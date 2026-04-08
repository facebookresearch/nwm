"""
Lightweight LoRA (Low-Rank Adaptation) implementation for CDiT models.

Injects trainable low-rank A/B matrices into nn.Linear layers,
enabling parameter-efficient fine-tuning for distillation.
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    A drop-in wrapper around nn.Linear that adds a low-rank A*B decomposition.
    Output = original_linear(x) + (x @ A^T @ B^T) * (alpha / rank)
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA decomposition: W_delta = B @ A
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already zeros

        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        # Original output
        out = self.original_linear(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into the original linear layer (irreversible)."""
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_linear.weight.add_(delta_w.to(self.original_linear.weight.dtype))

    @property
    def weight(self):
        return self.original_linear.weight

    @property
    def bias(self):
        return self.original_linear.bias


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 16.0, target_modules=None):
    """
    Inject LoRA into all matching nn.Linear layers in the model.

    Args:
        model: The CDiT model to inject LoRA into
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        target_modules: list of substrings to match against module names.
            If None, injects into attention and MLP layers in CDiT blocks.
    """
    if target_modules is None:
        # Default: inject into CDiT block attention and MLP layers
        target_modules = [
            "attn.qkv",      # self-attention QKV projection
            "attn.proj",     # self-attention output projection
            "cttn.in_proj",  # cross-attention input projection
            "cttn.out_proj", # cross-attention output projection
            "mlp.fc1",       # MLP first layer
            "mlp.fc2",       # MLP second layer
        ]

    injected = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Check if this module matches any target
        if not any(target in name for target in target_modules):
            continue

        # Get parent module and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = parts[0]

        # Replace with LoRA-wrapped version
        lora_linear = LoRALinear(module, rank=rank, alpha=alpha)
        lora_linear.to(module.weight.device)
        lora_linear.to(module.weight.dtype) # also match dtype just in case
        setattr(parent, attr_name, lora_linear)
        injected += 1

    print(f"Injected LoRA (rank={rank}, alpha={alpha}) into {injected} linear layers")
    return injected


def merge_lora(model: nn.Module):
    """
    Merge all LoRA weights into original weights and replace LoRALinear
    with plain nn.Linear. After merging, inference has zero overhead.
    """
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        module.merge_weights()

        # Replace LoRALinear with the original nn.Linear
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = parts[0]

        # Unfreeze original weights
        module.original_linear.weight.requires_grad = True
        if module.original_linear.bias is not None:
            module.original_linear.bias.requires_grad = True

        setattr(parent, attr_name, module.original_linear)
        merged += 1

    print(f"Merged {merged} LoRA layers into base weights")
    return merged


def get_lora_params(model: nn.Module):
    """Return only the LoRA A and B parameters for the optimizer."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def count_lora_params(model: nn.Module):
    """Count total LoRA trainable parameters."""
    total = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            total += module.lora_A.numel() + module.lora_B.numel()
    return total
