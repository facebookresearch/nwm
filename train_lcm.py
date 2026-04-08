"""
Progressive Distillation Training Script for CDiT-S/2.

Trains a student model (with LoRA) to match 2 teacher DDIM steps in 1 step,
reducing inference from 250 → 125 → 62 steps across two rounds.

Usage:
    # Round 1: 250 → 125
    python train_lcm.py --config config/lcm_distill_sacson.yaml --round 1

    # Round 2: 125 → 62 (uses merged round 1 output)
    python train_lcm.py --config config/lcm_distill_sacson.yaml --round 2 \
        --teacher-ckpt logs/lcm_distill_sacson/checkpoints/round1_merged.pth.tar
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import logging
import os
import yaml
from copy import deepcopy
from time import time

from torch.utils.data import DataLoader, ConcatDataset
from diffusers.models import AutoencoderKL

from models import CDiT_models
from diffusion import create_diffusion
from diffusion.progressive_distillation import ProgressiveDistiller
from datasets import TrainingDataset
from lora import inject_lora, merge_lora, get_lora_params, count_lora_params
from misc import transform


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    device = torch.device("cuda")

    # Load configs
    with open("config/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # Determine round settings
    round_key = f"round_{args.round}"
    assert round_key in config, f"Round config '{round_key}' not found in config"
    teacher_steps = config[round_key]["teacher_steps"]
    student_steps = config[round_key]["student_steps"]

    # Setup experiment dir
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Progressive Distillation Round {args.round}: {teacher_steps} → {student_steps} steps")

    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    latent_size = config['image_size'] // 8
    num_cond = config['context_size']

    # Load teacher model
    teacher_ckpt_path = args.teacher_ckpt or config['teacher_checkpoint']
    logger.info(f"Loading teacher from {teacher_ckpt_path}")

    teacher = CDiT_models[config['model']](
        context_size=num_cond, input_size=latent_size, in_channels=4
    ).to(device)

    ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    if "ema" in ckpt:
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['ema'].items()}
    elif "model" in ckpt:
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    else:
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}

    res = teacher.load_state_dict(state_dict, strict=True)
    logger.info(f"Teacher loaded: {res}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Create student (copy of teacher + LoRA)
    student = deepcopy(teacher).to(device)
    n_injected = inject_lora(
        student,
        rank=config.get('lora_rank', 16),
        alpha=config.get('lora_alpha', 16.0),
    )
    logger.info(f"Injected LoRA into {n_injected} layers")

    n_lora = count_lora_params(student)
    n_total = sum(p.numel() for p in student.parameters())
    logger.info(f"Total params: {n_total:,}, LoRA params: {n_lora:,} ({100*n_lora/n_total:.1f}%)")

    # Create diffusion schedules
    teacher_diffusion = create_diffusion(str(teacher_steps))
    student_diffusion = create_diffusion(str(student_steps))
    distiller = ProgressiveDistiller(teacher_diffusion, student_diffusion)
    logger.info(f"Teacher: {teacher_diffusion.num_timesteps} steps, Student: {student_diffusion.num_timesteps} steps")

    # Setup optimizer (only LoRA params)
    lora_params = get_lora_params(student)
    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0)
    logger.info(f"Optimizer: AdamW, lr={lr}, {len(lora_params)} parameter groups")

    # Setup dataset
    train_datasets = []
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "train" not in data_config:
            continue
        if "distance" in data_config:
            min_dist_cat = data_config["distance"]["min_dist_cat"]
            max_dist_cat = data_config["distance"]["max_dist_cat"]
        else:
            min_dist_cat = config["distance"]["min_dist_cat"]
            max_dist_cat = config["distance"]["max_dist_cat"]
        if "len_traj_pred" in data_config:
            len_traj_pred = data_config["len_traj_pred"]
        else:
            len_traj_pred = config["len_traj_pred"]

        dataset = TrainingDataset(
            data_folder=data_config["data_folder"],
            data_split_folder=data_config["train"],
            dataset_name=dataset_name,
            image_size=config["image_size"],
            min_dist_cat=min_dist_cat,
            max_dist_cat=max_dist_cat,
            len_traj_pred=len_traj_pred,
            context_size=config["context_size"],
            normalize=config["normalize"],
            goals_per_obs=int(data_config.get("goals_per_obs", 4)),
            transform=transform,
            predefined_index=None,
            traj_stride=1,
        )
        train_datasets.append(dataset)
        logger.info(f"Dataset '{dataset_name}': {len(dataset)} samples")

    train_dataset = ConcatDataset(train_datasets)
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    logger.info(f"Total training samples: {len(train_dataset):,}")

    # Training loop
    max_steps = args.max_steps if args.max_steps > 0 else config.get('distill_steps', 10000)
    log_every = args.log_every
    save_every = args.save_every

    student.train()
    running_loss = 0.0
    log_steps = 0
    start_time = time()

    logger.info(f"Starting distillation for {max_steps} steps...")
    train_step = 0
    epoch = 0

    while train_step < max_steps:
        epoch += 1
        for x, y, rel_t in loader:
            if train_step >= max_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                with torch.no_grad():
                    B, T = x.shape[:2]
                    x_flat = x.flatten(0, 1)
                    x_flat = vae.encode(x_flat).latent_dist.sample().mul_(0.18215)
                    x_encoded = x_flat.unflatten(0, (B, T))

                num_goals = T - num_cond
                x_start = x_encoded[:, num_cond:].flatten(0, 1)
                x_cond = x_encoded[:, :num_cond].unsqueeze(1).expand(
                    B, num_goals, num_cond,
                    x_encoded.shape[2], x_encoded.shape[3], x_encoded.shape[4]
                ).flatten(0, 1)
                y_flat = y.flatten(0, 1)
                rel_t_flat = rel_t.flatten(0, 1)

                model_kwargs = dict(y=y_flat, x_cond=x_cond, rel_t=rel_t_flat)

                # Compute distillation loss
                loss_dict = distiller.distillation_losses(
                    teacher_model=teacher.forward,
                    student_model=student.forward,
                    x_start=x_start,
                    model_kwargs=model_kwargs,
                )
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            if config.get('grad_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=config['grad_clip_val'])
            opt.step()

            running_loss += loss.detach().item()
            log_steps += 1
            train_step += 1

            if train_step % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(
                    f"[Round {args.round}] step={train_step:06d}/{max_steps} "
                    f"loss={avg_loss:.6e} steps/s={steps_per_sec:.2f}"
                )
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_step % save_every == 0:
                ckpt_path = f"{checkpoint_dir}/round{args.round}_step{train_step:06d}.pth.tar"
                torch.save({
                    "student": student.state_dict(),
                    "opt": opt.state_dict(),
                    "step": train_step,
                    "round": args.round,
                    "teacher_steps": teacher_steps,
                    "student_steps": student_steps,
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint (with LoRA)
    final_lora_path = f"{checkpoint_dir}/round{args.round}_final_lora.pth.tar"
    torch.save({
        "student": student.state_dict(),
        "step": train_step,
        "round": args.round,
        "teacher_steps": teacher_steps,
        "student_steps": student_steps,
    }, final_lora_path)
    logger.info(f"Saved final LoRA checkpoint: {final_lora_path}")

    # Merge LoRA and save clean checkpoint
    merge_lora(student)
    merged_path = f"{checkpoint_dir}/round{args.round}_merged.pth.tar"
    torch.save({
        "ema": student.state_dict(),
        "model": student.state_dict(),
        "step": train_step,
        "round": args.round,
        "student_steps": student_steps,
    }, merged_path)
    logger.info(f"Saved merged checkpoint: {merged_path}")
    logger.info(f"Round {args.round} complete! Use {merged_path} as teacher for next round.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Distillation for CDiT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--round", type=int, required=True, choices=[1, 2],
                        help="Distillation round (1: 250→125, 2: 125→62)")
    parser.add_argument("--teacher-ckpt", type=str, default=None,
                        help="Override teacher checkpoint path")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Override max distillation steps (0 = use config)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--bfloat16", type=int, default=1)
    args = parser.parse_args()
    main(args)
