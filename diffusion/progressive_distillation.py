"""
Progressive Distillation for diffusion models.

Implements the core logic for training a student model to match
2 teacher DDIM steps in 1 student step, following Salimans & Ho (2022).
"""

import torch
import torch as th
import numpy as np

from .gaussian_diffusion import GaussianDiffusion, _extract_into_tensor


class ProgressiveDistiller:
    """
    Progressive distillation: train student to match 2 teacher DDIM steps in 1.

    Args:
        teacher_diffusion: SpacedDiffusion with teacher's step count (e.g. 250)
        student_diffusion: SpacedDiffusion with student's step count (e.g. 125)
    """

    def __init__(self, teacher_diffusion, student_diffusion):
        self.teacher_diffusion = teacher_diffusion
        self.student_diffusion = student_diffusion

        self.teacher_steps = teacher_diffusion.num_timesteps
        self.student_steps = student_diffusion.num_timesteps

        assert self.teacher_steps == 2 * self.student_steps, \
            f"Teacher steps ({self.teacher_steps}) must be exactly 2x student steps ({self.student_steps})"

    @torch.no_grad()
    def _teacher_ddim_step(self, model, x_t, t_index, model_kwargs):
        """
        Perform one deterministic DDIM step with the teacher model.
        t_index is the step index in the teacher's schedule (0 = most noisy).
        Returns denoised sample x_{t-1}.
        """
        diff = self.teacher_diffusion
        B = x_t.shape[0]
        device = x_t.device

        # Create timestep tensor
        t = th.tensor([t_index] * B, device=device, dtype=th.long)

        # Get model prediction (epsilon)
        out = diff.p_mean_variance(
            model, x_t, t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )

        pred_xstart = out["pred_xstart"]
        eps = diff._predict_eps_from_xstart(x_t, t, pred_xstart)

        # DDIM deterministic update (eta=0)
        alpha_bar = _extract_into_tensor(diff.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = _extract_into_tensor(diff.alphas_cumprod_prev, t, x_t.shape)

        mean_pred = (
            pred_xstart * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev) * eps
        )

        return mean_pred, pred_xstart

    def distillation_losses(
        self,
        teacher_model,
        student_model,
        x_start,
        model_kwargs=None,
        noise=None,
    ):
        """
        Compute progressive distillation loss for one batch.

        For each sample:
        1. Pick a random student timestep t_s in [0, student_steps)
        2. Map to teacher timestep pair: t_teacher = 2*t_s+1 and t_teacher-1 = 2*t_s
        3. Noise x_0 at teacher's timestep 2*t_s+1
        4. Teacher: 2 DDIM steps from t=2*t_s+1 → t=2*t_s → t=2*t_s-1
        5. Student: 1 step prediction at t_s
        6. Loss = MSE between student output and teacher target

        Args:
            teacher_model: frozen teacher model (callable)
            student_model: student model with LoRA (callable)
            x_start: clean latents [B, C, H, W]
            model_kwargs: conditioning dict (y, x_cond, rel_t)
            noise: optional pre-specified noise
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        B = x_start.shape[0]
        device = x_start.device
        t_diff = self.teacher_diffusion
        s_diff = self.student_diffusion

        # Sample random student timestep indices (0-indexed, where higher = noisier)
        # We sample from [1, student_steps) to ensure we always have valid teacher pairs
        t_student = th.randint(1, self.student_steps, (B,), device=device)

        # Map to teacher timestep indices
        # Student step t_s corresponds to the integration from t_s to t_s - 1.
        # This aligns with teacher steps 2*t_s and 2*t_s - 1.
        t_teacher_start = 2 * t_student  # noisy end
        t_teacher_mid = 2 * t_student - 1        # after 1 teacher DDIM step
        # After 2 teacher DDIM steps, we reach 2*t_s - 1, which maps to student t_s - 1

        # Clamp teacher start to valid range
        t_teacher_start = t_teacher_start.clamp(max=self.teacher_steps - 1)

        # Noise x_0 to teacher's starting timestep
        x_noisy = t_diff.q_sample(x_start, t_teacher_start, noise=noise)

        # Teacher: 2 DDIM steps (deterministic, no grad, per-sample timesteps)
        with th.no_grad():
            teacher_target = self._batched_teacher_2step(
                teacher_model, x_noisy, t_teacher_start, t_teacher_mid, x_start, model_kwargs
            )

        # Student: predict x at student's target timestep from x_noisy
        # The student uses its own diffusion schedule
        student_pred = self._student_denoise(
            student_model, x_noisy, t_student, t_teacher_start, model_kwargs
        )

        # Loss: MSE between student prediction and teacher target
        # Compute in float32 to prevent catastrophic cancellation with very small errors in bfloat16
        loss = th.mean((student_pred.float() - teacher_target.float()) ** 2, dim=[1, 2, 3])

        return {"loss": loss, "mse": loss}

    @torch.no_grad()
    def _batched_teacher_2step(self, model, x_t, t_start, t_mid, x_start, model_kwargs):
        """
        Perform 2 sequential DDIM steps with the teacher, handling per-sample timesteps.
        t_start, t_mid are [B] tensors (indices into teacher schedule).
        Returns the target x after 2 teacher DDIM steps.
        """
        diff = self.teacher_diffusion

        # --- Step 1: denoise from t_start ---
        out1 = diff.p_mean_variance(
            model, x_t, t_start,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        pred_xstart_1 = out1["pred_xstart"]
        eps_1 = diff._predict_eps_from_xstart(x_t, t_start, pred_xstart_1)

        alpha_bar_start = _extract_into_tensor(diff.alphas_cumprod, t_start, x_t.shape)
        alpha_bar_mid = _extract_into_tensor(diff.alphas_cumprod, t_mid, x_t.shape)

        x_mid = (
            pred_xstart_1 * th.sqrt(alpha_bar_mid)
            + th.sqrt(1 - alpha_bar_mid) * eps_1
        )

        # --- Step 2: denoise from t_mid ---
        out2 = diff.p_mean_variance(
            model, x_mid, t_mid,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        pred_xstart_2 = out2["pred_xstart"]
        eps_2 = diff._predict_eps_from_xstart(x_mid, t_mid, pred_xstart_2)

        # Target: the x at the timestep *before* t_mid (i.e., t_mid - 1)
        # Using alphas_cumprod_prev for the DDIM target
        alpha_bar_prev = _extract_into_tensor(diff.alphas_cumprod_prev, t_mid, x_t.shape)

        target = (
            pred_xstart_2 * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev) * eps_2
        )

        return target

    def _student_denoise(self, model, x_noisy, t_student, t_teacher_start, model_kwargs):
        """
        Student predicts the target from x_noisy.
        The student sees x_noisy (noised at teacher's t_start) and predicts x0,
        then we project to the student's target noise level.
        """
        s_diff = self.student_diffusion

        # Student model forward at student timestep
        out = s_diff.p_mean_variance(
            model, x_noisy, t_student,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        eps = s_diff._predict_eps_from_xstart(x_noisy, t_student, pred_xstart)

        # Project to t_student - 1 using DDIM update
        alpha_bar_prev = _extract_into_tensor(
            s_diff.alphas_cumprod_prev, t_student, x_noisy.shape
        )

        student_pred = (
            pred_xstart * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev) * eps
        )

        return student_pred
