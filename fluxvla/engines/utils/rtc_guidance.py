# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-time Real-Time Chunking (RTC) guidance utilities.

Provides prefix weight computation and velocity correction for
the 'guidance' RTC method (models not trained with RTC).
Called from predict_action() in FlowMatchingHead and PI0FlowMatching.

References:
    - Physical Intelligence Kinetix: realtime_action, get_prefix_weights
      https://github.com/Physical-Intelligence/real-time-chunking-kinetix
"""

import math

import torch


def compute_prefix_weights(n_action_steps,
                           prefix_len,
                           decay_end,
                           schedule='exp',
                           device='cpu'):
    """Compute per-timestep prefix weights for RTC guidance.

    Reference: PI Kinetix get_prefix_weights.

    Args:
        n_action_steps: Total action chunk length.
        prefix_len: Number of steps already executed (locked prefix).
        decay_end: Position where guidance decays to zero.
        schedule: Weight decay schedule ('exp', 'linear', 'ones', 'zeros').
        device: Torch device.

    Returns:
        Tensor of shape (n_action_steps,) with values in [0, 1].
    """
    weights = torch.zeros(n_action_steps, device=device)

    if prefix_len <= 0:
        return weights

    # Locked prefix: weight = 1.0
    locked = min(prefix_len, n_action_steps)
    weights[:locked] = 1.0

    # Decay region: prefix_len .. decay_end
    if decay_end > prefix_len:
        decay_len = decay_end - prefix_len
        for i in range(decay_len):
            pos = prefix_len + i
            if pos >= n_action_steps:
                break
            frac = i / max(decay_len - 1, 1)  # 0..1

            if schedule == 'exp':
                # Exponential decay (recommended)
                weights[pos] = math.exp(-3.0 * frac)
            elif schedule == 'linear':
                weights[pos] = 1.0 - frac
            elif schedule == 'ones':
                weights[pos] = 1.0
            elif schedule == 'zeros':
                weights[pos] = 0.0
            else:
                raise ValueError(f'Unknown schedule: {schedule}')

    return weights


def compute_guidance_weight(t, max_guidance_weight):
    """Compute guidance weight for a single denoising step.

    Reference: PI Kinetix realtime_action.

    The weight is large when t is close to 0 (noisy) and small when t
    is close to 1 (clean), ensuring strong guidance early in denoising
    and minimal interference near completion.

    Args:
        t: Denoising time (0 = noise, 1 = clean).
        max_guidance_weight: Upper bound for guidance weight.

    Returns:
        Scalar guidance weight.
    """
    eps = 1e-6
    t_clamped = max(t, eps)
    one_minus_t = max(1.0 - t, eps)

    inv_r2 = (t_clamped**2 + one_minus_t**2) / (one_minus_t**2)
    c = one_minus_t / t_clamped
    return min(c * inv_r2, max_guidance_weight)


def apply_rtc_guidance(x_t,
                       denoise_fn,
                       prev_actions,
                       prefix_weights,
                       t,
                       max_guidance_weight,
                       use_vjp=False):
    """Correct velocity to align prefix with prev_actions.

    Args:
        x_t: Current noisy actions (B, T, D).
        denoise_fn: Callable x_t -> v_t.
        prev_actions: Previous action chunk to guide towards (B, L, D),
            where L <= T. Only the first L steps are guided.
        prefix_weights: Per-position weights (T,).
        t: Denoising progress (0=noise, 1=clean).
        max_guidance_weight: Upper bound for guidance weight.
        use_vjp: If True, backprop error through denoiser Jacobian
            (more accurate, ~2-3x slower).
    """
    eps = 1e-6
    one_minus_t = max(1.0 - t, eps)
    w = prefix_weights[None, :, None]  # (1, T, 1)
    gw = compute_guidance_weight(t, max_guidance_weight)

    def compute_full_error(x_1):
        L_prev = prev_actions.shape[1]
        if L_prev == 0:
            return torch.zeros_like(x_1)

        error_actual = (prev_actions - x_1[:, :L_prev, :]) * w[:, :L_prev, :]
        error_full = torch.zeros_like(x_1)
        error_full[:, :L_prev, :] = error_actual
        return error_full

    if use_vjp:
        with torch.enable_grad():
            x_g = x_t.detach().requires_grad_(True)
            v_t = denoise_fn(x_g)
            x_1 = x_g + v_t * one_minus_t
            error_full = compute_full_error(x_1)
            correction = torch.autograd.grad(
                x_1, x_g, grad_outputs=error_full)[0]
        return v_t.detach() + gw * correction.detach()
    else:
        v_t = denoise_fn(x_t)
        x_1 = x_t + v_t * one_minus_t
        error_full = compute_full_error(x_1)
        return v_t + gw * error_full
