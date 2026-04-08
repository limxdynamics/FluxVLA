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
"""Training-time Real-Time Chunking (RTC).

Implements RTC by setting per-position time to 1.0 at delay positions,
so x_t = (1-1)*noise + 1*actions = actions (known prefix).
Loss is masked out at delay positions.

Reference:
    Physical Intelligence Kinetix — simulated_delay
    https://github.com/Physical-Intelligence/real-time-chunking-kinetix
"""

import torch


def sample_training_delay(batch_size,
                          max_delay,
                          distribution='exponential',
                          device='cpu'):
    """Sample per-batch delay values for training-time RTC.

    Args:
        batch_size: Number of samples in the batch.
        max_delay: Maximum delay value (exclusive).
        distribution: 'exponential' (favors small delays) or 'uniform'.
        device: Torch device.

    Returns:
        Tensor of shape (batch_size,) with integer delay
        values in [0, max_delay).
    """
    if max_delay <= 0:
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    if distribution == 'exponential':
        # Reference: PI Kinetix w = exp(arange(max_delay)[::-1])
        weights = torch.exp(
            torch.arange(max_delay, dtype=torch.float32,
                         device=device).flip(0))
        weights = weights / weights.sum()
        delays = torch.multinomial(
            weights.expand(batch_size, -1), num_samples=1).squeeze(-1)
    elif distribution == 'uniform':
        delays = torch.randint(0, max_delay, (batch_size, ), device=device)
    else:
        raise ValueError(f'Unknown distribution: {distribution}. '
                         f"Expected 'exponential' or 'uniform'.")

    return delays


def apply_rtc_time_conditioning(time,
                                action_masks,
                                delays,
                                n_action_steps,
                                clean_time=1.0):
    """Apply RTC conditioning: set time to clean_time at delay positions.

    Reference: PI Kinetix — delay positions get time forced so that
    x_t collapses to actions (known prefix). Loss is excluded.

    The clean_time value depends on the model's time convention:
        - FlowMatchingHead: time=1.0 is clean  (clean_time=1.0)
        - PI0FlowMatching: time=0.0 is clean   (clean_time=0.0)

    Args:
        time: Scalar time per sample (B,).
        action_masks: Action masks (B, T) or None.
        delays: Per-sample delay values (B,), integer tensor.
        n_action_steps: Number of action steps T.
        clean_time: Time value meaning "fully clean" in the model's
            convention. Default 1.0.

    Returns:
        Tuple of (per_position_time, conditioned_masks):
            per_position_time: (B, T) — clean_time at delay positions.
            conditioned_masks: (B, T) — 0.0 at delay positions.
    """
    B = time.shape[0]
    device = time.device

    # Expand scalar time to per-position: (B,) -> (B, T)
    per_position_time = time[:, None].expand(B, n_action_steps).clone()

    # Build delay mask: (B, T)
    positions = torch.arange(n_action_steps, device=device).unsqueeze(0)
    delay_mask = positions < delays.unsqueeze(1)

    # Delay positions: time = clean_time
    per_position_time[delay_mask] = clean_time

    # Mask out delay positions from loss
    if action_masks is not None:
        conditioned_masks = action_masks.clone()
    else:
        conditioned_masks = torch.ones(
            B, n_action_steps, dtype=time.dtype, device=device)
    conditioned_masks[delay_mask] = 0.0

    return per_position_time, conditioned_masks
