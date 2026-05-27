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
"""Shared RA-BC loss reduction helpers."""

from typing import Optional

import torch


def _to_loss_tensor(value: torch.Tensor,
                    reference: torch.Tensor) -> torch.Tensor:
    """Convert ``value`` to the same device and dtype as ``reference``."""
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype)


def _expand_to_losses(value: torch.Tensor,
                      losses: torch.Tensor) -> torch.Tensor:
    """Expand a per-sample or per-token tensor to the loss tensor shape."""
    while value.ndim < losses.ndim:
        value = value.unsqueeze(-1)
    return value.expand_as(losses)


def reduce_action_bc_loss(losses: torch.Tensor,
                          action_mask: Optional[torch.Tensor] = None,
                          sample_weight: Optional[torch.Tensor] = None,
                          eps: float = 1e-8) -> torch.Tensor:
    """Reduce unreduced BC losses with optional sample-level weights.

    Args:
        losses (torch.Tensor): Element-wise BC losses, usually shaped
            ``[batch, horizon, action_dim]``.
        action_mask (Optional[torch.Tensor]): Valid-action mask. It can be
            shaped like ``[batch, horizon]`` or exactly like ``losses``.
        sample_weight (Optional[torch.Tensor]): Per-sample RA-BC weights
            shaped like ``[batch]``. When omitted, this function matches the
            existing mean-over-valid-elements behavior.
        eps (float): Numerical stability constant.

    Returns:
        torch.Tensor: Scalar reduced loss.
    """
    if action_mask is None and sample_weight is None:
        return losses.mean()

    if action_mask is None:
        valid = torch.ones_like(losses)
    else:
        valid = _expand_to_losses(_to_loss_tensor(action_mask, losses), losses)

    if sample_weight is None:
        return (losses * valid).sum() / (valid.sum() + eps)

    weights = _expand_to_losses(_to_loss_tensor(sample_weight, losses), losses)
    weighted_valid = valid * weights
    return (losses * weighted_valid).sum() / (weighted_valid.sum() + eps)
