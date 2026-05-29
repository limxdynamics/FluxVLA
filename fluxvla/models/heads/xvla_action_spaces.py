# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

# Origin: Modified from
# Upstream-Repo: 2toINF/X-VLA
# Upstream-Path: models/action_hub.py
# Upstream-Ref: origin/main@6bc2513f5f1cbec715cc668b414392a6cae5c671
# SPDX-License-Identifier: Apache-2.0
#
# Notes: Adapted into FluxVLA's action-space registry; the EE6D / joint / auto
# layouts preserve X-VLA semantics while fitting FluxVLA's module structure.

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_REGISTRY: Dict[str, Type['BaseActionSpace']] = {}


def register_action(name: str):
    """Decorator for registering a new XVLA action space."""

    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> "
                           f'{ACTION_REGISTRY[key]}')
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def build_action_space(name: str, **kwargs) -> 'BaseActionSpace':
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. "
                       f'Available: {list(ACTION_REGISTRY.keys())}')
    return ACTION_REGISTRY[key](**kwargs)


class BaseActionSpace(nn.Module):
    """Abstract base class for XVLA action-space definitions."""

    name: str = 'base'
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(pred, target, action_masks=action_masks)

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del mode
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return action


def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(
            f'{name} contains out-of-range indices {bad} for action dim D={D}')


def _select_mask(action_masks: Optional[torch.Tensor],
                 idx=None) -> Optional[torch.Tensor]:
    if action_masks is None:
        return None
    if action_masks.ndim < 3 or idx is None:
        return action_masks
    if isinstance(idx, tuple):
        idx = list(idx)
    return action_masks[:, :, idx]


def _masked_reduce(losses: torch.Tensor,
                   action_masks: Optional[torch.Tensor]) -> torch.Tensor:
    if action_masks is None:
        return losses.mean()
    mask = action_masks.to(device=losses.device, dtype=losses.dtype)
    while mask.ndim < losses.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(losses)
    return (losses * mask).sum() / (mask.sum() + 1e-8)


def _masked_mse(pred: torch.Tensor,
                target: torch.Tensor,
                action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
    losses = F.mse_loss(pred, target, reduction='none')
    return _masked_reduce(losses, action_masks)


def _masked_bce_with_logits(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    return _masked_reduce(losses, action_masks)


@register_action('ee6d')
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target, action_masks=None):
        assert pred.shape == target.shape, 'pred/target shapes must match'
        _, _, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, 'gripper_idx')

        g_losses = [
            _masked_bce_with_logits(
                pred[:, :, gi],
                target[:, :, gi],
                _select_mask(action_masks, gi),
            ) for gi in self.gripper_idx
        ]
        gripper_loss = sum(g_losses) / len(
            self.gripper_idx) * self.GRIPPER_SCALE

        pos_loss = (_masked_mse(
            pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1],
            _select_mask(action_masks, self.POS_IDX_1)) + _masked_mse(
                pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2],
                _select_mask(action_masks, self.POS_IDX_2))) * self.XYZ_SCALE

        rot_loss = (_masked_mse(
            pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1],
            _select_mask(action_masks, self.ROT_IDX_1)) + _masked_mse(
                pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2],
                _select_mask(action_masks, self.ROT_IDX_2))) * self.ROT_SCALE

        return {
            'position_loss': pos_loss,
            'rotate6D_loss': rot_loss,
            'gripper_loss': gripper_loss,
        }

    def preprocess(self, proprio, action, mode='train'):
        del mode
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        if action.size(-1) > max(self.gripper_idx):
            action[...,
                   self.gripper_idx] = torch.sigmoid(action[...,
                                                            self.gripper_idx])
        return action


@register_action('joint')
class JointActionSpace(BaseActionSpace):
    """Joint-space layout with joints + gripper only."""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target, action_masks=None):
        assert pred.shape == target.shape
        _, _, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, 'gripper_idx')

        g_losses = [
            _masked_bce_with_logits(
                pred[:, :, gi],
                target[:, :, gi],
                _select_mask(action_masks, gi),
            ) for gi in self.gripper_idx
        ]
        gripper_loss = sum(g_losses) / len(
            self.gripper_idx) * self.GRIPPER_SCALE

        joints_idx = tuple(
            i for i in range(D) if i not in set(self.gripper_idx))
        joints_loss = _masked_mse(
            pred[:, :, joints_idx],
            target[:, :, joints_idx],
            _select_mask(action_masks, joints_idx),
        ) * self.JOINTS_SCALE

        return {
            'joints_loss': joints_loss,
            'gripper_loss': gripper_loss,
        }

    def preprocess(self, proprio, action, mode='train'):
        del mode
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        if action.size(-1) > max(self.gripper_idx):
            action[...,
                   self.gripper_idx] = torch.sigmoid(action[...,
                                                            self.gripper_idx])
        return action


@register_action('agibot_ee6d')
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot variant of EE6DActionSpace using MSE for all components."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target, action_masks=None):
        assert pred.shape == target.shape
        _, _, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, 'gripper_idx')

        gripper_loss = _masked_mse(
            pred[:, :, self.gripper_idx],
            target[:, :, self.gripper_idx],
            _select_mask(action_masks, self.gripper_idx),
        ) * self.GRIPPER_SCALE
        pos_loss = (_masked_mse(
            pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1],
            _select_mask(action_masks, self.POS_IDX_1)) + _masked_mse(
                pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2],
                _select_mask(action_masks, self.POS_IDX_2))) * self.XYZ_SCALE
        rot_loss = (_masked_mse(
            pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1],
            _select_mask(action_masks, self.ROT_IDX_1)) + _masked_mse(
                pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2],
                _select_mask(action_masks, self.ROT_IDX_2))) * self.ROT_SCALE

        return {
            'position_loss': pos_loss,
            'rotate6D_loss': rot_loss,
            'gripper_loss': gripper_loss,
        }

    def preprocess(self, proprio, action, mode='train'):
        del mode
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return action


@register_action('auto')
class AutoActionSpace(BaseActionSpace):
    """Action space that adapts model outputs to the dataset action width."""

    JOINTS_SCALE = 100.0

    def __init__(self, ori_action_dim: int, max_dim: int = 20):
        super().__init__()
        self.ori_action_dim = ori_action_dim
        self.dim_action = max_dim
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.ori_action_dim:
            if x.size(-1) < self.ori_action_dim:
                pad_shape = list(
                    x.shape[:-1]) + [self.ori_action_dim - x.size(-1)]
                pad = x.new_zeros(pad_shape)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., :self.ori_action_dim]

        pad_shape = list(
            x.shape[:-1]) + [self.dim_action - self.ori_action_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_ori_action_dim(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :self.ori_action_dim]

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, (
            f'Shape mismatch: pred {pred.shape} vs target {target.shape}')

        action_idx = tuple(range(self.ori_action_dim))
        joints_loss = _masked_mse(
            pred[:, :, :self.ori_action_dim],
            target[:, :, :self.ori_action_dim],
            _select_mask(action_masks, action_idx),
        ) * self.JOINTS_SCALE
        return {'joints_loss': joints_loss}

    def preprocess(self,
                   proprio: torch.Tensor,
                   action: torch.Tensor,
                   mode: str = 'train'):
        del mode
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self._trim_to_ori_action_dim(action)


__all__ = [
    'ACTION_REGISTRY',
    'AGIBOTEE6DActionSpace',
    'AutoActionSpace',
    'BaseActionSpace',
    'EE6DActionSpace',
    'JointActionSpace',
    'build_action_space',
    'register_action',
]
