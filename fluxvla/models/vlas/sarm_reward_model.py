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

import copy
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy

from fluxvla.datasets.utils.sarm_utils import (load_temporal_proportions,
                                               normalize_stage_tau,
                                               pad_state_to_max_dim)
from fluxvla.engines import VLAS, initialize_overwatch
from fluxvla.models.backbones.llms.sarm import SARMBackbone
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class SARMRewardModel(BaseVLA):

    def __init__(self,
                 annotation_mode: str = 'single_stage',
                 llm_backbone: Optional[Dict] = None,
                 clip_model_name_or_path: str = (
                     './checkpoints/clip-vit-base-patch32'),
                 data_root_path: Optional[str | List[str]] = None,
                 hidden_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 8,
                 max_state_dim: int = 32,
                 dropout: float = 0.1,
                 n_obs_steps: int = 8,
                 frame_gap: int = 30,
                 max_rewind_steps: int = 4,
                 num_cameras: int = 1,
                 freeze_clip_backbone: bool = True,
                 freeze_llm_backbone: bool = False,
                 num_sparse_stages: int = 1,
                 sparse_subtask_names: Optional[List[str]] = None,
                 sparse_temporal_proportions: Optional[List[float]] = None,
                 num_dense_stages: Optional[int] = None,
                 dense_subtask_names: Optional[List[str]] = None,
                 dense_temporal_proportions: Optional[List[float]] = None,
                 pretrained_name_or_path: Optional[str] = None,
                 *args,
                 **kwargs) -> None:
        del args, kwargs
        self.annotation_mode = annotation_mode
        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap
        self.max_rewind_steps = max_rewind_steps
        self.gt_stage_ratio = 0.75

        dataset_root = data_root_path[0] if isinstance(
            data_root_path, list) else data_root_path
        if annotation_mode == 'single_stage':
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ['task']
            self.sparse_temporal_proportions = [1.0]
            self.num_dense_stages = None
            self.dense_subtask_names = None
            self.dense_temporal_proportions = None
        else:
            if dataset_root is not None:
                meta_root = Path(dataset_root) / 'meta'
                if annotation_mode == 'dual':
                    sparse_names, sparse_props = load_temporal_proportions(
                        meta_root, 'sparse')
                    self.num_sparse_stages = len(sparse_names)
                    self.sparse_subtask_names = sparse_names
                    self.sparse_temporal_proportions = sparse_props
                else:
                    self.num_sparse_stages = 1
                    self.sparse_subtask_names = ['task']
                    self.sparse_temporal_proportions = [1.0]
                dense_names, dense_props = load_temporal_proportions(
                    meta_root, 'dense')
                self.num_dense_stages = len(dense_names)
                self.dense_subtask_names = dense_names
                self.dense_temporal_proportions = dense_props
            else:
                self.num_sparse_stages = num_sparse_stages
                self.sparse_subtask_names = sparse_subtask_names or ['task']
                self.sparse_temporal_proportions = (
                    sparse_temporal_proportions or [1.0])
                self.num_dense_stages = num_dense_stages
                self.dense_subtask_names = dense_subtask_names
                self.dense_temporal_proportions = dense_temporal_proportions

        dense_stage_count = int(self.num_dense_stages
                                or self.num_sparse_stages)
        if self.uses_dual_heads:
            self.num_dense_stages = dense_stage_count

        llm_backbone_cfg = copy.deepcopy(llm_backbone) if llm_backbone else {}
        llm_backbone_cfg.setdefault('type', 'SARMBackbone')
        llm_backbone_cfg.setdefault('pretrained_name_or_path',
                                    clip_model_name_or_path)
        llm_backbone_cfg.setdefault('hidden_dim', hidden_dim)
        llm_backbone_cfg.setdefault('max_state_dim', max_state_dim)
        llm_backbone_cfg.setdefault('num_layers', num_layers)
        llm_backbone_cfg.setdefault('num_heads', num_heads)
        llm_backbone_cfg.setdefault('dropout', dropout)
        llm_backbone_cfg.setdefault('num_cameras', num_cameras)
        llm_backbone_cfg.setdefault('freeze_clip_backbone',
                                    freeze_clip_backbone)
        llm_backbone_cfg['num_sparse_stages'] = self.num_sparse_stages
        llm_backbone_cfg['num_dense_stages'] = dense_stage_count

        super().__init__(
            llm_backbone=llm_backbone_cfg,
            pretrained_name_or_path=pretrained_name_or_path,
            freeze_vision_backbone=True,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_vlm_backbone=True,
            freeze_projector=True,
            enable_mixed_precision_training=True,
            ignore_index=-100,
            norm_stats=None,
            name_mapping=None,
            strict_mapping=False,
        )

        backbone_cfg = llm_backbone_cfg
        self.clip_model_name_or_path = backbone_cfg['pretrained_name_or_path']
        self.hidden_dim = backbone_cfg['hidden_dim']
        self.num_heads = backbone_cfg['num_heads']
        self.num_layers = backbone_cfg['num_layers']
        self.max_state_dim = backbone_cfg['max_state_dim']
        self.dropout = backbone_cfg['dropout']
        self.num_cameras = backbone_cfg['num_cameras']
        self.freeze_clip_backbone = backbone_cfg['freeze_clip_backbone']
        self.all_module_keys = ['llm_backbone']
        self.trainable_module_keys = ['llm_backbone']

    @property
    def config(self):
        return None

    @property
    def uses_dual_heads(self) -> bool:
        return self.annotation_mode in ['dense_only', 'dual']

    def freeze_backbones(self) -> None:
        backbone = self._backbone()
        backbone.requires_grad_(not self.freeze_llm_backbone)
        if self.freeze_clip_backbone:
            backbone.clip_model.requires_grad_(False)
        self.trainable_module_keys = []
        if any(param.requires_grad for param in self.parameters()):
            self.trainable_module_keys.append('llm_backbone')

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _backbone(self) -> SARMBackbone:
        assert self.llm_backbone is not None
        return cast(SARMBackbone, self.llm_backbone)

    def _num_classes(self, scheme: str) -> int:
        if scheme == 'sparse':
            return int(self.num_sparse_stages)
        assert self.num_dense_stages is not None
        return int(self.num_dense_stages)

    def _gen_stage_emb(self, num_classes: int,
                       targets: torch.Tensor) -> torch.Tensor:
        idx = targets.long().clamp(min=0, max=num_classes - 1)
        stage_onehot = F.one_hot(idx, num_classes=num_classes).float()
        return stage_onehot.unsqueeze(1)

    def _valid_mask(self, lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        return torch.arange(
            seq_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    def _run_scheme(self, image_features: torch.Tensor,
                    text_features: torch.Tensor, state_features: torch.Tensor,
                    lengths: torch.Tensor, targets: torch.Tensor,
                    scheme: str) -> Dict[str, torch.Tensor]:
        backbone = self._backbone()
        num_classes = self._num_classes(scheme)
        gt_stage = torch.floor(targets).long().clamp(0, num_classes - 1)
        gt_tau = torch.remainder(targets, 1.0)
        valid_mask = self._valid_mask(lengths, targets.shape[1])

        stage_logits = backbone.stage_model(image_features, text_features,
                                            state_features, lengths, scheme)
        if torch.rand(1, device=targets.device).item() < self.gt_stage_ratio:
            stage_emb = self._gen_stage_emb(num_classes, targets)
        else:
            stage_idx = stage_logits.argmax(dim=-1)
            stage_onehot = F.one_hot(
                stage_idx, num_classes=num_classes).float()
            stage_emb = stage_onehot.unsqueeze(1)

        tau_pred = backbone.subtask_model(image_features, text_features,
                                          state_features, lengths, stage_emb,
                                          scheme)

        stage_loss = F.cross_entropy(stage_logits[valid_mask],
                                     gt_stage[valid_mask])
        tau_loss = F.mse_loss(tau_pred[valid_mask], gt_tau[valid_mask])
        stage_pred = stage_logits.argmax(dim=-1)
        raw_progress = stage_pred.float() + tau_pred
        normalized_progress = normalize_stage_tau(
            raw_progress,
            num_stages=num_classes,
            temporal_proportions=self.sparse_temporal_proportions
            if scheme == 'sparse' else self.dense_temporal_proportions,
            subtask_names=self.sparse_subtask_names
            if scheme == 'sparse' else self.dense_subtask_names,
        )
        assert isinstance(normalized_progress, torch.Tensor)
        stage_acc = (
            stage_pred[valid_mask] == gt_stage[valid_mask]).float().mean()
        return {
            'stage_loss': stage_loss,
            'tau_loss': tau_loss,
            'stage_acc': stage_acc,
            'pred_progress': normalized_progress,
        }

    def forward(self,
                images: torch.Tensor,
                text_input_ids: torch.Tensor,
                text_attention_mask: torch.Tensor,
                states: torch.Tensor,
                lengths: torch.Tensor,
                sparse_targets: torch.Tensor,
                dense_targets: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        device = self._device()
        backbone = self._backbone()
        if images.dim() == 5:
            images = images.unsqueeze(2)
        images = images.to(
            device=device, dtype=next(backbone.clip_model.parameters()).dtype)
        text_input_ids = text_input_ids.to(device=device)
        text_attention_mask = text_attention_mask.to(device=device)
        states = states.to(device=device, dtype=images.dtype)
        lengths = lengths.to(device=device)
        sparse_targets = sparse_targets.to(device=device, dtype=images.dtype)
        if dense_targets is not None:
            dense_targets = dense_targets.to(device=device, dtype=images.dtype)

        states = pad_state_to_max_dim(states, self.max_state_dim)
        image_features = backbone.encode_images(images)
        text_features = backbone.encode_text(text_input_ids,
                                             text_attention_mask)

        sparse_result = self._run_scheme(image_features, text_features, states,
                                         lengths, sparse_targets, 'sparse')
        total_loss = sparse_result['stage_loss'] + sparse_result['tau_loss']
        output = {
            'loss': total_loss,
            'sparse_stage_loss': sparse_result['stage_loss'].detach(),
            'sparse_tau_loss': sparse_result['tau_loss'].detach(),
            'sparse_stage_acc': sparse_result['stage_acc'].detach(),
            'pred_progress': sparse_result['pred_progress'].detach(),
        }

        if self.uses_dual_heads and dense_targets is not None:
            dense_result = self._run_scheme(image_features, text_features,
                                            states, lengths, dense_targets,
                                            'dense')
            total_loss = total_loss + dense_result[
                'stage_loss'] + dense_result['tau_loss']
            output.update({
                'loss':
                total_loss,
                'dense_stage_loss':
                dense_result['stage_loss'].detach(),
                'dense_tau_loss':
                dense_result['tau_loss'].detach(),
                'dense_stage_acc':
                dense_result['stage_acc'].detach(),
                'pred_dense_progress':
                dense_result['pred_progress'].detach(),
            })
        return output

    @torch.inference_mode()
    def predict_progress(self,
                         images: torch.Tensor,
                         text_input_ids: torch.Tensor,
                         text_attention_mask: torch.Tensor,
                         states: torch.Tensor,
                         lengths: Optional[torch.Tensor] = None,
                         head_mode: str = 'sparse',
                         return_all_frames: bool = False,
                         frame_index: Optional[int] = None) -> torch.Tensor:
        device = self._device()
        backbone = self._backbone()
        if images.dim() == 5:
            images = images.unsqueeze(2)
        images = images.to(
            device=device, dtype=next(backbone.clip_model.parameters()).dtype)
        text_input_ids = text_input_ids.to(device=device)
        text_attention_mask = text_attention_mask.to(device=device)
        states = states.to(device=device, dtype=images.dtype)
        if lengths is None:
            lengths = torch.full((images.shape[0], ),
                                 images.shape[1],
                                 dtype=torch.long,
                                 device=device)
        else:
            lengths = lengths.to(device=device)

        states = pad_state_to_max_dim(states, self.max_state_dim)
        image_features = backbone.encode_images(images)
        text_features = backbone.encode_text(text_input_ids,
                                             text_attention_mask)
        num_classes = self._num_classes(head_mode)
        stage_logits = backbone.stage_model(image_features, text_features,
                                            states, lengths, head_mode)
        stage_idx = stage_logits.argmax(dim=-1)
        stage_onehot = F.one_hot(stage_idx, num_classes=num_classes).float()
        tau_pred = backbone.subtask_model(image_features, text_features,
                                          states, lengths,
                                          stage_onehot.unsqueeze(1), head_mode)
        raw_progress = stage_idx.float() + tau_pred
        normalized_progress = normalize_stage_tau(
            raw_progress,
            num_stages=num_classes,
            temporal_proportions=self.sparse_temporal_proportions
            if head_mode == 'sparse' else self.dense_temporal_proportions,
            subtask_names=self.sparse_subtask_names
            if head_mode == 'sparse' else self.dense_subtask_names,
        )
        assert isinstance(normalized_progress, torch.Tensor)
        if return_all_frames:
            return normalized_progress
        target_index = self.n_obs_steps if frame_index is None else frame_index
        return normalized_progress[:, target_index]

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(
            _module_wrap_policy, module_classes={nn.TransformerEncoderLayer})
