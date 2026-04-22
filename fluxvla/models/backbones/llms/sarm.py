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

from typing import Optional, cast

import torch
import torch.nn as nn
from transformers import CLIPModel

from fluxvla.engines import LLM_BACKBONES
from fluxvla.engines.utils.hf_hub import resolve_hf_local_path


class StageTransformer(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 vis_emb_dim: int = 512,
                 text_emb_dim: int = 512,
                 state_dim: int = 32,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 num_cameras: int = 1,
                 num_classes_sparse: int = 4,
                 num_classes_dense: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.transformer_layer_cls = nn.TransformerEncoderLayer

        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            4 * d_model,
            dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        fused_in = d_model * (num_cameras + 2)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            'sparse':
            nn.Linear(d_model, num_classes_sparse),
            'dense':
            nn.Linear(d_model, num_classes_dense),
        })

    def _prep_lang(self, lang_emb: torch.Tensor, batch_size: int, seq_len: int,
                   dim: int) -> torch.Tensor:
        if lang_emb.dim() == 3:
            return self.lang_proj(lang_emb).unsqueeze(1)
        return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, seq_len, dim)

    def forward(self,
                img_seq: torch.Tensor,
                lang_emb: torch.Tensor,
                state: torch.Tensor,
                lengths: torch.Tensor,
                scheme: str = 'sparse') -> torch.Tensor:
        batch_size, num_cameras, seq_len, _ = img_seq.shape
        dim = self.d_model
        device = img_seq.device

        vis_proj = self.visual_proj(img_seq)
        state_proj = self.state_proj(state).unsqueeze(1)
        lang_proj = self._prep_lang(lang_emb, batch_size, seq_len, dim)
        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)
        x[:, :num_cameras, 0, :] = x[:, :num_cameras, 0, :] + self.first_pos

        x_tokens = x.view(batch_size, (num_cameras + 2) * seq_len, dim)
        token_len = x_tokens.size(1)
        base_mask = torch.arange(
            seq_len, device=device).expand(batch_size,
                                           seq_len) >= lengths.unsqueeze(1)
        padding_mask = base_mask.unsqueeze(1).expand(batch_size,
                                                     num_cameras + 2,
                                                     seq_len).reshape(
                                                         batch_size, token_len)
        causal_mask = torch.triu(
            torch.ones(token_len, token_len, device=device, dtype=torch.bool),
            diagonal=1)
        hidden = self.transformer(
            x_tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        hidden = hidden.view(batch_size, num_cameras + 2, seq_len,
                             dim).permute(0, 2, 1,
                                          3).reshape(batch_size, seq_len,
                                                     (num_cameras + 2) * dim)
        fused = self.fusion_backbone(hidden)
        return self.heads[scheme](fused)


class SubtaskTransformer(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 vis_emb_dim: int = 512,
                 text_emb_dim: int = 512,
                 state_dim: int = 32,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 num_cameras: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.transformer_layer_cls = nn.TransformerEncoderLayer

        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            4 * d_model,
            dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        fused_in = d_model * (num_cameras + 3)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            'sparse': nn.Linear(d_model, 1),
            'dense': nn.Linear(d_model, 1),
        })

    def _prep_lang(self, lang_emb: torch.Tensor, batch_size: int, seq_len: int,
                   dim: int) -> torch.Tensor:
        if lang_emb.dim() == 3:
            return self.lang_proj(lang_emb).unsqueeze(1)
        return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, seq_len, dim)

    def _stage_to_dmodel(self, stage_prior: torch.Tensor) -> torch.Tensor:
        batch_size, one, seq_len, num_classes = stage_prior.shape
        dim = self.d_model
        if dim == num_classes:
            return stage_prior
        if dim > num_classes:
            pad = torch.zeros(
                batch_size,
                one,
                seq_len,
                dim - num_classes,
                device=stage_prior.device,
                dtype=stage_prior.dtype)
            return torch.cat([stage_prior, pad], dim=-1)
        return stage_prior[..., :dim]

    def forward(self,
                img_seq: torch.Tensor,
                lang_emb: torch.Tensor,
                state: torch.Tensor,
                lengths: torch.Tensor,
                stage_prior: torch.Tensor,
                scheme: str = 'sparse') -> torch.Tensor:
        batch_size, num_cameras, seq_len, _ = img_seq.shape
        dim = self.d_model
        device = img_seq.device

        vis_proj = self.visual_proj(img_seq)
        state_proj = self.state_proj(state).unsqueeze(1)
        lang_proj = self._prep_lang(lang_emb, batch_size, seq_len, dim)
        stage_emb = self._stage_to_dmodel(stage_prior)
        x = torch.cat([vis_proj, lang_proj, state_proj, stage_emb], dim=1)
        x[:, :num_cameras, 0, :] = x[:, :num_cameras, 0, :] + self.first_pos

        x_tokens = x.view(batch_size, (num_cameras + 3) * seq_len, dim)
        token_len = x_tokens.size(1)
        base_mask = torch.arange(
            seq_len, device=device).expand(batch_size,
                                           seq_len) >= lengths.unsqueeze(1)
        padding_mask = base_mask.unsqueeze(1).expand(batch_size,
                                                     num_cameras + 3,
                                                     seq_len).reshape(
                                                         batch_size, token_len)
        causal_mask = torch.triu(
            torch.ones(token_len, token_len, device=device, dtype=torch.bool),
            diagonal=1)
        hidden = self.transformer(
            x_tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        hidden = hidden.view(batch_size, num_cameras + 3, seq_len,
                             dim).permute(0, 2, 1,
                                          3).reshape(batch_size, seq_len,
                                                     (num_cameras + 3) * dim)
        fused = self.fusion_backbone(hidden)
        return torch.sigmoid(self.heads[scheme](fused)).squeeze(-1)


@LLM_BACKBONES.register_module()
class SARMBackbone(nn.Module):

    transformer_layer_cls = nn.TransformerEncoderLayer

    def __init__(self,
                 pretrained_name_or_path: Optional[str] = None,
                 clip_model_name_or_path: Optional[str] = None,
                 hidden_dim: int = 768,
                 max_state_dim: int = 32,
                 num_layers: int = 8,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 num_cameras: int = 1,
                 num_sparse_stages: int = 1,
                 num_dense_stages: int = 1,
                 freeze_clip_backbone: bool = True):
        super().__init__()
        clip_model_name_or_path = (
            pretrained_name_or_path or clip_model_name_or_path)
        if clip_model_name_or_path is None:
            raise ValueError('`pretrained_name_or_path` or '
                             '`clip_model_name_or_path` must be provided '
                             'for SARMBackbone.')
        clip_model_name_or_path = resolve_hf_local_path(
            clip_model_name_or_path)
        self.pretrained_name_or_path = clip_model_name_or_path
        self.clip_model_name_or_path = clip_model_name_or_path
        self.clip_model = CLIPModel.from_pretrained(clip_model_name_or_path)
        projection_dim = self.clip_model.config.projection_dim
        self.stage_model = StageTransformer(
            d_model=hidden_dim,
            vis_emb_dim=projection_dim,
            text_emb_dim=projection_dim,
            state_dim=max_state_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            dropout=dropout,
            num_cameras=num_cameras,
            num_classes_sparse=num_sparse_stages,
            num_classes_dense=num_dense_stages,
        )
        self.subtask_model = SubtaskTransformer(
            d_model=hidden_dim,
            vis_emb_dim=projection_dim,
            text_emb_dim=projection_dim,
            state_dim=max_state_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            dropout=dropout,
            num_cameras=num_cameras,
        )
        self.freeze_clip_backbone = freeze_clip_backbone
        if self.freeze_clip_backbone:
            self.clip_model.requires_grad_(False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_cameras, channels, height, width = (
            images.shape)
        flat_images = cast(
            torch.FloatTensor,
            images.reshape(batch_size * seq_len * num_cameras, channels,
                           height, width).float(),
        )
        image_features = self.clip_model.get_image_features(
            pixel_values=flat_images)
        image_features = image_features.reshape(batch_size, seq_len,
                                                num_cameras, -1)
        return image_features.permute(0, 2, 1, 3).contiguous()

    def encode_text(self, text_input_ids: torch.Tensor,
                    text_attention_mask: torch.Tensor) -> torch.Tensor:
        return self.clip_model.get_text_features(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
