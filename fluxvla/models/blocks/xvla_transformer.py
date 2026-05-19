# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
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

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if fused scaled dot-product attention is available."""
    return hasattr(F, 'scaled_dot_product_attention')


class Mlp(nn.Module):
    """MLP used by the XVLA transformer blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(
            nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate='tanh')
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (norm_layer(hidden_features)
                     if norm_layer is not None else nn.Identity())
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with SDPA fallback."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(
            B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def basic_init(module: nn.Module) -> None:
    """Apply XVLA's default linear-layer initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor,
                       dim: int,
                       max_period: int = 100) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(start=0, end=half, dtype=t.dtype, device=t.device) /
        half)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DomainAwareLinear(nn.Module):
    """Linear layer with per-domain weights and biases."""

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor,
                domain_id: torch.LongTensor) -> torch.Tensor:
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y


class TransformerBlock(nn.Module):
    """Pre-LN transformer block used by XVLA heads."""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.1,
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SoftPromptedTransformer(nn.Module):
    """Multi-modal, domain-aware transformer with optional soft prompts."""

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(
                multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(
                multi_modal_input_size, hidden_size, num_domains=num_domains)
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size,
                                             hidden_size)

        self.pos_emb = nn.Parameter(
            torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio,
            hidden_size,
            num_domains=num_domains,
        )
        self.action_decoder = DomainAwareLinear(
            hidden_size, dim_action, num_domains=num_domains)

        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(
                num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        self.apply(basic_init)

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, num_actions = action_with_noise.shape[:2]

        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions,
                                                   self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions,
                                                     proprio.shape[-1])
        action_tokens = torch.cat(
            [action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens, domain_id)

        if self.use_hetero_proj:
            x = torch.cat([
                x,
                self.vlm_proj(vlm_features, domain_id),
                self.aux_visual_proj(aux_visual_inputs, domain_id),
            ],
                          dim=1)
        else:
            x = torch.cat(
                [x, self.vlm_proj(vlm_features),
                 self.aux_visual_proj(aux_visual_inputs)],
                dim=1,
            )

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f'Sequence length {seq_len} exceeds '
                f'max_len_seq={self.pos_emb.shape[1]}.')
        x = x + self.pos_emb[:, :seq_len, :]

        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(
                B, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        for block in self.blocks:
            x = block(x)

        return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)


__all__ = [
    'Attention',
    'DomainAwareLinear',
    'Mlp',
    'SoftPromptedTransformer',
    'TransformerBlock',
    'basic_init',
    'timestep_embedding',
]
