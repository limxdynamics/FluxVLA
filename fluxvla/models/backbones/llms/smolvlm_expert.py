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

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import (_or_policy,
                                         transformer_auto_wrap_policy)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from fluxvla.engines import LLM_BACKBONES
from fluxvla.engines.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


@LLM_BACKBONES.register_module()
class SmolVLMExpert(nn.Module):
    """SmolVLA action expert -- a smaller Llama model with optional
    cross-attention K/V reshaping.

    Built from explicit config parameters rather than derived
    at runtime.

    Args:
        hidden_size: Expert hidden dimension.
        num_hidden_layers: Number of expert transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads.
        head_dim: Attention head dimension.
        intermediate_size: FFN intermediate size. If -1, computed
            automatically from hidden_size.
        vocab_size: Vocabulary size (unused but required for config).
        attention_bias: Whether attention layers use bias.
        rms_norm_eps: RMSNorm epsilon.
        hidden_act: Activation function name.
        max_position_embeddings: Maximum sequence length.
        attention_mode: 'self_attn' or 'cross_attn'.
        vlm_kv_dim: VLM KV dimension for cross-attn mode. Required
            when attention_mode='cross_attn'.
        self_attn_every_n_layers: In cross_attn mode, use self-attn
            every N layers (others use cross-attn). -1 = all cross.
    """

    def __init__(
        self,
        hidden_size: int = 720,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        head_dim: int = 64,
        intermediate_size: int = -1,
        vocab_size: int = 49280,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = 'silu',
        max_position_embeddings: int = 8192,
        attention_mode: str = 'cross_attn',
        vlm_kv_dim: int = 320,
        self_attn_every_n_layers: int = -1,
        torch_dtype: str = 'bfloat16',
    ):
        super().__init__()

        if intermediate_size < 0:
            intermediate_size = self._compute_intermediate_size(hidden_size)

        expert_config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
        )

        self.expert = LlamaModel(expert_config)
        self.expert.embed_tokens = None

        # Cast to match lerobot's bf16 expert. Must happen BEFORE
        # K/V replacement so newly created nn.Linear projections
        # remain float32 (matching lerobot behaviour).
        _dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        if torch_dtype in _dtype_map:
            self.expert = self.expert.to(dtype=_dtype_map[torch_dtype])

        self.attention_mode = attention_mode
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.vlm_kv_dim = vlm_kv_dim

        # Per-layer attention mode: True = self-attn, False = cross-attn.
        # Determined once here and consumed by SmolVLAFlowMatching to
        # avoid recomputing (and misaligning) in the forward loop.
        self.is_self_attn = []

        # For cross-attn mode, reshape K/V projections on non-self-attn
        # layers so they accept VLM-dimension key/value states.
        # New nn.Linear layers stay float32 (matches lerobot).
        if 'cross' in attention_mode:
            expert_kv_dim = num_key_value_heads * head_dim
            for layer_idx in range(len(self.expert.layers)):
                is_self = (
                    self_attn_every_n_layers > 0
                    and layer_idx % self_attn_every_n_layers == 0)
                self.is_self_attn.append(is_self)
                if is_self:
                    continue
                self.expert.layers[layer_idx].self_attn.k_proj = \
                    nn.Linear(
                        vlm_kv_dim, expert_kv_dim,
                        bias=attention_bias)
                self.expert.layers[layer_idx].self_attn.v_proj = \
                    nn.Linear(
                        vlm_kv_dim, expert_kv_dim,
                        bias=attention_bias)
        else:
            self.is_self_attn = [True] * len(self.expert.layers)

        overwatch.info(
            f'Built SmolVLMExpert: hidden={hidden_size}, '
            f'layers={num_hidden_layers}, '
            f'attn_mode={attention_mode}',
            ctx_level=1)

    @staticmethod
    def _compute_intermediate_size(
        hidden_dim,
        ffn_dim_multiplier=4,
        multiple_of=256,
    ):
        """Compute intermediate FFN size aligned to `multiple_of`."""
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of)
        return hidden_dim

    @property
    def layers(self):
        return self.expert.layers

    @property
    def hidden_size(self):
        return self.expert.config.hidden_size

    def norm(self, hidden_states):
        return self.expert.norm(hidden_states)

    def get_fsdp_wrapping_policy(self) -> Callable:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaRMSNorm)

        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )

        def match_linear(module, *args, **kwargs):
            return isinstance(module, (nn.Linear, nn.LayerNorm, LlamaRMSNorm))

        return partial(
            _or_policy,
            policies=[transformer_block_policy, match_linear],
        )
