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
from typing import Callable, Dict, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import (_or_policy,
                                         transformer_auto_wrap_policy)
from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMModel

from fluxvla.engines import VLM_BACKBONES
from fluxvla.engines.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


@VLM_BACKBONES.register_module()
class SmolVLMBackbone(nn.Module):
    """SmolVLM backbone wrapping the HF SmolVLMModel.

    Builds the complete SmolVLM (vision + connector + text) from
    explicit config dicts and exposes sub-components for
    split-component VLA consumption.

    Args:
        vision_config: SmolVLMVisionConfig parameters.
        text_config: LlamaConfig parameters (model_type defaults
            to 'llama').
        scale_factor: Pixel-shuffle scale factor for connector.
        num_vlm_layers: Number of text layers to keep (-1 = all).
    """

    def __init__(
        self,
        vision_config: Dict,
        text_config: Dict,
        scale_factor: int = 4,
        num_vlm_layers: int = -1,
        torch_dtype: str = 'bfloat16',
    ):
        super().__init__()

        config = SmolVLMConfig(
            vision_config=vision_config,
            text_config=text_config,
            scale_factor=scale_factor,
        )

        self.vlm = SmolVLMModel(config)

        # Cast to match lerobot's AutoModelForImageTextToText.from_pretrained(
        # ..., torch_dtype="bfloat16") behaviour.
        _dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        if torch_dtype in _dtype_map:
            self.vlm = self.vlm.to(dtype=_dtype_map[torch_dtype])

        if num_vlm_layers > 0:
            overwatch.info(
                f'Reducing text layers to {num_vlm_layers}', ctx_level=1)
            self.vlm.text_model.layers = \
                self.vlm.text_model.layers[:num_vlm_layers]

        v_hidden = config.vision_config.hidden_size
        t_hidden = config.text_config.hidden_size
        t_layers = len(self.vlm.text_model.layers)
        overwatch.info(
            f'Built SmolVLMBackbone: '
            f'vision_hidden={v_hidden}, '
            f'text_hidden={t_hidden}, '
            f'text_layers={t_layers}, '
            f'scale_factor={scale_factor}',
            ctx_level=1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            'SmolVLMBackbone does not support end-to-end forward(). '
            'SmolVLAFlowMatching interleaves VLM and expert layers, '
            'use forward_vision(), forward_connector(), layers, etc.')

    # -- Vision / connector forward --

    def forward_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision encoder on a single image.

        Args:
            pixel_values: (B, 3, H, W).

        Returns:
            (B, N_patches, vision_hidden_size).
        """
        return self.vlm.vision_model(
            pixel_values.to(
                dtype=self.vlm.vision_model.dtype)).last_hidden_state

    def forward_connector(self, image_features: torch.Tensor) -> torch.Tensor:
        """Run connector (pixel-shuffle + projection).

        Args:
            image_features: (B, N_patches, vision_hidden_size).

        Returns:
            (B, N_projected, text_hidden_size).
        """
        return self.vlm.connector(image_features)

    # -- Text model access --

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.vlm.text_model.get_input_embeddings()(tokens)

    @property
    def text_model(self):
        return self.vlm.text_model

    @property
    def layers(self):
        return self.vlm.text_model.layers

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.vlm.text_model.norm(hidden_states)

    # -- Config properties --

    @property
    def hidden_size(self):
        return self.vlm.config.text_config.hidden_size

    @property
    def head_dim(self):
        return self.vlm.config.text_config.head_dim

    @property
    def num_attention_heads(self):
        return self.vlm.config.text_config.num_attention_heads

    @property
    def num_key_value_heads(self):
        return self.vlm.config.text_config.num_key_value_heads

    @property
    def config(self):
        return self.vlm.config.text_config

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer

    # -- Training utilities --

    def enable_gradient_checkpointing(self):
        self.vlm.text_model.gradient_checkpointing_enable()

    def get_fsdp_wrapping_policy(self) -> Callable:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaRMSNorm)
        from transformers.models.smolvlm.modeling_smolvlm import \
            SmolVLMEncoderLayer

        llm_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        vision_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={SmolVLMEncoderLayer},
        )

        def match_linear(module, *args, **kwargs):
            return isinstance(module, (nn.Linear, nn.LayerNorm, LlamaRMSNorm))

        return partial(
            _or_policy,
            policies=[llm_policy, vision_policy, match_linear],
        )
