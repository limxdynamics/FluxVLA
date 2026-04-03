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
from typing import Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VLDecoderLayer

from fluxvla.engines import VLM_BACKBONES
from .hf_vlm import VLMBackbone


@VLM_BACKBONES.register_module()
class QWen2_5VL(VLMBackbone):
    """
    HuggingFace-compatible wrapper for PaliGemma VLMs.
    Inherits from VLMBackbone and provides PaliGemma-specific token
    and configuration handling. This includes explicitly adding a padding
    token to the tokenizer and resizing the model embeddings accordingly.
    Registered into the `VLM_BACKBONES` registry for flexible instantiation.
    Args:
        vlm_backbone_id (str): Identifier string for this backbone.
        vlm_config (Dict, optional): Configuration dictionary for the
            VLM.
        vlm_path (Optional[str]): Path to the VLM model weights.
            If None, the model will be loaded from the HuggingFace hub.
            Defaults to None.
    """

    def __init__(self,
                 vlm_backbone_id: str,
                 vlm_config: Dict = None,
                 vlm_path: Optional[str] = None) -> None:
        super().__init__(vlm_backbone_id, vlm_config, vlm_path=vlm_path)

        if hasattr(self.vlm.config, 'attn_implementation'):
            self.vlm.config.attn_implementation = 'flash_attention_2'

        if hasattr(self.vlm.config, 'use_memory_efficient_attention'):
            self.vlm.config.use_memory_efficient_attention = True

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen2_5_VLDecoderLayer

    def forward(self,
                images: List[torch.Tensor],
                lang_tokens: torch.Tensor,
                img_masks: torch.Tensor,
                lang_masks: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                *args,
                **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass through the QWen2_5VL model.

        Args:
            images (List[torch.Tensor]): Input image tensor.
            lang_tokens (torch.Tensor): Input language tensor.
            img_masks [torch.Tensor]: Image attention mask.
            lang_masks (Optional[torch.Tensor]): Language attention mask.
            image_grid_thw (Optional[torch.LongTensor]): Image grid dimensions.

        Returns:
            CausalLMOutputWithPast: Output of the model containing logits
                and other information.
        """
        batch_size = images.shape[0]
        pixel_values = images
        num_images = image_grid_thw.shape[1]
        image_grid_thw = image_grid_thw.reshape(
            (batch_size * num_images, image_grid_thw.shape[-1]))
        img_emb = self.vlm.get_image_features(
            pixel_values, image_grid_thw)  # (B, L_img, 2048)
        img_emb = torch.stack(img_emb).reshape(batch_size, -1,
                                               img_emb[0].shape[-1])

        text_embeds = self.vlm.get_input_embeddings()(lang_tokens)
        inputs_embeds = torch.cat([img_emb, text_embeds],
                                  dim=1)  # (B,S+L_img,2048)
        attention_mask = torch.cat([
            torch.cat([
                img_masks[:, i:i + 1].repeat(1, img_emb.shape[1] // num_images)
                for i in range(img_masks.shape[1])
            ],
                      dim=1), lang_masks
        ],
                                   dim=1)
        outputs = self.vlm.model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            attention_mask=attention_mask,
            pixel_values=None)
        return outputs.last_hidden_state, attention_mask, attention_mask

    def get_fsdp_wrapping_policy(self) -> Callable:
        """xian
        Returns a function used to determine which modules to wrap with FSDP.

        Returns:
            Callable: Wrapping policy function.
        """
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        )
        return transformer_block_policy
