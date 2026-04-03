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

import math
from functools import partial
from typing import Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma.modeling_gemma import (GemmaAttention,
                                                      GemmaDecoderLayer,
                                                      GemmaMLP, GemmaRMSNorm)

from fluxvla.engines import VLM_BACKBONES
from .hf_vlm import VLMBackbone


@VLM_BACKBONES.register_module()
class PaliGemma(VLMBackbone):
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
                 vlm_path: Optional[str] = None,
                 use_llm: Optional[bool] = False) -> None:
        super().__init__(vlm_backbone_id, vlm_config, vlm_path=vlm_path)
        # To handle fsdp wrapping problem.
        self.embed_tokens = nn.Embedding(vlm_config['vocab_size'],
                                         vlm_config['hidden_size'])
        del self.vlm.language_model.embed_tokens
        self.use_llm = use_llm

    def forward(self,
                images: List[torch.Tensor],
                lang_tokens: torch.Tensor,
                img_masks: torch.Tensor,
                lang_masks: Optional[torch.Tensor] = None,
                *args,
                **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass through the PaliGemma model.

        Args:
            images (List[torch.Tensor]): Input image tensor.
            lang_tokens (torch.Tensor): Input language tensor.
            img_masks [torch.Tensor]: Image attention mask.
            lang_masks (Optional[torch.Tensor]): Language attention mask.

        Returns:
            CausalLMOutputWithPast: Output of the model containing logits
                and other information.
        """
        embs = list()
        pad_masks = list()
        attn_masks = list()
        images = images.unflatten(1, (-1, 3)).permute(1, 0, 2, 3, 4)
        img_masks = img_masks.permute(1, 0)
        for image, img_mask in zip(images, img_masks):
            if hasattr(self.vlm, 'get_image_features'):
                img_emb = self.vlm.get_image_features(image)
            else:
                img_emb = self.vlm.model.get_image_features(image)
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            attn_masks += [0] * num_img_embs
        lang_emb = self.embed_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        attn_masks += [0] * num_lang_embs
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        attn_masks = torch.tensor(
            attn_masks, dtype=torch.bool, device=pad_masks.device)
        attn_masks = attn_masks[None, :].expand(bsize, len(attn_masks))
        if self.use_llm:
            outputs = self.vlm.model(
                inputs_embeds=embs,
                use_cache=False,
                attention_mask=pad_masks,
                pixel_values=None)
            embs = outputs.last_hidden_state
        return embs, pad_masks, attn_masks

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GemmaDecoderLayer

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns a function used to determine which modules to wrap with FSDP.

        Returns:
            Callable: Wrapping policy function.
        """
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GemmaAttention, GemmaMLP, GemmaRMSNorm},
        )
        return transformer_block_policy
