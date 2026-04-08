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

from typing import Dict, Sequence, Type

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from fluxvla.engines import LLM_BACKBONES
from .hf_causal_llm import HFCausalLLMBackbone


@LLM_BACKBONES.register_module()
class LLaMa2LLMBackbone(HFCausalLLMBackbone):
    """
    HuggingFace-compatible wrapper for LLaMa-2 LLMs.

    Inherits from HFCausalLLMBackbone and provides LLaMa-2 specific token
    and configuration handling. This includes explicitly adding a padding
    token to the tokenizer and resizing the model embeddings accordingly.

    Registered into the `LLM_BACKBONES` registry for flexible instantiation.

    Args:
        llm_backbone_id (str): Identifier string for this backbone.
        llm_family (str): Family/type name (e.g., "llama2").
        llm_path (str): HF model ID or local checkpoint path.
        llm_max_length (int, optional): Maximum sequence length supported.
        hf_token (Optional[str], optional): HuggingFace auth token.
        inference_mode (bool, optional): Whether to load in inference mode.
        pad_token_id (int, optional): ID for the padding token. Defaults to 0.
        tokenizer_length (int, optional): Length of the tokenizer's
            vocabulary. Defaults to 32000.
    """

    def __init__(self,
                 llm_backbone_id: str,
                 llm_family: str,
                 llm_path: str,
                 llm_config: Dict = None,
                 llm_max_length: int = 2048,
                 hf_token: str = None,
                 inference_mode: bool = False,
                 pad_token_id: int = 0,
                 tokenizer_length: int = 32001) -> None:
        super().__init__(
            llm_backbone_id,
            llm_family=llm_family,
            llm_path=llm_path,
            llm_config=llm_config,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode)

        # Add a PAD token explicitly for LLaMa-2

        # Assign PAD token ID in config
        self.llm.config.pad_token_id = pad_token_id

        # Resize model's token embeddings with padding
        self.llm.resize_token_embeddings(
            tokenizer_length, pad_to_multiple_of=64)

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1],
                self.llm.lm_head)
