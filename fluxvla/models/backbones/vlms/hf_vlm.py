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

from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configs import VLM_BACKBONE_CONFIGS


class VLMBackbone(nn.Module):

    def __init__(self,
                 vlm_backbone_id: str,
                 vlm_config: Dict,
                 vlm_path: Optional[str] = None):
        super().__init__()
        self.vlm_backbone_id = vlm_backbone_id
        vlm_cls = VLM_BACKBONE_CONFIGS[vlm_backbone_id]['model_cls']
        vlm_cfg = VLM_BACKBONE_CONFIGS[vlm_backbone_id]['config']
        if vlm_config is None:
            assert vlm_path is not None, 'vlm_config must be provided if vlm_pretrained_config is specified'  # noqa: E501
            vlm_config = vlm_cfg.from_pretrained(vlm_path)
            self.vlm = vlm_cls.from_pretrained(vlm_path, config=vlm_config)
        else:
            vlm_config = vlm_cfg(**vlm_config)
            if vlm_path is not None:
                self.vlm = vlm_cls.from_pretrained(vlm_path, config=vlm_config)
            else:
                self.vlm = vlm_cls(config=vlm_config)

        self.config = self.vlm.config

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns a function used to determine which modules to wrap with FSDP.

        Returns:
            Callable: Wrapping policy function.
        """
        ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """
        Enables gradient checkpointing on the LLM model to save memory.
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the model.

        Args:
            input_ids (Optional[torch.LongTensor]): Token IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            position_ids (Optional[torch.LongTensor]): Position indices.
            past_key_values (Optional[List[torch.FloatTensor]]):
                Cached KV pairs.
            inputs_embeds (Optional[torch.FloatTensor]): Embedded inputs.
            labels (Optional[torch.LongTensor]): Labels for loss computation.
            use_cache (Optional[bool]): Whether to use cache.
            output_attentions (Optional[bool]): Return attention scores.
            output_hidden_states (Optional[bool]): Return hidden states.
            return_dict (Optional[bool]): Return output as dict.

        Returns:
            CausalLMOutputWithPast: Model output.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Embeds input token IDs using the model's embedding layer.

        Args:
            input_ids (torch.LongTensor): Input token IDs.

        Returns:
            torch.Tensor: Embedded token vectors.
        """
        ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> Type[nn.Module]:
        """
        Returns the class of the transformer's basic layer.

        Returns:
            Type[nn.Module]: Transformer block class.
        """
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        """
        Returns the appropriate dtype for half precision (e.g., bf16 or fp16).

        Returns:
            torch.dtype: Half precision type.
        """
        ...

    @property
    @abstractmethod
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        """
        Returns a sequence of modules used in last-layer finetuning.

        Returns:
            Sequence[nn.Module]: Modules to be fine-tuned.
        """
        ...

    @property
    def embed_dim(self) -> int:
        """
        Returns the model's hidden size (embedding dimension).

        Returns:
            int: Embedding dimension.
        """
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        """
        Returns the pad token ID used by the tokenizer.

        Returns:
            int: Pad token ID.
        """
        return self.tokenizer.pad_token_id
