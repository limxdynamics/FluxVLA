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
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines import LLM_BACKBONES
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .configs import LLM_BACKBONE_CONFIGS

overwatch = initialize_overwatch(__name__)


@LLM_BACKBONES.register_module()
class HFCausalLLMBackbone(nn.Module):
    """HuggingFace Causal LLM Backbone
    This class is a wrapper around HuggingFace's Causal LLMs. It provides
    an interface for loading, tokenizing, and processing inputs for
    various LLMs. The class is designed to be extensible and can be
    customized for different LLM architectures.

    Args:
        llm_backbone_id (str): Unique identifier for the LLM backbone.
        llm_family (str): Family of the LLM (e.g., "llama2", "mistral").
        llm_path (str): Path to the LLM model. This can be a local path or
            a HuggingFace Hub path.
        llm_max_length (int, optional): Maximum length of the input
            sequences. Defaults to 2048.
        hf_token (str, optional): HuggingFace token for authentication.
        inference_mode (bool, optional): If True, the model is loaded in
            inference mode. Defaults to False.
    """

    def __init__(self,
                 llm_backbone_id: str,
                 llm_family: str,
                 llm_path: str = None,
                 llm_config: Dict = None,
                 llm_max_length: int = 2048,
                 hf_token: Optional[str] = None,
                 inference_mode: bool = False) -> None:
        super().__init__()
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode

        model_cls = LLM_BACKBONE_CONFIGS[llm_backbone_id]['model_cls']
        llm_cfg = LLM_BACKBONE_CONFIGS[llm_backbone_id]['config']

        # Initialize LLM (downloading from HF Hub if necessary) --> `model_cls`
        # is the actual {Model}ForCausalLM class!
        #   => Note: We're eschewing use of the AutoModel API so that we can be
        # more explicit about LLM-specific details
        if not self.inference_mode:
            overwatch.info(
                f'Loading [bold]{llm_family}[/] LLM from [underline]`{llm_path}`[/]',  # noqa: E501,W604
                ctx_level=1)
            if llm_config is None:
                llm_config = llm_cfg.from_pretrained(llm_path, token=hf_token)
                assert llm_path is not None, \
                    'If not in inference mode, `llm_path` must be provided!'
                self.llm = model_cls.from_pretrained(
                    llm_path, config=llm_config, trust_remote_code=True)
            else:
                llm_config = llm_cfg(**llm_config)
                if llm_path is not None:
                    self.llm = model_cls.from_pretrained(
                        llm_path, config=llm_config)
                else:
                    self.llm = model_cls(llm_config)

        # [Contract] `inference_mode` means we're loading from a pretrained checkpoint;  # noqa: E501
        # no need to load base weights!
        else:
            overwatch.info(
                f'Building empty [bold]{llm_family}[/] LLM from [underline]`{llm_path}`[/]',  # noqa: E501,W604
                ctx_level=1)
            llm_config = AutoConfig.from_pretrained(llm_path, token=hf_token)
            self.llm = model_cls._from_config(llm_config)

        self.llm.config.use_cache = False if not self.inference_mode else True
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        # Load (Fast) Tokenizer
        overwatch.info(
            f'Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API',  # noqa: E501
            ctx_level=1)

        self.config = self.llm.config

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a `transformer_auto_wrap_policy` where we wrap each instance of `self.transformer_layer_cls`"""  # noqa: E501
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self.transformer_layer_cls})

        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        """Dispatch to underlying LLM instance's `gradient_checkpointing_enable`; defined for all `PretrainedModel`."""  # noqa: E501
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    # [Contract] Should match the `forward` call of the underlying
    # `llm` instance!
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
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        return output

    @property
    def pad_token_id(self) -> int:
        """
        Returns the pad token ID used by the tokenizer.

        Returns:
            int: Pad token ID.
        """
        return self.tokenizer.pad_token_id
