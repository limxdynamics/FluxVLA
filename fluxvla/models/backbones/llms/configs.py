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

from transformers import (GemmaConfig, GemmaForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel, MistralConfig,
                          MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                          Qwen2Model)

LLM_BACKBONE_CONFIGS = {
    'mistral-v0.1-7b-pure_causal': {
        'config': MistralConfig,
        'llm_family': 'mistral',
        'model_cls': MistralForCausalLM,
        'hf_hub_path': 'mistralai/Mistral-7B-v0.1'
    },

    # === Mistral Instruct v0.1 ===
    'mistral-v0.1-7b-instruct_causal': {
        'config': MistralConfig,
        'llm_family': 'mistral',
        'model_cls': MistralForCausalLM,
        'hf_hub_path': 'mistralai/Mistral-7B-Instruct-v0.1'
    },
    'llama2-7b-pure_causal': {
        'config': LlamaConfig,
        'llm_family': 'llama2',
        'model_cls': LlamaForCausalLM,
        'hf_hub_path': 'meta-llama/Llama-2-7b-hf'
    },
    'llama2-7b-pure': {
        'config': LlamaConfig,
        'llm_family': 'llama2',
        'model_cls': LlamaModel,
        'hf_hub_path': 'meta-llama/Llama-2-7b-hf'
    },
    'llama2-13b-pure_causal': {
        'config': LlamaConfig,
        'llm_family': 'llama2',
        'model_cls': LlamaForCausalLM,
        'hf_hub_path': 'meta-llama/Llama-2-13b-hf'
    },

    # === Meta LLaMa-2 Chat Models ===
    'llama2-7b-chat_causal': {
        'config': LlamaConfig,
        'llm_family': 'llama2',
        'model_cls': LlamaForCausalLM,
        'hf_hub_path': 'meta-llama/Llama-2-7b-chat-hf'
    },
    'llama2-13b-chat_causal': {
        'config': LlamaConfig,
        'llm_family': 'llama2',
        'model_cls': LlamaForCausalLM,
        'hf_hub_path': 'meta-llama/Llama-2-13b-chat-hf'
    },

    # === Vicuna v1.5 Chat Models ===
    'vicuna-v15-7b_causal': {
        'config': LlamaConfig,
        'llm_family': 'llama2',
        'model_cls': LlamaForCausalLM,
        'hf_hub_path': 'lmsys/vicuna-7b-v1.5'
    },
    'vicuna-v15-13b_causal': {
        'config': LlamaConfig,
        'llm_family': 'llama2',
        'model_cls': LlamaForCausalLM,
        'hf_hub_path': 'lmsys/vicuna-13b-v1.5'
    },

    # === Gemma Models ===
    'gemma-2b_causal': {
        'config': GemmaConfig,
        'llm_family': 'gemma',
        'model_cls': GemmaForCausalLM,
        'hf_hub_path': 'google/gemma-2b'
    },

    # === Qwen2 Models ===
    'qwen2-0.5b': {
        'config': Qwen2Config,
        'llm_family': 'qwen2',
        'model_cls': Qwen2Model,
        'hf_hub_path': 'Qwen/Qwen2-0.5B'
    },
    'qwen2-3b': {
        'config': Qwen2Config,
        'llm_family': 'qwen2',
        'model_cls': Qwen2Model,
        'hf_hub_path': 'Qwen/Qwen2-0.5B'
    },
    'qwen2_5-7b': {
        'config': Qwen2Config,
        'llm_family': 'qwen2',
        'model_cls': Qwen2Model,
        'hf_hub_path': 'Qwen/Qwen2_5-7B'
    },
    'qwen2-0.5b_causal': {
        'config': Qwen2Config,
        'llm_family': 'qwen2',
        'model_cls': Qwen2ForCausalLM,
        'hf_hub_path': 'Qwen/Qwen2-0.5B'
    },
    'qwen2_5-7b_causal': {
        'config': Qwen2Config,
        'llm_family': 'qwen2',
        'model_cls': Qwen2ForCausalLM,
        'hf_hub_path': 'Qwen/Qwen2_5-7B'
    },
}
