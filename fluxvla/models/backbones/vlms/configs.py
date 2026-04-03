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

from transformers import (PaliGemmaConfig, PaliGemmaForConditionalGeneration,
                          Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration)

VLM_BACKBONE_CONFIGS = dict(
    paligemma_3b_pt_224=dict(
        model_id='paligemma-3b-pt-224',
        config=PaliGemmaConfig,
        model_cls=PaliGemmaForConditionalGeneration,
    ),
    qwen2_5_3b_vl_pt_224=dict(
        model_id='qwen2-5-vl_pt_224',
        config=Qwen2_5_VLConfig,
        model_cls=Qwen2_5_VLForConditionalGeneration,
    ))
