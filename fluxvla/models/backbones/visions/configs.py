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

from transformers import (Dinov2Config, SiglipVisionConfig, SiglipVisionModel,
                          ViTForImageClassification)

VISION_BACKBONE_CONFIGS = dict(
    dino=dict(
        model_id='vit_large_patch14_reg4_dinov2',
        config=Dinov2Config,
        model_cls=ViTForImageClassification,
    ),
    siglip_224=dict(
        model_id='vit_so400m_patch14_siglip_224',
        config=SiglipVisionConfig,
        model_cls=SiglipVisionModel,
    ),
    siglip_384=dict(
        model_id='vit_so400m_patch14_siglip_384',
        config=SiglipVisionConfig,
        model_cls=SiglipVisionModel,
    ),
)
