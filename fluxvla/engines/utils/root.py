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

from .registry import Registry

TOKENIZERS = Registry('tokenizers')
TRANSFORMS = Registry('transforms')
DATASETS = Registry('datasets')
LLM_BACKBONES = Registry('llm_backbones')
VISION_BACKBONES = Registry('vision_backbones')
PROJECTORS = Registry('projectors')
HEADS = Registry('heads')
VLAS = Registry('vlas')
RUNNERS = Registry('runners')
COLLATORS = Registry('collators')
METRICS = Registry('metrics')
PROCESSORS = Registry('processors')
VLM_BACKBONES = Registry('vlm_backbones')
OPERATORS = Registry('operators')
