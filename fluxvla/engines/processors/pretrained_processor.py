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

from transformers import AutoProcessor

from ..utils import PROCESSORS


@PROCESSORS.register_module()
class PretrainedProcessor:

    def __init__(self, model_path: str, trust_remote_code: bool = True):
        """Load pretrained processor from the specified model path.

        Args:
            model_path (str): Path of the model to load the processor from.
            trust_remote_code (bool, optional): Whether to trust remote code.
                Defaults to True.
        """
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)

    def __call__(self, *args, **kwds):
        return self.processor(*args, **kwds)

    def save_pretrained(self, save_directory: str):
        """Save the processor to the specified directory.

        Args:
            save_directory (str): Directory where the processor will be saved.
        """
        self.processor.save_pretrained(save_directory)
