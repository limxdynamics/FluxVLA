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

import numpy as np
import torch

from fluxvla.engines import COLLATORS


@COLLATORS.register_module()
class NestedCollator:

    def __init__(self):
        pass

    def _nested_stack(self, items):
        """
        Recursively collate a list of items into a nested dictionary structure.
        This function handles different types of items, including dictionaries,
        numpy arrays, and PyTorch tensors. It stacks items along the
        first dimension for arrays and tensors, and collects values
        from dictionaries into lists.

        Args:
            items (list): A list of items to collate. Each item can be
                a dictionary, numpy array, or PyTorch tensor.

        Returns:
            dict: A nested dictionary where each key corresponds to a
                list of values from the input items. If the items
                are numpy arrays or tensors, they are stacked along
                the first dimension.
        """
        if isinstance(items[0], dict):
            return {
                k: self._nested_stack([d[k] for d in items])
                for k in items[0]
            }
        elif isinstance(items[0], np.ndarray):
            return np.stack(items, axis=0)
        elif torch.is_tensor(items[0]):
            return torch.stack(items, dim=0)

    def __call__(self, items):
        """
        Collate a list of items into a nested dictionary structure.

        Args:
            items (list): A list of dictionaries to collate.

        Returns:
            dict: A nested dictionary where each key corresponds to a
                list of values from the input dictionaries.
        """
        return self._nested_stack(items)
