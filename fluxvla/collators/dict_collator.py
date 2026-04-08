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
class DictCollator:

    def __init__(self, keys=None, meta_keys=None):
        """
        Args:
            keys (List[str]): Keys that should be stacked
                (e.g., ndarray, tensor).
            meta_keys (List[str]): Keys that should be left
                as list (e.g., strings, info).
        """
        self.keys = keys or []
        self.meta_keys = meta_keys or []

    def __call__(self, batch):
        """
        Collate function to be passed to DataLoader.
        Args:
            batch (List[Dict]): A list of sample dictionaries
        Returns:
            Dict[str, Any]: A batched dictionary
        """
        collated = {}
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]
            first_val = values[0]

            # stack keys
            if key in self.keys:
                if isinstance(first_val, np.ndarray):
                    collated[key] = torch.from_numpy(np.stack(values))
                elif isinstance(first_val, torch.Tensor):
                    collated[key] = torch.stack(values)
                else:
                    raise TypeError(
                        f"Key '{key}' marked as stackable, but got unsupported type {type(first_val)}"  # noqa: E501
                    )

            # meta keys: keep as list
            elif key in self.meta_keys:
                collated[key] = values

            # fallback
            else:
                raise KeyError(
                    f"Key '{key}' is not in keys or meta_keys. Please specify its behavior."  # noqa: E501
                )

        return collated
