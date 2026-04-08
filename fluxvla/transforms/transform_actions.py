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

from typing import Dict, List

import numpy as np

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class ProcessLiberoActions:

    def __init__(self, mask: List[bool] = None) -> None:
        """ProcessLiberoActions is a transform
        that modifies the actions in the data
        by subtracting the state values from
        the actions based on a mask.

        Args:
            mask (List[bool], optional): A list
                indicating which dimensions
                of the state should be subtracted from
                the actions.
                If None, no subtraction is performed.
        """
        self.mask = np.asarray(mask, dtype=bool)

    def __call__(self, data: Dict) -> Dict:
        if 'actions' not in data or self.mask is None:
            return data

        states, actions = data['states'], data['actions']
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, states[..., :dims], 0), axis=-2)
        data['actions'] = actions

        return data
