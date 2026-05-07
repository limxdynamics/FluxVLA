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

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    t0: float
    dt: float
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None

    def __post_init__(self):
        for name in ('velocities', 'accelerations'):
            arr = getattr(self, name)
            if arr is not None and arr.shape != self.positions.shape:
                raise ValueError(
                    f'{name} shape {arr.shape} != positions shape '
                    f'{self.positions.shape}')


def broadcast(v, n: int) -> list[float]:
    """Broadcast a scalar or sequence to a list of length *n*."""
    if isinstance(v, (int, float)):
        return [float(v)] * n
    return [float(x) for x in v]
