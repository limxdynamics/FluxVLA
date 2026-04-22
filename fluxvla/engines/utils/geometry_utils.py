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


def rotmat_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a 6D rotation representation."""
    rot_mat = np.asarray(rot_mat, dtype=np.float32)
    return np.concatenate([rot_mat[:3, 0], rot_mat[:3, 1]],
                          axis=-1).astype(np.float32)
