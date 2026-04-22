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


def resample_remaining(traj, offset):
    """Linearly interpolate remaining trajectory from a fractional offset.

    Args:
        traj: (N, D) sequential data (numpy array).
        offset: Fractional starting index, e.g. (t - t0) / dt.

    Returns:
        (M, D) resampled rows where M = N - int(offset).
    """
    N = traj.shape[0]
    M = N - int(offset)
    if M <= 0:
        return traj[:0]
    idx = np.clip(offset + np.arange(M), 0.0, N - 1.0)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, N - 1)
    alpha = (idx - lo)[:, np.newaxis]
    return traj[lo] + alpha * (traj[hi] - traj[lo])
