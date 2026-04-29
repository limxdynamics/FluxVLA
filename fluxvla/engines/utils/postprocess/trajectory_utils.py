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


def resample_remaining(traj: np.ndarray, offset: float) -> np.ndarray:
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


def compute_dynamic_horizon(
    actions: np.ndarray,
    config: Optional[dict] = None,
) -> Optional[int]:
    """Compute a motion-aware execution horizon.

    Analyzes the motion profile to ensure replanning only happens
    after real motion has begun.

    Args:
        actions: (N, D) action chunk.
        config: dict with keys:
            ``motion_threshold`` (default 0.05),
            ``min_motion_steps`` (default 3),
            ``max_horizon`` (default full trajectory).

    Returns:
        int number of steps to execute, or None if config is empty.
    """
    if not config:
        return None
    threshold = config.get('motion_threshold', 0.05)
    min_steps = config.get('min_motion_steps', 3)
    displacement = np.linalg.norm(actions - actions[0:1], axis=1)
    moving = np.nonzero(displacement > threshold)[0]
    if len(moving) > 0:
        horizon = int(moving[0]) + min_steps
    else:
        horizon = len(actions)
    cap = config.get('max_horizon', len(actions))
    return min(horizon, cap)
