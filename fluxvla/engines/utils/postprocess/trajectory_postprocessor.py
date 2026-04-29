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

from typing import Any, Dict, Optional

import numpy as np

from .joint_mpc import joint_mpc
from .ruckig_filter import ruckig_filter
from .trajectory import Trajectory


def _lerp_initial_state(
    prev_traj: Optional[Trajectory],
    reference_t0: float,
    dof_indices: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if prev_traj is None:
        return None, None, None

    elapsed = reference_t0 - prev_traj.t0
    offset = max(0.0, elapsed / prev_traj.dt)
    n = len(prev_traj.positions)

    if offset >= n:
        return prev_traj.positions[-1][dof_indices], np.zeros_like(
            prev_traj.positions[-1][dof_indices]), None

    lo, hi = min(int(offset), n - 1), min(int(offset) + 1, n - 1)
    alpha = offset - lo

    def _lerp(arr):
        if arr is None:
            return None
        return ((1 - alpha) * arr[lo] + alpha * arr[hi])[dof_indices]

    return (
        _lerp(prev_traj.positions),
        _lerp(prev_traj.velocities),
        _lerp(prev_traj.accelerations),
    )


def _resample_at_times(
    data: np.ndarray,
    t0: float,
    dt: float,
    times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate rows of ``data`` at absolute ``times``."""
    offsets = (times - t0) / dt
    n = len(data)
    offsets = np.clip(offsets, 0.0, n - 1.001)

    lo = np.floor(offsets).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    alpha = (offsets - lo)[:, np.newaxis]

    return data[lo] + alpha * (data[hi] - data[lo])


def _extract_stitch(
    prev_traj: Trajectory,
    t0: float,
    dt: float,
    num_stitch: int,
    dof_array: np.ndarray,
    max_len: int,
) -> tuple[int, Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray]]:
    """Extract rule-based stitch bridge from ``prev_traj``."""
    if num_stitch <= 0 or prev_traj is None:
        return 0, None, None, None

    S = min(num_stitch, max_len)
    times = t0 + np.arange(S) * dt
    stitch_pos = _resample_at_times(
        prev_traj.positions,
        prev_traj.t0,
        prev_traj.dt,
        times,
    )[:, dof_array]
    stitch_vel = _resample_at_times(
        prev_traj.velocities,
        prev_traj.t0,
        prev_traj.dt,
        times,
    )[:, dof_array] if prev_traj.velocities is not None else np.zeros_like(
        stitch_pos)
    stitch_acc = _resample_at_times(
        prev_traj.accelerations,
        prev_traj.t0,
        prev_traj.dt,
        times,
    )[:, dof_array] if prev_traj.accelerations is not None else np.zeros_like(
        stitch_pos)
    return S, stitch_pos, stitch_vel, stitch_acc


class TrajectoryPostprocessor:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._method = self.config.get('method', 'joint_mpc')
        if self._method not in ('joint_mpc', 'ruckig'):
            raise ValueError(f'Unknown postprocess method: {self._method}')

        self._mode = self.config.get('mode', 'tracking')
        if self._mode not in ('tracking', 'settle'):
            raise ValueError(f'Unknown mode: {self._mode!r}. '
                             f"Expected 'tracking' or 'settle'.")

    def process(self,
                traj: Trajectory,
                dof_indices: list[int],
                prev_traj: Optional[Trajectory] = None) -> Trajectory:
        if not self.config.get('enabled', False):
            return traj.copy()

        if len(dof_indices) == 0:
            raise ValueError('dof_indices must not be empty')
        dof_array = np.array(dof_indices, dtype=np.int64)

        full_targets = traj.positions[:, dof_array].copy()
        num_stitch = int(self.config.get('num_stitch', 0))
        S, stitch_pos, stitch_vel, stitch_acc = _extract_stitch(
            prev_traj, traj.t0, traj.dt, num_stitch, dof_array,
            len(full_targets))

        # ── Init state (at stitch boundary if S>0, else at traj.t0) ───────
        ref_t0 = traj.t0 + S * traj.dt if S > 0 else traj.t0
        init_pos, init_vel, init_acc = _lerp_initial_state(
            prev_traj, ref_t0, dof_array)
        if init_pos is None:
            init_pos = full_targets[S]
            init_vel = np.zeros_like(init_pos)
            init_acc = None

        targets = full_targets[S:]

        # ── Solver ────────────────────────────────────────────────────────
        tracking_weight = float(self.config.get('tracking_weight', 3.0))
        max_vel = self.config.get('max_velocity', 5.0)
        max_acc = float(self.config.get('max_acceleration', 20.0))
        max_jerk = float(self.config.get('max_jerk', 20.0))

        if self._method == 'joint_mpc':
            pos_dof, vel_dof, acc_dof = joint_mpc(
                targets=targets,
                dt=traj.dt,
                init_position=init_pos,
                init_velocity=init_vel,
                init_acceleration=init_acc,
                max_velocity=max_vel,
                max_acceleration=max_acc,
                max_jerk=max_jerk,
                tracking_weight=tracking_weight,
                terminal_weight=float(
                    self.config.get(
                        'terminal_weight',
                        1.0 if self._mode == 'settle' else 0.0,
                    )),
                settle_weight=float(
                    self.config.get(
                        'settle_weight',
                        1.0 if self._mode == 'settle' else 0.0,
                    )),
            )
        else:
            pos_dof, vel_dof, acc_dof = ruckig_filter(
                targets=targets,
                dt=traj.dt,
                init_position=init_pos,
                init_velocity=init_vel,
                init_acceleration=init_acc,
                max_velocity=max_vel,
                max_acceleration=max_acc,
                max_jerk=max_jerk,
                max_settle_steps=0 if self._mode == 'tracking' else
                self.config.get('max_settle_steps', 15),
                resample_settle=self.config.get('resample_settle', True),
            )

        # ── Assemble: bridge (rule-based) + solver output ─────────────────
        N = len(traj.positions)
        traj.positions = traj.positions.copy()
        traj.velocities = np.full_like(traj.positions, np.nan)
        traj.accelerations = np.full_like(traj.positions, np.nan)

        if S > 0:
            traj.positions[:, dof_array] = np.concatenate(
                [stitch_pos, pos_dof[:N - S]])
            traj.velocities[:, dof_array] = np.concatenate(
                [stitch_vel, vel_dof[:N - S]])
            traj.accelerations[:, dof_array] = np.concatenate(
                [stitch_acc, acc_dof[:N - S]])
        else:
            traj.positions[:, dof_array] = pos_dof[:N]
            traj.velocities[:, dof_array] = vel_dof[:N]
            traj.accelerations[:, dof_array] = acc_dof[:N]

        return traj
