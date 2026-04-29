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
from .trajectory_utils import Trajectory


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
        pos = prev_traj.positions[-1]
        if pos.shape[0] != len(dof_indices):
            pos = pos[dof_indices]
        return pos, None, None

    lo, hi = min(int(offset), n - 1), min(int(offset) + 1, n - 1)
    alpha = offset - lo

    def _lerp(arr):
        if arr is None:
            return None
        return (1 - alpha) * arr[lo] + alpha * arr[hi]

    def _select(x):
        if x is None:
            return None
        if x.shape[0] == len(dof_indices):
            return x
        return x[dof_indices]

    return (
        _select(_lerp(prev_traj.positions)),
        _select(_lerp(prev_traj.velocities)),
        _select(_lerp(prev_traj.accelerations)),
    )


class _JointMpcMethod:

    def process(self, traj: Trajectory, prev_traj: Optional[Trajectory],
                config: Dict[str, Any], dof_indices: list[int]) -> Trajectory:
        if len(dof_indices) == 0:
            raise ValueError('dof_indices must not be empty')
        dof_array = np.array(dof_indices, dtype=np.int64)

        init_pos, init_vel, init_acc = _lerp_initial_state(
            prev_traj, traj.t0, dof_array)
        if init_pos is None:
            init_pos = traj.positions[0, dof_array]
            init_vel = np.zeros_like(init_pos)
            init_acc = np.zeros_like(init_pos)

        pos_dof, vel_dof, acc_dof = joint_mpc(
            targets=traj.positions[:, dof_array],
            dt=traj.dt,
            init_position=init_pos,
            init_velocity=init_vel,
            init_acceleration=init_acc,
            max_velocity=config.get('max_velocity', 3.0),
            max_acceleration=config.get('max_acceleration', 10.0),
            max_jerk=config.get('max_jerk', 50.0),
            num_stitch=int(config.get('num_stitch', 0)),
            tracking_weight=float(config.get('tracking_weight', 1.0)),
        )

        traj.positions = traj.positions.copy()
        traj.positions[:, dof_array] = pos_dof
        traj.velocities = np.zeros_like(traj.positions)
        traj.velocities[:, dof_array] = vel_dof
        traj.accelerations = np.zeros_like(traj.positions)
        traj.accelerations[:, dof_array] = acc_dof

        return traj


class _RuckigMethod:

    def process(self, traj: Trajectory, prev_traj: Optional[Trajectory],
                config: Dict[str, Any], dof_indices: list[int]) -> Trajectory:
        from .ruckig_filter import ruckig_filter

        if len(dof_indices) == 0:
            raise ValueError('dof_indices must not be empty')
        dof_array = np.array(dof_indices, dtype=np.int64)

        init_pos, init_vel, init_acc = _lerp_initial_state(
            prev_traj, traj.t0, dof_array)
        if init_pos is None:
            init_pos = traj.positions[0, dof_array]
            init_vel = np.zeros_like(init_pos)
            init_acc = np.zeros_like(init_pos)

        pos_dof, vel_dof, acc_dof = ruckig_filter(
            targets=traj.positions[1:, dof_array],
            dt=traj.dt,
            init_position=init_pos,
            init_velocity=init_vel,
            init_acceleration=init_acc,
            max_velocity=config.get('max_velocity', 3.0),
            max_acceleration=config.get('max_acceleration', 10.0),
            max_jerk=config.get('max_jerk', 50.0),
            max_settle_steps=config.get('max_settle_steps', 0),
        )

        traj.positions = traj.positions.copy()
        traj.positions[:, dof_array] = pos_dof[:len(traj.positions)]
        traj.velocities = np.zeros_like(traj.positions)
        traj.velocities[:, dof_array] = vel_dof[:len(traj.positions)]
        traj.accelerations = np.zeros_like(traj.positions)
        traj.accelerations[:, dof_array] = acc_dof[:len(traj.positions)]

        return traj


class TrajectoryPostprocessor:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        method_name = self.config.get('method', 'joint_mpc')
        if method_name == 'joint_mpc':
            self.method = _JointMpcMethod()
        elif method_name == 'ruckig':
            try:
                from .ruckig_filter import ruckig_filter  # noqa: F401
            except ImportError:
                raise ImportError(
                    'ruckig package is required for the ruckig backend. '
                    'Install with: pip install ruckig')
            self.method = _RuckigMethod()
        else:
            raise ValueError(f'Unknown postprocess method: {method_name}')

    def process(self,
                traj: Trajectory,
                dof_indices: list[int],
                prev_traj: Optional[Trajectory] = None) -> Trajectory:
        if not self.config.get('enabled', False):
            return Trajectory(
                t0=traj.t0,
                dt=traj.dt,
                positions=traj.positions.copy(),
                velocities=traj.velocities.copy()
                if traj.velocities is not None else None,
                accelerations=traj.accelerations.copy()
                if traj.accelerations is not None else None,
            )

        return self.method.process(
            traj=traj,
            prev_traj=prev_traj,
            config=self.config,
            dof_indices=dof_indices,
        )
