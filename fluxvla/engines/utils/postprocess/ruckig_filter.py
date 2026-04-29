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
"""Generic Ruckig-based jerk-constrained trajectory filter.

Usage::

    # Tracking mode (real-time streaming, output length == N)
    positions, velocities, accelerations = ruckig_filter(
        targets=actions[:, dof_indices],
        dt=0.02,
        init_position=qpos,
        max_velocity=5.0, max_acceleration=20.0, max_jerk=20.0,
    )

    # Settle mode (keep stepping until final target reached at rest)
    positions, velocities, accelerations = ruckig_filter(
        targets=waypoints,
        dt=0.02,
        max_velocity=5.0, max_acceleration=20.0, max_jerk=20.0,
        max_settle_steps=1000,
    )
"""

from typing import Sequence, Union

import numpy as np

from .trajectory_utils import broadcast


def ruckig_filter(
    targets: np.ndarray,
    dt: float,
    init_position: np.ndarray = None,
    init_velocity: np.ndarray = None,
    init_acceleration: np.ndarray = None,
    max_velocity: Union[float, Sequence[float]] = 3.0,
    max_acceleration: Union[float, Sequence[float]] = 10.0,
    max_jerk: Union[float, Sequence[float]] = 50.0,
    max_settle_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter a trajectory through Ruckig to enforce jerk constraints.

    Stateless: all necessary state is passed in, all results are returned.

    Args:
        targets: (N, n_dof) desired waypoints.
        dt: Control timestep in seconds.
        init_position: (n_dof,) start position.  Defaults to targets[0].
        init_velocity: (n_dof,) start velocity.  Defaults to zeros.
        init_acceleration: (n_dof,) start acceleration.  Defaults to zeros.
        max_velocity: Per-DOF velocity limit.  Scalar is broadcast.
        max_acceleration: Per-DOF acceleration limit.  Scalar is broadcast.
        max_jerk: Per-DOF jerk limit.  Scalar is broadcast.
        max_settle_steps: Maximum extra steps after the last target to
            continue converging toward the final position at rest.
            0 (default) means no extra steps.  Extra settle points are
            resampled back to the original output length so the result
            shape is always (N + 1, n_dof) (init + N target steps).
    """
    from ruckig import InputParameter, OutputParameter, Result, Ruckig

    n_dof = targets.shape[1]

    otg = Ruckig(n_dof, dt)
    inp = InputParameter(n_dof)
    out = OutputParameter(n_dof)

    # When no init_position is given, use targets[0] as the starting point
    # and iterate over the remaining targets.
    if init_position is None:
        init_position = targets[0]
        targets = targets[1:]

    inp.current_position = np.asarray(init_position, dtype=float).tolist()
    inp.current_velocity = (
        np.asarray(init_velocity, dtype=float).tolist()
        if init_velocity is not None else [0.0] * n_dof)
    inp.current_acceleration = (
        np.asarray(init_acceleration, dtype=float).tolist()
        if init_acceleration is not None else [0.0] * n_dof)

    inp.max_velocity = broadcast(max_velocity, n_dof)
    inp.max_acceleration = broadcast(max_acceleration, n_dof)
    inp.max_jerk = broadcast(max_jerk, n_dof)

    # positions[0] = init state
    pos_list = [np.asarray(inp.current_position)]
    vel_list = [np.asarray(inp.current_velocity)]
    acc_list = [np.asarray(inp.current_acceleration)]

    n_targets = len(targets)
    for i in range(n_targets):
        inp.target_position = targets[i].tolist()
        inp.target_velocity = [0.0] * n_dof
        inp.target_acceleration = [0.0] * n_dof

        otg.update(inp, out)
        pos_list.append(list(out.new_position))
        vel_list.append(list(out.new_velocity))
        acc_list.append(list(out.new_acceleration))
        out.pass_to_input(inp)

    # Settle: keep stepping on the final target until reached or budget used
    if max_settle_steps > 0:
        inp.target_position = targets[-1].tolist()
        inp.target_velocity = [0.0] * n_dof
        inp.target_acceleration = [0.0] * n_dof
        for _ in range(max_settle_steps):
            if otg.update(inp, out) != Result.Working:
                break
            pos_list.append(list(out.new_position))
            vel_list.append(list(out.new_velocity))
            acc_list.append(list(out.new_acceleration))
            out.pass_to_input(inp)

    positions = np.array(pos_list)
    velocities = np.array(vel_list)
    accelerations = np.array(acc_list)

    # Resample settle points back to original length (init + n_targets).
    expected_len = n_targets + 1
    if len(positions) > expected_len:
        src_idx = np.linspace(0, len(positions) - 1, expected_len)
        lo = np.floor(src_idx).astype(int)
        hi = np.minimum(lo + 1, len(positions) - 1)
        alpha = (src_idx - lo)[:, np.newaxis]
        positions = positions[lo] + alpha * (positions[hi] - positions[lo])
        velocities = velocities[lo] + alpha * (velocities[hi] - velocities[lo])
        accelerations = (
            accelerations[lo] + alpha *
            (accelerations[hi] - accelerations[lo]))

    return positions, velocities, accelerations
