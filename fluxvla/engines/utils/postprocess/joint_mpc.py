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
"""Per-joint trajectory MPC via OSQP.

Formulates trajectory smoothing as a QP per joint with explicit
position / velocity / acceleration / jerk variables, triple-integrator
dynamics, and hard constraints on all derivatives.

Usage::

    positions, velocities, accelerations = joint_mpc(
        targets=actions[:, dof_indices],
        dt=0.02,
        init_position=qpos,
        max_velocity=5.0, max_acceleration=20.0, max_jerk=50.0,
    )
"""

from typing import Sequence, Union

import numpy as np
from scipy import sparse

from .trajectory_utils import broadcast

try:
    import osqp
except ImportError:
    osqp = None


def joint_mpc(
    targets: np.ndarray,
    dt: float,
    init_position: np.ndarray = None,
    init_velocity: np.ndarray = None,
    init_acceleration: np.ndarray = None,
    max_velocity: Union[float, Sequence[float]] = 3.0,
    max_acceleration: Union[float, Sequence[float]] = 10.0,
    max_jerk: Union[float, Sequence[float]] = 50.0,
    num_stitch: int = 0,
    tracking_weight: float = 1.0,
    reg_weight: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-joint MPC smoothing via OSQP.

    Each joint is solved as an independent QP.  Decision variables are
    internally scaled (v_s = v*dt, a_s = a*dt^2, j_s = j*dt^2) so that
    the dynamics have simple coefficients:

        q[t+1] - q[t] - v_s[t] = 0
        v_s[t+1] - v_s[t] - a_s[t] = 0
        a_s[t+1] - a_s[t] - j_s[t]*dt = 0

    Args:
        targets: (N, n_dof) desired waypoints.
        dt: Timestep in seconds.
        init_position: (n_dof,) start position. Defaults to targets[0].
        init_velocity: (n_dof,) start velocity. Defaults to zeros.
        init_acceleration: (n_dof,) start acceleration. Defaults to zeros.
        max_velocity: Per-DOF velocity limit. Scalar is broadcast.
        max_acceleration: Per-DOF acceleration limit. Scalar is broadcast.
        max_jerk: Per-DOF jerk limit. Scalar is broadcast.
        num_stitch: Fix the first ``num_stitch`` positions to targets.
            Constraints inside the stitched region are relaxed,
            only the boundary constraint is kept.
        tracking_weight: Weight on tracking cost (q[t] - target[t])^2.
        reg_weight: Diagonal regularisation to keep Q positive definite.

    Returns:
        (positions, velocities, accelerations) each of shape
        (T, n_dof) where T = len(targets).
    """
    if osqp is None:
        raise ImportError('osqp is required. Install with: pip install osqp')

    targets = np.asarray(targets, dtype=float)
    n_dof = targets.shape[1]

    if init_position is None:
        init_position = targets[0]
    init_position = np.asarray(init_position, dtype=float)
    init_velocity = (
        np.asarray(init_velocity, dtype=float)
        if init_velocity is not None else np.zeros(n_dof))
    init_acceleration = (
        np.asarray(init_acceleration, dtype=float)
        if init_acceleration is not None else np.zeros(n_dof))

    vel_limits = broadcast(max_velocity, n_dof)
    acc_limits = broadcast(max_acceleration, n_dof)
    jerk_limits = broadcast(max_jerk, n_dof)

    q_all = []
    v_all = []
    a_all = []

    for d in range(n_dof):
        q_j, v_j, a_j = _solve_one_joint(
            targets=targets[:, d],
            dt=dt,
            init_q=init_position[d],
            init_v=init_velocity[d],
            init_a=init_acceleration[d],
            vel_limit=vel_limits[d],
            acc_limit=acc_limits[d],
            jerk_limit=jerk_limits[d],
            num_stitch=num_stitch,
            tracking_weight=tracking_weight,
            reg_weight=reg_weight,
        )
        q_all.append(q_j)
        v_all.append(v_j)
        a_all.append(a_j)

    positions = np.stack(q_all, axis=1)
    velocities = np.stack(v_all, axis=1)
    accelerations = np.stack(a_all, axis=1)
    return positions, velocities, accelerations


# ---------------------------------------------------------------------------
#  Per-joint QP
# ---------------------------------------------------------------------------


def _solve_one_joint(
    targets: np.ndarray,  # [T]
    dt: float,
    init_q: float,
    init_v: float,
    init_a: float,
    vel_limit: float,
    acc_limit: float,
    jerk_limit: float,
    num_stitch: int,
    tracking_weight: float,
    reg_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(targets)

    # scaled limits
    v_lim = vel_limit * dt
    a_lim = acc_limit * dt * dt
    j_lim = jerk_limit * dt * dt

    n_q = T
    n_v = T
    n_a = T
    n_j = T - 1
    n_vars = n_q + n_v + n_a + n_j  # 4T - 1

    off_q = 0
    off_v = off_q + n_q
    off_a = off_v + n_v
    off_j = off_a + n_a

    # ---- cost ---------------------------------------------------------------
    Q = np.eye(n_vars) * reg_weight
    # tracking: q follows target
    for t in range(num_stitch, T):
        Q[off_q + t, off_q + t] += tracking_weight

    x_ref = np.zeros(n_vars)
    x_ref[off_q:off_q + T] = targets
    c = -Q @ x_ref

    # ---- dynamics (equality constraints, scaled variables) ------------------
    # q[t+1] - q[t] - v_s[t] = 0           (T-1 constraints)
    # v_s[t+1] - v_s[t] - a_s[t] = 0       (T-1 constraints)
    # a_s[t+1] - a_s[t] - j_s[t]*dt = 0    (T-1 constraints)
    n_dyn = 3 * (T - 1)
    A_dyn = sparse.lil_matrix((n_dyn, n_vars))
    b_dyn = np.zeros(n_dyn)

    row = 0
    for t in range(T - 1):
        # q dynamics
        A_dyn[row, off_q + t + 1] = 1.0
        A_dyn[row, off_q + t] = -1.0
        A_dyn[row, off_v + t] = -1.0
        row += 1
    for t in range(T - 1):
        # v dynamics
        A_dyn[row, off_v + t + 1] = 1.0
        A_dyn[row, off_v + t] = -1.0
        A_dyn[row, off_a + t] = -1.0
        row += 1
    for t in range(T - 1):
        # a -> j dynamics (j[t] is scaled by dt^2)
        A_dyn[row, off_a + t + 1] = 1.0
        A_dyn[row, off_a + t] = -1.0
        A_dyn[row, off_j + t] = -dt
        row += 1

    A_dyn = A_dyn.tocsc()

    if num_stitch > 0 and not np.isclose(init_q, targets[0]):
        raise ValueError(
            f'Stitch requires init_q == targets[0], got {init_q} vs '
            f'{targets[0]}')

    # ---- fixed position constraints ----------------------------------------
    # q[0] = init_q   (always)
    # if stitch: q[0..num_stitch-1] = targets[:num_stitch]
    # NOTE: when num_stitch > 0, q[0] is constrained to both init_q and
    # targets[0]; caller must ensure init_q == targets[0] for feasibility.
    n_fixed = 1 + num_stitch
    A_fix = sparse.lil_matrix((n_fixed, n_vars))
    b_fix = np.zeros(n_fixed)
    A_fix[0, off_q + 0] = 1.0
    b_fix[0] = init_q
    for t in range(num_stitch):
        A_fix[1 + t, off_q + t] = 1.0
        b_fix[1 + t] = targets[t]
    A_fix = A_fix.tocsc()

    # ---- box constraints ---------------------------------------------------
    lower = np.full(n_vars, -np.inf)
    upper = np.full(n_vars, np.inf)

    # v/a limits relaxed for indices strictly inside stitch region;
    # the boundary step (num_stitch-1) keeps constraints for continuity.
    # j has one fewer timestep (T-1 vs T), so its boundary is num_stitch-2.
    for t in range(n_v):
        if num_stitch > 0 and t < num_stitch - 1:
            continue
        lower[off_v + t] = -v_lim
        upper[off_v + t] = v_lim

    for t in range(n_a):
        if num_stitch > 0 and t < num_stitch - 1:
            continue
        lower[off_a + t] = -a_lim
        upper[off_a + t] = a_lim

    for t in range(n_j):
        if num_stitch > 0 and t < num_stitch - 2:
            continue
        lower[off_j + t] = -j_lim
        upper[off_j + t] = j_lim

    # ---- combine constraints ------------------------------------------------
    A_all = sparse.vstack([A_dyn, A_fix, sparse.eye(n_vars)])
    l_all = np.concatenate([b_dyn, b_fix, lower])
    u_all = np.concatenate([b_dyn, b_fix, upper])

    # ---- initial guess -----------------------------------------------------
    x0 = np.zeros(n_vars)
    x0[off_q:off_q + T] = np.linspace(init_q, targets[-1], T)

    # ---- solve -------------------------------------------------------------
    prob = osqp.OSQP()
    prob.setup(
        P=sparse.csc_matrix(Q),
        q=c,
        A=A_all.tocsc(),
        l=l_all,
        u=u_all,
        verbose=False,
        max_iter=1000,
        warm_start=False,
        polish=True,
    )

    res = prob.solve()
    if res.info.status not in ('solved', 'solved_inaccurate'):
        raise RuntimeError(f'OSQP failed for joint: {res.info.status}')

    x_opt = res.x
    q_opt = x_opt[off_q:off_q + T]
    v_scaled = x_opt[off_v:off_v + T]
    a_scaled = x_opt[off_a:off_a + T]

    v_opt = v_scaled / dt
    a_opt = a_scaled / (dt * dt)

    return q_opt, v_opt, a_opt
