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
position / velocity / acceleration / jerk variables (uniformly scaled
by dt), triple-integrator dynamics, and hard constraints on all
derivatives.

Usage::

    positions, velocities, accelerations = joint_mpc(
        targets=actions[:, dof_indices],
        dt=0.02,
        init_position=qpos,
        max_velocity=5.0, max_acceleration=20.0, max_jerk=50.0,
    )
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Union

import numpy as np
from scipy import sparse

from .trajectory_utils import broadcast

try:
    import osqp
except ImportError:
    osqp = None

# ---------------------------------------------------------------------------
#  Solver cache
# ---------------------------------------------------------------------------

# key → {
#     'P': csc,
#     'A_template': csc,
#     'b_dyn': ndarray,
#     'Q_diag': ndarray,
#     'off_*': int,
#     'n_vars': int,
#     'T': int,
#     'dt': float,
#     'solvers': list[osqp.OSQP | None],
# }
_solver_cache: dict = {}


def _cache_key(
    T: int,
    dt: float,
    tracking_weight: float,
    reg_weight: float,
    settle: bool,
    terminal_weight: float,
) -> tuple:
    return (T, dt, tracking_weight, reg_weight, settle, terminal_weight)


def _build_templates(
    T: int,
    dt: float,
    tracking_weight: float,
    reg_weight: float,
    settle: bool = False,
    terminal_weight: float = 10000.0,
) -> dict:
    """Build all DOF-independent QP template matrices.

    Scaling: v is natural [rad/s];  a_s = a·dt, j_s = j·dt.
    Dynamics:

        q[t+1] - q[t] - v[t] · dt       = 0
        v[t+1] - v[t] - a_s[t]          = 0
        a_s[t+1] - a_s[t] - j_s[t] · dt = 0

    Terminal constraints are SOFT — high cost weight, guaranteeing the QP
    is always feasible regardless of horizon or limits.
    """
    n_q, n_v, n_a, n_j = T, T, T, T - 1
    n_vars = n_q + n_v + n_a + n_j
    off_q, off_v, off_a, off_j = 0, n_q, n_q + n_v, n_q + n_v + n_a

    # Q diagonal: tracking on positions, soft terminal anchors
    Q_diag = np.full(n_vars, reg_weight)
    Q_diag[off_q:off_q + T] += tracking_weight
    Q_diag[off_q + T - 1] += terminal_weight  # attract q[-1] → target[-1]
    if settle:
        Q_diag[off_v + T - 1] += terminal_weight  # nudge v[-1] → 0
        Q_diag[off_a + T - 1] += terminal_weight  # nudge a[-1] → 0

    # Dynamics — all three tiers use dt on the input variable
    n_dyn = 3 * (T - 1)
    A_dyn = sparse.lil_matrix((n_dyn, n_vars))
    row = 0
    for t in range(T - 1):
        A_dyn[row, off_q + t + 1] = 1.0
        A_dyn[row, off_q + t] = -1.0
        A_dyn[row, off_v + t] = -dt  # v[·] natural, dt coeff
        row += 1
    for t in range(T - 1):
        A_dyn[row, off_v + t + 1] = 1.0
        A_dyn[row, off_v + t] = -1.0
        A_dyn[row, off_a + t] = -1.0  # a_s[·] = a·dt, unity coeff
        row += 1
    for t in range(T - 1):
        A_dyn[row, off_a + t + 1] = 1.0
        A_dyn[row, off_a + t] = -1.0
        A_dyn[row, off_j + t] = -dt  # j_s[·] = j·dt, dt coeff
        row += 1
    A_dyn = A_dyn.tocsc()
    b_dyn = np.zeros(n_dyn)

    I_sparse = sparse.eye(n_vars).tocsc()
    A_template = sparse.vstack([A_dyn, I_sparse]).tocsc()

    P = sparse.diags(Q_diag).tocsc()

    return {
        'P': P,
        'A_template': A_template,
        'b_dyn': b_dyn,
        'Q_diag': Q_diag,
        'off_q': off_q,
        'off_v': off_v,
        'off_a': off_a,
        'off_j': off_j,
        'n_vars': n_vars,
        'T': T,
        'dt': dt,
        'settle': settle,
        'solvers': [],
    }


def _solve_one_dof(
    cache: dict,
    d: int,
    target: np.ndarray,
    init_position: float,
    init_velocity: float,
    init_acceleration: float,
    v_lim: float,
    a_lim: float,
    j_lim: float,
    num_stitch: int = 0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Solve a single DOF — safe to call from a thread."""
    P = cache['P']
    A_template = cache['A_template']
    b_dyn = cache['b_dyn']
    Q_diag = cache['Q_diag']
    off_q = cache['off_q']
    off_v = cache['off_v']
    off_a = cache['off_a']
    off_j = cache['off_j']
    n_vars = cache['n_vars']
    _T = cache['T']
    _dt = cache['dt']

    # ── Build per-DOF l / u ────────────────────────────────────────────────
    a_lim_s = a_lim * _dt
    j_lim_s = j_lim * _dt

    lower = np.full(n_vars, -np.inf)
    upper = np.full(n_vars, np.inf)

    # v / a / j box constraints (set first, then overridden by fixed values)
    # v is natural [rad/s]; a_s = a·dt; j_s = j·dt
    for t in range(_T):
        if num_stitch > 0 and t < num_stitch - 1:
            continue
        lower[off_v + t] = -v_lim
        upper[off_v + t] = v_lim
    for t in range(_T):
        if num_stitch > 0 and t < num_stitch - 2:
            continue
        lower[off_a + t] = -a_lim_s
        upper[off_a + t] = a_lim_s
    for t in range(_T - 1):
        if num_stitch > 0 and t < num_stitch - 3:
            continue
        lower[off_j + t] = -j_lim_s
        upper[off_j + t] = j_lim_s

    # Initial state / stitch (mutually exclusive) — hard constraints
    if num_stitch == 0:
        lower[off_q + 0] = upper[off_q + 0] = init_position
        lower[off_v + 0] = upper[off_v + 0] = init_velocity
        lower[off_a + 0] = upper[off_a + 0] = init_acceleration * _dt
    else:
        for t in range(num_stitch):
            lower[off_q + t] = upper[off_q + t] = target[t]

    # Terminal constraints are SOFT — high cost weight, not hard bounds.
    # This guarantees the QP is always feasible regardless of horizon/limits.

    l_all = np.concatenate([b_dyn, lower])
    u_all = np.concatenate([b_dyn, upper])

    x_ref = np.zeros(n_vars)
    x_ref[off_q:off_q + _T] = target
    c_vec = -Q_diag * x_ref

    # ── Create or update solver ────────────────────────────────────────────
    prob = cache['solvers'][d]
    if prob is None:
        prob = osqp.OSQP()
        prob.setup(
            P=P,
            q=c_vec,
            A=A_template,
            l=l_all,
            u=u_all,
            verbose=False,
            max_iter=2000,
            polishing=False,
            eps_abs=1e-2)
        cache['solvers'][d] = prob
    else:
        prob.update(q=c_vec, l=l_all, u=u_all)

    # Always reset primal+dual: primal from target (bounds-feasible),
    # dual reset to zero — avoids stale duals from old q/l/u.
    m = A_template.shape[0]
    x0 = np.zeros(n_vars)
    x0[off_q:off_q + _T] = target
    if num_stitch == 0:
        x0[off_q + 0] = init_position
        x0[off_v + 0] = init_velocity
        x0[off_a + 0] = init_acceleration * _dt
    prob.warm_start(x=x0, y=np.zeros(m))
    res = prob.solve()

    if res.info.status not in ('solved', 'solved inaccurate'):
        print(f'[OSQP] suboptimal solution: {res.info.status} '
              f'(iter={res.info.iter}), using best-effort result')

    x_opt = res.x
    q_opt = x_opt[off_q:off_q + _T]
    v_opt = x_opt[off_v:off_v + _T]  # natural [rad/s]
    a_opt = x_opt[off_a:off_a + _T] / _dt

    return d, q_opt, v_opt, a_opt


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
    tracking_weight: float = 3.0,
    reg_weight: float = 5e-3,
    settle: bool = False,
    terminal_weight: float = 10000.0,
    parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-joint MPC smoothing via OSQP.

    Each joint is solved as an independent QP.  Decision variables are
    uniformly scaled by dt (v_s = v·dt, a_s = a·dt, j_s = j·dt) so that
    all three have comparable magnitude and a single ``reg_weight`` tunes
    regularisation across all derivatives.

    Dynamics:

        q[t+1] - q[t] - v_s[t]          = 0
        v_s[t+1] - v_s[t] - a_s[t] · dt = 0
        a_s[t+1] - a_s[t] - j_s[t] · dt = 0

    Terminal constraints are SOFT (high cost weight) to guarantee the QP
    is always feasible regardless of horizon or limits.

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
        settle: If True, add high cost on v[-1] and a[-1] to bring the
            trajectory to a stop at the final waypoint (default False).
        terminal_weight: Cost weight on terminal state (q[-1], and if
            settle also v[-1] / a[-1]).  Hard box constraints on
            v/a/j guarantee safety; this soft weight only nudges within
            the safe region (default 10000.0).
        parallel: Use thread parallelism for per-DOF solves (default True).

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

    T = len(targets)
    ck = _cache_key(T, dt, tracking_weight, reg_weight, settle,
                    terminal_weight)

    # ── Retrieve or build template (main thread) ───────────────────────────
    if ck not in _solver_cache:
        tmpl = _build_templates(T, dt, tracking_weight, reg_weight, settle,
                                terminal_weight)
        tmpl['solvers'] = [None] * n_dof
        _solver_cache[ck] = tmpl
    cache = _solver_cache[ck]
    if len(cache['solvers']) < n_dof:
        cache['solvers'].extend([None] * (n_dof - len(cache['solvers'])))

    # ── Solve all DOFs ─────────────────────────────────────────────────────
    positions = np.zeros((cache['T'], n_dof))
    velocities = np.zeros((cache['T'], n_dof))
    accelerations = np.zeros((cache['T'], n_dof))

    if parallel and n_dof > 1:
        with ThreadPoolExecutor(max_workers=min(n_dof, 12)) as ex:
            futures = {}
            for d in range(n_dof):
                f = ex.submit(
                    _solve_one_dof,
                    cache,
                    d,
                    targets[:, d],
                    float(init_position[d]),
                    float(init_velocity[d]),
                    float(init_acceleration[d]),
                    float(vel_limits[d]),
                    float(acc_limits[d]),
                    float(jerk_limits[d]),
                    num_stitch,
                )
                futures[f] = d

            for f in as_completed(futures):
                d, q_opt, v_opt, a_opt = f.result()
                positions[:, d] = q_opt
                velocities[:, d] = v_opt
                accelerations[:, d] = a_opt
    else:
        for d in range(n_dof):
            _, q_opt, v_opt, a_opt = _solve_one_dof(
                cache,
                d,
                targets[:, d],
                float(init_position[d]),
                float(init_velocity[d]),
                float(init_acceleration[d]),
                float(vel_limits[d]),
                float(acc_limits[d]),
                float(jerk_limits[d]),
                num_stitch,
            )
            positions[:, d] = q_opt
            velocities[:, d] = v_opt
            accelerations[:, d] = a_opt

    return positions, velocities, accelerations
