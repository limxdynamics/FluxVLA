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
"""Trajectory plotting utilities.

All functions are designed to run in a subprocess so they never block
the control loop.  Import matplotlib inside each function to keep it
out of the main process.
"""

import multiprocessing
import time
from typing import List, Optional

import numpy as np

from .trajectory_utils import Trajectory

_LAST_PLOT_TIME = 0.0


def plot_postprocess_comparison(
    dof_indices: List[int],
    cur_raw: Trajectory,
    cur_post: Trajectory,
    prev_raw: Optional[Trajectory] = None,
    prev_post: Optional[Trajectory] = None,
    plot_dofs: Optional[List[int]] = None,
    output_dir: str = 'work_dirs/postprocess_debug',
    title: str = 'Postprocess Debug: raw vs post-processed',
):
    """Spawn an async plot process comparing raw vs post-processed
    trajectories across adjacent inference steps."""
    global _LAST_PLOT_TIME
    now = time.time()
    if now - _LAST_PLOT_TIME < 0.5:
        return
    _LAST_PLOT_TIME = now

    dof_indices = list(dof_indices)
    local_plot_dofs = (
        plot_dofs if plot_dofs is not None else list(
            range(min(6, len(dof_indices)))))
    dof_labels = [f'DOF {i}' for i in dof_indices]

    kwargs = dict(
        dt=cur_raw.dt,
        dof_labels=dof_labels,
        plot_dofs=local_plot_dofs,
        cur_raw=cur_raw.positions[:, dof_indices].copy(),
        cur_post=cur_post.positions[:, dof_indices].copy(),
        cur_t0=cur_raw.t0,
        title=title,
        output_dir=output_dir,
    )

    if prev_raw is not None and prev_post is not None:
        kwargs['prev_raw'] = prev_raw.positions[:, dof_indices].copy()
        kwargs['prev_post'] = prev_post.positions[:, dof_indices].copy()
        kwargs['prev_t0'] = prev_raw.t0

    proc = multiprocessing.Process(
        target=_render_postprocess_comparison, kwargs=kwargs, daemon=True)
    proc.start()


def _render_postprocess_comparison(
    dt: float,
    dof_labels: List[str],
    plot_dofs: List[int],
    cur_raw: np.ndarray,
    cur_post: np.ndarray,
    cur_t0: float,
    prev_raw: Optional[np.ndarray] = None,
    prev_post: Optional[np.ndarray] = None,
    prev_t0: Optional[float] = None,
    title: str = 'Trajectory: raw vs post-processed',
    output_dir: str = 'work_dirs/traj_debug',
    filename: Optional[str] = None,
):
    """Render current/previous pre-vs-post actions into one plot figure."""
    import os

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_plots = len(plot_dofs)
    t0_ref = prev_t0 if prev_t0 is not None else cur_t0

    fig, axes = plt.subplots(
        n_plots, 1, figsize=(12, 3 * n_plots), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax_i, dof_i in enumerate(plot_dofs):
        ax = axes[ax_i]

        has_prev = (
            prev_raw is not None and prev_post is not None
            and prev_t0 is not None)
        if has_prev:
            n_prev = len(prev_raw)
            t_prev = (prev_t0 - t0_ref) + np.arange(n_prev) * dt
            ax.plot(
                t_prev, prev_raw[:, dof_i], 'b--', alpha=0.5, label='prev raw')
            ax.plot(
                t_prev,
                prev_post[:, dof_i],
                'b-',
                alpha=0.8,
                label='prev post')

        n_cur = len(cur_raw)
        t_cur = (cur_t0 - t0_ref) + np.arange(n_cur) * dt
        ax.plot(t_cur, cur_raw[:, dof_i], 'r--', alpha=0.5, label='cur raw')
        ax.plot(t_cur, cur_post[:, dof_i], 'r-', alpha=0.8, label='cur post')

        ax.axvline(
            x=t_cur[0],
            color='gray',
            linestyle=':',
            alpha=0.5,
            label='cur start')
        ax.set_ylabel(
            dof_labels[dof_i] if dof_i < len(dof_labels) else f'DOF {dof_i}')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = filename or f'chunk_{cur_t0:.3f}.png'
    fig.savefig(os.path.join(output_dir, fname), dpi=100)
    plt.close(fig)
