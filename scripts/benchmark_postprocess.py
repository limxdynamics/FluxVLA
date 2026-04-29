"""Benchmark for joint_mpc and ruckig_filter backends."""

import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fluxvla.engines.utils.postprocess import joint_mpc  # noqa: E402
from fluxvla.engines.utils.postprocess import ruckig_filter  # noqa: E402

matplotlib.use('Agg')

DT = 0.02
DOF = 12
N = 50
PLOT_DOFS = [0, 1, 2]
VEL_LIM = 5.0
ACC_LIM = 20.0
JERK_LIM = 20.0
SETTLE_ACC_LIM = 40.0
SETTLE_JERK_LIM = 80.0
NUM_WARMUP = 5
NUM_REPEAT = 10
OUTPUT_DIR = 'work_dirs/postprocess_benchmark'


def gen_traj(noise_std=0.15, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    scales = rng.uniform(0.5, 1.5, DOF)
    base = np.sin(t[:, None] * 2 * np.pi * 0.8 + np.arange(DOF) * 0.3) * 0.5
    base += np.sin(t[:, None] * 2 * np.pi * 2.0 + np.arange(DOF) * 0.7) * 0.2
    base *= scales[np.newaxis, :]
    noise = rng.normal(0, noise_std, (N, DOF)) * scales[np.newaxis, :]
    return (base + noise).astype(np.float64)


def estimate_jerk(acc, dt):
    return (acc[1:] - acc[:-1]) / dt


def _measure_latency(fn, targets_list, **kwargs):
    for i in range(NUM_WARMUP):
        fn(targets_list[i], DT, init_position=targets_list[i][0], **kwargs)
    t0 = time.perf_counter()
    for i in range(NUM_WARMUP, NUM_WARMUP + NUM_REPEAT):
        fn(targets_list[i], DT, init_position=targets_list[i][0], **kwargs)
    return (time.perf_counter() - t0) / NUM_REPEAT * 1000


def latency():
    print('─' * 50)
    print(f'  Latency (DOF={DOF}, N={N})')
    print('─' * 50)

    all_targets = [
        gen_traj(seed=42 + i) for i in range(NUM_WARMUP + NUM_REPEAT)
    ]
    common_kw = dict(
        max_velocity=VEL_LIM, max_acceleration=ACC_LIM, max_jerk=JERK_LIM)

    rows = {}
    rows['MPC tracking'] = _measure_latency(joint_mpc, all_targets,
                                            **common_kw)
    rows['Ruckig tracking'] = _measure_latency(ruckig_filter, all_targets,
                                               **common_kw)

    for name, ms in rows.items():
        print(f'  {name:>16s}  {ms:8.2f} ms')

    return rows


def _report_table(title, targets, acc_limit, jerk_limit, rows):
    print(f'\n  {title}')
    hdr = (f'  {"":>16s} │ {"RMS":>8s} │ {"max|v|":>8s} │ '
           f'{"max|a|":>8s} │ {"max|j|":>8s}')
    lim = (f'  {"limit":>16s} │ {"":>8s} │ {VEL_LIM:8.1f} │ '
           f'{acc_limit:8.1f} │ {jerk_limit:8.1f}')
    sep = '  ' + '─' * (len(hdr) - 2)
    print(hdr)
    print(lim)
    print(sep)
    for name, pos, vel, acc, jerk in rows:
        _pos, _vel, _acc = pos[:N], vel[:N], acc[:N]
        rms = np.sqrt(np.mean((_pos - targets)**2))
        vmax = np.max(np.abs(_vel))
        amax = np.max(np.abs(_acc))
        jmax = (
            np.max(np.abs(jerk[:N - 1])) if len(jerk) >= N - 1 else np.max(
                np.abs(jerk)))
        row = (f'  {name:>16s} │ {rms:8.4f} │ {vmax:8.2f} │ '
               f'{amax:8.1f} │ {jmax:8.1f}')
        print(row)


def quality():
    print()
    print('─' * 50)
    print('  Quality: tracking vs settle')
    print('─' * 50)

    targets = gen_traj(noise_std=0.3)
    init_q = targets[0].copy()
    init_v = np.zeros(DOF)
    common_kw = dict(
        init_position=init_q,
        init_velocity=init_v,
        init_acceleration=None,
        max_velocity=VEL_LIM,
        max_acceleration=ACC_LIM,
        max_jerk=JERK_LIM)

    # tracking mode: OSQP pure tracking
    tracking_pos, tracking_vel, tracking_acc = joint_mpc(
        targets, DT, **common_kw)

    # settle mode: OSQP tracking with terminal / settle weights
    settle_kw = dict(
        common_kw, max_acceleration=SETTLE_ACC_LIM, max_jerk=SETTLE_JERK_LIM)
    settle_pos, settle_vel, settle_acc = joint_mpc(
        targets, DT, terminal_weight=1.0, settle_weight=1.0, **settle_kw)

    # ruckig tracking (reference)
    r_pos, r_vel, r_acc = ruckig_filter(targets, DT, **common_kw)

    # ruckig settle: raw (keep extra steps) + resampled
    rsettle_kw = dict(
        common_kw, max_acceleration=SETTLE_ACC_LIM, max_jerk=SETTLE_JERK_LIM)
    rs_raw_pos, rs_raw_vel, rs_raw_acc = ruckig_filter(
        targets, DT, max_settle_steps=15, resample_settle=False, **rsettle_kw)
    rs_pos, rs_vel, rs_acc = ruckig_filter(
        targets, DT, max_settle_steps=15, **rsettle_kw)

    raw_j = np.zeros((N - 1, DOF))

    _report_table('Tracking', targets, ACC_LIM, JERK_LIM, [
        ('raw', targets, np.zeros_like(targets), np.zeros_like(targets),
         raw_j),
        ('MPC', tracking_pos, tracking_vel, tracking_acc,
         estimate_jerk(tracking_acc, DT)),
        ('Ruckig', r_pos, r_vel, r_acc, estimate_jerk(r_acc, DT)),
    ])

    _report_table('Settle', targets, SETTLE_ACC_LIM, SETTLE_JERK_LIM, [
        ('MPC settle', settle_pos, settle_vel, settle_acc,
         estimate_jerk(settle_acc, DT)),
        ('Ruckig settle', rs_pos, rs_vel, rs_acc, estimate_jerk(rs_acc, DT)),
    ])

    # Plot-only data: recompute on a 3-DOF subproblem so hidden DOFs do not
    # affect the visible trajectories.
    plot_targets = targets[:, PLOT_DOFS]
    plot_init_q = init_q[PLOT_DOFS]
    plot_init_v = np.zeros(len(PLOT_DOFS))
    plot_common_kw = dict(
        init_position=plot_init_q,
        init_velocity=plot_init_v,
        init_acceleration=None,
        max_velocity=VEL_LIM,
        max_acceleration=ACC_LIM,
        max_jerk=JERK_LIM)
    plot_settle_kw = dict(
        plot_common_kw,
        max_acceleration=SETTLE_ACC_LIM,
        max_jerk=SETTLE_JERK_LIM)

    plot_tracking = joint_mpc(plot_targets, DT, **plot_common_kw)
    plot_settle = joint_mpc(
        plot_targets,
        DT,
        terminal_weight=1.0,
        settle_weight=1.0,
        **plot_settle_kw)
    plot_ruckig = ruckig_filter(plot_targets, DT, **plot_common_kw)
    plot_ruckig_settle_raw = ruckig_filter(
        plot_targets,
        DT,
        max_settle_steps=15,
        resample_settle=False,
        **plot_settle_kw)
    plot_ruckig_settle = ruckig_filter(
        plot_targets, DT, max_settle_steps=15, **plot_settle_kw)

    return {
        'targets': targets,
        'init_q': init_q,
        'tracking': (tracking_pos, tracking_vel, tracking_acc),
        'settle': (settle_pos, settle_vel, settle_acc),
        'ruckig': (r_pos, r_vel, r_acc),
        'ruckig_settle': (rs_pos, rs_vel, rs_acc),
        'ruckig_settle_raw': (rs_raw_pos, rs_raw_vel, rs_raw_acc),
        'plot_targets': plot_targets,
        'plot_init_q': plot_init_q,
        'plot_tracking': plot_tracking,
        'plot_settle': plot_settle,
        'plot_ruckig': plot_ruckig,
        'plot_ruckig_settle': plot_ruckig_settle,
        'plot_ruckig_settle_raw': plot_ruckig_settle_raw,
    }


def _plot_mode(title,
               fname,
               targets,
               init_q,
               curves,
               acc_lim=ACC_LIM,
               jerk_lim=JERK_LIM):
    """Plot one mode with MPC + Ruckig overlaid."""
    print(f'  plotting {title} ...')

    plot_dofs = list(range(targets.shape[1]))
    dof_names = [f'joint_{d}' for d in PLOT_DOFS[:len(plot_dofs)]]
    t_ref = np.arange(N) * DT

    fig, axes = plt.subplots(
        len(plot_dofs), 4, figsize=(20, max(10, 2.4 * len(plot_dofs))))

    if len(plot_dofs) == 1:
        axes = axes[np.newaxis, :]

    for row, d in enumerate(plot_dofs):
        ax_p, ax_v, ax_a, ax_j = axes[row]

        ax_p.plot(t_ref, targets[:, d], 'k:', alpha=0.35, lw=1.0, label='raw')

        for label, pos, vel, acc, color, ls in curves:
            t = np.arange(len(pos)) * DT
            t_j = np.arange(len(pos) - 1) * DT
            jerk = estimate_jerk(acc, DT) if len(acc) > 1 else np.zeros(
                (len(acc) - 1, acc.shape[1]))

            kw = dict(color=color, ls=ls, lw=1.3)
            ax_p.plot(t, pos[:, d], label=label, **kw)
            ax_v.plot(t, vel[:, d], **kw)
            ax_a.plot(t, acc[:, d], **kw)
            ax_j.plot(t_j, jerk[:, d], **kw)

        ax_p.axhline(init_q[d], color='gray', ls=':', alpha=0.4)
        ax_p.set_ylabel(f'{dof_names[d]}\nposition')
        ax_p.grid(True, alpha=0.25)
        if row == 0:
            ax_p.set_title('Position')
            ax_p.legend(fontsize=7, loc='upper right')

        ax_v.axhline(VEL_LIM, color='gray', ls='--', alpha=0.3, lw=0.8)
        ax_v.axhline(-VEL_LIM, color='gray', ls='--', alpha=0.3, lw=0.8)
        ax_v.set_ylabel('velocity')
        ax_v.grid(True, alpha=0.25)
        if row == 0:
            ax_v.set_title('Velocity')

        ax_a.axhline(acc_lim, color='gray', ls='--', alpha=0.3, lw=0.8)
        ax_a.axhline(-acc_lim, color='gray', ls='--', alpha=0.3, lw=0.8)
        ax_a.set_ylabel('acceleration')
        ax_a.grid(True, alpha=0.25)
        if row == 0:
            ax_a.set_title('Acceleration')

        ax_j.axhline(jerk_lim, color='gray', ls='--', alpha=0.3, lw=0.8)
        ax_j.axhline(-jerk_lim, color='gray', ls='--', alpha=0.3, lw=0.8)
        ax_j.set_ylabel('jerk')
        ax_j.grid(True, alpha=0.25)
        if row == 0:
            ax_j.set_title('Jerk (finite diff)')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    axes[-1, 2].set_xlabel('Time (s)')
    axes[-1, 3].set_xlabel('Time (s)')

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


def plot(data):
    targets = data['plot_targets']
    init_q = data['plot_init_q']
    tracking_pos, tracking_vel, tracking_acc = data['plot_tracking']
    settle_pos, settle_vel, settle_acc = data['plot_settle']
    r_pos, r_vel, r_acc = data['plot_ruckig']
    rs_pos, rs_vel, rs_acc = data['plot_ruckig_settle']
    rs_raw_pos, rs_raw_vel, rs_raw_acc = data['plot_ruckig_settle_raw']

    _plot_mode(
        'Tracking mode — MPC vs Ruckig', 'tracking_mode.png', targets, init_q,
        [
            ('MPC', tracking_pos, tracking_vel, tracking_acc, 'tab:blue', '-'),
            ('Ruckig', r_pos, r_vel, r_acc, 'tab:green', '-'),
        ])

    _plot_mode(
        'Settle mode — MPC vs Ruckig (native settle + resample)',
        'settle_mode.png',
        targets,
        init_q, [
            ('MPC settle', settle_pos, settle_vel, settle_acc, 'tab:blue',
             '-'),
            ('Ruckig settle (raw)', rs_raw_pos, rs_raw_vel, rs_raw_acc,
             'tab:green', '--'),
            ('Ruckig settle (resample)', rs_pos, rs_vel, rs_acc, 'tab:green',
             '-'),
        ],
        acc_lim=SETTLE_ACC_LIM,
        jerk_lim=SETTLE_JERK_LIM)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    latency()
    data = quality()
    plot(data)
    print(f'\nDone. Plots in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
