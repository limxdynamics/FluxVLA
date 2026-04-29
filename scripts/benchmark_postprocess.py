"""Benchmark joint_mpc (OSQP) vs ruckig_filter — tracking & settle modes."""

import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from fluxvla.engines.utils.postprocess import joint_mpc  # noqa: E402
from fluxvla.engines.utils.postprocess import ruckig_filter  # noqa: E402

matplotlib.use('Agg')

DT = 0.02
DOF = 12
N = 50
VEL_LIM = 3.0
ACC_LIM = 10.0
JERK_LIM = 50.0
SETTLE_STEPS = 50
NUM_WARMUP = 5
NUM_REPEAT = 10
OUTPUT_DIR = 'work_dirs/postprocess_benchmark'


def gen_traj(noise_std=0.3, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    scales = rng.uniform(0.0, 2.0, DOF)
    base = np.sin(t[:, None] * 2 * np.pi * 0.8 + np.arange(DOF) * 0.3) * 0.5
    base += np.sin(t[:, None] * 2 * np.pi * 2.0 + np.arange(DOF) * 0.7) * 0.2
    base *= scales[np.newaxis, :]
    noise = rng.normal(0, noise_std, (N, DOF)) * scales[np.newaxis, :]
    return (base + noise).astype(np.float64)


def estimate_jerk(acc, dt):
    return (acc[1:] - acc[:-1]) / dt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Latency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _measure_latency(fn, targets_list, **kwargs):
    for i in range(NUM_WARMUP):
        fn(targets_list[i], DT, init_position=targets_list[i][0], **kwargs)
    t0 = time.perf_counter()
    for i in range(NUM_WARMUP, NUM_WARMUP + NUM_REPEAT):
        fn(targets_list[i], DT, init_position=targets_list[i][0], **kwargs)
    return (time.perf_counter() - t0) / NUM_REPEAT * 1000


def latency():
    print('=' * 70)
    print('  LATENCY: joint_mpc vs ruckig  (DOF=12, N=50)')
    print('=' * 70)

    all_targets = [
        gen_traj(seed=42 + i) for i in range(NUM_WARMUP + NUM_REPEAT)
    ]
    base_kw = dict(
        max_velocity=VEL_LIM, max_acceleration=ACC_LIM, max_jerk=JERK_LIM)

    mpc_track = _measure_latency(
        joint_mpc, all_targets, settle=False, **base_kw)
    mpc_settle = _measure_latency(
        joint_mpc, all_targets, settle=True, **base_kw)
    mpc_track_s = _measure_latency(
        joint_mpc, all_targets, settle=False, parallel=False, **base_kw)
    mpc_settle_s = _measure_latency(
        joint_mpc, all_targets, settle=True, parallel=False, **base_kw)
    r_track = _measure_latency(ruckig_filter, all_targets, **base_kw)
    r_settle = _measure_latency(
        ruckig_filter, all_targets, max_settle_steps=SETTLE_STEPS, **base_kw)

    print(f'  {"":>14s} │ {"Tracking ms":>11s} │ {"Settle ms":>11s}')
    print(f'  {"":>14s} │ {"":>11s} │ {"":>11s}')
    print(f'  {"MPC":>14s} │ {mpc_track:9.1f} ms │ {mpc_settle:9.1f} ms')
    print(f'  {"MPC (serial)":>14s} │ {mpc_track_s:9.1f} ms │ '
          f'{mpc_settle_s:9.1f} ms')
    print(f'  {"Ruckig":>14s} │ {r_track:9.2f} ms │ {r_settle:9.2f} ms')
    return (
        mpc_track,
        mpc_settle,
        mpc_track_s,
        mpc_settle_s,
        r_track,
        r_settle,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Quality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _report_table(title, targets, rows):
    print(f'\n  {title}')
    hdr = (f'  {"":>8s} │ {"RMS":>8s} │ {"max|v|":>8s} │ '
           f'{"max|a|":>8s} │ {"max|j|":>8s}')
    lim = (f'  {"limit":>8s} │ {"":>8s} │ {VEL_LIM:8.1f} │ '
           f'{ACC_LIM:8.1f} │ {JERK_LIM:8.1f}')
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
        row = (f'  {name:>8s} │ {rms:8.4f} │ {vmax:8.2f} │ '
               f'{amax:8.1f} │ {jmax:8.1f}')
        print(row)


def quality():
    print()
    print('=' * 70)
    print('  QUALITY: tracking RMS, max |v|/|a|/|jerk|')
    print('=' * 70)

    targets = gen_traj(noise_std=0.3)
    init_q = targets[0].copy()
    init_v = np.zeros(DOF)
    init_a = np.zeros(DOF)
    base_kw = dict(
        init_position=init_q,
        init_velocity=init_v,
        init_acceleration=init_a,
        max_velocity=VEL_LIM,
        max_acceleration=ACC_LIM,
        max_jerk=JERK_LIM)

    # ── Tracking ───────────────────────────────────────────────────────────
    mpc_t_pos, mpc_t_vel, mpc_t_acc = joint_mpc(
        targets, DT, settle=False, **base_kw)
    r_t_pos, r_t_vel, r_t_acc = ruckig_filter(targets, DT, **base_kw)

    # ── Settle ─────────────────────────────────────────────────────────────
    mpc_s_pos, mpc_s_vel, mpc_s_acc = joint_mpc(
        targets, DT, settle=True, **base_kw)
    r_sr_pos, r_sr_vel, r_sr_acc = ruckig_filter(
        targets,
        DT,
        max_settle_steps=SETTLE_STEPS,
        resample_settle=False,
        **base_kw)
    r_srsp_pos, r_srsp_vel, r_srsp_acc = ruckig_filter(
        targets,
        DT,
        max_settle_steps=SETTLE_STEPS,
        resample_settle=True,
        **base_kw)

    raw_j = np.zeros((N - 1, DOF))
    _report_table('Tracking', targets, [
        ('raw', targets, np.zeros_like(targets), np.zeros_like(targets),
         raw_j),
        ('MPC', mpc_t_pos, mpc_t_vel, mpc_t_acc, estimate_jerk(mpc_t_acc, DT)),
        ('Ruckig', r_t_pos, r_t_vel, r_t_acc, estimate_jerk(r_t_acc, DT)),
    ])
    _report_table('Settle', targets, [
        ('raw', targets, np.zeros_like(targets), np.zeros_like(targets),
         raw_j),
        ('MPC', mpc_s_pos, mpc_s_vel, mpc_s_acc, estimate_jerk(mpc_s_acc, DT)),
        ('R raw', r_sr_pos, r_sr_vel, r_sr_acc, estimate_jerk(r_sr_acc, DT)),
        ('R rsmpl', r_srsp_pos, r_srsp_vel, r_srsp_acc,
         estimate_jerk(r_srsp_acc, DT)),
    ])

    return (targets, init_q, mpc_t_pos, mpc_t_vel, mpc_t_acc, mpc_s_pos,
            mpc_s_vel, mpc_s_acc, r_t_pos, r_t_vel, r_t_acc, r_sr_pos,
            r_sr_vel, r_sr_acc, r_srsp_pos, r_srsp_vel, r_srsp_acc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _plot_one(title,
              fname,
              targets,
              init_q,
              pos_m,
              vel_m,
              acc_m,
              pos_r1,
              vel_r1,
              acc_r1,
              label_r1,
              pos_r2=None,
              vel_r2=None,
              acc_r2=None,
              label_r2=None):
    print(f'  plotting {title} ...')

    jerk_m = estimate_jerk(acc_m, DT)
    jerk_r1 = estimate_jerk(acc_r1, DT)
    jerk_r2 = estimate_jerk(acc_r2, DT) if acc_r2 is not None else None

    plot_dofs = [0, 1, 2]
    dof_names = ['left_arm_j1', 'left_arm_j2', 'left_arm_j3']
    t = np.arange(N) * DT
    t1 = np.arange(len(pos_r1)) * DT
    t_j = np.arange(N - 1) * DT
    t1_j = np.arange(len(pos_r1) - 1) * DT
    if pos_r2 is not None:
        t2 = np.arange(len(pos_r2)) * DT
        t2_j = np.arange(len(pos_r2) - 1) * DT

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))

    for row, d in enumerate(plot_dofs):
        ax_p, ax_v, ax_a, ax_j = axes[row]

        ax_p.plot(t, targets[:, d], 'k:', alpha=0.40, lw=1.2, label='raw')
        ax_p.plot(t, pos_m[:, d], 'r-', lw=1.3, label='MPC')
        ax_p.plot(
            t1, pos_r1[:, d], color='darkcyan', ls='-', lw=1.2, label=label_r1)
        if pos_r2 is not None:
            ax_p.plot(
                t2,
                pos_r2[:, d],
                color='teal',
                ls='--',
                lw=1.2,
                label=label_r2)
        ax_p.axhline(init_q[d], color='gray', ls=':', alpha=0.5)
        ax_p.set_ylabel(f'{dof_names[d]}\nposition')
        ax_p.grid(True, alpha=0.25)
        if row == 0:
            ax_p.set_title('Position')
            ax_p.legend(fontsize=7, loc='upper right')

        ax_v.plot(t, vel_m[:, d], 'r-', lw=1.3)
        ax_v.plot(t1, vel_r1[:, d], color='darkcyan', ls='-', lw=1.2)
        if vel_r2 is not None:
            ax_v.plot(t2, vel_r2[:, d], color='teal', ls='--', lw=1.2)
        ax_v.axhline(VEL_LIM, color='gray', ls='--', alpha=0.4, lw=0.8)
        ax_v.axhline(-VEL_LIM, color='gray', ls='--', alpha=0.4, lw=0.8)
        ax_v.set_ylabel('velocity')
        ax_v.grid(True, alpha=0.25)
        if row == 0:
            ax_v.set_title('Velocity')

        ax_a.plot(t, acc_m[:, d], 'r-', lw=1.3)
        ax_a.plot(t1, acc_r1[:, d], color='darkcyan', ls='-', lw=1.2)
        if acc_r2 is not None:
            ax_a.plot(t2, acc_r2[:, d], color='teal', ls='--', lw=1.2)
        ax_a.axhline(ACC_LIM, color='gray', ls='--', alpha=0.4, lw=0.8)
        ax_a.axhline(-ACC_LIM, color='gray', ls='--', alpha=0.4, lw=0.8)
        ax_a.set_ylabel('acceleration')
        ax_a.grid(True, alpha=0.25)
        if row == 0:
            ax_a.set_title('Acceleration')

        ax_j.plot(t_j, jerk_m[:, d], 'r-', lw=1.3)
        ax_j.plot(t1_j, jerk_r1[:, d], color='darkcyan', ls='-', lw=1.2)
        if jerk_r2 is not None:
            ax_j.plot(t2_j, jerk_r2[:, d], color='teal', ls='--', lw=1.2)
        ax_j.axhline(JERK_LIM, color='gray', ls='--', alpha=0.4, lw=0.8)
        ax_j.axhline(-JERK_LIM, color='gray', ls='--', alpha=0.4, lw=0.8)
        ax_j.set_ylabel('jerk')
        ax_j.grid(True, alpha=0.25)
        if row == 0:
            ax_j.set_title('Jerk (finite diff)')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    axes[-1, 2].set_xlabel('Time (s)')
    axes[-1, 3].set_xlabel('Time (s)')

    t_max = max(t[-1], t1[-1], t2[-1] if pos_r2 is not None else 0)
    for ax in axes.flat:
        ax.set_xlim(0, t_max)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    latency()
    (targets, init_q, mpc_t_pos, mpc_t_vel, mpc_t_acc, mpc_s_pos, mpc_s_vel,
     mpc_s_acc, r_t_pos, r_t_vel, r_t_acc, r_sr_pos, r_sr_vel, r_sr_acc,
     r_srsp_pos, r_srsp_vel, r_srsp_acc) = quality()

    _plot_one('MPC vs Ruckig — Tracking', 'comparison_tracking.png', targets,
              init_q, mpc_t_pos, mpc_t_vel, mpc_t_acc, r_t_pos, r_t_vel,
              r_t_acc, 'Ruckig')
    _plot_one('MPC vs Ruckig — Settle', 'comparison_settle.png', targets,
              init_q, mpc_s_pos, mpc_s_vel, mpc_s_acc, r_sr_pos, r_sr_vel,
              r_sr_acc, 'R raw', r_srsp_pos, r_srsp_vel, r_srsp_acc, 'R rsmpl')
    print(f'\nDone. Plots in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
