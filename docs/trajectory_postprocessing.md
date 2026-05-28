# Trajectory Postprocessing

Trajectory Postprocess sits between policy output and execution. Its role is to convert an action chunk into a trajectory that is more suitable for execution, with clearer bounds on higher-order quantities such as velocity, acceleration, and jerk, as well as better behavior at chunk boundaries.

## Problem

Trajectory postprocessing is introduced mainly to address three classes of issues.

1. A raw action chunk may contain local oscillations, and its implied velocity, acceleration, and jerk may be unnecessarily large.
2. Two adjacent chunks may appear reasonable in isolation while still producing a visible discontinuity at the boundary.
3. Under asynchronous execution, part of the previous chunk has often already been consumed by the controller, so the next chunk must connect to the actual execution state rather than simply replace the previous plan.

The first class concerns bounded higher-order behavior within a chunk, the second concerns cross-chunk continuity, and the third concerns continuity under time offset and partial execution history.

## Method

The current implementation consists of two parts: one for bounded higher-order behavior within the current chunk, and one for cross-chunk connection.

### Optimization component

This component acts directly on the current chunk. The input is a target trajectory and the output is a trajectory with tighter control over velocity, acceleration, and jerk. When required, the terminal state can also be driven closer to rest.

Two backends are currently provided.

#### `joint_mpc`

`joint_mpc` formulates the per-joint trajectory adjustment problem as a constrained optimization problem and solves it with OSQP.

Its main properties are:

- explicit modeling of position, velocity, acceleration, and jerk;
- direct constraints on these higher-order quantities;
- stronger tracking ability with respect to the target trajectory;
- higher computational cost and longer runtime.

The current implementation reduces part of the runtime cost through per-joint parallel solves and solver-cache reuse.
The benchmark performs warm-up before recording latency.

For `joint_mpc`, `mode='settle'` adds a light terminal-rest bias.

#### `ruckig_filter`

`ruckig_filter` uses Ruckig as a jerk-limited online trajectory generator.

Its main properties are:

- relatively direct implementation;
- stable behavior;
- easier satisfaction of higher-order bounds;
- no guarantee of strict target tracking;
- a tendency to introduce trajectory-level lag.

Here, lag mainly means that the trajectory response is delayed relative to the target trajectory, rather than higher per-call runtime.
Under the current usage, Ruckig behaves more like a step-by-step tracker, so it satisfies bounds more reliably but usually introduces some reference-tracking lag; by contrast, `joint_mpc` optimizes over the full horizon, usually achieves tighter target tracking, and incurs higher computational cost.

### Stitching mechanism

Stitching addresses cross-chunk connection rather than within-chunk optimization.

When stitching is enabled, the system:

1. resamples the previous processed trajectory onto the first timestamps of the current chunk;
2. uses those samples directly as the prefix of the current chunk;
3. continues solving only after the stitching boundary.

The purpose is to connect the new chunk to the actual execution history instead of treating it as a fresh trajectory.

### Relation to RTC

Trajectory Postprocess and RTC both improve cross-chunk continuity, but they operate at different levels.

- RTC acts during prediction, constraining or guiding generation so that the current chunk is more consistent with a known action prefix;
- Trajectory Postprocess acts after prediction and focuses on execution-side handling of higher-order bounds and boundary transitions.

In practice, the two can be used together: RTC addresses prediction-time consistency, while Trajectory Postprocess handles execution-time bounds and boundary behavior.

## Installation

The postprocessing backends and benchmark visualization depend on additional packages. If they are not already available in the current environment, install them explicitly with:

```bash
pip install osqp ruckig matplotlib
```

Package roles:

- `osqp`: required by `joint_mpc`;
- `ruckig`: required by `ruckig_filter`;
- `matplotlib`: required by benchmark plots and debug visualization.

## Key Implementation and Results

The implementation lives under `fluxvla/engines/utils/postprocess/`.

- `Trajectory`: stores timestamped positions, velocities, and accelerations;
- `TrajectoryPostprocessor`: handles initial state processing, stitching, backend dispatch, and result assembly;
- `joint_mpc`: OSQP-based per-joint optimization backend;
- `ruckig_filter`: Ruckig-based jerk-limited filtering backend;
- `plot_utils`: asynchronous plotting utilities for debugging and comparison.

At runtime, this capability is integrated into `AlohaInferenceRunner` and `Tron2InferenceRunner` through `inference.postprocess_config`.

### Tracking

Tracking is intended for rolling inference. The controller repeatedly receives fixed-length chunks, and the objective is to execute the near-term segment under higher-order bounds without requiring terminal rest.

The benchmark numbers on this page were measured on the current test machine with an Intel Xeon Platinum 8336C CPU @ 2.30GHz.

The latency numbers correspond to the following benchmark setting:

- degrees of freedom: 12
- horizon length: `N=50`

Under the default setting of [scripts/benchmark_postprocess.py](scripts/benchmark_postprocess.py), the average tracking latency is:

| Method | Tracking Latency |
| ------ | ---------------- |
| MPC    | 4.9 ms           |
| Ruckig | 1.17 ms          |

![Tracking comparison](../assets/postprocess_comparison_tracking.png)

This figure compares raw targets, `joint_mpc`, and `ruckig` in tracking mode.

The main observations are:

1. both methods satisfy velocity, acceleration, and jerk bounds to a good approximation;
2. `joint_mpc` tracks the target trajectory more closely;
3. `ruckig` introduces more lag than `joint_mpc`, though in practice this lag may still be acceptable depending on the task and controller.

### Settle

Settle is intended for single-frame inference, where one chunk is executed more like a complete motion segment than a streaming handoff.

In the current implementation:

- `joint_mpc` uses `mode='settle'` to add light terminal and settle weights within the original fixed-length horizon;
- `ruckig` uses `max_settle_steps` to continue converging beyond the last target, then returns a fixed-length result through `resample_settle=True` by default.

![Settle comparison](../assets/postprocess_comparison_settle.png)

`R raw` and `R rsmpl` are not different algorithms. They are two views of the same Ruckig settle trajectory:

- `R raw` keeps the naturally extended settle result;
- `R rsmpl` resamples it back to the original horizon length.

Resampling is needed because the current postprocessor and runners operate on fixed-length chunks.

Note: resampling compacts the extended settle trajectory back onto the original time grid, so output velocity, acceleration, and jerk may exceed the configured limits.

### Stitching

Stitching is intended for asynchronous execution, where the current chunk arrives after part of the previous chunk has already been executed.

![Stitching runtime plot](../assets/postprocess_stitch_runtime.jpg)

This runtime plot illustrates the stitching mechanism and can be obtained with `debug_plot=True`. Under the current implementation, the stitched result remains continuous in position, velocity, and acceleration at the boundary; in the discrete sampled sense, this can be viewed as C2 continuity at the stitching boundary. The stitched result follows the previous frame more naturally, while the non-stitched result exhibits a more visible bend near handoff.

## Configuration

Currently the trajectory postprocessor has been integrated into `AlohaInferenceRunner` and `Tron2InferenceRunner`. For other runners, refer to these two implementations as a reference for adaptation.

Only two common configurations are listed below: tracking and settle.

### Tracking

Tracking commonly corresponds to asynchronous execution. The recommended setting is therefore `async_execution=True` with stitching enabled.

```python
inference = dict(
    type='AlohaInferenceRunner',
    async_execution=True,
    execute_horizon=10,
    postprocess_config=dict(
        enabled=True,
        method='joint_mpc',
        mode='tracking',
        num_stitch=6,
        max_velocity=5.0,
        max_acceleration=20.0,
        max_jerk=20.0,
        tracking_weight=3.0,
    ))
```

For online debugging, plotting can be enabled:

```python
inference = dict(
    type='AlohaInferenceRunner',
    async_execution=True,
    execute_horizon=10,
    postprocess_config=dict(
        enabled=True,
        method='joint_mpc',
        mode='tracking',
        num_stitch=6,
        max_velocity=5.0,
        max_acceleration=20.0,
        max_jerk=20.0,
        tracking_weight=3.0,
        debug_plot=True,
        debug_plot_dofs=[0, 1, 2],
    ))
```

### Settle

Settle commonly corresponds to single-frame inference. The recommended setting is therefore `async_execution=False`.

```python
inference = dict(
    type='AlohaInferenceRunner',
    async_execution=False,
    execute_horizon=10,
    postprocess_config=dict(
        enabled=True,
        method='joint_mpc',
        mode='settle',
        num_stitch=0,
        max_velocity=5.0,
        max_acceleration=40.0,
        max_jerk=80.0,
        terminal_weight=1.0,
        settle_weight=1.0,
    ))
```

For OSQP-based `joint_mpc` settle mode, stop-start trajectories usually need
looser limits; `max_acceleration=40.0` and `max_jerk=80.0` are practical
starting points. `terminal_weight` and `settle_weight` control the trade-off
between terminal convergence and target tracking; the current default is a
light bias of 1.0 for each.

To use the Ruckig settle behavior:

```python
inference = dict(
    type='AlohaInferenceRunner',
    async_execution=False,
    execute_horizon=10,
    postprocess_config=dict(
        enabled=True,
        method='ruckig',
        mode='settle',
        num_stitch=0,
        max_velocity=5.0,
        max_acceleration=40.0,
        max_jerk=80.0,
        max_settle_steps=15,
    ))
```

## Adaptation Considerations

`AlohaInferenceRunner` and `Tron2InferenceRunner` already include full postprocessing support and can serve as reference implementations. When integrating this capability into a new robot or a new runner, refer to these two runners and consider the following points.

### 1. Low-level control interface

Whether trajectory postprocessing is needed should first be determined by the form of the robot's low-level control interface. Different platforms expose movej, servoj, MIT-style control, or force-control interfaces with different timing semantics, feedback structure, and built-in smoothing behavior, so the need for additional postprocessing should be evaluated together with the execution mechanism.

### 2. Controlled DOFs

Not every output dimension is suitable for the same postprocessing path.

- Arm joints are generally suitable for continuous postprocessing;
- grippers, suction states, and other discrete channels usually require separate handling.

### 3. Sync versus async execution

The need for stitching depends directly on the execution mode.

- Under synchronous execution, settle is usually preferred;
- under asynchronous execution, tracking + stitching is usually preferred, and continuity around the boundary should be validated.

### 4. Robot-side protection logic

Trajectory Postprocess is responsible for producing a trajectory that is more suitable for execution. Existing protection logic in the robot and the runner should remain intact. Integration should therefore verify that the postprocessed result does not interfere with clipping, emergency stop, or execution-side protection flows.

### 5. Limit Tuning

The limits (`max_velocity`, `max_acceleration`, `max_jerk`) control the trade-off between tracking fidelity and motion smoothness. Tighter limits usually produce smoother but more conservative motion; looser limits track targets more closely but can become more aggressive and less robust. Tune them against hardware behavior, and use `debug_plot` to inspect the actual postprocessed trajectory.

Recommended starting points:

| Mode     | `max_acceleration` | `max_jerk` |
| -------- | ------------------ | ---------- |
| tracking | 20.0               | 20.0       |
| settle   | 40.0               | 80.0       |
