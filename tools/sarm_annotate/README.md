# SARM Subtask Annotation

Tools for producing and maintaining the `sparse_*` / `dense_*` subtask columns
that FluxVLA's SARM models (and the official `lerobot.policies.sarm`
implementation) read from standard LeRobot episodes metadata.

Two complementary entry points:

1. **VLM-based pipeline** (`subtask_annotation.py`, ...) — ported from the
   HuggingFace LeRobot fork
   [`lerobot/data_processing/sarm_annotations/`](https://github.com/huggingface/lerobot).
   Uses a local Qwen3-VL model to auto-annotate episode videos.
2. **Manual pipeline** (`write_manual_stages.py`) — writes user-provided stage
   boundaries (or a trivial single-stage `"task"`) straight into the episodes
   metadata. No GPU / VLM required.

Both routes produce byte-compatible columns.

Published reference datasets on Hugging Face:

- Manually annotated training / inference dataset: [`limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv3.0)
- Unannotated dataset intended for manual or VLM labeling: [`limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv3.0)

Download them under `./datasets` with:

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
```

LeRobot v3.x video metadata sanity check:

- LeRobot v3.x allows either many episodes in one MP4 or one MP4 per episode.
- If many episodes share one MP4, each episode that points to that file must
  use correct `from_timestamp` / `to_timestamp` offsets.
- If videos are already split as `file-000.mp4`, `file-001.mp4`, ..., each
  episode should point to its own `file_index`, and `from_timestamp` will
  usually reset to `0.0`.
- If the directory contains multiple MP4 files but all episodes still point to
  `file-000.mp4`, the dataset metadata is malformed and should be fixed before
  training, annotation, or pushing to the Hub.

## What gets written

Takes a standard LeRobot v2.1 (`meta/episodes.jsonl`) or v3.x
(`meta/episodes/*.parquet`) dataset and adds these extra columns to the
episodes metadata:

- `sparse_subtask_names`, `sparse_subtask_start_frames`, `sparse_subtask_end_frames`
- `dense_subtask_names`, `dense_subtask_start_frames`, `dense_subtask_end_frames`
- (optional) `*_start_times`, `*_end_times`
- `meta/temporal_proportions_{sparse,dense}.json`

These are exactly the columns loaded by
[`fluxvla/datasets/utils/sarm_utils.py`](../../fluxvla/datasets/utils/sarm_utils.py)
and used by [`fluxvla/datasets/sarm_dataset.py`](../../fluxvla/datasets/sarm_dataset.py).

## Manual annotation — `write_manual_stages.py`

Use this when you already know the stage boundaries (human annotation,
heuristic script, up-stream task definition) and just want the columns
written. Works on both v2.1 and v3.x and (re)computes
`temporal_proportions_{sparse,dense}.json` from the data.

### Minimal: single-stage `"task"` over every episode

Bootstraps `single_stage` SARM training without touching frame numbers:

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
    --default-sparse auto
```

Matches `configs/sarm/sarm_single_stage.py`.

### Dense-only with a VLM-less fallback

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
    --spec specs/my_dense_stages.json \
    --default-sparse auto
```

Matches `configs/sarm/sarm_dense_only.py`.

### Dual multi-stage (both sparse and dense, per-episode spec)

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
    --spec specs/my_dual_stages.json
```

Matches `configs/sarm/sarm_dual.py`.

### Spec format

A JSON file (`.json` list) or JSONL (`.jsonl`) with one entry per episode.
Frame indices are **inclusive and 0-based**:

```jsonc
[
  {
    "episode_index": 0,
    "sparse": {
      "names":        ["reach", "grasp", "place"],
      "start_frames": [0,        60,      150],
      "end_frames":   [59,      149,      199]
    },
    "dense": {
      "names":        ["move_to_cup","close_gripper","lift","move_to_plate","lower","open_gripper"],
      "start_frames": [0,   40,  60, 100, 150, 185],
      "end_frames":   [39,  59,  99, 149, 184, 199],
      "start_times":  [0.00, 0.80, 1.20, 2.00, 3.00, 3.70],
      "end_times":    [0.78, 1.18, 1.98, 2.98, 3.68, 3.98]
    }
  },

  // shorthand: single "task" stage covering the whole episode
  { "episode_index": 1, "sparse": "auto", "dense": "auto" }
]
```

Anything not specified (either missing key or missing episode entry plus no
matching `--default-sparse` / `--default-dense`) is left untouched.

### Useful flags

- `--spec <file>`: per-episode stage spec (.json list or .jsonl).
- `--default-sparse auto` / `--default-dense auto`: fallback for episodes not
  in the spec (single `"task"` stage).
- `--skip-proportions`: do not (re)write `temporal_proportions_*.json`.

### Inspect / validate / reset

- `parse_sparse_episode_info.py --dataset-root <root>`: dumps
  `sparse_episode_info.json` with per-episode num_sparse_stages and computed
  temporal proportions. v3.x only.
- `parse_dense_episode_info.py --dataset-root <root>`: same for dense. v3.x
  only.
- `fix_sparse_annotations.py --dataset-root <root>`: forces sparse to a
  single `"task"` stage per episode. v3.x only — for v2.1 use
  `write_manual_stages.py --default-sparse auto`.
- `clear_written_annotations.py --dataset-root <root> [--apply]`: dry-run
  (default) or removal of all `sparse_*` / `dense_*` / `subtask_*` columns
  and `temporal_proportions_*.json`. v3.x only.

## VLM-based annotation — `subtask_annotation.py`

Uses a local Qwen3-VL model to auto-annotate episode videos. Three modes:

| Mode           | CLI invocation                                     | Intended FluxVLA config             |
| -------------- | -------------------------------------------------- | ----------------------------------- |
| `single_stage` | no `--sparse-subtasks` / `--dense-subtasks` args   | `configs/sarm/sarm_single_stage.py` |
| `dense_only`   | `--dense-only --dense-subtasks "Do A, Do B, Do C"` | `configs/sarm/sarm_dense_only.py`   |
| `dual`         | `--sparse-subtasks "..." --dense-subtasks "..."`   | `configs/sarm/sarm_dual.py`         |

### Requirements (VLM path only)

```bash
pip install "lerobot>=0.3.4" qwen-vl-utils transformers torch opencv-python pydantic
```

A GPU with enough VRAM for the chosen Qwen3-VL variant (≥16 GB recommended for
the 30B MoE model) is required for `VideoAnnotator`. The manual pipeline does
not need any of these.

### Practical notes

- `--model` may point to a Hugging Face repo ID, a concrete snapshot directory, or a local cache root such as `./checkpoints/Qwen3-VL-30B-A3B-Instruct`; the script resolves cache roots automatically to a loadable `snapshots/*` directory.
- If the default video decoder is unavailable in your environment, pass `--video-backend pyav` or `--video-backend video_reader`. Leaving it unset keeps LeRobot's default behavior.
- The auto-annotation path resolves the actual Qwen model class from config, so Qwen2-VL, Qwen2.5-VL, and Qwen3-VL variants do not need hard-coded class names.

### Examples

```bash
# Local dataset, dense-only
python tools/sarm_annotate/subtask_annotation.py \
    --repo-id ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
  --model ./checkpoints/Qwen3-VL-30B-A3B-Instruct \
    --video-key observation.images.cam_high \
  --video-backend pyav \
    --dense-only \
    --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

```bash
# HF Hub dataset, dual (sparse + dense)
python tools/sarm_annotate/subtask_annotation.py \
    --repo-id your-username/your-dataset \
    --video-key observation.images.base \
    --sparse-subtasks "Pick up the cup, Place it on the plate" \
    --dense-subtasks "Reach, Grasp, Lift, Transport, Release" \
    --push-to-hub
```

### Distributed per-episode runner

`run_vlm_dense_subset.py` wraps `subtask_annotation.py` to annotate a subset of
episodes of a single dataset in parallel worker processes — useful when a
single-GPU run would be too slow for a large dataset.

### Re-processing to cleaner splits

`subtask_annotation_timing.py` re-derives timing columns from an already
annotated dataset (e.g. smoothing subtask boundaries, converting timestamps).

## Consuming the annotations from FluxVLA

After either pipeline runs, the dataset is drop-in compatible with the SARM
configs:

```python
# any dense_only config under configs/sarm/
train_dataloader = dict(
    dataset=dict(
        type='SARMDataset',
        data_root_path='/path/to/annotated_dataset',
        annotation_type='dense',
        ...
    ),
)
```

No FluxVLA-specific annotation files (`sarm_*_annotations.jsonl`, etc.) are
produced or required — the annotations live inside the standard LeRobot
episodes metadata, which is also what `lerobot.policies.sarm` reads.

## Origin

| Local file                     | Upstream                                                                | Changes                                                    |
| ------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------- |
| `subtask_annotation.py`        | `lerobot/data_processing/sarm_annotations/subtask_annotation.py`        | Only import paths for sibling module.                      |
| `subtask_annotation_timing.py` | `lerobot/data_processing/sarm_annotations/subtask_annotation_timing.py` | Only import paths.                                         |
| `run_vlm_dense_subset.py`      | `lerobot/scripts/run_vlm_dense_subset.py`                               | Only import paths.                                         |
| `parse_sparse_episode_info.py` | `lerobot/parse_sparse_episode_info.py`                                  | Verbatim.                                                  |
| `parse_dense_episode_info.py`  | `lerobot/parse_dense_episode_info.py`                                   | Verbatim.                                                  |
| `fix_sparse_annotations.py`    | `lerobot/fix_sparse_annotations.py`                                     | Verbatim.                                                  |
| `clear_written_annotations.py` | `lerobot/clear_written_annotations.py`                                  | Verbatim.                                                  |
| `write_manual_stages.py`       | FluxVLA-native                                                          | Supports both v2.1 and v3.x from a simple JSON/JSONL spec. |

All upstream files retain their Apache 2.0 HuggingFace headers.
