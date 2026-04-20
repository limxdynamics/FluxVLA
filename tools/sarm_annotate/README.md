# SARM Subtask Annotation

Standalone annotation pipeline that produces the `sparse_*` / `dense_*` subtask
columns consumed by FluxVLA's SARM models and by the official
`lerobot.policies.sarm` implementation.

Ported, with only import-path adjustments, from the HuggingFace LeRobot fork:
[`lerobot/data_processing/sarm_annotations/`](https://github.com/huggingface/lerobot).

## What it does

Takes a standard LeRobot v2.1 / v3.x dataset and uses a local Qwen3-VL model to
identify when subtasks occur in each episode, then writes the results back to
the standard LeRobot episodes metadata as extra columns on
`meta/episodes.jsonl` (v2.1) or `meta/episodes/*.parquet` (v3.x):

- `sparse_subtask_names`, `sparse_subtask_start_frames`, `sparse_subtask_end_frames`
- `dense_subtask_names`, `dense_subtask_start_frames`, `dense_subtask_end_frames`
- (optional) `*_start_times`, `*_end_times`
- `meta/temporal_proportions_{sparse,dense}.json`

These are exactly the columns loaded by
[`fluxvla/datasets/utils/sarm_utils.py`](../../fluxvla/datasets/utils/sarm_utils.py)
and used by [`fluxvla/datasets/sarm_dataset.py`](../../fluxvla/datasets/sarm_dataset.py).

## Three annotation modes

| Mode           | CLI invocation                                               | Intended FluxVLA config                 |
|----------------|--------------------------------------------------------------|-----------------------------------------|
| `single_stage` | no `--sparse-subtasks` / `--dense-subtasks` args             | `configs/sarm/sarm_single_stage_*.py`   |
| `dense_only`   | `--dense-only --dense-subtasks "Do A, Do B, Do C"`           | `configs/sarm/sarm_dense_only_*.py`     |
| `dual`         | `--sparse-subtasks "..." --dense-subtasks "..."`             | `configs/sarm/sarm_dual_*.py`           |

`single_stage` auto-creates a single sparse `"task"` stage covering the whole
episode — useful for quick SARM training on datasets that do not yet have
human / VLM subtask annotations.

## Requirements

```bash
pip install "lerobot>=0.3.4" qwen-vl-utils transformers torch opencv-python pydantic
```

A GPU with enough VRAM for the chosen Qwen3-VL variant (≥16 GB recommended for
the 30B MoE model) is required for `VideoAnnotator`. The `single_stage` /
`fix_sparse`-style flows do not need a GPU.

## Usage

### From a local LeRobot dataset

```bash
python tools/sarm_annotate/subtask_annotation.py \
    --repo-id /path/to/your/lerobot_dataset \
    --video-key observation.images.image \
    --dense-only \
    --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

### Dual (sparse + dense)

```bash
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

## Relationship to the on-line FluxVLA SARM reader

After running the pipeline on a dataset, it becomes drop-in compatible with
FluxVLA:

```python
# configs/sarm/sarm_dense_only_libero_10.py
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

All scripts are ports of files from the HuggingFace LeRobot fork used on the
training server:

| Local file                       | Upstream                                                              |
|----------------------------------|-----------------------------------------------------------------------|
| `subtask_annotation.py`          | `lerobot/data_processing/sarm_annotations/subtask_annotation.py`      |
| `subtask_annotation_timing.py`   | `lerobot/data_processing/sarm_annotations/subtask_annotation_timing.py` |
| `run_vlm_dense_subset.py`        | `lerobot/scripts/run_vlm_dense_subset.py`                             |

Only the cross-module import paths were adjusted so the files work as
`tools/sarm_annotate/*.py` inside FluxVLA; all core annotation logic is
unchanged.
