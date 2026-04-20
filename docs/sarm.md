# SARM

FluxVLA's integration of [SARM (Stage-Aware Reward Modeling)](https://github.com/xdofai/opensarm) for long-horizon robot manipulation.

> **Paper**: [SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](https://arxiv.org/abs/2509.25358)  
> **Original Repository**: [https://github.com/xdofai/opensarm](https://github.com/xdofai/opensarm)

## SARM Checkpoints

For SARM-related workflows, keep all dependent models under `./checkpoints` and reference them with relative paths from config files.

Recommended local layout:

```text
checkpoints/
├── Qwen3-VL-32B-Instruct
├── clip-vit-base-patch32
├── sarm_dense_smoke
└── sarm_dense_only_flux_smoke.pt
```

Reserved local names:

- `./checkpoints/Qwen3-VL-32B-Instruct`: VLM used for external SARM annotation workflows.
- `./checkpoints/clip-vit-base-patch32`: CLIP backbone and tokenizer used by FluxVLA SARM configs.
- `./checkpoints/sarm_dense_smoke`: a tested SARM checkpoint directory produced during smoke validation.
- `./checkpoints/sarm_dense_only_flux_smoke.pt`: a FluxVLA-native SARM checkpoint file produced by local smoke training.

## SARM Usage

FluxVLA's SARM configs use relative checkpoint paths under `./checkpoints`, consistent with the rest of the project.

Current SARM configs:

- `configs/sarm/sarm_single_stage_libero_10.py`
- `configs/sarm/sarm_dense_only_libero_10.py`
- `configs/sarm/sarm_dual_libero_10.py`

These configs expect:

- CLIP backbone and tokenizer at `./checkpoints/clip-vit-base-patch32`
- Dataset metadata under the selected `data_root_path`

Notes:

- SARM annotations are read from the standard LeRobot episodes metadata
  (`meta/episodes.jsonl` for v2.1, `meta/episodes/*.parquet` for v3.x) as
  columns named `sparse_subtask_names` / `sparse_subtask_start_frames` /
  `sparse_subtask_end_frames` and their `dense_*` counterparts. This is the
  same source used by the official `lerobot.policies.sarm` implementation.
  Sparse annotations also fall back to unprefixed column names
  (`subtask_names`, `subtask_start_frames`, `subtask_end_frames`) for
  backwards compatibility. Separate `sarm_*_annotations.jsonl` files are no
  longer consulted.
- The external VLM used for annotation should also be placed under `./checkpoints`, for example `./checkpoints/Qwen3-VL-32B-Instruct`.
- `./checkpoints/sarm_dense_smoke` is a preserved smoke-test model directory from the external SARM workflow.
- FluxVLA's `scripts/infer_sarm_progress.py` expects a FluxVLA training checkpoint file such as `./checkpoints/sarm_dense_only_flux_smoke.pt`.
- `scripts/infer_sarm_progress.py` supports `--cfg-options` for dataset overrides and `--max-batches` for quick smoke validation.
- For LeRobot v2.1/v3.x style datasets, `task` can be stored as a task index. FluxVLA resolves it back to task text at read time from `tasks.jsonl` or `tasks.parquet` without modifying dataset files.
- If your dataset uses a camera key other than `observation.images.image`, override `train_dataloader.dataset.video_keys` and `inference_dataset.video_keys` with `--cfg-options`.

## Annotating a Dataset

SARM training reads per-episode subtask annotations from the standard LeRobot
episodes metadata (extra list columns on `meta/episodes.jsonl` for v2.1 /
`meta/episodes/*.parquet` for v3.x — the same format used by
`lerobot.policies.sarm`). FluxVLA ships two routes to produce them, both under
[`tools/sarm_annotate/`](../tools/sarm_annotate/README.md); pick whichever
fits your data.

### Column contract (what FluxVLA reads)

Every episode row is expected to carry list-valued columns:

- `sparse_subtask_names`, `sparse_subtask_start_frames`, `sparse_subtask_end_frames`
- `dense_subtask_names`, `dense_subtask_start_frames`, `dense_subtask_end_frames`
- optional: `*_start_times`, `*_end_times`

Frame indices are inclusive and 0-based. Sparse annotations also fall back to
unprefixed column names (`subtask_names`, `subtask_start_frames`,
`subtask_end_frames`) for backwards compatibility with older sparse-only
datasets.

In addition, `meta/temporal_proportions_sparse.json` /
`meta/temporal_proportions_dense.json` map each subtask name to its average
temporal share across the dataset. Both tools below write these for you.

### Route 1: manual stages (no GPU) — `write_manual_stages.py`

Use when stage boundaries come from humans, task definitions, or heuristic
scripts. Works on both v2.1 and v3.x.

**a) Single-stage `"task"` over every episode** — bootstraps
`single_stage` SARM training without any frame bookkeeping:

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
    --default-sparse auto
```

**b) Multi-stage sparse + dense** from a per-episode JSON spec:

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
    --spec /path/to/my_stages.json
```

where `my_stages.json` is:

```json
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
      "end_frames":   [39,  59,  99, 149, 184, 199]
    }
  },
  { "episode_index": 1, "sparse": "auto", "dense": "auto" }
]
```

`"auto"` is shorthand for a single `"task"` stage spanning the whole episode.
Missing entries / missing `sparse`/`dense` keys are left untouched unless you
pass `--default-sparse auto` / `--default-dense auto`.

**c) Dense-only with auto sparse fallback** (matches
`configs/sarm/sarm_dense_only_*.py`):

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
    --spec /path/to/my_dense_stages.json \
    --default-sparse auto
```

### Route 2: VLM-based auto-annotation — `subtask_annotation.py`

Runs a local Qwen3-VL model on the episode videos and writes the same columns
back. Requires a GPU (≥16 GB VRAM for the 30B MoE variant) and
`pip install qwen-vl-utils transformers`. Three modes:

| Mode           | CLI invocation                                               | Intended config                         |
|----------------|--------------------------------------------------------------|-----------------------------------------|
| `single_stage` | no `--sparse-subtasks` / `--dense-subtasks` args             | `configs/sarm/sarm_single_stage_*.py`   |
| `dense_only`   | `--dense-only --dense-subtasks "Do A, Do B, ..."`            | `configs/sarm/sarm_dense_only_*.py`     |
| `dual`         | `--sparse-subtasks "..." --dense-subtasks "..."`             | `configs/sarm/sarm_dual_*.py`           |

```bash
python tools/sarm_annotate/subtask_annotation.py \
  --repo-id /path/to/your/lerobot_dataset \
  --video-key observation.images.image \
  --dense-only \
  --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

For large datasets, `run_vlm_dense_subset.py` parallelises annotation across
worker processes.

### Inspect / validate / reset

- `tools/sarm_annotate/parse_sparse_episode_info.py` — per-episode
  `num_sparse_stages` + temporal proportions.
- `tools/sarm_annotate/parse_dense_episode_info.py` — same, for dense.
- `tools/sarm_annotate/fix_sparse_annotations.py` — force single-stage
  `"task"` on a v3.x dataset.
- `tools/sarm_annotate/clear_written_annotations.py --dataset-root <root>
  [--apply]` — dry-run / remove all SARM columns + proportions files from a
  v3.x dataset.

See [`tools/sarm_annotate/README.md`](../tools/sarm_annotate/README.md) for
the full list of scripts and their upstream provenance.

## Training

Example training command:

```bash
export WANDB_MODE=disabled
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/sarm/sarm_dense_only_libero_10.py \
  --work-dir ./work_dirs/sarm_dense_only_libero_10 \
  --cfg-options train_dataloader.per_device_batch_size=1
```

Example real-dataset smoke training command:

```bash
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/sarm/sarm_dense_only_libero_10.py \
  --work-dir ./tmp/sarm_dense_only_flux_smoke \
  --cfg-options \
    model.data_root_path=/path/to/your_dataset \
    train_dataloader.dataset.data_root_path=/path/to/your_dataset \
    inference_dataset.data_root_path=/path/to/your_dataset \
    train_dataloader.dataset.video_keys="['observation.images.cam_high']" \
    inference_dataset.video_keys="['observation.images.cam_high']" \
    train_dataloader.per_device_batch_size=1 \
    train_dataloader.per_device_num_workers=0 \
    runner.max_steps=1 \
    runner.max_epochs=None \
    runner.save_iter_interval=1
```

Expose the generated FluxVLA checkpoint under `./checkpoints`:

```bash
ln -sfn \
  "$PWD/tmp/sarm_dense_only_flux_smoke/checkpoints/latest-checkpoint.pt" \
  "$PWD/checkpoints/sarm_dense_only_flux_smoke.pt"
```

## Inference

Example offline progress inference command:

```bash
python scripts/infer_sarm_progress.py \
  --config configs/sarm/sarm_dense_only_libero_10.py \
  --ckpt-path ./checkpoints/sarm_dense_only_flux_smoke.pt \
  --output-path ./work_dirs/sarm_progress.jsonl \
  --head-mode dense \
  --batch-size 1
```

Example real-dataset smoke inference command:

```bash
python scripts/infer_sarm_progress.py \
  --config configs/sarm/sarm_dense_only_libero_10.py \
  --ckpt-path ./checkpoints/sarm_dense_only_flux_smoke.pt \
  --output-path ./tmp/sarm_progress_flux_smoke.jsonl \
  --head-mode dense \
  --batch-size 1 \
  --max-batches 1 \
  --cfg-options \
    model.data_root_path=/path/to/your_dataset \
    inference_dataset.data_root_path=/path/to/your_dataset \
    inference_dataset.video_keys="['observation.images.cam_high']"
```
