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

SARM training requires per-episode subtask annotations stored on the standard
LeRobot episodes metadata. FluxVLA ships the official annotation pipeline
(ported from HuggingFace LeRobot's
`lerobot/data_processing/sarm_annotations/`) under
[`tools/sarm_annotate/`](../tools/sarm_annotate/README.md). It runs a local
Qwen3-VL model on the episode videos and writes the results back as extra
columns (`sparse_subtask_*`, `dense_subtask_*`) to `meta/episodes.jsonl`
(v2.1) or `meta/episodes/*.parquet` (v3.x).

Three annotation modes are supported (matching the three SARM configs under
[`configs/sarm/`](../configs/sarm)):

- `single_stage` — no flags, auto-creates one sparse `"task"` stage per
  episode.
- `dense_only` — `--dense-only --dense-subtasks "Do A, Do B, ..."`.
- `dual` — `--sparse-subtasks "..." --dense-subtasks "..."`.

Example:

```bash
python tools/sarm_annotate/subtask_annotation.py \
  --repo-id /path/to/your/lerobot_dataset \
  --video-key observation.images.image \
  --dense-only \
  --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

See [`tools/sarm_annotate/README.md`](../tools/sarm_annotate/README.md) for
the full set of flags, dependencies (`lerobot>=0.3.4`, `qwen-vl-utils`,
`transformers`, …), and the distributed per-episode runner.

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
