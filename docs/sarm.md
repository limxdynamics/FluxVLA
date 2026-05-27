# SARM

FluxVLA's integration of [SARM (Stage-Aware Reward Modeling)](https://github.com/xdofai/opensarm) for long-horizon robot manipulation.

> **Paper**: [SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](https://arxiv.org/abs/2509.25358)
> **Original Repository**: [https://github.com/xdofai/opensarm](https://github.com/xdofai/opensarm)

## SARM Checkpoints

For SARM-related workflows, keep all dependent models under `./checkpoints` and reference them with relative paths from config files.

FluxVLA training checkpoints are written to the training output directory instead, typically under the `checkpoints/` subdirectory of the `--work-dir` you pass to `scripts/train.py`, for example `./work_dirs/<run_name>/checkpoints/latest-checkpoint.pt`.

Recommended local layout:

```text
checkpoints/
├── Qwen3-VL-30B-A3B-Instruct
└── clip-vit-base-patch32
```

Reserved local names:

- `./checkpoints/Qwen3-VL-30B-A3B-Instruct`: VLM used for external SARM annotation workflows.
- `./checkpoints/clip-vit-base-patch32`: CLIP backbone and tokenizer used by FluxVLA SARM configs.
- Training-generated FluxVLA checkpoints live under `--work-dir/checkpoints/` instead of the root `./checkpoints/` directory.

## SARM Usage

FluxVLA's SARM configs use relative checkpoint paths under `./checkpoints`, consistent with the rest of the project.

Current SARM configs:

- `configs/sarm/sarm_single_stage.py`
- `configs/sarm/sarm_dense_only.py`
- `configs/sarm/sarm_dual.py`

These starter configs now default to the published manual SARM dataset at `./datasets/SARM_manual_test_10Episodes_lerobotv3.0` and use `observation.images.cam_high`, which matches the released 10-episode example datasets on Hugging Face. The newer `./datasets/SARM_manual_test_10Episodes_lerobotv2.1` conversion has also been smoke-tested end to end with all three SARM configs by overriding the dataset roots. If you keep the dataset under a different local path or want to use another camera stream, override both `data_root_path` and `video_keys` with `--cfg-options`.

Published Hugging Face datasets:

- LeRobot v3.x manual annotations for training / inference: [`limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv3.0)
- LeRobot v3.x unlabeled dataset for manual or VLM labeling: [`limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv3.0)
- New LeRobot v2.1 manual conversion for training / inference and legacy-tool compatibility: [`limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv2.1`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv2.1)
- New LeRobot v2.1 unlabeled conversion for manual or VLM labeling workflows: [`limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv2.1`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv2.1)

One-command setup:

```bash
# Default: SARM manual v3.0 example data + CLIP checkpoint
bash scripts/setup_sarm_data_ckpts.sh

# LeRobot v2.1 SARM manual example data + CLIP checkpoint
bash scripts/setup_sarm_data_ckpts.sh --version v2

# Also prepare VLM auto-annotation inputs and Qwen3-VL
bash scripts/setup_sarm_data_ckpts.sh --with-vlm --mirror

# Prepare all released SARM v2.1/v3.0 examples and related checkpoints
bash scripts/setup_sarm_data_ckpts.sh --all
```

The script downloads SARM data from
[`limxdynamics/FluxVLAData`](https://huggingface.co/datasets/limxdynamics/FluxVLAData),
CLIP from
[`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32),
and optional Qwen3-VL from
[`Qwen/Qwen3-VL-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct).
It skips any dataset or checkpoint directory that is already present, so it is
safe to rerun. Use `--dry-run` to preview commands without downloading.

Download them locally under `./datasets` with:

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv2.1/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv2.1/*" --local-dir ./datasets
```

Use the `manual_*` datasets directly for training / inference. Use the `vlm_*` datasets as clean starting points for manual stage writing or VLM auto-annotation. Prefer the v2.1 pair when another tool expects `meta/episodes.jsonl` plus per-episode videos; prefer the v3.0 pair when you want to keep native LeRobot v3.x metadata layout.

LeRobot v3.x video metadata sanity check:

- LeRobot v3.x allows either many episodes in one MP4 or one MP4 per episode.
- If many episodes share one MP4, each episode that points to that file must
  use correct `from_timestamp` / `to_timestamp` offsets.
- If videos are already split as `file-000.mp4`, `file-001.mp4`, ..., each
  episode should point to its own `file_index`, and `from_timestamp` will
  usually reset to `0.0`.
- If the directory contains multiple MP4 files but all episodes still point to
  `file-000.mp4`, the dataset metadata is malformed and should be fixed before
  use.

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
- The external VLM used for annotation should also be placed under `./checkpoints`, for example `./checkpoints/Qwen3-VL-30B-A3B-Instruct`. If you pass a local Hugging Face cache root instead of a concrete snapshot directory, FluxVLA resolves it automatically to the matching `snapshots/*` entry.
- `scripts/infer_sarm_progress.py` expects a FluxVLA training checkpoint file, typically the `latest-checkpoint.pt` file under your chosen `--work-dir/checkpoints/` directory.
- `scripts/infer_sarm_progress.py` supports `--cfg-options` for dataset overrides and `--max-batches` for quick smoke validation.
- For LeRobot v2.1/v3.x style datasets, `task` can be stored as a task index. FluxVLA resolves it back to task text at read time from `tasks.jsonl` or `tasks.parquet` without modifying dataset files.
- For LeRobot v3.x style datasets, video paths may be described either by `videos/<key>/chunk_index` / `file_index` or equivalent chunk/file columns on episode metadata or parquet rows. FluxVLA accepts those variants without requiring dataset rewrites.
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
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv2.1 \
    --default-sparse auto
```

**b) Multi-stage sparse + dense** from a per-episode JSON spec:

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv2.1 \
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
`configs/sarm/sarm_dense_only.py`):

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv2.1 \
    --spec /path/to/my_dense_stages.json \
    --default-sparse auto
```

### Route 2: VLM-based auto-annotation — `subtask_annotation.py`

Runs a local Qwen3-VL model on the episode videos and writes the same columns
back. Requires a GPU (≥16 GB VRAM for the 30B MoE variant) and
`pip install qwen-vl-utils transformers`. Three modes:

| Mode           | CLI invocation                                    | Intended config                     |
| -------------- | ------------------------------------------------- | ----------------------------------- |
| `single_stage` | no `--sparse-subtasks` / `--dense-subtasks` args  | `configs/sarm/sarm_single_stage.py` |
| `dense_only`   | `--dense-only --dense-subtasks "Do A, Do B, ..."` | `configs/sarm/sarm_dense_only.py`   |
| `dual`         | `--sparse-subtasks "..." --dense-subtasks "..."`  | `configs/sarm/sarm_dual.py`         |

```bash
python tools/sarm_annotate/subtask_annotation.py \
  --repo-id ./datasets/SARM_vlm_test_10Episodes_lerobotv2.1 \
  --video-key observation.images.cam_high \
  --dense-only \
  --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

For large datasets, `run_vlm_dense_subset.py` parallelises annotation across
worker processes.

### Visualize annotation results

The VLM annotation script also ports LeRobot's SARM visualization path. It can
render already written sparse/dense annotations as PNGs with sampled frames and
a color-coded subtask timeline:

```bash
python tools/sarm_annotate/subtask_annotation.py \
  --repo-id ./datasets/SARM_manual_test_10Episodes_lerobotv2.1 \
  --video-key observation.images.cam_high \
  --visualize-only \
  --visualize-type both \
  --episodes 0 1 2 \
  --output-dir ./subtask_viz
```

For a normal VLM annotation run, visualizations are generated automatically at
the end. Set `--num-visualizations 0` to skip them, or use
`--visualize-type sparse`, `dense`, or `both` to choose which columns are
rendered.

### Inspect / validate / reset

- `tools/sarm_annotate/parse_sparse_episode_info.py` — per-episode
  `num_sparse_stages` + temporal proportions.
- `tools/sarm_annotate/parse_dense_episode_info.py` — same, for dense.
- `tools/sarm_annotate/fix_sparse_annotations.py` — force single-stage
  `"task"` on a v3.x dataset.
- `tools/sarm_annotate/clear_written_annotations.py --dataset-root <root> [--apply]` — dry-run / remove all SARM columns + proportions files from a
  v3.x dataset.

See [`tools/sarm_annotate/README.md`](../tools/sarm_annotate/README.md) for
the full list of scripts and their upstream provenance.

## Training

Example training command:

```bash
export WANDB_MODE=disabled
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/sarm/sarm_dense_only.py \
  --work-dir ./work_dirs/sarm_dense_only \
  --cfg-options train_dataloader.per_device_batch_size=1
```

Example minimal real-dataset training command:

```bash
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/sarm/sarm_dense_only.py \
  --work-dir ./work_dirs/sarm_dense_only_your_dataset \
  --cfg-options \
    model.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv2.1 \
    train_dataloader.dataset.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv2.1 \
    inference_dataset.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv2.1 \
    train_dataloader.dataset.video_keys="['observation.images.cam_high']" \
    inference_dataset.video_keys="['observation.images.cam_high']" \
    train_dataloader.per_device_batch_size=1 \
    train_dataloader.per_device_num_workers=0 \
    runner.max_steps=1 \
    runner.max_epochs=None \
    runner.save_iter_interval=1
```

## Inference

Example offline progress inference command:

```bash
python scripts/infer_sarm_progress.py \
  --config configs/sarm/sarm_dense_only.py \
  --ckpt-path ./work_dirs/sarm_dense_only/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/sarm_dense_only/sarm_progress.jsonl \
  --head-mode dense \
  --batch-size 1 \
  --num-visualizations 5 \
  --output-dir ./work_dirs/sarm_dense_only/sarm_viz
```

This post-training inference step also mirrors LeRobot's SARM prediction
visualization: each selected episode is rendered as a PNG with predicted
progress, stage probabilities, optional ground-truth progress, and sampled
frames. Set `--num-visualizations 0` to skip PNG generation, or use
`--visualize-only` to generate the PNGs without rewriting the JSONL file.

Example minimal real-dataset inference command:

```bash
python scripts/infer_sarm_progress.py \
  --config configs/sarm/sarm_dense_only.py \
  --ckpt-path ./work_dirs/sarm_dense_only_your_dataset/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/sarm_dense_only_your_dataset/sarm_progress.jsonl \
  --head-mode dense \
  --batch-size 1 \
  --max-batches 1 \
  --num-visualizations 1 \
  --output-dir ./work_dirs/sarm_dense_only_your_dataset/sarm_viz \
  --cfg-options \
    model.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv2.1 \
    inference_dataset.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv2.1 \
    inference_dataset.video_keys="['observation.images.cam_high']"
```

## RA-BC Progress Weights

FluxVLA also includes the SARM side of RA-BC (Reward-Aligned Behavior
Cloning): use a trained SARM checkpoint to precompute per-frame progress and
write a `sarm_progress.parquet` file. The parquet contains `index`,
`dataset_index`, `episode_index`, `frame_index`, and `progress_sparse` and/or
`progress_dense` columns, matching the format used by LeRobot's SARM RA-BC
weighting.

```bash
python scripts/compute_sarm_rabc_progress.py \
  --config configs/sarm/sarm_dense_only.py \
  --ckpt-path ./work_dirs/sarm_dense_only/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/sarm_dense_only/sarm_progress.parquet \
  --head-mode dense \
  --batch-size 1 \
  --stride 1
```

`--stride N` computes every Nth frame and linearly interpolates the skipped
frames, which is useful for larger datasets. The script scores the center
frame of each SARM observation window by default so the stored progress aligns
with the parquet `index` column. Use `--frame-index` only if you intentionally
need a different frame from the SARM sequence.

For downstream policy code that supports per-sample losses, the reusable
weight helper is available at `tools.sarm_rabc.SarmRABCWeights`. It expects
batches to contain a global `index` or `current_index` field and computes the
RA-BC delta weight from `progress[t + chunk_size] - progress[t]`.

## RA-BC Policy Training

Policy training can consume the SARM progress file through the generic
`sample_weight` path:

1. Set `expose_index=True` on the parquet dataset so transforms can see the
   same global frame index used in `sarm_progress.parquet`.
2. Insert `AttachRABCWeight` before `ProcessParquetInputs`.
3. Add `sample_weight` to the `DictCollator.keys` list.

```python
rabc = dict(
    enabled=True,
    weight_key='sample_weight',
    weighter=dict(
        type='SARMProgressWeighter',
        progress_path='./work_dirs/sarm_dense_only/sarm_progress.parquet',
        chunk_size=50,
        head_mode='dense',
    ),
)

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            type='ParquetDataset',
            expose_index=True,
            transforms=[
                dict(type='AttachRABCWeight', weighter=rabc['weighter']),
                dict(type='ProcessParquetInputs', ...),
                ...
            ],
        ),
    ),
)

runner = dict(
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'images', 'img_masks', 'lang_tokens', 'lang_masks',
            'actions', 'action_masks', 'sample_weight',
        ],
    ),
)
```

The loss weighting itself is implemented in
`fluxvla.engines.losses.reduce_action_bc_loss` and is currently wired into
PI0/PI05 flow matching, SmolVLA flow matching, shared flow-matching heads,
LLaVA action heads, OpenVLA token CE, and DreamZero's action/dynamics loss
path. If a batch does not include `sample_weight`, these models keep the
previous unweighted loss behavior.
