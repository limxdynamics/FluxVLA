# SARM

FluxVLA 对 [SARM（Stage-Aware Reward Modeling，阶段感知奖励建模）](https://github.com/xdofai/opensarm)的集成，面向长时序机器人操作任务。

> **论文**：[SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](https://arxiv.org/abs/2509.25358)
> **原始仓库**：[https://github.com/xdofai/opensarm](https://github.com/xdofai/opensarm)

## SARM Checkpoint 管理

所有 SARM 相关工作流所需的模型都应集中放在 `./checkpoints/` 下，配置文件通过相对路径引用。

训练产出的 FluxVLA 检查点则保存在训练输出目录下，默认使用你传给 `scripts/train.py --work-dir` 的目录，例如 `./work_dirs/<run_name>/checkpoints/latest-checkpoint.pt`。

推荐的本地目录结构：

```text
checkpoints/
├── Qwen3-VL-30B-A3B-Instruct
└── clip-vit-base-patch32
```

约定名称：

- `./checkpoints/Qwen3-VL-30B-A3B-Instruct`：外部 SARM 标注流程使用的 VLM。
- `./checkpoints/clip-vit-base-patch32`：FluxVLA SARM 配置使用的 CLIP 主干与 tokenizer。
- 训练生成的 FluxVLA checkpoint 默认放在 `--work-dir/checkpoints/` 下，而不是 `./checkpoints/` 根目录。

## SARM 使用方式

FluxVLA 的 SARM 配置统一使用 `./checkpoints` 下的相对路径，与项目其余部分一致。

当前 SARM 配置：

- `configs/sarm/sarm_single_stage.py`
- `configs/sarm/sarm_dense_only.py`
- `configs/sarm/sarm_dual.py`

这些文件现在默认指向已经发布到 Hugging Face 的 manual SARM 示例数据集 `./datasets/SARM_manual_test_10Episodes_lerobotv3.0`，并使用与该数据集匹配的 `observation.images.cam_high` 作为默认相机键。如果你把数据下载到了其他路径，或者想切换到其他相机流，可以通过 `--cfg-options` 覆盖 `data_root_path` 与 `video_keys`。

已发布的 Hugging Face 数据集：

- 训练 / 推理用的完整人工标注数据：[`limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv3.0)
- 供手工或 VLM 继续标注的无标注数据：[`limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv3.0`](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv3.0)

可通过以下命令下载到 `./datasets`：

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
```

LeRobot v3.x 视频元信息自检：

- LeRobot v3.x 既允许多个 episode 共用一个 MP4，也允许一个 episode 对应一个 MP4。
- 如果多个 episode 共用同一个 MP4，那么每个 episode 的 `from_timestamp` / `to_timestamp` 必须正确描述它在该视频中的片段区间。
- 如果视频本身已经拆成 `file-000.mp4`、`file-001.mp4` 这样的逐集文件，那么每个 episode 就应该指向各自的 `file_index`，且 `from_timestamp` 通常应回到 `0.0`。
- 如果目录里明明有多个 MP4，但所有 episode 仍都指向 `file-000.mp4`，那就是错误的 metadata，应先修正再训练或上传到 Hugging Face。

这些配置要求：

- `./checkpoints/clip-vit-base-patch32` 下已存在 CLIP 主干与 tokenizer。
- 在所选 `data_root_path` 下存在数据集元信息。

注意事项：

- SARM 标注直接从标准 LeRobot 的 episodes 元信息读取（v2.1 为 `meta/episodes.jsonl`，v3.x 为 `meta/episodes/*.parquet`），列名形如 `sparse_subtask_names` / `sparse_subtask_start_frames` / `sparse_subtask_end_frames` 及其对应的 `dense_*` 列。这和官方 `lerobot.policies.sarm` 使用的数据源一致。为兼容老的 sparse-only 数据集，sparse 还会回退到不带前缀的列名（`subtask_names`、`subtask_start_frames`、`subtask_end_frames`）。FluxVLA 不再读取独立的 `sarm_*_annotations.jsonl` 文件。
- 用于标注的外部 VLM 同样应放在 `./checkpoints` 下，例如 `./checkpoints/Qwen3-VL-30B-A3B-Instruct`。如果传入的是 Hugging Face 本地 cache 根目录而不是具体 snapshot，FluxVLA 会自动解析到对应的 `snapshots/*` 目录。
- FluxVLA 的 `scripts/infer_sarm_progress.py` 需要使用 FluxVLA 训练生成的 checkpoint 文件，通常直接指向 `--work-dir/checkpoints/latest-checkpoint.pt`。
- `scripts/infer_sarm_progress.py` 支持通过 `--cfg-options` 覆盖数据集配置，以及通过 `--max-batches` 做快速冒烟验证。
- 对于 LeRobot v2.1/v3.x 格式的数据集，`task` 字段可以是任务索引，FluxVLA 读取时会从 `tasks.jsonl` 或 `tasks.parquet` 动态还原为任务文本，不需要修改数据集文件。
- 对于 LeRobot v3.x 数据集，视频路径既可以来自 `videos/<key>/chunk_index` / `file_index`，也可以来自 `meta/episodes/*` 或数据行上的等价列；FluxVLA 会自动兼容这些字段名。
- 如果你的数据集使用的相机键不是 `observation.images.image`，可通过 `--cfg-options` 覆盖 `train_dataloader.dataset.video_keys` 与 `inference_dataset.video_keys`。

## 数据集标注

SARM 训练读取的是标准 LeRobot 每集元信息上的 subtask 标注（v2.1 的 `meta/episodes.jsonl`、v3.x 的 `meta/episodes/*.parquet` 上的 list 列，与 `lerobot.policies.sarm` 格式完全一致）。FluxVLA 在 [`tools/sarm_annotate/`](../tools/sarm_annotate/README_zh-CN.md) 下提供了两条互补的标注路线，根据数据情况选用其一即可。

### 列契约（FluxVLA 实际读取的字段）

每个 episode 应在元信息上带有以下 list 列：

- `sparse_subtask_names`、`sparse_subtask_start_frames`、`sparse_subtask_end_frames`
- `dense_subtask_names`、`dense_subtask_start_frames`、`dense_subtask_end_frames`
- 可选：`*_start_times`、`*_end_times`

帧号是 **闭区间、0-based**。sparse 还兼容不带前缀的老列名（`subtask_names`、`subtask_start_frames`、`subtask_end_frames`），以便老的 sparse-only 数据集能直接使用。

另外，`meta/temporal_proportions_sparse.json` / `meta/temporal_proportions_dense.json` 记录每个 subtask 名字在整个数据集上的平均时长占比。下面两种工具都会自动写入这两个文件。

### 路线 1：手动 stage（无需 GPU）—— `write_manual_stages.py`

当 stage 边界来自人工标注、任务定义或启发式脚本时使用；同时支持 v2.1 和 v3.x。

**a）整集只有一个 `"task"` stage**——最快的 `single_stage` SARM 启动方式，完全不用关心帧号：

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
    --default-sparse auto
```

**b）按 per-episode JSON spec 写多段 sparse + dense**：

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
    --spec /path/to/my_stages.json
```

其中 `my_stages.json` 形如：

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

`"auto"` 是简写，表示整集作为一个 `"task"` stage。未在 spec 中出现、且没有指定 `sparse`/`dense` 键的 episode 不会被修改，除非额外传入 `--default-sparse auto` / `--default-dense auto`。

**c）只有 dense 标注，sparse 用单段 auto 兜底**（匹配 `configs/sarm/sarm_dense_only.py`）：

```bash
python tools/sarm_annotate/write_manual_stages.py \
  --dataset-root ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
    --spec /path/to/my_dense_stages.json \
    --default-sparse auto
```

### 路线 2：VLM 自动标注 —— `subtask_annotation.py`

在本机跑 Qwen3-VL，对 episode 视频自动生成相同的列并写回。需要 GPU（30B MoE 版本建议 ≥16 GB 显存）和 `pip install qwen-vl-utils transformers`。支持三种模式：

| 模式           | CLI 用法                                          | 对应配置                            |
| -------------- | ------------------------------------------------- | ----------------------------------- |
| `single_stage` | 不传 `--sparse-subtasks` / `--dense-subtasks`     | `configs/sarm/sarm_single_stage.py` |
| `dense_only`   | `--dense-only --dense-subtasks "Do A, Do B, ..."` | `configs/sarm/sarm_dense_only.py`   |
| `dual`         | `--sparse-subtasks "..." --dense-subtasks "..."`  | `configs/sarm/sarm_dual.py`         |

```bash
python tools/sarm_annotate/subtask_annotation.py \
  --repo-id ./datasets/SARM_vlm_test_10Episodes_lerobotv3.0 \
  --model ./checkpoints/Qwen3-VL-30B-A3B-Instruct \
  --video-key observation.images.cam_high \
  --video-backend pyav \
  --dense-only \
  --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

对于大数据集，`run_vlm_dense_subset.py` 可以按 episode 并行启动多个 worker 进程加速标注。

如果当前环境里的默认视频解码后端不可用，可显式传入 `--video-backend pyav` 或 `--video-backend video_reader`；若不传，则保持 LeRobot 默认行为。

### 检视 / 校验 / 重置

- `tools/sarm_annotate/parse_sparse_episode_info.py`——按 episode 导出 `num_sparse_stages` 和 temporal proportions。
- `tools/sarm_annotate/parse_dense_episode_info.py`——同上，针对 dense。
- `tools/sarm_annotate/fix_sparse_annotations.py`——在 v3.x 数据集上强制写单段 `"task"` sparse。
- `tools/sarm_annotate/clear_written_annotations.py --dataset-root <root> [--apply]`——对 v3.x 数据集做 dry-run / 清除所有 SARM 列与 proportions 文件。

完整脚本清单及上游出处见 [`tools/sarm_annotate/README_zh-CN.md`](../tools/sarm_annotate/README_zh-CN.md)。

## 训练

示例训练命令：

```bash
export WANDB_MODE=disabled
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/sarm/sarm_dense_only.py \
  --work-dir ./work_dirs/sarm_dense_only \
  --cfg-options train_dataloader.per_device_batch_size=1
```

真实数据集最小训练示例：

```bash
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/sarm/sarm_dense_only.py \
  --work-dir ./work_dirs/sarm_dense_only_your_dataset \
  --cfg-options \
    model.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv3.0 \
    train_dataloader.dataset.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv3.0 \
    inference_dataset.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv3.0 \
    train_dataloader.dataset.video_keys="['observation.images.cam_high']" \
    inference_dataset.video_keys="['observation.images.cam_high']" \
    train_dataloader.per_device_batch_size=1 \
    train_dataloader.per_device_num_workers=0 \
    runner.max_steps=1 \
    runner.max_epochs=None \
    runner.save_iter_interval=1
```

## 推理

离线 progress 推理示例：

```bash
python scripts/infer_sarm_progress.py \
  --config configs/sarm/sarm_dense_only.py \
  --ckpt-path ./work_dirs/sarm_dense_only/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/sarm_dense_only/sarm_progress.jsonl \
  --head-mode dense \
  --batch-size 1
```

真实数据集最小推理示例：

```bash
python scripts/infer_sarm_progress.py \
  --config configs/sarm/sarm_dense_only.py \
  --ckpt-path ./work_dirs/sarm_dense_only_your_dataset/checkpoints/latest-checkpoint.pt \
  --output-path ./work_dirs/sarm_dense_only_your_dataset/sarm_progress.jsonl \
  --head-mode dense \
  --batch-size 1 \
  --max-batches 1 \
  --cfg-options \
    model.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv3.0 \
    inference_dataset.data_root_path=./datasets/SARM_manual_test_10Episodes_lerobotv3.0 \
    inference_dataset.video_keys="['observation.images.cam_high']"
```
