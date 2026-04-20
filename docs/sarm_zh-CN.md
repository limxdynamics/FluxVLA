# SARM

FluxVLA 对 [SARM（Stage-Aware Reward Modeling，阶段感知奖励建模）](https://github.com/xdofai/opensarm)的集成，面向长时序机器人操作任务。

> **论文**：[SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](https://arxiv.org/abs/2509.25358)
> **原始仓库**：[https://github.com/xdofai/opensarm](https://github.com/xdofai/opensarm)

## SARM Checkpoint 管理

所有 SARM 相关工作流所需的模型都应集中放在 `./checkpoints/` 下，配置文件通过相对路径引用。

推荐的本地目录结构：

```text
checkpoints/
├── Qwen3-VL-32B-Instruct
├── clip-vit-base-patch32
├── sarm_dense_smoke
└── sarm_dense_only_flux_smoke.pt
```

约定名称：

- `./checkpoints/Qwen3-VL-32B-Instruct`：外部 SARM 标注流程使用的 VLM。
- `./checkpoints/clip-vit-base-patch32`：FluxVLA SARM 配置使用的 CLIP 主干与 tokenizer。
- `./checkpoints/sarm_dense_smoke`：冒烟验证得到的 SARM 检查点目录。
- `./checkpoints/sarm_dense_only_flux_smoke.pt`：本地冒烟训练产出的 FluxVLA 原生 SARM 检查点文件。

## SARM 使用方式

FluxVLA 的 SARM 配置统一使用 `./checkpoints` 下的相对路径，与项目其余部分一致。

当前 SARM 配置：

- `configs/sarm/sarm_single_stage_libero_10.py`
- `configs/sarm/sarm_dense_only_libero_10.py`
- `configs/sarm/sarm_dual_libero_10.py`

这些配置要求：

- `./checkpoints/clip-vit-base-patch32` 下已存在 CLIP 主干与 tokenizer。
- 在所选 `data_root_path` 下存在数据集元信息。

注意事项：

- SARM 标注直接从标准 LeRobot 的 episodes 元信息读取（v2.1 为 `meta/episodes.jsonl`，v3.x 为 `meta/episodes/*.parquet`），列名形如 `sparse_subtask_names` / `sparse_subtask_start_frames` / `sparse_subtask_end_frames` 及其对应的 `dense_*` 列。这和官方 `lerobot.policies.sarm` 使用的数据源一致。为兼容老的 sparse-only 数据集，sparse 还会回退到不带前缀的列名（`subtask_names`、`subtask_start_frames`、`subtask_end_frames`）。FluxVLA 不再读取独立的 `sarm_*_annotations.jsonl` 文件。
- 用于标注的外部 VLM 同样应放在 `./checkpoints` 下，例如 `./checkpoints/Qwen3-VL-32B-Instruct`。
- `./checkpoints/sarm_dense_smoke` 是来自外部 SARM 工作流的冒烟测试模型目录，已保留。
- FluxVLA 的 `scripts/infer_sarm_progress.py` 需要使用形如 `./checkpoints/sarm_dense_only_flux_smoke.pt` 的 FluxVLA 训练检查点文件。
- `scripts/infer_sarm_progress.py` 支持通过 `--cfg-options` 覆盖数据集配置，以及通过 `--max-batches` 做快速冒烟验证。
- 对于 LeRobot v2.1/v3.x 格式的数据集，`task` 字段可以是任务索引，FluxVLA 读取时会从 `tasks.jsonl` 或 `tasks.parquet` 动态还原为任务文本，不需要修改数据集文件。
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
    --dataset-root /path/to/dataset \
    --default-sparse auto
```

**b）按 per-episode JSON spec 写多段 sparse + dense**：

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
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

**c）只有 dense 标注，sparse 用单段 auto 兜底**（匹配 `configs/sarm/sarm_dense_only_*.py`）：

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
    --spec /path/to/my_dense_stages.json \
    --default-sparse auto
```

### 路线 2：VLM 自动标注 —— `subtask_annotation.py`

在本机跑 Qwen3-VL，对 episode 视频自动生成相同的列并写回。需要 GPU（30B MoE 版本建议 ≥16 GB 显存）和 `pip install qwen-vl-utils transformers`。支持三种模式：

| 模式           | CLI 用法                                          | 对应配置                              |
| -------------- | ------------------------------------------------- | ------------------------------------- |
| `single_stage` | 不传 `--sparse-subtasks` / `--dense-subtasks`     | `configs/sarm/sarm_single_stage_*.py` |
| `dense_only`   | `--dense-only --dense-subtasks "Do A, Do B, ..."` | `configs/sarm/sarm_dense_only_*.py`   |
| `dual`         | `--sparse-subtasks "..." --dense-subtasks "..."`  | `configs/sarm/sarm_dual_*.py`         |

```bash
python tools/sarm_annotate/subtask_annotation.py \
  --repo-id /path/to/your/lerobot_dataset \
  --video-key observation.images.image \
  --dense-only \
  --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

对于大数据集，`run_vlm_dense_subset.py` 可以按 episode 并行启动多个 worker 进程加速标注。

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
  --config configs/sarm/sarm_dense_only_libero_10.py \
  --work-dir ./work_dirs/sarm_dense_only_libero_10 \
  --cfg-options train_dataloader.per_device_batch_size=1
```

真实数据集冒烟训练示例：

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

把生成的 FluxVLA 检查点软链到 `./checkpoints`：

```bash
ln -sfn \
  "$PWD/tmp/sarm_dense_only_flux_smoke/checkpoints/latest-checkpoint.pt" \
  "$PWD/checkpoints/sarm_dense_only_flux_smoke.pt"
```

## 推理

离线 progress 推理示例：

```bash
python scripts/infer_sarm_progress.py \
  --config configs/sarm/sarm_dense_only_libero_10.py \
  --ckpt-path ./checkpoints/sarm_dense_only_flux_smoke.pt \
  --output-path ./work_dirs/sarm_progress.jsonl \
  --head-mode dense \
  --batch-size 1
```

真实数据集冒烟推理示例：

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
