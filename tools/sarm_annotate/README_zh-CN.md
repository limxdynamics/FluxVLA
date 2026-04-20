# SARM Subtask 标注工具

一套工具用于生成与维护 FluxVLA SARM 模型（以及官方 `lerobot.policies.sarm`）读取的 `sparse_*` / `dense_*` subtask 列。全部结果都直接写入标准 LeRobot episodes 元信息。

两条互补的入口：

1. **基于 VLM 的自动标注流水线**（`subtask_annotation.py` 等）——从 HuggingFace LeRobot fork [`lerobot/data_processing/sarm_annotations/`](https://github.com/huggingface/lerobot) 移植，使用本地 Qwen3-VL 对 episode 视频自动打标。
2. **人工/脚本标注入口**（`write_manual_stages.py`）——把你已经给出的 stage 边界（或最简单的单段 `"task"`）直接写入 episodes 元信息。不需要 GPU 或 VLM。

两种路线写入的列是完全一致的。

## 写入的内容

在标准 LeRobot v2.1（`meta/episodes.jsonl`）或 v3.x（`meta/episodes/*.parquet`）数据集上追加这些列：

- `sparse_subtask_names`、`sparse_subtask_start_frames`、`sparse_subtask_end_frames`
- `dense_subtask_names`、`dense_subtask_start_frames`、`dense_subtask_end_frames`
- 可选：`*_start_times`、`*_end_times`
- `meta/temporal_proportions_{sparse,dense}.json`

这些正是 [`fluxvla/datasets/utils/sarm_utils.py`](../../fluxvla/datasets/utils/sarm_utils.py) 所读取、[`fluxvla/datasets/sarm_dataset.py`](../../fluxvla/datasets/sarm_dataset.py) 所使用的列。

## 人工标注 —— `write_manual_stages.py`

如果 stage 边界来自人工标注、任务定义或启发式脚本，直接用这个脚本写入即可。同时支持 v2.1 和 v3.x，并会基于写入的数据重新计算 `temporal_proportions_{sparse,dense}.json`。

### 最简：整集单段 `"task"`

在不填任何帧号的情况下为 `single_stage` SARM 训练做初始化：

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
    --default-sparse auto
```

匹配 `configs/sarm/sarm_single_stage_*.py`。

### 仅 dense（sparse 用 auto 兜底）

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
    --spec specs/my_dense_stages.json \
    --default-sparse auto
```

匹配 `configs/sarm/sarm_dense_only_*.py`。

### 双标注：per-episode 多段 sparse + dense

```bash
python tools/sarm_annotate/write_manual_stages.py \
    --dataset-root /path/to/dataset \
    --spec specs/my_dual_stages.json
```

匹配 `configs/sarm/sarm_dual_*.py`。

### Spec 格式

`.json` 列表或 `.jsonl`，每个 episode 一个条目。帧号为 **闭区间、0-based**：

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

  // 简写：整集一个 "task" stage
  { "episode_index": 1, "sparse": "auto", "dense": "auto" }
]
```

未给出的字段 / 未在 spec 中列出的 episode 保持不变，除非你通过 `--default-sparse` / `--default-dense` 指定了 fallback。

### 常用参数

- `--spec <file>`：per-episode stage spec（`.json` 列表或 `.jsonl`）。
- `--default-sparse auto` / `--default-dense auto`：为未在 spec 中出现的 episode 提供兜底（单段 `"task"`）。
- `--skip-proportions`：不要（重新）写 `temporal_proportions_*.json`。

### 检视 / 校验 / 重置

- `parse_sparse_episode_info.py --dataset-root <root>`：按 episode 导出 `sparse_episode_info.json`，内容包括 `num_sparse_stages` 与自动计算的 temporal proportions。仅支持 v3.x。
- `parse_dense_episode_info.py --dataset-root <root>`：同上，针对 dense。仅支持 v3.x。
- `fix_sparse_annotations.py --dataset-root <root>`：强制把 sparse 设为单段 `"task"`。仅支持 v3.x；v2.1 请改用 `write_manual_stages.py --default-sparse auto`。
- `clear_written_annotations.py --dataset-root <root> [--apply]`：默认 dry-run，加 `--apply` 后真正清空所有 `sparse_*` / `dense_*` / `subtask_*` 列以及 `temporal_proportions_*.json`。仅支持 v3.x。

## VLM 自动标注 —— `subtask_annotation.py`

使用本地 Qwen3-VL 对 episode 视频自动生成标注。支持三种模式：

| 模式           | CLI 用法                                                     | 对应 FluxVLA 配置                         |
|----------------|--------------------------------------------------------------|-------------------------------------------|
| `single_stage` | 不传 `--sparse-subtasks` / `--dense-subtasks`                | `configs/sarm/sarm_single_stage_*.py`     |
| `dense_only`   | `--dense-only --dense-subtasks "Do A, Do B, Do C"`           | `configs/sarm/sarm_dense_only_*.py`       |
| `dual`         | `--sparse-subtasks "..." --dense-subtasks "..."`             | `configs/sarm/sarm_dual_*.py`             |

### 依赖（仅 VLM 路线需要）

```bash
pip install "lerobot>=0.3.4" qwen-vl-utils transformers torch opencv-python pydantic
```

`VideoAnnotator` 需要足够显存的 GPU（30B MoE 版本建议 ≥16 GB）。人工标注路线不需要这些依赖。

### 示例

```bash
# 本地数据集，dense-only
python tools/sarm_annotate/subtask_annotation.py \
    --repo-id /path/to/your/lerobot_dataset \
    --video-key observation.images.image \
    --dense-only \
    --dense-subtasks "Move to object, Grasp object, Move to target, Place object"
```

```bash
# HuggingFace Hub 上的数据集，dual（sparse + dense）
python tools/sarm_annotate/subtask_annotation.py \
    --repo-id your-username/your-dataset \
    --video-key observation.images.base \
    --sparse-subtasks "Pick up the cup, Place it on the plate" \
    --dense-subtasks "Reach, Grasp, Lift, Transport, Release" \
    --push-to-hub
```

### 按 episode 并行

`run_vlm_dense_subset.py` 封装 `subtask_annotation.py`，用若干 worker 进程并行处理 episode 子集，单 GPU 面对大数据集时比较有用。

### 标注后再处理

`subtask_annotation_timing.py` 可以从已标注的数据集重新推导 timing 列（比如平滑 subtask 边界、转换时间戳）。

## FluxVLA 如何消费这些标注

任何一条路线跑完后，数据集都直接匹配 SARM 配置：

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

不需要任何 FluxVLA 专用标注文件（如已弃用的 `sarm_*_annotations.jsonl`）。所有标注都位于标准 LeRobot episodes 元信息里，和 `lerobot.policies.sarm` 的数据源完全相同。

## 出处与来源

| 本地文件                          | 上游                                                                     | 改动 |
|-----------------------------------|--------------------------------------------------------------------------|------|
| `subtask_annotation.py`           | `lerobot/data_processing/sarm_annotations/subtask_annotation.py`         | 仅调整 sibling 模块导入路径。|
| `subtask_annotation_timing.py`    | `lerobot/data_processing/sarm_annotations/subtask_annotation_timing.py`  | 仅调整导入路径。|
| `run_vlm_dense_subset.py`         | `lerobot/scripts/run_vlm_dense_subset.py`                                | 仅调整导入路径。|
| `parse_sparse_episode_info.py`    | `lerobot/parse_sparse_episode_info.py`                                   | 原样移植。|
| `parse_dense_episode_info.py`     | `lerobot/parse_dense_episode_info.py`                                    | 原样移植。|
| `fix_sparse_annotations.py`       | `lerobot/fix_sparse_annotations.py`                                      | 原样移植。|
| `clear_written_annotations.py`    | `lerobot/clear_written_annotations.py`                                   | 原样移植。|
| `write_manual_stages.py`          | FluxVLA 原生                                                              | 从 JSON/JSONL spec 同时支持 v2.1 与 v3.x 写入。|

所有上游文件都保留了原始的 Apache 2.0 HuggingFace 版权头。
