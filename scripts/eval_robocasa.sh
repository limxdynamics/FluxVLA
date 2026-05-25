#!/bin/bash
# RoboCasa evaluation launcher.
#
# Example:
#   bash scripts/eval_robocasa.sh \
#     --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
#     --ckpt-path work_dirs/groot_robocasa/checkpoints/step-020000.safetensors \
#     --output-dir work_dirs/groot_robocasa/eval_step020000 \
#     --cfg-options \
#       eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
#       eval.num_trials_per_task=20

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    scripts/eval.py "$@"
