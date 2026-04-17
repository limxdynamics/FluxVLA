#!/bin/bash
# Unified launcher for single-node or multi-node distributed evaluation.
# Auto-detects common distributed environment variable conventions:
#   - Standard torchrun / ali-style: NPROC_PER_NODE, WORLD_SIZE, RANK,
#     MASTER_ADDR, MASTER_PORT
#   - Vol-platform: MLP_WORKER_GPU, MLP_WORKER_NUM, MLP_ROLE_INDEX,
#     MLP_WORKER_0_HOST, MLP_WORKER_0_PORT
# Falls back to a sensible single-node default when none are set.

CONFIG=$1
CKPT_PATH=$2

NPROC_PER_NODE="${NPROC_PER_NODE:-${MLP_WORKER_GPU:-1}}"
WORLD_SIZE="${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "scripts/eval.py" \
  --config "${CONFIG}" \
  --ckpt-path "${CKPT_PATH}" \
  ${@:3}
