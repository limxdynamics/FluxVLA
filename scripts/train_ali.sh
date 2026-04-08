#!/bin/bash

CONFIG=${1:-"configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py"}
WORK_DIR=${2:-"work_dirs/gr00t_eagle_3b_libero_10_full_finetune"}

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "scripts/train.py" \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  ${@:3}
