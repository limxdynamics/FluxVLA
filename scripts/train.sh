#!/bin/bash

CONFIG=${1:-"configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py"}
WORK_DIR=${2:-"work_dirs/gr00t_eagle_3b_libero_10_full_finetune"}

torchrun \
  --nproc-per-node="${MLP_WORKER_GPU}" \
  --nnodes="${MLP_WORKER_NUM}" \
  --node_rank="${MLP_ROLE_INDEX}" \
  --master_addr="${MLP_WORKER_0_HOST}" \
  --master_port="${MLP_WORKER_0_PORT}" \
  "scripts/train.py" \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  ${@:3}
