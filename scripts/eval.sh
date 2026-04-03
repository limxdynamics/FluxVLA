#!/bin/bash

CONFIG=$1
CKPT_PATH=$2

torchrun \
  --nproc-per-node="${MLP_WORKER_GPU}" \
  --nnodes="${MLP_WORKER_NUM}" \
  --node_rank="${MLP_ROLE_INDEX}" \
  --master_addr="${MLP_WORKER_0_HOST}" \
  --master_port="${MLP_WORKER_0_PORT}" \
  "scripts/eval.py" \
  --config "${CONFIG}" \
  --ckpt-path "${CKPT_PATH}" \
  ${@:3}
