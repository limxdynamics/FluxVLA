#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/cpfs/users/yanis/FluxVLA

WORK_DIR="${WORK_DIR:-work_dirs/xvla_libero_spatial_hdf5_8gpu_bs16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

mkdir -p "${WORK_DIR}"

export PYTHONPATH=/mnt/data/cpfs/users/yanis/FluxVLA:${PYTHONPATH:-}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
/root/miniconda3/bin/conda run --no-capture-output -n fluxvla \
  torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  scripts/train.py \
  --config configs/xvla/xvla_libero_spatial_hdf5.py \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    runner.max_steps="${MAX_STEPS}" \
    train_dataloader.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
    runner.save_iter_interval="${SAVE_INTERVAL}" \
  2>&1 | tee "${WORK_DIR}/train.stdout.log"
