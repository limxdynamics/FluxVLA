#!/bin/bash
# GR00T RoboCasa finetuning launcher.
#
# Configure by environment variables, for example:
#   export ROBOCASA_DATASET_ROOT=/path/to/robocasa_lerobot_V2.1
#   export WORK_DIR=work_dirs/groot_robocasa_full
#   export PER_DEVICE_BS=16
#   export LEARNING_RATE=3e-5
#   export MAX_STEPS=60000
#   bash scripts/train_groot_robocasa.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py}"
WORK_DIR="${WORK_DIR:-work_dirs/groot_robocasa_finetune}"

if [[ "$CONFIG" == /configs/* ]]; then
    CONFIG="configs/${CONFIG#/configs/}"
    echo "[train_groot_robocasa][fix] CONFIG normalized to: $CONFIG"
fi

if [[ "$CONFIG" != /* && ! -f "$REPO/$CONFIG" ]]; then
    echo "[train_groot_robocasa][ERROR] config not found: $REPO/$CONFIG" >&2
    exit 1
fi

cd "$REPO"

if [[ -n "${ROBOCASA_DATASET_ROOT:-}" ]]; then
    if [[ ! -d "$ROBOCASA_DATASET_ROOT" ]]; then
        echo "[train_groot_robocasa][ERROR] ROBOCASA_DATASET_ROOT not found: $ROBOCASA_DATASET_ROOT" >&2
        exit 1
    fi
    mkdir -p datasets
    ln -sfnT "$ROBOCASA_DATASET_ROOT" datasets/robocasa_fluxvla
    echo "[train_groot_robocasa] datasets/robocasa_fluxvla -> $(readlink -f datasets/robocasa_fluxvla)"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${MLP_WORKER_GPU:-1}}"
WORLD_SIZE="${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}"
RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
export NPROC_PER_NODE WORLD_SIZE RANK MASTER_ADDR MASTER_PORT

PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
SHARDING_STRATEGY="${SHARDING_STRATEGY:-no-shard}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-False}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear-warmup+cosine-decay}"
MAX_STEPS="${MAX_STEPS:-60000}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
SAVE_ITER_INTERVAL="${SAVE_ITER_INTERVAL:-5000}"
SAVE_EPOCH_INTERVAL="${SAVE_EPOCH_INTERVAL:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-20}"
ACTIVE_TRACKERS="${ACTIVE_TRACKERS:-('jsonl','wandb')}"

GLOBAL_BS=$((PER_DEVICE_BS * NPROC_PER_NODE * WORLD_SIZE))
echo "[train_groot_robocasa] CONFIG=$CONFIG"
echo "[train_groot_robocasa] WORK_DIR=$WORK_DIR"
echo "[train_groot_robocasa] NPROC_PER_NODE=$NPROC_PER_NODE WORLD_SIZE=$WORLD_SIZE RANK=$RANK"
echo "[train_groot_robocasa] PER_DEVICE_BS=$PER_DEVICE_BS GLOBAL_BS(no grad-accum)=$GLOBAL_BS"
echo "[train_groot_robocasa] LR=$LEARNING_RATE WARMUP_RATIO=$WARMUP_RATIO WEIGHT_DECAY=$WEIGHT_DECAY"
echo "[train_groot_robocasa] LR_SCHEDULER_TYPE=$LR_SCHEDULER_TYPE MAX_STEPS=${MAX_STEPS:-<empty>} MAX_EPOCHS=${MAX_EPOCHS:-<empty>}"

CFG_OPTIONS=(
    runner.type=FSDPTrainRunner
    runner.sharding_strategy="$SHARDING_STRATEGY"
    train_dataloader.per_device_batch_size="$PER_DEVICE_BS"
    runner.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING"
    runner.learning_rate="$LEARNING_RATE"
    runner.weight_decay="$WEIGHT_DECAY"
    runner.warmup_ratio="$WARMUP_RATIO"
    runner.lr_scheduler_type="$LR_SCHEDULER_TYPE"
    runner.save_iter_interval="$SAVE_ITER_INTERVAL"
    runner.save_epoch_interval="$SAVE_EPOCH_INTERVAL"
    runner.max_keep_ckpts="$MAX_KEEP_CKPTS"
    "runner.metric.active_trackers=$ACTIVE_TRACKERS"
)

if [[ -n "$MAX_STEPS" ]]; then
    CFG_OPTIONS+=(runner.max_epochs=None)
    CFG_OPTIONS+=(runner.max_steps="$MAX_STEPS")
else
    MAX_EPOCHS="${MAX_EPOCHS:-8}"
    CFG_OPTIONS+=(runner.max_steps=None)
    CFG_OPTIONS+=(runner.max_epochs="$MAX_EPOCHS")
fi

exec bash scripts/train.sh \
    "$CONFIG" \
    "$WORK_DIR" \
    --cfg-options \
        "${CFG_OPTIONS[@]}"
