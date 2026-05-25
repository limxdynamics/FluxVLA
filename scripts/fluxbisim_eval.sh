#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preload system libffi.so.7 to fix ROS cv_bridge vs conda libffi conflict.
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}"

case "$(basename "$1")" in
    *pick_place_banana*)  TASK_ID=1 ;;
    *close_box*)          TASK_ID=2 ;;
    *pull_push_drawer*)   TASK_ID=3 ;;
    *screw_pitcher_lid*)  TASK_ID=4 ;;
    *handover_book*)      TASK_ID=5 ;;
esac

echo "${TASK_ID}" | python3 -u "${SCRIPT_DIR}/inference_real_robot.py" \
    --config "$1" --ckpt-path "$2" "${@:3}"
