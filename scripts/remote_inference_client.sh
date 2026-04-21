#!/bin/bash
# Remote inference client launcher with optional SSH tunnel.
#
# Usage:
#   bash scripts/remote_inference_client.sh <CONFIG> [OPTIONS]
#
# Basic (tunnel already set up or on same LAN):
#   bash scripts/remote_inference_client.sh \
#       configs/pi05/pi05_paligemma_ur3_remote_inference.py
#
# With auto SSH tunnel:
#   bash scripts/remote_inference_client.sh \
#       configs/pi05/pi05_paligemma_ur3_remote_inference.py \
#       --ssh-host user@server.example.com \
#       --ssh-port 22 \
#       --ssh-key ~/.ssh/id_rsa \
#       --local-port 5555 \
#       --remote-port 3333

set -e

CONFIG=""
SSH_HOST=""
SSH_PORT="22"
SSH_KEY=""
LOCAL_PORT="5555"
REMOTE_PORT="3333"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ssh-host)     SSH_HOST="$2";    shift 2 ;;
        --ssh-port)     SSH_PORT="$2";    shift 2 ;;
        --ssh-key)      SSH_KEY="$2";     shift 2 ;;
        --local-port)   LOCAL_PORT="$2";  shift 2 ;;
        --remote-port)  REMOTE_PORT="$2"; shift 2 ;;
        --cfg-options)
            shift
            EXTRA_ARGS="--cfg-options"
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
            done
            ;;
        *)
            if [ -z "$CONFIG" ]; then
                CONFIG="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    cat <<'USAGE'
Usage: bash scripts/remote_inference_client.sh <CONFIG> [OPTIONS]

Options:
  --ssh-host USER@HOST    SSH destination (enables auto tunnel)
  --ssh-port PORT         SSH port (default: 22)
  --ssh-key PATH          SSH private key file
  --local-port PORT       Local tunnel port (default: 5555)
  --remote-port PORT      Remote ZMQ server port (default: 3333)
  --cfg-options K=V ...   Override config values

Examples:
  # LAN / tunnel already running
  bash scripts/remote_inference_client.sh \
      configs/pi05/pi05_paligemma_ur3_remote_inference.py

  # Auto SSH tunnel
  bash scripts/remote_inference_client.sh \
      configs/pi05/pi05_paligemma_ur3_remote_inference.py \
      --ssh-host user@server.example.com \
      --ssh-port 57705 \
      --ssh-key ~/.ssh/my_key \
      --local-port 5555 \
      --remote-port 3333
USAGE
    exit 1
fi

TUNNEL_PID=""
cleanup() {
    if [ -n "$TUNNEL_PID" ]; then
        echo "[client] Closing SSH tunnel (pid=$TUNNEL_PID)..."
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ -n "$SSH_HOST" ]; then
    SSH_CMD="ssh"
    [ -n "$SSH_KEY" ]  && SSH_CMD="$SSH_CMD -i $SSH_KEY"
    SSH_CMD="$SSH_CMD -p $SSH_PORT -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} $SSH_HOST -N"

    echo "[client] Starting SSH tunnel..."
    echo "  $SSH_CMD"
    $SSH_CMD &
    TUNNEL_PID=$!
    sleep 2

    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "[client] ERROR: SSH tunnel failed to start."
        exit 1
    fi
    echo "[client] SSH tunnel running (pid=$TUNNEL_PID)"
    echo "  local :${LOCAL_PORT} -> remote :${REMOTE_PORT}"
fi

echo "[client] Starting inference with config: $CONFIG"
python scripts/inference.py --config "$CONFIG" $EXTRA_ARGS
