# Remote inference config for UR3 (ZMQ)
#
# GPU server side:
#   python -m fluxvla.engines.runners.serving.serve \
#       --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
#       --ckpt-path /path/to/checkpoint.pt \
#       --host 0.0.0.0 --port 3333
#
# SSH tunnel (if server not on LAN):
#   ssh -p <SSH_PORT> -i <KEY> \
#       -L 5555:127.0.0.1:3333 user@server -N
#
# Robot side (no GPU needed):
#   python scripts/inference.py \
#       --config configs/pi05/pi05_paligemma_ur3_remote_inference.py

inference = dict(
    type='URInferenceRunner',
    remote_inference=dict(
        server_host='127.0.0.1',
        server_port=5555,
        timeout_s=30.0,
        serializer='msgpack',
        compress=True,
        enable_profiling=True,
    ),
    seed=7,
    action_chunk=10,
    publish_rate=30,
    max_publish_step=10000,
    task_suite_name='private',
    state_dim=7,
)
