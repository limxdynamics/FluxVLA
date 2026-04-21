# FluxVLA Serving

ZMQ-based VLA model serving for remote robot inference.

## Architecture

```
serving/
├── serve.py          # CLI entry point: load model + start server
├── zmq_server.py     # PolicyServer (ZMQ event loop) + create_server factory
├── serializers.py    # Wire-format encode/decode (shared by client & server)
└── proto/
    ├── vla_service.proto      # Protobuf schema
    └── vla_service_pb2.py     # Generated Python code
```

Two-layer design:

```
PolicyServer        Layer 1: generic ZMQ REP loop + endpoint routing
    |
create_server()     Layer 2: wires VLA model into predict_action handler
                    (deserialize obs -> preprocess -> inference ->
                     denormalize -> serialize action)
```

## Environment Setup

### Server side (GPU machine)

Requires GPU + CUDA. Install the full FluxVLA training environment.

### Client side (robot machine, no GPU needed)

Minimal Python dependencies:

```
pyzmq>=27.0
msgpack>=1.0
numpy
```

If using `serializer='protobuf'`, also install:

```
protobuf>=5.26
```

**Note:** `protobuf` version must be compatible with `wandb` if wandb is also
installed. If you hit import errors, either:

- Upgrade both: `pip install --upgrade protobuf wandb`
- Or use `serializer='msgpack'` (no protobuf dependency, recommended for
  robot-side environments with pinned packages)

## Quick Start

### Step 1: Start the server (GPU machine)

```bash
python -m fluxvla.engines.runners.serving.serve \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --host 0.0.0.0 --port 3333 \
    --device cuda:0 --dtype bf16
```

The server loads the model, dataset pipeline, and denormalization transform,
then starts a ZMQ REP loop waiting for requests.

### Step 2: Set up network connection

If the robot cannot reach the GPU server directly (e.g. server is behind NAT
or on a cloud machine), use an SSH tunnel to forward a local port to the
server's ZMQ port:

```bash
ssh -p <SSH_PORT> -i <KEY_FILE> \
    -L <LOCAL_PORT>:127.0.0.1:<SERVER_ZMQ_PORT> \
    <USER>@<SERVER_PUBLIC_IP> -N
```

Example: server runs ZMQ on port 3333, forward local 5555 to it:

```bash
ssh -p 57705 -i ~/.ssh/my_key \
    -L 5555:127.0.0.1:3333 \
    user@server.example.com -N
```

Explanation:

- `-L 5555:127.0.0.1:3333`: binds local port 5555, tunnels traffic to
  `127.0.0.1:3333` on the remote server (where ZMQ is listening)
- `-N`: no remote shell, just forwarding
- `-p 57705`: SSH port if non-standard (default is 22)

After this, the robot connects to `127.0.0.1:5555` and traffic is tunneled
to the GPU server's ZMQ port.

If the robot and GPU server are on the same LAN, skip the tunnel and point
`server_host` directly to the server's LAN IP.

### Step 3: Create client config

Add ``remote_inference`` to any existing runner config.  The runner type
stays the same (``URInferenceRunner``, ``AlohaInferenceRunner``, etc.):

```python
# configs/pi05/pi05_paligemma_ur3_remote_inference.py
inference = dict(
    type='URInferenceRunner',          # same runner as local inference
    remote_inference=dict(              # add this block to enable remote mode
        server_host='127.0.0.1',       # localhost if using SSH tunnel
        server_port=5555,
        timeout_s=30.0,
        serializer='msgpack',          # 'msgpack' (recommended) or 'protobuf'
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
```

For Aloha, just change ``type='AlohaInferenceRunner'``.

### Step 4: Run the client (robot)

```bash
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_ur3_remote_inference.py
```

This will:

1. Build a `URInferenceRunner` with `remote_inference` enabled
2. `run_setup()` connects to the ZMQ server and sends a ping health-check
3. `run()` enters the ROS inference loop:
   collect obs -> encode -> ZMQ send -> receive -> decode action -> execute

Note: `--ckpt-path` is **not needed** for remote inference (the server loads
the model).

### Runtime interaction

During inference, the terminal prompts for task selection:

- Enter a task ID (e.g. `1`, `2`, ...) to run a predefined task
- Enter `0` to reset the robot to its prepare pose (local ROS, no network)
- Enter the number of times to repeat the task

### Supported runners

Any runner that inherits from `BaseInferenceRunner` supports remote mode
by adding `remote_inference` to its config:

| Runner | Robot |
|--------|-------|
| `URInferenceRunner` | UR3/UR5 single-arm |
| `AlohaInferenceRunner` | Aloha dual-arm |
| `AlohaRTCInferenceRunner` | Aloha dual-arm + RTC |

### Client config options (remote_inference dict)

| Option | Default | Description |
|--------|---------|-------------|
| `server_host` | `localhost` | GPU server IP or hostname |
| `server_port` | `5555` | ZMQ TCP port |
| `timeout_s` | `30.0` | ZMQ send/recv timeout (seconds) |
| `serializer` | `msgpack` | Wire format: `msgpack` or `protobuf` |
| `compress` | `True` | JPEG-compress images before sending |
| `enable_profiling` | `True` | Print avg latency every 50 calls |

## CLI Arguments

| Argument        | Default     | Description                                            |
| --------------- | ----------- | ------------------------------------------------------ |
| `--config`      | (required)  | Path to mmengine config file                           |
| `--ckpt-path`   | (required)  | Path to model checkpoint (.pt / .safetensors)          |
| `--host`        | `0.0.0.0`   | Bind address (`0.0.0.0` = all interfaces)              |
| `--port`        | `5555`      | ZMQ TCP port                                           |
| `--device`      | `cuda:0`    | CUDA device                                            |
| `--dtype`       | `bf16`      | Mixed precision: `bf16` / `fp16` / `fp32`              |
| `--dataset-key` | auto-detect | Config key for dataset pipeline: `inference` or `eval` |

## Wire Formats

Two serialization formats are supported. The first byte of every ZMQ message
determines the format:

| Format   | First byte | Pros                                  | Cons                     |
| -------- | ---------- | ------------------------------------- | ------------------------ |
| msgpack  | != `0x01`  | Flexible, zero schema, Python-native  | Larger payload           |
| protobuf | `0x01`     | Compact, cross-language (C++/Rust/Go) | Requires `.proto` schema |

Both formats use the same payload encoding for observation data:

- RGB images: JPEG-compressed bytes (configurable via `compress` flag)
- Numeric arrays: numpy `.npy` format bytes
- Strings: passed directly

## Endpoints

| Endpoint         | Input            | Output                             | Description        |
| ---------------- | ---------------- | ---------------------------------- | ------------------ |
| `predict_action` | obs + unnorm_key | action bytes + infer_time          | Model inference    |
| `ping`           | (none)           | `{status: ok}`                     | Health check       |
| `reset`          | (none)           | `{status: ok}`                     | Reset server state |
| `get_status`     | (none)           | uptime, request count, avg latency | Server stats       |
| `kill`           | (none)           | `{status: ok}`                     | Graceful shutdown  |

## Data Flow

```
Client (robot)                    Server (GPU)
     |                                 |
     |  encode obs (JPEG + npy)        |
     |  ────── ZMQ REQ ──────────────> |
     |                                 |  deserialize obs
     |                                 |  dataset transform (preprocess)
     |                                 |  vla.predict_action (GPU)
     |                                 |  denormalize action
     |                                 |  serialize action (npy)
     |  <────── ZMQ REP ────────────── |
     |  decode action                  |
     |  execute on robot               |
```
