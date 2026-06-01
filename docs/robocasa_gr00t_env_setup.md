# RoboCasa GR00T Environment Setup

This note explains how to prepare the extra dependencies, assets, datasets, and
checkpoints required by the GR00T RoboCasa training and evaluation config:

```bash
configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py
```

The config follows the GR00T N1.5 RoboCasa GR1 protocol, including GR1
state/action ordering, sin/cos state encoding, action horizon 16, random crop,
resize, color jitter, and `[-1, 1]` image normalization.

## What Must Be Downloaded

Large files are not stored in this Git repository. Download the required files
from Hugging Face or from the upstream project repositories before training or
evaluation.

Required files:

- Base GR00T checkpoint:
  `checkpoints/GR00T-N1.5-3B`
- Converted RoboCasa LeRobot dataset:
  `datasets/robocasa_fluxvla`
- RoboCasa tabletop assets used by the simulator.
- Dataset statistics used for evaluation:
  `work_dirs/official_groot_gr1_dataset_statistics.json`
- Optional finetuned GR00T RoboCasa checkpoints under `work_dirs/...`.

The commands below download files directly into the paths expected by the
config. Symlinks are not required for normal users. They are only a convenience
when the same large files already live on another local disk or shared storage.

Example download commands:

```bash
mkdir -p checkpoints datasets work_dirs

# Base GR00T checkpoint.
huggingface-cli download nvidia/GR00T-N1.5-3B \
  --local-dir checkpoints/GR00T-N1.5-3B

# Converted RoboCasa LeRobot dataset.
huggingface-cli download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include "robocasa_lerobot_V2.1/*" \
  --local-dir datasets/robocasa_fluxvla

# Official RoboCasa GR1 statistics for normalization / denormalization.
huggingface-cli download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include "official_groot_gr1_dataset_statistics.json" \
  --local-dir work_dirs

# Optional finetuned GR00T RoboCasa checkpoint.
huggingface-cli download limxdynamics/FluxVLAEngine \
  --include "gr00t_eagle_3b_robocasa_finetune/*" \
  --local-dir work_dirs/gr00t_eagle_3b_robocasa_finetune
```

If the RoboCasa tabletop assets are mirrored to Hugging Face, download them into
the RoboCasa GR1 task checkout. Otherwise, use the upstream asset downloader
shown below.

```bash
cd /path/to/robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y
```

## Fresh Installation

Use this path if you have not installed FluxVLA before.

1. Install the base FluxVLA environment by following the main README
   installation steps: create the conda environment, install PyTorch, install
   FlashAttention, install `av`, then run:

```bash
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

2. Install the additional RoboCasa runtime packages:

```bash
pip install gymnasium lxml
```

3. Install the patched robosuite build from
   [yinchimaoliang's repositories](https://github.com/yinchimaoliang?tab=repositories):

```bash
pip install git+https://github.com/yinchimaoliang/robosuite.git@7264a82
```

4. Install Isaac-GR00T and the RoboCasa GR1 task package from local checkouts:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git /path/to/Isaac-GR00T
cd /path/to/Isaac-GR00T
git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
pip install --no-deps -e /path/to/Isaac-GR00T

git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git \
  /path/to/robocasa-gr1-tabletop-tasks
cd /path/to/robocasa-gr1-tabletop-tasks
git checkout 4840e671596f93ca03651524b9f72ffb1aadfeff
pip install --no-deps -e /path/to/robocasa-gr1-tabletop-tasks
```

`--no-deps` is recommended for editable installs so the RoboCasa packages do
not replace the pinned FluxVLA model stack dependencies.

## Upgrade an Existing FluxVLA Environment

Use this path if you already have a working FluxVLA environment and only need
to add RoboCasa GR00T support.

1. Update your FluxVLA checkout to a commit that contains the RoboCasa GR00T
   config and runner.

2. Reinstall FluxVLA from the updated checkout:

```bash
pip install --no-deps --no-build-isolation -e .
```

3. Install the RoboCasa-specific packages:

```bash
pip install gymnasium lxml
pip install git+https://github.com/yinchimaoliang/robosuite.git@7264a82
```

4. Install Isaac-GR00T and `robocasa-gr1-tabletop-tasks` with `--no-deps`, as
   shown in the fresh installation section.

The patched robosuite build is needed because RoboCasa uses robosuite 1.5-style
APIs, while older stacks such as LIBERO still use some robosuite 1.4 import
paths and controller helpers. The patched commit restores compatibility shims
such as:

- `robosuite.load_controller_config(...)`
- `robosuite.environments.manipulation.single_arm_env.SingleArmEnv`
- `robosuite.robots.single_arm.SingleArm`
- `robosuite.models.robots.PandaOmron`

## Sanity Checks

Check robosuite compatibility:

```bash
python - <<'PY'
import robosuite

assert hasattr(robosuite, "load_controller_config")

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.robots import PandaOmron
from robosuite.robots.single_arm import SingleArm

print("robosuite compatibility imports OK")
print("SingleArmEnv:", SingleArmEnv)
print("SingleArm:", SingleArm)
print("PandaOmron:", PandaOmron)
PY
```

Check RoboCasa registration and reset:

```bash
python - <<'PY'
import gymnasium as gym
import robocasa  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

env_id = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
print("registered?", env_id in gym.envs.registry)

env = gym.make(env_id)
obs, info = env.reset()
print("env reset OK:", env_id, len(obs), "obs keys")
print(sorted(obs.keys())[:10])
env.close()
PY
```

If the second check fails because assets or datasets are unavailable, verify the
RoboCasa asset paths first. The Python imports should still pass.

## Training

Run a short smoke training job:

```bash
cd /path/to/FluxVLA

torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --work-dir work_dirs/smoke_groot_robocasa_train \
  --cfg-options \
    runner.type=FSDPTrainRunner \
    runner.sharding_strategy=no-shard \
    train_dataloader.per_device_batch_size=1 \
    runner.enable_gradient_checkpointing=False \
    runner.learning_rate=3e-5 \
    runner.weight_decay=1e-5 \
    runner.warmup_ratio=0.05 \
    runner.lr_scheduler_type=linear-warmup+cosine-decay \
    runner.max_epochs=None \
    runner.max_steps=2 \
    runner.save_iter_interval=1 \
    runner.max_keep_ckpts=2 \
    "runner.metric.active_trackers=('jsonl',)"
```

For full training, increase `--nproc-per-node`, `runner.max_steps`, and
`train_dataloader.per_device_batch_size` according to your GPU resources.

## Evaluation

Evaluate a finetuned GR00T RoboCasa checkpoint:

```bash
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --ckpt-path work_dirs/gr00t_eagle_3b_robocasa_finetune/checkpoints/step-060000.safetensors \
  --cfg-options \
    eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
    eval.output_dir=work_dirs/gr00t_eagle_3b_robocasa_finetune/eval_step060000 \
    eval.num_trials_per_task=20
```

Evaluate the base GR00T checkpoint on one task once:

```bash
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --ckpt-path checkpoints/GR00T-N1.5-3B \
  --cfg-options \
    eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
    eval.output_dir=work_dirs/smoke_eval_base \
    eval.task_list="['gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env']" \
    eval.num_trials_per_task=1 \
    eval.save_video=False
```

Prefer passing `eval.norm_stats_path=...` explicitly so the exact statistics
file used for evaluation is visible in the command history.
