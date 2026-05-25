# RoboCasa GR00T Environment Setup

This note describes the extra packages needed to run the GR00T RoboCasa
training and evaluation configs on top of the FluxVLA v1.x environment.

The base FluxVLA environment should remain compatible with existing tasks such
as LIBERO and ALOHA. The RoboCasa stack adds a newer robosuite path, RoboCasa
task registration, and Isaac-GR00T utilities.

## Base Environment

Start from the standard FluxVLA v1.x environment, including the existing core
versions from `requirements.txt`, for example:

```bash
torch==2.6.0
torchvision==0.21.0
transformers==4.53.2
mmengine==0.10.7
mujoco==3.3.2
robosuite==1.4.1
```

The RoboCasa GR00T integration does not require changing the model-side
dependencies such as PyTorch, Transformers, or MMEngine.

## Additional Packages

Install the extra runtime packages:

```bash
pip install gymnasium lxml
```

Install the patched robosuite build:

```bash
pip install git+https://github.com/yinchimaoliang/robosuite.git@7264a82
```

Install Isaac-GR00T and the RoboCasa GR1 task package from local checkouts:

```bash
pip install --no-deps -e /path/to/Isaac-GR00T
pip install --no-deps -e /path/to/robocasa-gr1-tabletop-tasks
```

`--no-deps` is recommended for editable installs so these packages do not
replace the pinned FluxVLA model stack dependencies.

## Why Use the Patched Robosuite

RoboCasa depends on robosuite 1.5-style APIs, while older stacks such as LIBERO
still use some robosuite 1.4 import paths and controller helpers. The patched
robosuite commit `7264a82` restores these compatibility shims and keeps
RoboCasa robot models available:

- `robosuite.load_controller_config(...)`
- `robosuite.environments.manipulation.single_arm_env.SingleArmEnv`
- `robosuite.robots.single_arm.SingleArm`
- `robosuite.models.robots.PandaOmron`

This allows the same environment to run RoboCasa GR00T evaluation while keeping
legacy LIBERO imports available.

## Sanity Checks

Run these checks after installation:

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

Then verify RoboCasa registration:

```bash
python - <<'PY'
import gymnasium as gym
import robocasa  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

env_id = "PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
env = gym.make(env_id)
obs, info = env.reset()
print("env reset OK:", env_id, len(obs), "obs keys")
env.close()
PY
```

If the second check fails because assets or datasets are unavailable, verify
the RoboCasa asset paths first. The Python imports should still pass.

## GR00T RoboCasa Config

The integration provides one GR00T RoboCasa finetuning config:

```bash
configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py
```

The config includes the official Isaac-GR00T-style training image pipeline:
random crop, resize, color jitter, and `[-1, 1]` image normalization. It can be
used for either small subsets or full RoboCasa GR1 datasets by changing the
dataset symlink or `ROBOCASA_DATASET_ROOT`.

## Checkpoints, Data, and Assets

Large assets are not stored in this repository. Download or prepare them
separately, then symlink them into the repo.

The GR00T base checkpoint should be available at:

```bash
checkpoints/GR00T-N1.5-3B
```

The converted RoboCasa LeRobot dataset should be available under:

```bash
datasets/robocasa_fluxvla
```

You can create this as a symlink to your converted RoboCasa LeRobot dataset:

```bash
mkdir -p datasets
ln -sfnT /path/to/robocasa_lerobot_V2.1 datasets/robocasa_fluxvla
```

For evaluation, make sure RoboCasa assets and raw videos are available according
to the RoboCasa / GR00T dataset instructions. If your converted LeRobot dataset
stores `videos/` as symlinks, those symlinks must resolve on the target machine.

For local shared storage, the same layout can be achieved with symlinks:

```bash
mkdir -p checkpoints datasets work_dirs
ln -sfnT /shared/checkpoints/GR00T-N1.5-3B checkpoints/GR00T-N1.5-3B
ln -sfnT /shared/datasets/robocasa_lerobot_V2.1 datasets/robocasa_fluxvla
```

## Training and Evaluation

Training can be launched with:

```bash
export ROBOCASA_DATASET_ROOT=/path/to/robocasa_lerobot_V2.1
export WORK_DIR=work_dirs/groot_robocasa_full
export PER_DEVICE_BS=16
export LEARNING_RATE=3e-5
export WARMUP_RATIO=0.05
export WEIGHT_DECAY=1e-5
export LR_SCHEDULER_TYPE=linear-warmup+cosine-decay
export MAX_STEPS=60000
export SAVE_ITER_INTERVAL=5000
export MAX_KEEP_CKPTS=20

bash scripts/train_groot_robocasa.sh
```

Evaluation can be launched with:

```bash
bash scripts/eval_robocasa.sh \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --ckpt-path work_dirs/groot_robocasa_full/checkpoints/step-060000.safetensors \
  --output-dir work_dirs/groot_robocasa_full/eval_step060000 \
  --cfg-options \
    eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
    eval.num_trials_per_task=20
```

## Notes

- Keep GR00T and RoboCasa editable installs separate from the base FluxVLA
  package so they can be upgraded independently.
- Prefer explicit `eval.norm_stats_path=...` during evaluation so the exact
  normalization statistics file is visible in the command history.
- If you need to run LIBERO and RoboCasa in separate environments, keep the
  base FluxVLA v1.x environment unchanged and create a second environment with
  the patched robosuite install above.
