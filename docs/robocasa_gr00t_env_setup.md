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

## GR00T RoboCasa Configs

The integration provides two GR00T RoboCasa configs:

```bash
configs/gr00t/gr00t_eagle_3b_robocasa_posttrain_24x30.py
configs/gr00t/gr00t_eagle_3b_robocasa_posttrain_24x30_official_aug.py
```

Use the `official_aug` config when matching the Isaac-GR00T fine-tuning image
pipeline. It adds random crop and color jitter before resize and normalization.

Both configs expect RoboCasa data to be available under:

```bash
datasets/robocasa_fluxvla
```

You can create this as a symlink to your converted RoboCasa LeRobot dataset:

```bash
mkdir -p datasets
ln -sfnT /path/to/robocasa_lerobot_V2.1 datasets/robocasa_fluxvla
```

## Notes

- Keep GR00T and RoboCasa editable installs separate from the base FluxVLA
  package so they can be upgraded independently.
- Prefer explicit `eval.norm_stats_path=...` during evaluation so the exact
  normalization statistics file is visible in the command history.
- If you need to run LIBERO and RoboCasa in separate environments, keep the
  base FluxVLA v1.x environment unchanged and create a second environment with
  the patched robosuite install above.
