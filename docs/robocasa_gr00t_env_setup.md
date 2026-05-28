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
# Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git /path/to/Isaac-GR00T
cd /path/to/Isaac-GR00T
git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96  # n1.5-release

pip install --no-deps -e /path/to/Isaac-GR00T

# RoboCasa GR1 tabletop tasks
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git /path/to/robocasa-gr1-tabletop-tasks
cd /path/to/robocasa-gr1-tabletop-tasks
git checkout 4840e671596f93ca03651524b9f72ffb1aadfeff

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

下载assert：

```bash
# 5. Download assets
cd robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y
```

Then verify RoboCasa registration:

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

### 开始训练和评测前，确保权重和数据集，统计量已经下载到本地，或从共享存储中建立好软链接。

## Training and Evaluation

Training can be launched with:

```bash
cd /path/to/FluxVLA
mkdir -p datasets
ln -sfnT /path/to/robocasa_lerobot_V2.1 datasets/robocasa_fluxvla

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

起训练时报错：
[rank0]: Traceback (most recent call last):
[rank0]:   File "/mnt/workspace/mnt/data/yiming/fluxvla-pr-gr00t-v2/scripts/train.py", line 352, in <module>
[rank0]:     train(args, cfg)
[rank0]:   File "/mnt/workspace/mnt/data/yiming/fluxvla-pr-gr00t-v2/scripts/train.py", line 304, in train
[rank0]:     dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
[rank0]:   File "/root/projects/FluxVLA/fluxvla/engines/utils/builder.py", line 209, in build_dataset_from_cfg
[rank0]:     return build_from_cfg(cfg, DATASETS, default_args)
[rank0]:   File "/root/projects/FluxVLA/fluxvla/engines/utils/builder.py", line 130, in build_from_cfg
[rank0]:     obj = obj_cls(**args)  # type: ignore
[rank0]: TypeError: DistributedRepeatingDataset.__init__() got an unexpected keyword argument 'dataset_statistics_path'

因为我们使用的是ali的v1版本的fluxvla镜像，里面的fluxvla是旧版本，没有对统计量的路径指定适配代码，现在通过显示指定解决，后面官方仓库镜像中拉取仓库中的更新即可。

在这里我重装一下fluxvla
cd /mnt/workspace/mnt/data/yiming/fluxvla-pr-gr00t-v2

python -m pip uninstall -y fluxvla FluxVLA || true

这里要使用 --no-build-isolation 重新装 不然会报缺少torch，这是 pip build isolation 导致的：它创建了一个临时构建环境，里面没有 torch，而 FluxVLA 的 setup.py 在构建阶段 import 了 torch。
python -m pip install --no-deps --no-build-isolation -e .

然后检查是否指向当前 PR 目录：
```bash
python - <<'PY'
import fluxvla
from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset
import inspect
print("fluxvla path:", fluxvla.__file__)
print("has dataset_statistics_path:",
      "dataset_statistics_path" in str(inspect.signature(DistributedRepeatingDataset.__init__)))
PY
```


Evaluation can be launched with:

```bash
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --ckpt-path work_dirs/groot_robocasa_full/checkpoints/step-060000.safetensors \
  --cfg-options \
    eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
    eval.output_dir=work_dirs/groot_robocasa_full/eval_step060000 \
    eval.num_trials_per_task=20

使用基础gr00t评测一个任务一次，不传参覆盖eval.task_list即使默认测评24任务，eval.num_trials_per_task每个任务评测次数
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

使用你微调后的gr00t评测一个任务一次
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --ckpt-path work_dirs/smoke_groot_robocasa_24x30/checkpoints/step-000002-epoch-00-loss=0.0256.safetensors \
  --cfg-options \
    eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
    eval.output_dir=work_dirs/smoke_groot_robocasa_24x30/eval_step_000002 \
    eval.task_list="['gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env']" \
    eval.num_trials_per_task=1 \
    eval.save_video=true
```

## Notes

- Keep GR00T and RoboCasa editable installs separate from the base FluxVLA
  package so they can be upgraded independently.
- Prefer explicit `eval.norm_stats_path=...` during evaluation so the exact
  normalization statistics file is visible in the command history.
- If you need to run LIBERO and RoboCasa in separate environments, keep the
  base FluxVLA v1.x environment unchanged and create a second environment with
  the patched robosuite install above.
