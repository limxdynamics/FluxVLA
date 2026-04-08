# FluxVLA引擎：专为具身智能打造的“一站式”VLA 工程平台

<p align="center">
  <img src="assets/fluxvla.png" alt="FluxVLA" width="600">
</p>

<div align="center">
<a href="https://huggingface.co/limxdynamics/FluxVLAEngine"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://fluxvla.limxdynamics.com"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://fluxvla.limxdynamics.com/zh"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://github.com/limxdynamics/FluxVLA/issues/1"><img src="https://img.shields.io/badge/微信-green?logo=wechat"></a>
<a href="https://github.com/limxdynamics/FluxVLA/issues/1"><img src="https://img.shields.io/badge/飞书-3370FF?logo=lark&logoColor=white"></a>
</div>

<div align="center">

[English](README.md) | 简体中文 | [日本語](README_ja.md)

</div>

FluxVLA Engine是面向具身智能落地应用的全链路一体化工程平台，以统一配置、标准接口、模块解耦、可部署为核心设计理念，构建从数据到真机部署的完整工程闭环，并以“标准化产学研基座”为目标，显著降低 VLA 研究与开发的工程门槛。

## 框架

<p align="center">
  <img src="assets/framework.png" alt="Framework Architecture" width="800">
</p>

## 性能

| Codebase                    | Libero-Spatial | Libero-Object | Libero-Goal | Libero-Long | Libero-Average |
| --------------------------- | -------------- | ------------- | ----------- | ----------- | -------------- |
| FluxVLA(GR00T)              | 96.4           | 93.8          | 93.6        | 83.5±1.5    | 91.8           |
| FluxVLA(Pi)                 | 99.4           | 99.4          | 98          | 96.8        | 98.4           |
| FluxVLA(Qwen3VL 0.6B+GR00T) | 98             | 99.2          | 95.2        | 87.2        | 94.9           |

## 📢 最新动态

**\[2026/04/03\]** 🔥 FluxVLA开源了。

## 🛠️ 安装

以下安装指南以 NVCC 12.4 为例。如果你的环境不同，请相应调整 CUDA 版本。

<details>
<summary><b>1. 创建 conda 环境</b></summary>

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
```

</details>

<details>
<summary><b>2. 安装 PyTorch（CUDA 版本）</b></summary>

> **重要**：在执行 `pip install -r requirements.txt` 之前，**必须**先从官方 CUDA 索引安装 PyTorch。默认 PyPI 索引无法获取 CUDA 版本构建。

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

对于其他 CUDA 版本，请将 `cu124` 替换为对应值（例如 `cu118`、`cu121`）。详见 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 。

</details>

<details>
<summary><b>3. 安装 flash-attention</b></summary>

方式 1：通过 pip 直接安装：

```bash
pip install psutil ninja packaging
# MAX_JOBS 控制并行编译线程数，请根据机器资源调整
MAX_JOBS=8 pip install flash-attn==2.5.5 --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases
```

方式 2：源码编译安装（若方式 1 失败，推荐使用）：

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.5
# MAX_JOBS 控制并行编译线程数，请根据机器资源调整
MAX_JOBS=8 python setup.py install
```

</details>

<details>
<summary><b>4. 安装 av</b></summary>

```bash
conda install -c conda-forge av=14.4.0
```

</details>

<details>
<summary><b>5. 安装 fluxvla 及其余依赖</b></summary>

```bash
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

> **说明**：`requirements.txt` 固定了 `torch==2.6.0`，以避免 pip 意外替换掉第 2 步安装的 CUDA 版 PyTorch。若需使用其他 torch 版本，请同时更新第 2 步命令与 `requirements.txt` 中的版本。

</details>

<details>
<summary><b>在线评估环境（LIBERO / EGL）</b></summary>

如果你要在不支持光线追踪的设备（如 A100）上评估 LIBERO，请参考 [EGL Device GPU Rendering Configuration](https://github.com/google-deepmind/mujoco/issues/572#issuecomment-2419965230)。

**安装系统依赖**

```bash
export MUJOCO_GL=egl
sudo apt install libegl-dev libgl1-mesa-dev libx11-dev libglew-dev libosmesa6-dev
```

**环境检查**

确认 `/proc/1/environ` 中包含以下环境变量：

- `NVIDIA_DRIVER_CAPABILITIES=all`
- `NVARCH=x86_64`
- `NVIDIA_REQUIRE_CUDA=cuda>=12.4`
- `brand=tesla` 且 `driver>=470`

**创建 EGL 配置文件**

创建文件 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`，内容如下：

```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
```

</details>

<details>
<summary><b>配置 pre-commit 钩子（可选但推荐）</b></summary>

为保证代码质量与一致性（尤其是 C++/CUDA 代码），建议安装 pre-commit 钩子：

```bash
pip install pre-commit
pre-commit install
```

这样会在每次提交前自动检查并格式化代码。

</details>

<details>
<summary><b>配置 Weights & Biases（wandb）</b></summary>

[Weights & Biases](https://wandb.ai/) 用于实验跟踪与可视化。配置方式如下：

1. 安装 wandb（已包含在 requirements.txt 中）：

```bash
pip install wandb
```

2. 登录你的 wandb 账号：

```bash
wandb login
```

3. 设置环境变量：

```bash
export WANDB_PROJECT=fluxvla        # 项目名（默认：fluxvla）
export WANDB_ENTITY=your-team-name  # 团队名或用户名（默认：None）
export WANDB_MODE=online            # online、offline 或 disabled（默认：online）
```

4. 如需在训练时禁用 wandb 日志，请设置：

```bash
export WANDB_MODE=disabled
```

说明：所有 wandb 配置都通过环境变量读取，无需在配置文件中额外设置。

</details>

## 📦 数据准备

<details>
<summary><b>直接使用我们准备好的数据</b></summary>

下载所需数据集并放到 `./datasets` 目录。请根据配置仅下载你需要的数据集。

| 数据集                 | 下载链接                                                                                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| libero-object          | [limxdynamics/FluxVLAData/libero_object_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_object_no_noops_lerobotv2.1)   |
| libero-spatial         | [limxdynamics/FluxVLAData/libero_spatial_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_spatial_no_noops_lerobotv2.1) |
| libero-10              | [limxdynamics/FluxVLAData/libero_10_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_10_no_noops_lerobotv2.1)           |
| libero-goal            | [limxdynamics/FluxVLAData/libero_goal_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_goal_no_noops_lerobotv2.1)       |
| modified_libero_rlds   | [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)                                                                           |
| RealRobot_AgileX_aloha | [limxdynamics/FluxVLAData/RealRobot_AgileX_aloha_lerobot_v2](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/RealRobot_AgileX_aloha_lerobot_v2)     |
| RealRobot_UR3_Chem     | [limxdynamics/FluxVLAData/RealRobot_UR3_Chem_lerobot_v2](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/RealRobot_UR3_Chem_lerobot_v2)             |

例如，下载 libero-10 数据集：

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "libero_10_no_noops_lerobotv2.1/*" --local-dir ./datasets
```

将 `libero_10_no_noops_lerobotv2.1` 替换为其他数据集对应的文件夹名即可下载。

</details>

<details>
<summary><b>私有数据集目录结构</b></summary>

若使用 fluxvla 在私有数据集上训练，需要先将原始数据（如 ALOHA 双臂机器人采集的 HDF5 文件）转换为 LeRobot Dataset v2.1 格式。详细的转换步骤请参考 [数据转换指南](docs/data_convert.md)。

转换后的数据集目录结构如下：

```
├── data
│   └── chunk000
│   │   └── episode_000000.parquet
│   │   └── episode_000001.parquet
│   │   └── ... (更多 parquet 文件)
│   │   └── episode_00000N.parquet
│   └── chunk001
│   └── ... (更多 chunk)
│   └── chunk00N
├── meta
│   └── episodes.jsonl
│   └── episodes_stats.jsonl
│   └── info.json
│   └── tasks.jsonl
├── videos
│   └── chunk000
│   │   └── camera name 0
│   │   │   └── episode_000000.mp4
│   │   │   └── episode_000001.mp4
│   │   │   └── ...(更多 mp4 文件)
│   │   │   └── episode_00000N.mp4
│   │   └── camera name 1
│   └── chunk001
│   └── ... (更多 chunk)
│   └── chunk00N
```

</details>

## Checkpoint 准备

下载所需预训练 checkpoint 并放到 `./checkpoints` 目录。请根据配置仅下载你需要的 checkpoint。

<details>
<summary><b>VLA 模型</b></summary>

| 模型        | 大小 | 下载链接                                                                                   |
| ----------- | ---- | ------------------------------------------------------------------------------------------ |
| GR00T N1.5  | 3B   | [🤗 Hugging Face](https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/main)                   |
| OpenVLA     | 7B   | [🤗 Hugging Face](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)           |
| PI0_base    | 3B   | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_base)    |
| PI05_base   | 3B   | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_base)   |
| PI05_libero | 3B   | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_libero) |

</details>

<details>
<summary><b>视觉语言模型（VLM）</b></summary>

| 模型       | 大小 | 下载链接                                                              |
| ---------- | ---- | --------------------------------------------------------------------- |
| Qwen2.5-VL | 3B   | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |

</details>

<details>
<summary><b>大语言模型（LLM）</b></summary>

| 模型     | 大小 | 下载链接                                                                     |
| -------- | ---- | ---------------------------------------------------------------------------- |
| Qwen 2.5 | 3B   | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B)                    |
| Qwen 2.5 | 7B   | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B)                    |
| Llama 2  | 7B   | [🤗 Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) |

</details>

<details>
<summary><b>视觉主干网络</b></summary>

| 模型                | 下载链接                                                                             |
| ------------------- | ------------------------------------------------------------------------------------ |
| ViT-Large (DINOv2)  | [🤗 Hugging Face](https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m) |
| ViT-SO400M (SigLIP) | [🤗 Hugging Face](https://huggingface.co/timm/ViT-SO400M-14-SigLIP)                  |
| SigLIP2             | [🤗 Hugging Face](https://huggingface.co/google/siglip2-base-patch16-224)            |
| paligemma           | [🤗 Hugging Face](https://huggingface.co/google/paligemma-3b-pt-224)                 |

> **提示**：可使用 `huggingface-cli download <model-name> --local-dir ./checkpoints/<model-name>` 加速下载。

</details>

<details>
<summary><b>已训练模型</b></summary>

你也可以下载已经使用 FluxVLA 训练好的模型，直接用于推理或评估。请放在 `./work_dirs` 目录下。

| 模型                      | 下载链接                                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| PI0.5 PaliGemma Libero-10 | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_10_full_finetune_bs64) |
| GR00T Eagle 3B Libero-10  | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_10_full_finetune_bs64) |

```bash
# 示例：从 limxdynamics/FluxVLAEngine 下载 PI0.5 checkpoint
huggingface-cli download limxdynamics/FluxVLAEngine --include "pi05_paligemma_libero_10_full_finetune_bs64/*" --local-dir ./checkpoints/pi05_paligemma_libero_10_full_finetune_bs64
```

</details>

## 🌟 特性

<details>
<summary><b>All-in-one：单配置文件管理全流程</b></summary>

- 支持通过一个配置文件统一管理数据、模型、训练、评测、推理与部署所需的关键参数（便于复现与部署）。

</details>

<details>
<summary><b>支持不同 VLA 模型</b></summary>

- 支持 OpenVLA、LlavaVLA、Gr00t、Pi0 与 Pi0.5。

</details>

<details>
<summary><b>支持不同模块</b></summary>

- 支持 Llama、Gemma 与 Qwen 系列 LLM 主干。
- 支持 DINOv2、SigLIP 视觉主干。
- 支持 PaliGemma 与 Qwen-VL VLM 主干。

</details>

<details>
<summary><b>支持不同训练策略</b></summary>

- 支持同时使用 FSDP 与 DDP，支持 LoRA 训练模式。
- 支持 train 后立即 eval（eval-after-train）。
- 支持从 checkpoint 恢复训练。

</details>

<details>
<summary><b>数据与权重格式</b></summary>

- 支持 Parquet 数据集，并支持加载 LeRobot 格式数据。
- 支持 safetensors 格式模型权重。

</details>

<details>
<summary><b>评估与推理能力</b></summary>

- 支持多 GPU 评估。
- 支持在无光追设备上评估 libero。
- 支持 [RTC (Real-Time Chunking)](docs/rtc.md)，提升跨 chunk 轨迹连续性。
- 支持 GR00T 与 PI0.5 推理加速；详见 [Inference Acceleration](docs/inference_acceleration.md)，包含 Triton 融合核、CUDA Graph 捕获与 CUDA 自定义算子。

</details>

<p align="center">
  <img src="assets/VLA_speedup.png" alt="VLA Speedup" width="800">
</p>

## 使用方式

<details>
<summary><b>本地调试</b></summary>

```
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/train.py --config [CONFIG_PATH] --work-dir [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE]
```

例如：

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py --work-dir ./checkpoints/pi05_paligemma_libero_10_full_finetune --cfg-options train_dataloader.per_device_batch_size=2
```

</details>

<details>
<summary><b>本地评估</b></summary>

```
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/eval.py --config [CONFIG_PATH] --ckpt-path [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

例如：

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/eval.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors
```

</details>

<details>
<summary><b>集群训练</b></summary>

```
export WANDB_MODE=disabled
bash scripts/train.sh [CONFIG] [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] train_dataloader.batch_size=[GLOBAL_BATCH_SIZE] runner.max_steps=[MAX_STEPS] runner.save_interval=[SAVE_INTERVAL] runner.max_keep_ckpts=[MAX_KEEP_CKPTS] --eval-after-train
```

</details>

<details>
<summary><b>从 checkpoint 恢复训练</b></summary>

要从 checkpoint 恢复训练，可使用 `--resume-from` 参数指定 checkpoint 文件路径。训练会从已保存的 global step、epoch、模型状态与优化器状态继续。

**本地训练示例：**

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --work-dir ./work_dirs/pi05_paligemma_libero_10_full_finetune \
  --resume-from ./work_dirs/pi05_paligemma_libero_10_full_finetune/checkpoints/checkpoint_epoch_5.pt \
  --cfg-options train_dataloader.per_device_batch_size=2
```

**集群训练示例：**

```
export WANDB_MODE=disabled
bash scripts/train.sh [CONFIG] [WORK_DIR] \
  --resume-from [CHECKPOINT_PATH] \
  --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] runner.max_steps=[MAX_STEPS]
```

</details>

<details>
<summary><b>集群评估</b></summary>

```
export WANDB_MODE=disabled
bash scripts/eval.sh [CONFIG] [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

</details>

<details>
<summary><b>真机推理</b></summary>

在真实机器人上运行推理时，请先在机器人端安装好环境，然后执行以下命令：

```
python scripts/inference_real_robot.py --config [CONFIG] -- ckpt-path [CKPT_PATH]
```

</details>

## 常见问题（FAQ）

<details>
<summary><b>Q：下载模型或数据集时，连接 Hugging Face 有问题。</b></summary>

A：如果遇到 Hugging Face 连接问题（如下载慢、超时、连接被拒绝），可以在执行命令前设置以下环境变量，使用 [hf-mirror](https://hf-mirror.com)：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

</details>

<details>
<summary><b>Q：`conda install av` 解析环境很慢。</b></summary>

A：可使用 `libmamba` 求解器加速依赖解析：

```bash
conda install -c conda-forge av=14.4.0 --solver=libmamba
```

</details>

<details>
<summary><b>Q：GR00T 在 LIBERO 上评估结果不稳定。</b></summary>

A：这是预期现象。GR00T 在 LIBERO 上的表现对随机种子、硬件环境和训练 epoch 数都较敏感。这些因素的小变化都可能导致评估结果明显波动。建议使用多个随机种子进行实验，并依据评估表现选择最优 checkpoint。

</details>

<details>
<summary><b>Q：执行 `pip install -r requirements.txt` 时构建 `egl_probe` 失败，报错 `RuntimeError: CMake must be installed`。</b></summary>

A：`egl_probe` 需要 CMake 才能构建。可通过 conda（推荐）或 apt 安装：

```bash
conda install -c conda-forge cmake
# 或
sudo apt install cmake
```

> **说明**：不要使用 `pip install cmake`，pip 版本是 Python 封装，在 pip 隔离构建环境中可能失败。

</details>

<details>
<summary><b>Q：`egl_probe` 构建失败，提示 `Compatibility with CMake < 3.5 has been removed from CMake`。</b></summary>

A：这通常是因为你的 CMake 版本对 `egl_probe` 的 CMakeLists.txt 来说过新。安装前先设置以下环境变量：

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -r requirements.txt
```

</details>

<details>
<summary><b>Q：安装后出现 NumPy 版本错误（如 `RuntimeError: Numpy is not available` 或版本不兼容警告）。</b></summary>

A：安装过程中某些依赖可能覆盖了固定的 NumPy 版本。直接重装正确版本即可：

```bash
pip install numpy==1.26.4
```

</details>

<details>
<summary><b>Q：在 RTX 5090 上推理失败（如 Triton kernel 错误或 CUDA 兼容性问题）。</b></summary>

A：RTX 5090（Blackwell 架构）需要更新版本的 Triton。请升级到 Triton 3.2.0 或更高版本：

```bash
pip install triton==3.2.0
```

</details>

## 支持

如果你在使用本仓库时遇到问题，欢迎联系我们。你可以直接联系 [mason@limxdynamics.com](mason@limxdynamics.com) 和 [wayne@limxdynamics.com](wayne@limxdynamics.com)，或在 Github 提交 issue 获取帮助。

## 🙏 致谢

本项目受益于以下开源项目与社区工作，在此一并致谢：

- [LeRobot](https://github.com/huggingface/lerobot)
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main)
- [OpenVLA](https://github.com/openvla/openvla)
- [OpenPI (pi0)](https://github.com/Physical-Intelligence/openpi)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Qwen](https://github.com/QwenLM)
- [Triton](https://github.com/triton-lang/triton)
- [RTC](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [Training RTC](https://arxiv.org/pdf/2512.05964)
- [Realtime-VLA](https://github.com/Dexmal/realtime-vla)

## 路线图

- 支持更多视觉主干网络。
- 支持更多 VLM 主干。
- 支持更多 VLA 方法。
- 支持使用 VLM 数据或思维链（CoT）数据进行训练。
- RLDS 数据集将废弃并被 Parquet 数据集替代。
- logger 功能将完整实现。
- 支持 issacsim。
- 支持SARM
