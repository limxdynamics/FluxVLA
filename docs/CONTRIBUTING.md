# Contributing to FluxVLA

Thank you for your interest in contributing to FluxVLA! This document provides guidelines and instructions for contributing to the project.

English | [简体中文](#简体中文)

______________________________________________________________________

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Project Structure](#project-structure)
- [Adding a New VLA Model](#adding-a-new-vla-model)
- [Submitting a Bug Report](#submitting-a-bug-report)
- [License](#license)
- [Contact](#contact)

______________________________________________________________________

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for everyone. Please be respectful and constructive in all interactions. Harassment, discrimination, and disruptive behavior will not be tolerated.

## How Can I Contribute?

There are many ways to contribute to FluxVLA:

- **Bug Reports**: Found a bug? Open a GitHub Issue with detailed reproduction steps.
- **New VLA Models**: Add support for new VLA architectures (see [Adding a New VLA Model](#adding-a-new-vla-model)).
- **New Modules**: Contribute new LLM backbones, vision backbones, or VLM backbones.
- **Performance Optimization**: Improve inference speed with Triton kernels, CUDA operators, or CUDA Graph optimizations.
- **Documentation**: Fix typos, improve clarity, or add new guides.
- **Bug Fixes**: Pick up an open issue and submit a fix.

Before starting work on a **new feature or model**, please open an Issue first to discuss the design with the maintainers.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/FluxVLA.git
cd FluxVLA
git remote add upstream https://github.com/FluxVLA/FluxVLA.git
```

### 2. Set Up the Development Environment

Follow the [Installation guide](../README.md#-installation) in the README to set up your conda environment, install PyTorch, flash-attention, and other dependencies.

### 3. Install Pre-commit Hooks (Required)

```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### Branch Naming

Create a descriptive branch from the latest `main`:

```bash
git fetch upstream
git checkout -b <type>/<short-description> upstream/main
```

Use these prefixes:

| Prefix      | Purpose                               |
| ----------- | ------------------------------------- |
| `feat/`     | New feature or model                  |
| `fix/`      | Bug fix                               |
| `docs/`     | Documentation only                    |
| `refactor/` | Code refactoring (no behavior change) |
| `perf/`     | Performance improvement               |
| `test/`     | Adding or updating tests              |

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>

<optional body>
```

Examples:

```bash
git commit -m "feat(models): add support for CogACT model"
git commit -m "fix(data): handle missing camera keys in parquet loader"
git commit -m "docs(readme): fix installation command for CUDA 12.1"
git commit -m "perf(gr00t): add Triton fused attention kernel"
```

For multi-line commit messages with a body:

```bash
git commit -m "feat(models): add support for CogACT model

Implement CogACT backbone with pretrained weight loading.
Includes unit tests and updated documentation."
```

### Pull Request Process

1. **One PR per issue** — keep changes focused and reviewable.
2. **Link the related Issue** in the PR description (e.g., `Closes #42`).
3. **Describe your changes** — explain *what* you changed, *why*, and *how to test* it.
4. **Ensure all checks pass** — pre-commit hooks and CI must be green.
5. **Keep commits clean** — squash fixup commits before requesting review.
6. **Be responsive** — address review feedback promptly.

## Code Style

Code style is enforced automatically via pre-commit hooks. The key tools are:

### Python

| Tool                                                        | Version | Purpose                                |
| ----------------------------------------------------------- | ------- | -------------------------------------- |
| [yapf](https://github.com/google/yapf)                      | v0.32.0 | Code formatting (based on PEP 8 style) |
| [isort](https://github.com/PyCQA/isort)                     | v5.11.5 | Import sorting                         |
| [flake8](https://github.com/PyCQA/flake8)                   | v5.0.4  | Linting                                |
| [codespell](https://github.com/codespell-project/codespell) | v2.2.1  | Spell checking                         |

### C++ / CUDA

| Tool                                                         | Version | Purpose         |
| ------------------------------------------------------------ | ------- | --------------- |
| [clang-format](https://clang.llvm.org/docs/ClangFormat.html) | v18.1.4 | Code formatting |
| [cpplint](https://github.com/cpplint/cpplint)                | v1.6.1  | Style linting   |

### Markdown

| Tool                                                    | Version | Purpose             |
| ------------------------------------------------------- | ------- | ------------------- |
| [mdformat](https://github.com/executablebooks/mdformat) | v0.7.9  | Markdown formatting |

### General

- Naming follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#316-naming) (and [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#Naming) for C++/CUDA code).
- Use single quotes for Python strings (enforced by `double-quote-string-fixer`).
- Use LF line endings (enforced by `mixed-line-ending --fix=lf`).
- Remove trailing whitespace and ensure files end with a newline.
- Maximum line length is **79 characters** for Python code.

## Project Structure

```
FluxVLA/
├── fluxvla/            # Main Python package
│   ├── models/         # VLA model implementations (GR00T, PI0, PI0.5, OpenVLA, LlavaVLA)
│   ├── data/           # Data loading and processing pipelines
│   └── ...
├── configs/            # Training / evaluation / inference configs
│   ├── gr00t/          #   GR00T model configs
│   ├── pi0/            #   PI0 model configs
│   ├── pi05/           #   PI0.5 model configs
│   ├── openvla/        #   OpenVLA model configs
│   └── llava/          #   LlavaVLA model configs
├── scripts/            # Entry-point scripts (train, eval, inference)
├── tools/              # Utility scripts and helper tools
├── test/               # Unit tests
├── assets/             # Images and static resources
├── docs/               # Documentation
├── datasets/           # Dataset storage (not committed to git)
├── checkpoints/        # Model checkpoints (not committed to git)
├── requirements.txt    # Python dependencies
└── setup.py            # Package setup with CUDA extensions
```

## Adding a New VLA Model

Adding a new VLA model is one of the most impactful contributions. Here is the general workflow:

### Step 1: Open a Discussion Issue

Before writing code, open an Issue titled `[Model] Add support for <ModelName>` to discuss the architecture, data format requirements, and expected evaluation benchmarks.

### Step 2: Implement the Model

1. Create a new directory under `fluxvla/models/` for your model.
2. Implement the model following the existing patterns (e.g., `fluxvla/models/` for GR00T or PI0.5).
3. Ensure your model works with the existing data pipeline (`fluxvla/data/`).

### Step 3: Add Configuration Files

1. Create a new directory under `configs/<model_name>/`.
2. Provide at least one training config and one evaluation config.
3. Follow the single-config-file principle — one config should manage the full workflow.

### Step 4: Provide Evaluation Results

- Run evaluation on at least one LIBERO benchmark suite (e.g., `libero-10`).
- Include results in your PR description with the exact config and checkpoint used.
- If applicable, provide real-robot evaluation results as well.

### Step 5: Update Documentation

- Add your model to the supported models list in the README.
- Add yourself to `docs/CONTRIBUTORS.md` with a summary of your contributions.

## Submitting a Bug Report

When reporting a bug, please include:

1. **Environment information**:

   - OS and kernel version
   - Python version
   - PyTorch version and CUDA version (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
   - GPU model (`nvidia-smi`)
   - FluxVLA version or commit hash (`git rev-parse HEAD`)

2. **Steps to reproduce**: The exact commands and config file used.

3. **Expected vs. actual behavior**: What you expected to happen and what actually happened.

4. **Error logs**: Full traceback or error message (use code blocks for formatting).

## License

By contributing to FluxVLA, you agree that your contributions will be licensed under the [Apache License 2.0](../LICENSE). All new files should not include license headers — the repository-level LICENSE file applies to all contents.

## Contact

- **Maintainers**: [mason@limxdynamics.com](mailto:mason@limxdynamics.com), [wayne@limxdynamics.com](mailto:wayne@limxdynamics.com)
- **WeChat Group**: [Join](https://github.com/FluxVLA/FluxVLA/issues/1)
- **Feishu Group**: [Join](https://github.com/FluxVLA/FluxVLA/issues/1)
- **GitHub Issues**: [https://github.com/FluxVLA/FluxVLA/issues](https://github.com/FluxVLA/FluxVLA/issues)

______________________________________________________________________

<a id="简体中文"></a>

# 贡献指南

感谢你对 FluxVLA 的关注！本文档介绍如何参与项目贡献。

## 行为准则

我们致力于为所有人提供友好、包容的环境。请在所有互动中保持尊重和建设性态度。

## 贡献方式

- **Bug 报告**：发现问题请提 Issue，附上详细的复现步骤。
- **新模型支持**：添加新的 VLA 模型架构（参见上方 [Adding a New VLA Model](#adding-a-new-vla-model)）。
- **新模块**：贡献新的 LLM 骨干网络、视觉骨干网络或 VLM 骨干网络。
- **性能优化**：通过 Triton 算子、CUDA 算子或 CUDA Graph 加速推理。
- **文档改进**：修正错别字、提升文档清晰度或添加新指南。
- **Bug 修复**：认领已有 Issue 并提交修复。

对于**新功能或新模型**，请先开 Issue 讨论设计方案，获得维护者认可后再开发。

## 快速开始

1. Fork 仓库并克隆到本地。
2. 按照 [README 安装指南](../README_zh-CN.md) 搭建开发环境。
3. 安装 pre-commit hooks（**必须**）：`pip install pre-commit && pre-commit install`

## 开发规范

- **分支命名**：`feat/xxx`、`fix/xxx`、`docs/xxx`、`refactor/xxx`、`perf/xxx`
- **Commit 消息**：遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。
- **Pull Request**：一个 PR 只解决一个问题；关联相关 Issue；描述改动内容、原因和测试方法。

## 代码风格

代码风格通过 pre-commit hooks 自动检查，主要工具：

- **Python**：yapf 格式化、isort 排序、flake8 检查、codespell 拼写检查
- **C++/CUDA**：clang-format 格式化、cpplint 检查
- **Markdown**：mdformat 格式化
- 命名遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html#316-naming)（C++/CUDA 遵循 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html#Naming)）
- 使用单引号字符串、LF 换行符、无尾部空格

## Bug 报告

提交 Bug 报告时请包含：

1. 环境信息（OS、Python、PyTorch、CUDA 版本、GPU 型号）
2. 复现步骤（使用的命令和配置文件）
3. 预期行为 vs 实际行为
4. 完整的错误日志

## 许可证

向 FluxVLA 贡献代码即表示你同意你的贡献将按 [Apache License 2.0](../LICENSE) 许可。

## 联系方式

- **维护者邮箱**：[mason@limxdynamics.com](mailto:mason@limxdynamics.com)、[wayne@limxdynamics.com](mailto:wayne@limxdynamics.com)
- **微信群**：[加入](https://github.com/FluxVLA/FluxVLA/issues/1)
- **飞书群**：[加入](https://github.com/FluxVLA/FluxVLA/issues/1)
- **GitHub Issues**：[https://github.com/FluxVLA/FluxVLA/issues](https://github.com/FluxVLA/FluxVLA/issues)
