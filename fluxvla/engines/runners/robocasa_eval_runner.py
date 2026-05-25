# ============================================================
# RobocasaEvalRunner — Robocasa 仿真评测 Runner
# ============================================================
#
# 对标 LiberoEvalRunner，在 FluxVLA 框架中评测模型在 Robocasa 环境上的表现。
#
# 与 LiberoEvalRunner 的核心差异:
#   - 环境: gymnasium API (5 元组) vs LIBERO 自定义 API (4 元组)
#   - 动作: 29 维关节角度 dict vs 7 维末端 flat list
#   - 状态: 29 维关节角度 vs 8 维 eef_pos+quat+gripper
#   - 图像: 单相机 ego_view vs 双相机 agentview+wrist
#   - 归一化: min_max vs mean_std
#   - 无 gripper binarize/invert 后处理
#   - 无等待机制 (LIBERO 需前 10 步等物体掉落)
#
# 环境要求:
#   运行时需要 PYTHONPATH 前置 robosuite 1.5.1 源码路径:
#   PYTHONPATH=/root/projects/robosuite:$PYTHONPATH
#
# 作者: yiming | 创建: 2026-04-14
# ============================================================

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from safetensors.torch import load_file

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)

# Robocasa GR1 29 维动作拆分为 env.step 所需的 dict 格式。
# 这里使用官方 GR00T N1.5 fourier_gr1_arms_waist 顺序：
# left_arm + right_arm + left_hand + right_hand + waist。
ROBOCASA_ACTION_KEYS = {
    'action.left_arm':   (0, 7),    # 左臂 7 维
    'action.right_arm':  (7, 14),   # 右臂 7 维
    'action.left_hand':  (14, 20),  # 左手 6 维
    'action.right_hand': (20, 26),  # 右手 6 维
    'action.waist':      (26, 29),  # 腰部 3 维
}


@RUNNERS.register_module()
class RobocasaEvalRunner:
    """Runner for evaluating VLA models on Robocasa simulation tasks.

    Args:
        cfg: 完整配置对象 (包含 model 段)
        seed: 随机种子
        ckpt_path: 模型权重路径 (.safetensors 或 .pt)
        model_family: 模型族 ('pi0' 等)
        task_list: Robocasa gymnasium 环境名列表
        dataset: 评测数据集配置 (观测预处理 transform pipeline)
        denormalize_action: 动作反归一化配置
        eval_chunk_size: 每次推理取多少步动作执行，默认 10
        max_episode_steps: 单 episode 最大步数，默认 720
        num_trials_per_task: 每个任务的评测 episode 数，默认 50
        mixed_precision_dtype: 混合精度类型，默认 'bf16'
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_list: List[str],
                 dataset: Dict,
                 denormalize_action: Dict,
                 eval_chunk_size: int = 10,
                 max_episode_steps: int = 720,
                 num_trials_per_task: int = 50,
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True,
                 unnorm_key: str = 'robocasa_gr1_test',
                 output_dir: Optional[str] = None,
                 save_video: bool = True,
                 video_steps_per_render: int = 2,
                 video_fps: int = 10,
                 norm_stats_path: Optional[str] = None,
                 grouped_norm_stats: bool = False,
                 norm_stats_group_names: Optional[List[str]] = None,
                 **kwargs):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)

        self.device_id = overwatch.local_rank()

        # --- 构建模型 ---
        if hasattr(cfg, 'inference_model'):
            self.vla = build_vla_from_cfg(cfg.inference_model).eval()
        else:
            self.vla = build_vla_from_cfg(cfg.model).eval()

        # --- 加载权重 ---
        if ckpt_path is not None:
            assert Path(ckpt_path).exists(), \
                f'Checkpoint not found: {ckpt_path}'

            # 支持三种格式：目录（多文件 safetensors）、单个 safetensors、.pt 文件
            if os.path.isdir(ckpt_path):
                # 目录路径：加载所有 .safetensors 文件并合并
                overwatch.info(f'Loading checkpoint from directory: {ckpt_path}')
                state_dict = dict()
                safetensors_files = sorted([
                    f for f in os.listdir(ckpt_path)
                    if f.endswith('.safetensors') and f.startswith('model-')
                ])
                if not safetensors_files:
                    raise FileNotFoundError(
                        f'No model-*.safetensors files found in {ckpt_path}')
                for file in safetensors_files:
                    file_path = os.path.join(ckpt_path, file)
                    overwatch.info(f'  Loading {file}...')
                    state_dict.update(load_file(file_path, device='cpu'))
                overwatch.info(f'Loaded {len(safetensors_files)} safetensors files')
            elif ckpt_path.endswith('.safetensors'):
                # 单个 safetensors 文件
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                # .pt 文件
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint)

            # 应用 name_mapping（如果配置中有定义且需要）
            # 这对于加载官方预训练模型（如 GR00T-N1.5-3B）非常重要
            # 注意：name_mapping 通常在 cfg.model 中，而不是 cfg.inference_model
            #
            # 智能检测：只有当 checkpoint 的 key 与映射源前缀匹配时才应用映射
            # - 官方 GR00T-N1.5-3B：key 以 'backbone.eagle_model.*' 开头 → 需要映射
            # - FluxVLA 自训的 checkpoint：key 以 'vlm_backbone.vlm.*' 开头 → 不需要映射
            # 这样保证对 PI0.5/LIBERO/自训 groot 等所有管线向后兼容
            model_cfg = cfg.model if hasattr(cfg, 'model') else cfg.inference_model
            if 'name_mapping' in model_cfg and model_cfg['name_mapping']:
                # 检测 checkpoint 的 key 格式
                # model_cfg.name_mapping 的 value 是 checkpoint 的源前缀
                ckpt_prefixes = list(model_cfg['name_mapping'].values())
                needs_mapping = any(
                    any(k.startswith(p) for k in state_dict.keys())
                    for p in ckpt_prefixes
                )

                if needs_mapping:
                    overwatch.info(
                        'Detected checkpoint with external key format, '
                        'applying name_mapping...')
                    mapped_state_dict = {}
                    for model_key, ckpt_key in model_cfg['name_mapping'].items():
                        for k, v in state_dict.items():
                            if k.startswith(ckpt_key):
                                new_key = k.replace(ckpt_key, model_key, 1)
                                mapped_state_dict[new_key] = v
                    state_dict = mapped_state_dict
                    overwatch.info(
                        f'Applied name_mapping: {len(state_dict)} keys after mapping')
                else:
                    overwatch.info(
                        'Checkpoint already in native format, '
                        'skipping name_mapping')

            # Handle shared tensors (e.g., embed_tokens and lm_head in GR00T)
            from fluxvla.engines.utils.checkpoint_utils import handle_shared_tensors
            state_dict = handle_shared_tensors(
                state_dict, self.vla.state_dict(), overwatch)

            self.vla.load_state_dict(state_dict, strict=True)

        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.grouped_norm_stats = grouped_norm_stats
        self.norm_stats_group_names = norm_stats_group_names or []
        work_dir = Path(self.ckpt_path).resolve().parent.parent

        # --- 统计量: 显式 norm_stats_path、单文件 dataset_statistics.json 或分组统计 ---
        if norm_stats_path is not None and self.grouped_norm_stats:
            raise ValueError(
                'norm_stats_path cannot be used together with grouped_norm_stats')

        if self.grouped_norm_stats:
            assert len(self.norm_stats_group_names) == len(task_list), (
                'norm_stats_group_names 必须与 task_list 等长，一一对应')
            self._stats_full_by_group: Dict[str, dict] = {}
            self._denorm_by_group: Dict[str, Any] = {}
            for g in sorted(set(self.norm_stats_group_names)):
                gpath = work_dir / f'dataset_statistics_{g}.json'
                assert gpath.is_file(), (
                    f'[grouped_norm_stats] 缺少 {gpath}（分组训练后应由 '
                    f'save_grouped_dataset_statistics 生成）')
                with open(gpath, 'r', encoding='utf-8') as f:
                    self._stats_full_by_group[g] = json.load(f)
                assert unnorm_key in self._stats_full_by_group[g], (
                    f'{gpath} 中缺少 unnorm_key={unnorm_key!r}')
            for g in self._stats_full_by_group:
                da_cfg = copy.deepcopy(denormalize_action)
                da_cfg['norm_stats'] = self._stats_full_by_group[g]
                self._denorm_by_group[g] = build_transform_from_cfg(da_cfg)
            first_g = self.norm_stats_group_names[0]
            data_stat_path = str(work_dir / f'dataset_statistics_{first_g}.json')
            dataset['norm_stats'] = data_stat_path
            dataset['unnorm_key'] = unnorm_key
            self.dataset = build_dataset_from_cfg(dataset)
            self.denormalize_action = self._denorm_by_group[first_g]
        else:
            data_stat_path = (
                str(Path(norm_stats_path).expanduser().resolve())
                if norm_stats_path is not None
                else os.path.join(work_dir, 'dataset_statistics.json'))
            assert os.path.exists(data_stat_path), \
                f'dataset_statistics.json not found at {data_stat_path}'
            denormalize_action['norm_stats'] = data_stat_path
            dataset['norm_stats'] = data_stat_path
            dataset['unnorm_key'] = unnorm_key
            self.dataset = build_dataset_from_cfg(dataset)
            self.denormalize_action = build_transform_from_cfg(denormalize_action)

        if model_family == 'groot' and hasattr(self.dataset, 'transforms'):
            transform_names = [
                t.__class__.__name__ for t in self.dataset.transforms
            ]
            if 'PreparePromptWithState' in transform_names:
                raise RuntimeError(
                    'GR00T RoboCasa eval must not use PreparePromptWithState. '
                    'That is the PI0.5 text-state prompt and will make this '
                    f'evaluation invalid. Current transforms: {transform_names}')

        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        self.task_list = task_list
        self.max_episode_steps = max_episode_steps
        self.num_trials_per_task = num_trials_per_task
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.unnorm_key = unnorm_key
        self.distributed_state = overwatch.distributed_state

        self.save_video = save_video
        self.video_steps_per_render = video_steps_per_render
        self.video_fps = video_fps

        # 加载 norm_stats 到模型 (predict_action 可能需要)
        if self.grouped_norm_stats:
            first_g = self.norm_stats_group_names[0]
            self.vla.norm_stats = self._stats_full_by_group[first_g]
        elif os.path.isfile(data_stat_path):
            with open(data_stat_path, 'r') as f:
                self.vla.norm_stats = json.load(f)

        self._active_denorm = self.denormalize_action

    def run_setup(self):
        """初始化 GPU 和模型。"""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(self.device_id)
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        self.vla.cuda(self.device_id)

    def run(self):
        """执行 Robocasa 评测循环。"""
        import gymnasium as gym

        # 前置依赖健康检查 —— 避免新机器复现时被误导性的
        # `ImportError: cannot import name 'PandaOmron'` 。
        # 根因: robocasa 0.2.0 (GR1 tabletop) 硬要求 robosuite 1.5.x,
        # 但 fluxvla site-packages 里装的是 robosuite 1.4.1 (LIBERO 用)。
        # 正确做法见 docs/robocasa_docs_yiming/integration_logs/01_env_dependency.md:
        #   运行时需在命令前加 `PYTHONPATH=/root/projects/robosuite:$PYTHONPATH`
        #   让 Python 优先加载本地 1.5.1 源码。
        import robosuite as _robosuite_check
        if _robosuite_check.__version__ not in ("1.5.0", "1.5.1"):
            raise RuntimeError(
                f"[RobocasaEvalRunner] robosuite 版本不兼容: "
                f"当前 {_robosuite_check.__version__} (位于 {_robosuite_check.__file__}), "
                f"robocasa 要求 1.5.x。\n"
                f"请在启动命令前加: "
                f"PYTHONPATH=/root/projects/robosuite:$PYTHONPATH\n"
                f"详见 docs/robocasa_docs_yiming/integration_logs/01_env_dependency.md 第四节。"
            )

        # 触发 Robocasa gymnasium 环境注册
        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

        num_tasks = len(self.task_list)
        global_episodes = list(
            range(num_tasks * self.num_trials_per_task))

        overwatch.info(f'Robocasa Eval: {num_tasks} tasks, '
                       f'{self.num_trials_per_task} trials each')
        overwatch.info(f'Model family: {self.model_family}, '
                       f'chunk_size: {self.eval_chunk_size}')

        rank = overwatch.rank()
        world_size = overwatch.world_size()
        local_episodes = global_episodes[rank::world_size]
        num_local_episodes = math.ceil(len(global_episodes) / world_size)

        data_time = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = f'EVAL-robocasa-{self.model_family}-{data_time}'
        if self.output_dir is not None:
            work_dir = Path(self.output_dir).expanduser().resolve()
        else:
            work_dir = Path(self.ckpt_path).resolve().parent.parent
        work_dir.mkdir(parents=True, exist_ok=True)
        log_filepath = os.path.join(work_dir, run_id + '.txt')
        log_file = open(log_filepath, 'w')

        total_episodes = torch.zeros(1, device=torch.cuda.current_device())
        total_successes = torch.zeros(1, device=torch.cuda.current_device())

        pbar = None
        if rank == 0:
            pbar = tqdm.tqdm(total=len(global_episodes),
                             desc='Robocasa Eval', dynamic_ncols=True)

        for idx in range(num_local_episodes):
            if idx >= len(local_episodes):
                step_tensor = torch.zeros(
                    1, device=torch.cuda.current_device())
            else:
                local_id = local_episodes[idx]
                task_id = local_id // self.num_trials_per_task
                trial_id = local_id % self.num_trials_per_task
                env_name = self.task_list[task_id]

                overwatch.info(
                    f'Task {task_id} ({env_name}), Trial {trial_id}')
                log_file.write(
                    f'Task {task_id} ({env_name}), Trial {trial_id}\n')

                # --- 创建 Robocasa 环境 ---
                env = gym.make(env_name)
                obs, info = env.reset(seed=self.seed + local_id)

                # 分组统计: 每任务切换 state 归一化 blob / 反归一化器 / vla.norm_stats
                if self.grouped_norm_stats:
                    gname = self.norm_stats_group_names[task_id]
                    self.dataset.set_active_stats_blob(
                        self._stats_full_by_group[gname][self.unnorm_key])
                    self._active_denorm = self._denorm_by_group[gname]
                    self.vla.norm_stats = self._stats_full_by_group[gname]
                else:
                    if hasattr(self.dataset, 'set_active_stats_blob'):
                        self.dataset.set_active_stats_blob(None)
                    self._active_denorm = self.denormalize_action

                # 从 obs 中提取任务描述
                # 注意: 训练时 task_description 来自 tasks.jsonl 原文
                # (如 "unlocked_waist: pick up the bottled water, ...")
                # 评测必须保留整句，不能 split(':'), 否则 prompt 与训练不一致
                # 会导致模型预测的动作完全不匹配任务。
                task_desc = obs.get(
                    'annotation.human.coarse_action', '')
                overwatch.info(f'Task desc: {task_desc}')
                log_file.write(f'Task desc: {task_desc}\n')

                # --- 评测循环 ---
                success = False
                replay_images = []
                video_step_count = 1
                t = 0

                while t < self.max_episode_steps:
                    # 构造输入 dict 给 dataset transform
                    obs['task_description'] = task_desc
                    batch, _ = self.dataset(obs)
                    debug_info = getattr(self.dataset, 'last_debug', {})
                    if t == 0:
                        state_arr = batch['states'].detach().float().cpu().numpy() if 'states' in batch else None
                        prompt_preview = debug_info.get('text') or debug_info.get('prompt') or ''
                        overwatch.info(
                            f'Prompt preview: {prompt_preview[:240]}')
                        log_file.write(
                            f'Prompt preview: {prompt_preview[:240]}\n')
                        if state_arr is not None:
                            overwatch.info(
                                f'State range: min={state_arr.min():.6g}, max={state_arr.max():.6g}')
                            log_file.write(
                                f'State range: min={state_arr.min():.6g}, max={state_arr.max():.6g}\n')
                    batch['unnorm_key'] = self.unnorm_key

                    # --- 模型推理 ---
                    with torch.autocast(
                            'cuda', dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training):
                        with torch.no_grad():
                            actions = self.vla.predict_action(**batch)

                    # actions shape: (1, chunk_size, max_action_dim)
                    if len(actions.shape) == 3:
                        actions = actions[
                            0, :self.eval_chunk_size, :].cpu().numpy()
                    else:
                        actions = actions[0, None, :].cpu().numpy()

                    if t == 0:
                        overwatch.info(
                            f'Normalized action chunk: '
                            f'min={actions.min():.6g}, max={actions.max():.6g}, '
                            f'mean={actions.mean():.6g}')
                        log_file.write(
                            f'Normalized action chunk: '
                            f'min={actions.min():.6g}, max={actions.max():.6g}, '
                            f'mean={actions.mean():.6g}\n')

                    # --- 执行动作 chunk ---
                    for action in actions:
                        # 反归一化: [-1,1] → 原始关节角度
                        denorm_input = dict(
                            action=action,
                            task_suite_name=self.unnorm_key,
                        )
                        action_denormed = self._active_denorm(
                            denorm_input)

                        if t == 0:
                            overwatch.info(
                                f'Denorm action: '
                                f'min={action_denormed.min():.6g}, '
                                f'max={action_denormed.max():.6g}')
                            log_file.write(
                                f'Denorm action: '
                                f'min={action_denormed.min():.6g}, '
                                f'max={action_denormed.max():.6g}\n')

                        # 拆分 29 维 → Robocasa dict 格式
                        action_dict = {}
                        for key, (start, end) in \
                                ROBOCASA_ACTION_KEYS.items():
                            action_dict[key] = \
                                action_denormed[start:end]

                        # 环境执行
                        obs, reward, terminated, truncated, info = \
                            env.step(action_dict)

                        # 收集每步的图像帧用于视频录制
                        if self.save_video:
                            video_step_count += 1
                            if video_step_count % self.video_steps_per_render == 0:
                                frame = env.render()
                                if frame is not None:
                                    replay_images.append(frame.copy())

                        t += 1
                        if info.get('success', False):
                            success = True
                            break
                        if terminated or truncated:
                            break

                    if success or terminated or truncated:
                        break

                # --- 记录结果 ---
                if success:
                    total_successes += 1
                total_episodes += 1

                result_str = 'SUCCESS' if success else 'FAIL'
                overwatch.info(
                    f'  Result: {result_str} (steps={t})')
                log_file.write(
                    f'  Result: {result_str} (steps={t})\n')

                # --- 保存评测视频 ---
                if self.save_video and replay_images:
                    video_dir = os.path.join(work_dir, 'rollouts')
                    os.makedirs(video_dir, exist_ok=True)
                    task_short = env_name.split('/')[-1][:30]
                    video_path = os.path.join(
                        video_dir,
                        f'task{task_id}_trial{trial_id}_{result_str}_{task_short}.mp4')
                    try:
                        writer = imageio.get_writer(
                            video_path, fps=self.video_fps, codec='h264',
                            output_params=['-pix_fmt', 'yuv420p'])
                        for frame in replay_images:
                            if isinstance(frame, np.ndarray):
                                if frame.dtype != np.uint8:
                                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                                writer.append_data(frame)
                        writer.close()
                        overwatch.info(
                            f'  Video saved: {video_path} '
                            f'({len(replay_images)} frames)')
                    except Exception as e:
                        overwatch.warning(
                            f'  Video save failed: {e}')

                env.close()
                step_tensor = torch.ones(
                    1, device=torch.cuda.current_device())

            # --- 分布式同步 ---
            dist.barrier()
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(step_tensor.item()))

            global_episodes_done = total_episodes.clone()
            global_successes_done = total_successes.clone()
            dist.all_reduce(global_episodes_done, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_successes_done, op=dist.ReduceOp.SUM)

            if rank == 0:
                n_ep = int(global_episodes_done[0])
                n_succ = int(global_successes_done[0])
                rate = n_succ / n_ep * 100 if n_ep > 0 else 0
                overwatch.info(
                    f'Progress: {n_ep} episodes, '
                    f'{n_succ} successes ({rate:.1f}%)')
                log_file.write(
                    f'Progress: {n_ep} episodes, '
                    f'{n_succ} successes ({rate:.1f}%)\n')
                log_file.flush()

        dist.barrier()
        log_file.close()
        exit(0)
