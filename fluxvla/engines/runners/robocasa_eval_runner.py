# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RoboCasa simulation evaluation runner."""

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .base_eval_runner import BaseEvalRunner

overwatch = initialize_overwatch(__name__)

# Split RoboCasa GR1 29D actions into the dict format required by env.step.
# FluxVLA converted data order: left_arm + left_hand + right_arm + right_hand
# + waist.
ROBOCASA_FLUXVLA_ACTION_KEYS = {
    'action.left_arm': (0, 7),  # left arm, 7D
    'action.left_hand': (7, 13),  # left hand, 6D
    'action.right_arm': (13, 20),  # right arm, 7D
    'action.right_hand': (20, 26),  # right hand, 6D
    'action.waist': (26, 29),  # waist, 3D
}

# Official GR00T N1.5 fourier_gr1_arms_waist order:
# left_arm + right_arm + left_hand + right_hand + waist.
ROBOCASA_N15_ACTION_KEYS = {
    'action.left_arm': (0, 7),  # left arm, 7D
    'action.right_arm': (7, 14),  # right arm, 7D
    'action.left_hand': (14, 20),  # left hand, 6D
    'action.right_hand': (20, 26),  # right hand, 6D
    'action.waist': (26, 29),  # waist, 3D
}


@RUNNERS.register_module()
class RobocasaEvalRunner(BaseEvalRunner):
    """Runner for evaluating VLA models on Robocasa simulation tasks.

    Args:
        cfg: Full config object containing the model section.
        seed: Random seed.
        ckpt_path: Path to the model checkpoint.
        model_family: Model family name, such as ``pi0`` or ``groot``.
        task_list: RoboCasa Gymnasium environment names.
        dataset: Evaluation dataset config.
        denormalize_action: Action denormalization transform config.
        eval_chunk_size: Number of predicted actions executed per step.
        max_episode_steps: Maximum number of environment steps per episode.
        num_trials_per_task: Number of trials for each task.
        mixed_precision_dtype: Mixed precision dtype name.
        enable_mixed_precision_training: Whether autocast is enabled.
        unnorm_key: Top-level key in the dataset statistics.
        output_dir: Optional output directory for logs and videos.
        save_video: Whether to save rollout videos.
        action_order: Action split order. Defaults to ``n15`` for GR00T and
            ``fluxvla`` otherwise.
        norm_stats_path: Optional explicit dataset statistics path.
        grouped_norm_stats: Whether to load one statistics file per group.
        norm_stats_group_names: Per-task group names for grouped statistics.
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
                 norm_stats_path: Optional[str] = None,
                 action_order: Optional[str] = None,
                 grouped_norm_stats: bool = False,
                 norm_stats_group_names: Optional[List[str]] = None,
                 **kwargs):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg)

        self.device_id = overwatch.local_rank()

        # Build model.
        self.vla = self.build_eval_vla(cfg)

        # Load checkpoint weights.
        if ckpt_path is not None:
            assert Path(ckpt_path).exists(), \
                f'Checkpoint not found: {ckpt_path}'

            # Support checkpoint directories, single safetensors, and .pt
            # files.
            if os.path.isdir(ckpt_path):
                # Directory path: merge all sharded safetensors files.
                overwatch.info(
                    f'Loading checkpoint from directory: {ckpt_path}')
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
                overwatch.info(
                    f'Loaded {len(safetensors_files)} safetensors files')
            elif ckpt_path.endswith('.safetensors'):
                # Single safetensors file.
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                # PyTorch checkpoint.
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint)

            # Apply name_mapping only when the checkpoint uses source prefixes.
            # This is needed for official checkpoints such as GR00T-N1.5-3B.
            # name_mapping is usually defined in cfg.model, not
            # cfg.inference_model.
            #
            # Detection rules:
            # - official GR00T-N1.5-3B uses source prefixes and needs mapping
            # - FluxVLA-trained checkpoints already use model-native prefixes
            # This keeps PI0.5, LIBERO, and fine-tuned GR00T paths compatible.
            model_cfg = (
                cfg.model if hasattr(cfg, 'model') else cfg.inference_model)
            if 'name_mapping' in model_cfg and model_cfg['name_mapping']:
                # model_cfg.name_mapping keys are model-native prefixes, while
                # values are external checkpoint source prefixes. Fine-tuned
                # checkpoints are already native and must not be remapped.
                def _has_prefix(key: str, prefix: str) -> bool:
                    return key == prefix or key.startswith(f'{prefix}.')

                model_prefixes = list(model_cfg['name_mapping'].keys())
                ckpt_prefixes = list(model_cfg['name_mapping'].values())
                has_native_keys = any(
                    any(_has_prefix(k, p) for k in state_dict.keys())
                    for p in model_prefixes)
                needs_mapping = any(
                    any(_has_prefix(k, p) for k in state_dict.keys())
                    for p in ckpt_prefixes) and not has_native_keys

                if needs_mapping:
                    overwatch.info(
                        'Detected checkpoint with external key format, '
                        'applying name_mapping...')
                    mapped_state_dict = {}
                    for model_key, ckpt_key in \
                            model_cfg['name_mapping'].items():
                        for k, v in state_dict.items():
                            if _has_prefix(k, ckpt_key):
                                new_key = model_key + k[len(ckpt_key):]
                                mapped_state_dict[new_key] = v
                    state_dict = mapped_state_dict
                    overwatch.info(
                        f'Applied name_mapping: {len(state_dict)} keys after '
                        f'mapping')
                else:
                    overwatch.info('Checkpoint already in native format, '
                                   'skipping name_mapping')

            # Handle shared tensors (e.g., embed_tokens and lm_head in GR00T)
            from fluxvla.engines.utils.checkpoint_utils import \
                handle_shared_tensors
            state_dict = handle_shared_tensors(state_dict,
                                               self.vla.state_dict(),
                                               overwatch)

            self.vla.load_state_dict(state_dict, strict=True)

        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.grouped_norm_stats = grouped_norm_stats
        self.norm_stats_group_names = norm_stats_group_names or []
        work_dir = Path(self.ckpt_path).resolve().parent.parent

        # Statistics can come from an explicit path, a default file, or groups.
        if norm_stats_path is not None and self.grouped_norm_stats:
            raise ValueError('norm_stats_path cannot be used together with '
                             'grouped_norm_stats')

        if self.grouped_norm_stats:
            assert len(self.norm_stats_group_names) == len(task_list), (
                'norm_stats_group_names must have the same length as '
                'task_list')
            self._stats_full_by_group: Dict[str, dict] = {}
            self._denorm_by_group: Dict[str, Any] = {}
            for g in sorted(set(self.norm_stats_group_names)):
                gpath = work_dir / f'dataset_statistics_{g}.json'
                assert gpath.is_file(), (
                    f'[grouped_norm_stats] missing {gpath}. Grouped training '
                    f'should create it via save_grouped_dataset_statistics')
                with open(gpath, 'r', encoding='utf-8') as f:
                    self._stats_full_by_group[g] = json.load(f)
                assert unnorm_key in self._stats_full_by_group[g], (
                    f'{gpath} does not contain unnorm_key={unnorm_key!r}')
            for g in self._stats_full_by_group:
                da_cfg = copy.deepcopy(denormalize_action)
                da_cfg['norm_stats'] = self._stats_full_by_group[g]
                self._denorm_by_group[g] = build_transform_from_cfg(da_cfg)
            first_g = self.norm_stats_group_names[0]
            data_stat_path = str(work_dir /
                                 f'dataset_statistics_{first_g}.json')
            dataset['norm_stats'] = data_stat_path
            dataset['unnorm_key'] = unnorm_key
            self.dataset = build_dataset_from_cfg(dataset)
            self.denormalize_action = self._denorm_by_group[first_g]
        else:
            data_stat_path = (
                str(Path(norm_stats_path).expanduser().resolve())
                if norm_stats_path is not None else os.path.join(
                    work_dir, 'dataset_statistics.json'))
            assert os.path.exists(data_stat_path), \
                f'dataset_statistics.json not found at {data_stat_path}'
            denormalize_action['norm_stats'] = data_stat_path
            dataset['norm_stats'] = data_stat_path
            dataset['unnorm_key'] = unnorm_key
            self.dataset = build_dataset_from_cfg(dataset)
            self.denormalize_action = build_transform_from_cfg(
                denormalize_action)

        if model_family == 'groot' and hasattr(self.dataset, 'transforms'):
            transform_names = [
                t.__class__.__name__ for t in self.dataset.transforms
            ]
            if 'PreparePromptWithState' in transform_names:
                raise RuntimeError(
                    'GR00T RoboCasa eval must not use PreparePromptWithState. '
                    'That is the PI0.5 text-state prompt and will make this '
                    f'evaluation invalid. Current transforms: '
                    f'{transform_names}')

        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        if action_order is None:
            action_order = 'n15' if model_family == 'groot' else 'fluxvla'
        if action_order not in ('fluxvla', 'n15'):
            raise ValueError(f'Unsupported action_order={action_order}. '
                             "Expected 'fluxvla' or 'n15'.")
        self.action_keys = (
            ROBOCASA_N15_ACTION_KEYS
            if action_order == 'n15' else ROBOCASA_FLUXVLA_ACTION_KEYS)
        self.action_order = action_order
        self.task_list = task_list
        self.max_episode_steps = max_episode_steps
        self.num_trials_per_task = num_trials_per_task
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.unnorm_key = unnorm_key
        self.distributed_state = overwatch.distributed_state

        self.save_video = save_video

        # Attach norm_stats to the model for heads that consume them.
        if self.grouped_norm_stats:
            first_g = self.norm_stats_group_names[0]
            self.vla.norm_stats = self._stats_full_by_group[first_g]
        elif os.path.isfile(data_stat_path):
            with open(data_stat_path, 'r') as f:
                self.vla.norm_stats = json.load(f)

        self._active_denorm = self.denormalize_action

    def run_setup(self):
        """Initialize CUDA placement and model state."""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(self.device_id)
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        self.vla.cuda(self.device_id)

    def run(self):
        """Run the RoboCasa evaluation loop."""
        import gymnasium as gym
        # Fail early with a clear message when the wrong robosuite version is
        # imported. RoboCasa GR1 tabletop needs robosuite 1.5.x, while the
        # package installed for LIBERO may still be robosuite 1.4.1.
        import robosuite as _robosuite_check
        if _robosuite_check.__version__ not in ('1.5.0', '1.5.1'):
            raise RuntimeError(
                f'[RobocasaEvalRunner] incompatible robosuite version: '
                f'got {_robosuite_check.__version__} from '
                f'{_robosuite_check.__file__}. robocasa requires 1.5.x.\n'
                f'Please start with: '
                f'prepend /root/projects/robosuite to PYTHONPATH\n'
                f'See docs/robocasa_docs_yiming/integration_logs/'
                f'01_env_dependency.md for details.')

        # Trigger RoboCasa Gymnasium environment registration.
        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

        num_tasks = len(self.task_list)
        global_episodes = list(range(num_tasks * self.num_trials_per_task))

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
            pbar = tqdm.tqdm(
                total=len(global_episodes),
                desc='Robocasa Eval',
                dynamic_ncols=True)

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

                # Create RoboCasa environment.
                env = gym.make(env_name)
                obs, info = env.reset(seed=self.seed + local_id)

                # In grouped mode, switch stats, denormalizer, and
                # vla.norm_stats per task.
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

                # Keep the task description identical to the training text from
                # tasks.jsonl. Do not split prefixes such as
                # "unlocked_waist: ..."; otherwise the prompt distribution
                # shifts.
                task_desc = obs.get('annotation.human.coarse_action', '')
                overwatch.info(f'Task desc: {task_desc}')
                log_file.write(f'Task desc: {task_desc}\n')

                # Evaluation loop.
                success = False
                replay_images = []
                t = 0

                while t < self.max_episode_steps:
                    # Build input dict for the dataset transform pipeline.
                    obs['task_description'] = task_desc
                    batch, replay_img = self.dataset(obs)
                    debug_info = getattr(self.dataset, 'last_debug', {})
                    if t == 0:
                        state_arr = (
                            batch['states'].detach().float().cpu().numpy()
                            if 'states' in batch else None)
                        prompt_preview = (
                            debug_info.get('text') or debug_info.get('prompt')
                            or '')
                        overwatch.info(
                            f'Prompt preview: {prompt_preview[:240]}')
                        log_file.write(
                            f'Prompt preview: {prompt_preview[:240]}\n')
                        if state_arr is not None:
                            state_min = format(state_arr.min(), '.6g')
                            state_max = format(state_arr.max(), '.6g')
                            overwatch.info(f'State range: min={state_min}, '
                                           f'max={state_max}')
                            log_file.write(f'State range: min={state_min}, '
                                           f'max={state_max}\n')
                    batch['unnorm_key'] = self.unnorm_key
                    if replay_img is not None:
                        replay_images.append(replay_img)

                    # Model inference.
                    with torch.autocast(
                            'cuda',
                            dtype=self.mixed_precision_dtype,
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
                        action_min = format(actions.min(), '.6g')
                        action_max = format(actions.max(), '.6g')
                        action_mean = format(actions.mean(), '.6g')
                        overwatch.info(f'Normalized action chunk: '
                                       f'min={action_min}, '
                                       f'max={action_max}, '
                                       f'mean={action_mean}')
                        log_file.write(f'Normalized action chunk: '
                                       f'min={action_min}, '
                                       f'max={action_max}, '
                                       f'mean={action_mean}\n')

                    # Execute one action chunk.
                    for action in actions:
                        # Denormalize from [-1, 1] to raw joint positions.
                        denorm_input = dict(
                            action=action,
                            task_suite_name=self.unnorm_key,
                        )
                        action_denormed = self._active_denorm(denorm_input)

                        if t == 0:
                            denorm_min = format(action_denormed.min(), '.6g')
                            denorm_max = format(action_denormed.max(), '.6g')
                            overwatch.info(f'Denorm action: '
                                           f'min={denorm_min}, '
                                           f'max={denorm_max}')
                            log_file.write(f'Denorm action: '
                                           f'min={denorm_min}, '
                                           f'max={denorm_max}\n')

                        # Split 29D action into RoboCasa's dict action format.
                        action_dict = {}
                        for key, (start, end) in self.action_keys.items():
                            action_dict[key] = action_denormed[start:end]

                        # Step the environment.
                        obs, reward, terminated, truncated, info = \
                            env.step(action_dict)

                        # Collect rendered frames for rollout videos.
                        if self.save_video:
                            frame = obs.get('video.ego_view_pad_res256_freq20',
                                            None)
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

                # Record result.
                if success:
                    total_successes += 1
                total_episodes += 1

                result_str = 'SUCCESS' if success else 'FAIL'
                overwatch.info(f'  Result: {result_str} (steps={t})')
                log_file.write(f'  Result: {result_str} (steps={t})\n')

                # Save rollout video.
                if self.save_video and replay_images:
                    video_dir = os.path.join(work_dir, 'rollouts')
                    os.makedirs(video_dir, exist_ok=True)
                    task_short = env_name.split('/')[-1][:30]
                    video_name = (
                        f'task{task_id}_trial{trial_id}_{result_str}_'
                        f'{task_short}.mp4')
                    video_path = os.path.join(video_dir, video_name)
                    try:
                        writer = imageio.get_writer(
                            video_path,
                            fps=20,
                            codec='h264',
                            output_params=['-pix_fmt', 'yuv420p'])
                        for frame in replay_images:
                            if isinstance(frame, np.ndarray):
                                if frame.dtype != np.uint8:
                                    frame = (frame * 255).clip(0, 255).astype(
                                        np.uint8)
                                writer.append_data(frame)
                        writer.close()
                        overwatch.info(f'  Video saved: {video_path} '
                                       f'({len(replay_images)} frames)')
                    except Exception as e:
                        overwatch.warning(f'  Video save failed: {e}')

                env.close()
                step_tensor = torch.ones(1, device=torch.cuda.current_device())

            # Distributed synchronization.
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
                rate_text = format(rate, '.1f')
                overwatch.info(f'Progress: {n_ep} episodes, '
                               f'{n_succ} successes ({rate_text}%)')
                log_file.write(f'Progress: {n_ep} episodes, '
                               f'{n_succ} successes ({rate_text}%)\n')
                log_file.flush()

        dist.barrier()
        log_file.close()
        exit(0)
