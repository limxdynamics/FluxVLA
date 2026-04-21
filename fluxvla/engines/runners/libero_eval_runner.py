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

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from libero.libero import benchmark
from safetensors.torch import load_file

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.eval_utils import (get_libero_dummy_action,
                                              get_libero_env,
                                              save_rollout_video)
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class LiberoEvalRunner:
    """Runner for evaluating models using Hugging Face Transformers.
    This class sets up the evaluation environment, loads the model,
    and runs the evaluation process.
    Args:
        cfg (Dict): Configuration dictionary containing model and
            evaluation settings.
        seed (int): Random seed for reproducibility.
        ckpt_path (str): Path to the model checkpoint.
        model_family (str): Model family for evaluation.
        task_suite_name (str): Name of the task suite for evaluation.
        dataset (Dict): Configuration for the dataset to be used in evaluation.
        denormalize_action (Dict): Configuration for denormalizing actions.
        eval_chunk_size (int): Size of the chunks for evaluation.
            Default is 1.
        resize_size (int): Size to which images will be resized.
            Default is 224.
        num_trials_per_task (int): Number of trials per task in the evaluation.
            Default is 50.
        num_steps_wait (int): Number of steps to wait before
            starting evaluation.
            Default is 10.
        mixed_precision_dtype (str): Data type for mixed precision training.
            Default is 'bf16'.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
            Default is True.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_suite_name: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 eval_chunk_size: int = 1,
                 resize_size: int = 224,
                 num_trials_per_task: int = 50,
                 num_steps_wait: int = 10,
                 task_ids: Optional[List[int]] = None,
                 task_horizons: Dict = None,
                 use_xvla_client_semantics: bool = False,
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)
        self.device_id = overwatch.local_rank()
        if hasattr(cfg, 'inference_model'):
            self.vla = build_vla_from_cfg(cfg.inference_model).eval()
        else:
            self.vla = build_vla_from_cfg(cfg.model).eval()
        # Load checkpoint weights if ckpt_path is provided
        if ckpt_path is not None:
            assert Path.exists(Path(ckpt_path)), \
                f'Checkpoint path {ckpt_path} does not exist!'

            if ckpt_path.endswith('.safetensors'):
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            self.vla.load_state_dict(state_dict, strict=True)
        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        data_stat_path = os.path.join(
            Path(self.ckpt_path).resolve().parent.parent,
            'dataset_statistics.json')  # noqa: E501
        assert os.path.exists(data_stat_path), \
            f'Dataset statistics file not found at {data_stat_path}!'
        # Load dataset and denormalization action
        denormalize_action['norm_stats'] = data_stat_path
        dataset['task_suite_name'] = task_suite_name
        dataset['norm_stats'] = data_stat_path
        self.dataset = build_dataset_from_cfg(dataset)
        self.denormalize_action = build_transform_from_cfg(denormalize_action)
        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        self.task_suite_name = task_suite_name
        self.resize_size = resize_size
        self.num_trials_per_task = num_trials_per_task
        self.num_steps_wait = num_steps_wait
        self.task_ids = None if task_ids is None else sorted(
            {int(task_id)
             for task_id in task_ids})
        self.task_horizons = task_horizons or {}
        self.use_xvla_client_semantics = use_xvla_client_semantics
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.distributed_state = overwatch.distributed_state

        if os.path.isfile(data_stat_path):
            with open(data_stat_path, 'r') as f:
                norm_stats = json.load(f)
            self.vla.norm_stats = norm_stats
        else:
            overwatch.warning(
                'WARNING: No local dataset_statistics.json file found for current checkpoint.\n'  # noqa: E501
                'You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.'  # noqa: E501
                'Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`.'  # noqa: E501
            )

    @staticmethod
    def _mat_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
        return np.concatenate([rot_mat[:3, 0], rot_mat[:3, 1]],
                              axis=-1).astype(np.float32)

    @classmethod
    def _build_closed_loop_state(cls, env) -> np.ndarray:
        ee_pos = np.asarray(env.env.robots[0].controller.ee_pos,
                            dtype=np.float32)
        ee_rot6d = cls._mat_to_rot6d(
            np.asarray(env.env.robots[0].controller.ee_ori_mat,
                       dtype=np.float32))
        arm = np.concatenate([ee_pos, ee_rot6d, np.array([0.0],
                              dtype=np.float32)],
                             axis=-1)
        state = np.zeros((20,), dtype=np.float32)
        state[:10] = arm
        return state

    def run_setup(self):
        """Set up the evaluation environment and model."""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(device_id := self.device_id)  # noqa: F841
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        if self.enable_mixed_precision_training:
            self.vla.to(
                device=self.device_id, dtype=self.mixed_precision_dtype)
        else:
            self.vla.cuda(self.device_id)

    def run(self):
        """Run the evaluation process."""
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        if self.task_ids is None:
            eval_task_ids = list(range(num_tasks_in_suite))
        else:
            invalid = [
                task_id for task_id in self.task_ids
                if task_id < 0 or task_id >= num_tasks_in_suite
            ]
            if invalid:
                raise ValueError(
                    f'Invalid task_ids for {self.task_suite_name}: {invalid}; '
                    f'valid range is [0, {num_tasks_in_suite - 1}]')
            eval_task_ids = self.task_ids
        global_episodes = [
            task_id * self.num_trials_per_task + trial_id
            for task_id in eval_task_ids
            for trial_id in range(self.num_trials_per_task)
        ]
        overwatch.info(f'Task suite: {self.task_suite_name}')
        overwatch.info(
            f'Running evaluation on {len(eval_task_ids)} task(s) '
            f'with {self.num_trials_per_task} trials each.')
        overwatch.info(f'Evaluated task IDs: {eval_task_ids}')
        overwatch.info(f'Using model family: {self.model_family}')
        overwatch.info(f'Using resize size: {self.resize_size}')
        overwatch.info(f'Using evaluation chunk size: {self.eval_chunk_size}')
        overwatch.info(
            f'Using mixed precision dtype: {self.mixed_precision_dtype}')
        rank = overwatch.rank()
        world_size = overwatch.world_size()
        local_episodes = global_episodes[rank::world_size]
        num_local_episodes = math.ceil(len(global_episodes) / world_size)
        data_time = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = f'EVAL-{self.task_suite_name}-{self.model_family}-{data_time}'  # noqa: E501
        local_log_filepath = os.path.join(
            Path(self.ckpt_path).resolve().parent.parent, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        total_episodes, total_successes = torch.zeros(
            1, device=torch.cuda.current_device()), torch.zeros(
                1, device=torch.cuda.current_device())
        unnorm_key = self.task_suite_name
        if rank == 0:
            pbar = tqdm.tqdm(
                total=len(global_episodes),
                desc='Evaluation',
                dynamic_ncols=True)
        else:
            pbar = None
        if self.model_family == 'openvla':
            # In some cases, the key must be manually modified (e.g. after
            # training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            candidate_unnorm_key = f'{unnorm_key}_no_noops'
            if (unnorm_key not in self.vla.norm_stats
                    and candidate_unnorm_key in self.vla.norm_stats):
                unnorm_key = candidate_unnorm_key
            assert unnorm_key in self.vla.norm_stats, (
                f'Action un-norm key {unnorm_key} '
                'not found in VLA norm_stats!')
        for id in range(num_local_episodes):
            if id >= len(local_episodes):
                step_tensor = torch.zeros(
                    1, device=torch.cuda.current_device())
            else:
                local_id = local_episodes[id]
                # Get task ID from local episode index
                task_id = local_id // self.num_trials_per_task
                # Get trial ID within the task
                trial_id = local_id % self.num_trials_per_task

                # Log the current task and trial
                overwatch.info(f'Evaluating Task {task_id}, Trial {trial_id}')
                log_file.write(
                    f'Evaluating Task {task_id}, Trial {trial_id}\n')

                # Initialize the task suite and environment
                # Get task
                task = task_suite.get_task(task_id)

                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)

                # Initialize LIBERO environment and task description
                env, task_description = get_libero_env(task, resolution=256)
                overwatch.info(f'\nTask: {task_description}')
                log_file.write(f'\nTask: {task_description}\n')

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[trial_id])
                if self.use_xvla_client_semantics:
                    for robot in env.env.robots:
                        robot.controller.use_delta = False
                is_new_episode = True

                # Setup
                t = 0
                replay_images = []
                closed_loop_state = None
                if self.task_suite_name in self.task_horizons:
                    max_steps = self.task_horizons[self.task_suite_name]
                elif self.task_suite_name == 'libero_spatial':
                    max_steps = 220  # longest training demo has 193 steps
                elif self.task_suite_name == 'libero_object':
                    max_steps = 280  # longest training demo has 254 steps
                elif self.task_suite_name == 'libero_goal':
                    max_steps = 300  # longest training demo has 270 steps
                elif self.task_suite_name == 'libero_10':
                    max_steps = 520  # longest training demo has 505 steps
                elif self.task_suite_name == 'libero_90':
                    max_steps = 400  # longest training demo has 373 steps

                overwatch.info(f'Starting episode {trial_id+1}...')

                log_file.write(f'Starting episode {trial_id+1}...\n')
                while t < max_steps + self.num_steps_wait:
                    # IMPORTANT: Do nothing for the first
                    # few timesteps
                    # because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < self.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action())
                        t += 1
                        continue
                    obs['task_description'] = task_description
                    obs['is_new_episode'] = is_new_episode
                    batch, replay_img = self.dataset(obs)
                    is_new_episode = False
                    if self.use_xvla_client_semantics:
                        if closed_loop_state is None:
                            closed_loop_state = self._build_closed_loop_state(
                                env)
                        batch['states'] = torch.from_numpy(
                            closed_loop_state).to(
                                device=batch['images'].device,
                                dtype=torch.bfloat16).unsqueeze(0)
                    batch['unnorm_key'] = unnorm_key
                    if len(replay_images) == 0:
                        replay_images.append(replay_img)
                    with torch.autocast(
                            'cuda',
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training):
                        with torch.no_grad():
                            actions = self.vla.predict_action(**batch)
                    if len(actions.shape) == 3:
                        actions = actions[
                            0, :self.eval_chunk_size, :].float().cpu().numpy()
                    else:
                        assert len(actions.shape) == 2, \
                            f'Unexpected action shape: {actions.shape}'
                        actions = actions[0, None, :].float().cpu().numpy()
                    for action in actions:
                        inputs = dict(
                            action=action,
                            task_suite_name=self.task_suite_name,
                        )
                        action_denormed = self.denormalize_action(inputs)
                        if self.use_xvla_client_semantics:
                            closed_loop_state[:9] = action[:9].astype(
                                np.float32)
                        obs, reward, done, info = env.step(
                            action_denormed.tolist())
                        obs['task_description'] = task_description
                        batch, replay_img = self.dataset(obs)
                        replay_images.append(replay_img)
                        if done:
                            total_successes += 1
                            break
                        t += 1
                    if done:
                        break
                total_episodes += 1
                step_tensor = torch.ones(1, device=torch.cuda.current_device())
                # Save a replay video of the episode
                save_rollout_video(
                    replay_images,
                    local_id,
                    success=done,
                    task_description=task_description,
                    work_dir=Path(self.ckpt_path).resolve().parent.parent,
                    log_file=log_file)
                env.close()

                # except Exception as e:
                #     print(f'Error during action prediction: {e}')
                #     log_file.write(f'Caught exception: {e}\n')
                #     action = get_libero_dummy_action()
            dist.barrier()
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(step_tensor.item()))

            global_episodes = total_episodes.clone()
            global_successes = total_successes.clone()
            dist.all_reduce(global_episodes, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_successes, op=dist.ReduceOp.SUM)
            done = done.item() if isinstance(done, torch.Tensor) else done
            if rank == 0:
                # Log current results
                overwatch.info(
                    f'# episodes completed so far: {int(global_episodes[0])}')
                success_rate = (global_successes[0] / global_episodes[0] * 100)
                success_text = (f'# successes: {int(global_successes[0])} '
                                f'({success_rate:.1f}%)')  # noqa: E231
                overwatch.info(success_text)
                log_file.write(f'Success: {done}\n')
                log_file.write(
                    f'# episodes completed so far: {global_episodes[0]}\n')
                success_log = (f'# successes: {global_successes[0]} '
                               f'({success_rate:.1f}%)\n')  # noqa: E231
                log_file.write(success_log)
                log_file.flush()
        dist.barrier()
        exit(0)
