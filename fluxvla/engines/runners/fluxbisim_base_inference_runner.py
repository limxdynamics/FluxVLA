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

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils import build_operator_from_cfg, initialize_overwatch
from ..utils.name_map import str_to_dtype

overwatch = initialize_overwatch(__name__)


class BaseInferenceRunnerSim:
    """Base class for simulation robot inference runners.

    Provides model setup, observation handling, task management, and a
    template inference loop with four overridable phases:

        _preprocess  → observe + build model inputs
        _predict_action  → model inference (normalized actions)
        _postprocess_actions → denormalize to robot command space
        _execute_actions → send commands to the robot (abstract)

    Unlike the real-robot ``BaseInferenceRunner``, the simulation
    environment only advances after each action is executed, so there
    is no need for ROS rate limiting, RTC, or async trajectory
    execution.  Episodes are driven by environment reset signals
    rather than interactive user input.

    Subclasses should implement robot-specific methods like
    get_observation and _execute_actions. Override other phases as
    needed.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 task_suite_name: str = 'private',
                 state_dim: int = 7,
                 action_chunk: int = 32,
                 publish_rate: int = 30,
                 max_publish_step: int = 10000,
                 use_robot_base: bool = False,
                 disable_puppet_arm: bool = False,
                 camera_names: Optional[List[str]] = None,
                 operator: Dict = None,
                 task_descriptions: Dict = None,
                 mixed_precision_dtype: str = 'float32',
                 enable_mixed_precision: bool = True,
                 smooth_alpha: float = 0.8):
        """Initialize the base simulation inference runner.

        Args:
            cfg (Dict): Configuration dictionary for the VLA model
            seed (int): Random seed for reproducibility
            ckpt_path (str): Path to model checkpoint file
            dataset (Dict): Dataset configuration dictionary
            denormalize_action (Dict): Action denormalization configuration
            task_suite_name (str, optional): Name of task suite.
                Defaults to 'private'.
            state_dim (int, optional): Dimension of robot state vector.
                Defaults to 7.
            action_chunk (int, optional): Number of actions to predict at once.
                Defaults to 32.
            publish_rate (int, optional): Simulation control rate in Hz.
                Defaults to 30.
            max_publish_step (int, optional): Maximum steps per episode.
                Defaults to 10000.
            use_robot_base (bool, optional): Whether to use mobile base.
                Defaults to False.
            disable_puppet_arm (bool, optional): Whether to disable puppet arm.
                Defaults to False.
            camera_names (List[str], optional): Names of camera feeds.
                Defaults to None.
            operator (Dict, optional): Simulation operator configuration.
                If None, uses default operator configuration.
            task_descriptions (Dict, optional): Task descriptions mapping.
                If None, uses empty dict.
            mixed_precision_dtype (str, optional): Data type string for
                mixed-precision inference. Defaults to 'float32'.
            enable_mixed_precision (bool, optional): Whether to enable
                mixed-precision inference. Defaults to True.
            smooth_alpha (float, optional): Exponential smoothing factor
                for action chunks. ``1.0`` disables smoothing and smaller
                values produce smoother but slower-changing actions.
                Defaults to 1.0.

        Raises:
            AssertionError: If dataset statistics file is not found
        """
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)

        self.ckpt_path = ckpt_path
        data_stat_path = os.path.join(
            Path(ckpt_path).resolve().parent.parent, 'dataset_statistics.json')
        assert os.path.exists(data_stat_path), (
            f'Dataset statistics file not found at {data_stat_path}!')

        denormalize_action['norm_stats'] = data_stat_path
        dataset['norm_stats'] = data_stat_path
        dataset['model_path'] = os.path.dirname(os.path.dirname(ckpt_path))

        self.dataset = build_dataset_from_cfg(dataset)
        self.denormalize_action = build_transform_from_cfg(denormalize_action)

        self.vla = build_vla_from_cfg(cfg.inference_model)
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

        self.seed = seed
        self.state_dim = state_dim
        self.action_chunk = action_chunk
        self.publish_rate = publish_rate
        self.max_publish_step = max_publish_step
        self.use_robot_base = use_robot_base
        self.disable_puppet_arm = disable_puppet_arm
        self.camera_names = camera_names or []
        self.task_suite_name = task_suite_name

        self.ros_operator = build_operator_from_cfg(operator)
        self.observation_window = None

        self.task_descriptions = task_descriptions or {}

        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision = enable_mixed_precision
        self.smooth_alpha = smooth_alpha
        self._last_action = None

    def _apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Apply JPEG compression and decompression to image.

        This transformation aligns the inference images with training data
        by applying the same JPEG compression artifacts that may have been
        present during dataset collection.

        Args:
            img (np.ndarray): Input BGR image array

        Returns:
            np.ndarray: JPEG-processed BGR image array
        """
        encoded_img = cv2.imencode('.jpg', img)[1].tobytes()
        decoded_img = cv2.imdecode(
            np.frombuffer(encoded_img, np.uint8), cv2.IMREAD_COLOR)
        return decoded_img

    def _get_task_description(self, task_id: str) -> str:
        """Get task description for given task ID.

        Args:
            task_id (str): Task identifier string

        Returns:
            str: Human-readable task description
        """
        return self.task_descriptions.get(task_id, '')

    def run_setup(self):
        """Set up the inference environment.

        Configures the model for evaluation mode, moves it to GPU,
        and sets random seeds for reproducibility.
        """
        set_seed_everywhere(self.seed)
        self.vla.eval()
        if self.enable_mixed_precision:
            self.vla.to(device='cuda', dtype=self.mixed_precision_dtype)
        else:
            self.vla.cuda()
        overwatch.info(f'Model loaded and moved to GPU '
                       f'(dtype={self.mixed_precision_dtype}). '
                       f'Seed set to {self.seed}')

    def run(self,
            initial_instruction:
            str = 'place it in the brown paper bag with right arm'):
        """Run the main inference loop.

        Executes continuous robotic manipulation tasks based on
        vision-language instructions. Unlike the real-robot runner,
        episodes are separated by environment reset signals from the
        simulation rather than interactive user input.

        Args:
            initial_instruction (str, optional): Default task instruction.
                Defaults to 'place it in the brown paper bag with right arm'.

        Note:
            This method runs indefinitely until ROS shutdown is requested.
            After each episode, it waits for a simulation reset signal
            before starting the next one.
        """
        import rospy

        instruction = self._get_user_task_instruction(initial_instruction)
        overwatch.info(f'Starting sim inference: "{instruction}"')

        with torch.inference_mode():
            while not rospy.is_shutdown():
                success = self._run_episode(instruction)
                if not success:
                    overwatch.info('Episode failed, waiting for environment '
                                   'reset before next episode')
                self._wait_for_env_reset()

    def _run_episode(self, instruction: str) -> bool:
        """Run a single episode: preprocess → predict → postprocess → execute.

        Subclasses should override individual phases rather than this
        method.

        Args:
            instruction (str): Task instruction to use for this episode

        Returns:
            bool: True if task completed successfully (env reset received
                before max steps), False if max steps reached without
                receiving env reset.
        """
        import rospy

        t = 0
        self.observation_window = None
        self._reset_action_smoothing()

        overwatch.info(f'Episode start: {instruction}')

        while t < self.max_publish_step and not rospy.is_shutdown():
            if self._check_env_reset():
                overwatch.info(f'Environment reset detected, '
                               f'episode completed after {t} steps')
                return True

            inputs = self._preprocess(instruction)

            with torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision):
                raw_action = self._predict_action(inputs)

            actions = self._postprocess_actions(raw_action)
            self._execute_actions(actions)

            t += self.action_chunk
            overwatch.info(f'Step {t}')

        return False

    def _check_env_reset(self):
        """Check whether the operator received an environment reset signal.

        Returns:
            bool: True if a reset signal has been received, False
                otherwise.  Always returns False if the operator does
                not implement ``check_env_reset``.
        """
        return (hasattr(self.ros_operator, 'check_env_reset')
                and self.ros_operator.check_env_reset())

    def _wait_for_env_reset(self):
        """Clear the current reset flag and block until the next one.

        Polls the operator at 10 Hz until a new environment reset
        signal arrives from the simulation.
        """
        import rospy

        overwatch.info('Waiting for environment reset to start next episode…')

        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            if self._check_env_reset():
                overwatch.info('Environment reset received, '
                               'starting new episode after 1 second...')
                if hasattr(self.ros_operator, 'clear_env_reset_flag'):
                    self.ros_operator.clear_env_reset_flag()
                rospy.sleep(1.0)
                return
            rate.sleep()

    # ---- Pipeline phases (override in subclass as needed) ----

    def _preprocess(self, instruction: str) -> dict:
        """Observe environment and build model inputs.

        Args:
            instruction (str): Task description for this chunk.

        Returns:
            dict: Model-ready inputs from the dataset transform.
        """
        obs = self.update_observation_window()
        obs['task_description'] = instruction
        return self.dataset(obs)

    def _predict_action(self, inputs: dict):
        """Run model inference to produce normalized actions.

        Override to add timing or other prediction logic.

        Args:
            inputs (dict): Model inputs from _preprocess.

        Returns:
            Tensor: Normalized action tensor.
        """
        return self.vla.predict_action(**inputs)

    def _postprocess_actions(self, raw_action):
        """Denormalize raw actions into robot command space.

        Override to add trajectory stitching, smoothing, etc.

        Args:
            raw_action: Normalized action tensor from _predict_action.

        Returns:
            np.ndarray: Denormalized actions, truncated to action_chunk.
        """
        denormalized = self.denormalize_action(
            dict(action=raw_action.cpu().numpy()))
        actions = np.asarray(
            denormalized[:self.action_chunk], dtype=np.float32)
        return self._smooth_action_chunk(actions)

    def _reset_action_smoothing(self):
        """Reset cross-chunk smoothing state for a new episode."""
        self._last_action = None

    def _get_unsmoothed_action_dims(self, action_dim: int):
        """Return action dimensions that should bypass smoothing."""
        del action_dim
        return ()

    def _smooth_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        """Smooth an action chunk using the previous action as context."""
        if actions.size == 0:
            return actions

        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        smooth_alpha = np.clip(self.smooth_alpha, 0.0, 1.0)
        if smooth_alpha >= 1.0:
            self._last_action = actions[-1].copy()
            return actions

        if (self._last_action is None
                or self._last_action.shape != actions[0].shape):
            self._last_action = actions[0].copy()

        smoothed = np.empty_like(actions)
        smoothed[:] = actions

        unsmoothed_dims = tuple(
            self._get_unsmoothed_action_dims(actions.shape[1]))
        smoothed_mask = np.ones(actions.shape[1], dtype=bool)
        if unsmoothed_dims:
            valid_dims = [
                dim for dim in unsmoothed_dims if 0 <= dim < actions.shape[1]
            ]
            smoothed_mask[valid_dims] = False

        if not np.any(smoothed_mask):
            self._last_action = actions[-1].copy()
            return actions

        smoothed[0, smoothed_mask] = (
            smooth_alpha * actions[0, smoothed_mask] +
            (1.0 - smooth_alpha) * self._last_action[smoothed_mask])
        for i in range(1, actions.shape[0]):
            smoothed[i, smoothed_mask] = (
                smooth_alpha * actions[i, smoothed_mask] +
                (1.0 - smooth_alpha) * smoothed[i - 1, smoothed_mask])

        self._last_action = smoothed[-1].copy()
        return smoothed

    def _get_user_task_instruction(self, default_instruction: str) -> str:
        """Get task instruction from user input.

        Args:
            default_instruction (str): Default instruction if no valid input

        Returns:
            str: Task instruction string
        """
        task_id = input('Enter task ID (or press Enter for default): ').strip()
        if task_id and task_id in self.task_descriptions:
            return self._get_task_description(task_id)
        return default_instruction

    # ---- Abstract methods that subclasses must implement ----

    def get_observation(self):
        """Get synchronized observation data from simulation topics.

        This method should be implemented by subclasses to handle
        robot-specific observation collection.

        Returns:
            Tuple: Robot-specific observation data
        """
        raise NotImplementedError(
            'Subclasses must implement get_observation method')

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        This method should be implemented by subclasses to handle
        robot-specific observation window management.

        Returns:
            Dict: Latest observation data
        """
        raise NotImplementedError(
            'Subclasses must implement update_observation_window method')

    def _execute_actions(self, actions: np.ndarray):
        """Execute a sequence of robot actions.

        This method should be implemented by subclasses to handle
        robot-specific action execution.  Unlike the real-robot
        runner, this method does not receive a ROS rate limiter
        because the simulation steps synchronously.

        Args:
            actions (np.ndarray): Array of denormalized robot actions
        """
        raise NotImplementedError(
            'Subclasses must implement _execute_actions method')

    def cleanup(self):
        """Clean up resources and shutdown gracefully."""
        overwatch.info('Cleaning up BaseInferenceRunnerSim')
        self._reset_action_smoothing()
        if self.observation_window is not None:
            self.observation_window.clear()
