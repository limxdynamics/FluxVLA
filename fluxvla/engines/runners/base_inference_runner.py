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
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils import build_operator_from_cfg, initialize_overwatch
from ..utils.name_map import str_to_dtype

overwatch = initialize_overwatch(__name__)


class BaseInferenceRunner:
    """Base class for robot inference runners.

    Provides model setup, observation handling, task management, and a
    template inference loop with four overridable phases:

        _preprocess  → observe + build model inputs
        _predict_action  → model inference (normalized actions)
        _postprocess_actions → denormalize to robot command space
        _execute_actions → send commands to the robot (abstract)

    Subclasses should implement robot-specific methods like get_ros_observation
    and _execute_actions. Override other phases as needed.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: str,
                 ckpt_path: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 task_suite_name: str = 'private',
                 state_dim: int = 7,
                 action_chunk: int = 32,
                 publish_rate: int = 30,
                 max_publish_step: int = 10000,
                 use_eval_collector: bool = False,
                 use_robot_base: bool = False,
                 disable_puppet_arm: bool = False,
                 camera_names: Optional[List[str]] = None,
                 operator: Dict = None,
                 task_descriptions: Dict = None,
                 task_pose_sequences: Dict = None,
                 mixed_precision_dtype: str = 'float32',
                 enable_mixed_precision: bool = True):
        """Initialize the base inference runner.

        Args:
            cfg (Dict): Configuration dictionary for the VLA model
            seed (str): Random seed for reproducibility
            ckpt_path (str): Path to model checkpoint file
            dataset (Dict): Dataset configuration dictionary
            denormalize_action (Dict): Action denormalization configuration
            task_suite_name (str, optional): Name of task suite.
                Defaults to 'private'.
            state_dim (int, optional): Dimension of robot state vector.
                Defaults to 7.
            action_chunk (int, optional): Number of actions to predict at once.
                Defaults to 32.
            publish_rate (int, optional): ROS publishing rate in Hz.
                Defaults to 30.
            max_publish_step (int, optional): Maximum steps per episode.
                Defaults to 10000.
            use_eval_collector (bool, optional): Whether to use evaluation
                data collector. Defaults to False.
            use_robot_base (bool, optional): Whether to use mobile base.
                Defaults to False.
            disable_puppet_arm (bool, optional): Whether to disable puppet arm.
                Defaults to False.
            camera_names (List[str], optional): Names of camera feeds.
                Defaults to None.
            operator (Dict, optional): ROS operator configuration.
                If None, uses default operator configuration.
            task_descriptions (Dict, optional): Task descriptions mapping.
                If None, uses empty dict.
            task_pose_sequences (Dict, optional): Task pose sequences mapping.
                If None, uses empty dict.

        Raises:
            AssertionError: If dataset statistics file is not found
        """
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)

        # Initialize paths and validate dataset statistics
        self.ckpt_path = ckpt_path
        data_stat_path = os.path.join(
            Path(ckpt_path).resolve().parent.parent, 'dataset_statistics.json')
        assert os.path.exists(data_stat_path), (
            f'Dataset statistics file not found at {data_stat_path}!')

        # Configure dataset and denormalization
        denormalize_action['norm_stats'] = data_stat_path
        dataset['norm_stats'] = data_stat_path
        dataset['model_path'] = os.path.dirname(os.path.dirname(ckpt_path))

        # Build components
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

        # Store configuration parameters
        self.seed = seed
        self.state_dim = state_dim
        self.action_chunk = action_chunk
        self.publish_rate = publish_rate
        self.max_publish_step = max_publish_step
        self.use_eval_collector = use_eval_collector
        self.use_robot_base = use_robot_base
        self.disable_puppet_arm = disable_puppet_arm
        self.camera_names = camera_names or []
        self.task_suite_name = task_suite_name

        # Initialize ROS operator and observation window
        self.ros_operator = build_operator_from_cfg(operator)
        self.observation_window = None

        # Initialize task configurations
        self.task_descriptions = task_descriptions or {}
        self.task_pose_sequences = task_pose_sequences or {}

        # Mixed precision settings
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision = enable_mixed_precision

        # Action context: SimpleNamespace shared between _predict_action,
        # _postprocess_actions, and _execute_actions within one iteration.
        # Becomes _prev_ctx in the next iteration for cross-chunk continuity.
        self._prev_ctx = None
        self._action_ctx = SimpleNamespace()

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
        return self.task_descriptions.get(
            task_id, 'place it in the brown paper bag with right arm')

    def execute_task_pose(self, task_id: str):
        """Execute pose sequence for a specific task.

        Args:
            task_id (str): Task identifier string

        Note:
            This is a base implementation that does nothing.
            Subclasses should override this method to implement
            robot-specific pose execution.
        """
        if task_id in self.task_pose_sequences:
            overwatch.info(f'Executing pose sequence for task {task_id}')
            # Base implementation - subclasses should override
            pass

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

        Executes continuous robotic manipulation tasks based on vision-language
        instructions. The loop handles task selection, action prediction,
        and robot control with proper error handling and user interaction.

        Args:
            initial_instruction (str, optional): Default task instruction.
                Defaults to 'place it in the brown paper bag with right arm'.

        Note:
            This method runs indefinitely until ROS shutdown is requested.
            It provides interactive task selection and automatic robot
            control based on VLA model predictions.
        """
        import rospy

        overwatch.info('Starting inference runner')

        # Main inference loop
        with torch.inference_mode():
            while not rospy.is_shutdown():
                self._run_episode(initial_instruction)

    def _run_episode(self, default_instruction: str):
        """Run a single episode: preprocess → predict → postprocess → execute.

        Subclasses should override individual phases rather than this method.

        Args:
            default_instruction (str): Default task instruction to use
        """
        import rospy

        t = 0
        rate = rospy.Rate(self.publish_rate)

        while t < self.max_publish_step and not rospy.is_shutdown():
            instructions = self._get_user_task_instruction(default_instruction)
            self._prev_ctx = None
            for instruction in instructions:
                self._action_ctx = SimpleNamespace()
                self._action_ctx.instruction = instruction
                inputs = self._preprocess(instruction)

                with torch.autocast(
                        'cuda',
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision):
                    raw_action = self._predict_action(inputs)

                actions = self._postprocess_actions(raw_action)
                self._execute_actions(actions, rate)

                self._prev_ctx = self._action_ctx
                t += self.action_chunk
                overwatch.info(f'Published Step {t}')

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

        Override to add RTC guidance, timing, or other prediction logic.

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
        return denormalized[:self.action_chunk]

    def _get_user_task_instruction(self, default_instruction: str) -> str:
        """Get task instruction from user input.

        Args:
            default_instruction (str): Default instruction if no valid input

        Returns:
            str: Task instruction string
        """
        task_id = input('Enter task ID (or press Enter for default): ').strip()
        if task_id == '0':
            # Reset to preparation pose
            self._move_to_prepare_pose()
            task_id = input('Enter task ID after reset: ').strip()

        if task_id in self.task_pose_sequences:
            self.execute_task_pose(task_id)
            input('Enter task ID (or press Enter for default): ').strip()

        num_times = int(input('Number of times to repeat the task: '))
        task_description = self._get_task_description(task_id)
        return [task_description] * num_times

    def get_observation_statistics(self) -> Dict:
        """Get statistics about current observation data.

        Returns:
            Dict: Statistics including queue lengths and timing information
        """
        if self.observation_window is None:
            return {'status': 'not_initialized'}

        return {
            'window_length': len(self.observation_window),
            'window_maxlen': self.observation_window.maxlen,
            'has_current_obs': len(self.observation_window) > 0,
            'camera_names': self.camera_names,
            'state_dim': self.state_dim,
            'action_chunk': self.action_chunk,
        }

    def cleanup(self):
        """Clean up resources and shutdown gracefully."""
        overwatch.info('Cleaning up BaseInferenceRunner')

        self._prev_ctx = None
        self._action_ctx = SimpleNamespace()

        # Clear observation window
        if self.observation_window is not None:
            self.observation_window.clear()

        overwatch.info('BaseInferenceRunner cleanup completed')

    # Abstract methods that subclasses must implement
    def get_ros_observation(self):
        """Get synchronized observation data from ROS topics.

        This method should be implemented by subclasses to handle
        robot-specific observation collection.

        Returns:
            Tuple: Robot-specific observation data
        """
        raise NotImplementedError(
            'Subclasses must implement get_ros_observation method')

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        This method should be implemented by subclasses to handle
        robot-specific observation window management.

        Returns:
            Dict: Latest observation data
        """
        raise NotImplementedError(
            'Subclasses must implement update_observation_window method')

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute a sequence of robot actions.

        This method should be implemented by subclasses to handle
        robot-specific action execution.

        Args:
            actions (np.ndarray): Array of denormalized robot actions
            rate: ROS rate limiter for action timing
        """
        raise NotImplementedError(
            'Subclasses must implement _execute_actions method')
