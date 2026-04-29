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

import time
from typing import Dict, List, Tuple

import numpy as np

from ..utils.postprocess import (Trajectory, TrajectoryPostprocessor,
                                 compute_dynamic_horizon, resample_remaining)
from ..utils.postprocess.plot_utils import plot_postprocess_comparison
from ..utils.root import RUNNERS
from .base_inference_runner import BaseInferenceRunner


@RUNNERS.register_module()
class AlohaInferenceRunner(BaseInferenceRunner):
    """Runner for Aloha dual-arm robot inference tasks.

    This runner handles real-time inference tasks for dual-arm robotic
    manipulation using Vision-Language-Action (VLA) models. It manages ROS
    communication, observation collection, action prediction, and robot control
    for both arms in a synchronized manner.

    The runner supports various camera configurations, action chunking,
    and provides a complete inference pipeline from sensor data to
    dual-arm robot actuation.

    Args:
        gripper_threshold (float, optional): Threshold for gripper action.
            Defaults to 0.05.

        prepare_pose (List[float], optional): Prepare pose for the robot.
            Defaults to None.
    """

    def __init__(self,
                 gripper_threshold: float = 0.05,
                 prepare_pose: List[float] = None,
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 dynamic_horizon: dict = None,
                 postprocess_config: dict = None,
                 *args,
                 **kwargs):
        self.gripper_threshold = gripper_threshold
        self.async_execution = async_execution
        self.execute_horizon = execute_horizon
        self.dynamic_horizon_config = dynamic_horizon or {}
        self.postprocess_config = postprocess_config or {}
        # Set Aloha-specific defaults
        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = [
                'cam_high', 'cam_left_wrist', 'cam_right_wrist'
            ]

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'AlohaOperator',
                'img_front_topic': '/camera_f/color/image_raw',
                'img_left_topic': '/camera_l/color/image_raw',
                'img_right_topic': '/camera_r/color/image_raw',
                'img_front_depth_topic': '/camera_f/depth/image_raw',
                'img_left_depth_topic': '/camera_l/depth/image_raw',
                'img_right_depth_topic': '/camera_r/depth/image_raw',
                'puppet_arm_left_cmd_topic': '/master/joint_left',
                'puppet_arm_right_cmd_topic': '/master/joint_right',
                'puppet_arm_left_topic': '/puppet/joint_left',
                'puppet_arm_right_topic': '/puppet/joint_right',
                'robot_base_topic': '/odom_raw',
                'robot_base_cmd_topic': '/cmd_vel',
            }

        # Initialize Aloha-specific task descriptions
        if 'task_descriptions' not in kwargs or kwargs[
                'task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1':
                'pick up the robot dog toy with right arm',
                '2':
                'place it in the brown paper bag',
                '3':
                'pick up the yellow chicken with right arm',
                '4':
                'touch the brown paper bag with left arm',
                '5':
                'push the brown paper bag with left arm',
                '6':
                'pick up the red tomato with right arm',
                '7':
                'grasp the upper edge of the brown paper bag with left arm',
                '8':
                'grasp the bottom edge of the brown paper bag with right arm',
                '9':
                'pull in opposite directions to open the brown paper bag '
                'with both arm',
            }

        # Call parent constructor
        super().__init__(*args, **kwargs)

        self.dt = 1.0 / self.publish_rate
        self.trajectory_postprocessor = TrajectoryPostprocessor(
            config=self.postprocess_config)

        if prepare_pose is None:
            # Initialize other special poses
            self.prepare_pose = ([
                0.071799504, 2.2534682520000002, -1.219353044,
                1.4872231080000002, -0.9562277480000001, -0.603963612, 0.073
            ], [
                0.035167104000000005, 1.4271285280000001, -0.394077404,
                -1.575856072, -0.86801344, 0.584007676, 0.0766
            ])
        else:
            self.prepare_pose = prepare_pose

    def get_ros_observation(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 'JointState',  # noqa: F821
               'JointState']:  # noqa: F821
        """Get synchronized observation data from ROS topics.

        Continuously polls the ROS operator for synchronized sensor data
        including RGB images from three cameras and joint states from both
        arms.

        Returns:
            Tuple containing:
                - img_front (np.ndarray): Front camera RGB image
                - img_left (np.ndarray): Left camera RGB image
                - img_right (np.ndarray): Right camera RGB image
                - puppet_arm_left (JointState): Left arm joint states
                - puppet_arm_right (JointState): Right arm joint states

        Note:
            This method blocks until synchronized data is available.
            It uses ROS rate limiting for consistent timing.
        """
        import rospy

        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)

        rate = rospy.Rate(self.publish_rate)
        print_flag = True
        rate.sleep()

        while not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    overwatch.info(
                        'Synchronization failed in get_ros_observation')
                    print_flag = False
                rate.sleep()
                continue

            print_flag = True
            (img_front, img_left, img_right, img_front_depth, img_left_depth,
             img_right_depth, puppet_arm_left, puppet_arm_right,
             robot_base) = result

            return (img_front, img_left, img_right, puppet_arm_left,
                    puppet_arm_right)

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        Maintains a sliding window of observations for temporal context.
        The window includes robot joint positions from both arms and
        camera images from three viewpoints.

        Returns:
            Dict: Latest observation containing:
                - 'qpos': Joint positions from both arms (14 dimensions)
                - Camera images keyed by camera names

        Note:
            The first observation in a new window is a dummy placeholder
            to maintain consistent window size.
        """
        from collections import deque

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            # Add dummy observation for initialization
            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        # Get current sensor data
        img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = (
            self.get_ros_observation())

        # Apply JPEG compression to match training conditions
        img_front = self._apply_jpeg_compression(img_front)
        img_left = self._apply_jpeg_compression(img_left)
        img_right = self._apply_jpeg_compression(img_right)

        # Combine joint positions from both arms
        qpos = np.concatenate((np.array(
            puppet_arm_left.position), np.array(puppet_arm_right.position)),
                              axis=0)

        # Create observation dictionary
        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_front,  # cam_high
            self.camera_names[1]: img_left,  # cam_left_wrist
            self.camera_names[2]: img_right,  # cam_right_wrist
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _move_to_prepare_pose(self):
        """Move robot to predefined preparation pose."""
        if self.prepare_pose is not None:
            left_pose, right_pose = self.prepare_pose
            self.ros_operator.move_to_joints(left_pose, right_pose)

    def _predict_action(self, inputs: dict):
        self._action_ctx.inference_start = time.time()
        raw_action = self.vla.predict_action(**inputs)
        return raw_action

    # Action layout: [left_arm(7), right_arm(7), base(2)]
    LEFT_ARM_COLS = slice(0, 7)
    RIGHT_ARM_COLS = slice(7, 14)
    BASE_COLS = slice(14, 16)
    LEFT_GRIPPER_COL = 6
    RIGHT_GRIPPER_COL = 13
    GRIPPER_CLOSED = -0.01
    DEFAULT_JOINT_DOF_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

    def _compute_horizon(self, actions):
        dynamic = compute_dynamic_horizon(actions, self.dynamic_horizon_config)
        if dynamic is not None:
            return dynamic
        return self.execute_horizon

    def _postprocess_actions(self, raw_action):
        """Denormalize, apply jerk-constrained smoothing, and snap grippers."""
        raw_chunk_actions = super()._postprocess_actions(raw_action)
        raw_trajectory = Trajectory(
            t0=self._action_ctx.inference_start,
            dt=self.dt,
            positions=raw_chunk_actions,
        )
        self._action_ctx.raw_trajectory = raw_trajectory

        previous_trajectory = getattr(self._prev_ctx, 'trajectory', None)
        processed_trajectory = self.trajectory_postprocessor.process(
            traj=raw_trajectory,
            dof_indices=self.DEFAULT_JOINT_DOF_INDICES,
            prev_traj=previous_trajectory)

        for col in (self.LEFT_GRIPPER_COL, self.RIGHT_GRIPPER_COL):
            col_data = processed_trajectory.positions[:, col]
            col_data[col_data < self.gripper_threshold] = self.GRIPPER_CLOSED

        self._action_ctx.trajectory = processed_trajectory

        self._debug_plot(processed_trajectory)
        return processed_trajectory.positions

    def _debug_plot(self, processed_traj):
        if not self.postprocess_config.get('debug_plot', False):
            return
        prev = self._prev_ctx
        prev_raw = getattr(prev, 'raw_trajectory', None)
        prev_post = getattr(prev, 'trajectory', None)

        plot_postprocess_comparison(
            dof_indices=self.DEFAULT_JOINT_DOF_INDICES,
            cur_raw=self._action_ctx.raw_trajectory,
            cur_post=processed_traj,
            prev_raw=prev_raw,
            prev_post=prev_post,
            plot_dofs=self.postprocess_config.get('debug_plot_dofs'),
        )

    def _execute_actions(self, actions, rate):
        """Execute dual-arm actions (sync or async).

        In async mode, skips steps that elapsed during inference.
        Uses dynamic horizon to avoid replanning during startup delay.
        """
        if self.disable_puppet_arm:
            return

        ctx = self._action_ctx
        horizon = self._compute_horizon(actions)

        if self.async_execution and self._prev_ctx is not None:
            ctx.action_timestamp = ctx.inference_start
            offset = max(0.0, (time.time() - ctx.action_timestamp) / self.dt)
            actions = resample_remaining(actions, offset)
        else:
            ctx.action_timestamp = time.time()
            if horizon is not None:
                actions = actions[:horizon]

        self.ros_operator.execute_trajectory(
            actions[:, self.LEFT_ARM_COLS],
            actions[:, self.RIGHT_ARM_COLS],
            dt=self.dt,
            async_exec=self.async_execution,
            base_velocity=actions[:, self.BASE_COLS]
            if self.use_robot_base else None)

        if self.async_execution and horizon is not None:
            time.sleep(horizon * self.dt)

        prev = self._prev_ctx
        if prev is not None and hasattr(prev, 'action_timestamp'):
            dt = ctx.action_timestamp - prev.action_timestamp
            print(f'[FPS] {1.0 / dt: .1f} chunk/s ({dt: .3f}s)')

    def cleanup(self):
        """Clean up resources."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Cleaning up AlohaInferenceRunner')

        if hasattr(self.ros_operator, 'stop_trajectory'):
            self.ros_operator.stop_trajectory()

        super().cleanup()

        overwatch.info('AlohaInferenceRunner cleanup completed')
