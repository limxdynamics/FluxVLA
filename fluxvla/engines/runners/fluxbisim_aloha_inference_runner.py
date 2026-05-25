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

from collections import deque
from typing import Dict, List

import numpy as np

from ..utils.root import RUNNERS
from .fluxbisim_base_inference_runner import BaseInferenceRunnerSim


@RUNNERS.register_module()
class AlohaInferenceRunnerSim(BaseInferenceRunnerSim):
    """Runner for Aloha dual-arm robot inference tasks in simulation.

    This runner handles inference tasks for dual-arm robotic manipulation
    using Vision-Language-Action (VLA) models. It manages ROS
    communication, observation collection, action prediction, and robot
    control for both arms in a synchronized manner.

    Unlike the real-robot ``AlohaInferenceRunner``, execution is always
    synchronous (no async thread or RTC) because the simulation
    environment only advances after each action is consumed.

    Args:
        gripper_threshold (float, optional): Threshold for gripper action.
            Defaults to 0.05.

        prepare_pose (List[float], optional): Prepare pose for the robot.
            Defaults to None.
    """

    LEFT_GRIPPER_COL = 6
    RIGHT_GRIPPER_COL = 13
    GRIPPER_CLOSED = 0.0

    def __init__(self,
                 gripper_threshold: float = 0.05,
                 prepare_pose: List[float] = None,
                 execute_horizon: int = None,
                 *args,
                 **kwargs):
        self.gripper_threshold = gripper_threshold
        self.execute_horizon = execute_horizon

        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = [
                'cam_high', 'cam_left_wrist', 'cam_right_wrist'
            ]

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'AlohaOperatorSim',
                'img_front_topic': '/camera_f/color/image_raw',
                'img_left_topic': '/camera_l/color/image_raw',
                'img_right_topic': '/camera_r/color/image_raw',
                'puppet_arm_left_cmd_topic': '/master/joint_left',
                'puppet_arm_right_cmd_topic': '/master/joint_right',
                'puppet_arm_left_topic': '/puppet/joint_left',
                'puppet_arm_right_topic': '/puppet/joint_right',
                'robot_base_topic': '/odom_raw',
                'robot_base_cmd_topic': '/cmd_vel',
            }

        super().__init__(*args, **kwargs)

        self.dt = 1.0 / self.publish_rate

        if prepare_pose is None:
            self.prepare_pose = ([
                0.071799504, 2.253468252, -1.219353044, 1.487223108,
                -0.956227748, -0.603963612, 0.073
            ], [
                0.035167104, 1.427128528, -0.394077404, -1.575856072,
                -0.86801344, 0.584007676, 0.0766
            ])
        else:
            self.prepare_pose = prepare_pose

    def get_ros_observation(self):
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

        rate = rospy.Rate(self.publish_rate)
        rate.sleep()

        while not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if result:
                (img_front, img_left, img_right, _, _, _, puppet_arm_left,
                 puppet_arm_right, _) = result
                return (img_front, img_left, img_right, puppet_arm_left,
                        puppet_arm_right)
            rate.sleep()

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
        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)
            dummy_obs = {'qpos': None}
            for cam in self.camera_names:
                dummy_obs[cam] = None
            self.observation_window.append(dummy_obs)

        img_front, img_left, img_right, arm_left, arm_right = (
            self.get_ros_observation())

        img_front = self._apply_jpeg_compression(img_front)
        img_left = self._apply_jpeg_compression(img_left)
        img_right = self._apply_jpeg_compression(img_right)

        qpos = np.concatenate(
            (np.array(arm_left.position), np.array(arm_right.position)),
            axis=0)

        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_front,
            self.camera_names[1]: img_left,
            self.camera_names[2]: img_right,
        }
        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _postprocess_actions(self, raw_action):
        """Denormalize and snap near-closed grippers to fully closed."""
        actions = super()._postprocess_actions(raw_action)
        for col in (self.LEFT_GRIPPER_COL, self.RIGHT_GRIPPER_COL):
            actions[:,
                    col] = np.where(actions[:, col] < self.gripper_threshold,
                                    self.GRIPPER_CLOSED, actions[:, col])
        return actions

    def _get_unsmoothed_action_dims(self, action_dim: int):
        """Keep gripper commands crisp instead of smoothing them."""
        del action_dim
        return (self.LEFT_GRIPPER_COL, self.RIGHT_GRIPPER_COL)

    def _execute_actions(self, actions):
        """Execute dual-arm actions synchronously.

        Unlike the real-robot runner, execution is always synchronous
        (no async thread) because the simulation only advances after
        each command is consumed.
        """
        if self.disable_puppet_arm:
            return

        if self.execute_horizon is not None:
            actions = actions[:self.execute_horizon]

        self.ros_operator.execute_trajectory(
            actions[:, :7],
            actions[:, 7:14],
            dt=self.dt,
            base_velocity=(actions[:, 14:16] if self.use_robot_base else None))

    def cleanup(self):
        """Clean up resources."""
        from ..utils import initialize_overwatch
        ow = initialize_overwatch(__name__)
        ow.info('Cleaning up AlohaInferenceRunnerSim')
        super().cleanup()
