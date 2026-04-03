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

from typing import Dict, Tuple

import numpy as np

from ..utils.root import RUNNERS
from .base_inference_runner import BaseInferenceRunner


@RUNNERS.register_module()
class URInferenceRunner(BaseInferenceRunner):
    """Runner for Universal Robot inference tasks.

    This runner handles real-time inference tasks for robotic manipulation
    using Vision-Language-Action (VLA) models. It manages ROS communication,
    observation collection, action prediction, and robot control in a
    synchronized manner.

    The runner supports various camera configurations, action chunking,
    and provides a complete inference pipeline from sensor data to
    robot actuation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Universal Robot inference runner."""
        # Set UR-specific defaults
        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = [
                'cam_high', 'cam_right_wrist', 'cam_left_wrist'
            ]

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'UROperator',
                'img_left_topic': '/wrist_camera/color/image_raw',
                'img_front_topic': '/front_camera/color/image_raw',
                'puppet_arm_left_topic': '/joint_states',
                'puppet_gripper_left_topic': '/gripper/position',
                'puppet_ee_pose_left_topic': '/arm/tcp_pose',
                'use_depth_image': False,
            }

        # Initialize UR-specific task descriptions
        if 'task_descriptions' not in kwargs or kwargs[
                'task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1': 'pick up the onion',
                '2': 'pick up the bitter melon',
                '3': 'pick up the peach',
                '4': 'put the shanghai green into the bamboo basket',
                '5': 'put the bread into the plate',
                '6': 'put the apple into the gray plate',
                '7': 'put the bitter melon into the pink plate',
            }

        # Initialize UR-specific task pose sequences
        if 'task_pose_sequences' not in kwargs or kwargs[
                'task_pose_sequences'] is None:
            kwargs['task_pose_sequences'] = {
                # Add UR-specific task pose sequences here if needed
            }

        # Call parent constructor
        super().__init__(*args, **kwargs)

        # Initialize UR-specific poses
        self.prepare_pose = [
            2.3911736011505127, -1.7057769934283655, 2.1696739196777344,
            -0.5096147696124476, 1.5789384841918945, -15.709390354140687
        ]  # horizontal, joint angles

    def get_ros_observation(
            self) -> Tuple[np.ndarray, np.ndarray, 'JointState',  # noqa: F821
                           'StampedFloat32']:  # noqa: F821
        """Get synchronized observation data from ROS topics.

        Continuously polls the ROS operator for synchronized sensor data
        including RGB images, joint states, and gripper position.

        Returns:
            Tuple containing:
                - img_front (np.ndarray): Front camera RGB image
                - img_left (np.ndarray): Left camera RGB image
                - puppet_arm_left (JointState): Robot arm joint states
                - puppet_gripper_left (StampedFloat32): Gripper position

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
            (img_front, img_left, img_front_depth, img_left_depth,
             puppet_arm_left, puppet_ee_pose_left, puppet_gripper_left,
             frame_time_min, frame_time_max) = result

            return (img_front, img_left, puppet_arm_left, puppet_gripper_left)

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        Maintains a sliding window of observations for temporal context.
        The window includes robot joint positions, gripper state, and
        camera images from multiple viewpoints.

        Returns:
            Dict: Latest observation containing:
                - 'qpos': Joint positions and gripper state
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
        img_front, img_left, puppet_arm_left, puppet_gripper_left = (
            self.get_ros_observation())

        # Apply JPEG compression to match training conditions
        img_front = self._apply_jpeg_compression(img_front)
        img_left = self._apply_jpeg_compression(img_left)

        # Combine joint positions and gripper state
        # must set to [2,1,0,3,4,5] for ur
        qpos = np.concatenate([
            np.array(puppet_arm_left.position)[[2, 1, 0, 3, 4, 5]],
            np.array([puppet_gripper_left.data])
        ],
                              axis=0)

        # Create observation dictionary
        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_front,  # cam_high
            self.camera_names[2]: img_left,  # cam_left_wrist
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _move_to_prepare_pose(self):
        """Move robot to predefined preparation pose."""
        self.ros_operator.movej(self.prepare_pose)
        self.ros_operator.movegrip(0.085)  # Open gripper

    def execute_task_pose(self, task_id: str):
        """Execute pose sequence for a specific task.

        Args:

            task_id (str): Task identifier string

        Note:
            If the task_id is not found in task_pose_sequences, this method
            does nothing (no error is raised).
        """
        if task_id in self.task_pose_sequences:
            pose_sequence = self.task_pose_sequences[task_id]
            for joint_angles, gripper_position in pose_sequence:
                # Move to joint position
                self.ros_operator.movel(joint_angles)
                # Set gripper position
                self.ros_operator.movegrip(gripper_position[0])

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute a sequence of robot actions.

        Args:
            actions (np.ndarray): Array of denormalized robot actions
            rate: ROS rate limiter for action timing
        """
        for action in actions:
            # Send joint commands (first 6 dimensions)
            self.ros_operator.servoj(action[:6])

            # Send gripper command (last dimension)
            self.ros_operator.movegrip(action[6])

            rate.sleep()
