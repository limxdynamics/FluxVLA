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

from ..utils.root import RUNNERS
from ..utils.trajectory_utils import resample_remaining
from .base_inference_runner import BaseInferenceRunner


@RUNNERS.register_module()
class Tron2InferenceRunner(BaseInferenceRunner):
    """Runner for Tron2 dual-arm robot inference tasks.

    This runner handles real-time inference tasks for dual-arm robotic
    manipulation using Vision-Language-Action (VLA) models. It manages
    ROS communication, observation collection, action prediction,
    and robot control for both arms in a synchronized manner.

    The runner supports various camera configurations, action chunking,
    and provides a complete inference pipeline from sensor data to
    dual-arm robot actuation.

    Args:
        gripper_threshold (float, optional): Gripper 0-1; below -> 0 (closed).
            Defaults to 0.1.

        prepare_pose (List[float], optional): Prepare pose for the robot.
            Defaults to None.

        enable_head_control (bool, optional): Whether the runner should send
            head commands during prepare pose execution and trajectory
            execution. Defaults to False.

    """

    def __init__(self,
                 gripper_threshold: float = 0.1,
                 prepare_pose: List[float] = None,
                 enable_head_control: bool = False,
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 *args,
                 **kwargs):
        self.gripper_threshold = gripper_threshold
        self.enable_head_control = enable_head_control
        self.async_execution = async_execution
        self.execute_horizon = execute_horizon
        # Set Tron2-specific defaults
        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = [
                'cam_high', 'cam_left_wrist', 'cam_right_wrist'
            ]

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'Tron2Operator',
                'img_top_topic': '/camera/top/color/image_raw',
                'img_left_topic': '/camera/left/color/image_rect_raw',
                'img_right_topic': '/camera/right/color/image_rect_raw',
                'img_top_depth_topic': '/camera/top/depth/image_raw',
                'img_left_depth_topic': '/camera/left/depth/image_rect_raw',
                'img_right_depth_topic': '/camera/right/depth/image_rect_raw',
                'joint_state_topic': '/joint_states',
                'gripper_state_topic': '/gripper_state',
            }

        # Initialize Tron2-specific task descriptions
        if 'task_descriptions' not in kwargs or kwargs[
                'task_descriptions'] is None:
            kwargs['task_descriptions'] = {'1': 'Complete the task.'}

        # Call parent constructor
        super().__init__(*args, **kwargs)

        self.dt = 1.0 / self.publish_rate

        if prepare_pose is None:
            # Initialize Tron2-specific prepare poses
            # [left(7), right(7), head_pitch, head_yaw,
            #  left_gripper(0-1), right_gripper(0-1)]
            self.prepare_pose = [
                [
                    1.2, 0, 0, -2.5, 0, 0, 0, 1.2, 0, 0, -2.5, 0, 0, 0, 0, 0,
                    1, 1
                ],
                [
                    0, 0.24, 0, -2.5, 0.24, 0, 0, 0, -0.24, 0, -2.5, -0.24, 0,
                    0, 0, 0, 1, 1
                ],
                [
                    0, 0.24, 0, -1.56, 0.24, 0, 0, 0, -0.24, 0, -1.56, -0.24,
                    0, 0, 0, 0, 1, 1
                ],
            ]
        else:
            self.prepare_pose = prepare_pose

    def get_ros_observation(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 'JointState',  # noqa: F821
               'JointState', 'JointState']:  # noqa: F821
        """Get synchronized observation data from ROS topics.

        Continuously polls the ROS operator for synchronized sensor data
        including RGB images from three cameras and joint states from both
        arms.

        Returns:
            Tuple containing:
                - img_top (np.ndarray): Top camera RGB image
                - img_left (np.ndarray): Left camera RGB image
                - img_right (np.ndarray): Right camera RGB image
                - arm_left (object): Left arm joint states (7 DOF)
                - arm_right (object): Right arm joint states (7 DOF)
                - robot_gripper (JointState): Gripper 0-1 (operator /100)

        Note:
            This method blocks until synchronized data is available.
            It uses time.sleep for consistent timing.
        """
        import time

        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)

        rate_period = 1.0 / self.publish_rate
        print_flag = True
        time.sleep(rate_period)

        while True:
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    overwatch.info(
                        'Synchronization failed in get_ros_observation')
                    print_flag = False
                time.sleep(rate_period)
                continue

            print_flag = True
            (img_top, img_left, img_right, img_top_depth, img_left_depth,
             img_right_depth, arm_left, arm_right, head,
             robot_gripper) = result

            return (img_top, img_left, img_right, arm_left, arm_right, head,
                    robot_gripper)

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        Maintains a sliding window of observations for temporal context.
        The window includes robot joint positions from both arms and
        camera images from three viewpoints.

        Returns:
            Dict: Latest observation containing:
                - 'qpos': 18 dims: 7 left + 7 right + 2 head
                    + 1 left_grip(0-1) + 1 right_grip(0-1)
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
        (img_top, img_left, img_right, arm_left, arm_right, head,
         robot_gripper) = self.get_ros_observation()

        # Apply JPEG compression to match training conditions
        img_top = self._apply_jpeg_compression(img_top)
        img_left = self._apply_jpeg_compression(img_left)
        img_right = self._apply_jpeg_compression(img_right)

        # Joints + head + grippers (0-1 from operator /100)
        # [left(7), right(7), head_pitch, head_yaw,
        #  left_gripper(1), right_gripper(1)]
        gripper_pos = robot_gripper.position
        left_gripper = np.array(gripper_pos[0:1])
        right_gripper = np.array(gripper_pos[1:2])
        qpos = np.concatenate(
            (np.array(arm_left.position), np.array(arm_right.position),
             np.array(head.position), left_gripper, right_gripper),
            axis=0,
        )

        # Create observation dictionary
        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_top,  # cam_high
            self.camera_names[1]: img_left,  # cam_left_wrist
            self.camera_names[2]: img_right,  # cam_right_wrist
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _move_to_prepare_pose(self):
        """Move robot to predefined preparation pose.

        Supports prepare_pose as:
        - 18-dim: [left(7), right(7), head(2), left_gripper(0-1),
          right_gripper(0-1)]
        - List of 18-dim lists: execute each pose sequentially
        """
        if self.prepare_pose is None:
            return

        # Check if it's a list of poses or single pose
        if isinstance(self.prepare_pose[0], (list, tuple, np.ndarray)):
            # Multiple poses - execute sequentially
            poses = self.prepare_pose
        else:
            # Single pose
            poses = [self.prepare_pose]

        for pose in poses:
            pose = np.array(pose)
            left_joints = pose[:7]
            right_joints = pose[7:14]
            # head at indices 14-15, grippers at 16-17
            head_joints = (
                list(pose[14:16])
                if self.enable_head_control and len(pose) > 15 else None)
            left_gripper = pose[16] if len(pose) > 16 else None
            right_gripper = pose[17] if len(pose) > 17 else None

            self.ros_operator.move_to_targets(
                left_joints,
                right_joints,
                head=head_joints,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                control_rate=30)

        self.last_actions = None

    def _predict_action(self, inputs: dict):
        self._action_ctx.inference_start = time.time()
        raw_action = self.vla.predict_action(**inputs)
        return raw_action

    # Action layout: [left_arm(7), right_arm(7), head(2),
    # left_gripper(1), right_gripper(1)]
    LEFT_GRIPPER_COL = 16
    RIGHT_GRIPPER_COL = 17
    GRIPPER_CLOSED = 0.0

    def _postprocess_actions(self, raw_action):
        """Denormalize and snap near-closed grippers to fully closed."""
        actions = super()._postprocess_actions(raw_action)
        for col in (self.LEFT_GRIPPER_COL, self.RIGHT_GRIPPER_COL):
            actions[:,
                    col] = np.where(actions[:, col] < self.gripper_threshold,
                                    self.GRIPPER_CLOSED, actions[:, col])
        return actions

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute a chunk of dual-arm robot actions.

        In async mode, skips elapsed steps and executes in background thread.
        """
        if self.disable_puppet_arm:
            return

        ctx = self._action_ctx

        if self.async_execution and self._prev_ctx is not None:
            ctx.action_timestamp = ctx.inference_start
            offset = (time.time() - ctx.action_timestamp) / self.dt
            actions = resample_remaining(actions, offset)
        else:
            ctx.action_timestamp = time.time()
            if self.execute_horizon is not None:
                actions = actions[:self.execute_horizon]

        head_trajectory = actions[:,
                                  14:16] if self.enable_head_control else None

        self.ros_operator.execute_trajectory(
            left_arm_trajectory=actions[:, :7],
            right_arm_trajectory=actions[:, 7:14],
            left_gripper_trajectory=actions[:, 16],
            right_gripper_trajectory=actions[:, 17],
            head_trajectory=head_trajectory,
            dt=self.dt,
            async_exec=self.async_execution)

        if self.async_execution and self.execute_horizon is not None:
            time.sleep(self.execute_horizon * self.dt)

    def cleanup(self):
        """Clean up resources."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Cleaning up Tron2InferenceRunner')

        if hasattr(self.ros_operator, 'stop_trajectory'):
            self.ros_operator.stop_trajectory()

        super().cleanup()

        overwatch.info('Tron2InferenceRunner cleanup completed')
