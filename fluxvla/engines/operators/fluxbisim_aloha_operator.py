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

import numpy as np

from fluxvla.engines.utils.root import OPERATORS


@OPERATORS.register_module()
class AlohaOperatorSim:
    """ALOHA operator for ROS-based dual-arm robot control in simulation.

    This class handles dual-arm robot control, multi-camera sensor data
    collection, and synchronization for ALOHA robotic systems in a ROS
    environment. Supports RGB and depth image streams from multiple cameras,
    joint states for dual arms, and mobile base control.
    """

    def __init__(self,
                 img_left_topic,
                 img_right_topic,
                 img_front_topic,
                 img_left_depth_topic,
                 img_right_depth_topic,
                 img_front_depth_topic,
                 puppet_arm_left_topic,
                 puppet_arm_right_topic,
                 robot_base_topic,
                 puppet_arm_left_cmd_topic,
                 puppet_arm_right_cmd_topic,
                 robot_base_cmd_topic,
                 use_depth_image=False,
                 use_robot_base=False,
                 arm_steps_length=None,
                 publish_rate=30):
        """Initialize AlohaOperatorSim with ROS topics configuration.

         Args:
            img_left_topic (str): ROS topic for left camera RGB image
            img_right_topic (str): ROS topic for right camera RGB image
            img_front_topic (str): ROS topic for front camera RGB image
            img_left_depth_topic (str): ROS topic for left depth image
            img_right_depth_topic (str): ROS topic for right depth image
            img_front_depth_topic (str): ROS topic for front depth image
            puppet_arm_left_topic (str): ROS topic for left arm joint states
            puppet_arm_right_topic (str): ROS topic for right arm joint states
            robot_base_topic (str): ROS topic for mobile base odometry
            puppet_arm_left_cmd_topic (str): ROS topic for left arm commands
            puppet_arm_right_cmd_topic (str): ROS topic for right arm commands
            robot_base_cmd_topic (str): ROS topic for base velocity commands
            use_depth_image (bool, optional): Whether to use depth images.
                Defaults to False.
            use_robot_base (bool, optional): Whether to use mobile base.
                Defaults to False.
            arm_steps_length (list, optional): Step sizes for each joint
                in continuous motion. Defaults to
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02].
            publish_rate (int, optional): Publishing rate in Hz.
                Defaults to 30.
        """
        self.img_left_topic = img_left_topic
        self.img_right_topic = img_right_topic
        self.img_front_topic = img_front_topic
        self.img_left_depth_topic = img_left_depth_topic
        self.img_right_depth_topic = img_right_depth_topic
        self.img_front_depth_topic = img_front_depth_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_arm_right_topic = puppet_arm_right_topic
        self.robot_base_topic = robot_base_topic
        self.puppet_arm_left_cmd_topic = puppet_arm_left_cmd_topic
        self.puppet_arm_right_cmd_topic = puppet_arm_right_cmd_topic
        self.robot_base_cmd_topic = robot_base_cmd_topic
        self.use_depth_image = use_depth_image
        self.use_robot_base = use_robot_base

        if arm_steps_length is None:
            arm_steps_length = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        self.arm_steps_length = arm_steps_length
        self.publish_rate = publish_rate

        # Latest sensor readings (replaces deque buffering)
        self._img_left_msg = None
        self._img_right_msg = None
        self._img_front_msg = None
        self._img_left_depth_msg = None
        self._img_right_depth_msg = None
        self._img_front_depth_msg = None
        self._arm_left_msg = None
        self._arm_right_msg = None
        self._robot_base_msg = None

        # Set by the /env/reset callback when the sim environment restarts
        self._env_reset_received = False
        self._last_completed_chunk_id = -1
        self._next_chunk_id = 0

        self._init_ros()

    def _init_ros(self):
        """Initialize ROS node, subscribers, and publishers."""
        import rospy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Int32, String

        rospy.init_node('aloha_sim_operator', anonymous=True)

        rospy.Subscriber(
            self.img_left_topic, Image, self._on_img_left, queue_size=1)
        rospy.Subscriber(
            self.img_right_topic, Image, self._on_img_right, queue_size=1)
        rospy.Subscriber(
            self.img_front_topic, Image, self._on_img_front, queue_size=1)
        rospy.Subscriber(
            self.puppet_arm_left_topic,
            JointState,
            self._on_arm_left,
            queue_size=1)
        rospy.Subscriber(
            self.puppet_arm_right_topic,
            JointState,
            self._on_arm_right,
            queue_size=1)

        if self.use_robot_base:
            from nav_msgs.msg import Odometry
            rospy.Subscriber(
                self.robot_base_topic,
                Odometry,
                self._on_robot_base,
                queue_size=1)

        # Subscribe to depth image topics if enabled
        if self.use_depth_image:
            rospy.Subscriber(
                self.img_left_depth_topic,
                Image,
                self._on_img_left_depth,
                queue_size=1)
            rospy.Subscriber(
                self.img_right_depth_topic,
                Image,
                self._on_img_right_depth,
                queue_size=1)
            rospy.Subscriber(
                self.img_front_depth_topic,
                Image,
                self._on_img_front_depth,
                queue_size=1)

        rospy.Subscriber(
            '/env/reset', String, self._on_env_reset, queue_size=1)
        rospy.Subscriber(
            '/env/chunk_done', Int32, self._on_chunk_done, queue_size=1)

        self.arm_left_pub = rospy.Publisher(
            self.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.arm_right_pub = rospy.Publisher(
            self.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        if self.use_robot_base:
            self.base_pub = rospy.Publisher(
                self.robot_base_cmd_topic, Twist, queue_size=10)

    # ---- Callbacks: store latest message only ----

    def _on_img_left(self, msg):
        """Callback function for left camera RGB image messages.

        Args:
            msg: ROS Image message from left camera
        """
        self._img_left_msg = msg

    def _on_img_right(self, msg):
        """Callback function for right camera RGB image messages.

        Args:
            msg: ROS Image message from right camera
        """
        self._img_right_msg = msg

    def _on_img_front(self, msg):
        """Callback function for front camera RGB image messages.

        Args:
            msg: ROS Image message from front camera
        """
        self._img_front_msg = msg

    def _on_img_left_depth(self, msg):
        """Callback function for left camera depth image messages.

        Args:
            msg: ROS Image message from left depth camera
        """
        self._img_left_depth_msg = msg

    def _on_img_right_depth(self, msg):
        """Callback function for right camera depth image messages.

        Args:
            msg: ROS Image message from right depth camera
        """
        self._img_right_depth_msg = msg

    def _on_img_front_depth(self, msg):
        """Callback function for front camera depth image messages.

        Args:
            msg: ROS Image message from front depth camera
        """
        self._img_front_depth_msg = msg

    def _on_arm_left(self, msg):
        """Callback function for left arm joint state messages.

        Args:
            msg: ROS JointState message from left arm
        """
        self._arm_left_msg = msg

    def _on_arm_right(self, msg):
        """Callback function for right arm joint state messages.

        Args:
            msg: ROS JointState message from right arm
        """
        self._arm_right_msg = msg

    def _on_robot_base(self, msg):
        """Callback function for robot base odometry messages.

        Args:
            msg: ROS Odometry message from mobile base
        """
        self._robot_base_msg = msg

    def _on_env_reset(self, msg):
        """Callback function for environment reset messages.

        Args:
            msg: ROS String message with reset signal
        """
        if msg.data == 'environment_reset':
            self._env_reset_received = True

    def _on_chunk_done(self, msg):
        """Record the latest action chunk completed by the simulator."""
        self._last_completed_chunk_id = msg.data

    # ---- Environment reset ----

    def check_env_reset(self):
        """Return True if the simulation sent an environment-reset signal."""
        return self._env_reset_received

    def clear_env_reset_flag(self):
        """Clear the pending reset flag so a new episode can begin."""
        self._env_reset_received = False

    def _wait_for_chunk_done(self, chunk_id):
        """Block until the simulator confirms completion of this chunk."""
        import rospy

        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            if (self._env_reset_received
                    or self._last_completed_chunk_id >= chunk_id):
                return True
            rate.sleep()
        return False

    # ---- Observation ----

    def get_frame(self):
        """Get synchronized frame data from all sensors.

        Converts stored ROS Image messages to OpenCV arrays and returns
        them together with joint states and base odometry. Depth images
        are processed when use_depth_image is True to keep the return
        signature compatible with the real-robot operator.

        Returns:
            tuple or False: If successful, returns tuple containing:
                (img_front, img_left, img_right, img_front_depth,
                 img_left_depth, img_right_depth, puppet_arm_left,
                 puppet_arm_right, robot_base)
                If failed, returns False.
        """
        from cv_bridge import CvBridge

        if (self._img_left_msg is None or self._img_right_msg is None
                or self._img_front_msg is None or self._arm_left_msg is None
                or self._arm_right_msg is None):
            return False

        # Check depth image availability if enabled
        if self.use_depth_image:
            if (self._img_left_depth_msg is None
                    or self._img_right_depth_msg is None
                    or self._img_front_depth_msg is None):
                return False

        bridge = CvBridge()
        img_left = bridge.imgmsg_to_cv2(self._img_left_msg, 'passthrough')
        img_right = bridge.imgmsg_to_cv2(self._img_right_msg, 'passthrough')
        img_front = bridge.imgmsg_to_cv2(self._img_front_msg, 'passthrough')

        # Process depth images if enabled
        img_left_depth = None
        img_right_depth = None
        img_front_depth = None
        if self.use_depth_image:
            img_left_depth = bridge.imgmsg_to_cv2(self._img_left_depth_msg,
                                                  'passthrough')
            img_right_depth = bridge.imgmsg_to_cv2(self._img_right_depth_msg,
                                                   'passthrough')
            img_front_depth = bridge.imgmsg_to_cv2(self._img_front_depth_msg,
                                                   'passthrough')

        robot_base = self._robot_base_msg if self.use_robot_base else None
        return (img_front, img_left, img_right, img_front_depth,
                img_left_depth, img_right_depth, self._arm_left_msg,
                self._arm_right_msg, robot_base)

    def get_current_joint_states(self):
        """Get current joint states from both arms.

        Returns:
            tuple: (left_position, right_position) as numpy arrays,
                or (None, None) if data is not available.
        """
        left = (
            np.array(self._arm_left_msg.position)
            if self._arm_left_msg else None)
        right = (
            np.array(self._arm_right_msg.position)
            if self._arm_right_msg else None)
        return left, right

    # ---- Action execution ----

    def execute_step(self, left, right, base_velocity=None):
        """Execute a single-step action command.

        Sends one set of joint positions to both arms, and optionally
        a base velocity command.

        Args:
            left (list): Joint positions for left arm (7 values, radians).
            right (list): Joint positions for right arm (7 values, radians).
            base_velocity (list, optional): Base velocity
                [linear_x, angular_z].
        """
        self._send_joints(left, right)
        if base_velocity is not None:
            self._send_base_velocity(base_velocity)

    def execute_trajectory(self,
                           left_trajectory,
                           right_trajectory,
                           dt=0.1,
                           base_velocity=None):
        """Execute trajectories for both arms.

        Unlike the real-robot operator, there is no ``async_exec``
        option because the simulation advances synchronously.

        Args:
            left_trajectory (np.ndarray): Left arm trajectory [n_steps x 7].
            right_trajectory (np.ndarray): Right arm trajectory [n_steps x 7].
            dt (float): Time step between trajectory points in seconds.
            base_velocity (np.ndarray, optional): Mobile base velocity
                trajectory [n_steps x 2] (linear_x, angular_z).
        """
        import rospy

        left_traj = np.asarray(left_trajectory)
        right_traj = np.asarray(right_trajectory)
        base_vel = (
            np.asarray(base_velocity) if base_velocity is not None else None)

        if len(left_traj) == 0:
            return

        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1

        rate = rospy.Rate(1.0 / dt)
        for i in range(len(left_traj)):
            if rospy.is_shutdown() or self._env_reset_received:
                break
            self._send_joints(
                left_traj[i].tolist(),
                right_traj[i].tolist(),
                chunk_id=chunk_id,
                action_idx=i)
            if base_vel is not None:
                self._send_base_velocity(base_vel[i])
            rate.sleep()

        self._wait_for_chunk_done(chunk_id)

    # ---- Internal helpers ----

    def _send_joints(self, left, right, chunk_id=0, action_idx=0):
        """Publish joint commands to both puppet arms."""
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header

        stamp = rospy.Time.now()
        joint_names = [f'joint{i}' for i in range(7)]
        # Encode both chunk_id and action_idx in frame_id because
        # ROS1 overwrites header.seq with its own auto-increment counter.
        frame_id = f'{chunk_id}:{action_idx}'  # noqa: E231

        left_msg = JointState()
        left_msg.header = Header()
        left_msg.header.stamp = stamp
        left_msg.header.frame_id = frame_id
        left_msg.name = joint_names
        left_msg.position = left
        self.arm_left_pub.publish(left_msg)

        right_msg = JointState()
        right_msg.header = Header()
        right_msg.header.stamp = stamp
        right_msg.header.frame_id = frame_id
        right_msg.name = joint_names
        right_msg.position = right
        self.arm_right_pub.publish(right_msg)

    def _send_base_velocity(self, vel):
        """Publish velocity commands to mobile base."""
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = vel[0]
        msg.angular.z = vel[1]
        self.base_pub.publish(msg)
