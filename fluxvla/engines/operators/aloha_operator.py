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

import threading
from collections import deque

import numpy as np

from fluxvla.engines.utils.root import OPERATORS


@OPERATORS.register_module()
class AlohaOperator:
    """ALOHA operator for ROS-based dual-arm robot control.

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
        """Initialize AlohaOperator with ROS topics configuration.

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
        # Store configuration parameters
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

        # Set default arm step lengths if not provided
        if arm_steps_length is None:
            arm_steps_length = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        self.arm_steps_length = arm_steps_length
        self.publish_rate = publish_rate

        # Initialize components
        self._init()
        self._init_ros()

    def _init(self):
        """Initialize internal data structures and OpenCV bridge."""
        from cv_bridge import CvBridge

        self.bridge = CvBridge()

        # Initialize message queues for images
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()

        # Initialize message queues for robot states
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

        # Trajectory execution state
        self._traj_thread = None
        self._traj_stop_event = threading.Event()

    def get_frame(self):
        """Get synchronized frame data from all sensors.

        Synchronizes RGB images, depth images (if enabled), joint states
        for both arms, and mobile base odometry (if enabled) based on
        timestamps.

        Returns:
            tuple or False: If successful, returns tuple containing:
                (img_front, img_left, img_right, img_front_depth,
                 img_left_depth, img_right_depth, puppet_arm_left,
                 puppet_arm_right, robot_base)
                If failed, returns False.
        """
        # Check if all required RGB image queues have data
        if (len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0
                or len(self.img_front_deque) == 0):
            return False

        # Check depth image queues if depth is enabled
        if self.use_depth_image:
            if (len(self.img_left_depth_deque) == 0
                    or len(self.img_right_depth_deque) == 0
                    or len(self.img_front_depth_deque) == 0):
                return False

        # Calculate minimum frame time across all sensors
        frame_time = self._calculate_frame_time()

        # Check if all sensors have data at the calculated frame time
        if not self._check_sensor_data_availability(frame_time):
            return False

        # Synchronize and extract data
        return self._synchronize_and_extract_data(frame_time)

    def _calculate_frame_time(self):
        """Calculate the minimum frame time across all sensors.

        Returns:
            float: Minimum timestamp across all available sensors
        """
        timestamps = [
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec()
        ]

        if self.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_right_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec()
            ])

        return min(timestamps)

    def _check_sensor_data_availability(self, frame_time):
        """Check if all sensors have data at the specified frame time.

        Args:
            frame_time (float): Target frame timestamp

        Returns:
            bool: True if all sensors have data, False otherwise
        """
        # Check RGB image availability
        if (len(self.img_left_deque) == 0
                or self.img_left_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.img_right_deque) == 0 or
                self.img_right_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.img_front_deque) == 0 or
                self.img_front_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        # Check arm data availability
        if (len(self.puppet_arm_left_deque) == 0
                or self.puppet_arm_left_deque[-1].header.stamp.to_sec() <
                frame_time):
            return False
        if (len(self.puppet_arm_right_deque) == 0
                or self.puppet_arm_right_deque[-1].header.stamp.to_sec() <
                frame_time):
            return False

        # Check depth image availability if enabled
        if self.use_depth_image:
            if (len(self.img_left_depth_deque) == 0
                    or self.img_left_depth_deque[-1].header.stamp.to_sec() <
                    frame_time):
                return False
            if (len(self.img_right_depth_deque) == 0
                    or self.img_right_depth_deque[-1].header.stamp.to_sec() <
                    frame_time):
                return False
            if (len(self.img_front_depth_deque) == 0
                    or self.img_front_depth_deque[-1].header.stamp.to_sec() <
                    frame_time):
                return False

        # Check robot base availability if enabled
        if self.use_robot_base:
            if (len(self.robot_base_deque) == 0
                    or self.robot_base_deque[-1].header.stamp.to_sec() <
                    frame_time):
                return False

        return True

    def _synchronize_and_extract_data(self, frame_time):
        """Synchronize queues and extract data at the specified frame time.

        Args:
            frame_time (float): Target synchronization timestamp

        Returns:
            tuple: Synchronized sensor data
        """
        # Synchronize and extract RGB images
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(),
                                             'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(),
                                              'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(),
                                              'passthrough')

        # Synchronize and extract arm data
        while (self.puppet_arm_left_deque[0].header.stamp.to_sec() <
               frame_time):
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while (self.puppet_arm_right_deque[0].header.stamp.to_sec() <
               frame_time):
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        # Extract depth images if enabled
        img_left_depth = None
        img_right_depth = None
        img_front_depth = None
        if self.use_depth_image:
            while (self.img_left_depth_deque[0].header.stamp.to_sec() <
                   frame_time):
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(
                self.img_left_depth_deque.popleft(), 'passthrough')

            while (self.img_right_depth_deque[0].header.stamp.to_sec() <
                   frame_time):
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(
                self.img_right_depth_deque.popleft(), 'passthrough')

            while (self.img_front_depth_deque[0].header.stamp.to_sec() <
                   frame_time):
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(
                self.img_front_depth_deque.popleft(), 'passthrough')

        # Extract robot base data if enabled
        robot_base = None
        if self.use_robot_base:
            while (self.robot_base_deque[0].header.stamp.to_sec() <
                   frame_time):
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth,
                img_left_depth, img_right_depth, puppet_arm_left,
                puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        """Callback function for left camera RGB image messages.

        Args:
            msg: ROS Image message from left camera
        """
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        """Callback function for right camera RGB image messages.

        Args:
            msg: ROS Image message from right camera
        """
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        """Callback function for front camera RGB image messages.

        Args:
            msg: ROS Image message from front camera
        """
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        """Callback function for left camera depth image messages.

        Args:
            msg: ROS Image message from left depth camera
        """
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        """Callback function for right camera depth image messages.

        Args:
            msg: ROS Image message from right depth camera
        """
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        """Callback function for front camera depth image messages.

        Args:
            msg: ROS Image message from front depth camera
        """
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        """Callback function for left arm joint state messages.

        Args:
            msg: ROS JointState message from left arm
        """
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        """Callback function for right arm joint state messages.

        Args:
            msg: ROS JointState message from right arm
        """
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        """Callback function for robot base odometry messages.

        Args:
            msg: ROS Odometry message from mobile base
        """
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def _init_ros(self):
        """Initialize ROS node, subscribers, and publishers."""
        import rospy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image, JointState

        rospy.init_node('joint_state_publisher', anonymous=True)

        # Subscribe to RGB image topics
        rospy.Subscriber(
            self.img_left_topic,
            Image,
            self.img_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.img_right_topic,
            Image,
            self.img_right_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.img_front_topic,
            Image,
            self.img_front_callback,
            queue_size=1000,
            tcp_nodelay=True)

        # Subscribe to depth image topics if enabled
        if self.use_depth_image:
            rospy.Subscriber(
                self.img_left_depth_topic,
                Image,
                self.img_left_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)
            rospy.Subscriber(
                self.img_right_depth_topic,
                Image,
                self.img_right_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)
            rospy.Subscriber(
                self.img_front_depth_topic,
                Image,
                self.img_front_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)

        # Subscribe to arm joint state topics
        rospy.Subscriber(
            self.puppet_arm_left_topic,
            JointState,
            self.puppet_arm_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.puppet_arm_right_topic,
            JointState,
            self.puppet_arm_right_callback,
            queue_size=1000,
            tcp_nodelay=True)

        # Subscribe to robot base topic
        rospy.Subscriber(
            self.robot_base_topic,
            Odometry,
            self.robot_base_callback,
            queue_size=1000,
            tcp_nodelay=True)

        # Initialize command publishers
        self.puppet_arm_left_publisher = rospy.Publisher(
            self.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(
            self.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(
            self.robot_base_cmd_topic, Twist, queue_size=10)

    def get_current_joint_states(self):
        """Get current joint states from both arms.

        Returns:
            tuple: (left_position, right_position) as numpy arrays,
                or (None, None) if data is not available.
        """
        left_pos = None
        right_pos = None

        if len(self.puppet_arm_left_deque) > 0:
            left_pos = np.array(self.puppet_arm_left_deque[-1].position)
        if len(self.puppet_arm_right_deque) > 0:
            right_pos = np.array(self.puppet_arm_right_deque[-1].position)

        return left_pos, right_pos

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
                           async_exec=False,
                           base_velocity=None):
        """Execute trajectories for both arms.

        Args:
            left_trajectory (np.ndarray): Left arm trajectory [n_steps x 7].
            right_trajectory (np.ndarray): Right arm trajectory [n_steps x 7].
            dt (float): Time step between trajectory points in seconds.
            async_exec (bool): If True, execute in background thread.
                If False (default), block until trajectory completes.
            base_velocity (np.ndarray, optional): Mobile base velocity
                trajectory [n_steps x 2] (linear_x, angular_z).
        """
        left_traj = np.asarray(left_trajectory)
        right_traj = np.asarray(right_trajectory)
        base_vel = (
            np.asarray(base_velocity) if base_velocity is not None else None)

        # Stop any existing trajectory (old event stays set → old thread exits)
        self._traj_stop_event.set()
        # Fresh event for the new trajectory
        self._traj_stop_event = threading.Event()

        stop_event = self._traj_stop_event
        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_trajectory,
                args=(left_traj, right_traj, dt, base_vel, stop_event),
                daemon=True)
            self._traj_thread.start()
        else:
            self._run_trajectory(left_traj, right_traj, dt, base_vel,
                                 stop_event)

    def stop_trajectory(self):
        """Stop the currently executing trajectory."""
        self._traj_stop_event.set()

    def is_trajectory_running(self):
        """Check if a trajectory is currently being executed."""
        return (self._traj_thread is not None and self._traj_thread.is_alive())

    def move_to_joints(self, left, right):
        """Move arms to target positions with step-wise interpolation.

        Blocks until target positions are reached.

        Args:
            left (list): Target joint positions for left arm in radians.
            right (list): Target joint positions for right arm in radians.
        """
        import rospy

        rate = rospy.Rate(self.publish_rate)
        left_arm = None
        right_arm = None

        # Wait for initial arm states
        while not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)

            if left_arm is not None and right_arm is not None:
                break
            rate.sleep()

        # Calculate movement directions
        left_symbol = [
            1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))
        ]
        right_symbol = [
            1 if right[i] - right_arm[i] > 0 else -1
            for i in range(len(right))
        ]

        flag = True
        step = 0

        # Interpolation loop
        while flag and not rospy.is_shutdown():
            # Calculate differences
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [
                abs(right[i] - right_arm[i]) for i in range(len(right))
            ]

            flag = False

            # Update left arm positions
            for i in range(len(left)):
                if left_diff[i] < self.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.arm_steps_length[i]
                    flag = True

            # Update right arm positions
            for i in range(len(right)):
                if right_diff[i] < self.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.arm_steps_length[i]
                    flag = True

            self._send_joints(left_arm, right_arm)

            step += 1
            print(f'move_to_joints: step {step}')
            rate.sleep()

    def _send_joints(self, left, right):
        """Publish joint commands to both puppet arms."""
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header

        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [
            'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5',
            'joint6'
        ]

        # Publish left arm command
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)

        # Publish right arm command
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def _send_base_velocity(self, vel):
        """Publish velocity commands to mobile base."""
        from geometry_msgs.msg import Twist

        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]

        self.robot_base_publisher.publish(vel_msg)

    def _run_trajectory(self, left_traj, right_traj, dt, base_vel, stop_event):
        """Execute trajectory step by step.

        Args:
            left_traj (np.ndarray): Left arm trajectory [n_steps x 7].
            right_traj (np.ndarray): Right arm trajectory [n_steps x 7].
            dt (float): Time step between points in seconds.
            base_vel (np.ndarray, optional): Base velocity [n_steps x 2].
            stop_event (threading.Event): Event to signal early stop.
        """
        import rospy

        stop = stop_event
        rate = rospy.Rate(1.0 / dt)

        for i in range(len(left_traj)):
            if rospy.is_shutdown() or stop.is_set():
                break
            self._send_joints(left_traj[i].tolist(), right_traj[i].tolist())
            if base_vel is not None:
                self._send_base_velocity(base_vel[i])
            rate.sleep()
