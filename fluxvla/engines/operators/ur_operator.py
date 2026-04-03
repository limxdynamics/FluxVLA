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
from collections import deque

from fluxvla.engines.utils.root import OPERATORS


def replace_last_segment(input_string, new_segment='camera_info'):
    """Replace the last segment of a path-like string.

    Args:
        input_string (str): The input path-like string
        new_segment (str): The new segment to replace with

    Returns:
        str: String with last segment replaced
    """
    # Find the position of the last '/'
    last_slash_index = input_string.rfind('/')

    # If '/' is found, replace the part after the last '/'
    if last_slash_index != -1:
        return input_string[:last_slash_index + 1] + new_segment
    else:
        # If no '/' is found, return the new segment directly
        return new_segment


@OPERATORS.register_module()
class UROperator:
    """Universal Robot operator for ROS-based robotic arm control.

    This class handles robot arm control, sensor data collection, and
    synchronization for Universal Robot arms in a ROS environment.
    Supports RGB and depth image streams, joint states, end-effector
    poses, and gripper control.
    """

    def __init__(self,
                 img_left_topic,
                 img_front_topic,
                 puppet_arm_left_topic,
                 puppet_ee_pose_left_topic,
                 puppet_gripper_left_topic,
                 use_depth_image=False,
                 img_left_depth_topic=None,
                 img_front_depth_topic=None):
        """Initialize UROperator with ROS topics configuration.

        Args:
            img_left_topic (str): ROS topic for left camera RGB image
            img_front_topic (str): ROS topic for front camera RGB image
            puppet_arm_left_topic (str): ROS topic for puppet arm joint states
            puppet_ee_pose_left_topic (str): ROS topic for end-effector pose
            puppet_gripper_left_topic (str): ROS topic for gripper state
            use_depth_image (bool, optional): Whether to use depth images.
                Defaults to False.
            img_left_depth_topic (str, optional): ROS topic for left depth
                image. Required when use_depth_image=True.
            img_front_depth_topic (str, optional): ROS topic for front depth
                image. Required when use_depth_image=True.

        Raises:
            ValueError: When use_depth_image=True but depth topics not provided
        """
        self.img_left_topic = img_left_topic
        self.img_front_topic = img_front_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_ee_pose_left_topic = puppet_ee_pose_left_topic
        self.puppet_gripper_left_topic = puppet_gripper_left_topic
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
        self.img_front_depth_topic = img_front_depth_topic

        # Validate depth image configuration
        if self.use_depth_image:
            if not img_left_depth_topic or not img_front_depth_topic:
                raise ValueError(
                    'When use_depth_image=True, both img_left_depth_topic '
                    'and img_front_depth_topic must be provided')

        self._init_count()
        self._init()
        self._init_ros()

    def _init_count(self):
        """Initialize error counters for different data streams."""
        self.rgb_left_count = 0
        self.rgb_right_count = 0
        self.rgb_front_count = 0
        self.depth_left_count = 0
        self.depth_right_count = 0
        self.depth_front_count = 0

    def _init(self):
        """Initialize internal data structures and OpenCV bridge."""
        from cv_bridge import CvBridge

        # Initialize error counters
        self.rgb_l = 0
        self.rgb_f = 0
        self.rgb_r = 0
        self.depth_l = 0
        self.depth_f = 0
        self.depth_r = 0

        self.last_time_step = 0
        self.bridge = CvBridge()

        # Initialize message queues
        self.img_left_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_ee_pose_left_deque = deque()
        self.puppet_gripper_left_deque = deque()

    def get_frame(self, slop=0.7):
        """Get synchronized frame data from all sensors.

        Synchronizes RGB images, depth images (if enabled), joint states,
        end-effector poses, and gripper states based on timestamps.

        Args:
            slop (float, optional): Maximum allowed time difference between
                sensors in seconds. Defaults to 0.7.

        Returns:
            tuple or False: If successful, returns tuple containing:
                (img_front, img_left, img_front_depth, img_left_depth,
                 puppet_arm_left, puppet_ee_pose_left, puppet_gripper_left,
                 frame_time, frame_time_max)
                If failed, returns False.
        """
        # Check if all required data queues have data
        required_queues_empty = (
            len(self.img_left_deque) == 0 or len(self.img_front_deque) == 0)

        depth_queues_empty = (
            self.use_depth_image and (len(self.img_left_depth_deque) == 0
                                      or len(self.img_front_depth_deque) == 0))

        if required_queues_empty or depth_queues_empty:
            self._handle_empty_queues()
            return False

        # Calculate minimum frame time across all sensors
        frame_time = self._calculate_frame_time()

        # Check if all sensors have data at the calculated frame time
        if not self._check_sensor_data_availability(frame_time):
            return False

        # Update timing information
        self.last_time_step = frame_time

        # Reset error counters
        self.rgb_l = 0
        self.rgb_f = 0
        self.rgb_r = 0
        self.depth_l = 0
        self.depth_f = 0
        self.depth_r = 0

        # Synchronize all data queues to frame_time
        frame_time_max = self._synchronize_queues(frame_time)

        # Check if synchronization is within acceptable tolerance
        if abs(frame_time_max - frame_time) > slop:
            self._flush_outdated_data(frame_time)
            return False

        # Extract synchronized data
        return self._extract_synchronized_data()

    def _handle_empty_queues(self):
        """Handle empty data queues by incrementing error counters."""
        if len(self.img_left_deque) == 0:
            self.rgb_l += 1
            if self.rgb_l > 3:
                print('Error left RGB', str(time.time()))

        if len(self.img_front_deque) == 0:
            self.rgb_f += 1
            if self.rgb_f > 3:
                print('Error front RGB', str(time.time()))

        if self.use_depth_image:
            if len(self.img_left_depth_deque) == 0:
                self.depth_l += 1
                if self.depth_l > 3:
                    print('Error left Depth')

            if len(self.img_front_depth_deque) == 0:
                self.depth_f += 1
                if self.depth_f > 3:
                    print('Error front Depth')

    def _calculate_frame_time(self):
        """Calculate the minimum frame time across all sensors.

        Returns:
            float: Minimum timestamp across all available sensors
        """
        timestamps = [
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec()
        ]

        if self.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
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
        checks = [(self.img_left_deque, '2'), (self.img_front_deque, '3'),
                  (self.puppet_arm_left_deque, '5'),
                  (self.puppet_ee_pose_left_deque, '6'),
                  (self.puppet_gripper_left_deque, 'g')]

        for deque_obj, error_code in checks:
            if (len(deque_obj) == 0
                    or deque_obj[-1].header.stamp.to_sec() < frame_time):
                print(error_code)
                return False

        if self.use_depth_image:
            depth_checks = [(self.img_left_depth_deque, '7'),
                            (self.img_front_depth_deque, '8')]

            for deque_obj, error_code in depth_checks:
                if (len(deque_obj) == 0
                        or deque_obj[-1].header.stamp.to_sec() < frame_time):
                    print(error_code)
                    return False

        return True

    def _synchronize_queues(self, frame_time):
        """Synchronize all data queues to the specified frame time.

        Args:
            frame_time (float): Target synchronization timestamp

        Returns:
            float: Maximum timestamp after synchronization
        """
        frame_time_max = 0

        # Synchronize RGB image queues
        queues_to_sync = [
            self.img_left_deque, self.img_front_deque,
            self.puppet_arm_left_deque, self.puppet_ee_pose_left_deque,
            self.puppet_gripper_left_deque
        ]

        for queue in queues_to_sync:
            while queue[0].header.stamp.to_sec() < frame_time:
                queue.popleft()
            frame_time_max = max(frame_time_max,
                                 queue[0].header.stamp.to_sec())

        # Synchronize depth image queues if enabled
        if self.use_depth_image:
            depth_queues = [
                self.img_left_depth_deque, self.img_front_depth_deque
            ]

            for queue in depth_queues:
                while queue[0].header.stamp.to_sec() < frame_time:
                    queue.popleft()
                frame_time_max = max(frame_time_max,
                                     queue[0].header.stamp.to_sec())

        return frame_time_max

    def _flush_outdated_data(self, frame_time):
        """Remove outdated data from all queues.

        Args:
            frame_time (float): Timestamp threshold for data removal
        """
        queues_to_flush = [
            self.img_left_deque, self.img_front_deque,
            self.img_left_depth_deque, self.img_front_depth_deque
        ]

        for queue in queues_to_flush:
            while (len(queue) > 0
                   and queue[0].header.stamp.to_sec() <= frame_time):
                if queue in [self.img_left_deque, self.img_front_deque]:
                    print('302', str(time.time()))
                queue.popleft()

    def _extract_synchronized_data(self):
        """Extract synchronized data from all queues.

        Returns:
            tuple: Synchronized sensor data
        """
        # Extract RGB images
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(),
                                              'passthrough')
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(),
                                             'passthrough')

        # Extract robot state data
        puppet_arm_left = self.puppet_arm_left_deque.popleft()
        puppet_ee_pose_left = self.puppet_ee_pose_left_deque.popleft()
        puppet_gripper_left = self.puppet_gripper_left_deque.popleft()

        # Extract depth images if enabled
        img_left_depth = None
        img_front_depth = None
        if self.use_depth_image:
            img_left_depth = self.bridge.imgmsg_to_cv2(
                self.img_left_depth_deque.popleft(), 'passthrough')
            img_front_depth = self.bridge.imgmsg_to_cv2(
                self.img_front_depth_deque.popleft(), 'passthrough')

        return (img_front, img_left, img_front_depth, img_left_depth,
                puppet_arm_left, puppet_ee_pose_left, puppet_gripper_left,
                self.last_time_step, self.last_time_step)

    def img_left_callback(self, msg):
        """Callback function for left camera RGB image messages.

        Args:
            msg: ROS Image message from left camera
        """
        if len(self.img_left_deque) >= 20000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_front_callback(self, msg):
        """Callback function for front camera RGB image messages.

        Args:
            msg: ROS Image message from front camera
        """
        if len(self.img_front_deque) >= 20000:
            print('352', str(time.time()))
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        """Callback function for left camera depth image messages.

        Args:
            msg: ROS Image message from left depth camera
        """
        if len(self.img_left_depth_deque) >= 20000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        """Callback function for front camera depth image messages.

        Args:
            msg: ROS Image message from front depth camera
        """
        if len(self.img_front_depth_deque) >= 20000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        """Callback function for puppet arm joint state messages.

        Args:
            msg: ROS JointState message from puppet arm
        """
        if len(self.puppet_arm_left_deque) >= 20000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_ee_pose_left_callback(self, msg):
        """Callback function for puppet end-effector pose messages.

        Args:
            msg: ROS PoseStamped message from end-effector
        """
        if len(self.puppet_ee_pose_left_deque) >= 20000:
            self.puppet_ee_pose_left_deque.popleft()
        self.puppet_ee_pose_left_deque.append(msg)

    def puppet_gripper_left_callback(self, msg):
        """Callback function for puppet gripper state messages.

        Args:
            msg: ROS StampedFloat32 message from gripper
        """
        if len(self.puppet_gripper_left_deque) >= 20000:
            self.puppet_gripper_left_deque.popleft()
        self.puppet_gripper_left_deque.append(msg)

    def _init_ros(self):
        """Initialize ROS node, subscribers, publishers, and camera info."""
        import rospy
        from geometry_msgs.msg import Pose, PoseStamped
        from robotiq.msg import StampedFloat32
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import Float32

        rospy.init_node('record_episodes', anonymous=True)
        camera_info_topics = []

        # Subscribe to RGB image topics
        rospy.Subscriber(
            self.img_left_topic,
            Image,
            self.img_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_left_topic))

        rospy.Subscriber(
            self.img_front_topic,
            Image,
            self.img_front_callback,
            queue_size=1000,
            tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_front_topic))

        # Subscribe to depth image topics if enabled
        if self.use_depth_image:
            rospy.Subscriber(
                self.img_left_depth_topic,
                Image,
                self.img_left_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)
            camera_info_topics.append(
                replace_last_segment(self.img_left_depth_topic))

            rospy.Subscriber(
                self.img_front_depth_topic,
                Image,
                self.img_front_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)
            camera_info_topics.append(
                replace_last_segment(self.img_front_depth_topic))

        # Subscribe to robot state topics
        rospy.Subscriber(
            self.puppet_arm_left_topic,
            JointState,
            self.puppet_arm_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.puppet_ee_pose_left_topic,
            PoseStamped,
            self.puppet_ee_pose_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.puppet_gripper_left_topic,
            StampedFloat32,
            self.puppet_gripper_left_callback,
            queue_size=1000,
            tcp_nodelay=True)

        # Initialize command publishers
        self.movel_pub = rospy.Publisher('/cmd/movel', Pose, queue_size=10)
        self.servoj_pub = rospy.Publisher(
            '/cmd/servoj', JointState, queue_size=10)
        self.movej_pub = rospy.Publisher(
            '/cmd/movej', JointState, queue_size=10)
        self.servol_pub = rospy.Publisher('/cmd/servol', Pose, queue_size=10)
        self.movegrip_pub = rospy.Publisher(
            '/cmd/gripper', Float32, queue_size=10)

        # Collect camera information
        self.cam_info_dict = {}
        for topic in camera_info_topics:
            camera_info = rospy.wait_for_message(topic, CameraInfo, timeout=5)
            self.cam_info_dict[topic] = {
                'rostopic': topic,
                'height': camera_info.height,
                'width': camera_info.width,
                'distortion_model': camera_info.distortion_model,
                'D': camera_info.D,
                'K': camera_info.K,
                'R': camera_info.R,
                'P': camera_info.P,
                'binning_x': camera_info.binning_x,
                'binning_y': camera_info.binning_y
            }

    def movel(self, eepose):
        """Move the robot arm to specified end-effector pose using
            linear motion.

        Args:
            eepose (list): List of 7 elements representing end-effector pose:
                [x, y, z, qx, qy, qz, qw] where (x,y,z) is position in meters
                and (qx,qy,qz,qw) is orientation quaternion.

        Raises:
            ValueError: If eepose doesn't contain exactly 7 elements
        """
        from geometry_msgs.msg import Point, Pose, Quaternion

        if len(eepose) != 7:
            raise ValueError('End-effector pose must contain exactly 7 '
                             'elements: [x, y, z, qx, qy, qz, qw]')

        msg = Pose()
        msg.position = Point(x=eepose[0], y=eepose[1], z=eepose[2])
        msg.orientation = Quaternion(
            x=eepose[3], y=eepose[4], z=eepose[5], w=eepose[6])
        self.movel_pub.publish(msg)

    def movej(self, qpos):
        """Move robot arm to specified joint positions.

        Args:
            qpos (list): List of 6 joint positions in radians for UR arm:
                [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        """
        from sensor_msgs.msg import JointState

        ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        msg = JointState()
        msg.name = ur_joint_names
        msg.position = qpos
        self.movej_pub.publish(msg)

    def servol(self, eepose):
        """Move robot arm to specified end-effector pose using servo mode.

        Servo mode provides continuous, real-time control with higher
        update frequency compared to movel().

        Args:
            eepose (list): List of 7 elements representing end-effector pose:
                [x, y, z, qx, qy, qz, qw] where (x,y,z) is position in meters
                and (qx,qy,qz,qw) is orientation quaternion.

        Raises:
            ValueError: If eepose doesn't contain exactly 7 elements
        """
        from geometry_msgs.msg import Point, Pose, Quaternion

        if len(eepose) != 7:
            raise ValueError('End-effector pose must contain exactly 7 '
                             'elements: [x, y, z, qx, qy, qz, qw]')

        msg = Pose()
        msg.position = Point(x=eepose[0], y=eepose[1], z=eepose[2])
        msg.orientation = Quaternion(
            x=eepose[3], y=eepose[4], z=eepose[5], w=eepose[6])
        self.servol_pub.publish(msg)

    def servoj(self, qpos):
        """Move robot arm to specified joint positions using servo mode.

        Servo mode provides continuous, real-time control with higher
        update frequency compared to movej().

        Args:
            qpos (list): List of 6 joint positions in radians for UR arm:
                [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        """
        from sensor_msgs.msg import JointState

        ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        msg = JointState()
        msg.name = ur_joint_names
        msg.position = qpos
        self.servoj_pub.publish(msg)

    def movegrip(self, gripper_position):
        """Move robot gripper to specified position.

        Args:
            gripper_position (float): Gripper opening width in meters.
                Valid range is [0, 0.085] where 0 is fully closed and
                0.085 is fully open.

        Note:
            The exact range may depend on the specific gripper model.
            Consult gripper documentation for precise specifications.
        """
        from std_msgs.msg import Float32

        msg = Float32()
        msg.data = gripper_position
        self.movegrip_pub.publish(msg)
