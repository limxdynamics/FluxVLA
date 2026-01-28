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
import threading
import time
import uuid
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import websocket

from fluxvla.engines.utils.root import OPERATORS

DEFAULT_HEAD_JOINT_POSITIONS = [1.047, -0.0139998]


class NumpySafeEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@OPERATORS.register_module()
class Tron2Operator:
    """Tron2 operator for dual-arm robot control.

    This class handles dual-arm robot control, multi-camera sensor data
    collection, and synchronization for Tron2 robotic systems.
    Supports RGB and depth image streams from multiple cameras,
    and joint states for dual arms + head (16 DOF merged).
    """

    def __init__(
            self,
            # Camera topics
            img_left_topic: str = '/camera/left/color/image_rect_raw',
            img_right_topic: str = '/camera/right/color/image_rect_raw',
            img_top_topic: str = '/camera/top/color/image_raw',
            # Depth image topics (required if use_depth_image=True)
            img_left_depth_topic: str = '/camera/left/depth/image_rect_raw',
            img_right_depth_topic: str = '/camera/right/depth/image_rect_raw',
            img_top_depth_topic: str = '/camera/top/depth/image_raw',
            # Joint state topics
            joint_state_topic: str = '/joint_states',
            # Gripper topics
            gripper_state_topic: str = '/gripper_state',
            # End-effector pose topics
            ee_pose_left_topic: str = '/left_arm/ee_pose',
            ee_pose_right_topic: str = '/right_arm/ee_pose',
            # Options
            use_depth_image: bool = False,
            arm_steps_length: Optional[List[float]] = None,
            # WebSocket options
            robot_ip: str = '10.192.1.2',
            ws_port: int = 5000,
            ws_accid: str = None,
            enable_base_control: bool = False,
            trajectory_exec_mode: str = 'movej'):
        """Initialize Tron2Operator with ROS topics configuration.

        Topics can be specified via constructor args or environment variables.

        Args:
            img_left_topic: ROS topic for left camera RGB image
            img_right_topic: ROS topic for right camera RGB image
            img_top_topic: ROS topic for top camera RGB image
            img_left_depth_topic: ROS topic for left depth image
            img_right_depth_topic: ROS topic for right depth image
            img_top_depth_topic: ROS topic for top depth image
            joint_state_topic: ROS topic for merged joint states
                (16 DOF: 14 arm + 2 head)
            gripper_state_topic: gripper state (hw 0-100, we /100 -> 0-1)
            ee_pose_left_topic: ROS topic for left end-effector pose
            ee_pose_right_topic: ROS topic for right end-effector pose
            use_depth_image: Whether to use depth images. Defaults to False.
            arm_steps_length: Step sizes for each arm joint (14 DOF).
            robot_ip: Robot IP address for WebSocket connection.
            ws_port: WebSocket port. Defaults to 5000.
            ws_accid: WebSocket account ID (required for robot control).
            enable_base_control: Whether to enable base control.
            trajectory_exec_mode: Trajectory execution mode, one of
                {'movej', 'servoj'}. Defaults to 'movej'.
        """
        # Camera topics
        self.img_left_topic = img_left_topic
        self.img_right_topic = img_right_topic
        self.img_top_topic = img_top_topic
        self.img_left_depth_topic = img_left_depth_topic
        self.img_right_depth_topic = img_right_depth_topic
        self.img_top_depth_topic = img_top_depth_topic

        # Joint state topics (16 DOF: 14 arm + 2 head)
        self.joint_state_topic = joint_state_topic

        # Gripper topics
        self.gripper_state_topic = gripper_state_topic

        # End-effector pose topics
        self.ee_pose_left_topic = ee_pose_left_topic
        self.ee_pose_right_topic = ee_pose_right_topic

        # Feature flags
        self.use_depth_image = use_depth_image
        self.enable_base_control = enable_base_control
        self.trajectory_exec_mode = trajectory_exec_mode

        # WebSocket configuration
        self.robot_ip = robot_ip
        self.ws_port = ws_port
        self.ws_client = None
        self.ws_accid = ws_accid  # None means auto-detect from server
        self.ws_connected = False
        self.ws_lock = threading.Lock()

        # Set default arm step lengths if not provided (14 DOF)
        if arm_steps_length is None:
            arm_steps_length = [0.02] * 14
        self.arm_steps_length = arm_steps_length

        # ServoJ gains
        # Per-arm: abad, hip, yaw, knee, wrist_yaw, wrist_pitch, wrist_roll
        self.servoj_kp = [
            420, 420, 300, 300, 200, 200, 200, 420, 420, 300, 300, 200, 200,
            200, 60, 60
        ]
        self.servoj_kd = [
            12, 12, 15, 15, 10, 10, 10, 12, 12, 15, 15, 10, 10, 10, 3, 3
        ]
        self.servoj_frequency = 500  # Hz, minimum for stable control

        # Tron2 joint names
        self.joint_names = [
            'abad_L_Joint',
            'hip_L_Joint',
            'yaw_L_Joint',
            'knee_L_Joint',
            'wrist_yaw_L_Joint',
            'wrist_pitch_L_Joint',
            'wrist_roll_L_Joint',
            'abad_R_Joint',
            'hip_R_Joint',
            'yaw_R_Joint',
            'knee_R_Joint',
            'wrist_yaw_R_Joint',
            'wrist_pitch_R_Joint',
            'wrist_roll_R_Joint',
            'head_pitch_Joint',
            'head_yaw_Joint',
        ]
        self.gripper_names = ['left_gripper', 'right_gripper']

        # Initialize ROS for receiving sensor data
        self._init_ros()

        # Initialize WebSocket for robot control
        self._init_websocket()

        self.json_encoder = NumpySafeEncoder

    # ========== Public Control API ==========

    def execute_step(self,
                     left,
                     right,
                     head=None,
                     left_gripper=None,
                     right_gripper=None,
                     duration: float = 0.1):
        """Execute a single-step command."""
        self._send_joints(left=left, right=right, duration=duration)
        self._send_gripper(
            left_opening=left_gripper, right_opening=right_gripper)
        self._send_head(head_positions=head, duration=duration)

    def move_to_targets(self,
                        left,
                        right,
                        control_rate: int = 30,
                        left_gripper: float = None,
                        right_gripper: float = None,
                        head: List[float] = None):
        """Move to target state with interpolation."""
        rate_period = 1.0 / control_rate

        self._send_head(head_positions=head, duration=5.0)
        if left_gripper is not None or right_gripper is not None:
            self._send_gripper(
                left_opening=left_gripper, right_opening=right_gripper)

        left_arm = None
        right_arm = None
        while True:
            if len(self.joint_state_deque) > 0:
                msg = self.joint_state_deque[-1]
                positions = list(msg.position) if msg.position else [0.0] * 16
                left_arm, right_arm = positions[:7], positions[7:14]
                break
            time.sleep(rate_period)

        left_symbol = [
            1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))
        ]
        right_symbol = [
            1 if right[i] - right_arm[i] > 0 else -1
            for i in range(len(right))
        ]

        flag = True
        while flag:
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [
                abs(right[i] - right_arm[i]) for i in range(len(right))
            ]

            flag = False

            for i in range(len(left)):
                if left_diff[i] < self.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.arm_steps_length[i]
                    flag = True

            for i in range(len(right)):
                if right_diff[i] < self.arm_steps_length[7 + i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.arm_steps_length[7 +
                                                                            i]
                    flag = True

            self._send_joints(
                left=left_arm, right=right_arm, duration=rate_period)

            time.sleep(rate_period)

    def execute_trajectory(self,
                           left_arm_trajectory,
                           right_arm_trajectory,
                           left_gripper_trajectory,
                           right_gripper_trajectory,
                           head_trajectory=None,
                           dt: float = 0.1,
                           async_exec: bool = False):
        """Execute trajectory in sync or async mode."""
        self._traj_stop_event.set()
        self._traj_stop_event = threading.Event()
        stop_event = self._traj_stop_event

        args = (left_arm_trajectory, right_arm_trajectory,
                left_gripper_trajectory, right_gripper_trajectory,
                head_trajectory, dt, stop_event)

        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_trajectory, args=args, daemon=True)
            self._traj_thread.start()
        else:
            self._run_trajectory(*args)

    def stop_trajectory(self):
        """Stop the currently executing trajectory."""
        self._traj_stop_event.set()

    def is_trajectory_running(self):
        """Check if a trajectory is currently being executed."""
        return (self._traj_thread is not None and self._traj_thread.is_alive())

    def set_light_effect(self, effect: int):
        """Set robot light effect via WebSocket."""
        self._ws_send_request('request_light_effect', {'effect': effect})

    def emergency_stop(self):
        """Trigger emergency stop via WebSocket."""
        self._ws_send_request('request_emgy_stop', {})

    def close(self):
        """Close WebSocket connection."""
        if self.ws_client:
            self.ws_client.close()
            self.ws_connected = False

    # ========== Motion Execution Helpers ==========

    def _run_trajectory(self, left_arm_trajectory, right_arm_trajectory,
                        left_gripper_trajectory, right_gripper_trajectory,
                        head_trajectory, dt, stop_event):
        """Run trajectory in selected execution mode."""
        if stop_event.is_set():
            return

        n = len(left_arm_trajectory)
        if n == 0:
            return

        if self.trajectory_exec_mode == 'movej':
            self._run_trajectory_movej(
                left_arm_trajectory=left_arm_trajectory,
                right_arm_trajectory=right_arm_trajectory,
                left_gripper_trajectory=left_gripper_trajectory,
                right_gripper_trajectory=right_gripper_trajectory,
                head_trajectory=head_trajectory,
                dt=dt,
                stop_event=stop_event)
        elif self.trajectory_exec_mode == 'servoj':
            self._run_trajectory_servoj(
                left_arm_trajectory=left_arm_trajectory,
                right_arm_trajectory=right_arm_trajectory,
                left_gripper_trajectory=left_gripper_trajectory,
                right_gripper_trajectory=right_gripper_trajectory,
                head_trajectory=head_trajectory,
                dt=dt,
                stop_event=stop_event)
        else:
            raise ValueError('Unsupported trajectory_exec_mode: '
                             f'{self.trajectory_exec_mode}. '
                             "Expected one of {'movej', 'servoj'}")

    def _run_trajectory_movej(self, left_arm_trajectory, right_arm_trajectory,
                              left_gripper_trajectory,
                              right_gripper_trajectory, head_trajectory, dt,
                              stop_event):
        """Run trajectory with movej step execution."""
        n = len(left_arm_trajectory)
        for i in range(n):
            if stop_event.is_set():
                return
            head = head_trajectory[i] if head_trajectory is not None else None
            self._send_joints(
                left=left_arm_trajectory[i],
                right=right_arm_trajectory[i],
                duration=dt)
            self._send_gripper(
                left_opening=float(left_gripper_trajectory[i]),
                right_opening=float(right_gripper_trajectory[i]))
            self._send_head(head_positions=head, duration=dt)
            time.sleep(dt)

    def _resolve_head_positions(self,
                                head_positions: Optional[List[float]] = None
                                ) -> List[float]:
        """Resolve head positions for ServoJ.

        Priority:
            1. Explicitly provided head positions.
            2. Latest head joint state.
            3. Default head joint positions.
        """
        if head_positions is not None and len(head_positions) >= 2:
            return [float(head_positions[0]), float(head_positions[1])]

        if len(self.joint_state_deque) > 0:
            msg = self.joint_state_deque[-1]
            positions = list(msg.position) if msg.position else None
            if positions is not None and len(positions) >= 2:
                return [float(positions[-2]), float(positions[-1])]

        return list(DEFAULT_HEAD_JOINT_POSITIONS)

    def _run_trajectory_servoj(self, left_arm_trajectory, right_arm_trajectory,
                               left_gripper_trajectory,
                               right_gripper_trajectory, head_trajectory, dt,
                               stop_event):
        """Run trajectory with servoj interpolation execution.

        ServoJ requires 16-dim commands: 14 arm joints + 2 head joints.
        Head is included in the servo command directly (no separate
        ``_send_head`` call).
        """
        n = len(left_arm_trajectory)

        # When head_trajectory is not provided, hold the current head position
        # so the head does not jump to zero.
        if head_trajectory is None:
            current_head = self._resolve_head_positions()
            if len(self.joint_state_deque) == 0:
                print('[WARN] No joint state available, '
                      f'head defaults to {DEFAULT_HEAD_JOINT_POSITIONS}')

        # Build 16-dim waypoints: left_arm(7) + right_arm(7) + head(2)
        wps = []
        for i in range(n):
            head = (
                self._resolve_head_positions(head_trajectory[i])
                if head_trajectory is not None else current_head)
            wps.append(
                list(left_arm_trajectory[i]) + list(right_arm_trajectory[i]) +
                head)

        servo_dt = 1.0 / self.servoj_frequency
        steps_per_seg = max(int(dt * self.servoj_frequency), 1)

        t0 = time.perf_counter()
        global_step = 0

        for seg in range(len(wps) - 1):
            if stop_event.is_set():
                return
            q0, q1 = wps[seg], wps[seg + 1]
            self._send_gripper(
                left_opening=float(left_gripper_trajectory[seg]),
                right_opening=float(right_gripper_trajectory[seg]))

            for step in range(1, steps_per_seg + 1):
                q_cmd = [
                    a + (step / steps_per_seg) * (b - a)
                    for a, b in zip(q0, q1)
                ]
                global_step += 1
                self._servo_step(q_cmd, t0, global_step, servo_dt)

        self._send_gripper(
            left_opening=float(left_gripper_trajectory[n - 1]),
            right_opening=float(right_gripper_trajectory[n - 1]))

        return

    # ========== Motion Execution ==========

    def _servo_step(self, q_cmd: List[float], t0: float, step: int,
                    servo_dt: float):
        """Send one servo command and wait for the next tick.

        Args:
            q_cmd: Joint positions in our convention (n-dim).
            t0: Trajectory start time (perf_counter).
            step: 1-based global step index.
            servo_dt: Servo period in seconds.
        """
        n = len(q_cmd)
        self._ws_send_request(
            'request_servoj', {
                'q': list(q_cmd),
                'v': [0.0] * n,
                'kp': self.servoj_kp[:n],
                'kd': self.servoj_kd[:n],
                'tau': [0.0] * n,
                'mode': [0] * n,
                'na': 0,
            })

        target_time = t0 + step * servo_dt
        remaining = target_time - time.perf_counter()
        if remaining > 1e-3:
            time.sleep(remaining - 1e-3)
        while time.perf_counter() < target_time:
            pass

    # ========== WebSocket Initialization ==========

    def _init_websocket(self):
        """Initialize WebSocket connection using WebSocketApp (async mode)."""
        self.ws_url = f'ws://{self.robot_ip}:{self.ws_port}'

        # Create WebSocketApp with callbacks
        self.ws_client = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._ws_on_open,
            on_message=self._ws_on_message,
            on_close=self._ws_on_close,
            on_error=self._ws_on_error)

        # Configure socket send and receive buffer sizes (2MB each)
        # Helps prevent data loss when sending high-frequency data
        self.ws_client.sock_opt = [('socket', 'SO_SNDBUF', 2 * 1024 * 1024),
                                   ('socket', 'SO_RCVBUF', 2 * 1024 * 1024)]

        # Start WebSocket in background thread
        self._ws_thread = threading.Thread(
            target=self._ws_run_forever, daemon=True)
        self._ws_thread.start()

        # Wait for connection
        timeout = 5.0
        start_time = time.time()
        while not self.ws_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.ws_connected:
            print(f'Tron2Operator WebSocket connected to {self.ws_url}')
        else:
            raise ConnectionError(
                f'Tron2Operator WebSocket connection timeout to {self.ws_url}')

        # If ws_accid was not provided, wait for auto-detection from server
        if self.ws_accid is None:
            accid_timeout = 5.0
            start_time = time.time()
            while (self.ws_accid is None
                   and (time.time() - start_time) < accid_timeout):
                time.sleep(0.1)
            if self.ws_accid is not None:
                print(f'Tron2Operator auto-detected ws_accid: {self.ws_accid}')
            else:
                print('Tron2Operator WARNING: ws_accid auto-detection '
                      'timed out, control commands may not work')

    def _ws_run_forever(self):
        """Run WebSocket client loop in background thread."""
        try:
            self.ws_client.run_forever()
        except Exception as e:
            print(f'WebSocket run_forever error: {e}')

    def _ws_on_open(self, ws):
        """WebSocket on_open callback."""
        self.ws_connected = True

    def _ws_on_message(self, ws, message):
        """WebSocket on_message callback - handles all incoming messages."""
        try:
            response = json.loads(message)
            title = response.get('title', '')
            resp_data = response.get('data', {})
            recv_accid = response.get('accid', None)

            # Auto-detect accid from the first server message
            if self.ws_accid is None and recv_accid is not None:
                self.ws_accid = recv_accid

            # Filter messages by accid - only process messages for this robot
            if recv_accid != self.ws_accid:
                return

            # Skip frequent robot info notifications
            if title == 'notify_robot_info':
                return

            # Check for invalid request notification
            if title == 'notify_invalid_request':
                print(f'WebSocket invalid request: {resp_data}')
                return

            # Check result for command responses
            if isinstance(resp_data, dict) and 'result' in resp_data:
                result = resp_data['result']
                if result != 'success':
                    print(f'WebSocket command failed [{title}]: {result}')

        except json.JSONDecodeError:
            print(f'WebSocket invalid JSON: {message}')

    def _ws_on_close(self, ws, close_status_code, close_msg):
        """WebSocket on_close callback."""
        self.ws_connected = False
        print(f'WebSocket closed: {close_status_code} - {close_msg}')

    def _ws_on_error(self, ws, error):
        """WebSocket on_error callback."""
        print(f'WebSocket error: {error}')

    def _generate_ws_guid(self) -> str:
        """Generate a unique GUID for WebSocket requests.

        Returns:
            str: UUID string
        """
        return str(uuid.uuid4())

    def _ws_send_request(self, title: str, data: dict = None):
        """Send a WebSocket request to the robot (non-blocking).

        Args:
            title: Request title/type
            data: Request data dictionary
        """
        if data is None:
            data = {}

        message = {
            'accid': self.ws_accid,
            'title': title,
            'timestamp': int(time.time() * 1000),
            'guid': self._generate_ws_guid(),
            'data': data
        }

        with self.ws_lock:
            try:
                if self.ws_client and self.ws_connected:
                    self.ws_client.send(
                        json.dumps(message, cls=self.json_encoder))
            except Exception as e:
                print(f'WebSocket send error: {e}')

    # ========== ROS Initialization ==========

    def _init_ros(self):
        """Initialize ROS node, subscribers, and runtime state."""
        import rospy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import Image, JointState

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Initialize message queues for images
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_top_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_top_depth_deque = deque()

        # Initialize message queues for robot states
        self.joint_state_deque = deque()
        self.joint_cmd_deque = deque()
        self.gripper_state_deque = deque()

        # End-effector pose queues
        self.ee_pose_left_deque = deque()
        self.ee_pose_right_deque = deque()

        # Motion execution state (sync/async interface)
        self._traj_thread = None
        self._traj_stop_event = threading.Event()

        # Initialize ROS node if not already done
        if rospy.get_name() == '/unnamed':
            rospy.init_node('tron2_operator_node', anonymous=True)

        # Subscribe to RGB image topics
        rospy.Subscriber(self.img_left_topic, Image, self.img_left_callback)
        rospy.Subscriber(self.img_right_topic, Image, self.img_right_callback)
        rospy.Subscriber(self.img_top_topic, Image, self.img_top_callback)

        # Subscribe to depth image topics if enabled
        if self.use_depth_image:
            rospy.Subscriber(self.img_left_depth_topic, Image,
                             self.img_left_depth_callback)
            rospy.Subscriber(self.img_right_depth_topic, Image,
                             self.img_right_depth_callback)
            rospy.Subscriber(self.img_top_depth_topic, Image,
                             self.img_top_depth_callback)

        # Subscribe to joint state topic (16 DOF)
        rospy.Subscriber(self.joint_state_topic, JointState,
                         self.joint_state_callback)

        # Subscribe to gripper state topic
        rospy.Subscriber(self.gripper_state_topic, JointState,
                         self.gripper_state_callback)

        # Subscribe to end-effector pose topics
        rospy.Subscriber(self.ee_pose_left_topic, PoseStamped,
                         self.ee_pose_left_callback)
        rospy.Subscriber(self.ee_pose_right_topic, PoseStamped,
                         self.ee_pose_right_callback)

        rospy.loginfo('Tron2Operator ROS node initialized')
        rospy.loginfo(f'  Joint state topic: {self.joint_state_topic}')
        rospy.loginfo(f'  Gripper state topic: {self.gripper_state_topic}')

    # ========== Sensor Data Methods ==========

    def get_frame(self):
        """Get synchronized frame data from all sensors.

        Synchronizes RGB images, depth images (if enabled), and joint states
        for both arms based on timestamps.

        Returns:
            tuple or False: If successful, returns tuple containing:
                (img_top, img_left, img_right, img_top_depth,
                 img_left_depth, img_right_depth, left_arm,
                 right_arm, head, gripper)
                If failed, returns False.
        """
        # Check if all required RGB image queues have data
        if (len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0
                or len(self.img_top_deque) == 0):
            return False

        # Check depth image queues if depth is enabled
        if self.use_depth_image:
            if (len(self.img_left_depth_deque) == 0
                    or len(self.img_right_depth_deque) == 0
                    or len(self.img_top_depth_deque) == 0):
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
            self.img_top_deque[-1].header.stamp.to_sec()
        ]

        if self.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_right_depth_deque[-1].header.stamp.to_sec(),
                self.img_top_depth_deque[-1].header.stamp.to_sec()
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
        if (len(self.img_top_deque) == 0
                or self.img_top_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        # Check joint state availability
        if (len(self.joint_state_deque) == 0 or
                self.joint_state_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.gripper_state_deque) == 0
                or self.gripper_state_deque[-1].header.stamp.to_sec() <
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
            if (len(self.img_top_depth_deque) == 0
                    or self.img_top_depth_deque[-1].header.stamp.to_sec() <
                    frame_time):
                return False

        return True

    def _synchronize_and_extract_data(self, frame_time):
        """Synchronize queues and extract data at the specified frame time.

        Args:
            frame_time (float): Target synchronization timestamp

        Returns:
            tuple: Synchronized sensor data containing:
                (img_top, img_left, img_right, img_top_depth,
                 img_left_depth, img_right_depth, left_arm,
                 right_arm, head, gripper)
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

        while self.img_top_deque[0].header.stamp.to_sec() < frame_time:
            self.img_top_deque.popleft()
        img_top = self.bridge.imgmsg_to_cv2(self.img_top_deque.popleft(),
                                            'passthrough')

        # Extract from merged joint_state_deque (16 DOF: 14 arm + 2 head)
        while self.joint_state_deque[0].header.stamp.to_sec() < frame_time:
            self.joint_state_deque.popleft()
        joint_state_msg = self.joint_state_deque.popleft()

        # Split 16 DOF: left(0-6), right(7-13), head(14-15)
        positions = list(joint_state_msg.position
                         ) if joint_state_msg.position else [0.0] * 16
        velocities = list(joint_state_msg.velocity
                          ) if joint_state_msg.velocity else [0.0] * 16
        efforts = list(
            joint_state_msg.effort) if joint_state_msg.effort else [0.0] * 16

        # Create left/right arm data structures
        left_arm = type(
            'JointState', (), {
                'position': positions[:7],
                'velocity': velocities[:7],
                'effort': efforts[:7],
                'header': joint_state_msg.header
            })()
        right_arm = type(
            'JointState', (), {
                'position': positions[7:14],
                'velocity': velocities[7:14],
                'effort': efforts[7:14],
                'header': joint_state_msg.header
            })()
        head = type(
            'JointState', (), {
                'position': positions[14:16],
                'velocity': velocities[14:16],
                'effort': efforts[14:16],
                'header': joint_state_msg.header
            })()

        # Gripper: hw 0-100 (0=closed, 100=open); /100 -> 0-1 for obs/model
        while self.gripper_state_deque[0].header.stamp.to_sec() < frame_time:
            self.gripper_state_deque.popleft()
        gripper = self.gripper_state_deque.popleft()
        gripper.position = [p / 100 for p in gripper.position]

        # Extract depth images if enabled
        img_left_depth = None
        img_right_depth = None
        img_top_depth = None
        if self.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec(
            ) < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(
                self.img_left_depth_deque.popleft(), 'passthrough')

            while self.img_right_depth_deque[0].header.stamp.to_sec(
            ) < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(
                self.img_right_depth_deque.popleft(), 'passthrough')

            while self.img_top_depth_deque[0].header.stamp.to_sec(
            ) < frame_time:
                self.img_top_depth_deque.popleft()
            img_top_depth = self.bridge.imgmsg_to_cv2(
                self.img_top_depth_deque.popleft(), 'passthrough')

        return (img_top, img_left, img_right, img_top_depth, img_left_depth,
                img_right_depth, left_arm, right_arm, head, gripper)

    # ========== ROS Callbacks ==========

    def img_left_callback(self, msg):
        """Callback for left camera RGB image."""
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        """Callback for right camera RGB image."""
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_top_callback(self, msg):
        """Callback for top camera RGB image."""
        if len(self.img_top_deque) >= 2000:
            self.img_top_deque.popleft()
        self.img_top_deque.append(msg)

    def img_left_depth_callback(self, msg):
        """Callback for left camera depth image."""
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        """Callback for right camera depth image."""
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_top_depth_callback(self, msg):
        """Callback for top camera depth image."""
        if len(self.img_top_depth_deque) >= 2000:
            self.img_top_depth_deque.popleft()
        self.img_top_depth_deque.append(msg)

    def joint_state_callback(self, msg):
        """Callback for merged joint states (16 DOF: 14 arm + 2 head)."""
        if len(self.joint_state_deque) >= 2000:
            self.joint_state_deque.popleft()
        self.joint_state_deque.append(msg)

    def joint_cmd_callback(self, msg):
        """Callback for merged joint commands (16 DOF: 14 arm + 2 head)."""
        if len(self.joint_cmd_deque) >= 2000:
            self.joint_cmd_deque.popleft()
        self.joint_cmd_deque.append(msg)

    def gripper_state_callback(self, msg):
        """Gripper state callback. Hw 0-100; /100 in get_ros_observation and
        get_latest_gripper_state."""
        if len(self.gripper_state_deque) >= 2000:
            self.gripper_state_deque.popleft()
        self.gripper_state_deque.append(msg)

    def ee_pose_left_callback(self, msg):
        """Callback for left arm end-effector pose."""
        if len(self.ee_pose_left_deque) >= 2000:
            self.ee_pose_left_deque.popleft()
        self.ee_pose_left_deque.append(msg)

    def ee_pose_right_callback(self, msg):
        """Callback for right arm end-effector pose."""
        if len(self.ee_pose_right_deque) >= 2000:
            self.ee_pose_right_deque.popleft()
        self.ee_pose_right_deque.append(msg)

    # ========== Command Helpers ==========

    def _send_gripper(self,
                      left_opening: float = None,
                      right_opening: float = None,
                      left_speed: float = 100,
                      right_speed: float = 100,
                      left_force: float = 100,
                      right_force: float = 100):
        """Command gripper via WebSocket."""
        data = {}

        if left_opening is not None:
            data['left_opening'] = float(left_opening * 100)
            data['left_speed'] = float(left_speed)
            data['left_force'] = float(left_force)

        if right_opening is not None:
            data['right_opening'] = float(right_opening * 100)
            data['right_speed'] = float(right_speed)
            data['right_force'] = float(right_force)

        if data:
            self._ws_send_request('request_set_limx_2fclaw_cmd', data)

    def _send_head(self,
                   head_positions: Optional[List[float]] = None,
                   duration: float = 5.0):
        """Command head joints via WebSocket."""
        if head_positions is None:
            return
        self._ws_send_request('request_moveh', {
            'joint': list(head_positions),
            'time': duration
        })

    def _send_joints(self, left, right, duration: float):
        """Command arm joints via WebSocket movej in position mode."""
        self._ws_send_request('request_movej', {
            'joint': list(left) + list(right),
            'time': duration
        })

    def _send_base_velocity(self, vel):
        """TODO: support base command for unified operator interface."""
        if vel is None:
            return
        if not self.enable_base_control:
            return

    def _send_ee_pose(self,
                      left_pose=None,
                      right_pose=None,
                      duration: float = 5.0):
        """Command end-effector poses via WebSocket."""
        pose_data = []

        if left_pose is not None:
            pose_data.extend(left_pose['position'])
            pose_data.extend(left_pose['orientation'])
        else:
            pose_data.extend([0.0] * 12)

        if right_pose is not None:
            pose_data.extend(right_pose['position'])
            pose_data.extend(right_pose['orientation'])
        else:
            pose_data.extend([0.0] * 12)

        self._ws_send_request('request_movep', {
            'pos': list(pose_data),
            'time': duration
        })

    # ========== Utility Methods ==========

    def get_current_joint_states(self):
        """Get latest left/right arm positions as numpy arrays."""
        if len(self.joint_state_deque) == 0:
            return None, None
        msg = self.joint_state_deque[-1]
        positions = list(msg.position) if msg.position else [0.0] * 16
        return np.array(positions[:7]), np.array(positions[7:14])

    def get_latest_gripper_state(self) -> Optional[Tuple[float, float]]:
        """Latest gripper state mapped to [0, 1]."""
        if len(self.gripper_state_deque) == 0:
            return None
        msg = self.gripper_state_deque[-1]
        positions = list(msg.position) if msg.position else [0.0, 0.0]
        return (positions[0] / 100,
                positions[1] / 100) if len(positions) >= 2 else (0.0, 0.0)

    def get_latest_ee_pose(self) -> Optional[Tuple[dict, dict]]:
        """Get the latest end-effector poses."""
        if len(self.ee_pose_left_deque) == 0 or len(
                self.ee_pose_right_deque) == 0:
            return None

        left_msg = self.ee_pose_left_deque[-1]
        right_msg = self.ee_pose_right_deque[-1]

        left_pose = {
            'position': [
                left_msg.pose.position.x, left_msg.pose.position.y,
                left_msg.pose.position.z
            ],
            'orientation': [
                left_msg.pose.orientation.x, left_msg.pose.orientation.y,
                left_msg.pose.orientation.z, left_msg.pose.orientation.w
            ]
        }
        right_pose = {
            'position': [
                right_msg.pose.position.x, right_msg.pose.position.y,
                right_msg.pose.position.z
            ],
            'orientation': [
                right_msg.pose.orientation.x, right_msg.pose.orientation.y,
                right_msg.pose.orientation.z, right_msg.pose.orientation.w
            ]
        }

        return (left_pose, right_pose)
