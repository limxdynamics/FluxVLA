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
"""
Tron2 WebSocket control test script.
Connects to robot WS, sends movej/movep/light/stop (same protocol as
Tron2Operator). Requires: websocket-client.
"""

import json
import signal
import threading
import time
import uuid
from datetime import datetime

import websocket  # type: ignore

# Robot accid (SN); if None, taken from first server message
ACCID: str | None = None

# Robot IP/port (same defaults as Tron2Operator)
ROBOT_IP = '10.192.1.2'
ROBOT_PORT = 5000

should_exit = False
ws_client: websocket.WebSocketApp | None = None

# Latest joint state received from robot
_joint_state_lock = threading.Lock()
_latest_joint_state: dict | None = None
_joint_state_event = threading.Event()  # signalled when a new state arrives


def generate_guid() -> str:
    return str(uuid.uuid4())


def send_request(title: str, data: dict | None = None) -> None:
    """Send one WebSocket request."""
    global ACCID, ws_client

    if ws_client is None:
        print(f'[WARN] WebSocket not connected, cannot send: {title}')
        return

    if data is None:
        data = {}

    message = {
        'accid': ACCID,
        'title': title,
        'timestamp': int(time.time() * 1000),
        'guid': generate_guid(),
        'data': data,
    }

    message_str = json.dumps(message, ensure_ascii=False)
    try:
        ws_client.send(message_str)
        print(f'[SEND] {title}: {message_str}')
    except Exception as e:  # noqa: BLE001
        print(f'[ERROR] Send failed: {e}')


def send_get_joint_state() -> dict | None:
    """Request current joint state and wait for the response.

    Returns the joint state dict on success, or None on timeout.
    """
    _joint_state_event.clear()
    send_request('request_get_joint_state', {})
    # Wait up to 2 s for the response
    if _joint_state_event.wait(timeout=2.0):
        with _joint_state_lock:
            return _latest_joint_state
    print('[WARN] Timed out waiting for joint state.')
    return None


def print_joint_state(state: dict | None = None) -> None:
    """Pretty-print a joint state dict."""
    if state is None:
        state = send_get_joint_state()
    if state is None:
        return

    result = state.get('result', 'unknown')
    if result != 'success':
        print(f'[JOINT] Query failed: {result}')
        return

    names = state.get('names', [])
    q = state.get('q', [])
    dq = state.get('dq', [])
    tau = state.get('tau', [])

    print('\n' + '=' * 72)
    print(
        f"{'Joint':<25} {'Pos (rad)':>12} {'Vel (rad/s)':>12} {'Torque':>12}")
    print('-' * 72)
    for i, name in enumerate(names):
        pos = f'{q[i]:.4f}' if i < len(q) else 'N/A'
        vel = f'{dq[i]:.4f}' if i < len(dq) else 'N/A'
        trq = f'{tau[i]:.4f}' if i < len(tau) else 'N/A'
        print(f'{name:<25} {pos:>12} {vel:>12} {trq:>12}')
    print('=' * 72 + '\n')


def send_servoj(
    q: list[float],
    v: list[float] | None = None,
    kp: list[float] | None = None,
    kd: list[float] | None = None,
    tau: list[float] | None = None,
    mode: list[int] | None = None,
    na: int = 0,
) -> None:
    """Send a single ServoJ control frame.

    Args:
        q:    Target joint positions (14-dim).
        v:    Target joint velocities (14-dim, default zeros).
        kp:   Position gains (14-dim, default 100 each).
        kd:   Derivative gains (14-dim, default 10 each).
        tau:  Feed-forward torques (14-dim, default zeros).
        mode: Control mode per joint (14-dim, default zeros).
        na:   Reserved field (default 0).
    """
    # Per-arm: abad, hip, yaw, knee, wrist_yaw/pitch/roll
    _default_kp = [
        420, 420, 300, 300, 200, 200, 200, 420, 420, 300, 300, 200, 200, 200,
        60, 60
    ]
    _default_kd = [
        12, 12, 15, 15, 10, 10, 10, 12, 12, 15, 15, 10, 10, 10, 3, 3
    ]

    n = len(q)
    if v is None:
        v = [0.0] * n
    if kp is None:
        kp = _default_kp[:n]
    if kd is None:
        kd = _default_kd[:n]
    if tau is None:
        tau = [0.0] * n
    if mode is None:
        mode = [0] * n

    send_request(
        'request_servoj',
        {
            'q': q,
            'v': v,
            'kp': kp,
            'kd': kd,
            'tau': tau,
            'mode': mode,
            'na': na,
        },
    )


def run_servoj_demo(
    duration: float = 5.0,
    frequency: float = 500.0,
) -> None:
    """Interpolate from current pose to zero over *duration* s
    using ServoJ.
    """
    global should_exit

    dt = 1.0 / frequency
    num_joints = 16  # 14 arm joints + 2 head joints

    kp = [
        420, 420, 300, 300, 200, 200, 200, 420, 420, 300, 300, 200, 200, 200,
        60, 60
    ]
    kd = [12, 12, 15, 15, 10, 10, 10, 12, 12, 15, 15, 10, 10, 10, 3, 3]

    # Read current joint positions as start (16-dim including head)
    state = send_get_joint_state()
    if state is not None and state.get('result') == 'success':
        start_q = list(state['q'])[:num_joints]
        print(f'[SERVOJ] Start q: {[round(v, 4) for v in start_q]}')
    else:
        print('[SERVOJ] Failed to read joint state, aborting.')
        return

    # Target: arm joints go to zero, head joints stay at current position
    head_q = start_q[14:16]
    target_q = [0.0] * 14 + head_q

    steps = int(duration * frequency)
    head_str = [round(v, 4) for v in head_q]
    print(f'[SERVOJ] Interpolating arm to zero '
          f'(head held at {head_str}): '
          f'{steps} steps @ {frequency} Hz for {duration} s')

    for step in range(steps):
        if should_exit:
            break
        t0 = time.monotonic()

        # Linear interpolation: alpha goes from 0 to 1
        alpha = (step + 1) / steps
        q_cmd = [s + alpha * (t - s) for s, t in zip(start_q, target_q)]

        send_servoj(q=q_cmd, kp=kp, kd=kd)

        elapsed = time.monotonic() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print('[SERVOJ] Done.')


def handle_commands() -> None:
    """Handle user input in a separate thread."""
    global should_exit

    help_text = (
        'Commands: movej, moveh, movep, servoj, state, light, stop, exit\n')
    print(help_text)

    while not should_exit:
        try:
            command = input(
                'Command (movej/moveh/movep/servoj/state/light/stop/exit): ',
            ).strip().lower()
        except EOFError:
            should_exit = True
            break

        if command == 'exit':
            should_exit = True
            break

        if command == 'movej':
            # Target arm position (14 joints)
            joint_target = [1.2, 0, 0, -2.5, 0, 0, 0, 1.2, 0, 0, -2.5, 0, 0, 0]
            send_request(
                'request_movej',
                {
                    'joint': joint_target,
                    'time': 5
                },
            )
        elif command == 'moveh':
            # Head control: [pitch, yaw] in radians
            send_request(
                'request_moveh',
                {
                    'joint': [0.5, 0.5],
                    'time': 5
                },
            )
        elif command == 'movep':
            send_request(
                'request_movep',
                {
                    'pos': [
                        0.3,
                        0.2,
                        -0.3,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0.3,
                        -0.2,
                        -0.3,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                    ],
                    'time':
                    5,
                },
            )
        elif command == 'light':
            send_request(
                'request_light_effect',
                {'effect': 1},
            )
        elif command == 'state':
            print_joint_state()
        elif command == 'servoj':
            run_servoj_demo(duration=5.0, frequency=500.0)
        elif command == 'stop':
            send_request('request_emgy_stop', {})
        else:
            print('Unknown command.')
            print(help_text)


def on_open(ws: websocket.WebSocketApp) -> None:  # noqa: ARG001
    print('[INFO] WebSocket connected.')
    threading.Thread(target=handle_commands, daemon=True).start()


def on_message(ws: websocket.WebSocketApp,
               message: str) -> None:  # noqa: ARG001
    global ACCID, _latest_joint_state

    try:
        root = json.loads(message)
    except json.JSONDecodeError:
        print(f'[RECV] Not JSON: {message}')
        return

    title = root.get('title', '')
    data = root.get('data', {})
    accid_from_msg = root.get('accid')

    if ACCID is None and accid_from_msg is not None:
        ACCID = accid_from_msg
        print(f'[INFO] ACCID from server: {ACCID}')

    # ── Handle specific response / notification titles ──
    if title == 'response_get_joint_state':
        with _joint_state_lock:
            _latest_joint_state = data
        _joint_state_event.set()
        result = data.get('result', 'unknown')
        if result != 'success':
            print(f'[JOINT] Query failed: {result}')
        return

    if title == 'notify_servoJ':
        result = data.get('result', 'unknown')
        reason = {
            'fail_invalid_cmd': 'Invalid ServoJ command',
            'fail_motor': 'Motor error',
        }.get(result, result)
        print(f'\n[SERVOJ-NOTIFY] ServoJ failure: {reason}')
        return

    # Suppress high-frequency robot info broadcasts
    if title == 'notify_robot_info':
        return

    ts = datetime.now().isoformat()
    print(f'[RECV] {ts} title={title} data={data}')


def on_error(ws: websocket.WebSocketApp,
             error: Exception) -> None:  # noqa: ARG001
    print(f'[ERROR] WebSocket: {error}')


def on_close(
    ws: websocket.WebSocketApp,  # noqa: ARG001
    close_status_code: int | None,
    close_msg: str | None,
) -> None:
    print(f'[INFO] WebSocket closed: {close_status_code} {close_msg}')


def close_connection() -> None:
    """Close WebSocket connection."""
    global ws_client
    if ws_client is not None:
        try:
            ws_client.close()
        except Exception:
            pass
        ws_client = None


def signal_handler(signum: int, frame) -> None:  # noqa: ANN001, D401, ARG001
    """Handle SIGINT/SIGTERM for clean exit."""
    global should_exit
    print(f'\n[INFO] Signal {signum}, exiting...')
    should_exit = True
    close_connection()


def main() -> None:
    """Connect WebSocket and run event loop."""
    global ws_client

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    url = f'ws://{ROBOT_IP}:{ROBOT_PORT}'
    print(f'[INFO] Connecting to {url}')
    print("[INFO] Ctrl+C or 'exit' to quit.")

    ws_client = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error,
    )

    ws_client.sock_opt = [
        ('socket', 'SO_SNDBUF', 2 * 1024 * 1024),
        ('socket', 'SO_RCVBUF', 2 * 1024 * 1024),
    ]

    ws_client.run_forever()


if __name__ == '__main__':
    main()
