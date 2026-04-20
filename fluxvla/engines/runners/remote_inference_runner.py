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
"""ZMQ remote inference runner.

Inherits from BaseInferenceRunner but replaces local model inference with
a ZMQ client that delegates to a remote GPU server.  Robot-specific
observation and execution are provided by subclasses via multiple
inheritance (e.g. RemoteURInferenceRunner, RemoteAlohaInferenceRunner).
"""
from __future__ import annotations
import io
import threading
import time
from typing import Literal

import msgpack
import numpy as np
import torch
import zmq

from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .base_inference_runner import BaseInferenceRunner
from .serving.serializers import (FORMAT_PROTOBUF, decode_predict_response,
                                  encode_predict_request)

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class RemoteInferenceRunner(BaseInferenceRunner):
    """Base class for remote inference runners.

    Replaces local model with a ZMQ client that delegates inference to a
    remote GPU server.  Overrides ``run_setup``, ``_preprocess``,
    ``_predict_action``, and ``_postprocess_actions`` so that subclasses
    only need to provide robot-specific observation and execution methods.

    Designed for multiple inheritance::

        class RemoteURInferenceRunner(
            RemoteInferenceRunner, URInferenceRunner):
            pass
    """

    def __init__(
        self,
        server_host: str = 'localhost',
        server_port: int = 5555,
        timeout_s: float = 30.0,
        serializer: Literal['msgpack', 'protobuf'] = 'msgpack',
        compress: bool = True,
        enable_profiling: bool = True,
        **kwargs,
    ):
        """
        Args:
            server_host: Remote GPU server hostname or IP.
            server_port: Remote GPU server port.
            timeout_s: ZMQ send/recv timeout in seconds.
            serializer: Wire format -- 'msgpack' or 'protobuf'.
            compress: JPEG-compress RGB images before sending.
            enable_profiling: Print average latency every 50 calls.
            **kwargs: Forwarded to BaseInferenceRunner (via MRO).
        """
        # Pass ckpt_path=None so BaseInferenceRunner skips model loading
        kwargs.setdefault('ckpt_path', None)
        super().__init__(**kwargs)

        assert serializer in ('msgpack', 'protobuf'), \
            f"serializer must be 'msgpack' or 'protobuf', got '{serializer}'"
        self._serializer = serializer
        self._compress = compress
        self._server_host = server_host
        self._server_port = server_port
        self._server_address = f'tcp://{server_host}:{server_port}'
        self._timeout_ms = int(timeout_s * 1000)
        self._enable_profiling = enable_profiling

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(self._server_address)
        self._zmq_lock = threading.Lock()

        self._call_count = 0
        self._t_serialize = 0.0
        self._t_zmq = 0.0
        self._t_deserialize = 0.0
        self._t_total = 0.0
        self._t_server_infer = 0.0
        self._t_network = 0.0
        self._payload_bytes = 0
        self._resp_bytes = 0
        self.last_profile = {}

    def run_setup(self):
        """Verify remote server connectivity."""
        set_seed_everywhere(self.seed)
        if not self.ping():
            raise ConnectionError(
                f'Cannot reach VLA server at {self._server_address}')
        overwatch.info(f'Remote server OK at {self._server_address}. '
                       f'Seed set to {self.seed}')

    def _preprocess(self, instruction: str) -> dict:
        """Return raw observation for remote server preprocessing."""
        obs = self.update_observation_window()
        obs['task_description'] = instruction
        obs['unnorm_key'] = self.task_suite_name
        return obs

    def _predict_action(self, inputs: dict):
        """Serialize obs, send to remote server, return action tensor."""
        t_total_start = time.perf_counter()
        unnorm_key = inputs.pop('unnorm_key', '')

        t0 = time.perf_counter()
        obs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                obs[k] = v.cpu().numpy()
            else:
                obs[k] = v
        request = encode_predict_request(
            obs,
            str(unnorm_key),
            fmt=self._serializer,
            compress=self._compress)
        payload_size = len(request)
        t_serialize = time.perf_counter() - t0

        t1 = time.perf_counter()
        with self._zmq_lock:
            self._socket.send(request)
            raw_response = self._socket.recv()
        fmt_tag = FORMAT_PROTOBUF if self._serializer == 'protobuf' else 0
        response = decode_predict_response(raw_response, fmt=fmt_tag)
        t_zmq = time.perf_counter() - t1

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f"ZMQ server error: {response['error']}")

        t2 = time.perf_counter()
        action_buf = io.BytesIO(response['action_data'])
        arr = np.load(action_buf, allow_pickle=False)
        actions = torch.from_numpy(arr.copy())
        t_deserialize = time.perf_counter() - t2

        t_total = time.perf_counter() - t_total_start
        server_infer = response.get('infer_time', 0.0)
        resp_size = len(raw_response)
        t_network = t_zmq - server_infer

        self.last_profile = {
            'serialize_ms': t_serialize * 1000,
            'zmq_roundtrip_ms': t_zmq * 1000,
            'server_infer_ms': server_infer * 1000,
            'network_ms': t_network * 1000,
            'deserialize_ms': t_deserialize * 1000,
            'total_ms': t_total * 1000,
            'payload_kb': payload_size / 1024,
            'response_kb': resp_size / 1024,
        }

        if self._enable_profiling:
            self._call_count += 1
            self._t_serialize += t_serialize
            self._t_zmq += t_zmq
            self._t_deserialize += t_deserialize
            self._t_total += t_total
            self._t_server_infer += server_infer
            self._t_network += t_network
            self._payload_bytes += payload_size
            self._resp_bytes += resp_size

            if self._call_count % 50 == 0:
                n = self._call_count
                overwatch.info(
                    f'[RemoteInference] calls={n}  '
                    f'avg_total={self._t_total/n*1000:.1f}ms  '
                    f'avg_serialize={self._t_serialize/n*1000:.1f}ms  '
                    f'avg_zmq={self._t_zmq/n*1000:.1f}ms  '
                    f'avg_server={self._t_server_infer/n*1000:.1f}ms  '
                    f'avg_network={self._t_network/n*1000:.1f}ms  '
                    f'avg_deser={self._t_deserialize/n*1000:.1f}ms  '
                    f'avg_payload={self._payload_bytes/n/1024:.0f}KB  '
                    f'avg_resp={self._resp_bytes/n/1024:.0f}KB')

        return actions

    def _postprocess_actions(self, raw_action):
        """Server already denormalized; just convert to numpy and truncate."""
        return raw_action.cpu().numpy()[:self.action_chunk]

    def ping(self) -> bool:
        """Health-check the remote server."""
        try:
            request = msgpack.packb({'endpoint': 'ping'})
            with self._zmq_lock:
                self._socket.send(request)
                raw = self._socket.recv()
            resp = msgpack.unpackb(raw, raw=False)
            return resp.get('status') == 'ok'
        except zmq.error.ZMQError:
            return False

    def cleanup(self):
        """Release ZMQ resources and call parent cleanup."""
        overwatch.info('Cleaning up RemoteInferenceRunner')
        if hasattr(self, '_socket') and not self._socket.closed:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close()
        if hasattr(self, '_context'):
            self._context.term()
        super().cleanup()
