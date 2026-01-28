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

import torch

from ..utils.root import RUNNERS
from ..utils.trajectory_utils import resample_remaining
from .tron2_inference_runner import Tron2InferenceRunner


@RUNNERS.register_module()
class Tron2RTCInferenceRunner(Tron2InferenceRunner):
    """Tron2 inference runner with RTC prefix conditioning."""

    def __init__(self, rtc_config: dict = None, *args, **kwargs):
        self.rtc_config = rtc_config
        super().__init__(*args, **kwargs)

    def _predict_action(self, inputs):
        prev = self._prev_ctx
        ctx = self._action_ctx

        ctx.inference_start = time.time()

        if (prev is not None and self.rtc_config
                and self.rtc_config.get('enabled', False)):
            offset = (ctx.inference_start - prev.action_timestamp) / self.dt
            remaining = resample_remaining(prev.raw_actions[0], offset)[None]
            prefix_len = self.rtc_config.get('prefix_len')
            if prefix_len is None:
                prefix_len = int(prev.inference_elapsed * self.publish_rate)
            prefix_len = min(prefix_len, remaining.shape[1])
            if prefix_len > 0:
                inputs['prev_actions'] = torch.from_numpy(remaining).to(
                    device=inputs['states'].device,
                    dtype=inputs['states'].dtype)
                inputs['prefix_len'] = prefix_len
                inputs['rtc_config'] = self.rtc_config

        raw_action = self.vla.predict_action(**inputs)

        ctx.inference_elapsed = time.time() - ctx.inference_start
        ctx.raw_actions = raw_action.cpu().numpy()
        return raw_action
