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
"""Remote Aloha dual-arm inference runner.

Combines ZMQ remote inference (RemoteInferenceRunner) with Aloha-specific
ROS observation collection and dual-arm action execution
(AlohaInferenceRunner) via multiple inheritance.

MRO: RemoteAlohaInferenceRunner -> RemoteInferenceRunner
     -> AlohaInferenceRunner -> BaseInferenceRunner

Method resolution:
    run_setup / _preprocess / _predict_action / _postprocess_actions
        -> RemoteInferenceRunner (ZMQ remote)
    get_ros_observation / update_observation_window / _execute_actions
        -> AlohaInferenceRunner (local ROS, dual-arm)
"""
from ..utils.root import RUNNERS
from .aloha_inference_runner import AlohaInferenceRunner
from .remote_inference_runner import RemoteInferenceRunner


@RUNNERS.register_module()
class RemoteAlohaInferenceRunner(RemoteInferenceRunner, AlohaInferenceRunner):
    """Remote Aloha inference: dual-arm ROS + ZMQ remote model.

    All Aloha-specific ROS methods (get_ros_observation,
    update_observation_window, _execute_actions, etc.) are inherited from
    AlohaInferenceRunner.  All remote inference methods are inherited from
    RemoteInferenceRunner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
