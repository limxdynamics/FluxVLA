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
"""Remote UR3 inference runner.

Combines ZMQ remote inference (RemoteInferenceRunner) with UR-specific
ROS observation collection and action execution (URInferenceRunner) via
multiple inheritance.  No ROS code is duplicated.

MRO: RemoteURInferenceRunner -> RemoteInferenceRunner -> URInferenceRunner
     -> BaseInferenceRunner

Method resolution:
    run_setup / _preprocess / _predict_action / _postprocess_actions
        -> RemoteInferenceRunner (ZMQ remote)
    get_ros_observation / update_observation_window / _execute_actions
        -> URInferenceRunner (local ROS)
"""
from ..utils.root import RUNNERS
from .remote_inference_runner import RemoteInferenceRunner
from .ur_inference_runner import URInferenceRunner


@RUNNERS.register_module()
class RemoteURInferenceRunner(RemoteInferenceRunner, URInferenceRunner):
    """Remote UR3 inference: UR ROS observation/execution + ZMQ remote model.

    All UR-specific ROS methods (get_ros_observation,
    update_observation_window, _execute_actions, etc.) are inherited from
    URInferenceRunner.  All remote inference methods are inherited from
    RemoteInferenceRunner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
