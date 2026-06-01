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
"""Shared utilities for simulation evaluation runners."""

import os
from pathlib import Path
from typing import Any, Dict

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype

overwatch = initialize_overwatch(__name__)


class BaseEvalRunner:
    """Common helper mixin for environment evaluation runners.

    This base class intentionally keeps the task-specific evaluation loops in
    their concrete runners. It only centralizes small, side-effect-free helpers
    that are shared by LIBERO and RoboCasa evaluation code.
    """

    @staticmethod
    def build_eval_vla(cfg: Dict):
        """Build the VLA model used by an evaluation config."""
        from fluxvla.engines import build_vla_from_cfg

        if hasattr(cfg, 'inference_model'):
            return build_vla_from_cfg(cfg.inference_model).eval()
        return build_vla_from_cfg(cfg.model).eval()

    @staticmethod
    def default_stats_path(ckpt_path: str) -> str:
        """Return the checkpoint-relative dataset statistics path."""
        return os.path.join(
            Path(ckpt_path).resolve().parent.parent, 'dataset_statistics.json')

    def set_common_eval_attrs(self, cfg: Dict, seed: int, ckpt_path: str,
                              model_family: str, mixed_precision_dtype: str,
                              enable_mixed_precision_training: bool) -> None:
        """Set attributes shared by evaluation runners."""
        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.model_family = model_family
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.device_id = overwatch.local_rank()
        self.distributed_state = overwatch.distributed_state

    def update_model_norm_stats(self, norm_stats: Dict[str, Any]) -> None:
        """Attach normalization statistics to the wrapped VLA model."""
        self.vla.norm_stats = norm_stats
