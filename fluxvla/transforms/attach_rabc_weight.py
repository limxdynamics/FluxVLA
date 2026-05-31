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
"""Transforms for attaching RA-BC sample weights."""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from fluxvla.engines import TRANSFORMS


def _as_scalar(value: Any) -> Any:
    """Return a Python scalar from common tensor and array containers."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.flatten()[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _as_scalar(value[0])
    return value


class ConstantWeighter:
    """Return a fixed weight for every sample."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    def __call__(self, sample: Mapping[str, Any]) -> float:
        return self.weight


class SuccessRateWeighter:
    """Map a success flag in the sample to a BC weight."""

    def __init__(self,
                 success_key: str = 'success',
                 positive_weight: float = 1.0,
                 negative_weight: float = 0.0,
                 fallback_weight: float = 1.0) -> None:
        self.success_key = success_key
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.fallback_weight = float(fallback_weight)

    def __call__(self, sample: Mapping[str, Any]) -> float:
        success = _as_scalar(sample.get(self.success_key))
        if success is None:
            return self.fallback_weight
        return self.positive_weight if bool(success) else self.negative_weight


class ProgressDeltaWeighter:
    """Weight samples from progress deltas already present in the sample."""

    def __init__(self,
                 progress_key: str = 'progress',
                 future_progress_key: str = 'future_progress',
                 kappa: float = 0.01,
                 fallback_weight: float = 1.0) -> None:
        self.progress_key = progress_key
        self.future_progress_key = future_progress_key
        self.kappa = float(kappa)
        self.fallback_weight = float(fallback_weight)

    def __call__(self, sample: Mapping[str, Any]) -> float:
        progress = _as_scalar(sample.get(self.progress_key))
        future_progress = _as_scalar(sample.get(self.future_progress_key))
        if progress is None or future_progress is None:
            return self.fallback_weight
        delta = float(future_progress) - float(progress)
        if not np.isfinite(delta):
            return self.fallback_weight
        if delta > self.kappa:
            return 1.0
        if delta < 0.0:
            return 0.0
        return delta / max(self.kappa, 1e-8)


class SARMProgressWeighter:
    """Use precomputed SARM progress parquet to compute RA-BC weights."""

    def __init__(self,
                 progress_path: str,
                 chunk_size: int,
                 head_mode: str = 'sparse',
                 index_key: str = 'index',
                 fallback_weight: float = 1.0,
                 **kwargs) -> None:
        self.index_key = index_key
        self.fallback_weight = float(fallback_weight)
        kwargs.setdefault('fallback_weight', fallback_weight)
        kwargs.setdefault('device', 'cpu')
        from tools.sarm_rabc import SarmRABCWeights
        self.weighter = SarmRABCWeights(
            progress_path=progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            **kwargs)

    def __call__(self, sample: Mapping[str, Any]) -> float:
        index = _as_scalar(sample.get(self.index_key))
        if index is None:
            index = _as_scalar(sample.get('current_index'))
        if index is None:
            return self.fallback_weight
        return self.weighter.compute_weight(int(index))


def _build_weighter(config: Optional[Dict[str, Any]]):
    if config is None:
        return ConstantWeighter()
    if callable(config):
        return config
    if not isinstance(config, dict):
        raise TypeError(
            f'weighter must be a dict or callable, got {type(config)}')

    cfg = dict(config)
    weighter_type = cfg.pop('type')
    if weighter_type == 'ConstantWeighter':
        return ConstantWeighter(**cfg)
    if weighter_type == 'SuccessRateWeighter':
        return SuccessRateWeighter(**cfg)
    if weighter_type == 'ProgressDeltaWeighter':
        return ProgressDeltaWeighter(**cfg)
    if weighter_type == 'SARMProgressWeighter':
        return SARMProgressWeighter(**cfg)
    raise ValueError(f'Unsupported RA-BC weighter type: {weighter_type!r}')


@TRANSFORMS.register_module()
class AttachRABCWeight:
    """Attach one RA-BC sample weight to each training sample.

    Put this transform before transforms that rebuild the sample dictionary,
    such as ``ProcessParquetInputs``. Those transforms can then carry
    ``sample_weight`` through to the collator.
    """

    def __init__(self,
                 weighter: Optional[Dict[str, Any]] = None,
                 output_key: str = 'sample_weight',
                 default_weight: float = 1.0,
                 drop_index: bool = False) -> None:
        self.weighter = _build_weighter(weighter)
        self.output_key = output_key
        self.default_weight = float(default_weight)
        self.drop_index = drop_index

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        weight = self.weighter(data)
        if weight is None:
            weight = self.default_weight

        data[self.output_key] = np.asarray(weight, dtype=np.float32)
        if self.drop_index:
            data.pop('index', None)
        return data
