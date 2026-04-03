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
from typing import Dict, List

import numpy as np
import torch

from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.eval_utils import quat2axisangle
from fluxvla.engines.utils.robot_utils import (invert_gripper_action,
                                               normalize_gripper_action)


@TRANSFORMS.register_module()
class Normalize:
    """Normalize the data using provided statistics.
    This transform normalizes the data by subtracting
    the mean and dividing by the standard deviation.
    Supports different normalization types: 'mean_std',
        'quantile', or 'min_max'.

    Args:
        norm_stats (List): List of normalization statistics,
            where each element is a dictionary  containing
            'mean', 'std', 'q01', 'q99', 'min', and 'max' for each feature.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
    """

    def __init__(self,
                 norm_stats: List,
                 norm_type: str = 'mean_std',
                 strict: bool = False):
        self.norm_stats = norm_stats
        self.norm_type = norm_type
        self.strict = strict

    def __call__(self, data: Dict) -> Dict:
        if self.norm_stats is None:
            return data
        for key, value in data.items():
            if key in self.norm_stats.keys():
                if self.norm_type == 'quantile':
                    data[key] = self._normalize_quantile(
                        value, self.norm_stats[key])
                elif self.norm_type == 'min_max':
                    data[key] = self._normalize_min_max(
                        value, self.norm_stats[key])
                else:  # norm_type == 'mean_std'
                    data[key] = self._normalize(value, self.norm_stats[key])
        return data

    def _normalize(self, x, stats: Dict):
        return (x - torch.tensor(stats['mean'])) / (
            torch.tensor(stats['std']) + 1e-6)

    def _normalize_quantile(self, x, stats: torch.tensor):
        assert stats['q01'] is not None
        assert stats['q99'] is not None
        return (x - torch.tensor(stats['q01'])) / (torch.tensor(
            stats['q99']) - torch.tensor(stats['q01']) + 1e-6) * 2.0 - 1.0

    def _normalize_min_max(self, x, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        return (x - torch.tensor(stats['min'])) / (torch.tensor(
            stats['max']) - torch.tensor(stats['min']) + 1e-6) * 2.0 - 1.0


@TRANSFORMS.register_module()
class DenormalizeLiberoAction:
    """Denormalize the data using provided statistics.
    This transform reverses the normalization done using
    mean/std, quantiles, or min_max.

    Args:
        norm_stats (str or Dict): Normalization statistics,
            which can be a JSON string or a dictionary
            containing 'mean', 'std', 'q01', 'q99', 'min', and 'max' for each
            feature. If a string, it should be a JSON representation
            of the normalization statistics.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
        denorm_action (bool): If True, denormalize the action.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
        normalize_gripper_action (bool): If True, normalize
            the gripper action. This is useful for tasks
            where the gripper action is part of the state
            and needs to be denormalized.
        invert_gripper_action (bool): If True, invert the
            gripper action. This is useful for tasks where
            the gripper action is represented in a way that
            requires inversion (e.g., opening vs. closing).
            This is useful for tasks where the gripper action
            is represented in a way that requires inversion
            (e.g., opening vs. closing).
    """

    def __init__(self,
                 norm_stats: str,
                 action_dim: int = None,
                 norm_type: str = 'mean_std',
                 strict: bool = False,
                 denorm_action: bool = True,
                 normalize_gripper_action: bool = True,
                 invert_gripper_action: bool = True,
                 action_norm_mask: List[bool] = None):
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.strict = strict
        self.denorm_action = denorm_action
        self.normalize_gripper_action = normalize_gripper_action
        self.invert_gripper_action = invert_gripper_action
        self.action_norm_mask = action_norm_mask

    def __call__(self, data: Dict) -> Dict:
        """Denormalize the data using the provided statistics.
        This method denormalizes the action in the data
        if the `denorm_action` flag is set to True.
        It retrieves the normalization statistics based on
        the `task_suite_name` from the data and applies
        the appropriate denormalization method.  # noqa: E501

        Args:
            data (Dict): The data to be denormalized, which should
                contain keys that match the keys in `norm_stats`.
        """
        if self.norm_stats is not None and self.denorm_action:
            task_suite_name = data.get('task_suite_name', '')
            norm_stats = self.norm_stats[task_suite_name + '_no_noops']
            action = data.get('action', None)
            assert action is not None, \
                f'Action is not found in the data: {data.keys()}'
            if self.norm_type == 'quantile':
                action = self._denormalize_quantile(action,
                                                    norm_stats['action'])
            elif self.norm_type == 'min_max':
                action = self._denormalize_min_max(action,
                                                   norm_stats['action'])
            else:  # norm_type == 'mean_std'
                action = self._denormalize(action, norm_stats['action'])
        if self.normalize_gripper_action:
            action = normalize_gripper_action(action, binarize=True)
        if self.invert_gripper_action:
            action = invert_gripper_action(action)

        if self.action_dim is not None:
            action = action[:self.action_dim]
        return action

    def _denormalize(self, normalized_action: np.ndarray, stats: Dict):
        assert 'mean' in stats and stats['mean'] is not None
        assert 'std' in stats and stats['std'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]

        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(stats['mean'], dtype=bool)
        action = np.where(
            mask,
            normalized_action * np.array(stats['std']) +
            np.array(stats['mean']), normalized_action)
        return action

    def _denormalize_quantile(self, normalized_action: np.ndarray,
                              stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]
        if self.action_norm_mask is not None:
            mask = np.array(self.action_norm_mask)
        else:
            mask = np.ones_like(stats['q01'], dtype=bool)  # noqa: E501
        action_high = np.array(stats['q99'])
        action_low = np.array(stats['q01'])
        mask = np.array(mask)
        action = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) +
            action_low,  # noqa: E501
            normalized_action,
        )
        return action

    def _denormalize_min_max(self, normalized_action: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]
        if self.action_norm_mask is not None:
            mask = np.array(self.action_norm_mask)
        else:
            mask = np.ones_like(stats['min'], dtype=bool)
        action_high = np.array(stats['max'])
        action_low = np.array(stats['min'])
        mask = np.array(mask)
        action = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) +
            action_low,
            normalized_action,
        )
        return action


@TRANSFORMS.register_module()
class DenormalizePrivateAction(DenormalizeLiberoAction):
    """Denormalize the data using provided statistics.
    This transform reverses the normalization done using
    mean/std, quantiles, or min_max.

    Args:
        norm_stats (str or Dict): Normalization statistics,
            which can be a JSON string or a dictionary
            containing 'mean', 'std', 'q01', 'q99', 'min', and 'max' for each
            feature. If a string, it should be a JSON representation
            of the normalization statistics.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
        denorm_action (bool): If True, denormalize the action.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
        normalize_gripper_action (bool): If True, normalize
            the gripper action. This is useful for tasks
            where the gripper action is part of the state
            and needs to be denormalized.
        invert_gripper_action (bool): If True, invert the
            gripper action. This is useful for tasks where
            the gripper action is represented in a way that
            requires inversion (e.g., opening vs. closing).
            This is useful for tasks where the gripper action
            is represented in a way that requires inversion
            (e.g., opening vs. closing).
    """

    def __init__(self,
                 norm_stats: str,
                 action_dim: int = None,
                 norm_type: str = 'mean_std',
                 strict: bool = False,
                 denorm_action: bool = True,
                 normalize_gripper_action: bool = True,
                 invert_gripper_action: bool = True,
                 action_norm_mask: List[bool] = None):
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.strict = strict
        self.denorm_action = denorm_action
        self.action_norm_mask = action_norm_mask

    def __call__(self, data: Dict) -> Dict:
        """Denormalize the data using the provided statistics.
        This method denormalizes the action in the data
        if the `denorm_action` flag is set to True.
        It retrieves the normalization statistics based on
        the `task_suite_name` from the data and applies
        the appropriate denormalization method.  # noqa: E501

        Args:
            data (Dict): The data to be denormalized, which should
                contain keys that match the keys in `norm_stats`.
        """
        if self.norm_stats is not None and self.denorm_action:
            norm_stats = self.norm_stats['private']
            action = data.get('action', None)[0]
            assert action is not None, \
                f'Action is not found in the data: {data.keys()}'
            if self.norm_type == 'quantile':
                action = self._denormalize_quantile(action,
                                                    norm_stats['action'])
            elif self.norm_type == 'min_max':
                action = self._denormalize_min_max(action,
                                                   norm_stats['action'])
            else:  # norm_type == 'mean_std'
                action = self._denormalize(action, norm_stats['action'])
        return action


@TRANSFORMS.register_module()
class NormalizeStatesAndActions:
    """Normalize states and actions in the data.
    This transform normalizes the state and action
    dimensions in the data to match the specified
    action dimension. It pads the state and action
    dimensions to the specified action dimension.

    Args:
        action_dim (int): The dimension to which the state
            and action should be normalized.
        pad_value (float): The value to use for padding.
            Defaults to 0.0.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        state_key (str): The key in the data dictionary
            that contains the state information.
        action_key (str): The key in the data dictionary
            that contains the action information.
    """

    def __init__(self,
                 state_key: str,
                 action_key: str,
                 action_dim: int = None,
                 state_dim: int = None,
                 norm_type: str = 'mean_std',
                 pad_value: float = 0.0,
                 action_norm_mask: List[bool] = None,
                 *args,
                 **kwargs):
        self.state_key = state_key
        self.action_key = action_key
        self.norm_type = norm_type
        self.pad_value = pad_value
        self.action_dim = action_dim
        self.state_dim = state_dim
        if action_norm_mask is not None:
            assert len(action_norm_mask) == action_dim, \
                f'Action norm mask must be of length {action_dim}'
            self.action_norm_mask = action_norm_mask
        else:
            self.action_norm_mask = None

    def __call__(self, data: Dict) -> Dict:
        assert 'stats' in data, "Input data must contain 'stats' key"
        state_stats = data['stats'][self.state_key]
        action_stats = data['stats'][self.action_key]

        if self.norm_type == 'quantile':
            states = self._normalize_quantile(data['states'], state_stats)
            if 'actions' in data:
                actions = self._normalize_quantile(data['actions'],
                                                   action_stats,
                                                   self.action_norm_mask)
                data['actions'] = actions
            data['states'] = states
        elif self.norm_type == 'min_max':
            states = self._normalize_min_max(data['states'], state_stats)
            if 'actions' in data:
                actions = self._normalize_min_max(data['actions'],
                                                  action_stats,
                                                  self.action_norm_mask)
                data['actions'] = actions
            data['states'] = states
        else:  # norm_type == 'mean_std'
            states = self._normalize(data['states'], state_stats)
            if 'actions' in data:
                actions = self._normalize(data['actions'], action_stats,
                                          self.action_norm_mask)
                data['actions'] = actions
            data['states'] = states
        if self.state_dim is not None:
            data['states'] = np.zeros((self.state_dim))
            data['states'][:states.shape[0]] = states
        if self.action_dim is not None:
            data['actions'] = np.zeros(
                (data['actions'].shape[0], self.action_dim))
            data['actions'][:, :actions.shape[-1]] = actions
        return data

    def _normalize(self, x, stats: Dict, norm_mask: List[bool] = None):
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        return np.where(norm_mask, (x - np.array(stats['mean'])) /
                        (np.array(stats['std']) + 1e-6), x)

    def _normalize_quantile(self,
                            x,
                            stats: torch.tensor,
                            norm_mask: List[bool] = None):
        assert stats['q01'] is not None
        assert stats['q99'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        return np.where(
            norm_mask, (x - np.array(stats['q01'])) /
            (np.array(stats['q99']) - np.array(stats['q01']) + 1e-6) * 2.0 -
            1.0, x)

    def _normalize_min_max(self, x, stats: Dict, norm_mask: List[bool] = None):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        return np.where(
            norm_mask, (x - np.array(stats['min'])) /
            (np.array(stats['max']) - np.array(stats['min']) + 1e-6) * 2.0 -
            1.0, x)


@TRANSFORMS.register_module()
class LiberoProprioFromInputs:
    """Build and normalize Libero proprio state from inputs.

    Reads `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`,
    converts quaternion to axis-angle, concatenates into a
    state vector, and normalizes using `norm_stats[task_suite_name +
    '_no_noops']['proprio']`.

    Expects `task_suite_name` to be present in the input dict.

    Args:
        norm_stats (str | Dict): Path to JSON or dict of normalization stats.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'quantile'.
        pos_key (str): Key for end-effector position.
        quat_key (str): Key for end-effector quaternion.
        gripper_key (str): Key for gripper position.
        out_key (str): Output key for normalized state (default 'states').
    """

    def __init__(self,
                 norm_type: str = 'quantile',
                 state_dim: int = None,
                 pos_key: str = 'robot0_eef_pos',
                 quat_key: str = 'robot0_eef_quat',
                 gripper_key: str = 'robot0_gripper_qpos',
                 stat_key: str = 'proprio',
                 out_key: str = 'states') -> None:
        self.norm_type = norm_type
        self.state_dim = state_dim
        self.pos_key = pos_key
        self.quat_key = quat_key
        self.gripper_key = gripper_key
        self.out_key = out_key
        self.stat_key = stat_key

    def __call__(self, data: Dict) -> Dict:
        assert self.pos_key in data and self.quat_key in \
            data and self.gripper_key in data, \
            f'Missing proprio keys in data: {self.pos_key}, {self.quat_key}, {self.gripper_key}'  # noqa: E501
        robot0_eef_pos = np.asarray(data[self.pos_key])
        robot0_eef_quat = np.asarray(data[self.quat_key])
        robot0_gripper_qpos = np.asarray(data[self.gripper_key])

        state = np.concatenate((
            robot0_eef_pos,
            quat2axisangle(robot0_eef_quat),
            robot0_gripper_qpos,
        ))

        stats = data['norm_stats'][self.stat_key]
        if self.norm_type == 'quantile':
            state = self._normalize_quantile(state, stats)
        elif self.norm_type == 'min_max':
            state = self._normalize_min_max(state, stats)
        else:  # norm_type == 'mean_std'
            state = self._normalize(state, stats)

        out = dict(data)
        if self.state_dim is not None:
            out[self.out_key] = np.zeros((self.state_dim))
            out[self.out_key][:state.shape[0]] = state
        else:
            out[self.out_key] = state
        return out

    def _normalize(self, normalized_states: np.ndarray, stats: Dict):
        assert 'mean' in stats and stats['mean'] is not None
        assert 'std' in stats and stats['std'] is not None
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(stats['mean'], dtype=bool)
        states = np.where(
            mask,
            np.clip((normalized_states - np.array(stats['mean'])) /
                    np.array(stats['std']) + 1e-6, -1, 1), normalized_states)
        return states

    def _normalize_quantile(self, normalized_states: np.ndarray, stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None
        state_high = np.array(stats['q99'])
        state_low = np.array(stats['q01'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states

    def _normalize_min_max(self, normalized_states: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        state_high = np.array(stats['max'])
        state_low = np.array(stats['min'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states
