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

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from fluxvla.engines import HEADS


@HEADS.register_module()
class OpenVLAHead(nn.Module):
    """
    Head module for OpenVLA, responsible for decoding generated token IDs
    into continuous unnormalized action vectors.

    Args:
        norm_stats (Dict): Dictionary containing normalization statistics
            for each dataset, used to unnormalize predicted actions.
        vocab_size (int): Size of the vocabulary for action tokens.
        *args, **kwargs: Additional arguments passed to nn.Module.
    """

    def __init__(self, norm_stats: Dict[str, Dict[str,
                                                  Dict[str,
                                                       Dict[str,
                                                            List[float]]]]],
                 vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.normstats = norm_stats
        self.vocab_size = vocab_size

    @torch.inference_mode()
    def predict_action(self,
                       generated_ids: torch.Tensor,
                       unnorm_key: Optional[str] = None) -> np.ndarray:
        """
        Predicts and returns the continuous unnormalized action vector from
        the generated token IDs.

        Args:
            generated_ids (Tensor): Generated token IDs of shape [1, seq_len].
            unnorm_key (Optional[str]): Dataset key to retrieve unnormalize
                statistics. If None, assumes only one dataset is used.

        Returns:
            np.ndarray: Unnormalized continuous action vector.
        """
        # Get predicted token IDs corresponding to action
        predicted_action_token_ids = generated_ids[
            0, -self.get_action_dim(unnorm_key):]

        # Decode tokens into normalized actions
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Retrieve unnormalization statistics
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get(
            'mask', np.ones_like(action_norm_stats['q01'], dtype=bool))
        action_high = np.array(action_norm_stats['q99'])
        action_low = np.array(action_norm_stats['q01'])

        # Unnormalize the actions
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) *  # noqa: E222
            (action_high - action_low) +  # noqa: E222
            action_low,  # noqa: E501
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: Optional[str]) -> str:
        """
        Validates the unnorm_key or infers it if only one dataset is present.

        Args:
            norm_stats (Dict): Dictionary of normalization stats.
            unnorm_key (Optional[str]): Dataset name to validate.

        Returns:
            str: Validated dataset name.

        Raises:
            AssertionError: If key is missing or ambiguous.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                'Model trained on multiple datasets. Please provide an '
                '`unnorm_key` from: ' + str(norm_stats.keys()))  # noqa: W604
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f'The `unnorm_key` is invalid. Choose from: {norm_stats.keys()}'  # noqa: W604,E501
        )  # noqa: E501

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """
        Returns the dimensionality of the action space.

        Args:
            unnorm_key (Optional[str]): Dataset key to query.

        Returns:
            int: Action dimension.
        """
        unnorm_key = self._check_unnorm_key(self.normstats, unnorm_key)
        return len(self.normstats[unnorm_key]['action']['q01'])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """
        Retrieves unnormalization statistics for a given dataset.

        Args:
            unnorm_key (Optional[str]): Dataset key to query.

        Returns:
            Dict: Dictionary containing q01, q99, and optional mask.
        """
        unnorm_key = self._check_unnorm_key(self.normstats, unnorm_key)
        return self.normstats[unnorm_key]['action']
