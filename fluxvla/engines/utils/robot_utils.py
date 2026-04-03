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

import numpy as np


def invert_gripper_action(action):
    """
    Invert the gripper action in the last dimension of the action array.
    This is necessary for some environments where the gripper action is
    represented as a single value, and we need to invert it for the
    RLDS dataloader to align the gripper actions such that 0 = close, 1 = open.

    Args:
        action (np.ndarray): The action array, expected to have the gripper
            action in the last dimension.

    Returns:
        np.ndarray: The action array with the gripper action inverted.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def normalize_gripper_action(action, binarize=True):
    """Normalize the gripper action to be in the range [-1, +1] and optionally
    binarize it to -1 or +1.
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action (np.ndarray): The action array, expected to have the gripper
            action in the last dimension.
        binarize (bool): If True, binarize the gripper action to -1 or +1.
            Defaults to True.

    Returns:
        np.ndarray: The normalized (and optionally binarized) action array.
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[...,
           -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action
