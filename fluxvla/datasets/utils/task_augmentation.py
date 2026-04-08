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
task_augmentation.py

Contains basic logic for randomly zeroing out keys in the task specification.
"""

from typing import Dict

import tensorflow as tf

from .data_utils import to_padding


def delete_task_conditioning(traj: Dict, keep_image_prob: float) -> Dict:
    """
    Randomly drops either goal images or language instruction from the
        task dict.

    Only performs the operation if both language instruction and at least
        one goal
    image (keys starting with "image_" or "depth_") are present. For each step,
        the function chooses to keep the goal images with probability
        `keep_image_prob`. If images are dropped, language is kept, and vice
        versa.

    Pad masks are updated accordingly, and when all goal images are removed,
    the goal timestep is set to the final timestep.

    Args:
        traj (Dict): A trajectory dictionary containing "task" and "action".
        keep_image_prob (float): Probability of keeping goal images. Language
            is kept with probability (1 - keep_image_prob).

    Returns:
        Dict: Modified trajectory with task conditioning randomly removed.
    """
    if 'language_instruction' not in traj['task']:
        return traj

    # Find keys corresponding to goal images
    image_keys = {
        key
        for key in traj['task']
        if key.startswith('image_') or key.startswith('depth_')
    }
    if not image_keys:
        return traj

    traj_len = tf.shape(traj['action'])[0]

    # Generate random mask for image retention
    should_keep_images = tf.random.uniform([traj_len]) < keep_image_prob

    # Always keep language when image is dropped, and vice versa
    should_keep_images |= ~traj['task']['pad_mask_dict'][
        'language_instruction']  # noqa: E501

    # Zero out or retain each key based on strategy
    for key in image_keys | {'language_instruction'}:
        should_keep = (
            should_keep_images if key in image_keys else ~should_keep_images)

        traj['task'][key] = tf.where(
            should_keep,
            traj['task'][key],
            to_padding(traj['task'][key]),
        )
        traj['task']['pad_mask_dict'][key] = tf.where(
            should_keep,
            traj['task']['pad_mask_dict'][key],
            tf.zeros_like(traj['task']['pad_mask_dict'][key]),
        )

    # If images dropped, set goal timestep to final step
    traj['task']['timestep'] = tf.where(
        should_keep_images,
        traj['task']['timestep'],
        traj_len - 1,
    )

    return traj
