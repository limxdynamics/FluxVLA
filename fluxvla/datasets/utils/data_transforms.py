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
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline.
Trajectory transforms operate on a dictionary that represents a
single trajectory, meaning each tensor has the same leading
dimension (the trajectory length).
"""

import logging
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import dlimp as dl
import tensorflow as tf

import fluxvla.datasets.utils.goal_relabeling as goal_relabeling
import fluxvla.datasets.utils.task_augmentation as task_augmentation


def chunk_act_obs(traj: Dict,
                  window_size: int,
                  future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size
    `window_size` containing `window_size - 1` observations from
    the past and the current observation. "action" is given a new
    axis (at index 1) of size `window_size + future_action_window_size`
    containing `window_size - 1` actions from the past,
    the current action, and `future_action_window_size` actions from
    the future. "pad_mask" is added to "observation" and indicates
    whether an observation should be considered padding (i.e. if it
    had come from a timestep before the start of the trajectory).

    Args:
        traj (Dict): The trajectory to chunk.
        window_size (int): The size of the chunks to create.
        future_action_window_size (int): The number of future
            actions to include in the chunked actions.

    Returns:
        Dict: The chunked trajectory.
    """
    traj_len = tf.shape(traj['action'])[0]
    action_dim = traj['action'].shape[-1]
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1),
        [traj_len, window_size]) + tf.broadcast_to(
            tf.range(traj_len)[:, None], [traj_len, window_size])

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if 'timestep' in traj['task']:
        goal_timestep = traj['task']['timestep']
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj['observation'] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), traj['observation'])
    traj['action'] = tf.gather(traj['action'], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj['observation']['pad_mask'] = chunk_indices >= 0

    # if no absolute_action_mask was provided, assume all actions are relative
    if 'absolute_action_mask' not in traj and future_action_window_size > 0:
        logging.warning(
            'future_action_window_size > 0 but no absolute_action_mask'
            'was provided. Assuming all actions are relative for the'
            'purpose of making neutral actions.')
    absolute_action_mask = traj.get(
        'absolute_action_mask', tf.zeros([traj_len, action_dim],
                                         dtype=tf.bool))
    # absolute actions are repeated (already done during chunking)
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj['action'],
        tf.zeros_like(traj['action']),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj['action'] = tf.where(action_past_goal[:, :, None], neutral_actions,
                              traj['action'])

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length.

    Args:
        traj (Dict): The trajectory to subsample.
        subsample_length (int): The length to subsample to.

    returns:
        Dict: The subsampled trajectory.
    """
    traj_len = tf.shape(traj['action'])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should
    be treated as padding. =>> traj["observation"|"task"]["pad_mask_dict"] =
    {k: traj["observation"|"task"][k] is not padding}.

    Args:
        traj (Dict): The trajectory to add the padding mask to.

    Returns:
        Dict: The trajectory with the padding mask added.
    """
    traj_len = tf.shape(traj['action'])[0]

    for key in ['observation', 'task']:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(
                    traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]['pad_mask_dict'] = pad_mask_dict

    return traj


def augment(obs: Dict, seed: tf.Tensor,
            augment_kwargs: Union[Dict, Dict[str, Dict]]) -> Dict:
    """Augments images, skipping padding images."""
    image_names = {key[6:] for key in obs if key.startswith('image_')}

    if 'augment_order' in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, name in enumerate(image_names):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        logging.debug(f'Augmenting image_{name} with kwargs {kwargs}')
        obs[f'image_{name}'] = tf.cond(
            obs['pad_mask_dict'][f'image_{name}'],
            lambda: dl.transforms.augment_image(
                obs[f'image_{name}'],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f'image_{name}'],  # skip padding images
        )

    return obs


def decode_and_resize(
    obs: Dict,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
) -> Dict:
    """Decodes images and depth images, and then optionally resizes them."""
    image_names = {key[6:] for key in obs if key.startswith('image_')}
    depth_names = {key[6:] for key in obs if key.startswith('depth_')}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                f'No resize_size was provided for image_{name}. This will \
                    result in 1x1 padding images, which may cause errors if \
                        you mix padding and non-padding images.')
        image = obs[f'image_{name}']
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3),
                                 dtype=tf.uint8)
            else:
                image = tf.io.decode_image(
                    image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(
                f'Unsupported image dtype: found image_{name} with \
                    dtype {image.dtype}')
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f'image_{name}'] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                f'No depth_resize_size was provided for depth_{name}. \
                    This will result in 1x1  \
                        padding depth images, which may cause errors \
                            if you mix padding and non-padding images.')
        depth = obs[f'depth_{name}']

        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                depth = tf.zeros((*depth_resize_size.get(name, (1, 1)), 1),
                                 dtype=tf.float32)
            else:
                depth = tf.io.decode_image(
                    depth, expand_animations=False, dtype=tf.float32)[..., 0]
        elif depth.dtype != tf.float32:
            raise ValueError(
                f'Unsupported depth dtype: found depth_{name} with \
                    dtype {depth.dtype}')

        if name in depth_resize_size:
            depth = dl.transforms.resize_depth_image(
                depth, size=depth_resize_size[name])

        obs[f'depth_{name}'] = depth

    return obs


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a trajectory level. \
        Such transforms are usually some sort of "relabeling" (e.g., \
            filtering, chunking, adding goals, dropping keys).

    Transforms in this function should have the following properties:
        - They require access to an entire trajectory (i.e., they cannot \
            be applied frame-wise).
        - They are generally not CPU-intensive, mostly involving moving \
            and copying data.
        - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training \
            (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling \
            strategy to use, or None for no goal relabeling. See \
                `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword \
            arguments to pass to the goal relabeling function.
        window_size (int, optional): The length of the snippets that \
            trajectories are chunked into.
        future_action_window_size (int, optional): The number of \
            future actions beyond window_size to include in the chunked \
                actions.
        subsample_length (int, optional): If provided, trajectories \
            longer than this will be subsampled to this length (after goal \
                relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories \
            with no language labels.
        max_action: (float, optional): If provided, trajectories in \
            which *any* action dimension of *any* transition has an \
                absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in \
            which *any* proprio dimension of *any* transition has an \
                absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation \
            strategy to use, or None for no task augmentation. See \
                `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword \
            arguments to pass to the task augmentation function.
        num_parallel_calls (int, optional): number of parallel calls \
            for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if 'language_instruction' not in dataset.element_spec['task']:
            raise ValueError('skip_unlabeled=True but dataset does not have \
                    language labels.')

        dataset = dataset.filter(lambda x: tf.math.reduce_any(x['task'][
            'language_instruction'] != ''))

    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(
            tf.math.abs(x['action']) <= max_action))

    if max_proprio is not None and 'proprio' in dataset.element_spec[
            'observation']:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(
            tf.math.abs(x['observation']['proprio']) <= max_proprio))

    # marks which entries of the observation and task dicts are padding
    dataset = dataset.traj_map(add_pad_mask_dict, num_parallel_calls)

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case
    # it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions, giving them a new axis
    # at index 1 of size `window_size` and
    # `window_size + future_action_window_size`, respectively
    dataset = dataset.traj_map(
        partial(
            chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    return dataset


def apply_per_dataset_frame_transforms(
    dataset: dl.DLataset,
    chunk_filter_fn: Optional[Callable] = None,
):
    """
    Optionally applied *per-dataset* transforms that happen at
        a frame level.

    Args:
        chunk_filter_fn (callable, optional): Filter function for chunks.
    """
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[Dict, Dict[str, Dict]] = {},
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a frame level. These
    transforms are usually more CPU-intensive, (e.g., decoding or
    resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image
            augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments
            to pass to the image augmentation function. See
            `dlimp.transforms.augment_image` for documentation of
            these kwargs. If a dict of dicts is provided, then key
            "k" will be used for "image_{k}" (names determined by
            `image_obs_keys` in `make_dataset_from_rlds`).
            Augmentation will be skipped for missing keys (so
            pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]):
            If provided, images will be resized to this size. If a
            dict of tuples is provided, then key "k" will be used for
            "image_{k}" (names determined by `image_obs_keys` in
            `make_dataset_from_rlds`). Resizing will be
            skipped for missing keys (so pass an empty dict to
            skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int,
            int]]): Same as resize_size, but for depth images.
        num_parallel_calls (int): number of parallel calls for frame_map
            operations. Default to AUTOTUNE.
    """

    # Convenience wrapper that takes a function that operates on a
    # non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked
    # "task" dict
    def apply_obs_transform(fn: Callable[[Dict], Dict], frame: Dict) -> Dict:
        frame['task'] = fn(frame['task'])
        frame['observation'] = dl.vmap(fn)(frame['observation'])
        return frame

    # Decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size),
        ),
        num_parallel_calls,
    )

    if train:
        # Augment all images with the same seed, skipping padding images
        def aug(frame: dict):
            seed = tf.random.uniform([2],
                                     maxval=tf.dtypes.int32.max,
                                     dtype=tf.int32)
            aug_fn = partial(
                augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset
