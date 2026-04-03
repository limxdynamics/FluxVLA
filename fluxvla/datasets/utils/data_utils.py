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
data_utils.py

Additional RLDS-specific data utilities.
"""

import hashlib
import inspect
import json
import os
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from fluxvla.engines.utils.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def tree_map(fn: Callable, tree: Dict) -> Dict:
    """Applies a function to each leaf in a nested dictionary structure.

    Args:
        fn (Callable): Function to apply to each leaf.
        tree (Dict): Nested dictionary structure.

    Returns:
        Dict: New dictionary with the same structure as `tree`, but with
            `fn` applied to each leaf.
    """
    return {
        k: tree_map(fn, v) if isinstance(v, dict) else fn(v)
        for k, v in tree.items()
    }


def tree_merge(*trees: Dict) -> Dict:
    """Merges multiple dictionaries into a single dictionary.

    Args:
        *trees (Dict): Dictionaries to merge.

    Returns:
        Dict: Merged dictionary with the same structure as
            the input dictionaries.
    """
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    """Generates a padding tensor of the same shape
    and type as the input tensor.

    Args:
        tensor (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Padding tensor of the same shape
            and type as the input tensor.
    """
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), '')
    else:
        raise ValueError(
            f'Cannot generate padding for tensor of type {tensor.dtype}.')


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    # Normalize to Mean = 0, Stdev = 1
    NORMAL = 'normal'
    # Normalize to Interval = [-1, 1]
    BOUNDS = 'bounds'
    # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    BOUNDS_Q99 = 'bounds_q99'
    # fmt: on


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ('primary', ),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.
    NORMAL,
) -> Dict[str, Any]:
    """
    Constructs a dictionary of keyword arguments for loading a dataset from
    Open-X Embodiment (OXE).

    Args:
        dataset_name (str): The name of the dataset to load.
        data_root_dir (Path): Path to the root directory containing
            the dataset.
        load_camera_views (Tuple[str]): Tuple of camera view names to load.
        load_depth (bool): Whether to include depth observations.
        load_proprio (bool): Whether to include proprioceptive (state) inputs.
        load_language (bool): Whether to include language instructions.
        action_proprio_normalization_type (NormalizationType): Specifies the
            normalization type for action and proprio inputs.

    Returns:
        Dict[str, Any]: A dictionary of dataset loading arguments compatible
            with downstream pipelines.
    """
    from .configs import (OXE_DATASET_CONFIGS, OXE_STANDARDIZATION_TRANSFORMS,
                          ActionEncoding)

    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    if dataset_kwargs['action_encoding'] not in [
            ActionEncoding.EEF_POS, ActionEncoding.EEF_R6
    ]:
        raise ValueError(
            f'Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 actions '
            'supported!')

    # For EEF_POS & EEF_R6, only the gripper dimension is absolute.
    # Normalize all other action dimensions.
    if dataset_kwargs['action_encoding'] is ActionEncoding.EEF_POS:
        dataset_kwargs['absolute_action_mask'] = [False] * 6 + [True]
        dataset_kwargs['action_normalization_mask'] = [True] * 6 + [False]
    elif dataset_kwargs['action_encoding'] is ActionEncoding.EEF_R6:
        dataset_kwargs['absolute_action_mask'] = [False] * 9 + [True]
        dataset_kwargs['action_normalization_mask'] = [True] * 9 + [False]

    dataset_kwargs[
        'action_proprio_normalization_type'] = action_proprio_normalization_type  # noqa: E501

    # Validate camera views
    if len(missing_keys := (set(load_camera_views) -
                            set(dataset_kwargs['image_obs_keys']))) > 0:
        raise ValueError(f'Cannot load `{dataset_name}`; missing camera views \
                `{missing_keys}`')

    # Filter image and depth observation keys by view
    dataset_kwargs['image_obs_keys'] = {
        k: v
        for k, v in dataset_kwargs['image_obs_keys'].items()
        if k in load_camera_views
    }
    dataset_kwargs['depth_obs_keys'] = {
        k: v
        for k, v in dataset_kwargs['depth_obs_keys'].items()
        if k in load_camera_views
    }

    # Remove unused keys
    dataset_kwargs.pop('state_encoding')
    dataset_kwargs.pop('action_encoding')
    if not load_depth:
        dataset_kwargs.pop('depth_obs_keys')
    if not load_proprio:
        dataset_kwargs.pop('state_obs_keys')

    # Add language instruction key if enabled
    if load_language:
        dataset_kwargs['language_key'] = 'language_instruction'

    # Assign standardization function
    dataset_kwargs['standardize_fn'] = OXE_STANDARDIZATION_TRANSFORMS[
        dataset_name]

    # Merge auxiliary kwargs if any
    if 'aux_kwargs' in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop('aux_kwargs'))

    return {
        'name': dataset_name,
        'data_dir': str(data_root_dir),
        **dataset_kwargs
    }


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    load_camera_views: Tuple[str] = ('primary', ),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.
    NORMAL,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Constructs a list of dataset kwargs and corresponding sampling weights
    for a mixture of Open-X Embodiment datasets.

    Args:
        data_root_dir (Path): Path to the root directory of all datasets.
        mixture_spec (List[Tuple[str, float]]): A list of (dataset_name,
            weight) pairs specifying the datasets and their sampling weights.
        load_camera_views (Tuple[str]): Camera views to load for image
            observations.
        load_depth (bool): Whether to include depth observations.
        load_proprio (bool): Whether to include proprioceptive inputs.
        load_language (bool): Whether to include language instructions.
        action_proprio_normalization_type (NormalizationType): Specifies the
            normalization strategy for action and proprioception.

    Returns:
        Tuple[List[Dict[str, Any]], List[float]]: A list of dataset kwargs
            dictionaries and their corresponding sampling weights. Duplicates
            and invalid entries are skipped with warnings.
    """
    included_datasets, filtered_mixture_spec = set(), []

    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            overwatch.warning(
                f'Skipping Duplicate Dataset: {(d_name, d_weight)}')
            continue
        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble dataset kwargs and weights
    per_dataset_kwargs, sampling_weights = [], []
    for d_name, d_weight in filtered_mixture_spec:
        try:
            per_dataset_kwargs.append(
                make_oxe_dataset_kwargs(
                    d_name,
                    data_root_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    action_proprio_normalization_type,
                ))
            sampling_weights.append(d_weight)
        except ValueError as e:
            overwatch.warning(f'Skipping {d_name} due to Error: {e}')

    return per_dataset_kwargs, sampling_weights


# === State / Action Processing Primitives ===


# ruff: noqa: B023
def normalize_action_and_proprio(traj: Dict, metadata: Dict,
                                 normalization_type: NormalizationType):
    """
    Normalizes the 'action' and 'proprio' fields in a trajectory based on
    the specified normalization type and metadata.

    Args:
        traj (Dict): A trajectory dictionary containing raw action and
            proprioceptive values.
        metadata (Dict): A dictionary containing normalization statistics
            for both 'action' and 'proprio'. Each field must include 'mean'
            and 'std' for 'normal', or 'min'/'max' or 'q01'/'q99' for bounds-
            based normalization. An optional 'mask' field can be used to
            selectively normalize dimensions.
        normalization_type (NormalizationType): Specifies the normalization
            method. Supported values are:
            - 'normal': standard score normalization
            - 'bounds': scales to [-1, 1] using min/max bounds
            - 'bounds_q99': scales to [-1, 1] using 1st/99th percentiles

    Returns:
        Dict: The normalized trajectory with updated action and proprio fields.

    Raises:
        ValueError: If an unknown normalization type is provided.
    """
    keys_to_normalize = {'action': 'action', 'proprio': 'observation/proprio'}

    if normalization_type == 'normal':
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get(
                'mask', tf.ones_like(metadata[key]['mean'], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    (x - metadata[key]['mean']) /
                    (metadata[key]['std'] + 1e-8),
                    x,
                ),
            )
        return traj

    elif normalization_type in ['bounds', 'bounds_q99']:
        for key, traj_key in keys_to_normalize.items():
            if normalization_type == 'bounds':
                low = metadata[key]['min']
                high = metadata[key]['max']
            elif normalization_type == 'bounds_q99':
                low = metadata[key]['q01']
                high = metadata[key]['q99']

            mask = metadata[key].get(
                'mask', tf.ones_like(metadata[key]['min'], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) /
                                     (high - low + 1e-8) - 1, -1, 1),
                    x,
                ),
            )

            # Map unused dimensions (min == max) to zero
            zeros_mask = metadata[key]['min'] == metadata[key]['max']
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(zeros_mask, 0.0, x),
            )

        return traj

    raise ValueError(f'Unknown Normalization Type {normalization_type}')


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open
    (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate
    values. We relabel those intermediate values based on the state that is
    reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value,
    we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry

    Args:
        actions (tf.Tensor): Tensor of gripper actions.

    Returns:
        tf.Tensor: Binarized gripper actions.
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32),
                       lambda: is_open_float[i])

    return tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """inverts gripper actions (0 = open; 1 = closed) to (1 - actions).

    Args:
        actions (tf.Tensor): Tensor of gripper actions.

    Returns:
        tf.Tensor: Inverted gripper actions.
    """
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions to absolute gripper states.

    Relative actions:
        +1 = open gripper
        -1 = close gripper
         0 = no change

    Output:
        1.0 = gripper is open
        0.0 = gripper is closed

    Assumes the first non-zero action is meaningful (not redundant).
    If all actions are zero, gripper is assumed open throughout.

    Args:
        actions (tf.Tensor): 1D float tensor of relative gripper actions.

    Returns:
        tf.Tensor: 1D float tensor of absolute gripper states (0.0 or 1.0).
    """
    # Threshold actions to {-1, 0, 1}
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1,
                                   tf.where(closing_mask, -1, 0))

    # Define scan update function
    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry,
                       lambda: thresholded_actions[i])

    # Initialize first state based on first non-zero relative action
    start = -1 * thresholded_actions[tf.argmax(
        thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Accumulate absolute states
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)

    # Convert {-1, 1} to {0.0, 1.0}
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Relabels the action field in a Bridge trajectory using state differences.

    Computes movement actions as the difference between consecutive
    proprioceptive states (first 6 dims of 'state'). Keeps the gripper action
    (last dim) from the original action field. Discards the final timestep,
    since there is no action following it.

    Args:
        traj (Dict[str, Any]): A trajectory dictionary containing fields
            'observation' (with 'state') and 'action'.

    Returns:
        Dict[str, Any]: A truncated trajectory with updated 'action' field
            based on state deltas and original gripper signal.
    """
    movement_actions = traj['observation']['state'][1:, :6] - \
        traj['observation']['state'][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated['action'] = tf.concat(
        [movement_actions, traj['action'][:-1, -1:]], axis=1)
    return traj_truncated


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_kwargs_list: List[Dict[str, Any]],
                        dataset_weights: List[int]) -> None:
    """
    Pretty-prints the list of datasets being loaded along with their sampling
    weights.

    Args:
        dataset_kwargs_list (List[Dict[str, Any]]): A list of dataset
            configuration dictionaries, each containing a 'name' key.
        dataset_weights (List[int]): A list of sampling weights corresponding
            to each dataset.
    """
    print(
        '\n###################################################################'
    )
    print(f'# Loading the following {len(dataset_kwargs_list)} datasets '
          f"(incl. sampling weight):{'': >24} #")
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs['name'])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        '###################################################################\n'
    )


def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Computes or loads cached statistics for a dataset.

    The function checks whether statistics have been computed before using a
    hash of the given dependencies. If available, the statistics are loaded
    from cache. Otherwise, they are computed from scratch.

    Statistics include mean, std, min, max, 1st and 99th percentile for
    'action' and 'proprio' fields, as well as the total number of transitions
    and trajectories.

    Args:
        dataset (dl.DLataset): The dataset to compute statistics for.
        hash_dependencies (Tuple[str, ...]): A list of string identifiers
            used to compute a unique hash key for caching.
        save_dir (Optional[str]): Optional directory to save the statistics.
            If not provided, a local fallback path is used.

    Returns:
        Dict: A dictionary containing computed or cached dataset statistics.

    Raises:
        ValueError: If the dataset has infinite cardinality and statistics
            cannot be computed.
    """
    unique_hash = hashlib.sha256(
        ''.join(hash_dependencies).encode('utf-8'),
        usedforsecurity=False).hexdigest()

    local_path = os.path.expanduser(
        os.path.join('~', '.cache', 'orca',
                     f'dataset_statistics_{unique_hash}.json'))
    if save_dir is not None:
        path = tf.io.gfile.join(save_dir,
                                f'dataset_statistics_{unique_hash}.json')
    else:
        path = local_path

    if tf.io.gfile.exists(path):
        overwatch.info(f'Loading existing dataset statistics from {path}.')
        with tf.io.gfile.GFile(path, 'r') as f:
            metadata = json.load(f)
        return metadata

    if os.path.exists(local_path):
        overwatch.info(
            f'Loading existing dataset statistics from {local_path}.')
        with open(local_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    dataset = dataset.traj_map(
        lambda traj: {
            'action':
            traj['action'],
            'proprio': (traj['observation']['proprio'] if 'proprio' in traj[
                'observation'] else tf.zeros_like(traj['action'])),
        })

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError(
            'Cannot compute dataset statistics for infinite datasets.')

    overwatch.info(
        'Computing dataset statistics. This may take a bit, but should only '
        'need to happen once.')
    actions, proprios = [], []
    num_transitions, num_trajectories = 0, 0

    for traj in tqdm(
            dataset.iterator(),
            total=cardinality
            if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
        actions.append(traj['action'])
        proprios.append(traj['proprio'])
        num_transitions += traj['action'].shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)
    metadata = {
        'action': {
            'mean': actions.mean(0).tolist(),
            'std': actions.std(0).tolist(),
            'max': actions.max(0).tolist(),
            'min': actions.min(0).tolist(),
            'q01': np.quantile(actions, 0.01, axis=0).tolist(),
            'q99': np.quantile(actions, 0.99, axis=0).tolist(),
        },
        'proprio': {
            'mean': proprios.mean(0).tolist(),
            'std': proprios.std(0).tolist(),
            'max': proprios.max(0).tolist(),
            'min': proprios.min(0).tolist(),
            'q01': np.quantile(proprios, 0.01, axis=0).tolist(),
            'q99': np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        'num_transitions': num_transitions,
        'num_trajectories': num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, 'w') as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        overwatch.warning(
            f'Could not write dataset statistics to {path}. Writing to '
            f'{local_path} instead.')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'w') as f:
            json.dump(metadata, f)

    return metadata


def save_dataset_statistics(dataset_statistics, run_dir):
    """
    Saves non-grouped dataset statistics to a JSON file named
    'dataset_statistics.json'.

    This function ensures that all NumPy arrays in the statistics are converted
    to Python lists or scalars so that the JSON file is valid.

    Args:
        dataset_statistics (dict): Dictionary containing dataset statistics,
            typically with 'private' key containing the actual statistics.
        run_dir (Path): Directory where the JSON file should be saved.

    Returns:
        None
    """

    def _convert_numpy_arrays(stats):
        """Helper function to convert numpy arrays to lists/scalars."""
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    stats[key] = value.tolist()
                elif isinstance(value, dict):
                    _convert_numpy_arrays(value)
        return stats

    out_path = os.path.join(run_dir, 'dataset_statistics.json')
    with open(out_path, 'w') as f_json:
        converted_stats = _convert_numpy_arrays(dataset_statistics.copy())
        json.dump(converted_stats, f_json, indent=2)
    overwatch.info(f'Saved dataset statistics file at path {out_path}')


def save_grouped_dataset_statistics(grouped_dataset_statistics, run_dir):
    """
    Saves grouped dataset statistics to separate JSON files for each group.

    This function ensures that all NumPy arrays in the statistics are converted
    to Python lists or scalars so that the JSON files are valid.

    Args:
        grouped_dataset_statistics (dict): Dictionary with group names as keys,
            where each value contains the statistics for that group.
        run_dir (Path): Directory where the JSON files should be saved.

    Returns:
        None
    """

    def _convert_numpy_arrays(stats):
        """Helper function to convert numpy arrays to lists/scalars."""
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    stats[key] = value.tolist()
                elif isinstance(value, dict):
                    _convert_numpy_arrays(value)
        return stats

    # Save separate files for each group
    for group_name, group_stats in grouped_dataset_statistics.items():
        out_path = os.path.join(run_dir,
                                f'dataset_statistics_{group_name}.json')
        with open(out_path, 'w') as f_json:
            converted_stats = _convert_numpy_arrays(group_stats.copy())
            json.dump(converted_stats, f_json, indent=2)
        overwatch.info(f'Saved dataset statistics for group {group_name} '
                       f'at path {out_path}')


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """
    Allocates an integer number of threads to datasets based on sampling
    weights.

    Ensures each dataset gets at least one thread. If n is None, returns
    AUTOTUNE for each dataset.

    Args:
        n (Optional[int]): Total number of threads to allocate. If None,
            AUTOTUNE is used.
        weights (np.ndarray): Sampling weights corresponding to each dataset.

    Returns:
        np.ndarray: Integer array of thread counts for each dataset.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), 'Weights must be non-negative'
    assert len(weights) <= n, (
        'Number of threads must be at least as large as number of datasets')
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give 1 to entries that would otherwise get less than 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate remaining threads
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][:int(n)]:
        allocation[i] += 1

    return allocation


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.
    NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """
    Loads an RLDS dataset and restructures it into a standard trajectory
    format.

    Applies optional standardization, normalizes action/proprio fields, and
    supports extraction of image, depth, state, and language observations.

    Args:
        name (str): Name of the RLDS dataset (e.g., "bridge_dataset:1.0.0").
        data_dir (str): Path to the directory containing dataset files.
        train (bool): Whether to load training (True) or validation (False)
            split.
        standardize_fn (Callable, optional): Function to standardize raw traj
            dict.
        shuffle (bool): Whether to shuffle files (not trajectories).
        image_obs_keys (Dict[str, Optional[str]]): Mapping from new image keys
            to existing keys in raw data. If value is None, inserts padding.
        depth_obs_keys (Dict[str, Optional[str]]): Same as `image_obs_keys`,
            but for depth images. Keys are prefixed with "depth_".
        state_obs_keys (List[Optional[str]]): List of 1D state keys to extract
            and concatenate as 'proprio'. None values result in 0-padding.
        language_key (str, optional): Key to extract language instructions
            from.
        action_proprio_normalization_type (NormalizationType): Type of
            normalization: "normal", "bounds", or "bounds_q99".
        dataset_statistics (dict or str, optional): Stats for normalization,
            or path to JSON file containing them. Computed if not provided.
        absolute_action_mask (List[bool], optional): Indicates which action
            dims are absolute (rest are treated as relative).
        action_normalization_mask (List[bool], optional): Indicates which
            action dims should be normalized. Others are skipped.
        num_parallel_reads (int): Number of parallel TFRecord read workers.
        num_parallel_calls (int): Number of parallel calls for preprocessing.

    Returns:
        Tuple[dl.DLataset, dict]: Processed DLataset and normalization
            statistics.

    The resulting dataset contains per-timestep fields:
        - observation:
            - image_{key}: RGB image tensor or padding
            - depth_{key}: Depth image tensor or padding
            - proprio: 1D state vector
            - timestep: Integer timestep
        - task:
            - language_instruction (if `language_key` provided)
        - action: Action vector (optionally normalized)
        - dataset_name: Name of the originating dataset
    """
    # TODO: Figure out why need this.
    tf.config.set_visible_devices([], 'GPU')
    REQUIRED_KEYS = {'observation', 'action'}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(f'Trajectory is missing keys: '
                             f'{REQUIRED_KEYS - set(traj.keys())}. '
                             'Did you write a `standardize_fn`?')

        traj_len = tf.shape(traj['action'])[0]
        old_obs = traj['observation']
        new_obs = {}

        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f'image_{new}'] = tf.repeat('', traj_len)
            else:
                new_obs[f'image_{new}'] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f'depth_{new}'] = tf.repeat('', traj_len)
            else:
                new_obs[f'depth_{new}'] = old_obs[old]

        if state_obs_keys:
            new_obs['proprio'] = tf.concat([
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None else tf.cast(old_obs[key], tf.float32)
                for key in state_obs_keys
            ],
                                           axis=1)

        new_obs['timestep'] = tf.range(traj_len)

        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f'Language key {language_key} has dtype '
                    f'{traj[language_key].dtype}, but must be tf.string.')
            task['language_instruction'] = traj.pop(language_key)

        traj = {
            'observation': new_obs,
            'task': task,
            'action': tf.cast(traj['action'], tf.float32),
            'dataset_name': tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj['action'].shape[-1]:
                raise ValueError(
                    f'Length of absolute_action_mask '
                    f'({len(absolute_action_mask)}) does not match action '
                    f'dimension ({traj["action"].shape[-1]}).')
            traj['absolute_action_mask'] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask,
                                     dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    builder = tfds.builder(name, data_dir=data_dir)

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, 'r') as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder,
            split='all',
            shuffle=False,
            num_parallel_reads=num_parallel_reads).traj_map(
                restructure, num_parallel_calls)

        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn)
                if standardize_fn is not None else '',
            ),
            save_dir=builder.data_dir,
        )

    dataset_statistics = tree_map(np.array, dataset_statistics)

    if action_normalization_mask is not None:
        if len(action_normalization_mask) != \
                dataset_statistics['action']['mean'].shape[-1]:
            raise ValueError(
                f'Length of normalization mask '
                f'({len(action_normalization_mask)}) does not match action '
                f'dimension ({dataset_statistics["action"]["mean"].shape[-1]}).'  # noqa: E501
            )
        dataset_statistics['action']['mask'] = np.array(
            action_normalization_mask)

    if 'val' not in builder.info.splits:
        split = 'train[:95%]' if train else 'train[95%:]'
    else:
        split = 'train' if train else 'val'

    dataset = dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=shuffle,
        num_parallel_reads=num_parallel_reads)
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics


def make_interleaved_dataset(
    dataset_kwargs_list: List[Dict],
    sample_weights: Optional[List[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: Optional[Dict] = None,
    frame_transform_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """
    Builds an interleaved multi-dataset of frame-level samples.

    Loads multiple RLDS datasets using `make_dataset_from_rlds`, transforms
    each with trajectory/frame transforms, then interleaves and optionally
    batches the result.

    Args:
        dataset_kwargs_list (List[Dict]): List of dataset config dicts. Each
            is passed to `make_dataset_from_rlds`.
        sample_weights (List[float], optional): Sampling weights for each
            dataset. Defaults to uniform.
        train (bool): Whether to construct the training or validation split.
        shuffle_buffer_size (int): Number of frames to keep in the shuffle
            buffer.
        traj_transform_kwargs (Dict, optional): Config for applying trajectory
            transforms.
        frame_transform_kwargs (Dict, optional): Config for applying
            frame-level transforms.
        batch_size (int, optional): Batch size for frame batching. If None, no
            batching applied.
        balance_weights (bool): If True, sample weights are scaled by dataset
            size to ensure balanced sampling.
        traj_transform_threads (int, optional): Total threads to use across
            datasets for trajectory transforms. If None, uses AUTOTUNE.
        traj_read_threads (int, optional): Total threads to use across datasets
            for file reading. If None, uses AUTOTUNE.

    Returns:
        dl.DLataset: Interleaved frame-level dataset.
        int: Effective dataset length (in frames) before all datasets complete
            one epoch.
        dict: A dictionary of dataset statistics for each dataset.
    """

    from .data_transforms import (apply_frame_transforms,
                                  apply_per_dataset_frame_transforms,
                                  apply_trajectory_transforms)

    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            'sample_weights must be None or have length equal to dataset count.'  # noqa: E501
        )

    if traj_transform_kwargs is None or frame_transform_kwargs is None:
        raise ValueError(
            'Missing `traj_transform_kwargs` or `frame_transform_kwargs`.')

    dataset_sizes, all_dataset_statistics = [], {}
    for dataset_kwargs in dataset_kwargs_list:
        data_kwargs = deepcopy(dataset_kwargs)
        if 'dataset_frame_transform_kwargs' in data_kwargs:
            data_kwargs.pop('dataset_frame_transform_kwargs')
        _, dataset_statistics = make_dataset_from_rlds(
            **data_kwargs, train=train)
        dataset_sizes.append(dataset_statistics['num_transitions'])
        all_dataset_statistics[dataset_kwargs['name']] = dataset_statistics

    primary_dataset_indices = np.array(
        [idx for idx, weight in enumerate(sample_weights) if weight == 1.0])

    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)

    sample_weights = np.array(sample_weights)
    sample_weights = sample_weights / np.sum(sample_weights)

    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    dataset_len = int((np.array(dataset_sizes) /
                       sample_weights)[primary_dataset_indices].max())

    threads_per_dataset = allocate_threads(traj_transform_threads,
                                           sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    overwatch.info('Threads per Dataset: %s', threads_per_dataset)
    overwatch.info('Reads per Dataset: %s', reads_per_dataset)

    overwatch.info('Constructing datasets...')
    datasets = []
    for dataset_kwargs, threads, reads in zip(dataset_kwargs_list,
                                              threads_per_dataset,
                                              reads_per_dataset):
        frame_kwargs = dataset_kwargs.pop('dataset_frame_transform_kwargs') \
            if 'dataset_frame_transform_kwargs' in dataset_kwargs else {}

        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs['name']],
        )

        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)

        dataset = apply_per_dataset_frame_transforms(dataset, **frame_kwargs)
        datasets.append(dataset)

    dataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)

    if not train:
        dataset = dataset.take(shuffle_buffer_size).cache()

    dataset = dataset.shuffle(shuffle_buffer_size)

    overwatch.info('Applying frame transforms on dataset...')
    dataset = apply_frame_transforms(
        dataset, **frame_transform_kwargs, train=train)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.with_ram_budget(1)
    dataset.sample_weights = sample_weights

    return dataset, dataset_len, all_dataset_statistics
