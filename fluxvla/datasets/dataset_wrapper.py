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

from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset

from fluxvla.engines import (DATASETS, build_dataset_from_cfg,
                             initialize_overwatch)

overwatch = initialize_overwatch(__name__)


@DATASETS.register_module()
class DistributedRepeatingDataset(IterableDataset):
    """A distributed dataset that repeats indefinitely.
    This dataset is designed to be used in a distributed setting,
    where each worker gets a shard of the dataset. It supports
    shuffling and can be used with PyTorch's DataLoader.

    Now supports three formats:
    1. Single dataset (dict): Single dataset configuration
    2. List of datasets (list of dict): Treats all datasets as one
        concatenated dataset
    3. Grouped datasets (dict of list of dict): Groups datasets
        by keys, each group has separate statistics and is
        treated as a separate dataset

    Args:
        datasets (dict, list of dict, or dict of list of dict):
            Configuration for the dataset(s) to be wrapped.
        statistic_keys (list[str]): Keys for which
            to collect statistics.
        name_mappings (dict, optional): Mappings for
            statistic names. Defaults to None.
        shuffle (bool): Whether to shuffle the dataset
            at each epoch.
        seed (int): Seed for random number generation.
        statistic_name (str): Name for the statistics collection.
        dim (int, optional): Target dimension for padding/copying data.
            If provided, data will be padded/copied to be an integer
            multiple of this dimension. Defaults to None.
    """

    def __init__(self,
                 datasets: Union[Dict, List[Dict], Dict[str, List[Dict]]],
                 statistic_keys: List[str],
                 name_mappings: Dict = None,
                 shuffle: bool = True,
                 seed: int = 42,
                 statistic_name: str = 'private',
                 dim: Optional[int] = None,
                 dataset_statistics: Optional[Dict] = None) -> None:
        self.shuffle = shuffle
        self.seed = seed
        self.statistic_name = statistic_name
        self.dim = dim
        # Determine the dataset format and initialize accordingly
        if isinstance(datasets, dict) and not (isinstance(
                list(datasets.values())[0], list) if datasets else False):
            # Case 1: Single dataset (dict)
            if isinstance(datasets, dict):
                self.dataset = build_dataset_from_cfg(datasets)
            else:
                # Already built dataset object
                self.dataset = datasets

            assert hasattr(self.dataset, '__getitem__') and hasattr(self.dataset, '__len__'), \
                'The wrapped dataset must implement __getitem__ and __len__ methods.'  # noqa: E501

            stats = self.dataset.stats
            self.total_len = len(self.dataset)
            self.is_grouped = False
            self.is_list = False

            if dataset_statistics is None:
                self.dataset_statistics = self.get_dataset_statistics(
                    stats, statistic_keys, name_mappings)
            else:
                self.dataset_statistics = dataset_statistics

        elif isinstance(datasets, list):
            # Case 2: List of datasets (list of dict)
            self.datasets = []
            for ds_cfg in datasets:
                ds = build_dataset_from_cfg(ds_cfg) if isinstance(
                    ds_cfg, dict) else ds_cfg
                self.datasets.append(ds)

            # Validate all datasets
            for i, ds in enumerate(self.datasets):
                assert hasattr(ds, '__getitem__') and hasattr(ds, '__len__'), \
                    f'Dataset {i} must implement __getitem__ and __len__ methods.'  # noqa: E501

            # Calculate cumulative lengths for efficient indexing
            self.dataset_lens = [len(ds) for ds in self.datasets]
            self.cumulative_lens = np.cumsum([0] + self.dataset_lens).tolist()
            self.total_len = self.cumulative_lens[-1]

            # Collect dataset statistics if available
            stats = []
            for ds in self.datasets:
                assert hasattr(ds, 'stats'), \
                    'Each dataset must have a stats attribute for statistics collection.'  # noqa: E501
                if hasattr(ds, 'stats'):
                    stats.extend(ds.stats)

            self.is_grouped = False
            self.is_list = True

            self.dataset_statistics = self.get_dataset_statistics(
                stats, statistic_keys, name_mappings)

        else:
            # Case 3: Grouped datasets (dict of list of dict)
            self.grouped_datasets = {}
            self.grouped_dataset_lens = {}
            self.grouped_cumulative_lens = {}
            self.grouped_dataset_statistics = {}

            for group_name, group_datasets in datasets.items():
                # Build all datasets in this group
                group_ds_list = []
                for ds_cfg in group_datasets:
                    ds = build_dataset_from_cfg(ds_cfg) if isinstance(
                        ds_cfg, dict) else ds_cfg
                    group_ds_list.append(ds)

                # Validate all datasets in this group
                for i, ds in enumerate(group_ds_list):
                    assert hasattr(ds, '__getitem__') and hasattr(ds, '__len__'), \
                        f'Dataset {i} in group {group_name} must implement __getitem__ and __len__ methods.'  # noqa: E501

                # Calculate cumulative lengths for this group
                group_lens = [len(ds) for ds in group_ds_list]
                group_cumulative_lens = np.cumsum([0] + group_lens).tolist()

                # Collect dataset statistics for this group
                group_stats = []
                for ds in group_ds_list:
                    assert hasattr(ds, 'stats'), \
                        f'Each dataset in group {group_name} must have a stats attribute for statistics collection.'  # noqa: E501
                    if hasattr(ds, 'stats'):
                        group_stats.extend(ds.stats)

                # Store group information
                self.grouped_datasets[group_name] = group_ds_list
                self.grouped_dataset_lens[group_name] = group_lens
                self.grouped_cumulative_lens[
                    group_name] = group_cumulative_lens

                # Calculate statistics for this group
                self.grouped_dataset_statistics[
                    group_name] = self.get_dataset_statistics(
                        group_stats, statistic_keys, name_mappings)

            # Calculate total length across all groups
            self.total_len = sum(group_cumulative_lens[-1]
                                 for group_cumulative_lens in
                                 self.grouped_cumulative_lens.values())

            self.is_grouped = True
            self.is_list = False

        # Get the rank and world size from the overwatch
        self.rank = overwatch.rank()
        self.world_size = overwatch.world_size()

    def _get_item_from_global_idx(self, global_idx):
        """Get item from global index, handling single, list,
            and grouped cases.

        Args:
            global_idx (int): The global index of the item to get.

        Returns:
            The item from the dataset.
        """
        if self.is_grouped:
            # Case 3: Grouped datasets
            # Find which group the global index belongs to
            current_idx = 0
            for group_name, group_cumulative_lens in \
                    self.grouped_cumulative_lens.items():
                group_total_len = group_cumulative_lens[-1]
                if global_idx < current_idx + group_total_len:
                    # Index belongs to this group
                    local_idx = global_idx - current_idx

                    # Find which dataset in the group
                    dataset_idx = np.searchsorted(
                        group_cumulative_lens[1:], local_idx, side='right')
                    dataset_local_idx = local_idx - group_cumulative_lens[
                        dataset_idx]

                    # Get the dataset and its statistics
                    dataset = self.grouped_datasets[group_name][dataset_idx]
                    group_statistics = self.grouped_dataset_statistics[
                        group_name]

                    return dataset.__getitem__(dataset_local_idx,
                                               group_statistics)

                current_idx += group_total_len

            raise IndexError(f'Global index {global_idx} is out of range')

        elif self.is_list:
            # Case 2: List of datasets
            # Binary search to find which dataset the index belongs to
            dataset_idx = np.searchsorted(
                self.cumulative_lens[1:], global_idx, side='right')
            local_idx = global_idx - self.cumulative_lens[dataset_idx]
            return self.datasets[dataset_idx].__getitem__(
                local_idx, self.dataset_statistics)
        else:
            # Case 1: Single dataset
            return self.dataset.__getitem__(global_idx,
                                            self.dataset_statistics)

    def get_dataset_statistics(self,
                               stats,
                               static_keys: List[str],
                               name_mappings: Optional[Dict] = None) -> Dict:
        """Collect and combine statistics from multiple datasets.

        Args:
            stats (list[dict]): List of statistics from each dataset.
            static_keys (list[str]): Keys for which to collect statistics.
            name_mappings (dict, optional): Mappings for statistic names.
                Defaults to None.
        Returns:
            dict: Combined statistics for the specified keys.
        """
        dataset_statistics = defaultdict(lambda: defaultdict(list))

        # Collect statistics from each dataset
        for stat in stats:
            for key in static_keys:
                if key not in stat['stats']:
                    raise KeyError(f'Dataset is missing key {key!r}.')

                stat_data = stat['stats'][key]

                # Collect basic statistics
                dataset_statistics[key]['min'].append(stat_data['min'])
                dataset_statistics[key]['max'].append(stat_data['max'])
                dataset_statistics[key]['mean'].append(stat_data['mean'])
                dataset_statistics[key]['std'].append(stat_data['std'])

                # Collect count if available
                if 'count' in stat_data:
                    dataset_statistics[key]['count'].append(stat_data['count'])

                # Collect quantiles if available
                if 'q01' in stat_data:
                    dataset_statistics[key]['q01'].append(stat_data['q01'])
                if 'q99' in stat_data:
                    dataset_statistics[key]['q99'].append(stat_data['q99'])

                # Collect other quantiles if available
                for q in ['q25', 'q50', 'q75']:
                    if q in stat_data:
                        dataset_statistics[key][q].append(stat_data[q])

        # Combine collected statistics
        metadata = {}

        for key in static_keys:
            # Handle name mappings
            if name_mappings and key in name_mappings:
                if isinstance(name_mappings[key], str):
                    mapped_keys = [name_mappings[key]]
                else:
                    assert isinstance(name_mappings[key], list), \
                        f"Expected list for key '{key}' in name_mappings, got {type(name_mappings[key])}"  # noqa: E501
                    mapped_keys = name_mappings[key]
            else:
                mapped_keys = [key]

            # Combine statistics for this key
            key_stats = dataset_statistics[key]
            # Apply dimension padding to individual
            # dataset statistics if dim is specified
            if self.dim is not None:
                for stat_type in ['min', 'max', 'mean', 'std', 'q01',
                                  'q99']:  # noqa: E501
                    if stat_type in key_stats and len(
                            key_stats[stat_type]) > 0:
                        padded_stats = []
                        for stat in key_stats[stat_type]:
                            padded_stat = self._pad_statistics_to_dim(
                                {stat_type: stat})[stat_type]
                            padded_stats.append(padded_stat)
                        key_stats[stat_type] = padded_stats

            # Global min and max
            global_min = np.array(key_stats['min']).min(axis=0).tolist()
            global_max = np.array(key_stats['max']).max(axis=0).tolist()

            # Compute weighted mean and combined std
            if 'count' in key_stats and len(key_stats['count']) > 0:
                # If count information is available,
                # use it for weighted calculations
                means = np.array(key_stats['mean'])
                counts = np.array(key_stats['count']).repeat(means.shape[1], 1)
                stds = np.array(key_stats['std'])

                # Weighted mean
                total_count = counts.sum()
                if total_count > 0:
                    weighted_mean = np.average(
                        means, weights=counts, axis=0).tolist()

                    # Merge standard deviations using the formula:
                    # Var_combined = Σ(n_i * (σ_i^2 + (μ_i -
                    # μ_combined)^2)) / Σn_i
                    variances = stds**2
                    mean_diffs_sq = (means - weighted_mean)**2

                    combined_variance = np.sum(
                        counts *
                        (variances + mean_diffs_sq), axis=0) / total_count
                    combined_std = np.sqrt(combined_variance).tolist()
                else:
                    weighted_mean = np.array(
                        key_stats['mean']).mean(axis=0).tolist()
                    combined_std = np.array(
                        key_stats['std']).mean(axis=0).tolist()
            else:
                # if no count information, use simple averages
                weighted_mean = np.array(
                    key_stats['mean']).mean(axis=0).tolist()

                # mean of stds as an approximation
                stds = np.array(key_stats['std'])
                combined_std = np.sqrt(np.mean(stds**2, axis=0)).tolist()

            # Handle quantiles
            # Ideally, we would want to merge quantiles more accurately,
            # but without raw data, we can only approximate.

            # Approach:
            # 1: Use weighted average if counts are available
            # 2: Use median of quantiles as a fallback
            # 3: If neither is available, set to None

            q01_value = None
            q99_value = None

            if 'q01' in key_stats and len(key_stats['q01']) > 0:
                q01_array = np.array(key_stats['q01'])
                if 'count' in key_stats and len(key_stats['count']) > 0:
                    # Weighted average
                    q01_value = np.average(
                        q01_array, weights=counts, axis=0).tolist()
                else:
                    # Use median as a compromise
                    q01_value = np.median(q01_array, axis=0).tolist()

            if 'q99' in key_stats and len(key_stats['q99']) > 0:
                q99_array = np.array(key_stats['q99'])
                if 'count' in key_stats and len(key_stats['count']) > 0:
                    # Weighted average
                    q99_value = np.average(
                        q99_array, weights=counts, axis=0).tolist()
                else:
                    # Use median as a compromise
                    q99_value = np.median(q99_array, axis=0).tolist()

            # Set combined statistics for all mapped keys
            for mapped_key in mapped_keys:
                metadata[mapped_key] = {
                    'mean': weighted_mean,
                    'std': combined_std,
                    'min': global_min,
                    'max': global_max,
                    'q01': q01_value,
                    'q99': q99_value
                }

                # Add other quantiles if available
                for q in ['q25', 'q50', 'q75']:
                    if q in key_stats and len(key_stats[q]) > 0:
                        q_array = np.array(key_stats[q])
                        if 'count' in key_stats and len(
                                key_stats['count']) > 0:
                            q_value = np.average(
                                q_array, weights=counts, axis=0).tolist()
                        else:
                            q_value = np.median(q_array, axis=0).tolist()
                        metadata[mapped_key][q] = q_value

        return {self.statistic_name: metadata}

    def _pad_statistics_to_dim(self, stats_dict):
        """Pad statistics to be an integer multiple of self.dim.

        Args:
            stats_dict: Dictionary containing statistics
                (mean, std, min, max, q01, q99)

        Returns:
            Dictionary with padded statistics
        """
        if self.dim is None:
            return stats_dict

        padded_stats = {}
        for key, value in stats_dict.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                # Convert to numpy array for easier manipulation
                if isinstance(value, list):
                    value = np.array(value)

                # Get the first dimension (sequence length)
                orig_len = value.shape[0]
                # Calculate target length as integer multiple of dim
                target_len = ((orig_len + self.dim - 1) // self.dim) * self.dim

                if target_len == orig_len:
                    # No padding needed
                    padded_stats[key] = value.tolist() if isinstance(
                        value, np.ndarray) else value
                else:
                    # Pad by copying the original data
                    repeat_times = (target_len + orig_len - 1) // orig_len
                    padded_value = np.tile(value, (repeat_times, ) + (1, ) *
                                           (value.ndim - 1))[:target_len]
                    padded_stats[key] = padded_value.tolist()
            else:
                # Non-array data or scalar, keep as is
                padded_stats[key] = value

        return padded_stats

    # If we had access to more raw data,
    # we could implement a more precise merging of quantiles.
    def combine_quantiles_precisely(self,
                                    values_list,
                                    counts_list,
                                    quantiles=[0.01, 0.25, 0.5, 0.75, 0.99]):
        # This requires more raw data information;
        # here is just a sample framework
        pass

    def __iter__(self):
        epoch = 0
        while True:
            # Create indices for the entire virtual concatenated dataset
            indices = np.arange(self.total_len)
            if self.shuffle:
                rng = np.random.default_rng(self.seed + epoch)
                rng.shuffle(indices)

            # Distribute the indices across the world size
            shard = indices[self.rank::self.world_size].tolist()

            for idx in shard:
                yield self._get_item_from_global_idx(idx)

            epoch += 1

    def __len__(self):
        """Return the total length of all datasets."""
        return self.total_len
