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

"""HDF5 dataset for X-VLA LIBERO training.

Reads directly from X-VLA's HDF5 format (abs_action_6d key) so no parquet
conversion is needed.  Mirrors X-VLA's LiberoHandler + BaseHDF5Handler logic
exactly, but as a map-style Dataset compatible with DistributedRepeatingDataset.

HDF5 layout expected (per file):
    abs_action_6d          [T, 10]  xyz(3)+rot6d(6)+gripper_raw(1)
    <observation_key[0]>   [T,]     JPEG-encoded bytes  (e.g. observation/third_image)
    <observation_key[1]>   [T,]     JPEG-encoded bytes  (e.g. observation/wrist_image)
    <language_instruction_key>  ()  bytes scalar

Meta JSON layout (X-VLA style):
    {
      "dataset_name": "libero_spatial",
      "datalist": ["path/to/demo_0.hdf5", ...],
      "observation_key": ["observation/third_image", "observation/wrist_image"],
      "language_instruction_key": "language_instruction"
    }
"""

import io
import json
from typing import Dict, List

import h5py
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode

from fluxvla.engines import DATASETS, build_transform_from_cfg


@DATASETS.register_module()
class LiberoHDF5Dataset(Dataset):
    """Map-style dataset over X-VLA LIBERO HDF5 files.

    Args:
        meta_path (str): Path to X-VLA meta JSON file.
        transforms (List[Dict]): Transform configs applied after image loading.
            Typically just ProcessXVLAPrompts; image aug is done internally.
        num_actions (int): Action chunk size. Defaults to 10.
        num_views (int): Number of camera views to load. Defaults to 3.
        embodiment_id (int): Embodiment ID passed to the model. Defaults to 3.
        training (bool): Enable color-jitter augmentation. Defaults to True.
        statistic_name (str): Key used by DistributedRepeatingDataset.
            Defaults to 'private'.
        image_size (int): Resize target for images. Defaults to 224.
    """

    def __init__(
        self,
        meta_path: str,
        transforms: List[Dict],
        num_actions: int = 10,
        num_views: int = 3,
        embodiment_id: int = 3,
        training: bool = True,
        statistic_name: str = 'private',
        image_size: int = 224,
    ) -> None:
        super().__init__()

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.hdf5_paths: List[str] = meta['datalist']
        self.obs_keys: List[str] = meta.get(
            'observation_key',
            ['observation/third_image', 'observation/wrist_image'],
        )
        self.lang_key: str = meta.get('language_instruction_key',
                                      'language_instruction')
        self.num_actions = num_actions
        self.num_views = num_views
        self.embodiment_id = embodiment_id
        self.training = training
        self.statistic_name = statistic_name

        # Required by DistributedRepeatingDataset; empty = no normalization.
        self.stats: List = []

        # ------------------------------------------------------------------
        # Pre-load non-image data and build flat sample index.
        # X-VLA's LiberoHandler samples idx in range(0, max(0, T - 10)),
        # queries a 1s future window from idx on a 30Hz time axis via
        # interpolation, then uses the first step as proprio and the
        # remaining 10 steps as actions.
        # ------------------------------------------------------------------
        self._traj_cache: List[Dict] = []  # per-file: {traj, lang, T, time_axis}
        self._index: List[tuple] = []      # (file_idx, idx)

        for file_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, 'r') as f:
                abs_a = f['abs_action_6d'][()]          # (T, 10)
                lang_ds = f[self.lang_key]
                raw = lang_ds[()]
                lang = raw.decode() if isinstance(raw, (bytes, bytearray)) \
                    else raw[0].decode()

            T = abs_a.shape[0]
            # Binarize gripper to {0,1} — matches LiberoHandler: (a[:,9:] > 0.0)
            gripper_bin = (abs_a[:, 9:] > 0.0).astype(np.float32)
            arm1 = np.concatenate([abs_a[:, :9], gripper_bin], axis=-1)  # (T,10)
            traj_20d = np.zeros((T, 20), dtype=np.float32)
            traj_20d[:, :10] = arm1  # arm2 stays zeros (single-arm LIBERO)

            time_axis = np.arange(T, dtype=np.float32) / 30.0
            self._traj_cache.append({
                'traj': traj_20d,
                'lang': lang,
                'T': T,
                'time_axis': time_axis,
            })

            for idx in range(0, max(0, T - self.num_actions)):
                # Match BaseHDF5Handler.iter_episode(): skip near-static
                # samples before shuffling / sharding the training stream.
                if np.abs(traj_20d[idx + 1] - traj_20d[idx]).max() < 1e-5:
                    continue
                self._index.append((file_idx, idx))

        # ------------------------------------------------------------------
        # Image augmentation — identical to X-VLA datasets/dataset.py
        # ------------------------------------------------------------------
        aug = [
            tv_transforms.Resize((image_size, image_size),
                                  interpolation=InterpolationMode.BICUBIC),
        ]
        if training:
            aug.append(
                tv_transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                          saturation=0.2, hue=0.0))
        aug += [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
        ]
        self.image_aug = tv_transforms.Compose(aug)

        # Post-image transforms (e.g. ProcessXVLAPrompts)
        self.transforms = [build_transform_from_cfg(t) for t in transforms]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int, dataset_statistics=None) -> Dict:
        file_idx, idx = self._index[index]
        cache = self._traj_cache[file_idx]
        traj_20d: np.ndarray = cache['traj']   # (T, 20)
        lang: str = cache['lang']
        T: int = cache['T']
        time_axis: np.ndarray = cache['time_axis']

        cur = float(time_axis[idx])
        q = np.linspace(
            cur,
            min(cur + 1.0, float(time_axis.max())),
            self.num_actions + 1,
            dtype=np.float32,
        )
        interp = interp1d(
            time_axis,
            traj_20d,
            axis=0,
            bounds_error=False,
            fill_value=(traj_20d[0], traj_20d[-1]),
        )
        traj_seq = interp(q).astype(np.float32)
        states = traj_seq[0].copy()
        actions = traj_seq[1:].copy()
        action_masks = np.ones((self.num_actions,), dtype=np.float32)

        # Load images from HDF5.
        # X-VLA drops the first image frame globally, so images[v][idx] maps to
        # raw HDF5 frame idx + 1.
        images = []
        img_masks = []
        path = self.hdf5_paths[file_idx]
        with h5py.File(path, 'r') as f:
            for obs_key in self.obs_keys[:self.num_views]:
                raw_bytes = f[obs_key][idx + 1]
                pil_img = self._decode_jpeg(raw_bytes)
                img_chw = self.image_aug(pil_img).numpy()  # (3, H, W) float32
                images.append(img_chw)
                img_masks.append(True)

        # Pad to num_views with zero images if fewer views available
        while len(images) < self.num_views:
            images.append(np.zeros_like(images[0]))
            img_masks.append(False)

        images_arr = np.stack(images, axis=0)          # (V, 3, H, W)

        data = {
            'states': states,
            'images': images_arr,
            'img_masks': np.array(img_masks, dtype=bool),
            'actions': actions,
            'action_masks': action_masks,
            'task_description': lang,
            'embodiment_ids': np.array(self.embodiment_id),
            # Passthrough keys expected by collator meta_keys
            'stats': {},
            'info': {},
            'timestamp': float(idx),
            'prompt': '',
        }

        for transform in self.transforms:
            data = transform(data)

        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_jpeg(raw) -> Image.Image:
        """Decode JPEG bytes (or numpy bytes scalar) to PIL RGB image."""
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        elif not isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw)
        return Image.open(io.BytesIO(raw)).convert('RGB')
