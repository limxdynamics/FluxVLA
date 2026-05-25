# ============================================================
# Robocasa 评测 Transform — 观测预处理 + 动作反归一化
# ============================================================
#
# 评测时的观测预处理 pipeline:
#   ProcessRobocasaEvalInputs → NormalizeStatesAndActions
#   → PreparePromptWithState → ProcessPrompts
#   → ResizeImages → SimpleNormalizeImages
#
# 动作反归一化:
#   DenormalizeRobocasaAction: min_max 反归一化，无 gripper 后处理
#
# 作者: yiming | 创建: 2026-04-14
# ============================================================

import json
import copy
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from fluxvla.engines.utils.root import TRANSFORMS, DATASETS

# 关节布局必须和训练数据严格一致！
# 参见 scripts/convert_robocasa_for_fluxvla.py 顶部注释:
#   "提取后 29 维排列: left_arm(7) + left_hand(6) + right_arm(7) + right_hand(6) + waist(3)"
# 以及 RobocasaEvalRunner 的 ROBOCASA_ACTION_KEYS 字典
# 历史坑: v1 版本把 right_arm 和 left_hand 位置写反，
#        导致 eval 时 state 从第 7 维开始整段错位 → 12/12 fail。
ROBOCASA_STATE_KEYS = [
    'state.left_arm',    # 7 维 → [0:7]
    'state.left_hand',   # 6 维 → [7:13]
    'state.right_arm',   # 7 维 → [13:20]
    'state.right_hand',  # 6 维 → [20:26]
    'state.waist',       # 3 维 → [26:29]
]

ROBOCASA_GR1_FLUXVLA_ORDER = {
    'left_arm': (0, 7),
    'left_hand': (7, 13),
    'right_arm': (13, 20),
    'right_hand': (20, 26),
    'waist': (26, 29),
}

ROBOCASA_GR1_N15_ORDER = {
    'left_arm': (0, 7),
    'right_arm': (7, 14),
    'left_hand': (14, 20),
    'right_hand': (20, 26),
    'waist': (26, 29),
}

ROBOCASA_GR1_N15_GROUPS = [
    'left_arm',
    'right_arm',
    'left_hand',
    'right_hand',
    'waist',
]


def _robocasa_gr1_permutation(source_order: Dict[str, tuple]) -> List[int]:
    """Return indices that convert a flat GR1 vector to official N1.5 order."""
    indices = []
    for name in ROBOCASA_GR1_N15_GROUPS:
        start, end = source_order[name]
        indices.extend(range(start, end))
    return indices


@TRANSFORMS.register_module()
class RobocasaGR1N15Bridge:
    """Align FluxVLA-converted RoboCasa GR1 vectors to official GR00T N1.5.

    The current FluxVLA parquet/eval bridge stores GR1 vectors as
    left_arm + left_hand + right_arm + right_hand + waist. Official N1.5
    `fourier_gr1_arms_waist` uses left_arm + right_arm + left_hand +
    right_hand + waist, applies sin/cos to state, and min-max normalizes
    action only. This transform performs the order conversion and state
    encoding while also reordering flat action statistics.
    """

    def __init__(self,
                 source_order: str = 'fluxvla',
                 state_key: str = 'states',
                 action_key: str = 'actions',
                 stats_action_key: str = 'action',
                 apply_state_sincos: bool = True,
                 reorder_actions: bool = True,
                 reorder_action_stats: bool = True):
        if source_order != 'fluxvla':
            raise ValueError(
                f'Unsupported source_order={source_order}. '
                'Only the existing FluxVLA RoboCasa order is supported.')
        self.state_key = state_key
        self.action_key = action_key
        self.stats_action_key = stats_action_key
        self.apply_state_sincos = apply_state_sincos
        self.reorder_actions = reorder_actions
        self.reorder_action_stats = reorder_action_stats
        self.permutation = np.array(
            _robocasa_gr1_permutation(ROBOCASA_GR1_FLUXVLA_ORDER),
            dtype=np.int64)

    def _reorder_flat(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if value.shape[-1] != len(self.permutation):
            raise ValueError(
                f'Expected GR1 vector dim {len(self.permutation)}, '
                f'got {value.shape[-1]} for shape {value.shape}')
        return value[..., self.permutation]

    def _state_sincos_n15(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if value.shape[-1] != len(self.permutation):
            raise ValueError(
                f'Expected GR1 state dim {len(self.permutation)}, '
                f'got {value.shape[-1]} for shape {value.shape}')
        encoded = []
        for name in ROBOCASA_GR1_N15_GROUPS:
            start, end = ROBOCASA_GR1_FLUXVLA_ORDER[name]
            group = value[..., start:end]
            encoded.extend([np.sin(group), np.cos(group)])
        return np.concatenate(encoded, axis=-1)

    def _reorder_action_stats(self, data: Dict) -> None:
        if not self.reorder_action_stats or 'stats' not in data:
            return
        stats = copy.deepcopy(data['stats'])
        if self.stats_action_key not in stats:
            data['stats'] = stats
            return
        action_stats = stats[self.stats_action_key]
        for key in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
            values = action_stats.get(key)
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.shape[-1] == len(self.permutation):
                action_stats[key] = arr[..., self.permutation].tolist()
        data['stats'] = stats

    def __call__(self, data: Dict) -> Dict:
        if self.state_key in data:
            if self.apply_state_sincos:
                data[self.state_key] = self._state_sincos_n15(
                    data[self.state_key])
            else:
                data[self.state_key] = self._reorder_flat(data[self.state_key])

        if self.reorder_actions and self.action_key in data:
            data[self.action_key] = self._reorder_flat(data[self.action_key])

        self._reorder_action_stats(data)
        return data


@TRANSFORMS.register_module()
class ProcessRobocasaEvalInputs:
    """从 Robocasa gymnasium obs dict 中提取图像和状态。

    Robocasa obs 格式:
        video.ego_view_pad_res256_freq20: (256, 256, 3) uint8
        video.ego_view_bg_crop_pad_res256_freq20: (256, 256, 3) uint8
        state.left_arm: (7,)
        state.right_arm: (7,)
        state.left_hand: (6,)
        state.right_hand: (6,)
        state.waist: (3,)
        annotation.human.coarse_action: str

    输出:
        images: list of numpy (H, W, 3)
        states: numpy (29,)
        task_description: str
        replay_img: numpy (H, W, 3) 用于录制回放视频
    """

    def __init__(self,
                 img_key: str = 'video.ego_view_pad_res256_freq20',
                 resize_size: int = 224,
                 center_crop_scale: Optional[float] = None,
                 normalize: bool = True,
                 embodiment_id: Optional[int] = None):
        if center_crop_scale is not None and not (0 < center_crop_scale <= 1):
            raise ValueError(
                f'center_crop_scale must be in (0, 1], got {center_crop_scale}')
        self.img_key = img_key
        self.resize_size = resize_size
        self.center_crop_scale = center_crop_scale
        self.normalize = normalize  # 是否做 pixel/255 归一化
        # 与训练侧 ProcessParquetInputs 配置对齐；当前评测管线不消费该字段，仅避免 cfg 报错
        self.embodiment_id = embodiment_id

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        if self.center_crop_scale is None or self.center_crop_scale == 1:
            return img
        h, w = img.shape[:2]
        crop_h = max(1, int(round(h * self.center_crop_scale)))
        crop_w = max(1, int(round(w * self.center_crop_scale)))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return img[top:top + crop_h, left:left + crop_w]

    def __call__(self, data: Dict) -> Dict:
        result = {}

        # --- 图像提取 + 处理 ---
        img = data.get(self.img_key, None)
        if img is not None:
            # 保存原始图像用于回放视频
            result['replay_img'] = img.copy()

            # 官方 GR00T eval 的 VideoCrop(scale=0.95) 在 eval 模式下为 center crop。
            img = self._center_crop(img)

            # Resize 到模型期望尺寸 (robocasa 输出 256x256 → 224x224)
            if img.shape[0] != self.resize_size or img.shape[1] != self.resize_size:
                img = cv2.resize(img, (self.resize_size, self.resize_size))

            if self.normalize:
                # PI0.5: 归一化 uint8 [0,255] → float32 [0,1]
                img = img.astype(np.float32) / 255.0
                # HWC → CHW
                img = np.transpose(img, (2, 0, 1))  # (3, 224, 224)
                # 转为 tensor
                pixel_values = torch.from_numpy(img).float()  # (3, 224, 224)
                result['pixel_values'] = pixel_values
            else:
                # GR00T: 保留 uint8 [0,255]，输出为 'pixel_values' 供后续 TransformImage 处理
                # 保持 HWC 格式，不转 CHW
                result['pixel_values'] = img  # (224, 224, 3) uint8

            result['img_masks'] = np.array([True])
        else:
            raise ValueError(
                f'Image key {self.img_key} not found in obs. '
                f'Available keys: {list(data.keys())}')

        # --- 状态提取: 拼接 29 维关节角度 ---
        state_parts = []
        for key in ROBOCASA_STATE_KEYS:
            val = data.get(key, None)
            if val is not None:
                state_parts.append(np.array(val, dtype=np.float64))
        if state_parts:
            result['states'] = np.concatenate(state_parts)  # (29,)
        else:
            raise ValueError(
                f'State keys not found in obs. '
                f'Available keys: {list(data.keys())}')

        # --- 任务描述 ---
        result['task_description'] = data.get('task_description', '')

        if self.embodiment_id is not None:
            result['embodiment_ids'] = np.array(
                self.embodiment_id, dtype=np.int32)

        # --- 传递 norm_stats (如果有) ---
        if 'norm_stats' in data:
            result['norm_stats'] = data['norm_stats']
        if 'stats' in data:
            result['stats'] = data['stats']

        return result


@TRANSFORMS.register_module()
class DenormalizeRobocasaAction:
    """Robocasa 动作反归一化。

    与 DenormalizeLiberoAction 的差异:
    - stats key 不附加 '_no_noops' 后缀
    - 无 gripper binarize / invert 后处理
    - 默认 min_max 归一化

    Args:
        norm_stats: 归一化统计量文件路径或 dict
        action_dim: 实际动作维度 (29)
        norm_type: 归一化类型 ('min_max')
    """

    def __init__(self,
                 norm_stats: str,
                 action_dim: int = 29,
                 norm_type: str = 'min_max',
                 clip_actions: bool = False,
                 stats_order: str = 'fluxvla'):
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.clip_actions = clip_actions
        if stats_order != 'fluxvla':
            raise ValueError(
                f'Unsupported stats_order={stats_order}. '
                'Only existing FluxVLA RoboCasa statistics are supported.')
        self.stats_permutation = np.array(
            _robocasa_gr1_permutation(ROBOCASA_GR1_FLUXVLA_ORDER),
            dtype=np.int64)

    def __call__(self, data: Dict) -> np.ndarray:
        """反归一化动作。

        Args:
            data: dict with 'action' (numpy) and 'task_suite_name' (str)

        Returns:
            numpy array of denormalized action, shape (action_dim,)
        """
        task_key = data.get('task_suite_name', '')
        # Robocasa 不用 '_no_noops' 后缀
        if task_key in self.norm_stats:
            stats = self.norm_stats[task_key]
        else:
            raise KeyError(
                f'Stats key "{task_key}" not found. '
                f'Available: {list(self.norm_stats.keys())}')

        action = data['action']
        action_stats = self._reorder_action_stats(stats['action'])

        # 先截断到实际维度
        action = action[:self.action_dim]

        if self.norm_type == 'min_max':
            action = self._denormalize_min_max(action, action_stats)
        elif self.norm_type == 'mean_std':
            mean = np.array(action_stats['mean'])[:self.action_dim]
            std = np.array(action_stats['std'])[:self.action_dim]
            action = action * std + mean
        else:
            raise ValueError(f'Unknown norm_type: {self.norm_type}')

        return action

    def _reorder_action_stats(self, action_stats: Dict) -> Dict:
        action_stats = copy.deepcopy(action_stats)
        for key in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
            values = action_stats.get(key)
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.shape[-1] == len(self.stats_permutation):
                action_stats[key] = arr[..., self.stats_permutation].tolist()
        return action_stats

    def _denormalize_min_max(self, action: np.ndarray,
                              stats: Dict) -> np.ndarray:
        low = np.array(stats['min'])[:self.action_dim]
        high = np.array(stats['max'])[:self.action_dim]
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        # min_max 反归一化: action ∈ [-1, 1] → [low, high]
        return 0.5 * (action + 1) * (high - low) + low


@DATASETS.register_module()
class RobocasaEvalDataset:
    """Robocasa 评测数据集 — 将 gymnasium obs 转换为模型输入 batch。

    类似 LiberoParquetEvalDataset，但适配 Robocasa obs 格式。

    Args:
        norm_stats: 归一化统计量路径或 dict
        unnorm_key: stats dict 中的 key (如 'robocasa_gr1_test')
        transforms: transform 配置列表
    """

    def __init__(self,
                 norm_stats: Any = None,
                 unnorm_key: str = 'robocasa_gr1_test',
                 transforms: List[Dict] = None,
                 **kwargs) -> None:
        from fluxvla.engines import build_transform_from_cfg

        self.transforms = [build_transform_from_cfg(t)
                           for t in (transforms or [])]
        self.unnorm_key = unnorm_key
        # 分组评测时由 RobocasaEvalRunner 每任务 set_active_stats_blob 覆盖
        self._active_stats_blob: Optional[Dict] = None
        self.last_debug: Dict[str, Any] = {}

        if isinstance(norm_stats, str) and norm_stats:
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats

    def set_active_stats_blob(self, blob: Optional[Dict]) -> None:
        """指定本步使用的 stats（proprio/action/...），用于分组 dataset_statistics。

        blob 为 dataset_statistics*.json 中 unnorm_key 对应的那一层 dict。
        传 None 则恢复为使用 self.norm_stats[self.unnorm_key]。
        """
        self._active_stats_blob = blob

    def __call__(self, inputs: Dict) -> tuple:
        """将 Robocasa obs 转换为模型输入。

        Args:
            inputs: gymnasium obs dict

        Returns:
            (batch_dict, replay_img)
        """
        data = dict(inputs)

        # 注入归一化统计量 (NormalizeStatesAndActions 需要)
        if self._active_stats_blob is not None:
            data['stats'] = self._active_stats_blob
        elif self.norm_stats is not None and self.unnorm_key in self.norm_stats:
            data['stats'] = self.norm_stats[self.unnorm_key]

        # 执行 transform pipeline
        for t in self.transforms:
            data = t(data)

        replay_img = data.get('replay_img', None)
        self.last_debug = {
            'task_description': data.get('task_description', ''),
            'prompt': data.get('prompt', ''),
            'text': data.get('text', ''),
        }

        # 组装 batch (与 LiberoParquetEvalDataset 格式一致)
        assert 'lang_tokens' in data and 'lang_masks' in data, \
            'Prompt transform must provide lang_tokens and lang_masks'

        tokens = torch.tensor(data['lang_tokens'])
        token_mask = data['lang_masks'].tolist() if hasattr(
            data['lang_masks'], 'tolist') else list(data['lang_masks'])

        pixel_values = data['pixel_values']
        img_masks = data.get('img_masks', None)
        if img_masks is None:
            num_imgs = pixel_values.shape[0] // 3
            img_masks = [True] * num_imgs
        else:
            img_masks = list(img_masks)

        batch = dict(
            images=pixel_values.cuda().unsqueeze(0),
            img_masks=torch.tensor([img_masks]).cuda(),
            lang_tokens=tokens.unsqueeze(0).cuda(),
            lang_masks=torch.tensor(token_mask).unsqueeze(0).cuda(),
        )

        if 'states' in data:
            batch['states'] = torch.from_numpy(
                data['states']).bfloat16().cuda().unsqueeze(0)

        # GR00T FlowMatchingInferenceHead.predict_action 会 copy_ 到 buffer，不能为 None。
        # 若上游未显式传入 embodiment_id，则退回到历史默认 0。
        bsz = batch['images'].shape[0]
        dev = batch['images'].device
        if 'embodiment_ids' in data:
            eid = torch.as_tensor(
                data['embodiment_ids'], dtype=torch.long, device=dev).reshape(-1)
            if eid.numel() == 1 and bsz > 1:
                eid = eid.expand(bsz)
            elif eid.numel() != bsz:
                raise ValueError(
                    f'embodiment_ids length {eid.numel()} != batch size {bsz}')
            batch['embodiment_ids'] = eid
        else:
            batch['embodiment_ids'] = torch.zeros(bsz, dtype=torch.long, device=dev)

        return batch, replay_img
