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

import torch
import torch.nn.functional as F

from fluxvla.engines import VLAS
from fluxvla.engines.utils.model_utils import create_sinusoidal_pos_embedding
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .pi0_flowmatching import PI0FlowMatching

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI05FlowMatching(PI0FlowMatching):
    """PI0 Flow Matching Model for Vision-Language Alignment.
    Implemented based on https://arxiv.org/abs/2504.16054

    This model is designed to handle vision-language alignment tasks
    using flow matching techniques, leveraging a vision backbone,
    language model backbone, projector, and a VLA head.

    Args:
        state_proj (Dict): Configuration dictionary for the state
            projector.
        action_in_proj (Dict): Configuration dictionary for the action
            input projector.
        action_out_proj (Dict): Configuration dictionary for the action
            output projector.
        action_time_mlp_in (Dict): Configuration dictionary for the action
            time MLP input.
        action_time_mlp_out (Dict): Configuration dictionary for the action
            time MLP output.
        vlm_backbone (str): Identifier for the vision-language model backbone.
        vla_head (str): Identifier for the vision-language alignment head.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
        freeze_vision_backbone (bool): Whether to freeze the vision backbone.
        freeze_llm_backbone (bool): Whether to freeze the language model
            backbone.
        freeze_projector (bool): Whether to freeze the projector.
        vision_backbone_fp32 (bool): Whether to use FP32 for the vision
            backbone.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer
            of the model.
        ignore_index (int): Index to ignore in loss calculations.
        norm_stats (Dict, optional): Normalization statistics for the model.
        **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(self, **kwargs):
        rtc_cfg = kwargs.get('rtc_training_config')
        if rtc_cfg and rtc_cfg.get('enabled', False):
            raise ValueError(
                'PI05FlowMatching does not support training-time RTC. '
                'Its architecture cannot inject per-position timesteps '
                'without model modifications. Please disable '
                'rtc_training_config or use test-time RTC (guidance) '
                'instead.')
        super().__init__(**kwargs)

    def predict_action(self,
                       *args,
                       rtc_config=None,
                       prev_actions=None,
                       prefix_len=0,
                       **kwargs):
        if (prev_actions is not None and prefix_len > 0 and rtc_config
                and rtc_config.get('method', 'prefix') == 'prefix'):
            raise ValueError(
                'PI05FlowMatching does not support RTC prefix mode at '
                'inference. Its embed_suffix only accepts a scalar timestep '
                'and cannot handle per-position time injection. '
                "Use method='guidance' for test-time RTC instead.")
        return super().predict_action(
            *args,
            rtc_config=rtc_config,
            prev_actions=prev_actions,
            prefix_len=prefix_len,
            **kwargs)

    def embed_suffix(self, states, noisy_actions, timestep):
        """Embed the suffix tokens for the Pi0 head.

        Args:
            state (torch.Tensor): The state tensor of shape (bsize, state_dim).
            noisy_actions (torch.Tensor): The noisy actions tensor of shape
                (bsize, n_action_steps, action_dim).
            timestep (torch.Tensor): The timestep tensor of shape (bsize,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
                containing the embedded suffix tokens, padding masks,
                and attention masks.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        bsize = states.shape[0]
        dtype = states.dtype
        device = states.device
        if self.state_proj is not None:
            state_emb = self.state_proj(states)
            embs.append(state_emb[:, None, :])
            pad_masks.append(
                torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        # Set attention masks so that image and language
        # inputs do not attend to state or actions

        # Embed timestep using sine-cosine positional
        # encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device)
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = F.silu(self.time_mlp_in(time_emb))
        time_emb = F.silu(self.time_mlp_out(time_emb))
        # Add to input tokens
        embs.append(action_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state
        # inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, time_emb
