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

from functools import partial
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.distributions import Beta

from fluxvla.engines import HEADS
from fluxvla.models.blocks import SelfAttentionTransformer
from fluxvla.models.blocks.cross_attention_dit import DiT


def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=device) * (
                torch.log(torch.tensor(10000.0)) / half_dim)
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class CategorySpecificLinear(nn.Module):
    """
    Category specific linear layer for DiT.

    Args:
        num_categories: The number of categories.
        input_dim: The dimension of the input.
        hidden_dim: The dimension of the hidden states.
    """

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(
            0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """
    Category specific MLP for DiT.

    Args:
        num_categories: The number of categories.
        input_dim: The dimension of the input.
        hidden_dim: The dimension of the hidden states.
        output_dim: The dimension of the output.
    """

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim,
                                             hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim,
                                             output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """
    Multi-embodiment action encoder for DiT.

    Args:
        action_dim: The dimension of the action.
        hidden_size: The dimension of the hidden states.
        num_embodiments: The number of embodiments.
    """

    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim,
                                         hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size,
                                         hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size,
                                         hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T) -- per-sample or per-position
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Accept (B,) or (B, T) timesteps
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2:
            assert timesteps.shape == (B, T)
        else:
            raise ValueError(f'Expected timesteps shape (B,) or (B,T), got '
                             f'{timesteps.shape}')

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@HEADS.register_module()
class FlowMatchingHead(nn.Module):
    """
    FlowMatchingHead for DiT.

    Args:
        hidden_size: The dimension of the hidden states.
        state_dim: The dimension of the state.
        input_embedding_dim: The dimension of the input embedding.
        action_dim: The dimension of the action.
        num_inference_timesteps: The number of inference timesteps.
        max_num_embodiments: The maximum number of embodiments.
        use_vlln: Whether to use VLLN.
        num_target_vision_tokens: The number of target vision tokens.
        backbone_embedding_dim: The dimension of the backbone embedding.
        vl_self_attention_cfg: The configuration for the VL self-attention.
        add_positional_embeddings: Whether to add positional embeddings.
        max_seq_len: The maximum sequence length.
        num_timestep_buckets: The number of timestep buckets.
        noise_s: The noise scale.
        noise_beta_alpha: The alpha for the noise beta distribution.
        noise_beta_beta: The beta for the noise beta distribution.
        num_steps: The number of steps.
        diffusion_model_cfg: The configuration for the diffusion model.
    """

    def __init__(self,
                 hidden_size: int,
                 state_dim: int,
                 input_embedding_dim: int,
                 action_dim: int,
                 num_inference_timesteps: int,
                 max_num_embodiments: int = 32,
                 use_vlln: bool = True,
                 num_target_vision_tokens: int = 32,
                 backbone_embedding_dim: int = 2048,
                 vl_self_attention_cfg: Dict = dict(
                     attention_head_dim=64,
                     dropout=0.2,
                     final_dropout=True,
                     num_attention_heads=32,
                     num_layers=4,
                     positional_embeddings=None),
                 add_positional_embeddings: bool = True,
                 max_seq_len: int = 1024,
                 num_timestep_buckets: int = 1000,
                 noise_s: float = 0.999,
                 noise_beta_alpha: float = 1.5,
                 noise_beta_beta: float = 1.0,
                 num_steps: int = 10,
                 diffusion_model_cfg: Dict = dict(
                     attention_head_dim=48,
                     cross_attention_dim=2048,
                     dropout=0.2,
                     final_dropout=True,
                     interleave_self_attention=True,
                     norm_type='ada_norm',
                     num_attention_heads=32,
                     num_layers=16,
                     output_dim=1024,
                     positional_embeddings=None),
                 ori_action_dim=None,
                 rtc_training_config=None,
                 *args,
                 **kwargs):
        super().__init__()
        self.rtc_training_config = rtc_training_config
        self.hidden_size = hidden_size
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=state_dim,
            hidden_dim=hidden_size,
            output_dim=input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=input_embedding_dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )
        self.model = DiT(**diffusion_model_cfg)
        self.input_embedding_dim = input_embedding_dim
        self.action_dim = action_dim
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_s = noise_s
        self.future_tokens = nn.Embedding(num_target_vision_tokens,
                                          self.input_embedding_dim)
        self.add_positional_embeddings = add_positional_embeddings
        self.num_steps = num_steps
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(backbone_embedding_dim)
            if use_vlln else nn.Identity())
        self.vl_self_attention = (
            SelfAttentionTransformer(
                **vl_self_attention_cfg) if use_vlln else nn.Identity())
        if add_positional_embeddings:
            self.position_embedding = nn.Embedding(max_seq_len,
                                                   self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.ori_action_dim = ori_action_dim

    def forward(self, input_features: torch.Tensor, states: torch.Tensor,
                attention_mask: torch.Tensor, embodiment_ids: torch.Tensor,
                actions: torch.Tensor, action_masks: torch.Tensor, **kwargs):
        input_features = self.vlln(input_features)
        input_features = self.vl_self_attention(input_features)
        state_features = self.state_encoder(
            states.unsqueeze(1), embodiment_ids)
        noise = torch.randn(
            actions.shape, device=actions.device, dtype=actions.dtype)
        t_scalar = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype)
        T = actions.shape[1]
        if (self.rtc_training_config
                and self.rtc_training_config.get('enabled', False)):
            from fluxvla.engines.utils.rtc_training import (
                apply_rtc_time_conditioning, sample_training_delay)
            delays = sample_training_delay(
                batch_size=actions.shape[0],
                max_delay=self.rtc_training_config.get('max_delay', 5),
                distribution=self.rtc_training_config.get(
                    'distribution', 'exponential'),
                device=actions.device)
            t, action_masks = apply_rtc_time_conditioning(
                t_scalar, action_masks, delays, T)  # (B, T)
        else:
            t = t_scalar.unsqueeze(1).expand(-1, T)  # (B, T)

        t_encoder = (t * self.num_timestep_buckets).long()  # (B, T)
        noisy_trajectory = (
            1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * actions
        velocity = actions - noise
        action_features = self.action_encoder(noisy_trajectory, t_encoder,
                                              embodiment_ids)

        # Maybe add position embedding.
        if self.add_positional_embeddings:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=actions.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dim.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            input_features.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features),
                            dim=1)

        vl_attn_mask = attention_mask

        # DiT AdaLN uses global time — always pass scalar (B,)
        t_global = (t_scalar * self.num_timestep_buckets).long()
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=input_features,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_global,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_ids)
        pred_actions = pred[:, -actions.shape[1]:]

        if self.ori_action_dim is not None:
            pred_actions = pred_actions[:, :, :self.ori_action_dim]
            velocity = velocity[:, :, :self.ori_action_dim]

        # Slice out only the action portion of pred and target.
        loss = F.mse_loss(
            pred_actions, velocity,
            reduction='none') * action_masks.unsqueeze(-1)
        loss = loss.sum() / (action_masks.sum() * actions.shape[-1])

        return dict(
            pred_actions=pred_actions,
            loss=loss,
        )

    def denoise_step(self,
                     actions,
                     input_features,
                     state_features,
                     attention_mask,
                     embodiment_ids,
                     t_global,
                     t_encoder=None):
        """Single denoising step extracted from predict_action loop.

        Args:
            actions: Current noisy actions (B, T, D).
            input_features: Processed VL features (B, S, H).
            state_features: Encoded state (B, 1, H).
            attention_mask: VL attention mask.
            embodiment_ids: Embodiment IDs (B,).
            t_global: Discretized timestep (B,) for DiT AdaLN.
            t_encoder: Per-position timestep (B, T) for action
                encoder. If None, uses t_global.

        Returns:
            Predicted velocity (B, T, D).
        """
        t_enc = t_encoder if t_encoder is not None else t_global
        action_features = self.action_encoder(actions, t_enc, embodiment_ids)
        if self.add_positional_embeddings:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=actions.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            input_features.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features),
                            dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=input_features,
            encoder_attention_mask=attention_mask,
            timestep=t_global,
        )
        pred = self.action_decoder(model_output, embodiment_ids)
        return pred[:, -self.num_steps:]

    def _predict_action_plain(self, actions, denoise, batch_size, device, dt):
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            t_global = torch.full((batch_size, ),
                                  fill_value=t_discretized,
                                  device=device)
            v = denoise(actions, t_global)
            actions = actions + dt * v
        return actions

    def _predict_action_prefix_rtc(self, actions, denoise, batch_size, device,
                                   dt, prev_actions, prefix_len):
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            t_global = torch.full((batch_size, ),
                                  fill_value=t_discretized,
                                  device=device)

            actions[:, :prefix_len] = prev_actions[:, :prefix_len]
            t_enc = torch.full((batch_size, self.num_steps),
                               fill_value=t_discretized,
                               dtype=torch.long,
                               device=device)
            t_enc[:, :prefix_len] = self.num_timestep_buckets
            v = denoise(actions, t_global, t_enc)
            actions = actions + dt * v

        actions[:, :prefix_len] = prev_actions[:, :prefix_len]
        return actions

    def _predict_action_guidance_rtc(self, actions, denoise, batch_size,
                                     device, dt, prev_actions, prefix_len,
                                     rtc_config):
        from fluxvla.engines.utils.rtc_guidance import (apply_rtc_guidance,
                                                        compute_prefix_weights)

        prefix_weights = compute_prefix_weights(
            self.num_steps,
            prefix_len,
            rtc_config.get('decay_end', prefix_len * 2),
            rtc_config.get('schedule', 'exp'),
            device=device)
        max_gw = rtc_config.get('max_guidance_weight', 5.0)
        use_vjp = rtc_config.get('use_vjp', False)

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            t_global = torch.full((batch_size, ),
                                  fill_value=t_discretized,
                                  device=device)
            v = apply_rtc_guidance(
                actions,
                lambda x: denoise(x, t_global),
                prev_actions,
                prefix_weights,
                t_cont,
                max_gw,
                use_vjp=use_vjp)
            actions = actions + dt * v
        return actions

    def predict_action(self,
                       input_features: torch.Tensor,
                       states: torch.Tensor,
                       attention_mask: torch.Tensor,
                       embodiment_ids: torch.Tensor,
                       prev_actions=None,
                       prefix_len: int = 0,
                       rtc_config: dict = None):
        device = input_features.device
        input_features = self.vlln(input_features)
        input_features = self.vl_self_attention(input_features)
        batch_size = input_features.shape[0]
        state_features = self.state_encoder(
            states.unsqueeze(1), embodiment_ids)
        actions = torch.randn(
            size=(batch_size, self.num_steps, self.action_dim),
            dtype=input_features.dtype,
            device=input_features.device,
        )
        dt = 1.0 / self.num_inference_timesteps

        if (prev_actions is not None and self.ori_action_dim is not None
                and prev_actions.shape[-1] < self.action_dim):
            pad_size = self.action_dim - prev_actions.shape[-1]
            prev_actions = F.pad(prev_actions, (0, pad_size), value=0.0)

        rtc_method = None
        if prev_actions is not None and prefix_len > 0 and rtc_config:
            rtc_method = rtc_config.get('method', 'prefix')

        def denoise(x, t_global, t_encoder=None):
            return self.denoise_step(x, input_features, state_features,
                                     attention_mask, embodiment_ids, t_global,
                                     t_encoder)

        if rtc_method == 'prefix':
            actions = self._predict_action_prefix_rtc(
                actions=actions,
                denoise=denoise,
                batch_size=batch_size,
                device=device,
                dt=dt,
                prev_actions=prev_actions,
                prefix_len=prefix_len)
        elif rtc_method == 'guidance':
            actions = self._predict_action_guidance_rtc(
                actions=actions,
                denoise=denoise,
                batch_size=batch_size,
                device=device,
                dt=dt,
                prev_actions=prev_actions,
                prefix_len=prefix_len,
                rtc_config=rtc_config)
        else:
            actions = self._predict_action_plain(
                actions=actions,
                denoise=denoise,
                batch_size=batch_size,
                device=device,
                dt=dt)

        if self.ori_action_dim is not None:
            actions = actions[:, :, :self.ori_action_dim]
        return actions

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns a function used to determine which modules to wrap with FSDP.
        """
        return partial(
            _module_wrap_policy,
            module_classes=set([SelfAttentionTransformer, DiT]),
        )
