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

import math
from typing import Dict

import torch
import torch.nn.functional as F

from fluxvla.engines import HEADS
from fluxvla.ops.atomic_ops import dit_block_cross, dit_block_self, vl_sa_block
from fluxvla.ops.triton.position_embedding import \
    fused_position_embedding_add_inplace
from .flow_matching_head import FlowMatchingHead


def _timestep_embedding(timesteps, channels=256):
    """Sinusoidal timestep embedding (diffusers-compatible,
    flip_sin_to_cos=True, downscale_freq_shift=1)."""
    half_dim = channels // 2
    exponent = -math.log(10000) * torch.arange(
        half_dim, dtype=torch.float32, device=timesteps.device) / (
            half_dim - 1)
    emb = timesteps[:, None].float() * exponent[None, :].exp()
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    return emb


def _sinusoidal_pos_encoding(timesteps, embedding_dim):
    """Sinusoidal positional encoding for action encoder. timesteps: (B, T)."""
    timesteps = timesteps.float()
    half_dim = embedding_dim // 2
    exponent = -torch.arange(
        half_dim, dtype=torch.float, device=timesteps.device) * (
            math.log(10000.0) / half_dim)
    freqs = timesteps.unsqueeze(-1) * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


def _cat_linear(x, W, b, cat_ids):
    """Functional CategorySpecificLinear: x @ W[cat] + b[cat]."""
    return torch.bmm(x, W[cat_ids]) + b[cat_ids].unsqueeze(1)


def _cat_mlp(x, W1, b1, W2, b2, cat_ids):
    """Functional CategorySpecificMLP: relu(linear1) -> linear2."""
    return _cat_linear(
        F.relu(_cat_linear(x, W1, b1, cat_ids)), W2, b2, cat_ids)


def _action_encode(actions, timesteps, W1_W, W1_b, W2_W, W2_b, W3_W, W3_b,
                   cat_ids, hidden_size):
    """Functional MultiEmbodimentActionEncoder."""
    B, T, _ = actions.shape
    timesteps = timesteps.unsqueeze(1).expand(-1, T)
    a_emb = _cat_linear(actions, W1_W, W1_b, cat_ids)
    tau_emb = _sinusoidal_pos_encoding(timesteps,
                                       hidden_size).to(dtype=a_emb.dtype)
    x = torch.cat([a_emb, tau_emb], dim=-1)
    x = F.silu(_cat_linear(x, W2_W, W2_b, cat_ids))
    return _cat_linear(x, W3_W, W3_b, cat_ids)


@HEADS.register_module()
class FlowMatchingInferenceHead(FlowMatchingHead):
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
                 max_input_seq_len: int = 600,
                 *args,
                 **kwargs):
        super().__init__(hidden_size, state_dim, input_embedding_dim,
                         action_dim, num_inference_timesteps,
                         max_num_embodiments, use_vlln,
                         num_target_vision_tokens, backbone_embedding_dim,
                         vl_self_attention_cfg, add_positional_embeddings,
                         max_seq_len, num_timestep_buckets, noise_s,
                         noise_beta_alpha, noise_beta_beta, num_steps,
                         diffusion_model_cfg, ori_action_dim, *args, **kwargs)

        # ---- Derived hyperparameters ----
        E = max_num_embodiments
        sd = state_dim
        hs = hidden_size
        ied = input_embedding_dim
        ad = action_dim
        bed = backbone_embedding_dim
        nvt = num_target_vision_tokens
        msl = max_seq_len
        temb_ch = 256  # timestep embedding channels (sinusoidal)

        # VL self-attention
        vl_nh = vl_self_attention_cfg['num_attention_heads']
        vl_hd = vl_self_attention_cfg['attention_head_dim']
        vl_nl = vl_self_attention_cfg['num_layers']
        vl_dim = vl_nh * vl_hd
        vl_ff = vl_dim * 4

        # DiT
        dit_nh = diffusion_model_cfg['num_attention_heads']
        dit_hd = diffusion_model_cfg['attention_head_dim']
        dit_nl = diffusion_model_cfg['num_layers']
        dit_cad = diffusion_model_cfg['cross_attention_dim']
        dit_od = diffusion_model_cfg['output_dim']
        dit_dim = dit_nh * dit_hd
        dit_ff = dit_dim * 4
        dit_ns = dit_nl // 2  # self-attn block count (odd layers)

        # Store for record_run
        self.vl_nh = vl_nh
        self.vl_hd = vl_hd
        self.vl_nl = vl_nl
        self.vl_dim = vl_dim
        self.vl_ff = vl_ff
        self.dit_nh = dit_nh
        self.dit_hd = dit_hd
        self.dit_nl = dit_nl
        self.dit_dim = dit_dim
        self.dit_ff = dit_ff
        self.max_input_seq_len = max_input_seq_len
        self.backbone_embedding_dim = bed
        self.state_dim = sd

        self.weights = {
            # State encoder (embodiment-conditioned)
            'state_encoder_layer1_W':
            torch.empty(E, sd, hs, dtype=torch.bfloat16, device='cuda'),
            'state_encoder_layer1_b':
            torch.empty(E, hs, dtype=torch.bfloat16, device='cuda'),
            'state_encoder_layer2_W':
            torch.empty(E, hs, ied, dtype=torch.bfloat16, device='cuda'),
            'state_encoder_layer2_b':
            torch.empty(E, ied, dtype=torch.bfloat16, device='cuda'),
            # Action encoder (embodiment-conditioned)
            'action_encoder_W1_W':
            torch.empty(E, ad, ied, dtype=torch.bfloat16, device='cuda'),
            'action_encoder_W1_b':
            torch.empty(E, ied, dtype=torch.bfloat16, device='cuda'),
            'action_encoder_W2_W':
            torch.empty(E, 2 * ied, ied, dtype=torch.bfloat16, device='cuda'),
            'action_encoder_W2_b':
            torch.empty(E, ied, dtype=torch.bfloat16, device='cuda'),
            'action_encoder_W3_W':
            torch.empty(E, ied, ied, dtype=torch.bfloat16, device='cuda'),
            'action_encoder_W3_b':
            torch.empty(E, ied, dtype=torch.bfloat16, device='cuda'),
            # Action decoder (embodiment-conditioned)
            'action_decoder_layer1_W':
            torch.empty(E, hs, hs, dtype=torch.bfloat16, device='cuda'),
            'action_decoder_layer1_b':
            torch.empty(E, hs, dtype=torch.bfloat16, device='cuda'),
            'action_decoder_layer2_W':
            torch.empty(E, hs, ad, dtype=torch.bfloat16, device='cuda'),
            'action_decoder_layer2_b':
            torch.empty(E, ad, dtype=torch.bfloat16, device='cuda'),
            # DiT timestep encoder
            'dit_timestep_linear1_w':
            torch.empty(dit_dim, temb_ch, dtype=torch.bfloat16, device='cuda'),
            'dit_timestep_linear1_b':
            torch.empty(dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_timestep_linear2_w':
            torch.empty(dit_dim, dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_timestep_linear2_b':
            torch.empty(dit_dim, dtype=torch.bfloat16, device='cuda'),
            # DiT transformer blocks (stacked)
            'dit_norm1_linear_w':
            torch.empty(
                dit_nl,
                2 * dit_dim,
                dit_dim,
                dtype=torch.bfloat16,
                device='cuda'),
            'dit_norm1_linear_b':
            torch.empty(
                dit_nl, 2 * dit_dim, dtype=torch.bfloat16, device='cuda'),
            # Self-attention blocks (odd) - merged QKV
            'dit_self_qkv_w':
            torch.empty(
                dit_ns,
                3 * dit_dim,
                dit_dim,
                dtype=torch.bfloat16,
                device='cuda'),
            'dit_self_qkv_b':
            torch.empty(
                dit_ns, 3 * dit_dim, dtype=torch.bfloat16, device='cuda'),
            # Cross-attention blocks (even) - Q separate, KV merged
            'dit_cross_q_w':
            torch.empty(
                dit_ns, dit_dim, dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_cross_q_b':
            torch.empty(dit_ns, dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_cross_kv_w':
            torch.empty(
                dit_ns,
                2 * dit_dim,
                dit_cad,
                dtype=torch.bfloat16,
                device='cuda'),
            'dit_cross_kv_b':
            torch.empty(
                dit_ns, 2 * dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_attn_out_w':
            torch.empty(
                dit_nl, dit_dim, dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_attn_out_b':
            torch.empty(dit_nl, dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_ff_up_w_T':
            torch.empty(
                dit_nl, dit_dim, dit_ff, dtype=torch.bfloat16, device='cuda'),
            'dit_ff_up_b':
            torch.empty(dit_nl, dit_ff, dtype=torch.bfloat16, device='cuda'),
            'dit_ff_down_w':
            torch.empty(
                dit_nl, dit_dim, dit_ff, dtype=torch.bfloat16, device='cuda'),
            'dit_ff_down_b':
            torch.empty(dit_nl, dit_dim, dtype=torch.bfloat16, device='cuda'),
            # DiT output projection
            'dit_proj_out_1_w':
            torch.empty(
                2 * dit_dim, dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_proj_out_1_b':
            torch.empty(2 * dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_proj_out_2_w':
            torch.empty(dit_od, dit_dim, dtype=torch.bfloat16, device='cuda'),
            'dit_proj_out_2_b':
            torch.empty(dit_od, dtype=torch.bfloat16, device='cuda'),
            # Future tokens
            'future_tokens_w':
            torch.empty(nvt, ied, dtype=torch.bfloat16, device='cuda'),
            # VLLN (LayerNorm)
            'vlln_w':
            torch.empty(bed, dtype=torch.float32, device='cuda'),
            'vlln_b':
            torch.empty(bed, dtype=torch.float32, device='cuda'),
            # VL self-attention (stacked)
            'vl_sa_norm1_w':
            torch.empty(vl_nl, vl_dim, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_norm1_b':
            torch.empty(vl_nl, vl_dim, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_qkv_w':
            torch.empty(
                vl_nl, 3 * vl_dim, vl_dim, dtype=torch.bfloat16,
                device='cuda'),
            'vl_sa_qkv_b':
            torch.empty(
                vl_nl, 3 * vl_dim, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_attn_out_w':
            torch.empty(
                vl_nl, vl_dim, vl_dim, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_attn_out_b':
            torch.empty(vl_nl, vl_dim, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_norm3_w':
            torch.empty(vl_nl, vl_dim, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_norm3_b':
            torch.empty(vl_nl, vl_dim, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_ff_up_w_T':
            torch.empty(
                vl_nl, vl_dim, vl_ff, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_ff_up_b':
            torch.empty(vl_nl, vl_ff, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_ff_down_w':
            torch.empty(
                vl_nl, vl_dim, vl_ff, dtype=torch.bfloat16, device='cuda'),
            'vl_sa_ff_down_b':
            torch.empty(vl_nl, vl_dim, dtype=torch.bfloat16, device='cuda'),
            # Position embedding
            'position_embedding_w':
            torch.empty(msl, ied, dtype=torch.bfloat16, device='cuda'),
        }

        self.loaded_weights = False

    def _load_weights_and_buffer(self):
        # State encoder
        self.weights['state_encoder_layer1_W'].copy_(
            self.state_encoder.layer1.W.data)
        self.weights['state_encoder_layer1_b'].copy_(
            self.state_encoder.layer1.b.data)
        self.weights['state_encoder_layer2_W'].copy_(
            self.state_encoder.layer2.W.data)
        self.weights['state_encoder_layer2_b'].copy_(
            self.state_encoder.layer2.b.data)
        # Action encoder
        self.weights['action_encoder_W1_W'].copy_(
            self.action_encoder.W1.W.data)
        self.weights['action_encoder_W1_b'].copy_(
            self.action_encoder.W1.b.data)
        self.weights['action_encoder_W2_W'].copy_(
            self.action_encoder.W2.W.data)
        self.weights['action_encoder_W2_b'].copy_(
            self.action_encoder.W2.b.data)
        self.weights['action_encoder_W3_W'].copy_(
            self.action_encoder.W3.W.data)
        self.weights['action_encoder_W3_b'].copy_(
            self.action_encoder.W3.b.data)
        # Action decoder
        self.weights['action_decoder_layer1_W'].copy_(
            self.action_decoder.layer1.W.data)
        self.weights['action_decoder_layer1_b'].copy_(
            self.action_decoder.layer1.b.data)
        self.weights['action_decoder_layer2_W'].copy_(
            self.action_decoder.layer2.W.data)
        self.weights['action_decoder_layer2_b'].copy_(
            self.action_decoder.layer2.b.data)
        # DiT timestep encoder
        self.weights['dit_timestep_linear1_w'].copy_(
            self.model.timestep_encoder.timestep_embedder.linear_1.weight)
        self.weights['dit_timestep_linear1_b'].copy_(
            self.model.timestep_encoder.timestep_embedder.linear_1.bias)
        self.weights['dit_timestep_linear2_w'].copy_(
            self.model.timestep_encoder.timestep_embedder.linear_2.weight)
        self.weights['dit_timestep_linear2_b'].copy_(
            self.model.timestep_encoder.timestep_embedder.linear_2.bias)
        # DiT transformer blocks (16 blocks)
        blocks = self.model.transformer_blocks
        nl = self.dit_nl
        self.weights['dit_norm1_linear_w'].copy_(
            torch.stack([blocks[i].norm1.linear.weight for i in range(nl)]))
        self.weights['dit_norm1_linear_b'].copy_(
            torch.stack([blocks[i].norm1.linear.bias for i in range(nl)]))
        # Self-attention merged QKV (odd blocks)
        self_indices = list(range(1, nl, 2))
        self.weights['dit_self_qkv_w'].copy_(
            torch.stack([
                torch.cat([
                    blocks[i].attn1.to_q.weight, blocks[i].attn1.to_k.weight,
                    blocks[i].attn1.to_v.weight
                ],
                          dim=0) for i in self_indices
            ]))
        self.weights['dit_self_qkv_b'].copy_(
            torch.stack([
                torch.cat([
                    blocks[i].attn1.to_q.bias, blocks[i].attn1.to_k.bias,
                    blocks[i].attn1.to_v.bias
                ],
                          dim=0) for i in self_indices
            ]))
        # Cross-attention Q + merged KV (even blocks)
        cross_indices = list(range(0, nl, 2))
        self.weights['dit_cross_q_w'].copy_(
            torch.stack([blocks[i].attn1.to_q.weight for i in cross_indices]))
        self.weights['dit_cross_q_b'].copy_(
            torch.stack([blocks[i].attn1.to_q.bias for i in cross_indices]))
        self.weights['dit_cross_kv_w'].copy_(
            torch.stack([
                torch.cat(
                    [blocks[i].attn1.to_k.weight, blocks[i].attn1.to_v.weight],
                    dim=0) for i in cross_indices
            ]))
        self.weights['dit_cross_kv_b'].copy_(
            torch.stack([
                torch.cat(
                    [blocks[i].attn1.to_k.bias, blocks[i].attn1.to_v.bias],
                    dim=0) for i in cross_indices
            ]))
        self.weights['dit_attn_out_w'].copy_(
            torch.stack([blocks[i].attn1.to_out[0].weight for i in range(nl)]))
        self.weights['dit_attn_out_b'].copy_(
            torch.stack([blocks[i].attn1.to_out[0].bias for i in range(nl)]))
        self.weights['dit_ff_up_w_T'].copy_(
            torch.stack([blocks[i].ff.net[0].proj.weight
                         for i in range(nl)]).permute(0, 2, 1))
        self.weights['dit_ff_up_b'].copy_(
            torch.stack([blocks[i].ff.net[0].proj.bias for i in range(nl)]))
        self.weights['dit_ff_down_w'].copy_(
            torch.stack([blocks[i].ff.net[2].weight for i in range(nl)]))
        self.weights['dit_ff_down_b'].copy_(
            torch.stack([blocks[i].ff.net[2].bias for i in range(nl)]))
        # DiT output projection
        self.weights['dit_proj_out_1_w'].copy_(self.model.proj_out_1.weight)
        self.weights['dit_proj_out_1_b'].copy_(self.model.proj_out_1.bias)
        self.weights['dit_proj_out_2_w'].copy_(self.model.proj_out_2.weight)
        self.weights['dit_proj_out_2_b'].copy_(self.model.proj_out_2.bias)
        # Future tokens
        self.weights['future_tokens_w'].copy_(self.future_tokens.weight)
        # VLLN (LayerNorm)
        self.weights['vlln_w'].copy_(self.vlln.weight)
        self.weights['vlln_b'].copy_(self.vlln.bias)
        # VL self-attention blocks
        vl_blocks = self.vl_self_attention.transformer_blocks
        vl_n = self.vl_nl
        self.weights['vl_sa_norm1_w'].copy_(
            torch.stack([vl_blocks[i].norm1.weight for i in range(vl_n)]))
        self.weights['vl_sa_norm1_b'].copy_(
            torch.stack([vl_blocks[i].norm1.bias for i in range(vl_n)]))
        self.weights['vl_sa_qkv_w'].copy_(
            torch.stack([
                torch.cat([
                    vl_blocks[i].attn1.to_q.weight,
                    vl_blocks[i].attn1.to_k.weight,
                    vl_blocks[i].attn1.to_v.weight
                ],
                          dim=0) for i in range(vl_n)
            ]))
        self.weights['vl_sa_qkv_b'].copy_(
            torch.stack([
                torch.cat([
                    vl_blocks[i].attn1.to_q.bias, vl_blocks[i].attn1.to_k.bias,
                    vl_blocks[i].attn1.to_v.bias
                ],
                          dim=0) for i in range(vl_n)
            ]))
        self.weights['vl_sa_attn_out_w'].copy_(
            torch.stack(
                [vl_blocks[i].attn1.to_out[0].weight for i in range(vl_n)]))
        self.weights['vl_sa_attn_out_b'].copy_(
            torch.stack(
                [vl_blocks[i].attn1.to_out[0].bias for i in range(vl_n)]))
        self.weights['vl_sa_norm3_w'].copy_(
            torch.stack([vl_blocks[i].norm3.weight for i in range(vl_n)]))
        self.weights['vl_sa_norm3_b'].copy_(
            torch.stack([vl_blocks[i].norm3.bias for i in range(vl_n)]))
        self.weights['vl_sa_ff_up_w_T'].copy_(
            torch.stack([
                vl_blocks[i].ff.net[0].proj.weight for i in range(vl_n)
            ]).permute(0, 2, 1))
        self.weights['vl_sa_ff_up_b'].copy_(
            torch.stack(
                [vl_blocks[i].ff.net[0].proj.bias for i in range(vl_n)]))
        self.weights['vl_sa_ff_down_w'].copy_(
            torch.stack([vl_blocks[i].ff.net[2].weight for i in range(vl_n)]))
        self.weights['vl_sa_ff_down_b'].copy_(
            torch.stack([vl_blocks[i].ff.net[2].bias for i in range(vl_n)]))
        # Position embedding
        self.weights['position_embedding_w'].copy_(
            self.position_embedding.weight)

        self.buffers = {
            'input_features':
            torch.empty(
                1,
                self.max_input_seq_len,
                self.backbone_embedding_dim,
                dtype=torch.bfloat16,
                device='cuda'),
            'states':
            torch.empty(
                1, self.state_dim, dtype=torch.bfloat16, device='cuda'),
            'state_features':
            torch.empty(
                1,
                1,
                self.input_embedding_dim,
                dtype=torch.bfloat16,
                device='cuda'),
            'actions':
            torch.empty(
                1,
                self.num_steps,
                self.action_dim,
                dtype=torch.bfloat16,
                device='cuda'),
            'embodiment_ids':
            torch.zeros(1, dtype=torch.long, device='cuda'),
        }

        self.graph = torch.cuda.CUDAGraph()
        self.record_graph()

    def record_graph(self):
        for i in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.graph.capture_begin()
            self.record_run()
            self.graph.capture_end()

    def record_run(self):
        """Core denoising computation for CUDA Graph capture."""
        input_features = self.buffers['input_features']
        embodiment_ids = self.buffers['embodiment_ids']

        # VLLN (LayerNorm)
        input_features = F.layer_norm(input_features,
                                      [input_features.shape[-1]],
                                      self.weights['vlln_w'],
                                      self.weights['vlln_b'])

        # VL self-attention blocks
        for i in range(self.vl_nl):
            input_features = vl_sa_block(
                input_features,
                self.weights['vl_sa_norm1_w'][i],
                self.weights['vl_sa_norm1_b'][i],
                self.weights['vl_sa_qkv_w'][i],
                self.weights['vl_sa_qkv_b'][i],
                self.weights['vl_sa_attn_out_w'][i],
                self.weights['vl_sa_attn_out_b'][i],
                self.weights['vl_sa_norm3_w'][i],
                self.weights['vl_sa_norm3_b'][i],
                self.weights['vl_sa_ff_up_w_T'][i],
                self.weights['vl_sa_ff_up_b'][i],
                self.weights['vl_sa_ff_down_w'][i],
                self.weights['vl_sa_ff_down_b'][i],
                self.vl_nh,
                self.vl_hd,
                ff_features=self.vl_dim,
                ff_hidden=self.vl_ff)
        self.buffers['input_features'].copy_(input_features)

        # # State encoder
        self.buffers['state_features'].copy_(
            _cat_mlp(self.buffers['states'].unsqueeze(1),
                     self.weights['state_encoder_layer1_W'],
                     self.weights['state_encoder_layer1_b'],
                     self.weights['state_encoder_layer2_W'],
                     self.weights['state_encoder_layer2_b'], embodiment_ids))

        dt = 1.0 / self.num_inference_timesteps
        inner_dim = self.dit_dim
        actions = self.buffers['actions']
        device = actions.device

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(1, ), fill_value=t_discretized, device=device)

            # Action encoder
            action_features = _action_encode(
                actions, timesteps_tensor, self.weights['action_encoder_W1_W'],
                self.weights['action_encoder_W1_b'],
                self.weights['action_encoder_W2_W'],
                self.weights['action_encoder_W2_b'],
                self.weights['action_encoder_W3_W'],
                self.weights['action_encoder_W3_b'], embodiment_ids,
                self.input_embedding_dim)

            # Position embedding
            if self.add_positional_embeddings:
                action_features = fused_position_embedding_add_inplace(
                    action_features, self.weights['position_embedding_w'])

            # Prepare DiT input
            future_tok = self.weights['future_tokens_w'].unsqueeze(0)
            sa_embs = torch.cat(
                (self.buffers['state_features'], future_tok, action_features),
                dim=1)

            # DiT timestep encoding
            temb = _timestep_embedding(timesteps_tensor).to(
                dtype=actions.dtype)
            temb = F.silu(
                F.linear(temb, self.weights['dit_timestep_linear1_w'],
                         self.weights['dit_timestep_linear1_b']))
            temb = F.linear(temb, self.weights['dit_timestep_linear2_w'],
                            self.weights['dit_timestep_linear2_b'])

            # DiT transformer blocks
            hidden_states = sa_embs
            cross_idx, self_idx = 0, 0
            for i in range(self.dit_nl):
                if i % 2 == 1:
                    hidden_states = dit_block_self(
                        hidden_states,
                        temb,
                        self.weights['dit_norm1_linear_w'][i],
                        self.weights['dit_norm1_linear_b'][i],
                        self.weights['dit_self_qkv_w'][self_idx],
                        self.weights['dit_self_qkv_b'][self_idx],
                        self.weights['dit_attn_out_w'][i],
                        self.weights['dit_attn_out_b'][i],
                        self.weights['dit_ff_up_w_T'][i],
                        self.weights['dit_ff_up_b'][i],
                        self.weights['dit_ff_down_w'][i],
                        self.weights['dit_ff_down_b'][i],
                        self.dit_nh,
                        self.dit_hd,
                        inner_dim,
                        ff_features=inner_dim,
                        ff_hidden=self.dit_ff)
                    self_idx += 1
                else:
                    hidden_states = dit_block_cross(
                        hidden_states,
                        input_features,
                        temb,
                        self.weights['dit_norm1_linear_w'][i],
                        self.weights['dit_norm1_linear_b'][i],
                        self.weights['dit_cross_q_w'][cross_idx],
                        self.weights['dit_cross_q_b'][cross_idx],
                        self.weights['dit_cross_kv_w'][cross_idx],
                        self.weights['dit_cross_kv_b'][cross_idx],
                        self.weights['dit_attn_out_w'][i],
                        self.weights['dit_attn_out_b'][i],
                        self.weights['dit_ff_up_w_T'][i],
                        self.weights['dit_ff_up_b'][i],
                        self.weights['dit_ff_down_w'][i],
                        self.weights['dit_ff_down_b'][i],
                        self.dit_nh,
                        self.dit_hd,
                        inner_dim,
                        ff_features=inner_dim,
                        ff_hidden=self.dit_ff)
                    cross_idx += 1

            # DiT output
            shift, scale = F.linear(
                F.silu(temb), self.weights['dit_proj_out_1_w'],
                self.weights['dit_proj_out_1_b']).chunk(
                    2, dim=1)
            hidden_states = (
                F.layer_norm(hidden_states, [inner_dim]) *
                (1 + scale[:, None]) + shift[:, None])
            model_output = F.linear(hidden_states,
                                    self.weights['dit_proj_out_2_w'],
                                    self.weights['dit_proj_out_2_b'])

            # Action decoder
            pred = _cat_mlp(model_output,
                            self.weights['action_decoder_layer1_W'],
                            self.weights['action_decoder_layer1_b'],
                            self.weights['action_decoder_layer2_W'],
                            self.weights['action_decoder_layer2_b'],
                            embodiment_ids)

            pred_velocity = pred[:, -self.num_steps:]
            actions = actions + dt * pred_velocity

        self.buffers['actions'].copy_(actions)

    def predict_action(self, input_features: torch.Tensor,
                       states: torch.Tensor, attention_mask: torch.Tensor,
                       embodiment_ids: torch.Tensor, *args, **kwargs):
        if not self.loaded_weights:
            self._load_weights_and_buffer()
            self.loaded_weights = True

        self.buffers['input_features'].copy_(input_features)
        self.buffers['states'].copy_(states)
        self.buffers['embodiment_ids'].copy_(embodiment_ids)
        self.buffers['actions'].copy_(
            torch.randn(
                size=(1, self.num_steps, self.action_dim),
                dtype=input_features.dtype,
                device=input_features.device))

        self.graph.replay()

        actions = self.buffers['actions']
        if self.ori_action_dim is not None:
            actions = actions[:, :, :self.ori_action_dim]
        return actions
