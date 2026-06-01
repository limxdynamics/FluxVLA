import math

import torch

from fluxvla.engines import VLAS
# yapf: disable
from fluxvla.ops.atomic_ops import (AttnMultiKey,
                                    adarms_matmul_k_1024_32_bias_res_rowwise,
                                    adarms_norm_mod_proj_rowwise,
                                    conv2d_embed_res, layer_norm_matmul_bias,
                                    layer_norm_matmul_bias_gelu,
                                    layer_norm_QKV_matmul_bias, matmul_attn_v,
                                    matmul_bias_res, matmul_bias_small,
                                    matmul_k_1024_2560_qkv_rope, matmul_res,
                                    matmul_res_gate, matmul_small_gate,
                                    matmul_split_k_bias_res, rms_matmul_gate,
                                    rms_matmul_qkv_rope)
from fluxvla.ops.triton.attention_triton_ops import (
    matmul_abT_scale, softmax_kernel_masklen, softmax_kernel_prefix_suffix)
# yapf: enable
from .pi05_flowmatching import PI05FlowMatching


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _posemb_sincos_torch(t: torch.Tensor,
                         embedding_dim: int = 1024,
                         min_period: float = 4e-3,
                         max_period: float = 4.0) -> torch.Tensor:
    fraction = torch.linspace(
        0.0, 1.0, embedding_dim // 2, dtype=torch.float32, device=t.device)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = t[:, None] * (1.0 / period)[None, :] * (2.0 * math.pi)
    return torch.cat([torch.sin(sinusoid_input),
                      torch.cos(sinusoid_input)],
                     dim=-1)


def _apply_prefill_mask(buffers) -> None:
    buffers['diffusion_noise'].mul_(buffers['prefill_inv_mask'])
    buffers['diffusion_noise'].add_(buffers['prefill_actions'] *
                                    buffers['prefill_mask'])


def vision_encoder(weights, buffers, num_views, num_vit_layers=27):
    num_patches = 256
    vit_hidden = 1152
    vit_intermediate = 4304
    vit_num_heads = 16
    vit_head_dim = 72
    vit_qkv_hidden = 3 * vit_hidden
    grid_size = 16
    patch_size = 14

    conv2d_embed_res(buffers['observation_images_normalized'],
                     weights['vision_patch_embedding_w'],
                     weights['vision_patch_embedding_b'],
                     weights['vision_position_embedding'], buffers['vision_x'],
                     grid_size, patch_size, num_patches, vit_hidden)

    for i in range(num_vit_layers):
        layer_norm_QKV_matmul_bias(
            buffers['vision_x'], weights['vision_pre_attn_norm_w'][i],
            weights['vision_pre_attn_norm_b'][i],
            weights['vision_attn_qkv_w'][i], weights['vision_attn_qkv_b'][i],
            buffers['vision_QKV'], buffers['vision_x_norm'], num_patches,
            vit_hidden, vit_qkv_hidden)

        attn = AttnMultiKey(buffers['vision_QKV'], num_patches, vit_num_heads,
                            vit_head_dim, vit_hidden)

        matmul_bias_res(attn, weights['vision_attn_o_w'][i],
                        weights['vision_attn_o_b'][i], buffers['vision_x'],
                        buffers['vision_x'], buffers['vision_x_split_k_buf'],
                        num_patches, vit_hidden)

        layer_norm_matmul_bias_gelu(
            buffers['vision_x'], weights['vision_pre_ffn_norm_w'][i],
            weights['vision_pre_ffn_norm_b'][i], weights['vision_ffn_up_w'][i],
            weights['vision_ffn_up_b'][i], buffers['vision_hidden'],
            buffers['vision_x_norm'], num_patches, vit_hidden,
            vit_intermediate)

        matmul_split_k_bias_res(buffers['vision_hidden'],
                                weights['vision_ffn_down_w'][i],
                                weights['vision_ffn_down_b'][i],
                                buffers['vision_x'], buffers['vision_x'],
                                buffers['vision_x_split_k_buf'], num_patches,
                                vit_intermediate, vit_hidden)


def transformer_encoder(weights,
                        buffers,
                        encoder_seq_len,
                        num_encoder_layers=18):
    layer_norm_matmul_bias(
        buffers['vision_x'],
        weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'],
        buffers['encoder_x'],
        buffers['vision_x_norm'],
        num_patches=256,
        in_features=1152,
        out_features=2048,
        eps=1e-5)

    for i in range(num_encoder_layers):
        rms_matmul_qkv_rope(
            buffers['encoder_x'],
            weights['encoder_attn_qkv_w'][i],
            buffers['encoder_rope_weights'],
            buffers['encoder_Q'],
            buffers['encoder_K'][i, :encoder_seq_len],
            buffers['encoder_V'][i, :encoder_seq_len],
            buffers['encoder_x_norm'],
            hidden_dim=2048,
            head_dim=256,
            num_kv_heads=8)

        if i != num_encoder_layers - 1:
            scale = 1.0 / (256**0.5)
            total_queries = buffers['encoder_Q'].shape[0]
            total_keys = encoder_seq_len
            matmul_abT_scale[(((total_queries + 31) // 32) *
                              ((total_keys + 31) // 32), )](
                                  buffers['encoder_Q'],
                                  buffers['encoder_K'][i, :encoder_seq_len],
                                  buffers['encoder_logits_buf'],
                                  total_queries,
                                  total_keys,
                                  256,
                                  scale,
                                  BLOCK_SIZE_M=32,
                                  BLOCK_SIZE_N=32,
                                  BLOCK_SIZE_K=64)

            softmax_kernel_masklen[((total_queries + 3) // 4, )](
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                buffers['valid_encoder_len'],
                buffers['encoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024)

            matmul_attn_v(
                buffers['encoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_ctx_buf'],
                head_dim=256)

            matmul_res(
                buffers['encoder_ctx_buf'].view(-1, 2048),
                weights['encoder_attn_o_w'][i],
                buffers['encoder_x'],
                in_features=2048,
                out_features=2048)

            rms_matmul_gate(
                buffers['encoder_x'],
                weights['encoder_ffn_gate_w'][i],
                weights['encoder_ffn_up_w'][i],
                buffers['encoder_hidden'],
                buffers['encoder_x_norm'],
                hidden_dim=2048,
                intermediate_dim=16384)

            matmul_res(
                buffers['encoder_hidden'],
                weights['encoder_ffn_down_w'][i],
                buffers['encoder_x'],
                in_features=16384,
                out_features=2048)


def transformer_decoder_rtc(weights,
                            buffers,
                            encoder_seq_len,
                            num_decoder_layers=18,
                            num_steps=10):
    for step in range(num_steps):
        matmul_bias_small(
            buffers['diffusion_noise'],
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x'],
            in_features=32,
            out_features=1024,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_K=32,
        )
        seq_len = buffers['decoder_x'].shape[0]
        for i in range(num_decoder_layers):
            adarms_norm_mod_proj_rowwise(
                buffers['decoder_x'],
                buffers['decoder_adarms_mod_attn'][step, i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
            )
            matmul_k_1024_2560_qkv_rope(
                buffers['x_normed_buf'],
                weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len +
                                     seq_len],
                buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len +
                                     seq_len],
            )
            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale[(((total_queries + 31) // 32) *
                              ((total_keys + 31) // 32), )](
                                  buffers['decoder_q_buf'],
                                  buffers['encoder_K'][i, :encoder_seq_len +
                                                       seq_len],
                                  buffers['decoder_logits_buf'],
                                  total_queries,
                                  total_keys,
                                  256,
                                  256**-0.5,
                                  BLOCK_SIZE_M=32,
                                  BLOCK_SIZE_N=32,
                                  BLOCK_SIZE_K=64)
            softmax_kernel_prefix_suffix[((total_queries + 3) // 4, )](
                buffers['decoder_logits_buf'],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )
            matmul_attn_v(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                buffers['decoder_q_buf'],
                head_dim=256)
            matmul_res_gate(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf'],
                in_features=2048,
                out_features=1024)
            adarms_norm_mod_proj_rowwise(
                buffers['decoder_x'],
                buffers['decoder_adarms_mod_ffn'][step, i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
            )
            seq_len = buffers['decoder_x'].shape[0]
            matmul_small_gate[((seq_len + 127) // 128, (4096 + 63) // 64)](
                buffers['x_normed_buf'],
                weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i],
                buffers['decoder_hidden'],
                seq_len,
                1024,
                4096,
            )
            matmul_res_gate(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf'],
                in_features=4096,
                out_features=1024,
                BLOCK_SIZE_N=16,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_K=256)

        adarms_matmul_k_1024_32_bias_res_rowwise(
            buffers['decoder_x'],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_adarms_mod_final'][step],
            weights['decoder_action_out_proj_w'],
            weights['decoder_action_out_proj_b'],
            buffers['diffusion_noise'],
            buffers['diffusion_noise'],
        )
        _apply_prefill_mask(buffers)


def pi05rtc_model(weights,
                  buffers,
                  num_views,
                  encoder_seq_len,
                  num_vit_layers=27,
                  num_encoder_layers=18,
                  num_decoder_layers=18,
                  num_steps=10):
    vision_encoder(weights, buffers, num_views, num_vit_layers)
    transformer_encoder(weights, buffers, encoder_seq_len, num_encoder_layers)
    transformer_decoder_rtc(weights, buffers, encoder_seq_len,
                            num_decoder_layers, num_steps)


@VLAS.register_module()
class PI05FlowMatchingRTCInference(PI05FlowMatching):
    """Inference variant of PI05FlowMatching with Triton acceleration.

    Combines vision_encoder + transformer_encoder + transformer_decoder into
    ONE CUDA Graph, eliminating inter-graph overhead.

    Args:
        num_views (int): Number of camera views. Default: 2.
        *args: Forwarded to :class:`PI05FlowMatching`.
        **kwargs: Forwarded to :class:`PI05FlowMatching`.
    """

    def __init__(self,
                 num_views=3,
                 triton_max_prompt_len=48,
                 num_steps=10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_views = num_views
        self.triton_max_prompt_len = triton_max_prompt_len
        self.num_steps = num_steps
        self._triton_ready = False
        self._cuda_graph = None
        self._cuda_graph_ready = False

    def _init_buffers(self):
        nv = self.num_views
        enc = self._encoder_seq_len
        dec = self._decoder_seq_len
        num_kv_layers = max(self._num_encoder_layers, self._num_decoder_layers)
        bf = torch.bfloat16
        dev = 'cuda'

        img = self._vit_image_size
        np_ = self._vit_num_patches
        vh = self._vit_hidden
        vi = self._vit_intermediate
        eh = self._enc_hidden
        ei = self._enc_intermediate
        dh = self._dec_hidden
        di = self._dec_intermediate
        hd = self._head_dim
        nkv = self._num_kv_heads
        ad = self._action_dim

        self._triton_bufs = {
            'observation_images_normalized':
            torch.zeros(nv, img, img, 3, dtype=bf, device=dev),
            'vision_x':
            torch.zeros(nv, np_, vh, dtype=bf, device=dev),
            'vision_x_norm':
            torch.zeros(nv, np_, vh, dtype=bf, device=dev),
            'vision_QKV':
            torch.zeros(nv, np_, 3 * vh, dtype=bf, device=dev),
            'vision_hidden':
            torch.zeros(nv, np_, vi, dtype=bf, device=dev),
            'vision_x_split_k_buf':
            torch.zeros(nv * np_ * vh * 4, dtype=torch.float32, device=dev),
            'encoder_rope_weights':
            torch.zeros(enc, hd, dtype=bf, device=dev),
            'encoder_x':
            torch.zeros(enc, eh, dtype=bf, device=dev),
            'encoder_x_norm':
            torch.zeros(enc, eh, dtype=bf, device=dev),
            'encoder_K':
            torch.zeros(num_kv_layers, enc + dec, hd, dtype=bf, device=dev),
            'encoder_V':
            torch.zeros(num_kv_layers, enc + dec, hd, dtype=bf, device=dev),
            'encoder_Q':
            torch.zeros(enc * nkv, hd, dtype=bf, device=dev),
            'encoder_hidden':
            torch.zeros(enc, ei, dtype=bf, device=dev),
            'valid_encoder_len':
            torch.zeros((1, ), dtype=torch.int32, device=dev),
            'encoder_logits_buf':
            torch.zeros(enc * nkv, enc, dtype=torch.float32, device=dev),
            'encoder_attn_buf':
            torch.zeros(enc * nkv, enc, dtype=bf, device=dev),
            'encoder_ctx_buf':
            torch.zeros(enc * nkv, hd, dtype=bf, device=dev),
            'decoder_rope_weights':
            torch.zeros(dec, hd, dtype=bf, device=dev),
            'decoder_x':
            torch.zeros(dec, dh, dtype=bf, device=dev),
            'decoder_x_buf':
            torch.zeros(dec, dh, dtype=bf, device=dev),
            'decoder_action_buf':
            torch.zeros(dec, ad, dtype=bf, device=dev),
            'decoder_time_emb':
            torch.zeros(dec, dh, dtype=bf, device=dev),
            'decoder_adarms_mod_attn':
            torch.empty(
                (self._num_steps, self._num_decoder_layers, dec, dh * 3),
                dtype=bf,
                device=dev),
            'decoder_adarms_mod_ffn':
            torch.empty(
                (self._num_steps, self._num_decoder_layers, dec, dh * 3),
                dtype=bf,
                device=dev),
            'decoder_adarms_mod_final':
            torch.empty((self._num_steps, dec, dh * 3), dtype=bf, device=dev),
            # 'decoder_style':
            # torch.zeros(dec, ds, dtype=bf, device=dev),
            # 'decoder_norm_factor_buf':
            # torch.zeros(dec, dtype=bf, device=dev),
            'decoder_q_buf':
            torch.zeros(dec * nkv, hd, dtype=bf, device=dev),
            'decoder_logits_buf':
            torch.zeros(dec * nkv, enc + dec, dtype=torch.float32, device=dev),
            'decoder_attn_buf':
            torch.zeros(dec * nkv, enc + dec, dtype=bf, device=dev),
            'decoder_hidden':
            torch.zeros(dec, di, dtype=bf, device=dev),
            # 'decode_split_k_buf':
            # torch.zeros(2, dec, dh, dtype=torch.float32, device=dev),
            'x_normed_buf':
            torch.zeros(dec, dh, dtype=bf, device=dev),
            'gate_buf':
            torch.zeros(dec, dh, dtype=bf, device=dev),
            'prefill_actions':
            torch.zeros((dec, ad), dtype=bf, device=dev),
            'prefill_mask':
            torch.zeros((dec, 1), dtype=bf, device=dev),
            'prefill_inv_mask':
            torch.ones((dec, 1), dtype=bf, device=dev),
            'diffusion_noise':
            torch.zeros(dec, ad, dtype=bf, device=dev),
        }

    def _init_rope_table(self):
        prefix_alloc = self.num_views * 256 + self._max_prompt_len
        max_pos = prefix_alloc - 1 + self._decoder_seq_len
        position_ids = torch.arange(max_pos + 1, device='cuda')
        inv_freq = 1.0 / (10000**(
            torch.arange(0, 256, 2, dtype=torch.float32, device='cuda') / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]],
                                     2).view(-1, 256)
        self._triton_bufs['encoder_rope_weights'].copy_(
            self._rope_table[:prefix_alloc])

    def _get_decoder_rope_weights(self, prompt_len):
        start = self.num_views * 256 + prompt_len
        end = start + self._decoder_seq_len
        return self._rope_table[start:end]

    def _build_adarms_mod_bases(self):
        '''
        precompute the time embeddings for each step
        '''
        t_embeds = []
        for step in range(self._num_steps):
            time_embed = self._triton_weights['decoder_time_embeds'][
                step].float()
            time_embed = _swish(
                torch.matmul(
                    time_embed,
                    self._triton_weights['decoder_time_mlp_in_w'].float()) +
                self._triton_weights['decoder_time_mlp_in_b'].float())
            time_embed = _swish(
                torch.matmul(
                    time_embed,
                    self._triton_weights['decoder_time_mlp_out_w'].float()) +
                self._triton_weights['decoder_time_mlp_out_b'].float())
            t_embeds.append(time_embed)
        self._time_step_emb = torch.stack(t_embeds, dim=0).to(torch.float32)

        zero_embed = _posemb_sincos_torch(
            torch.zeros(1, dtype=torch.float32, device='cuda'))[0]
        zero_embed = _swish(
            torch.matmul(zero_embed, self.
                         _triton_weights['decoder_time_mlp_in_w'].float()) +
            self._triton_weights['decoder_time_mlp_in_b'].float())
        zero_embed = _swish(
            torch.matmul(
                zero_embed,
                self._triton_weights['decoder_time_mlp_out_w'].float()) +
            self._triton_weights['decoder_time_mlp_out_b'].float())
        self._time_zero_emb = zero_embed.to(torch.float32)

        attn_w = self._triton_weights['decoder_pre_attn_norm_mod_w'].float()
        attn_b = self._triton_weights['decoder_pre_attn_norm_mod_b'].float()
        ffn_w = self._triton_weights['decoder_pre_ffn_norm_mod_w'].float()
        ffn_b = self._triton_weights['decoder_pre_ffn_norm_mod_b'].float()
        final_w = self._triton_weights['decoder_final_norm_mod_w'].float()
        final_b = self._triton_weights['decoder_final_norm_mod_b'].float()

        self._base_adarms_mod_attn_vec = (
            torch.einsum('sd,ldh->slh', self._time_step_emb, attn_w) +
            attn_b[None, :, :]).to(torch.bfloat16)
        self._base_adarms_mod_ffn_vec = (
            torch.einsum('sd,ldh->slh', self._time_step_emb, ffn_w) +
            ffn_b[None, :, :]).to(torch.bfloat16)
        self._base_adarms_mod_final_vec = (
            torch.matmul(self._time_step_emb, final_w) + final_b[None, :]).to(
                torch.bfloat16)

        self._base_adarms_mod_attn_t0 = (
            torch.einsum('d,ldh->lh', self._time_zero_emb, attn_w) +
            attn_b).to(torch.bfloat16)
        self._base_adarms_mod_ffn_t0 = (
            torch.einsum('d,ldh->lh', self._time_zero_emb, ffn_w) + ffn_b).to(
                torch.bfloat16)
        self._base_adarms_mod_final_t0 = (
            torch.matmul(self._time_zero_emb, final_w) + final_b).to(
                torch.bfloat16)

    def _update_runtime_adarms_mods(self, prefill_len: int):
        '''
        update the runtime adarms mods to timestep=0
        '''
        self._triton_bufs['decoder_adarms_mod_attn'][:] = (
            self._base_adarms_mod_attn_vec[:, :, None, :])
        self._triton_bufs['decoder_adarms_mod_ffn'][:] = (
            self._base_adarms_mod_ffn_vec[:, :, None, :])
        self._triton_bufs['decoder_adarms_mod_final'][:] = (
            self._base_adarms_mod_final_vec[:, None, :])

        if prefill_len > 0:
            self._triton_bufs[
                'decoder_adarms_mod_attn'][:, :, :prefill_len, :] = (
                    self._base_adarms_mod_attn_t0[None, :, None, :])
            self._triton_bufs[
                'decoder_adarms_mod_ffn'][:, :, :prefill_len, :] = (
                    self._base_adarms_mod_ffn_t0[None, :, None, :])
            self._triton_bufs[
                'decoder_adarms_mod_final'][:, :prefill_len, :] = (
                    self._base_adarms_mod_final_t0[None, None, :])

    def _prepare_prefill(self, diffusion_noise, prefill_actions,
                         action_prefill_len):
        prefill_len = int(action_prefill_len or 0)
        prefill_len = max(0, min(prefill_len, self._decoder_seq_len))

        self._triton_bufs['prefill_actions'].zero_()
        self._triton_bufs['prefill_mask'].zero_()
        self._triton_bufs['prefill_inv_mask'].fill_(1)
        if prefill_len == 0:
            return 0
        if prefill_actions is None:
            raise ValueError('prefix_len > 0 requires prev_actions.')

        pref = torch.as_tensor(
            prefill_actions, device='cuda', dtype=torch.bfloat16)
        if pref.ndim == 3:
            pref = pref[0]
        if pref.shape[-1] < self._action_dim:
            pad = torch.zeros(
                pref.shape[0],
                self._action_dim - pref.shape[-1],
                dtype=torch.bfloat16,
                device=pref.device)
            pref = torch.cat([pref, pad], dim=-1)

        copy_len = min(prefill_len, int(pref.shape[0]), self._decoder_seq_len)
        self._triton_bufs['prefill_actions'][:copy_len].copy_(pref[:copy_len])
        self._triton_bufs['prefill_mask'][:copy_len].fill_(1)
        self._triton_bufs['prefill_inv_mask'][:copy_len].fill_(0)
        diffusion_noise[:copy_len].copy_(
            self._triton_bufs['prefill_actions'][:copy_len])
        return copy_len

    def _run_forward(self):
        pi05rtc_model(self._triton_weights, self._triton_bufs, self.num_views,
                      self._encoder_seq_len, self._num_vit_layers,
                      self._num_encoder_layers, self._num_decoder_layers,
                      self._num_steps)

    def _build_cuda_graph(self):
        print('[Triton Inference] Recording CUDA Graph ...')
        for _ in range(3):
            self._run_forward()
        torch.cuda.synchronize()

        self._cuda_graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._cuda_graph.capture_begin()
            self._run_forward()
            self._cuda_graph.capture_end()
        torch.cuda.synchronize()

        self._cuda_graph_ready = True
        print('[Triton Inference] CUDA Graph recorded successfully!')

    def _triton_forward(self,
                        images_nhwc,
                        prompt_embeds,
                        prompt_len,
                        diffusion_noise,
                        prev_actions=None,
                        prefix_len: int = 0):
        """Run the full unified Triton inference pipeline.

        Args:
            images_nhwc: images in [num_views, H, W, C] bfloat16 format.
            prompt_embeds: language embeddings [prompt_len, 2048] bfloat16.
            prompt_len: actual prompt token count (int).
            diffusion_noise: initial noise [chunk_size, 32] bfloat16.

        Returns:
            Denoised actions [chunk_size, 32] bfloat16.
        """
        self._triton_bufs['observation_images_normalized'].copy_(images_nhwc)
        start = self.num_views * 256
        self._triton_bufs['encoder_x'][start:start +
                                       prompt_len].copy_(prompt_embeds)
        self._triton_bufs['valid_encoder_len'].fill_(start + prompt_len)
        self._triton_bufs['decoder_rope_weights'].copy_(
            self._get_decoder_rope_weights(prompt_len))
        self._triton_bufs['diffusion_noise'].copy_(diffusion_noise)
        prefill_len = self._prepare_prefill(
            self._triton_bufs['diffusion_noise'], prev_actions, prefix_len)
        self._update_runtime_adarms_mods(prefill_len)

        if not self._cuda_graph_ready:
            self._build_cuda_graph()

        self._cuda_graph.replay()
        return self._triton_bufs['diffusion_noise']

    def predict_action(self,
                       images,
                       lang_tokens,
                       states,
                       img_masks=None,
                       lang_masks=None,
                       past_key_values=None,
                       noise=None,
                       prev_actions=None,
                       prefix_len: int = 0,
                       rtc_config: dict = None,
                       *args,
                       **kwargs):
        if not self._triton_ready:
            self.materialize_inference_weights(
                num_views=self.num_views,
                max_prompt_len=self.triton_max_prompt_len,
                chunk_size=self.n_action_steps,
                num_steps=self.num_steps)
            self._triton_ready = True

        pixel_values = images.unflatten(1, (-1, 3))[0]
        images_nhwc = pixel_values.permute(0, 2, 3, 1).contiguous().bfloat16()

        prompt_len = (
            int(lang_masks[0].sum().item())
            if lang_masks is not None else lang_tokens.shape[1])
        embed_device = self.llm_backbone.embed_tokens.weight.device
        prompt_tokens = lang_tokens[0, :prompt_len].to(embed_device)
        lang_emb = self.llm_backbone.embed_tokens(prompt_tokens)
        lang_emb = (lang_emb * math.sqrt(lang_emb.shape[-1])).to(
            device='cuda', dtype=torch.bfloat16)

        chunk_size = self.n_action_steps
        if noise is None:
            noise_t = torch.randn(
                chunk_size,
                self.max_action_dim,
                dtype=torch.bfloat16,
                device=states.device)
        else:
            noise_t = noise[0].to(dtype=torch.bfloat16)
        if noise_t.shape[-1] < self._action_dim:
            pad = torch.zeros(
                noise_t.shape[0],
                self._action_dim - noise_t.shape[-1],
                dtype=torch.bfloat16,
                device=noise_t.device)
            noise_t = torch.cat([noise_t, pad], dim=-1)

        rtc_method = None
        if prev_actions is not None and prefix_len > 0 and rtc_config:
            if rtc_config.get('enabled', True):
                rtc_method = rtc_config.get('method', 'prefix')
        if rtc_method not in (None, 'prefix'):
            raise NotImplementedError(
                'PI05FlowMatchingRTCInference only supports prefix RTC.')

        denoised = self._triton_forward(
            images_nhwc,
            lang_emb,
            prompt_len,
            noise_t,
            prev_actions=prev_actions,
            prefix_len=prefix_len if rtc_method == 'prefix' else 0)
        result = denoised[:, :self.max_action_dim].unsqueeze(0).float()

        return result

    def _prepare_adarms_cond(self, num_steps):
        """Pre-compute sinusoidal time embeddings for each step."""
        dt = -1.0 / num_steps
        time_val = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        min_period = 4e-3
        max_period = 4.0
        embedding_dim = 1024
        fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device='cuda')
        period = min_period * (max_period / min_period)**fraction
        time_embs = []
        for _ in range(num_steps):
            sinusoid_input = (
                time_val.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 *
                math.pi)
            emb = torch.cat(
                [torch.sin(sinusoid_input),
                 torch.cos(sinusoid_input)], dim=-1)
            time_embs.append(emb.to(torch.bfloat16))
            time_val = time_val + dt
        return torch.cat(time_embs, dim=0)

    def _move_triton_weights_to_cuda(self):
        for key, value in list(self._triton_weights.items()):
            if torch.is_tensor(value) and value.device.type != 'cuda':
                self._triton_weights[key] = value.contiguous().cuda()

    def _offload_eager_modules_to_cpu(self):
        for name in ('vision_backbone', 'llm_backbone', 'llm_expert',
                     'projector', 'action_in_proj', 'action_out_proj',
                     'time_mlp_in', 'time_mlp_out'):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, 'cpu'):
                module.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _materialize_action_time_weights(self) -> dict:
        weights = {}
        weights.update(
            self.action_in_proj.materialize_weights('decoder_action_in_proj'))
        weights.update(
            self.action_out_proj.materialize_weights(
                'decoder_action_out_proj'))
        weights.update(
            self.time_mlp_in.materialize_weights('decoder_time_mlp_in'))
        weights.update(
            self.time_mlp_out.materialize_weights('decoder_time_mlp_out'))
        return weights

    def materialize_inference_weights(self, num_views, max_prompt_len,
                                      chunk_size, num_steps):
        """Collect weights and build the Triton pipeline.

        Args:
            num_views (int): The number of views.
            max_prompt_len (int): The maximum prompt length.
            chunk_size (int): The chunk size.
            num_steps (int): Denoising steps.
        """
        self._triton_weights = {}
        self._triton_weights.update(self.vision_backbone.materialize_weights())
        self._triton_weights.update(
            self.llm_backbone.materialize_weights(role='llm'))
        self._triton_weights.update(
            self.llm_expert.materialize_weights(role='expert'))
        self._triton_weights.update(
            self.projector.materialize_weights(
                prefix='encoder_multi_modal_projector'))
        self._triton_weights.update(self._materialize_action_time_weights())
        self._triton_weights.update(
            {'decoder_time_embeds': self._prepare_adarms_cond(num_steps)})
        self._move_triton_weights_to_cuda()
        self._triton_weights['decoder_action_out_proj_w'].mul_(
            -1.0 / float(num_steps))
        self._triton_weights['decoder_action_out_proj_b'].mul_(
            -1.0 / float(num_steps))

        self._max_prompt_len = max_prompt_len
        self._num_steps = num_steps

        # Vision dimensions (from SigLIP config)
        vit_cfg = self.vision_backbone.vision.vision_model.config
        self._vit_image_size = vit_cfg.image_size
        self._vit_num_patches = (vit_cfg.image_size // vit_cfg.patch_size)**2
        self._vit_hidden = vit_cfg.hidden_size
        self._vit_intermediate = vit_cfg.intermediate_size
        self._num_vit_layers = vit_cfg.num_hidden_layers

        # Encoder dimensions (from Gemma backbone config)
        enc_cfg = self.llm_backbone.config
        self._enc_hidden = enc_cfg.hidden_size
        self._enc_intermediate = enc_cfg.intermediate_size
        self._num_encoder_layers = len(self.llm_backbone.layers)

        # Decoder dimensions (from Gemma expert config)
        dec_cfg = self.llm_expert.config
        self._dec_hidden = dec_cfg.hidden_size
        self._dec_intermediate = dec_cfg.intermediate_size
        self._dec_style_dim = 3 * self._dec_hidden
        self._num_decoder_layers = len(self.llm_expert.layers)

        # Shared attention dimensions
        self._head_dim = enc_cfg.head_dim
        self._num_kv_heads = enc_cfg.num_attention_heads
        self._action_dim = self.max_action_dim

        self._encoder_seq_len = (
            num_views * self._vit_num_patches + max_prompt_len)
        self._decoder_seq_len = chunk_size

        self._offload_eager_modules_to_cpu()
        self._init_buffers()
        self._init_rope_table()
        self._build_adarms_mod_bases()
        self._update_runtime_adarms_mods(prefill_len=0)

        self._cuda_graph = None
        self._cuda_graph_ready = False
