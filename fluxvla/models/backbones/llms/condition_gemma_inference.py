import torch

from fluxvla.engines import LLM_BACKBONES
from .condition_gemma import ConditionGemmaModel


@LLM_BACKBONES.register_module()
class ConditionGemmaInferenceModel(ConditionGemmaModel):
    """Inference variant of ConditionGemma with Triton weight prep.

    Extends :class:`ConditionGemmaModel` with :meth:`prepare_triton`,
    which extracts, fuses and reformats model weights for the
    Triton-based CUDA-graph inference pipeline
    (:class:`PI05FlowMatchingInference`).

    Weight pre-processing performed by :meth:`prepare_triton`:

    * **Encoder (role='llm')**: Fuses RMSNorm scale ``(1 + w)``
      into QKV / gate / up projections so that normalization and
      linear projection become a single matmul.  Converts Q/K
      weight layout from ``[heads, head_dim, in]`` to the
      interleaved format required by Triton fused-RoPE kernels.
    * **Decoder (role='expert')**: Collects AdaRMS modulation
      weights (``input_layernorm.dense``,
      ``post_attention_layernorm.dense``) and final-norm
      modulation weights.  Q/K are also converted to fused-RoPE
      layout, but RMSNorm is *not* fused because the decoder
      uses adaptive (input-dependent) normalization.

    All returned tensors are bf16, column-major (transposed),
    contiguous, and on CUDA — ready to be consumed by
    :meth:`PI05FlowMatchingInference.prepare_triton_inference`.

    Args:
        Inherits all arguments from :class:`ConditionGemmaModel`.

    Example::

        encoder = ConditionGemmaInferenceModel(**cfg_enc)
        decoder = ConditionGemmaInferenceModel(**cfg_dec)
        weights = {}
        weights.update(encoder.prepare_triton(role='llm'))
        weights.update(decoder.prepare_triton(role='expert'))
    """

    def _apply_rope_format_conversion(self,
                                      w_q,
                                      w_k,
                                      head_dim=256,
                                      num_heads_q=8,
                                      num_heads_k=1):
        """Convert Q/K weights from training format to Triton RoPE format."""
        in_dim = w_q.shape[1]
        out_dim_q = num_heads_q * head_dim

        w_q = w_q.view(num_heads_q, head_dim, in_dim)
        w_q = w_q.view(num_heads_q, 2, head_dim // 2, in_dim)
        w_q = w_q.permute(0, 2, 1, 3).reshape(out_dim_q, in_dim)

        w_k = w_k.view(2, head_dim // 2, in_dim)
        w_k = w_k.permute(1, 0, 2).reshape(head_dim, in_dim)

        return w_q, w_k

    def prepare_triton(self, role='llm') -> dict:
        weights = {}

        if role == 'llm':
            llm = self.layers
            n = len(self.layers)
            attn_qkv_w, attn_o_w = [], []
            ffn_gate_w, ffn_up_w, ffn_down_w = [], [], []

            for i in range(n):
                layer = llm[i]
                pre_attn_norm = layer.input_layernorm.weight.data.float()
                pre_ffn_norm = (
                    layer.post_attention_layernorm.weight.data.float())

                q_w = layer.self_attn.q_proj.weight.data.float()
                k_w = layer.self_attn.k_proj.weight.data.float()
                v_w = layer.self_attn.v_proj.weight.data.float()
                o_w = layer.self_attn.o_proj.weight.data.float()

                scale = (1 + pre_attn_norm).unsqueeze(0)
                q_w = q_w * scale
                k_w = k_w * scale
                v_w = v_w * scale

                q_w, k_w = self._apply_rope_format_conversion(q_w, k_w)
                qkv_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=1)
                attn_qkv_w.append(qkv_w.bfloat16().cuda())
                attn_o_w.append(o_w.T.contiguous().bfloat16().cuda())

                gate_w = layer.mlp.gate_proj.weight.data.float()
                up_w = layer.mlp.up_proj.weight.data.float()
                down_w = layer.mlp.down_proj.weight.data.float()

                ffn_scale = (1 + pre_ffn_norm).unsqueeze(0)
                gate_w = gate_w * ffn_scale
                up_w = up_w * ffn_scale

                ffn_gate_w.append(gate_w.T.contiguous().bfloat16().cuda())
                ffn_up_w.append(up_w.T.contiguous().bfloat16().cuda())
                ffn_down_w.append(down_w.T.contiguous().bfloat16().cuda())

            weights['encoder_attn_qkv_w'] = torch.stack(attn_qkv_w)
            weights['encoder_attn_o_w'] = torch.stack(attn_o_w)
            weights['encoder_ffn_gate_w'] = torch.stack(ffn_gate_w)
            weights['encoder_ffn_up_w'] = torch.stack(ffn_up_w)
            weights['encoder_ffn_down_w'] = torch.stack(ffn_down_w)

        else:  # expert
            expert = self.layers
            n = len(self.layers)

            attn_qkv_w, attn_o_w = [], []
            ffn_gate_w, ffn_up_w, ffn_down_w = [], [], []
            pre_attn_mod_w, pre_attn_mod_b = [], []
            pre_ffn_mod_w, pre_ffn_mod_b = [], []

            for i in range(n):
                layer = expert[i]

                pre_attn_mod_w.append(layer.input_layernorm.dense.weight.data.
                                      T.contiguous().bfloat16().cuda())
                pre_attn_mod_b.append(
                    layer.input_layernorm.dense.bias.data.bfloat16().cuda())
                pre_ffn_mod_w.append(
                    layer.post_attention_layernorm.dense.weight.data.T.
                    contiguous().bfloat16().cuda())
                pre_ffn_mod_b.append(layer.post_attention_layernorm.dense.bias.
                                     data.bfloat16().cuda())

                q_w = layer.self_attn.q_proj.weight.data.float()
                k_w = layer.self_attn.k_proj.weight.data.float()
                v_w = layer.self_attn.v_proj.weight.data.float()
                o_w = layer.self_attn.o_proj.weight.data.float()

                q_w, k_w = self._apply_rope_format_conversion(q_w, k_w)
                qkv_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=1)
                attn_qkv_w.append(qkv_w.bfloat16().cuda())
                attn_o_w.append(o_w.T.contiguous().bfloat16().cuda())

                ffn_gate_w.append(layer.mlp.gate_proj.weight.data.T.contiguous(
                ).bfloat16().cuda())
                ffn_up_w.append(layer.mlp.up_proj.weight.data.T.contiguous().
                                bfloat16().cuda())
                ffn_down_w.append(layer.mlp.down_proj.weight.data.T.contiguous(
                ).bfloat16().cuda())

            weights['decoder_attn_qkv_w'] = torch.stack(attn_qkv_w)
            weights['decoder_attn_o_w'] = torch.stack(attn_o_w)
            weights['decoder_ffn_gate_w'] = torch.stack(ffn_gate_w)
            weights['decoder_ffn_up_w'] = torch.stack(ffn_up_w)
            weights['decoder_ffn_down_w'] = torch.stack(ffn_down_w)
            weights['decoder_pre_attn_norm_mod_w'] = (
                torch.stack(pre_attn_mod_w))
            weights['decoder_pre_attn_norm_mod_b'] = (
                torch.stack(pre_attn_mod_b))
            weights['decoder_pre_ffn_norm_mod_w'] = (
                torch.stack(pre_ffn_mod_w))
            weights['decoder_pre_ffn_norm_mod_b'] = (
                torch.stack(pre_ffn_mod_b))

            weights['decoder_final_norm_mod_w'] = (
                self.norm.dense.weight.data.T.contiguous().bfloat16().cuda())
            weights['decoder_final_norm_mod_b'] = (
                self.norm.dense.bias.data.bfloat16().cuda())
        return weights
