import torch

from fluxvla.engines import VISION_BACKBONES
from .siglip_vit import SigLIPViTBackbone


@VISION_BACKBONES.register_module()
class SigLIPViTBackboneInference(SigLIPViTBackbone):

    def prepare_triton(self) -> dict:
        vm = self.vision.vision_model
        weights = {}

        embed = vm.embeddings
        encoder = vm.encoder

        # Patch Embedding: [out, in, kH, kW] → [kH, kW, in, out]
        patch_w = embed.patch_embedding.weight.data  # [1152, 3, 14, 14]
        weights['vision_patch_embedding_w'] = (
            patch_w.permute(2, 3, 1, 0).contiguous().bfloat16().cuda())
        if embed.patch_embedding.bias is not None:
            weights['vision_patch_embedding_b'] = (
                embed.patch_embedding.bias.data.bfloat16().cuda())

        # Position Embedding
        weights['vision_position_embedding'] = (
            embed.position_embedding.weight.data.bfloat16().cuda())

        # Transformer layers
        attn_qkv_w, attn_qkv_b = [], []
        attn_o_w, attn_o_b = [], []
        ffn_up_w, ffn_up_b = [], []
        ffn_down_w, ffn_down_b = [], []
        pre_attn_norm_w, pre_attn_norm_b = [], []
        pre_ffn_norm_w, pre_ffn_norm_b = [], []

        for layer in encoder.layers:
            attn = layer.self_attn
            mlp = layer.mlp

            q_w = attn.q_proj.weight.data  # [1152, 1152]
            k_w = attn.k_proj.weight.data
            v_w = attn.v_proj.weight.data
            qkv_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=1)
            attn_qkv_w.append(qkv_w)

            q_b = (
                attn.q_proj.bias.data
                if attn.q_proj.bias is not None else torch.zeros(1152))
            k_b = (
                attn.k_proj.bias.data
                if attn.k_proj.bias is not None else torch.zeros(1152))
            v_b = (
                attn.v_proj.bias.data
                if attn.v_proj.bias is not None else torch.zeros(1152))
            attn_qkv_b.append(torch.cat([q_b, k_b, v_b], dim=0))

            attn_o_w.append(attn.out_proj.weight.data.T)
            attn_o_b.append(
                attn.out_proj.bias.data if attn.out_proj.
                bias is not None else torch.zeros(1152))

            ffn_up_w.append(mlp.fc1.weight.data.T)
            ffn_up_b.append(mlp.fc1.bias.data)
            ffn_down_w.append(mlp.fc2.weight.data.T)
            ffn_down_b.append(mlp.fc2.bias.data)

            pre_attn_norm_w.append(layer.layer_norm1.weight.data)
            pre_attn_norm_b.append(layer.layer_norm1.bias.data)
            pre_ffn_norm_w.append(layer.layer_norm2.weight.data)
            pre_ffn_norm_b.append(layer.layer_norm2.bias.data)

        weights['vision_attn_qkv_w'] = (
            torch.stack(attn_qkv_w).bfloat16().cuda())
        weights['vision_attn_qkv_b'] = (
            torch.stack(attn_qkv_b).bfloat16().cuda())
        weights['vision_attn_o_w'] = (torch.stack(attn_o_w).bfloat16().cuda())
        weights['vision_attn_o_b'] = (torch.stack(attn_o_b).bfloat16().cuda())
        weights['vision_ffn_up_w'] = (torch.stack(ffn_up_w).bfloat16().cuda())
        weights['vision_ffn_up_b'] = (torch.stack(ffn_up_b).bfloat16().cuda())
        weights['vision_ffn_down_w'] = (
            torch.stack(ffn_down_w).bfloat16().cuda())
        weights['vision_ffn_down_b'] = (
            torch.stack(ffn_down_b).bfloat16().cuda())
        weights['vision_pre_attn_norm_w'] = (
            torch.stack(pre_attn_norm_w).bfloat16().cuda())
        weights['vision_pre_attn_norm_b'] = (
            torch.stack(pre_attn_norm_b).bfloat16().cuda())
        weights['vision_pre_ffn_norm_w'] = (
            torch.stack(pre_ffn_norm_w).bfloat16().cuda())
        weights['vision_pre_ffn_norm_b'] = (
            torch.stack(pre_ffn_norm_b).bfloat16().cuda())

        # Vision final norm
        assert hasattr(
            vm, 'post_layernorm'), 'SigLIP model must have post_layernorm'
        weights['vision_final_norm_w'] = (
            vm.post_layernorm.weight.data.bfloat16().cuda())
        weights['vision_final_norm_b'] = (
            vm.post_layernorm.bias.data.bfloat16().cuda())
        return weights
