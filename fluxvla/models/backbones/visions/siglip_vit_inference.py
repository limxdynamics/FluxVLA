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
            patch_w.permute(2, 3, 1, 0).contiguous().bfloat16())
        if embed.patch_embedding.bias is not None:
            weights['vision_patch_embedding_b'] = (
                embed.patch_embedding.bias.data.bfloat16())

        # Position Embedding
        weights['vision_position_embedding'] = (
            embed.position_embedding.weight.data.bfloat16())

        # Transformer layers. Preallocate final tensors instead of keeping
        # per-layer lists and stacking them, which lowers prepare-time peak
        # memory and matches the CPU-prepare path used by ConditionGemma.
        vision_attn_qkv_w = None
        vision_attn_qkv_b = None
        vision_attn_o_w = None
        vision_attn_o_b = None
        vision_ffn_up_w = None
        vision_ffn_up_b = None
        vision_ffn_down_w = None
        vision_ffn_down_b = None
        vision_pre_attn_norm_w = None
        vision_pre_attn_norm_b = None
        vision_pre_ffn_norm_w = None
        vision_pre_ffn_norm_b = None
        num_layers = len(encoder.layers)

        for i, layer in enumerate(encoder.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            q_w = attn.q_proj.weight.data  # [1152, 1152]
            k_w = attn.k_proj.weight.data
            v_w = attn.v_proj.weight.data
            qkv_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=1).bfloat16()

            q_b = (
                attn.q_proj.bias.data
                if attn.q_proj.bias is not None else torch.zeros(1152))
            k_b = (
                attn.k_proj.bias.data
                if attn.k_proj.bias is not None else torch.zeros(1152))
            v_b = (
                attn.v_proj.bias.data
                if attn.v_proj.bias is not None else torch.zeros(1152))
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0).bfloat16()

            attn_o_w = attn.out_proj.weight.data.T.bfloat16()
            attn_o_b = (attn.out_proj.bias.data if attn.out_proj.bias
                        is not None else torch.zeros(1152)).bfloat16()
            ffn_up_w = mlp.fc1.weight.data.T.bfloat16()
            ffn_up_b = mlp.fc1.bias.data.bfloat16()
            ffn_down_w = mlp.fc2.weight.data.T.bfloat16()
            ffn_down_b = mlp.fc2.bias.data.bfloat16()
            pre_attn_norm_w = layer.layer_norm1.weight.data.bfloat16()
            pre_attn_norm_b = layer.layer_norm1.bias.data.bfloat16()
            pre_ffn_norm_w = layer.layer_norm2.weight.data.bfloat16()
            pre_ffn_norm_b = layer.layer_norm2.bias.data.bfloat16()

            if vision_attn_qkv_w is None:
                vision_attn_qkv_w = torch.empty((num_layers, *qkv_w.shape),
                                                dtype=torch.bfloat16)
                vision_attn_qkv_b = torch.empty((num_layers, *qkv_b.shape),
                                                dtype=torch.bfloat16)
                vision_attn_o_w = torch.empty((num_layers, *attn_o_w.shape),
                                              dtype=torch.bfloat16)
                vision_attn_o_b = torch.empty((num_layers, *attn_o_b.shape),
                                              dtype=torch.bfloat16)
                vision_ffn_up_w = torch.empty((num_layers, *ffn_up_w.shape),
                                              dtype=torch.bfloat16)
                vision_ffn_up_b = torch.empty((num_layers, *ffn_up_b.shape),
                                              dtype=torch.bfloat16)
                vision_ffn_down_w = torch.empty(
                    (num_layers, *ffn_down_w.shape), dtype=torch.bfloat16)
                vision_ffn_down_b = torch.empty(
                    (num_layers, *ffn_down_b.shape), dtype=torch.bfloat16)
                vision_pre_attn_norm_w = torch.empty(
                    (num_layers, *pre_attn_norm_w.shape), dtype=torch.bfloat16)
                vision_pre_attn_norm_b = torch.empty(
                    (num_layers, *pre_attn_norm_b.shape), dtype=torch.bfloat16)
                vision_pre_ffn_norm_w = torch.empty(
                    (num_layers, *pre_ffn_norm_w.shape), dtype=torch.bfloat16)
                vision_pre_ffn_norm_b = torch.empty(
                    (num_layers, *pre_ffn_norm_b.shape), dtype=torch.bfloat16)

            vision_attn_qkv_w[i].copy_(qkv_w)
            vision_attn_qkv_b[i].copy_(qkv_b)
            vision_attn_o_w[i].copy_(attn_o_w)
            vision_attn_o_b[i].copy_(attn_o_b)
            vision_ffn_up_w[i].copy_(ffn_up_w)
            vision_ffn_up_b[i].copy_(ffn_up_b)
            vision_ffn_down_w[i].copy_(ffn_down_w)
            vision_ffn_down_b[i].copy_(ffn_down_b)
            vision_pre_attn_norm_w[i].copy_(pre_attn_norm_w)
            vision_pre_attn_norm_b[i].copy_(pre_attn_norm_b)
            vision_pre_ffn_norm_w[i].copy_(pre_ffn_norm_w)
            vision_pre_ffn_norm_b[i].copy_(pre_ffn_norm_b)

        weights['vision_attn_qkv_w'] = vision_attn_qkv_w
        weights['vision_attn_qkv_b'] = vision_attn_qkv_b
        weights['vision_attn_o_w'] = vision_attn_o_w
        weights['vision_attn_o_b'] = vision_attn_o_b
        weights['vision_ffn_up_w'] = vision_ffn_up_w
        weights['vision_ffn_up_b'] = vision_ffn_up_b
        weights['vision_ffn_down_w'] = vision_ffn_down_w
        weights['vision_ffn_down_b'] = vision_ffn_down_b
        weights['vision_pre_attn_norm_w'] = vision_pre_attn_norm_w
        weights['vision_pre_attn_norm_b'] = vision_pre_attn_norm_b
        weights['vision_pre_ffn_norm_w'] = vision_pre_ffn_norm_w
        weights['vision_pre_ffn_norm_b'] = vision_pre_ffn_norm_b

        # Vision final norm
        assert hasattr(
            vm, 'post_layernorm'), 'SigLIP model must have post_layernorm'
        weights['vision_final_norm_w'] = (
            vm.post_layernorm.weight.data.bfloat16())
        weights['vision_final_norm_b'] = (
            vm.post_layernorm.bias.data.bfloat16())
        return weights
