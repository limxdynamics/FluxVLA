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
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy
from transformers import LlamaConfig, LlamaModel

from fluxvla.engines import HEADS


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden
            features. If None, defaults to None.
        out_features (int, optional): Number of output
            features. If None, defaults to None.
        act_layer (nn.Module, optional): Activation layer.
            Defaults to nn.GELU.
        drop (float, optional): Dropout rate. Defaults to 0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimpleLLamaTransformer(nn.Module):

    def __init__(self, embedding_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        # 配置模型参数（Llama 默认使用 ROPE 和 Causal Attention）
        config = LlamaConfig(
            hidden_size=embedding_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * embedding_dim,  # FFN 中间层维度
        )
        self.transformer = LlamaModel(config)

    def forward(self, features, attention_mask=None):
        """
        Passes the input features through the transformer model.
        Args:
            features (torch.Tensor): Input features of shape
                (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor, optional): Attention
                mask of shape (batch_size, seq_len).
                Defaults to None.
        Returns:
            torch.Tensor: Output features of shape (batch_size,
                seq_len, embedding_dim).
        """
        outputs = self.transformer(
            inputs_embeds=features, attention_mask=attention_mask)
        return outputs.last_hidden_state


class Attention(nn.Module):
    """ Multi-Head Self Attention module.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable
            bias to query, key, value. Default: False.
        qk_scale (float | None, optional): Override default
            qk scale of head_dim ** -0.5 if set.
        attn_drop (float, optional): Dropout ratio of attention
            weight. Default: 0.0.
        proj_drop (float, optional): Dropout ratio of output.
            Default: 0.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """ Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of mlp hidden
            dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable
            bias to query, key, value. Default: False.
        qk_scale (float | None, optional): Override default
            qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout ratio. Default: 0.
        attn_drop (float, optional): Dropout ratio of attention
            weight. Default: 0.
        drop_path (float, optional): Stochastic depth rate.
            Default: 0.
        act_layer (nn.Module, optional): Activation layer.
            Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SimpleTransformer(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 num_layers: int,
                 num_heads: int = 8,
                 max_seq_len: int = 8192):
        super().__init__()

        self.layers = nn.ModuleList([
            Block(
                dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm) for i in range(num_layers)
        ])

        # 可学习的位置编码 (直接复用 nn.Embedding)
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, seq_len, _ = x.shape

        # 生成位置编码并叠加到输入特征
        position_ids = torch.arange(
            seq_len, device=x.device).expand(B, -1)  # (B, seq_len)
        pos_emb = self.position_embeddings(position_ids)  # (B, seq_len, C)
        x = x + pos_emb

        for layer in self.layers:
            # 将 attention_mask 传递给每一层
            x = layer(x)

        return x


@HEADS.register_module()
class LlavaActionHead(nn.Module):
    """
    Head module for OpenVLA, responsible for decoding generated token IDs
    into continuous unnormalized action vectors.

    Args:
        norm_stats (Dict): Dictionary containing normalization statistics
            for each dataset, used to unnormalize predicted actions.
        vocab_size (int): Size of the vocabulary for action tokens.
        *args, **kwargs: Additional arguments passed to nn.Module.
    """

    def __init__(self,
                 hidden_size: int,
                 state_dim: int,
                 num_layers: int = 1,
                 num_heads: int = 4,
                 traj_length: int = 10,
                 action_dim: int = 7,
                 act_decoder_dim: int = 64,
                 max_seq_len: int = 64,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_encoder = nn.Linear(state_dim, act_decoder_dim)
        self.proj_action_output_embed = nn.Linear(hidden_size, act_decoder_dim)
        self.decode_action = SimpleTransformer(
            embedding_dim=act_decoder_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len)
        self.action_mask_token_embedding = nn.Embedding(1, act_decoder_dim)
        self.traj_length = traj_length
        self.pred_action = nn.Linear(act_decoder_dim, action_dim)

    def forward(self, input_features: torch.Tensor, states: torch.Tensor,
                attention_mask: torch.Tensor, actions: torch.Tensor,
                action_masks: torch.Tensor, **kwargs):
        batch_size = input_features.shape[0]
        states_features = self.state_encoder(states)
        action_decode_query_token = self.action_mask_token_embedding.weight
        action_decode_query_token = action_decode_query_token.unsqueeze(
            0).repeat(batch_size, self.traj_length, 1)
        # Use the last valid token in the sequence as the action input
        batch_indexes = torch.arange(batch_size, device=input_features.device)
        valid_token_indexes = attention_mask.sum(1) - 1
        action_input_features = self.proj_action_output_embed(
            input_features[batch_indexes, valid_token_indexes].unsqueeze(
                1))  # (B, T, act_decoder_dim)
        y = torch.cat([
            states_features.unsqueeze(1), action_input_features,
            action_decode_query_token
        ],
                      dim=1)  # (B, T+1, act_decoder_dim)
        y = self.decode_action(y)
        action_decoder_output_embeddings = y[:, 2:]
        pred_actions = self.pred_action(action_decoder_output_embeddings)
        losses = F.mse_loss(
            pred_actions, actions,
            reduction='none') * action_masks.unsqueeze(-1)
        ret_dict = dict(
            pred_actions=pred_actions,
            loss=losses.sum() / (action_masks.sum() * actions.shape[-1]),
        )
        return ret_dict

    def predict_action(self, input_features: torch.Tensor,
                       states: torch.Tensor, attention_mask: torch.Tensor,
                       *args, **kwargs):
        batch_size = input_features.shape[0]
        states_features = self.state_encoder(states)
        action_decode_query_token = self.action_mask_token_embedding.weight
        action_decode_query_token = action_decode_query_token.unsqueeze(
            0).repeat(batch_size, self.traj_length, 1)
        batch_indexes = torch.arange(batch_size, device=input_features.device)
        valid_token_indexes = attention_mask.sum(1) - 1
        action_input_features = self.proj_action_output_embed(
            input_features[batch_indexes, valid_token_indexes].unsqueeze(
                1))  # (B, T, act_decoder_dim)
        y = torch.cat([
            states_features.unsqueeze(1), action_input_features,
            action_decode_query_token
        ],
                      dim=1)  # (B, T+1, act_decoder_dim)
        y = self.decode_action(y)
        action_decoder_output_embeddings = y[:, 2:]
        actions = self.pred_action(action_decoder_output_embeddings)
        return actions

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Define FSDP wrapping policy for the LlavaActionHead.

        Returns:
            Callable: A policy that wraps the head module.
        """
        return partial(
            _module_wrap_policy,
            module_classes={SimpleTransformer, Block, Mlp})
