import numpy as np
import torch.nn.functional as F
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange, repeat
# from fairscale.nn import checkpoint_wrapper
from torch import Tensor

from detectron2.data import MetadataCatalog, DatasetCatalog
from torchnlp.word_to_vector import GloVe
# from fairscale.nn import checkpoint_wrapper
import random
from .my_module import *
from .meta_arch.gdl import decouple_layer, AffineLayer
from sentence_transformers import SentenceTransformer



class Sequential(nn.Sequential):
    def forward(self, *x):
        for module in self:
            if type(x) == tuple:
                x = module(*x)
            else:
                x = module(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of channels query and key input channels are projected to,
            for computing the attention matrix. Defaults to number `num_q_input_channels`
        :param num_v_channels: Number of channels value input channels are projected to.
            Defaults to `num_qk_channels`.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param attn_mask: Boolean attention mask. Not needed/supported yet.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """
        if attn_mask is not None:
            raise NotImplementedError("attention masks not supported yet")

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads)
                   for x in [q, k, v])
        attn = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "b j -> (b h) () j", h=self.num_heads)
            attn_max_neg = -torch.finfo(attn.dtype).max
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b i j, b j c -> b i c", attn, v)
        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class SharedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of channels query and key input channels are projected to,
            for computing the attention matrix. Defaults to number `num_q_input_channels`
        :param num_v_channels: Number of channels value input channels are projected to.
            Defaults to `num_qk_channels`.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        # self._proj = nn.Linear(num_q_input_channels, num_qk_channels)
        self.kq_proj = nn.Linear(num_kv_input_channels, num_qk_channels)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param attn_mask: Boolean attention mask. Not needed/supported yet.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """
        if attn_mask is not None:
            raise NotImplementedError("attention masks not supported yet")

        q = self.kq_proj(x_q)
        k = self.kq_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads)
                   for x in [q, k, v])
        attn = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "b j -> (b h) () j", h=self.num_heads)
            attn_max_neg = -torch.finfo(attn.dtype).max
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b i j, b j c -> b i c", attn, v)
        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class SharedMultiHeadAttention2(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of channels query and key input channels are projected to,
            for computing the attention matrix. Defaults to number `num_q_input_channels`
        :param num_v_channels: Number of channels value input channels are projected to.
            Defaults to `num_qk_channels`.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        # self._proj = nn.Linear(num_q_input_channels, num_qk_channels)
        self.kq_proj = nn.Linear(num_kv_input_channels, num_qk_channels)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, pad_mask=None, attn_mask=None):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param attn_mask: Boolean attention mask. Not needed/supported yet.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """
        if attn_mask is not None:
            raise NotImplementedError("attention masks not supported yet")

        q = self.kq_proj(x_q)
        k = self.kq_proj(x_k)
        v = self.v_proj(x_v)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads)
                   for x in [q, k, v])
        attn = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "b j -> (b h) () j", h=self.num_heads)
            attn_max_neg = -torch.finfo(attn.dtype).max
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b i j, b j c -> b i c", attn, v)
        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head cross-attention (see `MultiHeadAttention` for details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """Multi-head attention of query input `x_q` to key/value input (`x_kv`) after (separately) applying layer
        normalization to these inputs."""
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SharedCrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head cross-attention (see `MultiHeadAttention` for details)."""
        super().__init__()
        self.attention = SharedMultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """Multi-head attention of query input `x_q` to key/value input (`x_kv`) after (separately) applying layer
        normalization to these inputs."""
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head self-attention (see `MultiHeadAttention` and for details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        """Multi-head attention of input `x` to itself after applying layer normalization to the input."""
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class CrossAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
        attention_residual: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )
        super().__init__(
            Residual(cross_attn) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor)),
        )


class SharedCrossAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
        attention_residual: bool = True,
    ):
        cross_attn = SharedCrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )
        super().__init__(
            Residual(cross_attn) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor)),
        )


class SelfAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )
        super().__init__(
            Residual(self_attn),
            Residual(MLP(num_channels, widening_factor)),
        )


class SelfAttentionBlock(Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                widening_factor=widening_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

        if activation_checkpointing:
            layers = [checkpoint_wrapper(
                layer, offload_to_cpu=activation_offloading) for layer in layers]

        super().__init__(*layers)


class MLP(Sequential):
    def __init__(self, num_channels: int, widening_factor: int):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels),
        )


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) + args[0]


class InputAdapter(nn.Module):
    def __init__(self, num_input_channels: int):
        """Transforms and position-encodes task-specific input to generic encoder input.

        :param num_input_channels: Number of channels of the generic encoder input produced by this adapter.
        """
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels

    def forward(self, x):
        raise NotImplementedError()


class OutputAdapter(nn.Module):
    def __init__(self, output_query: Tensor, init_scale: float):
        """Transforms generic decoder cross-attention output to task-specific output.

        :param output_query: Output query prototype (does not include batch dimension) used as query input to
            generic decoder cross-attention.
        :param init_scale: Output query parameter initialization scale.
        """
        super().__init__()
        self._output_query = nn.Parameter(output_query)
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self._output_query.normal_(0.0, init_scale)

    @property
    def num_output_query_channels(self):
        return self._output_query.shape[-1]

    def output_query(self, x):
        return repeat(self._output_query, "... -> b ...", b=x.shape[0])


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_queries: int = 1,
        num_output_query_channels: Optional[int] = None,
        init_scale: float = 0.02,
    ):

        if num_output_query_channels is None:
            num_output_query_channels = num_classes

        super().__init__(output_query=torch.empty(num_output_queries,
                                                  num_output_query_channels), init_scale=init_scale)
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        num_cross_attention_layers: int = 1,
        first_cross_attention_layer_shared: bool = False,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,
        num_self_attention_blocks: int = 1,
        first_self_attention_block_shared: bool = True,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input
            of shape (B, M, C) where B is the batch size, M the input sequence length and C the number of
            key/value input channels. C is determined by the `num_input_channels` property of the
            `input_adapter`.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights
            with subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks sharing weights between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights
            with subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers and residuals.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        self.input_adapter = input_adapter

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError(
                "num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=input_adapter.num_input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
            )
            return (
                checkpoint_wrapper(
                    layer, offload_to_cpu=activation_offloading) if activation_checkpointing else layer
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(
            num_latents, num_latent_channels))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.latent.normal_(0.0, init_scale)
            _init_parameters(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attention_layers > 1 and not self.first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attention_blocks > 1 and not self.first_self_attention_block_shared

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn_1(x_latent, x, pad_mask)
        x_latent = self.self_attn_1(x_latent)

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = cross_attn_n(x_latent, x, pad_mask)
            x_latent = self_attn_n(x_latent)

        return x_latent


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        output_adapter: OutputAdapter,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        cross_attention_widening_factor: int = 1,
        cross_attention_residual: bool = True,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder cross-attention output of shape (B, O, F) to task-specific
            output. B is the batch size, O the output sequence length and F the number of cross-attention output
            channels. F is determined by the `num_output_query_channels` property of the `output_adapter`.
        :param num_latent_channels: Number of latent channels (C_latent) as produced by a Perceiver IO encoder.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layers and residuals.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        cross_attn = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_adapter.num_output_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            attention_residual=cross_attention_residual,
            dropout=dropout,
        )

        if activation_checkpointing:
            cross_attn = checkpoint_wrapper(
                cross_attn, offload_to_cpu=activation_offloading)

        self.cross_attn = cross_attn
        self.output_adapter = output_adapter
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            _init_parameters(self, init_scale)

    def forward(self, x, **kwargs):
        output_query = self.output_adapter.output_query(x)
        output = self.cross_attn(output_query, x)
        return self.output_adapter(output, **kwargs)


class PerceiverIO(Sequential):
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)

    @property
    def encoder(self):
        return self[0]

    @property
    def decoder(self):
        return self[1]


class CustomPerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        num_cross_attention_layers: int = 3,  # 1
        first_cross_attention_layer_shared: bool = False,  # False
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,  # 6
        num_self_attention_blocks: int = 3,  # 1,
        first_self_attention_block_shared: bool = True,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.2,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input
            of shape (B, M, C) where B is the batch size, M the input sequence length and C the number of
            key/value input channels. C is determined by the `num_input_channels` property of the
            `input_adapter`.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights
            with subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks sharing weights between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights
            with subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers and residuals.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError(
                "num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            # if num_cross_attention_qk_channels == None:

            # layer = SharedCrossAttentionLayer(
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=num_cross_attention_qk_channels if num_cross_attention_qk_channels else num_latent_channels,
                num_qk_channels=num_cross_attention_qk_channels if num_cross_attention_qk_channels else num_latent_channels,
                num_v_channels=num_cross_attention_v_channels if num_cross_attention_v_channels else num_latent_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
            )
            return layer
            return (
                checkpoint_wrapper(
                    layer, offload_to_cpu=activation_offloading) if activation_checkpointing else layer
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels if num_self_attention_qk_channels else num_latent_channels,
                num_v_channels=num_self_attention_v_channels if num_self_attention_v_channels else num_latent_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()
        # self.self_attn_query = self_attn()
        # self.self_attn_kv = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

        d_ffn = 1024
        # ffn
        self.linear1 = nn.Linear(num_latent_channels, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, num_latent_channels)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(num_latent_channels)

        self._init_parameters(init_scale)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            _init_parameters(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attention_layers > 1 and not self.first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attention_blocks > 1 and not self.first_self_attention_block_shared

    def forward(self, x, query, pad_mask=None):
        # query =
        # b, *_ = x.shape

        # encode task-specific input
        # x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        # query = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn_1(query, x, pad_mask)
        x_latent = self.self_attn_1(x_latent)

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = cross_attn_n(query, x_latent, pad_mask)
            x_latent = self_attn_n(x_latent)
        x_latent = self.forward_ffn(x_latent)
        return x_latent


class ourCrossAttention(nn.Module):
    def __init__(self,
                 num_latent_channels: int,
                 num_cross_attention_heads: int = 4,
                 num_cross_attention_qk_channels: Optional[int] = None,
                 num_cross_attention_v_channels: Optional[int] = None,
                 num_cross_attention_layers: int = 3,  # 1
                 first_cross_attention_layer_shared: bool = False,  # False
                 cross_attention_widening_factor: int = 1,
                 num_self_attention_heads: int = 4,
                 num_self_attention_qk_channels: Optional[int] = None,
                 num_self_attention_v_channels: Optional[int] = None,
                 num_self_attention_layers_per_block: int = 6,  # 6
                 num_self_attention_blocks: int = 3,  # 1,
                 first_self_attention_block_shared: bool = True,
                 self_attention_widening_factor: int = 1,
                 dropout: float = 0.2,
                 init_scale: float = 0.02,
                 activation_checkpointing: bool = False,
                 activation_offloading: bool = False,) -> None:

        super().__init__()
        self.cross_attn = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=num_cross_attention_qk_channels if num_cross_attention_qk_channels else num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels if num_cross_attention_qk_channels else num_latent_channels,
            num_v_channels=num_cross_attention_v_channels if num_cross_attention_v_channels else num_latent_channels,
            widening_factor=cross_attention_widening_factor,
            dropout=dropout,
        )

        self.cross_attn = SharedMultiHeadAttention2(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=num_latent_channels,)
        d_ffn = 1024
        # ffn
        self.linear1 = nn.Linear(num_latent_channels, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, num_latent_channels)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(num_latent_channels)

        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            _init_parameters(self, init_scale)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, q, k, v, pad_mask=None):
        # query =
        # b, *_ = x.shape

        # encode task-specific input
        # x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        # query = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn(x_q=q, x_k=k, x_v=v, pad_mask=pad_mask)

        x_latent = self.forward_ffn(x_latent)
        return x_latent


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class FFN(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        d_ffn = 1024
        self.d_model = d_model
        # ffn
        self.linear1 = nn.Linear(self.d_model, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, self.d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(self.d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SingleHeadSiameseAttention(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""

    def __init__(self, d_model, dropout=0):
        super().__init__()
        self.n_head = 1
        self.d_model = d_model
        self.w_qk = nn.Linear(self.d_model, self.n_head *
                              self.d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_model, 0.5), dropout=dropout)
        nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(
            2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)
        self.ffn = FFN(self.d_model, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)

        # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(
            sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head,
                              self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1,
                                                    self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1,
                                                    self.d_model)  # (n_head * b) x lv x d_model
        # tsp = tsp.permute(2, 0, 1, 3).contiguous(
        # ).view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
        use_cosine = False
        if use_cosine:
            def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
            def cosine_func(x, y): return torch.einsum(
                'b i c, b j c -> b i j', norm(x), norm(y))
            output = cosine_func(q, k)
            output = F.relu(output)
            output = self.dropout(output)
            output = torch.bmm(output, v)
            # print('cosine_func:', output.shape)
        else:
            output, attn, log_attn = self.attention(q, k, v)
            # print('softmax:', output.shape)

        # tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * residual)
        # print('output', output.shape)
        # print('residual', residual.shape)
        # output3 = self.linear2(residual + output)
        output2 = self.linear2(residual - output)
        # output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )
        output = self.ffn(output)
        # assert 0
        return output

class OurSingleHeadSiameseAttention2(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""

    def __init__(self, d_model, dropout=0):
        super().__init__()
        self.n_head = 1
        self.d_model = d_model
        self.w_qk = nn.Linear(self.d_model, self.n_head *
                              self.d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_model, 0.5), dropout=dropout)
        nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(
            2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)
        self.ffn = FFN(self.d_model, dropout)

        self.active = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)

        # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(
            sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head,
                              self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1,
                                                    self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1,
                                                    self.d_model)  # (n_head * b) x lv x d_model
        # tsp = tsp.permute(2, 0, 1, 3).contiguous(
        # ).view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
        use_cosine = True
        if use_cosine:
            def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
            def cosine_func(x, y): return torch.einsum(
                'b i c, b j c -> b i j', norm(x), norm(y))
            output = cosine_func(q, k)
            output = self.active(output)
            output = self.dropout(output)
            output = torch.bmm(output, v)
            # print('cosine_func:', output.shape)
        else:
            output, attn, log_attn = self.attention(q, k, v)
            # print('softmax:', output.shape)

        # tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * residual)
        # print('output', output.shape)
        # print('residual', residual.shape)
        # output3 = self.linear2(residual + output)
        output2 = self.linear2(residual - output)
        # output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )
        output = self.ffn(output+residual)
        # assert 0
        return output


class OurSingleHeadSiameseAttention(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""

    def __init__(self, d_vis_feat, d_text_feat, d_model=1024, dropout=0):
        super().__init__()
        self.n_head = 1
        # self.d_model = d_model
        self.q_size = d_vis_feat + d_text_feat
        self.k_size = d_text_feat
        self.v_size = d_text_feat 
        self.d_model = d_model

        if d_vis_feat != d_text_feat:
            self.buffer_text = nn.Linear(self.k_size, self.q_size, bias=False)
        else:
            self.buffer_text = nn.Identity()
        self.w_qk = nn.Linear(self.q_size, self.d_model, bias=False)
        self.w_v = nn.Linear(self.v_size, self.d_model, bias=False)
        # self.w_o = nn.Linear(self.d_model, self.q_size, bias=False)

        use_norm=True
        norm_layer = nn.LayerNorm if use_norm else nn.Identity
        self.norm_q = norm_layer(self.d_model)
        self.norm_k = norm_layer(self.d_model)
        self.norm_v = norm_layer(self.d_model)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_model, 0.5), dropout=dropout)

        for w in [self.w_qk, self.w_v]:
            nn.init.normal_(w.weight, mean=0, std=np.sqrt(
                2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2), torch.nn.GELU())
        self.linear2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), torch.nn.GELU())
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)
        self.ffn = FFN(self.d_model, dropout)
        self.active = torch.nn.GELU()

        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, q, k, v):
        sz_b, len_q, _ = q.size()
        # print("q.size()", q.size())
        sz_b, len_k, _ = k.size()
        # print("k.size()", k.size())
        sz_b, len_v, _ = v.size()
        # print("v.size()", v.size())

        residual1 = q
        
        q = self.w_qk(q)
        # k = self.w_qk(self.buffer_text(k))
        # v = self.w_v(v)
        
        residual1 = q
        q = self.norm_q(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.norm_k(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = self.norm_v(v).view(sz_b, len_v, self.n_head, self.d_model)
        # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(
            sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head,
                              self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1,
                                                    self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1,
                                                    self.d_model)  # (n_head * b) x lv x d_model
        # tsp = tsp.permute(2, 0, 1, 3).contiguous(
        # ).view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
        use_cosine = False
        if use_cosine:
            def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
            def cosine_func(x, y): return torch.einsum(
                'b i c, b j c -> b i j', norm(x), norm(y))
            output = cosine_func(q, k)
            output = self.dropout(output)
            output = F.relu(output)
            output = torch.bmm(output, v)
            # print('cosine_func:', output.shape)
        else:
            output, attn, log_attn = self.attention(q, k, v)
            # print('softmax:', output.shape)

        # tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * residual1)
        # print('output', output.shape)
        # print('residual', residual.shape)
        # output3 = self.linear2(residual + output)
        # output2 = self.linear2(residual - output)
        output2 = self.linear2(residual1 - output)

        output = self.linear3(
            torch.cat([output1, output2, residual1], dim=2)
        )
        output = self.ffn(output)
        output = self.active(output)
        # output = self.active(self.w_o(output)) #+ residual2

        # assert 0
        return output


class SingleHeadSiameseAttention2(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""

    def __init__(self, d_model, dropout=0):
        super().__init__()
        self.n_head = 1
        self.d_model = d_model
        self.w_qk = nn.Linear(self.d_model, self.n_head *
                              self.d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_model, 0.5), dropout=dropout)
        nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(
            2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)
        self.ffn = FFN(self.d_model, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        q1 = inputs['q1']
        k1 = inputs['k1']
        v1 = inputs['v1']
        q2 = inputs['q2']
        k2 = inputs['k2']
        v2 = inputs['v2']

        res1 = q1
        res2 = q2

        output1 = self.subforward(q1, k1, v1)
        output2 = self.subforward(q2, k2, v2)

        output = self.linear1(output1*res1 + output2*res2)
        # print('output', output.shape)
        # print('residual', residual.shape)
        # output3 = self.linear2(residual + output)
        # output2 = self.linear2(residual - output)

        output2 = self.linear2((res1 - output1)*(res2-output2))

        output = self.linear3(
            torch.cat([output, output2, res1+res2], dim=2)
        )
        output = self.ffn(output)
        # assert 0
        return output

    def subforward(self, q, k, v):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)

        # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(
            sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head,
                              self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1,
                                                    self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1,
                                                    self.d_model)  # (n_head * b) x lv x d_model
        # tsp = tsp.permute(2, 0, 1, 3).contiguous(
        # ).view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
        use_cosine = False
        if use_cosine:
            def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
            def cosine_func(x, y): return torch.einsum(
                'b i c, b j c -> b i j', norm(x), norm(y))
            output = cosine_func(q, k)
            output = F.relu(output)
            output = self.dropout(output)
            output = torch.bmm(output, v)
            # print('cosine_func:', output.shape)
        else:
            output, attn, log_attn = self.attention(q, k, v)
            # print('softmax:', output.shape)

        # tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n_head * d_model)
        return output+residual

    def forward2(self, q, k, v):

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)

        # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(
            sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head,
                              self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1,
                                                    self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1,
                                                    self.d_model)  # (n_head * b) x lv x d_model
        # tsp = tsp.permute(2, 0, 1, 3).contiguous(
        # ).view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
        use_cosine = False
        if use_cosine:
            def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
            def cosine_func(x, y): return torch.einsum(
                'b i c, b j c -> b i j', norm(x), norm(y))
            output = cosine_func(q, k)
            output = F.relu(output)
            output = self.dropout(output)
            output = torch.bmm(output, v)
            # print('cosine_func:', output.shape)
        else:
            output, attn, log_attn = self.attention(q, k, v)
            # print('softmax:', output.shape)

        # tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * residual)
        # print('output', output.shape)
        # print('residual', residual.shape)
        # output3 = self.linear2(residual + output)
        output2 = self.linear2(residual - output)
        # output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )
        output = self.ffn(output+residual)
        # assert 0
        return output


class MultiHeadSiameseAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param d_model: Number of input channels.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        num_qk_channels_per_head = d_model * num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads
        self.w_qk = nn.Linear(d_model, d_model, bias=False)

        self.dummy = nn.Parameter(torch.Tensor(1, d_model))
        nn.init.normal_(self.dummy)

        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Sequential(nn.Linear(
            num_qk_channels_per_head, num_qk_channels_per_head // 2), nn.ReLU(inplace=True))
        #     nn.Identity()

        # )

        self.linear2 = nn.Sequential(
            nn.Linear(num_qk_channels_per_head, num_qk_channels_per_head // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(
            num_qk_channels_per_head * 2, num_qk_channels_per_head)
        self.ffn = FFN(num_qk_channels_per_head, dropout)

    def forward(self, q, k, v):
        # q = rearrange(q, "b n c -> n b c")
        # k = rearrange(k, "b n c -> n b c")
        # v = rearrange(v, "b n c -> n b c")
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)
        # print('v.shape', v.shape)
        # print()
        # assert 0
        sz_b, len_q, dim_q = q.size()
        # sz_b, len_k, _ = k.size()
        # sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q)  # .view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k)  # .view(sz_b, len_k, self.n_head, self.d_model)
        # v = v.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, dim_q).expand(sz_b, -1, -1)
        dummy_v = torch.zeros(sz_b, 1, dim_q, device=v.device)

        # dummy = dummy.reshape(sz_b, 1, dim_q)
        # dummy_v = dummy_v.reshape(sz_b, 1, dim_q)

        # print('dummy.shape', dummy.shape)
        # print('dummy_v.shape', dummy_v.shape)
        # print('k.shape', k.shape)
        # print('v.shape', v.shape)
        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q, k, v, residual = (rearrange(
            x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v, residual])

        output = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale
        output = output.softmax(dim=-1)

        # def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
        # def cosine_func(x, y): return torch.einsum(
        #     'b i c, b j c -> b i j', norm(x), norm(y))
        # output = cosine_func(q, k)

        # print('att.shape', output.shape)
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)
        # print('v.shape', v.shape)

        output = self.dropout(output)

        output = torch.einsum("b i j, b j c -> b i c", output, v)

        output1 = self.linear1(output * residual)
        output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )
        output = self.ffn(output + residual)
        output = rearrange(output, "(b h) n c -> b n (h c)", h=self.num_heads)
        # output = rearrange(output, "b n c -> n b c")
        # assert 0
        return output


class MultiHeadSiameseAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param d_model: Number of input channels.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        num_qk_channels_per_head = d_model // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads
        self.w_qk = nn.Linear(d_model, d_model, bias=False)

        self.dummy = nn.Parameter(torch.Tensor(1, d_model))
        nn.init.normal_(self.dummy)

        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Sequential(nn.Linear(
            num_qk_channels_per_head, num_qk_channels_per_head // 2), nn.ReLU(inplace=True))
        #     nn.Identity()

        # )

        self.linear2 = nn.Sequential(
            nn.Linear(num_qk_channels_per_head, num_qk_channels_per_head // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(
            num_qk_channels_per_head * 2, num_qk_channels_per_head)
        self.ffn = FFN(num_qk_channels_per_head, dropout)

    def forward(self, q, k, v, num_per_img=2):
        # q = rearrange(q, "b n c -> n b c")
        # k = rearrange(k, "b n c -> n b c")
        # v = rearrange(v, "b n c -> n b c")
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)
        # print('v.shape', v.shape)
        # print()
        # assert 0
        sz_b, len_q, dim_q = q.size()
        # sz_b, len_k, _ = k.size()
        # sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q)  # .view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k)  # .view(sz_b, len_k, self.n_head, self.d_model)
        # v = v.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, dim_q).expand(sz_b, -1, -1)
        dummy_v = torch.zeros(sz_b, 1, dim_q, device=v.device)

        # dummy = dummy.reshape(sz_b, 1, dim_q)
        # dummy_v = dummy_v.reshape(sz_b, 1, dim_q)

        # print('dummy.shape', dummy.shape)
        # print('dummy_v.shape', dummy_v.shape)
        # print('k.shape', k.shape)
        # print('v.shape', v.shape)
        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)

        q, k, v, residual = (rearrange(
            x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v, residual])

        # output = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale
        # output = output.softmax(dim=-1)

        def norm(x): return torch.nn.functional.normalize(x, p=2.0, dim=-1)
        def cosine_func(x, y): return torch.einsum(
            'b i c, b j c -> b i j', norm(x), norm(y))
        output = cosine_func(q, k)
        output = F.relu(output)

        # print('att.shape', output.shape)
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)
        # print('v.shape', v.shape)

        output = self.dropout(output)

        output = torch.einsum("b i j, b j c -> b i c", output, v)

        output1 = self.linear1(output * residual)
        output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )
        output = self.ffn(output + residual) + v
        output = rearrange(output, "(b h) n c -> b n (h c)", h=self.num_heads)
        # output = rearrange(output, "b n c -> n b c")
        # assert 0
        return output


class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = adj_matrix@node_feats
        node_feats = node_feats / num_neighbours
        return node_feats


class GraphAttention(nn.Module):
    def __init__(self, input_dim, num_layer=4, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList(
            [GCNLayer(input_dim, input_dim) for _ in range(num_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, w):
        # w_norm = F.normalize(w)
        w_norm = w
        weights = w_norm@w_norm.T
        res = x
        for l in self.layers:
            x = F.relu(l(x, weights))
            x = self.dropout(x)
        x = x + res

        return x


def _init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            
            
            
class LV_attention(nn.Module):
    def __init__(self,
                 input_size,
                 cfg=None,
                 is_multi=False,
                 output_size=0,
                 dropout=0):
        super().__init__()
        self.is_multi = is_multi
        self.device = 'cuda'
        self.output_size = output_size if output_size else input_size
        self.dropout = dropout
        self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        self.__init_language_model__(cfg)
        # self.__init_attention_layer__(input_size, num_super_cls)
        self.__init_attention_layer__(input_size)
        
        if self.student_training:
            # self.mlp_adapter = MLP(input_size, widening_factor=2)
            # self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
            )
            self.mlp_adapter2 = torch.nn.Sequential(
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
            )
    def __init_language_model__(self, cfg, num_clusters=6):
        text_dim = 300
        self.l_model = GloVe(name='6B', dim=text_dim)
        # self.l_model = GloVe(name='42B', dim=text_dim)

        dataset_name = cfg.DATASETS.TRAIN[0]

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

        # metadata_dict.novel_classes
        map_voc = {'aeroplane': 'aeroplane', 'bicycle': 'bicycle', 'boat': 'boat', 'bottle': 'bottle', 'car': 'car', 'cat': 'cat', 'chair': 'chair', 'diningtable': 'dining table', 'dog': 'dog', 'horse': 'horse',
                   'person': 'person', 'pottedplant': 'potted plant', 'sheep': 'sheep', 'train': 'train', 'tvmonitor': 'tv', 'bird': 'bird', 'bus': 'bus', 'cow': 'cow', 'motorbike': 'motorbike', 'sofa': 'sofa'}

        # print(base_classes)
        embed = torch.zeros(len(self.classes), text_dim).to(self.device)

        for id, name in enumerate(self.classes):
            text = map_voc[name]
            for i in text.split(' '):
                embed[id] = embed[id] + \
                    self.l_model[i][None, :].to(self.device)
        self.class_id = torch.arange(len(self.classes)+1)
        self.embed = embed
        self.w_bg_init = torch.randn(1, text_dim)
        # self.w_bg_init = torch.zeros(1, text_dim)
        self.w_bg = torch.nn.parameter.Parameter(
            self.w_bg_init.clone(), requires_grad=True)
        return
    def __init_attention_layer__(self, input_size):
        text_dim = 300
        init_scale = 0.02
        self.attention = SingleHeadSiameseAttention(input_size)
        self.proj_k = nn.Linear(input_size*2, input_size)
        self.proj2 = nn.Linear(text_dim, input_size)
        with torch.no_grad():
            _init_parameters(self.attention, init_scale)
            
    def forward(self, x):
        visual_feat = x[None, :]

        embed = torch.cat([self.embed, self.w_bg], dim=0)  # add bg
        embed = self.proj2(embed)[None, :]

        # visual_feat, embed
        # print(x.shape, embed.shape)
        weighted = torch.einsum(
            'b j, i j ->b i', x, embed[0])

        stext_feat = torch.einsum(
            'b i, i j ->b j', weighted, embed[0]
        )[None, :]

        val = torch.cat([visual_feat, stext_feat], dim=2)
        val = self.proj_k(val)

        stext_feat = F.relu(stext_feat)
        val = F.relu(val)

        sim2stext = self.attention(
            q=visual_feat, k=stext_feat, v=val)[0]
        sim2stext = F.relu(sim2stext)
        return sim2stext
