# Copyright IBM Corp. 2025
# License: Apache-2.0
# -----------------------------------------------------------------------------
# Adapted from NeuralCompression "torch_vct" module
#   (https://github.com/facebookresearch/NeuralCompression)
# Copyright (c) 2023–2024 Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
#   (see LICENSES/LICENSE-NEURALCOMPRESSION-MIT.txt)
#
# This file also includes components derived from:
#   VCT: A Video Compression Transformer
#   (https://arxiv.org/abs/2206.07307)
# Copyright 2022–2024 The Google Research Authors
# Licensed under the Apache License, Version 2.0
#   (see LICENSES/LICENSE-VCT-APACHE-2.0.txt)
# -----------------------------------------------------------------------------


"""Transformer layers."""

from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers_utils import init_weights_truncated_normal


class WindowMultiHeadAttention(nn.Module):
    """Windowed multi-head attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"Size of hidden units ({d_model}) not divisible by number "
                f"of head ({num_heads})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        # Linear projections for q, k, v
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.k = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

        # Add truncated normal initialization with stddev=0.02
        init_weights_truncated_normal(self.q)
        init_weights_truncated_normal(self.k)
        init_weights_truncated_normal(self.v)
        init_weights_truncated_normal(self.proj)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        v: Tensor,
        k: Tensor,
        q: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the windowed multi-head attention.

        Note that seq_len_kv must be an integer multiple of seq_len_q.

        :param v: (B', seq_len_kv, C) inputs.
        :param k: (B', seq_len_kv, C) inputs.
        :param q: (B', seq_len_q, C) inputs.
        :param mask: Optional mask, must broadcast to (B', num_heads, seq_len_q,
            seq_len_q), and must be in {0, 1}, 1s will be masked.
        :return: Output tensor of shape (B', seq_len_q, C), as well as the
            attention matrix used, shape (B', num_heads, seq_len_q, seq_len_kv).
        """
        B, seq_len_q, C = q.shape
        # B' = B * nH * nW

        assert C == self.d_model, f"Shape mismatch: {C} != {self.d_model}"

        _, seq_len_kv, _ = v.shape

        assert seq_len_kv % seq_len_q == 0, (
            f"seq_len_kv {seq_len_kv} not divisible by seq_len_q {seq_len_q}"
        )

        # project and reshape
        def reshape_proj(x: Tensor, linear: nn.Linear) -> Tensor:
            # x: (B', seq_len, C)
            x = linear(x)  # (B', seq_len, d_model)
            # reshape to (B', seq_len, num_heads, head_dim)
            x = x.view(B, -1, self.num_heads, self.head_dim)
            # permute to (B', num_heads, seq_len, head_dim)
            return x.permute(0, 2, 1, 3).contiguous()

        q_proj = reshape_proj(q, self.q) * self.scale
        k_proj = reshape_proj(k, self.k)
        v_proj = reshape_proj(v, self.v)

        # attention scores
        attn = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        # (B', num_heads, seq_q, seq_kv)

        if mask is not None:
            assert mask.shape[-2:] == (seq_len_q, seq_len_q), (
                f"Mask shape {mask.shape} is not compatible with "
                f"attention shape {attn.shape}."
            )
            # mask: True for positions to mask
            attn = attn.masked_fill(mask.bool(), float("-inf"))

            # blowup = seq_len_kv // seq_len_q
            # tile_pattern = [1] * mask.dim()
            # tile_pattern[-1] = blowup
            # attn = attn + torch.tile(mask, tile_pattern) * -1e6

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        out = torch.matmul(attn, v_proj)  # (B, num_heads, seq_q, head_dim)
        assert out.shape == (
            B,
            self.num_heads,
            seq_len_q,
            C // self.num_heads,
        ), "out shape mismatch"
        # this means head_dim = C // num_heads
        # combine heads
        out = out.permute(
            0, 2, 1, 3
        ).contiguous()  # (B', seq_q, num_heads, head_dim)
        out = out.view(B, seq_len_q, C)  # (B', seq_q, d_model)

        # project and dropout
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


def create_look_ahead_mask(size: int) -> Tensor:
    """Creates a look-ahead mask for autoregressive attention.

    :param size: Size of the mask.
    :return: A boolean mask of shape (size, size) where True indicates
        positions that should be masked (i.e., not attended to).
        The mask is upper triangular, meaning that for each position i,
        positions j > i are masked.
    """
    # mask[i,j] = True if j > i
    mask = torch.triu(torch.ones((size, size), dtype=torch.bool), diagonal=1)
    return mask


class StochasticDepth(nn.Module):
    """Stochastic depth layer.

    This is a dropout layer that randomly drops entire channels during training.
    It is used to prevent overfitting in deep networks.
    """

    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.drop_rate == 0.0:
            return inputs

        keep_prob = 1.0 - self.drop_rate
        # Generate binary mask of shape [batch, 1, 1, ...]
        shape = (inputs.size(0),) + (1,) * (inputs.dim() - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=inputs.dtype, device=inputs.device
        )
        binary_tensor = torch.floor(random_tensor)
        return inputs.div(keep_prob) * binary_tensor


class MLP(nn.Module):
    """MLP head for transformer blocks"""

    def __init__(
        self,
        in_features: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        """
        MLP head for transformer blocks
        Args:
            expansion_rate: rate at which the input tensor is expanded
            dropout_rate: dropout rate
            input_shape: shape of the input tensor, with the last dimension
                as the size of channel -- [N1, ... Nn, C]
        """
        super().__init__()

        # Initialize linear layers with truncated normal
        self.fc1 = nn.Linear(in_features=in_features, out_features=mlp_dim)
        init_weights_truncated_normal(self.fc1)
        self.fc2 = torch.nn.Linear(
            in_features=mlp_dim, out_features=in_features
        )
        init_weights_truncated_normal(self.fc2)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            features: tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            tensor of shape (batch_size, seq_len, hidden_dim)
        """
        input = self.fc1(input)
        input = self.act(input)
        input = self.dropout(input)
        input = self.fc2(input)
        return self.dropout(input)  # type: ignore[no-any-return]


class TransformerBlock(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        num_heads: int,  # 4
        mlp_dim: int,  # 4 (expansion rate)
        dropout_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm,
            eps=1e-6,
        ),
        is_decoder: bool = False,
    ) -> None:
        super().__init__()
        # Look-ahead mask for decoder
        if is_decoder:
            self.register_buffer(
                "look_ahead_mask",
                create_look_ahead_mask(seq_len),
            )
        else:
            self.look_ahead_mask = None
        self.is_decoder = is_decoder

        # Layer norms
        self.norm1a = norm_layer(d_model)
        self.norm1b = norm_layer(d_model)
        self.norm2a = norm_layer(d_model)
        self.norm2b = norm_layer(d_model)

        # Attention layers
        self.self_attention = WindowMultiHeadAttention(
            d_model, num_heads, attn_drop=dropout_rate, proj_drop=dropout_rate
        )
        self.cross_attention = WindowMultiHeadAttention(
            d_model, num_heads, attn_drop=dropout_rate, proj_drop=dropout_rate
        )

        # MLP heads
        self.mlp1 = MLP(d_model, mlp_dim, dropout_rate)
        self.mlp2 = MLP(d_model, mlp_dim, dropout_rate)

        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(
        self,
        features: Tensor,
        encoder_output: Optional[Tensor],
    ) -> Tensor:
        assert features.dim() == 3, (
            f"Expected (batch_size, seq_length, hidden_dim) "
            f"got {features.shape}"
        )

        # Validate encoder output for decoder mode
        if encoder_output is None and self.is_decoder:
            raise ValueError("Need `encoder_output` when running decoder.")
        # First block: self-attention
        shortcut = features
        x = self.norm1a(features)
        attn_out1, _ = self.self_attention(
            v=x,
            k=x,
            q=x,
            mask=self.look_ahead_mask,
        )
        x = shortcut + self.drop_path(attn_out1)
        x = x + self.drop_path(self.mlp1(self.norm1b(x)))

        # Second block: cross-attention (or self if encoder)
        shortcut = x
        x_norm = self.norm2a(x)
        v_in = encoder_output if encoder_output is not None else x_norm
        attn_out2, _ = self.cross_attention(
            v=v_in,
            k=v_in,
            q=x_norm,
            mask=None,
        )
        x = shortcut + self.drop_path(attn_out2)
        output = x + self.drop_path(self.mlp2(self.norm2b(x)))
        return output  # type: ignore[no-any-return]


class Transformer(nn.Module):
    """Stack of TransformerBlock layers."""

    def __init__(
        self,
        seq_len: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        d_model: int = 192,
        mlp_expansion: int = 4,
        dropout_rate: float = 0.1,
        is_decoder: bool = False,
        norm_layer: Callable[..., nn.Module] = partial(
            nn.LayerNorm,
            eps=1e-6,
        ),
    ) -> None:
        """Initialize the Transformer.

        :param seq_len: Sequence length.
        :param num_layers: Number of transformer layers.
        :param num_heads: Number of attention heads.
        :param d_model: Dimension of the model.
        :param mlp_expansion: Expansion rate for the MLP.
        :param dropout_rate: Dropout rate.
        :param is_decoder: Whether this is a decoder.
        :param norm_layer: Normalization layer to use.
        """
        super().__init__()
        self.is_decoder = is_decoder
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList()
        # Use ModuleList to hold layers
        for i in range(num_layers):
            self.layers.add_module(
                f"encoder_layer_{i}",
                TransformerBlock(
                    d_model=d_model,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    mlp_dim=d_model * mlp_expansion,
                    dropout_rate=dropout_rate,
                    drop_path_rate=dropout_rate,
                    norm_layer=norm_layer,
                    is_decoder=is_decoder,
                ),
            )

    def forward(
        self,
        latent: Tensor,
        encoder_output: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through a stack of TransformerBlock.

        Args:
            latent: Tensor of shape (B, seq_len, d_model)
            encoder_output: Optional encoder output for decoder layers
        Returns:
            Tensor of shape (B, seq_len, d_model)
        """
        assert latent.dim() == 3, (
            f"Expected (B, seq_len, d_model) got {latent.shape}"
        )
        if encoder_output is not None:
            assert latent.shape[-1] == encoder_output.shape[-1], (
                "Expected latent and encoder_output to have the same last "
                "dimension, "
                f"but got {latent.shape[-1]} and {encoder_output.shape[-1]}"
            )
        x = latent
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x


class EncoderSection(Transformer):
    """
    Formerly "T_sep" in the original paper.
    A wrapper around `Transformer` turning it into an Encoder by setting
    the following:
        - is_decoder=False
        - seq_length=0
        - encoder_output=None
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        mlp_expansion: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6
        ),
    ) -> None:
        super().__init__(
            seq_len=0,
            # NO-OP: used for look_ahead_mask only, no masking in encoder
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            mlp_expansion=mlp_expansion,
            dropout_rate=dropout,
            is_decoder=False,
            norm_layer=norm_layer,
        )

    def forward(
        self, latent: Tensor, encoder_output: Optional[Tensor] = None
    ) -> Tensor:
        return super().forward(latent, encoder_output=encoder_output)
