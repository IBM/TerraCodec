# Copyright IBM Corp. 2025
# License: Apache-2.0
# -----------------------------------------------------------------------------
# Adapted from NeuralCompression "torch_vct" module
#   (https://github.com/facebookresearch/NeuralCompression)
# Copyright (c) 2023–2024 Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
#   (see LICENSES/LICENSE-NEURALCOMPRESSION-MIT.txt)
#
# Adapted from CompressAI (https://github.com/InterDigitalInc/CompressAI)
# Copyright (c) 2021-2024, InterDigital Communications, Inc.
# Licensed under the BSD 3-Clause License
# (see LICENSES/LICENSE-COMPRESSAI-BSD-3.txt)
#
# This file also includes components derived from:
#   VCT: A Video Compression Transformer
#   (https://arxiv.org/abs/2206.07307)
# Copyright 2022–2024 The Google Research Authors
# Licensed under the Apache License, Version 2.0
#   (see LICENSES/LICENSE-VCT-APACHE-2.0.txt)
# -----------------------------------------------------------------------------


"""Auxiliary layers."""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def make_deconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def init_weights_truncated_normal(m: nn.Module) -> None:
    """
    Initialise weights with truncated normal.
    Weights that fall outside 2 stds are resampled.
    See torch.nn.init.trunc_normal_ for details.

    Args:
        m: weights

    Examples:
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights_truncated_normal)
    """
    std = 0.02
    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
        m.bias.data.fill_(0.01)
    else:
        raise ValueError(
            f"Unsupported layer type: {type(m)}. Only nn.Linear is supported."
        )


def make_embedding_layer(num_channels: int, d_model: int) -> nn.Linear:
    """
    Create a linear layer mapping num_channels to d_model with uniform init.

    :param num_channels: Size of input features (channels).
    :param d_model: Size of output features (model dimension).
    :return: nn.Linear layer with weights and bias initialized uniformly in
        [-1/sqrt(num_channels), 1/sqrt(num_channels)].
    """
    # (batch, ..., num_channels)
    layer = nn.Linear(num_channels, d_model, bias=True)
    bound = 1.0 / math.sqrt(num_channels)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)
    return layer


class StartSym(nn.Module):
    """
    Learnable start symbol prepender: shifts input and prefixes a learned token.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.sym: nn.Parameter = nn.Parameter(torch.empty(num_channels))
        nn.init.uniform_(self.sym.data, -3.0, 3.0)

    @staticmethod
    def _shift_to_the_right(x: Tensor, pad: Optional[Tensor] = None) -> Tensor:
        """
        Shift the input sequence to the right by one time-step, prepending `pad`.

        :param x: Tensor of shape (batch, seq_len, channels).
        :param pad: Optional Tensor of shape (batch, 1, channels) to use as the first token.
            If None, uses zeros of the appropriate shape.
        :return: Tensor of shape (batch, seq_len, channels) where output[:, 0, :] = pad[:, 0, :]
            and output[:, 1:, :] = x[:, :-1, :].
        """
        *dims, _, channels = x.shape
        expected_pad_shape = (*tuple(dims), 1, channels)
        if pad is None:
            pad = x.new_zeros(expected_pad_shape)
        elif pad.shape != expected_pad_shape:
            raise ValueError(
                f"Invalid pad shape: {pad.shape}, expected {expected_pad_shape}"
            )
        return torch.cat([pad, x[..., :-1, :]], dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Prefixes `x` with the learned start symbol.

        :param x: Tensor of shape (batch, *, channels).
        :return: Tensor of shape (batch, *, channels) with first token replaced by sym.
        """
        batch, *_, channels = x.shape

        pad = self.sym.view(1, 1, channels).expand(batch, 1, channels)
        return self._shift_to_the_right(x, pad=pad)


class LearnedPosition(nn.Module):
    """
    Single learned positional encoding.
    """

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._d_model = d_model
        stddev = 0.02
        self._pos: nn.Parameter = nn.Parameter(
            torch.empty(1, seq_len, d_model, dtype=torch.float32).normal_(
                std=stddev
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional embeddings to x.

        :param x: Tensor of shape (batch, seq_len, d_model).
        :return: Tensor of same shape with positional embeddings added.
        """
        expected_shape = (self._seq_len, self._d_model)
        if x.shape[-2:] != expected_shape:
            raise ValueError(
                f"Invalid input shape {x.shape[-2:]}, expected {expected_shape}"
            )
        return x + self._pos


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualUnit(nn.Module):
    """Simple residual unit."""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(N, N // 2),
            nn.ReLU(inplace=True),
            conv3x3(N // 2, N // 2),
            nn.ReLU(inplace=True),
            conv1x1(N // 2, N),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out  # type: ignore[no-any-return]


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        self.conv_a = nn.Sequential(
            ResidualUnit(N), ResidualUnit(N), ResidualUnit(N)
        )

        self.conv_b = nn.Sequential(
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out  # type: ignore[no-any-return]
