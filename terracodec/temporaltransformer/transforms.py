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

## This module contains modules implementing
# standard synthesis and analysis transforms.

import abc
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .layers_utils import AttentionBlock, make_conv, make_deconv


class Transform(nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def compression_channels(self) -> int:
        pass

    @abc.abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def decode(self, x: Tensor, frames_shape: torch.Size) -> Tensor:
        pass


class ResidualUnit(nn.Module):
    """Simple residual unit"""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            make_conv(N, N // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            make_conv(N // 2, N // 2, kernel_size=3),
            nn.ReLU(inplace=True),
            make_conv(N // 2, N, kernel_size=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out: Tensor = self.conv(x)
        out += identity
        out = self.activation(out)
        return out


class ELICTransform(Transform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_blocks: int = 3,
        channels: List[int] = [128, 160, 192, 192],
        compression_channels: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.analysis_transform = ELICAnalysis(
            in_channels=in_channels,
            num_residual_blocks=num_residual_blocks,
            channels=channels,
            compression_channels=compression_channels,
            max_frames=max_frames,
        )
        synthesis_channels = list(reversed(channels))
        self.synthesis_transform = ELICSynthesis(
            out_channels=out_channels,
            num_residual_blocks=num_residual_blocks,
            channels=synthesis_channels,
            max_frames=max_frames,
        )

    @property
    def compression_channels(self) -> int:
        return self.analysis_transform.compression_channels

    def encode(self, x: Tensor) -> Tensor:
        return self.analysis_transform(x)  # type: ignore

    def decode(self, x: Tensor, frames_shape: torch.Size) -> Tensor:
        return self.synthesis_transform(x, frames_shape)  # type: ignore


class ELICAnalysis(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_residual_blocks: int = 3,
        channels: List[int] = [128, 160, 192, 192],
        compression_channels: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """Analysis transform from ELIC (https://arxiv.org/abs/2203.10886), which
        can be configured to match the one from "Devil's in the Details"
        (https://arxiv.org/abs/2203.08450).

        Args:
            in_channels: number of input channels.
            num_residual_blocks: defaults to 3.
            channels: defaults to [128, 160, 192, 192].
            compression_channels: optional, defaults to None. If provided, it must equal
                the last element of `channels`.
            max_frames: optional, defaults to None. If provided, the input is chunked
                into max_frames elements, otherwise the entire batch is processed at
                once. This is useful when large sequences are to be processed and can
                be used to manage memory a bit better.
        """
        super().__init__()
        self._in_channels = in_channels
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {len(channels)}).")
        if (
            compression_channels is not None
            and compression_channels != channels[-1]
        ):
            raise ValueError(
                "output_channels specified but does not match channels: "
                f"{compression_channels} vs. {channels}"
            )
        self._compression_channels = (
            compression_channels
            if compression_channels is not None
            else channels[-1]
        )
        self._max_frames = max_frames

        def res_units(N: int) -> List[nn.Module]:
            """Creates a list of residual units."""
            return [ResidualUnit(N) for _ in range(num_residual_blocks)]

        channels = [self._in_channels] + channels

        self.transforms = nn.Sequential(
            make_conv(
                channels[0], channels[1], kernel_size=5, stride=2
            ),  # in_chan,256,256->128,128,128
            *res_units(channels[1]),
            make_conv(
                channels[1], channels[2], kernel_size=5, stride=2
            ),  # 128,128,128->160,64,64
            *res_units(channels[2]),
            AttentionBlock(channels[2]),
            make_conv(
                channels[2], channels[3], kernel_size=5, stride=2
            ),  # 160,64,64->192,32,32
            *res_units(channels[3]),
            make_conv(
                channels[3], channels[4], kernel_size=5, stride=2
            ),  # 192,32,32->192,16,16
            AttentionBlock(channels[4]),
        )

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 5, f"Expected [B, T, C, H, W] got {x.shape}"
        B, T, C, H, W = x.shape
        assert C == self._in_channels, (
            f"Expected {self._in_channels} channels, got {C}"
        )
        x = x.reshape(B * T, C, H, W)
        if self._max_frames is not None and B * T > self._max_frames:
            assert (B * T) % self._max_frames == 0, "Can't reshape!"
            f = (B * T) // self._max_frames
            x = x.reshape(f, self._max_frames, *x.shape[1:])
            x = torch.stack([self.transforms(chunk) for chunk in x], dim=0)
            x = torch.flatten(x, start_dim=0, end_dim=1)
        else:
            x = self.transforms(x)
        return x.reshape(B, T, *x.shape[1:])


class ELICSynthesis(nn.Module):
    def __init__(
        self,
        out_channels: int,
        num_residual_blocks: int = 3,
        channels: List[int] = [192, 192, 160, 128],
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Synthesis transform from ELIC (https://arxiv.org/abs/2203.10886).

        Args:
            out_channels: number of output channels (shall match the number
                of input channels of the analysis transform).
            num_residual_blocks: defaults to 3.
            channels: _defaults to [192, 160, 128, 3].
            max_frames: optional, defaults to None. If provided, the input is chunked
                into max_frames elements, otherwise the entire batch is processed at
                once. This is useful when large sequences are to be processed and can
                be used to manage memory a bit better.
        """
        super().__init__()
        self._out_channels = out_channels
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")

        self._compression_channels = channels[0]
        self._max_frames = max_frames
        channels = channels + [self._out_channels]

        def res_units(N: int) -> List:
            return [ResidualUnit(N) for _ in range(num_residual_blocks)]

        self.transforms = nn.Sequential(
            AttentionBlock(channels[0]),
            make_deconv(
                channels[0], out_channels=channels[1], kernel_size=5, stride=2
            ),
            *res_units(channels[1]),
            make_deconv(
                channels[1], out_channels=channels[2], kernel_size=5, stride=2
            ),
            AttentionBlock(channels[2]),
            *res_units(channels[2]),
            make_deconv(
                channels[2], out_channels=channels[3], kernel_size=5, stride=2
            ),
            *res_units(channels[3]),
            make_deconv(
                channels[3], out_channels=channels[4], kernel_size=5, stride=2
            ),
        )

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor, frames_shape: torch.Size) -> Tensor:
        """
        Args:
            x: the (reconstructed) latent embdeddings to be decoded to images.
            frames_shape: shape of the sequence to be reconstructed.
        Returns:
            reconstruction: reconstruction of the original sequence with shape
                [B, T, C, H, W] = frames_shape.
        """

        assert x.dim() == 5, f"Expected [B, T, C, H, W] got {x.shape}"
        B, T, C, H, W = x.shape
        assert C == self._compression_channels, (
            f"Expected {self._compression_channels} channels, got {C}"
        )
        x = x.reshape(B * T, C, H, W)
        if self._max_frames is not None and B * T > self._max_frames:
            assert (B * T) % self._max_frames == 0, "Can't reshape!"
            f = (B * T) // self._max_frames
            x = x.reshape(f, self._max_frames, C, H, W)
            x = torch.stack([self.transforms(chunk) for chunk in x], dim=0)
            x = torch.flatten(x, start_dim=0, end_dim=1)
        else:
            x = self.transforms(x)

        x = x.reshape(B, T, *x.shape[1:])
        return x[..., : frames_shape[-2], : frames_shape[-1]]
