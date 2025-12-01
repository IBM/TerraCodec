"""
Copyright IBM Corp. 2025
License: Apache-2.0

Adapted from CompressAI (https://github.com/InterDigitalInc/CompressAI)
Copyright (c) 2021-2025, InterDigital Communications, Inc
Licensed under BSD 3-Clause License: https://opensource.org/licenses/BSD-3-Clause

Modified to allow for multispectral and standardised data.
"""

import torch.nn as nn
from compressai.layers import AttentionBlock
from compressai.models import Elic2022Chandelier as CompressAIElic
from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv


class ELIC(CompressAIElic):
    """
    TerraCodec ELIC model.
    Adapts CompressAI's Elic2022Chandelier to:
    - Handle multispectral inputs via in_channels
    - Remove clamping to [0,1] during decompression for standardized data
    - Simplify forward return (only x_hat)
    - Return computed bits for compress()
    """

    def __init__(
        self, in_channels: int, N: int, M: int, image_size: int, **kwargs
    ):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(in_channels, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, in_channels, kernel_size=5, stride=2),
        )

    def forward(self, x):
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)

        return x_hat

    def compress(self, x):
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)
        outputs["bits"] = sum(len(s[0]) * 8 for s in outputs["strings"])

        return outputs

    def decompress(self, strings: list, shape: tuple, **kwargs) -> dict:
        """
        Decompress latent representation and reconstruct image.
        Removed clamping to [0,1] for standardized data and returning only x_hat.
        """
        y_out = self.latent_codec.decompress(strings, shape, **kwargs)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)

        return x_hat
