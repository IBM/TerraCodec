"""
Copyright IBM Corp. 2025
License: Apache-2.0

Adapted from CompressAI (https://github.com/InterDigitalInc/CompressAI)
Copyright (c) 2021-2025, InterDigital Communications, Inc
Licensed under BSD 3-Clause License: https://opensource.org/licenses/BSD-3-Clause

Modified to allow for multispectral and standardised data.
"""

import torch.nn as nn
from compressai.layers import GDN
from compressai.models import FactorizedPrior as CompressAIFactorizedPrior
from compressai.models.utils import conv, deconv


class FactorizedPrior(CompressAIFactorizedPrior):
    """
    TerraCodec Factorized Prior model.
    Adapts CompressAI's FactorizedPrior to:
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
            conv(in_channels, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(self.N, inverse=True),
            deconv(self.N, in_channels),
        )

        self.in_channels = in_channels
        self.N = N
        self.M = M
        self.image_size = image_size

    def forward(self, x):
        y = self.g_a(x)
        y_hat, _ = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return x_hat

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        bits = sum(len(s[0]) * 8 for s in [y_strings])

        return {"strings": [y_strings], "bits": bits, "shape": y.size()[-2:]}

    def decompress(self, strings: list, shape: tuple, **kwargs) -> dict:
        """
        Decompress latent representation and reconstruct image.
        Removed clamping to [0,1] for standardized data and returning only x_hat.
        """
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat)
        return x_hat
