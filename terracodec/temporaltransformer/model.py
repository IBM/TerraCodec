# Copyright IBM Corp. 2025
# License: Apache-2.0

"""
TerraCodecTT Codec wrapper for benchmarking against traditional codecs.

This module provides a TerraCodecTTCodec class that wraps the TerraCodecTT model to match
the same interface as traditional codecs for unified benchmarking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from terracodec.temporaltransformer.transforms import ELICTransform

from .pipeline import TerraCodecTTPipeline


class TemporalTransformer(nn.Module):
    """TerraCodecTT (TemporalTransformer) codec wrapper."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        flextec: bool = False,
        image_size: int = 256,
        in_channels: int = 12,
    ) -> None:
        super().__init__()
        self.device = device
        self.flextec = flextec
        self.img_dims = (image_size, image_size)
        self.model = TerraCodecTTPipeline(
            transform=ELICTransform(
                in_channels=in_channels,  # e.g., 12 for S2-L2A
                out_channels=in_channels,
                compression_channels=192,  # fixed for TerraCodecTT
            ),
            context_len=2,  # fixed for TerraCodecTT
            enable_latent_repacking=flextec,  # latent repacking only for FlexTEC
            enable_masking=flextec,  # masking only for FlexTEC
            fixed_image_hw=self.img_dims,
        )

    def setup_model_for_compression(self) -> None:
        """
        Setup the model for compression: enable compression and build entropy tables.
        """
        assert self.model is not None, "Model is not loaded."

        self.model.eval()

        # [NOTE: we would always assume latent_repacking is enabled too]
        if self.flextec:
            print("Enabling variable-rate compression in FlexTEC model.")
            self.model.enable_masked()

        # ---- Build/update entropy tables ON CPU exactly once
        self.model.enable_compression(force_update=True, test_mode=True)

        # ---- Move everything to target device
        self.model.to(self.device)

    # helper to split per-frame bit budget evenly across C bands without losing remainder
    def split_bits_per_band(self, bits: int, C: int) -> List[int]:
        q, r = divmod(int(bits), C)
        return [q + (1 if b < r else 0) for b in range(C)]

    @torch.inference_mode()
    def forward(
        self,
        input: torch.Tensor,
        quality: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Compress and decompress the input S2-L2A EO sequence tensor.

        NOTE: we assume already normalized and correctly cropped input tensor.

        :param input: Input tensor of shape (T, C, H, W) or (1, T, C, H, W),
            or (C, H, W) for single frame (T=1).
        :param quality: Compression quality (not used in TerraCodecTT)
            Used solely for FlexTEC.
        :return: Tuple of (reconstructed tensor, bits per frame)
        """
        assert self.model is not None, "Could not load TerraCodecTT model"

        self.model._code_to_strings = (
            False  # disable range coding inside forward
        )
        # Only accept following shapes:
        if input.dim() == 4:
            input = input.unsqueeze(0)  # add batch dim
        elif input.dim() == 3:
            input = input.unsqueeze(0).unsqueeze(0)  # add batch and time dim
        B, T, C, H, W = input.shape

        # Move to device
        input = input.to(self.device)

        with torch.autocast("cuda", enabled=False):  # forces fp32 inside
            output = self.model(
                input,
                enable_masking=self.flextec,
                k=quality if self.flextec else None,
            )

        # Enable range coding back and return
        self.model._code_to_strings = True
        return output

    @torch.inference_mode()
    def run(
        self,
        input: torch.Tensor,
        quality: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Compress and decompress the input S2-L2A EO sequence tensor.

        NOTE: we assume already normalized and correctly cropped input tensor.

        :param input: Input tensor of shape (T, C, H, W) or (1, T, C, H, W),
            or (C, H, W) for single frame (T=1).
        :param quality: Compression quality (not used in TerraCodecTT)
            Used solely for FlexTEC.
        :return: Tuple of (reconstructed tensor, bits per frame)
        """
        if self.flextec:
            assert quality is not None, "Quality must be provided for FlexTEC."
            return self.run_flextec(
                input,
                quality=quality,
            )

        assert self.model is not None, "Could not load TerraCodecTT model"

        # Only accept following shapes:
        if input.dim() == 4:
            input = input.unsqueeze(0)  # add batch dim
        elif input.dim() == 3:
            input = input.unsqueeze(0).unsqueeze(0)  # add batch and time dim
        B, T, C, H, W = input.shape
        assert B == 1, "Batch size > 1 not supported"

        # Check img_dims matches input H, W
        assert (H, W) == self.img_dims, (
            f"Input H, W {H, W} does not match expected {self.img_dims}"
        )

        # Move to device
        input = input.to(self.device)

        with torch.autocast("cuda", enabled=False):  # forces fp32 inside
            # Standard TerraCodecTT compression (encoding)
            *bottleneck_args, bits_per_frame = self.model.compress_eo_sequence(
                input,
            )

            # Decompression (decoding)
            recon = self.model.decompress_eo_sequence(
                input.shape,
                bottleneck_args=bottleneck_args,
                force_cpu=False,
            )

        return recon, bits_per_frame

    @torch.inference_mode()
    def compress(
        self,
        input: torch.Tensor,
        quality: Optional[Union[int, List[int]]] = None,
        forecast_mode: str = "deterministic",
    ) -> Dict[str, Any]:
        """
        Compress an entire sequence and return true bitstreams and metadata.

        NOTE: if self.flextec, then quality must be provided,
        and variable-rate compression will be used.

        Args:
            input: Tensor with shape (T, C, H, W) or (1, T, C, H, W).
            quality: int or list[int] (per-frame k), each in [0..seq_len_dec].
            forecast_mode: "deterministic" or "stochastic" for dropped tokens.

        Returns:
            A payload dict that can be fed to `decompress` to reconstruct.
        """
        assert self.model is not None

        # Normalize shape to [1, T, C, H, W]
        if input.dim() == 4:
            input = input.unsqueeze(0)
        elif input.dim() == 3:
            input = input.unsqueeze(0).unsqueeze(0)

        B, T, C, H, W = input.shape
        assert B == 1, "Batch size > 1 not supported"
        assert (H, W) == self.img_dims, (
            f"Input H, W {(H, W)} does not match expected {self.img_dims}"
        )

        # Move to device and ensure tables are built
        input = input.to(self.device)

        if not self.flextec:
            with torch.autocast("cuda", enabled=False):
                payload = self.model.compress(input)
        else:
            assert quality is not None, "Quality must be provided for FlexTEC."
            with torch.autocast("cuda", enabled=False):
                payload = self.model.compress_flextec(
                    eo_sequence=input,
                    quality=quality,
                    forecast_mode=forecast_mode,
                )
        return payload

    @torch.inference_mode()
    def decompress(self, payload: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress a payload previously returned by `compress`.
        Returns [1, T, C, H, W].
        """
        assert self.model is not None

        # Ensure we are on the right device
        self.model.to(self.device)

        if not self.flextec:
            with torch.autocast("cuda", enabled=False):
                recon = self.model.decompress(payload)
        else:
            with torch.autocast("cuda", enabled=False):
                recon = self.model.decompress_flextec(payload)

        return recon

    @torch.inference_mode()
    def run_flextec(
        self,
        input: torch.Tensor,
        quality: Union[int, List[int]],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Compress and decompress the input S2-L2A EO sequence tensor using FlexTEC.

        :param input: Normalized S2-L2A tensor. Allowed shapes:
            (B, T, C, H, W) with B=1 or (T, C, H, W) or (C, H, W) for single frame (T=1).
        :param quality: Compression quality in [1, 16].
        :return: Tuple of (reconstructed tensor, bits per frame)
        """
        assert self.model is not None, "Could not load FlexTEC model"

        # Only accept following shapes:
        if input.dim() == 4:
            input = input.unsqueeze(0)  # add batch dim
        elif input.dim() == 3:
            input = input.unsqueeze(0).unsqueeze(0)  # add batch and time dim
        B, T, C, H, W = input.shape
        assert B == 1, "Batch size > 1 not supported"

        # Check img_dims matches input H, W
        assert (H, W) == self.img_dims, (
            f"Input H, W {H, W} does not match expected {self.img_dims}"
        )

        # Move to device
        input = input.to(self.device)

        if isinstance(quality, int):
            assert (
                quality >= 0 and quality < self.model.entropy_model.seq_len_enc
            ), (
                f"Quality {quality} must be in [0, "
                f"{self.model.entropy_model.seq_len_enc - 1}] for FlexTEC."
            )
        elif isinstance(quality, list):
            assert len(quality) <= T, (
                f"Length of quality list must be <= number of frames ({T})"
            )
            for k in quality:
                assert k >= 0 and k < self.model.entropy_model.seq_len_enc, (
                    f"Quality {k} must be in [0, "
                    f"{self.model.entropy_model.seq_len_enc - 1}] for FlexTEC."
                )

        ks = [quality]
        with torch.autocast("cuda", enabled=False):  # forces fp32 inside
            # TerraCodecTT compression with masking
            recon_frames, rate_args = (
                self.model.multi_k_variable_rate_compression(
                    eo_sequence=input,
                    ks=ks,  # quality levels
                )
            )

            recon = recon_frames[0]  # [B, T, C, H, W]
            bits_per_frame = rate_args[0]  # [T]

        return recon, bits_per_frame
