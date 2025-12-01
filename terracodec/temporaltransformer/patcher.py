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


"""Helper class for patching and unpatching."""

import math
from typing import Literal, NamedTuple, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn


class Patched(NamedTuple):
    """Represents a patched tensor.

    Attributes:
      tensor: The patched tensor, shape (b * nH * nW, d, patch_size ** 2)
      num_patches: Tuple (nH, nW) indicating number of patches per batch.
    """

    tensor: Tensor
    num_patches: Tuple[int, int]


class Patcher(nn.Module):
    """Helper class for patching and unpatching."""

    def __init__(
        self,
        stride: int,
        enable_latent_repacking: bool = False,
        pad_mode: Literal[
            "CONSTANT",
            "constant",
            "REFLECT",
            "reflect",
            "SYMMETRIC",
            "symmetric",
        ] = "REFLECT",
    ) -> None:
        """Initializes the patch helper."""
        super().__init__()
        self.stride = stride
        self.pad_mode = pad_mode
        self.latent_repacking = enable_latent_repacking

    def _reflect_pad(self, tensor: Tensor, target_factor: int) -> Tensor:
        """
        PyTorch version of reflect padding to make height/width multiples of target_factor.

        :param tensor: Tensor of shape (B, C, H, W).
        :param target_factor: The factor to pad height and width to multiples of.
        :return: Padded tensor of shape (B, C, H_padded, W_padded).
        """
        assert tensor.dim() == 4, (
            f"Expected tensor of shape (B, C, H, W), got {tensor.shape}"
        )
        H, W = tensor.shape[-2:]
        H_padded = math.ceil(H / target_factor) * target_factor
        W_padded = math.ceil(W / target_factor) * target_factor
        pad_h = H_padded - H
        pad_w = W_padded - W
        # pad format: (pad_left, pad_right, pad_top, pad_bottom)
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    def _latent_repacking_repack(self, x: Tensor) -> Tensor:
        """
        Transpose channel slices between the `T = window_size**2` tokens of every block.

        * Forward  (patch  → latent_repacking layout)
        * Inverse  (latent_repacking → original layout) - the operation is self-inverse.

        Expects `x` shaped (B_blk, T, C) where C = T·k.
        """
        if not self.latent_repacking:
            return x
        B_blk, T, C = x.shape
        if C % T:
            raise ValueError(
                f"C ({C}) must be divisible by T ({T}) for latent_repacking."
            )
        k = C // T
        x = x.view(B_blk, T, T, k)  # (block, orig_tok, slice_id, k)
        x = x.permute(0, 2, 1, 3)  # swap orig_tok  ↔  slice_id
        return x.contiguous().view(B_blk, T, C)  # (block, new_tok, C)

    def _window_partition(
        self, tensor: Tensor, window_size: int, pad: bool = True
    ) -> Tensor:
        """
        PyTorch version: partition tensor into non-overlapping windows.

        :param tensor: Tensor of shape (B, C, H, W).
        :param window_size: size of each window.
        :param pad: whether to pad via reflect to make dims divisible.
        :return: Tensor of shape (B * nH * nW, window_size * window_size, C).
        """
        B, C, H, W = tensor.shape
        if H % window_size != 0 or W % window_size != 0:
            if not pad:
                raise ValueError(
                    f"Feature map sizes {(H, W)} not divisible "
                    f"by window size {window_size}."
                )
            tensor = self._reflect_pad(tensor, window_size)
            # tensor.shape = (B, C, H_padded, W_padded)
            H, W = tensor.shape[-2:]

        # reshape to (B, C, nH, window_size, nW, window_size)
        nH = H // window_size
        nW = W // window_size
        t = tensor.view(B, C, nH, window_size, nW, window_size)
        # permute to (B, nH, nW, window_size, window_size, C)
        # t = torch.einsum("bhiwjc->bhwijc", t)

        # Faster than einsum
        t = t.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nH, nW, ws, ws, C)
        patches = t.view(B * nH * nW, window_size * window_size, C)

        # latent_repacking latent repacking
        patches = self._latent_repacking_repack(patches)
        return patches

    def _extract_overlapping_patches(
        self, tensor: Tensor, size: int, stride: int = 1, padding: str = "VALID"
    ) -> Tensor:
        """
        PyTorch version using unfold to extract sliding patches.

        :param tensor: Tensor of shape (B, C, H, W).
        :param size: patch size.
        :param stride: patch stride.
        :param padding: 'VALID' or 'SAME'
        :return: Tensor of shape (B * nH * nW, size * size, C).
        """
        B, C, H, W = tensor.shape
        # Pad if needed
        if padding.upper() == "SAME":
            pad_h = max((math.ceil(H / stride) - 1) * stride + size - H, 0)
            pad_w = max((math.ceil(W / stride) - 1) * stride + size - W, 0)
            tensor = F.pad(
                tensor,
                (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ),  # by default, "constant" padding with 0
            )
        # Unfold
        patches = tensor.unfold(2, size, stride).unfold(3, size, stride)
        if padding.upper() == "SAME":
            assert patches.shape == (B, C, H // stride, W // stride, size, size)
        else:
            assert patches.shape == (
                B,
                C,
                (H - size) // stride + 1,
                (W - size) // stride + 1,
                size,
                size,
            )
        # patches: (B, C, nH, nW, size, size)
        nH = patches.size(2)
        nW = patches.size(3)

        patches = (
            patches.reshape(B, C, nH, nW, size * size)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .reshape(B * nH * nW, size * size, C)
        )

        # latent_repacking latent repacking
        patches = self._latent_repacking_repack(patches)
        return patches

    def _get_num_patches(
        self, H: int, W: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Returns the number of patches in H and W dimensions.

        :param H: Height of the input tensor.
        :param W: Width of the input tensor.
        :return: Tuple of (nH, nW) indicating number of patches in H and
            W dimensions, and (H_padded, W_padded) indicating the
            padded H and W.
        """
        # Initial pad to get all strides in.
        H_padded = math.ceil(H / self.stride) * self.stride
        W_padded = math.ceil(W / self.stride) * self.stride
        nH = H_padded // self.stride
        nW = W_padded // self.stride
        return (nH, nW), (H_padded, W_padded)

    def _pad(self, x: Tensor, patch_size: int) -> Tuple[Tensor, int, int]:
        """Pads `x` such that we can do VALID patch extraction.

        :param x: Tensor to be padded, shape [B, C, H, W].
        :param patch_size: Size of the patches to be extracted.
        :return: Tuple of padded tensor and number of patches.
        """
        if patch_size < self.stride:
            raise ValueError("`patch_size` must be greater than `stride`!")
        missing = patch_size - self.stride
        if missing % 2 != 0:
            raise ValueError("Can only handle even missing pixels.")

        H, W = x.shape[-2:]
        (nH, nW), (H_padded, W_padded) = self._get_num_patches(H, W)
        # compute padding
        pad_h = H_padded - H + missing
        pad_w = W_padded - W + missing
        # padding sequence (left,right,top,bottom) for F.pad
        left = missing // 2
        right = pad_w - left
        top = missing // 2
        bottom = pad_h - top
        return (
            F.pad(x, (left, right, top, bottom), mode=self.pad_mode),
            nH,
            nW,
        )

    def forward(self, t: Tensor, patch_size: int) -> Patched:
        """Pads and extracts patches, shape (b * num_patches, size ** 2, d).

        :param t: Tensor to be patched, shape [B, C, H, W]
        :param patch_size: Size of the patches to be extracted.
        :return: Patched object containing the patches and number of patches.
        """
        # Pad to allow valid extract
        t_padded, nH, nW = self._pad(t, patch_size)
        # extract_patches is the PyTorch version imported above
        patches = (
            self._window_partition(t_padded, patch_size, pad=False)
            if patch_size == self.stride
            else self._extract_overlapping_patches(
                t_padded, patch_size, self.stride, padding="VALID"
            )
        )
        return Patched(patches, (nH, nW))

    def unpatch(
        self,
        t: Tensor,
        nH: int,
        nW: int,
        crop: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """Goes back to [B, C, H, W].

        :param t: Tensor to be unpatched, shape (b * num_patches, size ** 2, d)
        :param nH: Number of patches in H dimension.
        :param nW: Number of patches in W dimension.
        :param crop: Optional tuple (h, w) to crop the output tensor.
        :return: Unpatched tensor, shape [B, C, H, W].
        """
        # Inverse latent_repacking repacking **before** spatial reconstruction
        t = self._latent_repacking_repack(t)

        _, seq_len, C = t.shape
        if seq_len != self.stride**2:
            raise ValueError("Sequence length does not match stride^2")
        # reshape to (B, nH, nW, stride, stride, C)
        t = t.view(-1, nH, nW, self.stride, self.stride, C)
        # permute to (B, C, nH, stride, nW, stride)
        t = t.permute(0, 5, 1, 3, 2, 4).contiguous()
        # merge to spatial dims
        t = t.reshape(-1, C, nH * self.stride, nW * self.stride)
        if crop:
            h, w = crop
            return t[..., :h, :w]  # [B, C, crop_h, crop_w]
        return t
