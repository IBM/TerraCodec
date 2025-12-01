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

import base64
import itertools
import random
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor

from .entropy_model import (
    PreviousLatent,
    TemporalEntropyModelOut,
    TerraCodecTTEntropyModel,
)
from .transforms import Transform

State = Tuple[PreviousLatent, ...]


class Bottleneck(NamedTuple):
    """
    Bottleneck of a single latent frame (frame)

    Args:
        latent_q: quantized latent, expected shape is [B, C, H, W]
            actual "compressed" representation that will either get sent through
            the entropy coder, or directly to the dequantizer on decoding.
        likelihood: likelihood/bits for the quantized latent, expected shape is
            [B', seq_len_dec, C] (no need to unpatch)
        z_cur: z_cur, optional, output of the entropy model,
            expected shape [B, d_model, H, W]

        -> by default, C=192, d_model=C*4=768
    """

    latent_q: Tensor
    likelihood: Tensor  # likelihood or bits
    masked_latent: Optional[Tensor] = None  # masked latent for synthesis
    z_cur: Optional[Tensor] = None
    masked_z_cur: Optional[Tensor] = None  # masked z_cur for synthesis
    mask: Optional[Tensor] = None  # mask used in variable token usage


class NetworkOut(NamedTuple):
    """Output of the entropy model decoder (._encode_and_decode_frames)

    Args:
        reconstruction: reconstruction of the latent,
            expected shape [B, C_emb, H_emb, W_emb]
        likelihood: likelihood/bits of the reconstruction, expected shape
            [B', seq_len_dec, C] (no need to unpatch)
        -> by default, C=192
        mask: Optional mask used in variable token usage, expected shape [B', seq_len_dec]
    """

    reconstruction: Tensor
    likelihood: Tensor
    mask: Optional[Tensor] = None  # mask used in variable token usage


class EncodeOut(NamedTuple):
    """Output of the entropy model encoder (.encode_frames)

    Args:
        bottleneck: Bottleneck object
    """

    bottleneck: Bottleneck


class PerChannelWeight(nn.Module):
    """Learn a weight per channel

    - A learnable 4-D tensor of shape [1, C, 1, 1].
        One scalar per latent channel.
    - On the very first (I-)frame, you have no real "previous" latent to feed
        into the temporal model. You need some stand-in.
    - PerChannelWeight.forward([B, C, H, W]) returns a full-sized fake-latent
        of shape [B, C, H, W], where channel-c everywhere is just that single
        weight [0,c,0,0].
    - This lets the entropy model get a "previous frame" of the correct shape
        without introducing arbitrary content. And because the per-channel weights
        are learnable, the network can discover the best "starting bias" for each
        channel.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.weight = nn.parameter.Parameter(
            torch.rand((1, num_channels, 1, 1))
        )

    def forward(
        self, latent_shape: Union[torch.Size, List[int], Tuple[int, ...]]
    ) -> Tensor:
        """
        Forward pass to generate per-channel weights.

        :param latent_shape: Shape tuple/list representing
            [B, C_emb, H_emb, W_emb].
        :return: Tensor of shape [B, C_emb, H_emb, W_emb] where each channel
            is weighted by a learnable scalar.
        """
        assert latent_shape[1] == self.weight.shape[1], (
            "channel length mismatch"
        )
        return self.weight * self.weight.new_ones(latent_shape)


class ResidualBlock(nn.Module):
    """Standard residual block"""

    def __init__(self, filters: int, kernel_size: int) -> None:
        """
        Initialize a residual block with two convolutional layers
        and a skip connection.

        :param filters: Number of input/output channels.
        :param kernel_size: Size of the convolution kernels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.01)  # default value
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Apply the residual block to the input.

        :param inputs: Input tensor of shape [B, C_emb, H_emb, W_emb].
        :return: Output tensor of shape [B, C_emb, H_emb, W_emb], equal to
            inputs + transformed inputs.
        """
        output = self.activation(self.conv1(inputs))
        output = self.activation(self.conv2(output))
        return output + inputs  # type: ignore[no-any-return]


class Dequantizer(nn.Module):
    """
    Implement dequantization: feed y' = y + f(z) to the synthesis transform,
    where y is the latent and z is transformer/entropy model features.
    """

    def __init__(self, num_channels: int, d_model: int) -> None:
        """
        Construct a Dequantizer module that adds transformer features to the quantized latent.

        :param num_channels: Number of channels in the latent (C).
        :param d_model: Dimension of the transformer features.
        """
        super().__init__()
        self._d_model = d_model
        self._num_channels = num_channels
        self.process_conv = nn.Sequential(
            nn.Conv2d(
                d_model, num_channels, kernel_size=1
            ),  # d_model, num_channels
            # same to a Linear layer, but generally faster in PyTorch
            nn.LeakyReLU(negative_slope=0.01),
            ResidualBlock(num_channels, kernel_size=3),
        )

    def forward(
        self, *, latent_q: Tensor, entropy_features: Optional[Tensor] = None
    ) -> Tensor:
        """
        Dequantize by adding a learned function of transformer features
        to the quantized latent.

        :param latent_q: Quantized latent tensor of shape
            [B, C_emb, H_emb, W_emb].
        :param entropy_features: Optional transformer features of shape
            [B, d_model, H_emb, W_emb].
        :return: Tensor of shape [B, C_emb, H_emb, W_emb] ready
            for the synthesis transform.
        """
        if entropy_features is None:
            # Create fake features with 0s only -- technically, this never hits
            b, _, h, w = latent_q.shape
            entropy_features = latent_q.new_zeros((b, self._d_model, h, w))

        return latent_q + self.process_conv(entropy_features)  # type: ignore[no-any-return]


class TerraCodecTTPipeline(nn.Module):
    """
    Initialize the TerraCodecTT pipeline with analysis/synthesis transforms and entropy model.

    :param transform: An autoencoder-like module with `encode` and `decode` methods.
    :param compression_channels: Number of channels in the latent space.
    :param context_len: Number of past frames to condition on in the entropy model.
    :param latent_repacking: Whether to use Latent Repacking feature.
    :param enable_masking: Whether to enable masking in the entropy model.
    """

    def __init__(
        self,
        transform: Transform,
        context_len: int = 2,
        enable_latent_repacking: bool = False,
        enable_masking: bool = False,
        fixed_image_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize the TerraCodecTT pipeline with a transform and entropy model.

        :param transform: An autoencoder-like module with `encode` and `decode` methods.
        :param context_len: Number of past frames to condition on in the entropy model.
        :param enable_latent_repacking: Whether to use Latent Repacking feature.
        :param enable_masking: Whether to enable masking in the entropy model.
        """
        super().__init__()
        # Transforms
        self.transform = transform
        self._pad_factor = 16
        compression_channels = transform.compression_channels

        # Latent Repacking and Masking
        self.latent_repacking = enable_latent_repacking
        self.masking = enable_masking

        # Entropy model
        self.entropy_model = TerraCodecTTEntropyModel(
            num_channels=compression_channels,
            context_len=context_len,
            enable_latent_repacking=enable_latent_repacking,
            enable_masking=enable_masking,
        )
        self._temporal_pad_token_maker = PerChannelWeight(
            num_channels=compression_channels
        )
        self._dequantizer = Dequantizer(
            compression_channels, self.entropy_model.d_model
        )
        self._context_len = context_len
        self.avail_ks = self.entropy_model.avail_ks

        # Compression state management
        self._code_to_strings = False
        self._definitive_tables_initialized = False

        # Shape cache for decompression without payload metadata
        self._fixed_image_hw: Optional[Tuple[int, int]] = (
            (int(fixed_image_hw[0]), int(fixed_image_hw[1]))
            if fixed_image_hw is not None
            else None
        )
        self._cached_in_channels: Optional[int] = None
        self._cached_latent_hw: Optional[Tuple[int, int]] = None
        self._cached_latent_channels: Optional[int] = None

    def _ensure_shape_cache(self) -> None:
        """Infer and cache (once) the input and latent shapes for this model.

        - in_channels from analysis transform (if available)
        - image_hw from constructor (fixed) or default to (256, 256)
        - latent_hw and latent_channels from a tiny dry run through encode
        """
        if (
            self._cached_in_channels is not None
            and self._cached_latent_hw is not None
            and self._cached_latent_channels is not None
        ):
            return

        # Try to infer input channels from transform
        in_ch = getattr(
            getattr(self.transform, "analysis_transform", object()),
            "in_channels",
            None,
        )
        if in_ch is None:
            # Fallback: many TerraCodec configs use 12 bands for S2-L2A
            in_ch = 12
        self._cached_in_channels = int(in_ch)

        # Image size: fixed for this model (e.g., 256x256); default if not set
        if self._fixed_image_hw is None:
            self._fixed_image_hw = (256, 256)

        H, W = self._fixed_image_hw
        device = next(self.parameters()).device

        # Do a one-off encode dry-run to probe latent shape
        with torch.no_grad():
            probe = torch.zeros(
                (1, 1, self._cached_in_channels, H, W), device=device
            )
            lat = self.transform.encode(probe)  # [1,1,C_emb,H_emb,W_emb]
            _, _, C_emb, H_emb, W_emb = lat.shape
            self._cached_latent_hw = (int(H_emb), int(W_emb))
            self._cached_latent_channels = int(C_emb)

    def enable_masked(self) -> None:
        """
        Enable latent repacking and masking.
        Used for inference with masking after training.
        """
        self.latent_repacking = True
        self.masking = True
        self.entropy_model.enable_masked()

    def get_dec_len(self) -> int:
        """
        Get the number of tokens per patch in the latent space.

        :return: Number of tokens per patch in the latent space.
        """
        return self.entropy_model.get_dec_len()

    ## ENCODING FUNCTIONS
    def _encode_frame(
        self,
        frame: Tensor,
        frame_index: int,
        state: Optional[State],
        enable_masking: bool,
        k: Optional[int] = None,
    ) -> Tuple[State, EncodeOut]:
        """
        Encode a single frame (I-frame or P-frame).
        :param frame: Latent tensor of shape [B, C, H, W].
        :param frame_index: Index of this frame in the sequence.
        :param state: Optional tuple of PreviousLatent for conditioning. If None
                      treat as an I-frame.
        :param enable_masking: Whether to enable masking
            in the entropy model.
        :param k: Optional integer representing the k value to use for
            variable token usage (Masking).
        :return: Tuple (new_state, EncodeOut) after encoding this frame.
        """
        is_iframe = frame_index == 0
        assert not is_iframe or state is None, (
            "I-frame should not have a previous state."
        )

        if is_iframe:
            # I-frame: create a fake previous latent for conditioning.
            fake_previous_latent = self._temporal_pad_token_maker(frame.shape)
            processed = self.entropy_model.process_previous_latent_q(
                fake_previous_latent
            )  # [B', Wp x Wp, d_model] = [B * nH * nW, Wp * Wp, 768]
            previous_latents: State = (processed,)
        else:
            # P-frame: use the provided state.
            assert state is not None, "P-frame should have a previous state."
            previous_latents = state

        if not self.training and self._code_to_strings:
            # `compress` runs encoding and decoding
            run_decode = frame_index <= 1
            output: TemporalEntropyModelOut = self.entropy_model.compress(
                latent_unquantized=frame,
                previous_latents=previous_latents,
                # Since compression is lossless, decode only the first couple of latents
                # to check for errors, and skip decoding afterwards
                run_decode=run_decode,
                validate_causal=False,
            )
        else:
            # forward pass through the entropy model
            output = self.entropy_model(
                latent_unquantized=frame,
                previous_latents=previous_latents,
                enable_masking=enable_masking,
                k=k,
            )

        if enable_masking and self.masking:
            # Check that mask dim is 2, and shape is [B', seq_len_dec]
            assert (
                output.mask is not None
                and output.mask.dim() == 2
                and output.mask.shape[0] == output.bits.shape[0]
                and output.mask.shape[1] == output.bits.shape[1]
            ), (
                "Mask should be present when masking is enabled, "
                "and have shape [B', seq_len_dec]."
                f" Got mask shape: {output.mask.shape}, "  # type: ignore
                f"bits shape: {output.bits.shape}"
            )

            bottleneck = Bottleneck(
                latent_q=output.perturbed_latent,
                likelihood=output.bits,
                z_cur=output.features,
                masked_latent=output.masked_perturbed_latent,
                masked_z_cur=output.masked_features,
                mask=output.mask,  # Mask used for variable token usage
            )
        else:
            bottleneck = Bottleneck(
                latent_q=output.perturbed_latent,
                likelihood=output.bits,
                z_cur=output.features,
            )

        # NOTE: During training, even when masking enabled, state is updated
        # with the unmasked latent_q, therefore employing Teacher Forcing,
        # on the sequence level
        new_state = self._update_state(
            latent_q=bottleneck.latent_q, state=state, is_iframe=is_iframe
        )

        return new_state, EncodeOut(bottleneck)

    def _encode_frame_masking(
        self,
        frame: Tensor,
        frame_index: int,
        ks: List[int],
        states: Optional[List[State]],
        forecast_mode: str,
    ) -> Tuple[List[State], List[EncodeOut]]:
        """
        Encode a single frame (I-frame or P-frame) with masking.

        :param frame: Latent tensor of shape [B, C, H_emb, W_emb].
        :param frame_index: Index of this frame in the sequence.
        :param ks: List of k values for masking.
        :param states: List of optional tuple of PreviousLatent for conditioning.
            If None, treat as an I-frame.
            One state per k in ks.
        :param forecast_mode: Forecast mode for token filling. "deterministic"
            for mu-only, "stochastic" for sampling from the distribution.
        :return: List of tuples (new_state, EncodeOut) after encoding this frame
            for each k in ks.
        """
        is_iframe = frame_index == 0
        assert not is_iframe or states is None, (
            "I-frame should not have a previous state."
        )

        if is_iframe:
            # I-frame: create a fake previous latent for conditioning.
            fake_previous_latent = self._temporal_pad_token_maker(frame.shape)
            processed = self.entropy_model.process_previous_latent_q(
                fake_previous_latent
            )  # [B', Wp x Wp, d_model] = [B * nH * nW, Wp * Wp, 768]
            fake_previous_latents: State = (processed,)
            previous_latents = [fake_previous_latents] * len(ks)
        else:
            # P-frame: use the provided state.
            assert states is not None, "P-frame should have a previous state."
            previous_latents = states

        if not self.training and self._code_to_strings:
            assert len(ks) == 1, (
                "Compression with masking supports only a single k value."
            )
            k = ks[0]
            run_decode = frame_index <= 1
            output: TemporalEntropyModelOut = (
                self.entropy_model.compress_multi_k_masking(
                    latent_unquantized=frame,
                    previous_latents=previous_latents[0],
                    k=k,
                    run_decode=run_decode,
                    validate_causal=False,
                    forecast_mode=forecast_mode,
                )
            )
            outputs = [output]
        else:
            outputs = self.entropy_model.ent_multi_k_variable_rate_compression(
                latent_unquantized=frame,
                ks=ks,
                previous_latents=previous_latents,
            )

        bottlenecks = [
            Bottleneck(
                latent_q=output.perturbed_latent,  # [B, C_emb, H_emb, W_emb]
                likelihood=output.bits,  # [B * nH * nW, Wc * Wc, C_emb]
                z_cur=output.features,  # [B, d_model, H_emb,W_emb]
            )
            for output in outputs
        ]

        new_states = [
            self._update_state(
                latent_q=bottleneck.latent_q,
                state=states[k_index] if states else None,
                is_iframe=is_iframe,
            )
            for k_index, bottleneck in enumerate(bottlenecks)
        ]

        return new_states, [EncodeOut(bottleneck) for bottleneck in bottlenecks]

    def _encode_frames(
        self,
        sequence: torch.Tensor,
        enable_masking: bool,
        k: Optional[int] = None,
    ) -> Iterator[EncodeOut]:
        """
        Generator over sequence to encode I and P frames in order.

        :param sequence: torch.Tensor wrapping latents
            of shape [B, T, C_emb, H_emb, W_emb].
        :param enable_masking: Whether to enable masking
            in the entropy model.
        :param k: Optional integer representing the k value to use for
            variable token usage (Masking).
        :param cache: Memoization cache.
        :yield: EncodeOut for each frame.
        """
        state = None
        num_frames = sequence.shape[1]
        for index in range(num_frames):
            frame = sequence[:, index, ...]  # [B, C_emb, H_emb, W_emb]
            state, encode_out = self._encode_frame(
                frame,
                index,
                state,
                enable_masking,
                k=k,
            )
            yield encode_out

    # DECODING
    def _decode_bottleneck(
        self, bottleneck: Bottleneck, enable_masking: bool
    ) -> Tensor:
        """
        Decode a bottleneck: run dequantizer to get synthesis input.

        :param bottleneck: Bottleneck for this frame.
        :param enable_masking: Whether to enable masking
        :return: Tensor to be used as input for the synthesis transform.
        """
        if enable_masking and self.masking:
            assert (
                bottleneck.masked_latent is not None
                and bottleneck.masked_z_cur is not None
            ), (
                "Masked latent and z_cur should be present "
                "when masking is enabled."
            )
        latent_q = (
            bottleneck.masked_latent
            if (enable_masking and self.masking)
            else bottleneck.latent_q
        )
        entropy_features = (
            bottleneck.masked_z_cur
            if (enable_masking and self.masking)
            else bottleneck.z_cur
        )
        synthesis_in: torch.Tensor = self._dequantizer(
            latent_q=latent_q,
            entropy_features=entropy_features,
        )  # [B, C_emb, H_emb, W_emb]

        # NB: we apply the transforms (analysis and synthesis) in the
        # forward pass on the entire eo_sequence
        # so here we return synthesis_in instead of reconstruction
        # and apply image_synthesis

        return synthesis_in

    # All frames (frames)
    def _decode_frames(
        self,
        bottlenecks: Iterator[Bottleneck],
        enable_masking: bool = False,
    ) -> Iterator[Tensor]:
        """
        Generator to decode a stream of bottlenecks into synthesis inputs.

        :param bottlenecks: Iterator of Bottleneck objects.
        :param cache: Memoization cache.
        :yield: Tuples (synthesis_input, metrics) per frame.
        """

        for bottleneck in bottlenecks:
            frame_reconstruction = self._decode_bottleneck(
                bottleneck, enable_masking
            )
            yield frame_reconstruction

    def _encode_and_decode_frames(
        self,
        frames: torch.Tensor,
        enable_masking: bool,
        k: Optional[int] = None,
    ) -> Iterator[NetworkOut]:
        """
        End-to-end encode and decode of latent frames.

        :param frames: torch.Tensor wrapping latents shape [B, T, C_emb, H_emb, W_emb].
        :param enable_masking: Whether to enable masking in the entropy model.
        :param k: Optional integer representing the k value to use for
            variable token usage (Masking).
        :yield: NetworkOut containing reconstruction, likelihood, and metrics.
        """
        encode_outs = self._encode_frames(
            frames,
            enable_masking,
            k=k,
        )

        # Iterate over `encode_outs` twice: once to decode and once to construct
        # and yield the NetworkOut object
        encode_outs, encode_outs_tee = itertools.tee(encode_outs)
        reconstruction_frames = self._decode_frames(
            (encode_out.bottleneck for encode_out in encode_outs_tee),
            enable_masking=enable_masking,
        )

        for rec_frame, encode_out in zip(reconstruction_frames, encode_outs):
            # rec_frame is padded [B, C_emb, H_emb, W_emb]

            yield NetworkOut(
                reconstruction=rec_frame,
                likelihood=encode_out.bottleneck.likelihood,
                mask=encode_out.bottleneck.mask,
                # Mask used for variable token usage, shape: [B', seq_len_dec]
            )

    def _update_state(
        self,
        latent_q: Tensor,
        state: Optional[State] = None,
        is_iframe: bool = False,
    ) -> State:
        """Update the temporal state given a new quantized latent.

        Handles both I-frame (no prior state) and P-frame cases.
        """
        # Detach latent_q as it will serve as a previous latent for the
        # next frame, and we do not want to backprop through it.
        if is_iframe:
            assert state is None, "I-frame should not have a previous state."
            latent_q = latent_q.detach()
        else:
            assert state is not None, "P-frame should have a previous state."
            # P-frames are not detached, as they are used for the next frame
            # and we want to backprop through them

        next_state_entry = self.entropy_model.process_previous_latent_q(
            latent_q
        )

        if state is None:
            # I-frame: state is just the new entry
            new_state: State = (next_state_entry,)
        else:
            # P-frame: append and trim to context length
            new_state = (*state, next_state_entry)
            new_state = new_state[-self._context_len :]

        assert len(new_state) <= self._context_len
        return new_state

    ### FORWARD & HELPERS ###
    def _pad(
        self, x: Tensor, sizes: Sequence[int], factor: Optional[int] = None
    ) -> Tensor:
        """
        Reflect-pad input to a multiple of downscale factor.

        :param x: Tensor of shape [B, T, C, H, W].
        :param sizes: Sequence (H, W) before padding.
        :param factor: Optional multiple; defaults to total downscale.
        :return: Padded tensor [B, T, C, H+pad_h, W+pad_w].
        """
        if factor is None:
            n_im_downscale = getattr(
                self.transform, "num_downsampling_layers", 0
            )
            n_hyper_downscale = getattr(
                self.entropy_model, "num_downsampling_layers", 0
            )
            factor = 2 ** (n_im_downscale + n_hyper_downscale)

        pad_h, pad_w = [(factor - (s % factor)) % factor for s in sizes]  # type: ignore
        # dims are in reverse -- W, H, C, so the below pads:
        # width dimension by 0, pad_w, height by 0, pad_h and no padding for channel
        return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, 0), "reflect")

    def forward(
        self,
        eo_sequence: torch.Tensor,
        enable_masking: bool = False,
        k: Optional[int] = None,
    ) -> Tuple[Tensor, List[int]]:
        """
        Forward pass: analysis, entropy coding, decode, synthesis.

        :param eo_sequence: tensor of shape [B, T, C, H, W].
        :param enable_masking: Forward pass with masked tokens
        (FlexTEC simulation during training).
        :param k: Optional integer representing the k value to use for
            variable token usage (Masking).
        :return: Tuple where:
            - reconstructions: reconstructed frames after synthesis
                of shape [B, T, C, H, W]
            - bits: list of bits.
        """
        assert eo_sequence.dim() == 5, (
            f"Expected eo_sequence to have 5 dimensions ([B, T, C, H, W]), got {eo_sequence.dim()}"
        )
        if not self.training:
            frames_shape = eo_sequence.shape[-2:]  # (H, W)
            inputs = self._pad(eo_sequence, frames_shape)
        else:
            frames_shape = None
            inputs = eo_sequence

        frames = self.transform.encode(
            inputs
        )  # [B, T, C_emb, H_emb, W_emb] = [B, T, 192, H // 16, W // 16]

        if enable_masking and k is None:
            # Sample one same 'k' for all samples and frames in the batch
            k = random.choice(self.avail_ks)

        # code and decode in a differentiable way
        res = self._encode_and_decode_frames(
            frames,
            enable_masking,
            k=k,
        )  # yields NetworkOut

        rec_frames: List[Tensor] = []
        likelihoods: List[Tensor] = []
        masks: List[Optional[Tensor]] = []
        for r in res:
            rec_frames.append(r.reconstruction)
            likelihoods.append(r.likelihood)
            masks.append(r.mask)

        masks_th: Optional[Tensor] = None
        if enable_masking:
            # mask_th, of shape [T, B', seq_len_dec]
            # masks is a T-length list of tensors
            # of shape [B', seq_len_dec] = [B * nH * nW, Wc * Wc]
            assert all(m is not None for m in masks), (
                "Masks should not be None when masking is enabled."
            )
            masks_th = torch.stack(masks)  # type: ignore[arg-type]

        reconstructions_embeddings = torch.stack(
            rec_frames, dim=1
        )  # [B, T, C_emb, H_emb, W_emb]

        # likelihoods is a T-length list of tensors
        # of shape [B', seq_len_dec, C_emb] = [B * nH * nW, Wc * Wc, C_emb]
        likelihoods_th = torch.stack(likelihoods)
        # likelhoods.shape = [T, B', seq_len_dec, C_emb]

        reconstructions_frames = self.transform.decode(
            reconstructions_embeddings, eo_sequence.shape
        )
        # reconstructions_frames.shape = [B, T, C, H, W]
        if not self.training:
            assert frames_shape is not None, "frames_shape not found"
            h, w = frames_shape
            reconstructions_frames = reconstructions_frames[..., :h, :w]

        bits = self.compute_rate(likelihoods_th, per_frame=True, mask=masks_th)

        # Cast to list of int
        bits_list = []
        for b in bits:
            bit_count = float(b.item())
            bits_list.append(int(bit_count))

        return reconstructions_frames, bits_list

    def multi_k_variable_rate_compression(
        self,
        eo_sequence: torch.Tensor,
        ks: List[Union[List[int], int]],
        forecast_mode: str = "deterministic",
    ) -> Tuple[
        List[Tensor],
        List[List[int]],
    ]:
        """
        Multi-k masking inference for a sequence of frames.
        This method encodes each frame with masking for each k in ks,
        decodes the bottlenecks, and returns the reconstructions and likelihoods

        [NOTE]: self.masking does not have to be enabled (i.e. True).

        :param eo_sequence: torch.Tensor with .eo_sequence_tensor
            [B, T, C, H, W]
        :param ks: List of k values or lists of k values for masking.
            Each k can be an int (same k for all frames) or a list of ints
            (one k per frame). If a list is shorter than the number of frames,
            it will be extended by repeating the last k.

        :return: Tuple (reconstructions [B, T, C, H, W], [likelihoods]),
            for each k in ks.
        """
        assert not self.training, (
            "multi_k_variable_rate_compression should not be used when training."
        )
        assert eo_sequence.dim() == 5, (
            f"Expected eo_sequence to have 5 dimensions, got {eo_sequence.dim()}"
        )
        eo_sequence_shape = eo_sequence.shape[-2:]  # (H, W)
        inputs = self._pad(eo_sequence, eo_sequence_shape)

        frames = self.transform.encode(inputs)
        # [B, T, C_emb, H_emb, W_emb]

        B, T, _, H_emb, W_emb = frames.shape

        # First we encode the frames with masking
        states = None
        encode_outputs: List[List[EncodeOut]] = []
        decoded_frames: List[List[Tensor]] = []

        _ks: List[List[int]] = [[] for _ in range(T)]
        # _ks[i] will be a list of k values for frame i,
        # where each k corresponds to a different k in ks.
        for k in ks:
            if isinstance(k, list):
                assert len(k) <= T, (
                    f"Expected k to be a list of length at most {T},"
                    " one for each frame in the sequence."
                )
                if len(k) < T:
                    k.extend([k[-1]] * (T - len(k)))

                for i, k_i in enumerate(k):
                    _ks[i].append(k_i)
            else:
                assert isinstance(k, int), (
                    f"Expected k to be int or list, got {type(k)}"
                )
                for i in range(T):
                    _ks[i].append(k)

        for index in range(T):
            frame = frames[:, index, ...]  # [B, C_emb, H_emb, W_emb]
            # Encode each frame with masking
            states, encode_outs = self._encode_frame_masking(
                frame,
                index,
                _ks[index],
                states,
                forecast_mode=forecast_mode,
            )
            encode_outputs.append(encode_outs)

            # Decode each bottleneck for each k
            dec_frames = self._decode_frames(
                (encode_out.bottleneck for encode_out in encode_outs),
            )
            decoded_frames.append(list(dec_frames))

        # Now we have encode_outputs and decoded_frames
        # encode_outputs[i][j] is EncodeOut for frame i with k=_ks[i][j]
        # decoded_frames[i][j] is the reconstructed frame i with k=_ks[i][j]

        reconstructions_frames = []
        bits: List[List[int]] = []

        # Iterate over each k in ks
        # and collect the reconstructions and likelihoods
        for k_index, _ in enumerate(ks):
            # For each k, we collect the decodeds and likelihoods
            dec_frames_k = [
                dec_frames[k_index] for dec_frames in decoded_frames
            ]
            dec_frames_k_th = torch.stack(
                dec_frames_k, dim=1
            )  # [B, T, C_emb, H_emb, W_emb]

            reconstructions_frames_k = self.transform.decode(
                dec_frames_k_th, eo_sequence.shape
            )  # [B, T, C, H, W]

            # Compression results
            assert k_index == 0, (
                "When coding to strings, only a single k is supported."
            )

            bits_k: List[int] = []  # [T]
            for i, encode_outs in enumerate(encode_outputs):
                encode_out = encode_outs[k_index]
                assert encode_out.bottleneck.likelihood.dim() == 1, (
                    f"Expected 1D likelihood for frame {i}, got {encode_out.bottleneck.likelihood.shape}"
                )

                bit_count = float(encode_out.bottleneck.likelihood.item())
                if not bit_count.is_integer():
                    raise ValueError(
                        f"Bit count should be integer, got {bit_count}"
                    )
                bits_k.append(int(bit_count))

            reconstructions_frames.append(reconstructions_frames_k)
            bits.append(bits_k)

        return (
            reconstructions_frames,  # List of tensors [B, T, C, H, W]
            bits,  # List of List of ints [T]
        )

    def _on_cpu(self) -> bool:
        """
        Check if all model parameters live on CPU.

        :return: True if on CPU, False otherwise.
        """
        cpu = torch.device("cpu")
        for param in self.parameters():
            if param.device != cpu:
                return False
        return True

    def compute_rate(
        self,
        frames_likelihoods: torch.Tensor,
        per_frame: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute bit-rate from likelihood tensor.

        Args:
            frames_likelihoods: [T, B', seq_len_dec, C_emb] tensor of likelihoods in (0,1].
            per_frame: if True, return [T] (bits per frame); else scalar total bits.
            mask: Optional boolean/broadcastable mask; True where tokens are excluded from rate.
                Accepted cases:
                    3D [T, B', seq_len_dec] tensor
                    or 1D [seq_len_dec] tensor
        Returns:
            Tensor: scalar or [T] tensor of bits.
        """
        eps = 1e-12  # to avoid log(0)
        # Convert likelihoods -> bits per token: -log2 p
        bits_tok = -torch.log2(frames_likelihoods.clamp_min(eps))
        seq_len = frames_likelihoods.shape[2]

        if mask is not None:
            if mask.dim() == 3:
                assert (
                    mask.shape[0] == frames_likelihoods.shape[0]
                    and mask.shape[1] == frames_likelihoods.shape[1]
                    and mask.shape[2] == frames_likelihoods.shape[2]
                ), (
                    "mask must be 3D with shape [T, B', seq_len_dec]; "
                    f"got {mask.shape} vs {frames_likelihoods.shape}"
                )
            elif mask.dim() == 1:
                assert mask.shape[0] == seq_len, (
                    "mask must be 1D and have length == seq_len_dec; "
                    f"got {mask.shape[0]} vs {seq_len}"
                )
                mask = mask.view(1, 1, -1)  # [1, 1, seq_len_dec]
            else:
                raise ValueError(
                    "mask must be either 3D [T, B', seq_len_dec] or 1D [seq_len_dec]"
                    f"; got {mask.dim()}D tensor."
                )

            rate_mask = (~mask)[..., None].to(
                device=bits_tok.device, dtype=bits_tok.dtype
            )  # [T, B', seq_len_dec, 1]
            bits_tok *= rate_mask  # zero out excluded positions

        # Reduce
        if per_frame:
            # sum over B', P, C -> keep T
            bits = bits_tok.sum(dim=(1, 2, 3))
        else:
            bits = bits_tok.sum()

        return bits

    def enable_compression(
        self,
        force_update: bool = False,
        test_mode: bool = False,
    ) -> None:
        """
        Enable compression mode with smart table updating.

        :param force_update: If True, force update compression tables regardless of state
        :param test_mode: If True, we are in test mode and should only update tables once.
        """
        self.eval()

        # Determine if tables should be updated
        should_update = force_update

        if test_mode and not self._definitive_tables_initialized:
            # In test mode, we only update tables once
            should_update = True
            self._definitive_tables_initialized = True

        if should_update:
            self.entropy_model.update(force=True)

        self._code_to_strings = True

    def disable_compression(self) -> None:
        """
        Disable compression mode and return to normal forward pass mode.
        """
        print("Disabling compression mode.")
        self._code_to_strings = False

    def is_compression_ready(self) -> bool:
        """
        Check if compression is properly configured and ready to use.

        :return: True if compression is enabled
        """
        return self._code_to_strings

    def compress_eo_sequence(
        self,
        eo_sequence: torch.Tensor,
    ) -> Tuple[Tensor, List[int]]:
        """
        Compress a single EO sequence to bit-strings.

        :param eo_sequence: tensor with batch size 1.
        :return: Tuple (reconstructed embeddings, list of bit counts).
        """

        # Ensure compression is properly configured
        if not self.is_compression_ready():
            # For testing/inference, force update tables once
            self.enable_compression(force_update=True)

        assert eo_sequence.dim() == 5 and eo_sequence.shape[0] == 1, (
            "Expected batch 1 eo_sequence of shape [1, T, C, H, W],"
            f" got {eo_sequence.shape}"
        )

        frames_shape = eo_sequence.shape[-2:]  # (H, W)
        x = self._pad(eo_sequence, frames_shape)
        frames = self.transform.encode(x)

        # `_encode_and_decode_frames` compresses to strings
        # if training we are in eval mode and _code_to_strings is set to True
        network_outs = self._encode_and_decode_frames(
            frames,
            enable_masking=False,
        )

        rec_frames = []
        bits: List[int] = []
        for i, res in enumerate(network_outs):
            rec_frames.append(res.reconstruction)  # [1, C_emb, H_emb, W_emb]
            assert res.likelihood.dim() == 1
            bit_count = float(res.likelihood.item())
            if not bit_count.is_integer():
                raise ValueError(
                    f"Bit count for frame {i} is not an integer: {bit_count}"
                )
            bits.append(int(bit_count))

        # Stack reconstructions to get [B, T, C_emb, H_emb, W_emb];
        # sum to get total bits
        return torch.stack(rec_frames, dim=1), bits

    def decompress_eo_sequence(
        self,
        frames_shape: torch.Size,
        bottleneck_args: Sequence,  # [rec_frames] of shape [B, T, C_emb, H_emb, W_emb]
        force_cpu: bool = False,
    ) -> Tensor:
        """
        Decompress stored bottleneck embeddings to final frames.

        :param frames_shape: Original tensor shape for synthesis.
        :param bottleneck_args: Sequence, first element is embeddings
            [B, T, C_emb, H_emb, W_emb].
        :param force_cpu: If True, error if model not on CPU.
        :return: Tensor of reconstructions [B, T, C, H, W].
        """
        if not self._on_cpu() and force_cpu:
            raise ValueError("Decompression not supported on GPU.")

        reconstructions_frames = self.transform.decode(
            bottleneck_args[0], frames_shape
        )

        assert len(frames_shape) == 5
        return reconstructions_frames  # type: ignore[no-any-return]

    # --- TEC-TT sequence-level API ---
    @torch.inference_mode()
    def compress(self, eo_sequence: torch.Tensor) -> Dict[str, Any]:
        """
        Sequence-level compression.

        Args:
            eo_sequence: Tensor of shape [1, T, C, H, W]. Batch size must be 1.

        Returns:
            payload: Dict with fields
                - shape: original tensor shape list [1, T, C, H, W]
                - latent_hw: [H_emb, W_emb]
                - num_channels: C_emb
                - strings: list of length T; each entry is a list of length
                  seq_len_dec where each element is a [str] with length 1.
                - bits: list[int] per frame (for convenience)
        """
        # Ensure compression tables ready (entropy coder CDFs)
        if not self.is_compression_ready():
            self.enable_compression(force_update=True)

        assert eo_sequence.dim() == 5 and eo_sequence.shape[0] == 1, (
            "Expected shape [1, T, C, H, W] with batch 1"
        )

        frames_shape = eo_sequence.shape  # [1, T, C, H, W]
        x = self._pad(eo_sequence, frames_shape[-2:])
        frames = self.transform.encode(x)  # [1, T, C_emb, H_emb, W_emb]

        _B, T, C_emb, H_emb, W_emb = frames.shape
        seq_len_dec = self.get_dec_len()

        state: Optional[State] = None
        all_strings: List[List[List[bytes]]] = []  # T, seq_len_dec, [str]
        bits_per_frame: List[int] = []

        for t in range(T):
            frame = frames[:, t, ...]  # [1, C_emb, H_emb, W_emb]
            is_iframe = t == 0

            # Build previous latent context
            if is_iframe:
                fake_prev = self._temporal_pad_token_maker(frame.shape)
                prev_proc = self.entropy_model.process_previous_latent_q(
                    fake_prev
                )
                previous_latents: State = (prev_proc,)
            else:
                assert state is not None
                previous_latents = state

            # Patchify current latent and build encoded context
            latent_patched, (nH, nW) = self.entropy_model.patcher(
                frame, self.entropy_model.window_size_dec
            )
            encoded_seqs = self.entropy_model._get_encoded_seqs(
                previous_latents
            )
            encoded = torch.cat(encoded_seqs, -2)

            # True range encoding to strings (no decoding here)
            strings, extra = self.entropy_model._encode(latent_patched, encoded)

            # Count bits in this frame
            bit_count = int(sum(len(s[0]) * 8 for s in strings))
            bits_per_frame.append(bit_count)
            all_strings.append(strings)

            # Get quantized latent from encode extra to advance temporal state
            quantized_patched: torch.Tensor = extra["quantized"]
            latent_q = self.entropy_model.patcher.unpatch(
                t=quantized_patched, nH=nH, nW=nW, crop=(H_emb, W_emb)
            )
            # Update state with the newly quantized latent
            state = self._update_state(
                latent_q=latent_q, state=state, is_iframe=is_iframe
            )

        return {
            "shape": list(frames_shape),
            "latent_hw": [int(H_emb), int(W_emb)],
            "num_channels": int(C_emb),
            "strings": all_strings,
            "bits": bits_per_frame,
            "seq_len_dec": int(seq_len_dec),
        }

    @torch.inference_mode()
    def decompress(
        self,
        payload: Dict[str, Any],
    ) -> Tensor:
        """
        Sequence-level decompression. Reconstruct all frames solely from
        the encoded bitstreams and minimal metadata.

        Expected payload format produced by `compress`.

        Args (inside payload):
            strings: nested list per frame and per token, base64-encoded ascii strings
                     (or raw bytes converted to str) produced by `compress`.

        Returns:
            Tensor [1, T, C, H, W]
        """
        # Ensure entropy tables and cached shapes
        if not self.is_compression_ready():
            self.enable_compression(force_update=True)

        strings_any = payload.get("strings") or payload.get("bitstreams")
        assert strings_any is not None, "Payload missing 'strings'/'bitstreams'"
        strings: List[List[List[Union[str, bytes]]]] = strings_any

        T_payload = len(strings)
        if "shape" in payload:
            shape_list = payload["shape"]
            assert shape_list[0] == 1, (
                "Batch size > 1 not supported by compress()/decompress() API"
            )
            assert shape_list[1] == T_payload, (
                "Mismatch between strings length and payload T"
            )

        self._ensure_shape_cache()

        assert (
            self._cached_in_channels is not None
            and self._cached_latent_hw is not None
            and self._cached_latent_channels is not None
            and self._fixed_image_hw is not None
        ), "Shape cache not initialized"

        in_C = self._cached_in_channels
        H, W = self._fixed_image_hw
        H_emb, W_emb = self._cached_latent_hw
        C_emb = self._cached_latent_channels

        T = len(strings)

        # Build tensors per frame by true arithmetic decoding
        synthesis_inputs: List[Tensor] = []

        state: Optional[State] = None
        for t in range(T):
            is_iframe = t == 0

            # Previous latent context
            if is_iframe:
                fake_prev_latent = torch.zeros(
                    (1, C_emb, H_emb, W_emb),
                    device=next(self.parameters()).device,
                )
                fake_prev_latent = self._temporal_pad_token_maker(
                    fake_prev_latent.shape
                )
                prev_proc = self.entropy_model.process_previous_latent_q(
                    fake_prev_latent
                )
                previous_latents: State = (prev_proc,)
            else:
                assert state is not None
                previous_latents = state

            # Build encoded context from previous latents
            encoded_seqs = self.entropy_model._get_encoded_seqs(
                previous_latents
            )
            encoded = torch.cat(encoded_seqs, -2)

            # Create dummy means/scales tensors (not used when use_output_from_encode=False)
            fake = self.entropy_model.patcher(
                torch.ones((1, C_emb, H_emb, W_emb), device=encoded.device),
                self.entropy_model.window_size_dec,
            )
            nH, nW = fake.num_patches
            Bp = nH * nW
            seq_len_dec = self.get_dec_len()
            encoded_means = torch.zeros(
                (Bp, seq_len_dec, C_emb), device=encoded.device
            )
            encoded_scales = torch.ones_like(encoded_means)

            # Convert base64-encoded strings back to bytes for entropy decoder
            strings_t_raw = strings[t]
            strings_t: List[List[bytes]] = []
            for tok in strings_t_raw:
                v = tok[0] if isinstance(tok, (list, tuple)) else tok
                if isinstance(v, bytes):
                    strings_t.append([v])
                elif isinstance(v, str):
                    try:
                        b = base64.b64decode(v.encode("ascii"))
                    except Exception:
                        b = v.encode("utf-8")
                    strings_t.append([b])
                else:
                    strings_t.append([str(v).encode("utf-8")])

            # True arithmetic decoding from strings -> quantized latent
            latent_q, z_cur = self.entropy_model._decode(
                strings=strings_t,
                encoded=encoded,
                shape=(H_emb, W_emb, C_emb),
                encoded_means=encoded_means,
                encoded_scales=encoded_scales,
                use_output_from_encode=False,
            )  # [1, C_emb, H_emb, W_emb]

            # Dequantize to synthesis input
            synthesis_in = self._dequantizer(
                latent_q=latent_q, entropy_features=z_cur
            )
            synthesis_inputs.append(synthesis_in)

            # Advance temporal state
            state = self._update_state(
                latent_q=latent_q, state=state, is_iframe=is_iframe
            )

        # Stack and run synthesis transform
        synthesis_stack = torch.stack(synthesis_inputs, dim=1)
        reconstructions = self.transform.decode(
            synthesis_stack, torch.Size([1, T, in_C, H, W])
        )
        return reconstructions

    # --- FlexTEC (variable-rate compression) API ---
    @torch.inference_mode()
    def compress_flextec(
        self,
        eo_sequence: torch.Tensor,
        quality: Union[int, List[int]],
        forecast_mode: str = "deterministic",
    ) -> Dict[str, Any]:
        """
        Variable-rate compression for FlexTEC.

        Args:
            eo_sequence: [1, T, C, H, W] (batch=1)
            quality: int in [0..seq_len_dec] or List[int] length T.
            forecast_mode: fill strategy for dropped tokens >= k
                           (default "deterministic" to match run_flextec).

        Returns:
            payload dict with fields:
                - strings: per-frame list of length k_t with bytestrings
                - quality: List[int] of length T
                - bits: List[int] bits per frame
                - seq_len_dec: decoder token length (for sanity)
        """
        # Ensure entropy tables are ready
        if not self.is_compression_ready():
            self.enable_compression(force_update=True)

        assert eo_sequence.dim() == 5 and eo_sequence.shape[0] == 1, (
            "Expected shape [1, T, C, H, W] with batch 1"
        )

        frames_shape = eo_sequence.shape
        x = self._pad(eo_sequence, frames_shape[-2:])
        frames = self.transform.encode(x)  # [1, T, C_emb, H_emb, W_emb]

        _B, T, C_emb, H_emb, W_emb = frames.shape
        seq_len_dec = self.get_dec_len()

        # Normalize quality per-frame list
        if isinstance(quality, int):
            q_list: List[int] = [int(quality)] * T
        else:
            assert len(quality) <= T, (
                f"quality list length {len(quality)} must be <= T={T}"
            )
            # extend by repeating last if needed
            q_list = list(map(int, quality))
            if len(q_list) < T:
                q_list.extend([q_list[-1]] * (T - len(q_list)))

        # Validate bounds
        for k in q_list:
            assert 0 <= k <= seq_len_dec, (
                f"quality (k) {k} must be in [0, {seq_len_dec}]"
            )

        state: Optional[State] = None
        all_strings: List[List[List[bytes]]] = []
        bits_per_frame: List[int] = []

        for t in range(T):
            frame = frames[:, t, ...]  # [1, C_emb, H_emb, W_emb]
            is_iframe = t == 0

            # Build previous latent context
            if is_iframe:
                fake_prev = self._temporal_pad_token_maker(frame.shape)
                prev_proc = self.entropy_model.process_previous_latent_q(
                    fake_prev
                )
                previous_latents: State = (prev_proc,)
            else:
                assert state is not None
                previous_latents = state

            # Patchify and build encoded context
            latent_patched, (nH, nW) = self.entropy_model.patcher(
                frame, self.entropy_model.window_size_dec
            )
            encoded_seqs = self.entropy_model._get_encoded_seqs(
                previous_latents
            )
            encoded = torch.cat(encoded_seqs, dim=-2)

            k = q_list[t]

            # Encode first k tokens to true bitstrings, hallucinate rest
            strings, extra = self.entropy_model._encode_masked(
                latent_patched=latent_patched,
                encoded=encoded,
                k=k,
                hallucination_mode=(
                    "deterministic"
                    if forecast_mode == "deterministic"
                    else "stochastic"
                ),
            )

            # Count bits in this frame (strings length == k)
            bit_count = int(sum(len(s[0]) * 8 for s in strings))
            bits_per_frame.append(bit_count)
            all_strings.append(strings)

            # Update temporal state using the quantized latent from encode
            quantized_patched: torch.Tensor = extra["quantized"]
            latent_q = self.entropy_model.patcher.unpatch(
                t=quantized_patched, nH=nH, nW=nW, crop=(H_emb, W_emb)
            )
            state = self._update_state(
                latent_q=latent_q, state=state, is_iframe=is_iframe
            )

        return {
            "strings": all_strings,
            "quality": q_list,
            "bits": bits_per_frame,
            "seq_len_dec": int(seq_len_dec),
        }

    @torch.inference_mode()
    def decompress_flextec(self, payload: Dict[str, Any]) -> Tensor:
        """
        Decompression for FlexTEC.

        The payload must contain:
            - strings: per-frame base64/bytes bitstrings (only first k tokens)
            - quality: List[int] per frame indicating k

        Returns: tensor [1, T, C, H, W]
        """
        # Ensure entropy tables and cached shapes
        if not self.is_compression_ready():
            self.enable_compression(force_update=True)

        self._ensure_shape_cache()

        assert (
            self._cached_in_channels is not None
            and self._cached_latent_hw is not None
            and self._cached_latent_channels is not None
            and self._fixed_image_hw is not None
        ), "Shape cache not initialized"

        in_C = self._cached_in_channels
        H, W = self._fixed_image_hw
        H_emb, W_emb = self._cached_latent_hw
        C_emb = self._cached_latent_channels

        all_strings_any = payload.get("strings")
        assert all_strings_any is not None, "Payload missing 'strings'"
        all_strings_raw: List[List[List[Union[str, bytes]]]] = all_strings_any

        q_list = payload.get("quality")
        assert isinstance(q_list, list) and len(q_list) == len(
            all_strings_raw
        ), "Payload must include 'quality' list matching #frames"

        T = len(all_strings_raw)

        # Build tensors per frame by masked arithmetic decoding
        synthesis_inputs: List[Tensor] = []
        state: Optional[State] = None

        for t in range(T):
            is_iframe = t == 0
            k = int(q_list[t])

            # Previous latent context
            if is_iframe:
                fake_prev_latent = torch.zeros(
                    (1, C_emb, H_emb, W_emb),
                    device=next(self.parameters()).device,
                )
                fake_prev_latent = self._temporal_pad_token_maker(
                    fake_prev_latent.shape
                )
                prev_proc = self.entropy_model.process_previous_latent_q(
                    fake_prev_latent
                )
                previous_latents: State = (prev_proc,)
            else:
                assert state is not None
                previous_latents = state

            # Encoded context from previous latents
            encoded_seqs = self.entropy_model._get_encoded_seqs(
                previous_latents
            )
            encoded = torch.cat(encoded_seqs, dim=-2)

            # Construct dummy means/scales (unused when use_output_from_encode=False)
            fake = self.entropy_model.patcher(
                torch.ones((1, C_emb, H_emb, W_emb), device=encoded.device),
                self.entropy_model.window_size_dec,
            )
            nH, nW = fake.num_patches
            Bp = nH * nW
            seq_len_dec = self.get_dec_len()
            encoded_means = torch.full(
                (Bp, seq_len_dec, C_emb), 100.0, device=encoded.device
            )
            encoded_scales = torch.full_like(encoded_means, 100.0)

            # Convert base64 strings to bytes for entropy decoder
            strings_t_raw = all_strings_raw[t]
            strings_t: List[List[bytes]] = []
            for tok in strings_t_raw:
                v = tok[0] if isinstance(tok, (list, tuple)) else tok
                if isinstance(v, bytes):
                    strings_t.append([v])
                elif isinstance(v, str):
                    try:
                        b = base64.b64decode(v.encode("ascii"))
                    except Exception:
                        b = v.encode("utf-8")
                    strings_t.append([b])
                else:
                    strings_t.append([str(v).encode("utf-8")])

            # Masked arithmetic decoding of first k tokens, hallucinate rest
            latent_q, z_cur = self.entropy_model._decode_masked(
                strings=strings_t,
                encoded=encoded,
                shape=(H_emb, W_emb, C_emb),
                k=k,
                encoded_means=encoded_means,
                encoded_scales=encoded_scales,
                use_output_from_encode=False,
            )  # [1, C_emb, H_emb, W_emb]

            # Dequantize to synthesis input and collect
            synthesis_in = self._dequantizer(
                latent_q=latent_q, entropy_features=z_cur
            )
            synthesis_inputs.append(synthesis_in)

            # Advance temporal state
            state = self._update_state(
                latent_q=latent_q, state=state, is_iframe=is_iframe
            )

        synthesis_stack = torch.stack(synthesis_inputs, dim=1)
        reconstructions = self.transform.decode(
            synthesis_stack, torch.Size([1, T, in_C, H, W])
        )
        return reconstructions
