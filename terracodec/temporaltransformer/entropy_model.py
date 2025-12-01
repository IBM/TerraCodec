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

import itertools
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .bottlenecks import GsnConditionalLocScaleShift
from .layers_utils import LearnedPosition, StartSym, make_embedding_layer
from .patcher import Patched, Patcher
from .transformer_layers import EncoderSection, Transformer

_LATENT_NORM_FAC: float = 35.0  # factor to scale latents by


def _unbatch(t: Tensor, dims: Tuple[int, int]) -> Tensor:
    """Reshapes first dimension, i.e. (b, ...) becomes (b', *dims, ...)."""
    b_in, *other_dims = t.shape
    b_out = b_in // (dims[0] * dims[1])
    return t.view(b_out, *dims, *other_dims)


class PreviousLatent(NamedTuple):
    """Previous latent with the following attributes

    Attributes:
        quantized: the quantized latent
        processed: the processed latent by running it through an encoder. See
            `TerraCodecTTEntropyModle.process_previous_latent_q` for more details.
    """

    quantized: Tensor
    processed: Tensor


class TemporalEntropyModelOut(NamedTuple):
    """Output of the TerraCodecTT temporal entropy model

    When Masking enabled, masked perturbed latent is the one that is used for synthesis
    transform, while perturbed latent is the one that is used for the next frame.
    k is not None, but the k value used for variable token usage.

    Attributes:
      perturbed_latent: noised (training=True) or quantized (training=False)
        latent. Tensor of shape [b', seq_len, C]
      bits: bits taken to transmit the latent. Tensor of shape: [b', seq_len, C]
      metrics: Metrics collected by the entropy model. Shape: [b', seq_len, C]
      features: (optional) features of the entropy model to be used by a
        synthesis transform for dequantizing.
        Tensor of shape: [B, d_model, H, W]
      mask (optional): mask for masked training, if enabled.
        Tensor of shape [b', seq_len_dec], where True indicates that the
        corresponding token is masked out.
    """

    perturbed_latent: Tensor
    bits: Tensor
    features: Optional[Tensor] = None
    masked_features: Optional[Tensor] = None
    masked_perturbed_latent: Optional[Tensor] = None
    mask: Optional[Tensor] = None


class TerraCodecTTEntropyModel(nn.Module):
    """
    Temporal Entropy Model
    """

    def __init__(
        self,
        num_channels: int = 192,
        context_len: int = 2,
        enable_latent_repacking: bool = False,
        enable_masking: bool = False,
        window_size_enc: int = 8,
        window_size_dec: int = 4,
        num_layers_encoder_sep: int = 6,
        num_layers_encoder_joint: int = 4,
        num_layers_decoder: int = 5,
        d_model: int = 768,
        num_head: int = 16,
        mlp_expansion: int = 4,
        drop_out_enc: float = 0.0,
        drop_out_dec: float = 0.0,
    ) -> None:
        """
        Temporal Entropy Model
        Args:
            num_channels: number of channels in the latent space,
                i.e. symbols per token. Defaults to 196.
            context_len: number of previous latents. Defaults to 2.
            enable_latent_repacking: whether to enable Latent Repacking.
            enable_masking: whether to enable masking.
            window_size_enc: window (patch) size in encoder.
                Defaults to 8.
            window_size_dec: window (patch) size in decoder.
                Defaults to 4.
            num_layers_encoder_sep: number of layers in the separate encoder.
                Defaults to 3.
            num_layers_encoder_joint: number of layers in the joint encoder.
                Defaults to 2.
            num_layers_decoder: number of layers in the decoder.
                Defaults to 5.
            d_model: feature dimensionality inside the model.
                Defaults to 768.
            num_head: number of attention heads in MHA layers.
                Defaults to 16.
            mlp_expansion: expansion *factor* for each MLP.
                Defaults to 4.
            drop_out_enc: dropout probability in encoder.
                Defaults to 0.0.
            drop_out_dec: dropout probability in decoder.
                Defaults to 0.0.
        """
        super().__init__()
        if window_size_enc < window_size_dec:
            raise ValueError(
                f"window_size_enc={window_size_enc} cannot be lower"
                f"than window_size_dec={window_size_dec}."
            )
        if num_channels < 0:
            raise ValueError(f"num_channels={num_channels} cannot be negative")
        self.num_channels = num_channels
        self.window_size_enc = window_size_enc
        self.window_size_dec = window_size_dec

        self.d_model = d_model
        # we will use compressai's GsnConditional as a bottleneck
        self.bottleneck = GsnConditionalLocScaleShift(
            num_scales=256, num_means=100, min_scale=0.01, tail_mass=(2 ** (-8))
        )

        self.range_bottleneck = None
        self.context_len = context_len

        # LR, Masking
        self.masking = enable_masking

        self.encoder_sep = EncoderSection(
            num_layers=num_layers_encoder_sep,
            num_heads=num_head,
            d_model=d_model,
            mlp_expansion=mlp_expansion,
            dropout=drop_out_enc,
        )
        self.encoder_joint = EncoderSection(
            num_layers=num_layers_encoder_joint,
            num_heads=num_head,
            d_model=d_model,
            mlp_expansion=mlp_expansion,
            dropout=drop_out_enc,
        )

        # Important here to pass in the info about adaptative patching
        self.patcher = Patcher(
            stride=window_size_dec,
            enable_latent_repacking=enable_latent_repacking,
            pad_mode="reflect",
        )

        self.learned_zero = StartSym(num_channels=num_channels)

        # Learned mask
        if enable_masking:
            self.mask_token = nn.Parameter(torch.empty(num_channels))
            nn.init.uniform_(self.mask_token, -1.0, 1.0)

            self.z_cur_mask_token = nn.Parameter(
                torch.empty(d_model)
            )  # Mask token for z_cur
            nn.init.uniform_(self.z_cur_mask_token, -1.0, 1.0)

        self.seq_len_enc = window_size_enc**2
        self.enc_position_sep = LearnedPosition(
            seq_len=self.seq_len_enc, d_model=d_model
        )
        self.enc_position_joint = LearnedPosition(
            seq_len=self.seq_len_enc * context_len, d_model=d_model
        )
        self.seq_len_dec = window_size_dec**2
        self.avail_ks = list(
            itertools.chain(
                *[[i] * (i) for i in range(1, self.seq_len_dec + 1)]
            )
        )

        self.dec_position = LearnedPosition(
            seq_len=self.seq_len_dec, d_model=d_model
        )

        self.post_embedding_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        self.encoder_embedding = make_embedding_layer(
            num_channels=num_channels, d_model=d_model
        )  # a single linear layer
        self.decoder_embedding = make_embedding_layer(
            num_channels=num_channels, d_model=d_model
        )  # a single linear layer

        self.decoder = Transformer(
            seq_len=self.seq_len_dec,
            num_layers=num_layers_decoder,
            num_heads=num_head,
            d_model=d_model,
            mlp_expansion=mlp_expansion,
            dropout_rate=drop_out_dec,
            is_decoder=True,
        )  # its forward method returns [B', seq_len_dec, d_model]

        def _make_final_heads(output_channels: int) -> nn.Module:
            # 3 stacked linear layers with leakyrelu activations
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, output_channels),
            )

        self.mean_head = _make_final_heads(num_channels)
        self.scale_head = _make_final_heads(num_channels)

    def get_dec_len(self) -> int:
        """
        Get the number of tokens per patch in the latent space.

        :return: Number of tokens per patch in the latent space.
        """
        return self.seq_len_dec

    def enable_masked(self) -> None:
        """
        Enable latent repacking and masking.
        Used for inference with masking after training.
        """
        self.masking = True
        self.latent_repacking = True
        self.patcher = Patcher(
            stride=self.window_size_dec,
            enable_latent_repacking=True,
            pad_mode="reflect",
        )

    @staticmethod
    def round_st(x: Tensor) -> Tensor:
        """
        Straight-through round

        Allows for gradient to pass through the rounding operation.
        """
        return (torch.round(x) - x).detach() + x

    def process_previous_latent_q(
        self, previous_latent_quantized: Tensor
    ) -> PreviousLatent:
        """Process previous quantized latent by passing it through the encoder.

        This can be used if previous latents go through expensive transforms
        before being fed to the entropy model, and will be stored in the
        `processed` field of the `PreviousLatent` tuple.

        The output of this function applied to all quantized latents should
        be fed to the `forward` method. This is used to improve efficiency,
        as it avoids calling expensive processing of previous latents at
        each time step.

        Args:
            previous_latent_quantized: previous quantized latent that is to be
                processed, expected shape [B, C_emb, H_emb, W_emb].

        Returns:
            PreviousLatent object with the processed latent in the processed
                field
        """
        patches: torch.Tensor
        patches, _ = self.patcher(
            previous_latent_quantized, self.window_size_enc
        )
        # [B', Wp x Wp, C_emb] where B' = B * nH * nW
        # Internally pads H_emb and W_emb so that when sliding overlapping
        # patches of Wp x Wp, with stride Wc, we get the same number of patches
        # (nH * nW) as when doing non-overlapping patching using Wc.

        patches = patches / _LATENT_NORM_FAC
        patches = self.encoder_embedding(patches)  # projects C_emb to d_model
        patches = self.post_embedding_layernorm(patches)
        patches = self.enc_position_sep(patches)  # [B', Wp x Wp, d_model]
        patches = self.encoder_sep(patches)  # [B', Wp x Wp, d_model]
        # [B', Wp x Wp, d_model] = [B * 16 (patches), 8 x 8 (tokens), 768]

        return PreviousLatent(previous_latent_quantized, processed=patches)

    def _embed_latent_q_patched(self, latent_q_patched: Tensor) -> Tensor:
        """Embed current patched latent for decoder

        The input latent is normalized, embedded in d_model dimension, and
        positional encoding is added.
        Args:
            latent_q_patched: tensor of shape [b', seq_len_dec, C]

        Returns:
            tensor of shape [b', seq_len_dec, d_model]
        """
        latent_q_patched = (
            latent_q_patched / _LATENT_NORM_FAC
        )  # [b', seq_len_dec, C]
        latent_q_patched = self.decoder_embedding(
            latent_q_patched
        )  # [b', seq_len_dec, d_model]
        latent_q_patched = self.post_embedding_layernorm(
            latent_q_patched
        )  # [b', seq_len_dec, d_model]
        return self.dec_position(latent_q_patched)  # type: ignore[no-any-return]

    def _terracodec_tt_encoder_forward(
        self,
        *,
        encoded_patched: Tensor,
        latent_q_patched: Tensor,
        processed_zjoint: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict the distribution of the current quantized patched latent

        :param encoded_patched: tensor of shape
            [*b, context_len*patch_enc^2, d_model],
            where with the defaults (patch_enc=8, d_model=768), so we have
            default expected shape [*b, 128, 768]
        :param latent_q_patched: tensor of shape [*b, patch_dec^2, num_channels]
            with the default patch_dec^2=16 and num_channels = 192, so we have
            default expected shape [*b, 16, 192]
        :param processed_zjoint: optional tensor of shape
            [*b, context_len * patch_enc^2, d_model], containing the
            processed joint latent, i.e. the output of the encoder_joint
            if not None, this will be used instead of encoding the
            `encoded_patched` tensor.

        :return: a tuple containing 4 tensors:
            mean, scale, z_cur and z_joint.
        """
        if encoded_patched.shape[-1] != self.d_model:
            raise ValueError(
                f"Context must have final dim {self.d_model}, "
                f"got shape={encoded_patched.shape}. "
                "Did you run `process_previous_latent_q`?"
            )

        # Shift in the token dimension, one position to the right
        # This allows autoregressive decoding
        latent_q_patched_shifted = self.learned_zero(latent_q_patched)
        del latent_q_patched  # should not be used after this line

        latent_q_patched_emb_shifted = self._embed_latent_q_patched(
            latent_q_patched_shifted
        )  # [B', Wc x Wc, d_model]

        if processed_zjoint is not None:
            encoded_patched = processed_zjoint
        else:
            encoded_patched = self.enc_position_joint(encoded_patched)
            encoded_patched = self.encoder_joint(
                encoded_patched
            )  # [B', context_len * Wp x Wp, d_model]

        # RUN DECODER
        dec_output = self.decoder(
            latent=latent_q_patched_emb_shifted,  # current quantized latent
            # (embedded and shifted)
            encoder_output=encoded_patched,  # = z_joint
        )  # [B', Wc x Wc, d_model] = z_cur

        # Project back to C_emb to obtain mu and sigma
        mean = self.mean_head(dec_output)
        scale = self.scale_head(dec_output)

        return mean, scale, dec_output, encoded_patched

    def _get_encoded_seqs(
        self, previous_latents: Sequence[PreviousLatent]
    ) -> List[Tensor]:
        """
        Extract the previously procesed latents, repeating them if the
        number of processed latents is less than the context legnth

        Args:
            previous_latents: sequence of sizse at most `context_len`,
                containing object of type PreviousLatent, with two attributes:
                    - `processed` tensor of shape [B', seq_len_enc, d_model]
                    - `quantized` tensor of shape [B, C_emb, H_emb, W_emb]
                        NOT needed in this method

        Returns:
            List of length `context_len` with tensors of shape
                [B', seq_len_enc, d_model], containing `processed` data only
                (encoder processed data).
        """
        encoded_seqs = [p.processed for p in previous_latents]
        if len(encoded_seqs) < self.context_len:
            if self.context_len == 2:
                # encoded_seqs is a list of size 1
                return encoded_seqs * 2  # [b', seq_len_enc, d_model]*2
            elif self.context_len == 3:
                return (
                    encoded_seqs * 3  # [b', seq_len_enc, d_model]*3
                    if len(encoded_seqs) == 1
                    # repeat the 0th twice
                    else [encoded_seqs[0]] * 2 + [encoded_seqs[1]]
                )
            else:
                ValueError(f"Unsupported context_len={self.context_len}")
        return encoded_seqs

    def forward(
        self,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
        enable_masking: bool = False,
        k: Optional[int] = None,
    ) -> TemporalEntropyModelOut:
        if self.masking and enable_masking:
            assert k is not None, "k must be provided when masking is enabled."
            return self.masking_forward(
                latent_unquantized=latent_unquantized,
                previous_latents=previous_latents,
                k=k,
            )
        return self.regular_forward(
            latent_unquantized=latent_unquantized,
            previous_latents=previous_latents,
        )

    def regular_forward(
        self,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
    ) -> TemporalEntropyModelOut:
        """
        Args:
            latent_unquantized: the latent to transmit (quantize),
                expected shape is [B, C_emb, H_emb, W_emb],
            previous_latents: previously transmitted (quantized) latents, should
                be of size at least one and at most `context_len`.
                Each PreviousLatent has
                    - quantized: floats (i.e. noised) tensor
                        of shape [B, C_emb, H_emb, W_emb]
                    - processed: [B', seq_len_enc, d_model]
                                = [B * nH * nW, Wc x Wc, d_model]

        Returns:
            TemporalEntropyModelOut, see docstring there.
        """
        H, W = latent_unquantized.shape[-2:]
        # encoded_seqs: list of tensors [B', Wp x Wp, d_model]
        encoded_seqs = self._get_encoded_seqs(previous_latents=previous_latents)
        b_enc, _, d_enc = encoded_seqs[0].shape  # [B', Wp x Wp, d_model]
        if d_enc != self.d_model:
            raise ValueError(f"Shape mismatch, {d_enc}!={self.d_model}")

        # Soft rounding via straight-through gradient estimation
        latent_q = self.round_st(
            latent_unquantized
        )  # [B, C_emb, H_emb, W_emb] (float32 but containing ints)

        # Patcher expects [B, C_emb, H_emb, W_emb]
        # and returns [B', seq_len_dec, C_emb], with seq_len_dec = Wc x Wc
        latent_q_patched, (nH, nW) = self.patcher(
            latent_q, self.window_size_dec
        )
        b_dec, seq_len, d_dec = (
            latent_q_patched.shape
        )  # b_dec, patch_size_dec^2, C
        if d_dec != self.num_channels:
            raise ValueError(
                f"Model dims don't match, {d_dec}!={self.num_channels}"
            )
        if b_dec != b_enc:
            raise ValueError(f"Batch dims don't match, got {b_enc} != {b_dec}!")
        assert seq_len == self.window_size_dec**2, "Error patching"

        # Transformer expects inputs to have channels in the last dim,
        # i.e. [B', ..., C_emb]

        # Transformer expects quantized latents
        mean, scale, dec_output, _ = self._terracodec_tt_encoder_forward(
            encoded_patched=torch.cat(encoded_seqs, dim=-2),  # cat on seq dim
            latent_q_patched=latent_q_patched,  # [B', seq_len_dec, C_emb]
        )
        assert mean.shape == latent_q_patched.shape, (
            f"Shape mismatch! {mean.shape} != {latent_q_patched.shape}"
        )

        # Unpatch z_cur back from [B', Wc x Wc, d_model]
        # to [B, d_model, H_emb, W_emb]
        decoder_features = self.patcher.unpatch(
            t=dec_output, nH=nH, nW=nW, crop=(H, W)
        )

        latent_unquantized_patched: torch.Tensor = self.patcher(
            latent_unquantized, self.window_size_dec
        ).tensor  # [b', seq_len_dec, C]

        assert (
            latent_unquantized_patched.shape == scale.shape
        )  # [B', seq_len_dec, C]

        # We use GaussianConditional from compressai as bottleneck, its forward
        # method returns (output, likelihood)
        # in TerraCodecTT scales and means are quantized in the bottleneck
        output, likelihood_bits = self.bottleneck(
            inputs=latent_unquantized_patched, scales=scale, means=mean
        )  # output is a noised version of the `inputs`
        assert output.shape == likelihood_bits.shape  # [B', seq_len_dec, C]

        # Unpatch the output back to [B, C_emb, H_emb, W_emb]
        output = self.patcher.unpatch(t=output, nH=nH, nW=nW, crop=(H, W))

        assert output.shape == latent_unquantized.shape
        # [B, C_emb, H_emb, W_emb]

        return TemporalEntropyModelOut(
            perturbed_latent=output,
            bits=likelihood_bits,
            features=decoder_features,  # z_cur
        )

    def masking_forward(
        self,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
        k: int,
    ) -> TemporalEntropyModelOut:
        """
        Forward pass for Masking in training mode.
        This method is similar to `forward`, but it enables variable token
        usage per patch; adaptively select how much to decode or transmit

        Args:
            latent_unquantized: the latent to transmit (quantize),
                expected shape is [B, C_emb, H_emb, W_emb],
            previous_latents: previously transmitted (quantized) latents, should
                be of size at least one and at most `context_len`.
                Each PreviousLatent has
                    - quantized: floats (i.e. noised) tensor
                        of shape [B, C_emb, H_emb, W_emb]
                    - processed: [B', seq_len_enc, d_model]
                                = [B * nH * nW, Wc x Wc, d_model]
            k: integer representing the k value to use for
                variable token usage.

        Returns:
            TemporalEntropyModelOut, see docstring there.
        """
        B, C, H, W = latent_unquantized.shape

        encoded_seqs = self._get_encoded_seqs(previous_latents=previous_latents)
        encoded = torch.cat(encoded_seqs, dim=-2)
        # shape is [B', context_len*WpxWp, d_model]

        latent_patched: torch.Tensor
        latent_patched, (nH, nW) = self.patcher(
            latent_unquantized, self.window_size_dec
        )

        # [B', Wc x Wc, C_emb] where B' = B * nH * nW
        Bp, dec_len, _ = latent_patched.shape

        # ---- Encode using variable token usage per patch
        # Quantize using STE
        latent_q_patched = self.round_st(latent_patched)

        full_mean, full_scale, dec_output, _ = (
            self._terracodec_tt_encoder_forward(
                encoded_patched=encoded, latent_q_patched=latent_q_patched
            )
        )

        # Retrive noised output and likelihood from bottleneck
        output, likelihood_bits = self.bottleneck(
            inputs=latent_patched, scales=full_scale, means=full_mean
        )  # shapes: [B', dec_len, C_emb], [B', dec_len, C_emb]

        # Returned masked from k-th token onwards + the usual.
        pos = torch.arange(dec_len, device=output.device)  # [dec_len]
        M = pos >= k  # [dec_len] True where we mask
        mask_token = self.mask_token.view(1, 1, -1)  # [1,1,C_emb], nn.Parameter
        z_cur_mask_token = self.z_cur_mask_token.view(1, 1, -1)  # [1,1,d_model]
        masked_output = torch.where(
            M.view(-1, 1).expand(Bp, -1, 1),  # [B', dec_len, 1]
            mask_token,  # [1, 1, C_emb]
            output,  # [B', dec_len, C_emb]
        )
        masked_z_cur = torch.where(
            M.view(-1, 1).expand(Bp, -1, 1),  # [B', dec_len, 1]
            z_cur_mask_token,  # [1, 1, d_model]
            dec_output,  # [B', dec_len, d_model]
        )

        # True Latents
        output = self.patcher.unpatch(t=output, nH=nH, nW=nW, crop=(H, W))
        decoder_features = self.patcher.unpatch(
            t=dec_output, nH=nH, nW=nW, crop=(H, W)
        )

        # Masked Latents
        masked_output = self.patcher.unpatch(
            t=masked_output, nH=nH, nW=nW, crop=(H, W)
        )
        masked_z_cur = self.patcher.unpatch(
            t=masked_z_cur, nH=nH, nW=nW, crop=(H, W)
        )
        assert output.shape == latent_unquantized.shape, (
            f"Shape mismatch! {output.shape} != {latent_unquantized.shape}"
        )
        assert masked_output.shape == latent_unquantized.shape, (
            f"Shape mismatch! {masked_output.shape} != {latent_unquantized.shape}"
        )

        return TemporalEntropyModelOut(
            perturbed_latent=output,  # Perturbed latent
            bits=likelihood_bits,  # Bits for each token
            features=decoder_features,  # z_cur
            masked_features=masked_z_cur,  # Masked z_cur
            masked_perturbed_latent=masked_output,  # Masked perturbed latent
            mask=M.view(1, -1)  # [1, dec_len]
            .expand(Bp, -1)
            .detach()
            .cpu(),  # [B', dec_len] mask
        )

    def ent_multi_k_variable_rate_compression(
        self,
        latent_unquantized: Tensor,
        ks: List[int],
        previous_latents: List[Tuple[PreviousLatent, ...]],
    ) -> List[TemporalEntropyModelOut]:
        """
        Forward pass for Masking with multiple k values.

        Args:
            latent_unquantized: [B, C_emb, H_emb, W_emb]
            ks: list of integers k (0..seq_len_dec)
            previous_latents: list of State tuples, one per k.

        Returns:
            List[TemporalEntropyModelOut], one per k.
        """
        assert len(ks) > 0, "ks must not be empty"
        assert len(previous_latents) == len(ks), (
            f"Number of previous latents ({len(previous_latents)}) "
            f"must match number of ks ({len(ks)})"
        )

        return [
            self.masking_inference(
                latent_unquantized=latent_unquantized,
                previous_latents=previous_latents[i],
                k=ks[i],
            )
            for i in range(len(ks))
        ]

    def propagate_from_first_cloud(
        self, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        token_mask: [B * nH * nW, Wc x Wc] boolean; True means 'token is cloudy'
        Returns: same shape; True means 'use predicted token' (from the first cloud onward)
        """
        # cumulative OR across tokens (raster order)
        return token_mask.to(torch.int32).cumsum(dim=-1).clamp_max(1).bool()

    def masking_inference(
        self,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
        k: int,
        cloud_mask: Optional[Tensor] = None,
        propagate_from_first_cloud: bool = False,  # False => interleave mode
    ) -> TemporalEntropyModelOut:
        """
        Forward pass for Masking in inference mode.
        This method is similar to `forward`, but it enables variable token
        usage per patch; adaptively select how much to decode or transmit

        Args:
            latent_unquantized: the latent to transmit (quantize),
                expected shape is [B, C_emb, H_emb, W_emb],
            previous_latents: previously transmitted (quantized) latents, should
                be of size at least one and at most `context_len`.
                Each PreviousLatent has
                    - quantized: floats (i.e. noised) tensor
                        of shape [B, C_emb, H_emb, W_emb]
                    - processed: [B', seq_len_enc, d_model]
                                = [B * nH * nW, Wc x Wc, d_model]
            k: integer representing the k value to use for
                variable token usage. This is the number of tokens to use
                from the quantized latent, starting from the first token.
            cloud_mask: optional tensor of shape [B, 1, H_emb, W_emb]
                representing the cloud mask for the current frame.
            propagate_from_first_cloud: if True, in cloud-aware masking,
                once the first cloudy patch is found, all subsequent patches
                (cloudy or cloud-free) will be predicted using the model.
                If False, only cloudy patches will be predicted, while
                cloud-free patches will use the full latent (all tokens).

        Returns:
            TemporalEntropyModelOut, see docstring there.
        """
        assert not self.training, (
            "Masked inference should not be used in training mode."
        )
        B, C, H, W = latent_unquantized.shape

        encoded_seqs = self._get_encoded_seqs(previous_latents=previous_latents)
        encoded = torch.cat(encoded_seqs, dim=-2)
        # shape is [B', context_len*WpxWp, d_model]

        latent_patched, (nH, nW) = self.patcher(
            latent_unquantized, self.window_size_dec
        )  # [Bp, Wc*Wc, C_emb]
        Bp, seq_len_dec, C_emb = latent_patched.shape

        if k != 0:
            assert cloud_mask is None, "If k != 0, cloud_mask must be None."

        # Patch cloud mask to token space
        if cloud_mask is not None:
            assert cloud_mask.shape == (B, 1, H, W), (
                f"cloud_mask must be {(B, 1, H, W)}, got {cloud_mask.shape}"
            )
            cloud_mask_patched: torch.Tensor
            cloud_mask_patched, _ = self.patcher(
                cloud_mask.float(), self.window_size_dec
            )  # [Bp, Wc*Wc, 1]
            cloud_mask_patched = cloud_mask_patched.squeeze(
                -1
            ).bool()  # [Bp, Wc*Wc]

            if propagate_from_first_cloud:
                prediction_selector = self.propagate_from_first_cloud(
                    cloud_mask_patched
                )  # [Bp, Wc*Wc] True => predict from first cloud onward
            else:
                prediction_selector = (
                    cloud_mask_patched.bool()
                )  # [Bp, Wc*Wc] True => predict only at cloudy tokens
        else:
            prediction_selector = None

        # Quantize using STE
        latent_q_patched = self.round_st(latent_patched)

        # Initial pass with full quantized latent to get means/scales
        full_mean, full_scale, dec_output, z_joint = (
            self._terracodec_tt_encoder_forward(
                encoded_patched=encoded, latent_q_patched=latent_q_patched
            )
        )

        # Entropy bottleneck sample (y_tilde) and likelihoods
        output, likelihood_bits = self.bottleneck(
            inputs=latent_patched, scales=full_scale, means=full_mean
        )  # 'output' starts as quantized/noised latents

        if cloud_mask is None:
            # Regular Masking: from k onward replace by predicted mean
            for i in range(k, self.seq_len_dec):
                mu, _, dec_out, _ = self._terracodec_tt_encoder_forward(
                    encoded_patched=encoded,
                    latent_q_patched=output,
                    processed_zjoint=z_joint,
                )
                output[:, i, :] = self.round_st(mu[:, i, :])
                dec_output[:, i, :] = dec_out[:, i, :]
        else:
            # Cloud-aware Masking:
            assert prediction_selector is not None
            assert prediction_selector.shape == (Bp, self.seq_len_dec), (
                f"prediction_selector must have shape {(Bp, self.seq_len_dec)}, "
                f"got {prediction_selector.shape}"
            )

            for b in range(Bp):
                # indices we actually plan to predict (ascending)
                pred_idx = torch.nonzero(
                    prediction_selector[b], as_tuple=False
                ).squeeze(-1)  # [num_pred]
                if pred_idx.numel() == 0:
                    # No prediction needed, all tokens are cloud-free
                    continue

                if propagate_from_first_cloud:
                    start_idx = int(pred_idx[0].item())
                    assert pred_idx.tolist() == list(
                        range(start_idx, int(seq_len_dec))
                    ), (
                        "In propagate mode, expected pred_idx to be a contiguous range "
                        "from the first cloudy token to the end."
                    )

                # Autoregressive loop over selected tokens only
                for i in pred_idx.tolist():
                    mu, _, dec_out, _ = self._terracodec_tt_encoder_forward(
                        encoded_patched=encoded[
                            b : b + 1
                        ],  # [1, ctx_len*Wp*Wp, d]
                        latent_q_patched=output[b : b + 1],  # [1, Wc*Wc, C_emb]
                        processed_zjoint=z_joint[
                            b : b + 1
                        ],  # [1, ctx_len*Wp*Wp, d]
                    )
                    # Replace this token by the predicted mean (quantized STE) — real tokens remain as-is elsewhere
                    output[b, i, :] = self.round_st(mu[0, i, :])
                    dec_output[b, i, :] = dec_out[0, i, :]

        # Unpatch
        output_img = self.patcher.unpatch(t=output, nH=nH, nW=nW, crop=(H, W))
        decoder_features = self.patcher.unpatch(
            t=dec_output, nH=nH, nW=nW, crop=(H, W)
        )

        return TemporalEntropyModelOut(
            perturbed_latent=output_img,
            bits=likelihood_bits,
            features=decoder_features,  # z_cur
        )

    def validate_causal(
        self, latent_q_patched: Tensor, encoded: Tensor
    ) -> None:
        """
        Validate that the masking is causal
        """
        # Run model, relying on masks to guarantee causality
        masked_means, masked_scales, _, z_joint = (
            self._terracodec_tt_encoder_forward(
                encoded_patched=encoded, latent_q_patched=latent_q_patched
            )
        )

        # latent_q_patched.shape = [B * nH * nW, seq_len_dec, C]

        # Run model iteratively, feeding progressively more input
        current_inp = torch.full_like(latent_q_patched, fill_value=10.0)
        autoreg_means = torch.full_like(latent_q_patched, fill_value=10.0)
        autoreg_scales = torch.full_like(latent_q_patched, fill_value=10.0)

        for i in range(self.seq_len_dec):
            # Note that the transformer starts using a zero sybmol,
            # so internally,it shifts the input to the right.
            # Thus, on the very first call (where i == 0), we want a dummy
            # input. Only afterwards we start filling in input, using it shifted
            if i > 0:  # first token is the learnt StartSym
                current_inp[:, i - 1, :] = latent_q_patched[:, i - 1, :]
            mean_i, scale_i, _, _ = self._terracodec_tt_encoder_forward(
                encoded_patched=encoded,
                latent_q_patched=current_inp,
                processed_zjoint=z_joint,
            )
            autoreg_means[:, i, :] = mean_i[:, i, :]
            autoreg_scales[:, i, :] = scale_i[:, i, :]

        isclose_means = autoreg_means.isclose(masked_means).all()
        isclose_scales = autoreg_scales.isclose(masked_scales).all()
        causal = isclose_means and isclose_scales
        if not causal:
            msg_mean = "" if isclose_means else "means"
            msg_scales = "" if isclose_scales else "scales"
            raise ValueError(
                f"Larger than expected discrepancy: {msg_mean} {msg_scales}"
            )

    def compress(
        self,
        *,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
        run_decode: bool = False,
        validate_causal: bool = False,
    ) -> TemporalEntropyModelOut:
        """
        Compress and decompress autoregressively. Can only handle batch size 1.

        Named "range_code" in the original implementation.

        Args:
            latent_unquantized: unquantized latent
                of shape [1, C_emb, H_emb, W_emb]
            previous_latents: a sequence of length at least 1 and at most
                `context_len`containing PreviousLatent objects which hold
                2 tensors:
                    - quantized: [1, C_emb, H_emb, W_emb]
                    - processed: latents passed through the encoder, expected
                        shape [b', seq_len_enc, d_model]
            run_decode: bool, defaults to False. Whether to run the actual
                decoding

        If run_decode=False (default after the third P-frame) the function skips
        the final arithmetic decode because the quantised values are already
        known (lossless transmit-and-play trick).

        Returns:
            TemporalEntropyModelOut object with the following components:
                - perturbed_latent: tensor of ints with same shape as the input
                    tensor `latent_unquatnized` -- [1, C_emb, H_emb, W_emb]
                - bits: number of bits used to compress the input latent,
                    float tensor
                - features: features tensor from the decoder,
                    shape [1, d_model, H, W]
        """
        B, C, H, W = latent_unquantized.shape
        assert B == 1, "Cannot handle batch yet."

        encoded_seqs = self._get_encoded_seqs(previous_latents)
        encoded = torch.cat(encoded_seqs, -2)  # concat on seq_len_enc dim
        # previously coded latents,
        # shape is [B', context_len*seq_len_enc, d_model]

        latent_patched, (nH, nW) = self.patcher(
            latent_unquantized, self.window_size_dec
        )  # [B', Wc x Wc, C_emb]

        if validate_causal:
            self.validate_causal(
                latent_q_patched=latent_patched, encoded=encoded
            )

        # Encoding: compress to strings - strings is a list of len seq_len_dec
        strings, extra = self._encode(latent_patched, encoded)
        means, scales, dec_output, quantized = extra.values()

        # z_cur = dec_output
        decoder_features = self.patcher.unpatch(
            t=dec_output,
            nH=nH,
            nW=nW,
            crop=(H, W),
        )  # [1, d_model, H_emb, W_emb]

        # Count bits in each sequence, each string is a list of len 1
        bits = [sum(len(string[0]) * 8 for string in strings)]

        # decoding:
        if not run_decode:
            # For performance, since coding is lossless,
            # real decode can be skipped
            decoded = torch.round(latent_unquantized)
        else:
            use_output_from_encode = True
            decoded, dec_output = self._decode(
                strings,
                encoded,
                shape=(H, W, C),
                encoded_means=means,
                encoded_scales=scales,
                use_output_from_encode=use_output_from_encode,
            )

            dequantized = self.patcher.unpatch(
                t=quantized,
                nH=nH,
                nW=nW,
                crop=(H, W),
            )
            if use_output_from_encode:
                # This should pass if `use_output_from_encode=True`
                assert (decoded == dequantized).all(), "Something went wrong!"
                assert (decoder_features == dec_output).all(), (
                    "Decoder features do not match!"
                )

        return TemporalEntropyModelOut(
            perturbed_latent=decoded,
            bits=torch.tensor(bits, dtype=torch.float32),
            features=decoder_features,
        )

    def compress_multi_k_masking(
        self,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
        k: int,
        run_decode: bool = False,
        validate_causal: bool = False,
        forecast_mode: str = "deterministic",
    ) -> TemporalEntropyModelOut:
        """
        Compress and (optionally) decompress autoregressively with masking.
        Can only handle batch size 1. "k" is the number of real tokens per patch.
        Remaining tokens are predicted (0-bit).

        Args:
            latent_unquantized: [1, C_emb, H_emb, W_emb]
            previous_latents: PreviousLatent sequence for conditioning.
            k: number of tokens per patch to transmit (0..seq_len_dec)
            run_decode: if True, also run decode and sanity checks.
        """
        assert latent_unquantized.shape[0] == 1, (
            "compress_multi_k_masking only supports batch=1"
        )

        B, C, H, W = latent_unquantized.shape
        latent_patched, (nH, nW) = self.patcher(
            latent_unquantized, self.window_size_dec
        )
        Bp, seq_len_dec, C = latent_patched.shape

        # Build encoded context from previous latents
        encoded_seqs = self._get_encoded_seqs(previous_latents=previous_latents)
        encoded = torch.cat(encoded_seqs, dim=-2)

        # Optional strict causality check (same as in _encode)
        if validate_causal:
            self.validate_causal(
                latent_q_patched=latent_patched, encoded=encoded
            )

        # Masked encoding
        strings, extra = self._encode_masked(
            latent_patched=latent_patched,
            encoded=encoded,
            k=k,
            hallucination_mode=forecast_mode,
        )
        means, scales, dec_output, quantized = extra.values()

        # z_cur = dec_output
        decoder_features = self.patcher.unpatch(
            t=dec_output,
            nH=nH,
            nW=nW,
            crop=(H, W),
        )  # [1, d_model, H_emb, W_emb]

        # bits: single scalar per sequence (sum over tokens/channels)
        bits = [sum(len(string[0]) * 8 for string in strings)]

        # Unpatchify to original latent shape
        dequantized = self.patcher.unpatch(
            t=quantized,
            nH=nH,
            nW=nW,
            crop=(H, W),
        )

        if run_decode:
            use_output_from_encode = True
            decoded, dec_out = self._decode_masked(
                strings=strings,
                encoded=encoded,
                shape=(H, W, C),
                k=k,
                encoded_means=means,
                encoded_scales=scales,
                use_output_from_encode=use_output_from_encode,
            )

            if use_output_from_encode:
                assert (decoded == dequantized).all(), "Something went wrong!"
                assert (decoder_features == dec_out).all(), (
                    "Decoder features do not match!"
                )

        return TemporalEntropyModelOut(
            perturbed_latent=dequantized,
            bits=torch.tensor(bits, dtype=torch.float32),
            features=decoder_features,
        )

    def _encode(
        self, latent_patched: Tensor, encoded: Tensor
    ) -> Tuple[List[List[bytes]], Dict[str, Tensor]]:
        """
        Compress patched latents to strings

        Args:
            latent_patched: unquantized latent of shape [B', seq_len_dec, C_emb]
                B' = #patches * actual batch size, which is 1
                (for compress/decompress)
                seq_len_dec = Wc x Wc, C_emb = num_channels

            encoded: the "features" used to code the latent,
                i.e. what we condition on to predict the distribution of
                the current latent. This should be a
                tensor of shape [B', context_len*seq_len_enc, d_model],
                where seq_len_enc = Wp x Wp

        Returns:
            strings (list of bytestrings), extra (dict with tensors).
                NB: In theory, nothing in the  dict (e.g. means and scales)
                should be be used at decode time. In practice, the below
                ._decode method uses them, to avoid issues with non-determinism
                of the transformer.
        """
        Bp, seq_len_dec, C = latent_patched.shape  # B' , Wc x Wc , C_emb

        quantized = torch.full_like(latent_patched, 10.0)
        autoreg_means = torch.full_like(latent_patched, fill_value=100.0)
        autoreg_scales = torch.full_like(latent_patched, fill_value=100.0)

        # the decoder output (z_cur) is a tensor of shape
        # [B', seq_len_dec, d_model], where seq_len_dec = Wc x Wc
        dec_output_shape = (*latent_patched.shape[:-1], self.d_model)
        dec_output = torch.full(
            dec_output_shape,
            fill_value=100.0,
            dtype=torch.float32,
            device=quantized.device,
        )

        strings: List[List[bytes]] = []
        z_joint: Optional[Tensor] = None

        for i in range(seq_len_dec):  # 0 … 15
            mu_i, sigma_i, dec_out_i, z_joint = (
                self._terracodec_tt_encoder_forward(
                    encoded_patched=encoded,
                    latent_q_patched=quantized,
                    processed_zjoint=z_joint,
                )
            )  # shapes [b', seq_len_dec, C]*2, [b', seq_len_dec, d_model]

            # take slice i
            mu_i = mu_i[:, i, :]  # [Bp, C]
            sigma_i = sigma_i[:, i, :]
            token_gt = latent_patched[:, i, :]  # ground-truth latent

            # range-encode token_i
            q_i, string_i = self.bottleneck.compress(
                inputs=token_gt.unsqueeze(0),  # add B dim = 1
                scales=sigma_i.unsqueeze(0),
                means=mu_i.unsqueeze(0),
            )  # q_i: [1, Bp, C]

            assert len(string_i) == 1, (
                f"Expected string of length 1, got {len(string_i)}"
            )

            strings.append(string_i)  # length-1 list of bytes

            # Write back the quantized token for next iteration
            quantized[:, i, :] = q_i.squeeze(0)
            autoreg_means[:, i, :] = mu_i
            autoreg_scales[:, i, :] = sigma_i
            dec_output[:, i, :] = dec_out_i[:, i, :]  # [Bp, d_model]

        # NOTE: `autoreg_means` and `autoreg_scales` must not be used at decode
        # time. However, due to transofmer non-determinism, we return them and
        # allow "fake" decoding by setting use_output_from_encode=True
        # in `._decode
        extra = dict(
            means=autoreg_means,
            scales=autoreg_scales,
            dec_output=dec_output,
            quantized=quantized,
        )
        return strings, extra

    @torch.inference_mode()
    def _encode_masked(
        self,
        latent_patched: Tensor,  # [B', seq_len_dec, C_emb]
        encoded: Tensor,  # [B', context_len*seq_len_enc, d_model]
        k: int,
        hallucination_mode: str = "deterministic",
        tau: float = 1.0,
    ) -> Tuple[List[List[bytes]], Dict[str, Tensor]]:
        """
        Partially compress an autoregressive latent sequence and then
        hallucinate the remaining tokens.

        Workflow for a single 4×4 block (seq_len_dec = 16):
          - For i < k: encode ground-truth token i with range coding.
          - For i ≥ k: predict μ_i, σ_i with transformer and either:
              * deterministic: write μ_i
              * stochastic: sample y_i ~ N(μ_i, σ_i²)
            and store rounded value in `quantized` (no bit-string).

        Returns:
            strings          - list with length = k
            extra["means"]   - μ for every token
            extra["scales"]  - σ for every token
            extra["quantized"] - final latent containing gt+sampled tokens
            extra["dec_output"] - decoder features (z_cur)
        """
        Bp, seq_len_dec, C = latent_patched.shape  # B', Wc×Wc, C_emb
        assert 0 <= k <= seq_len_dec, f"k must be in 0..{seq_len_dec}"

        quantized = torch.full_like(latent_patched, 10.0)
        autoreg_means = torch.full_like(latent_patched, 100.0)
        autoreg_scales = torch.full_like(latent_patched, 100.0)

        dec_output_shape = (*latent_patched.shape[:-1], self.d_model)
        dec_output = torch.full(
            dec_output_shape,
            100.0,
            dtype=torch.float32,
            device=latent_patched.device,
        )

        strings: List[List[bytes]] = []
        z_joint: Optional[Tensor] = None

        eps = 1e-6
        sigma_min = 1e-3
        sigma_max = 10.0

        for i in range(seq_len_dec):
            mu, sigma_pre, dec_out, z_joint = (
                self._terracodec_tt_encoder_forward(
                    encoded_patched=encoded,
                    latent_q_patched=quantized,
                    processed_zjoint=z_joint,
                )
            )

            sigma = F.softplus(sigma_pre) + eps
            sigma = sigma.clamp_min(sigma_min).clamp_max(sigma_max)

            mu_i, sigma_i = mu[:, i, :], sigma[:, i, :]  # [Bp, C]

            if i < k:
                # compress ground-truth token i
                token_gt = latent_patched[:, i, :]
                q_i, str_i = self.bottleneck.compress(
                    inputs=token_gt.unsqueeze(0),
                    scales=sigma_i.unsqueeze(0),
                    means=mu_i.unsqueeze(0),
                )  # q_i: [1,WcxWc,C]
                # Write back the quantized token for next iteration
                quantized[:, i, :] = q_i.squeeze(0)
                assert len(str_i) == 1, (
                    f"Expected string of length 1, got {len(str_i)}"
                )
                strings.append(str_i)
            else:
                # hallucinate token i
                if hallucination_mode == "deterministic":
                    sample = mu_i
                else:
                    # stochastic: sample from N(μ, σ²)
                    noise = torch.randn_like(mu_i)
                    sample = mu_i + sigma_i * tau * noise

                quantized[:, i, :] = torch.round(sample)

            autoreg_means[:, i, :] = mu_i
            autoreg_scales[:, i, :] = sigma_i
            dec_output[:, i, :] = dec_out[:, i, :]

        extra = dict(
            means=autoreg_means,
            scales=autoreg_scales,
            dec_output=dec_output,
            quantized=quantized,
        )
        return strings, extra

    def _decode(
        self,
        strings: List[List[bytes]],
        encoded: torch.Tensor,
        shape: Sequence[int],  # (H, W, C)
        encoded_means: torch.Tensor,  # [B', seq_len_dec, C]
        encoded_scales: torch.Tensor,  # [B', seq_len_dec, C]
        use_output_from_encode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress strings → quantized latent (patched → unpatched).

        NOTE:
        - `strings` is a list of length = seq_len_dec, each element is a [bytestring] of length 1.
        - If `use_output_from_encode=True`, we use (means, scales) from the encode pass
            to avoid tiny transformer nondeterminism at decode time (research-only trick).

        Args:
            strings: list of bytestrings, length should be seq_len_dec
            encoded: previous latents, passed to transformer
            shape: H, W, C
            encoded_means: means from encode (compresss) step
            encoded_scales: scales from encode (compresss) step
            use_output_from_encode: use encoded_means and encoded_scales from
                the encode step (not real compression), or use the transformer
                to compute them.
                Note there could be a discrepancy between the two due to the
                non-deterministic nature of transformers (which makes them
                suboptimal for pmf prediction and use in compression),
                so for research purposes it could be acceptable to use the
                output from the encoder.
        Returns:
            A tuple of tensors:
                - decoded quantized latent, shape [1, C, H, W]
                - decoder features, shape [1, d_model, H, W]
        """
        H, W, C = shape
        device = encoded_means.device
        Bp, seq_len_dec, Cchk = encoded_means.shape
        assert Cchk == C, f"Channel mismatch: {Cchk} != {C}"

        # Build a fake patch shape to unpatch later
        fake: Patched = self.patcher(
            torch.ones((1, C, H, W), device=device), self.window_size_dec
        )
        nH, nW = fake.num_patches

        # Running buffer of already-decompressed tokens (AR state)
        decompressed = torch.full_like(
            fake.tensor, fill_value=0.0
        )  # [B', seq_len_dec, C]

        z_joint: Optional[torch.Tensor] = None
        dec_output = torch.empty(
            Bp, seq_len_dec, self.d_model, device=encoded.device
        )

        for i in range(seq_len_dec):
            # Predict μ, σ for token i given previously written tokens in `decompressed`
            mean_all, scale_all, dec_out, z_joint = (
                self._terracodec_tt_encoder_forward(
                    encoded_patched=encoded,
                    latent_q_patched=decompressed,  # AR context
                    processed_zjoint=z_joint,
                )
            )  # shapes: [B', seq_len_dec, C] x2

            actual_mean_i = mean_all[:, i, :]  # [B', C]
            actual_scale_i = scale_all[:, i, :]  # [B', C]
            target_mean_i = encoded_means[:, i, :]
            target_scale_i = encoded_scales[:, i, :]

            if use_output_from_encode:
                mean_i, scale_i = target_mean_i, target_scale_i
            else:
                mean_i, scale_i = actual_mean_i, actual_scale_i

            # Range decode token i using the agreed μ, σ
            decoded_i = self.bottleneck.decompress(
                strings=strings[i],
                scales=scale_i.unsqueeze(0),  # add B dim
                means=mean_i.unsqueeze(0),
            )  # -> [B', C]

            decompressed[:, i, :] = decoded_i

            # Save causal feature for token i
            dec_output[:, i, :] = dec_out[:, i, :]

        # Turn patched [B', seq_len_dec, C] back into [1, C, H, W]
        unpatched_latent_q = self.patcher.unpatch(
            t=decompressed, nH=nH, nW=nW, crop=(H, W)
        )

        unpatched_dec_output = self.patcher.unpatch(
            t=dec_output, nH=nH, nW=nW, crop=(H, W)
        )
        return unpatched_latent_q, unpatched_dec_output

    def _decode_masked(
        self,
        strings: List[List[bytes]],
        encoded: Tensor,
        shape: Sequence[int],  # (H, W, C)
        k: int,
        encoded_means: Tensor,
        encoded_scales: Tensor,
        use_output_from_encode: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Decompress masked strings → quantized latent (patched → unpatched),
        where only the first k tokens were actually compressed and the rest
        were hallucinated at encode time.

        Args:
            strings: list of bytestrings, length = k
            encoded: previous latents, passed to transformer
            shape: (H, W, C)
            k: number of tokens that were actually compressed
            encoded_means / encoded_scales: means/scales from encode
            use_output_from_encode: whether to reuse those (research trick).

        Returns:
            decoded quantized latent [1, C, H, W]
            decoder features        [1, d_model, H, W]
        """
        H, W, C = shape
        _device = encoded_means.device
        Bp, seq_len_dec, Cchk = encoded_means.shape
        assert Cchk == C, f"Channel mismatch: {Cchk} != {C}"

        fake: Patched = self.patcher(
            torch.ones((1, C, H, W), device=_device), self.window_size_dec
        )
        nH, nW = fake.num_patches
        assert 0 <= k <= seq_len_dec, f"k must be in 0..{seq_len_dec}"
        assert len(strings) == k, f"Expected {k} strings, got {len(strings)}"

        decompressed = torch.full_like(
            fake.tensor, 10.0
        )  # [B', seq_len_dec, C]
        z_joint: Optional[Tensor] = None
        dec_output = torch.empty(Bp, seq_len_dec, self.d_model, device=_device)

        eps = 1e-6
        sigma_min = 1e-3
        sigma_max = 10.0

        for i in range(seq_len_dec):
            mu, sigma_pre, dec_out, z_joint = (
                self._terracodec_tt_encoder_forward(
                    encoded_patched=encoded,
                    latent_q_patched=decompressed,
                    processed_zjoint=z_joint,
                )
            )

            sigma = F.softplus(sigma_pre) + eps
            sigma = sigma.clamp_min(sigma_min).clamp_max(sigma_max)

            actual_mean_i = mu[:, i, :]
            actual_scale_i = sigma[:, i, :]

            target_mean_i = encoded_means[:, i, :]
            target_scale_i = encoded_scales[:, i, :]

            if use_output_from_encode:
                mean_i, scale_i = target_mean_i, target_scale_i
            else:
                mean_i, scale_i = actual_mean_i, actual_scale_i

            if i < k:
                decoded_i = self.bottleneck.decompress(
                    strings=strings[i],
                    scales=scale_i.unsqueeze(0),
                    means=mean_i.unsqueeze(0),
                )
                decompressed[:, i, :] = decoded_i
            else:
                # hallucinate token i
                decompressed[:, i, :] = torch.round(mean_i)

            dec_output[:, i, :] = dec_out[:, i, :]

        assert len(strings) == k, f"Expected {k} strings, got {len(strings)}"
        unpatched_latent_q = self.patcher.unpatch(
            t=decompressed, nH=nH, nW=nW, crop=(H, W)
        )
        unpatched_dec_output = self.patcher.unpatch(
            t=dec_output, nH=nH, nW=nW, crop=(H, W)
        )
        return unpatched_latent_q, unpatched_dec_output

    def update(self, force: bool = False) -> bool:
        """
        Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later compress
        and decompress with an actual entropy coder.

        Args:
            force: overwrite previous values (default: False)

        Returns:
            updated: True if one of the bottlenecks was updated.
        """
        check = getattr(self.bottleneck, "update", None)
        if check is not None:
            bottleneck_updated = self.bottleneck.update(force=force)
        else:
            bottleneck_updated = False

        return bottleneck_updated
