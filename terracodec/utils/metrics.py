# Copyright IBM Corp. 2025
# License: Apache-2.0

"""
Unified metrics for Earth Observation compression evaluation.

This module provides standardized implementations of quality metrics
(PSNR, SSIM, MS-SSIM, MSE) that operate on tensors in raw reflectance
units with shape [B, T, C, H, W].
"""

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

MetricMode = Literal["auto", "10k", "65k"]


def _get_data_range_and_clamp(
    tensor: torch.Tensor, mode: MetricMode
) -> Tuple[torch.Tensor, float]:
    """
    Get data range and optionally clamp tensor based on mode.

    Args:
        tensor: Input tensor in raw reflectance units
        mode: Metric computation mode

    Returns:
        Tuple of (processed_tensor, data_range)
    """
    if mode == "auto":
        # Auto-detect min/max range per batch
        data_range = tensor.max().item() - tensor.min().item()
        return tensor, data_range
    elif mode == "10k":
        # Clamp to [0, 10000] range, subtract 1000 offset for S2 data
        clamped = torch.clamp(tensor - 1000.0, 0.0, 10000.0)
        return clamped, 10000.0
    elif mode == "65k":
        # Use fixed range assuming 16-bit data [0, 65535]
        return tensor, 65535.0
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: MetricMode = "auto",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Mean Squared Error between prediction and target.

    Args:
        pred: Predicted tensor [B, T, C, H, W] in raw reflectance units
        target: Target tensor [B, T, C, H, W] in raw reflectance units
        mode: Metric computation mode (auto, 10k, 65k)
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        MSE value(s)
    """
    pred_processed, _ = _get_data_range_and_clamp(pred, mode)
    target_processed, _ = _get_data_range_and_clamp(target, mode)

    return F.mse_loss(pred_processed, target_processed, reduction=reduction)


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: MetricMode = "auto",
    dim: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio between prediction and target.

    Args:
        pred: Predicted tensor [B, T, C, H, W] in raw reflectance units
        target: Target tensor [B, T, C, H, W] in raw reflectance units
        mode: Metric computation mode (auto, 10k, 65k)
        dim: Dimensions to average over (default: average over all dimensions)

    Returns:
        PSNR value(s)
    """
    pred_processed, data_range = _get_data_range_and_clamp(pred, mode)
    target_processed, _ = _get_data_range_and_clamp(target, mode)

    # Reshape to [N, C, H, W] for torchmetrics
    B, T, C, H, W = pred_processed.shape
    pred_reshaped = pred_processed.view(B * T, C, H, W)
    target_reshaped = target_processed.view(B * T, C, H, W)

    if mode == "auto":
        data_range = float((target.max() - target.min()).item())

    return peak_signal_noise_ratio(
        pred_reshaped,
        target_reshaped,
        data_range=data_range,
        dim=dim if mode != "auto" else None,
    )


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: MetricMode = "auto",
) -> torch.Tensor:
    """
    Compute Structural Similarity Index Measure between prediction and target.

    Args:
        pred: Predicted tensor [B, T, C, H, W] in raw reflectance units
        target: Target tensor [B, T, C, H, W] in raw reflectance units
        mode: Metric computation mode (auto, 10k, 65k)

    Returns:
        SSIM value(s)
    """
    pred_processed, data_range = _get_data_range_and_clamp(pred, mode)
    target_processed, _ = _get_data_range_and_clamp(target, mode)

    # Reshape to [N, C, H, W] for torchmetrics
    B, T, C, H, W = pred_processed.shape
    pred_reshaped = pred_processed.view(B * T, C, H, W)
    target_reshaped = target_processed.view(B * T, C, H, W)

    return structural_similarity_index_measure(  # type: ignore
        pred_reshaped,
        target_reshaped,
        data_range=data_range if mode != "auto" else None,
    )


# Utils for MS-SSIM

# Default MS-SSIM betas
_DEFAULT_BETAS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _max_allowed_kernel(height: int, width: int, L: int) -> int:
    """From your earlier logic: satisfies TorchMetrics pre-check that depends on (kernel_size, L)."""
    d = max(1, (L - 1)) ** 2
    min_hw = min(height, width)
    # Need min(H, W) > (k - 1) * d  =>  k_max = floor((minHW - 1)/d) + 1
    k_max = ((min_hw - 1) // d) + 1
    # enforce odd and at least 1
    if k_max % 2 == 0:
        k_max -= 1
    return max(k_max, 1)


def _last_scale_size(h: int, w: int, L: int) -> tuple[int, int]:
    """Smallest spatial size used by MS-SSIM given L scales (since we pool after each of the first L-1 passes)."""
    div = 2 ** (L - 1)
    return max(1, h // div), max(1, w // div)


def _sigma_for_kernel(k: int) -> float:
    """
    Choose sigma so that TorchMetrics' internal gauss size equals `k`.

    TorchMetrics sets:
        gauss_kernel_size = int(3.5*s + 0.5) * 2 + 1

    For a target odd k >= 3, this function returns a sigma that yields
    exactly that kernel size.
    """
    if k < 3 or k % 2 == 0:
        raise ValueError(f"Kernel size must be an odd integer >=3, got {k}.")

    target = (k - 1) // 2
    lo = (target - 0.5) / 3.5
    hi = (target + 0.5) / 3.5
    sigma = 0.5 * (lo + hi)  # midpoint of valid interval
    return max(sigma, 0.15)  # keep positive lower bound


def _choose_ms_ssim_params(
    H: int, W: int, prefer_L: int = 5, prefer_k: int = 11
) -> tuple[int, float, tuple[float, ...]]:
    """
    Find a balanced (k, sigma, betas) that:
      - passes TorchMetrics' size checks
      - keeps reflection pad < current dim at the coarsest scale
      - respects the 2**L downsampling requirement
    """
    min_hw = min(H, W)
    max_L_possible = int(math.log2(min_hw)) if min_hw > 0 else 1
    max_L_possible = max(1, max_L_possible)  # at least 1 scale

    # Try from preferred L down to 1
    for L in range(min(prefer_L, max_L_possible), 0, -1):
        # Must also satisfy the downsampling requirement: min(H, W) >= 2**L
        if min_hw < (1 << L):
            continue

        # Constraint A (TorchMetrics precheck): k <= _max_allowed_kernel
        k1 = _max_allowed_kernel(H, W, L)

        # Constraint B (pad at last scale): (k-1)//2 < min(H_last, W_last)
        H_last, W_last = _last_scale_size(H, W, L)
        k2 = min(2 * H_last - 1, 2 * W_last - 1)

        k_max = max(1, min(k1, k2))

        # Pick k as large as allowed but not exceeding preference; enforce odd >= 3
        k = min(prefer_k, k_max)
        if k % 2 == 0:
            k -= 1
        if k < 3:
            # try the largest odd >=3 that fits
            k = max(3, k_max if k_max % 2 == 1 else k_max - 1)

        # still infeasible? then reduce scales
        if k < 3:
            continue

        # Choose sigma to match k under TorchMetrics' Gaussian rule
        sigma = _sigma_for_kernel(k)
        betas = _DEFAULT_BETAS[:L]
        return k, sigma, betas

    # Fallback: minimal viable parameters
    k = 3
    sigma = _sigma_for_kernel(k)
    betas = _DEFAULT_BETAS[:1]
    return k, sigma, betas


def compute_ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: "MetricMode" = "auto",
) -> torch.Tensor:
    """
    Compute Multi-Scale SSIM with parameters that auto-adapt to the input size.
    pred/target: [B, T, C, H, W]
    """
    pred_processed, data_range = _get_data_range_and_clamp(pred, mode)
    target_processed, _ = _get_data_range_and_clamp(target, mode)

    B, T, C, H, W = pred_processed.shape
    pred_reshaped = pred_processed.view(B * T, C, H, W)
    target_reshaped = target_processed.view(B * T, C, H, W)

    # Pick params that are safe at all scales (including the coarsest one)
    kernel_size, sigma, betas = _choose_ms_ssim_params(
        H, W, prefer_L=5, prefer_k=11
    )

    # If you want exact parity with 256×256 legacy runs:
    if H == W == 256:
        kernel_size = 11
        sigma = 1.5
        betas = _DEFAULT_BETAS

    return multiscale_structural_similarity_index_measure(
        pred_reshaped,
        target_reshaped,
        data_range=data_range if mode != "auto" else None,
        gaussian_kernel=True,  # keep Gaussian
        kernel_size=kernel_size,  # used for pre-checks in TorchMetrics
        sigma=sigma,  # makes the actual gaussian window size == kernel_size
        betas=betas,
    )


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: MetricMode = "auto",
    per_band: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute all quality metrics between prediction and target.

    Args:
        pred: Predicted tensor [B, T, C, H, W] in raw reflectance units
        target: Target tensor [B, T, C, H, W] in raw reflectance units
        mode: Metric computation mode (auto, 10k, 65k)
        per_band: If True, compute metrics per band.

    Returns:
        Dictionary or tuple of (MSE, PSNR, SSIM, MS-SSIM) values
    """
    # Assert that shapes match and that they are 5D
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        )
    if pred.dim() != 5:
        raise ValueError(
            f"Expected 5D tensors ([B, T, C, H, W]), got {pred.dim()}D"
        )

    if per_band:
        # Compute per band by reshaping C into batch dimension
        B, T, C, H, W = pred.shape
        pred = (
            pred.permute(0, 2, 1, 3, 4)
            .reshape(B * C, T, H, W)
            .unsqueeze(2)
            .contiguous()
        )
        target = (
            target.permute(0, 2, 1, 3, 4).reshape(B * C, T, H, W).unsqueeze(2)
        ).contiguous()
        # both now [B*C, T, 1, H, W]

    mse = compute_mse(pred, target, mode)
    psnr = compute_psnr(pred, target, mode, dim=(1, 2, 3))
    ssim = compute_ssim(pred, target, mode)
    ms_ssim = compute_ms_ssim(pred, target, mode)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
    }
