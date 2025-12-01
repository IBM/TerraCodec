# Copyright IBM Corp. 2025
# License: Apache-2.0

import torch


def s2_smooth_quantiles(
    array: torch.Tensor,
    tolerance: float = 0.02,
    scaling: float = 0.5,
    default: float = 2000.0,
) -> torch.Tensor:
    """
    Smooth the brightness of a tensor using quantile scaling.

    Args:
        array: The input tensor with shape (N, C, H, W) or (C, H, W)
            or (H, W) for grayscale
        tolerance: The quantile tolerance for scaling
        scaling: The scaling factor for brightness adjustment
        default: The default value to clip the brightness

    Returns:
        The scaled tensor with smoothed brightness
    """
    if array.ndim == 2:
        array = array[None, :, :]  # [1, H, W] for grayscale

    # Compute the low, median, and high quantiles
    q = torch.tensor(
        [tolerance, 0.5, 1.0 - tolerance],
        device=array.device,
        dtype=array.dtype,
    )
    limit_low, median, limit_high = torch.quantile(array, q)

    # Clip the high and low thresholds
    limit_high = limit_high.clamp(
        max=default
    )  # only scale pixels above `default`
    limit_low = limit_low.clamp(
        min=0.0, max=1000.0
    )  # only scale pixels below 1000

    # If the image is already bright (median > default/2), keep limit_low;
    # otherwise use 0
    limit_low = torch.where(
        median > (default / 2),
        limit_low,
        torch.tensor(0.0, device=array.device),
    )

    # First-pass smoothing
    array = torch.where(
        array >= limit_low, array, limit_low + (array - limit_low) * scaling
    )
    array = torch.where(
        array <= limit_high,
        array,
        limit_high + (array - limit_high) * scaling,
    )

    # Recompute tighter quantiles (1/10th tolerance)
    q2 = torch.tensor(
        [tolerance / 10, 1.0 - tolerance / 10],
        device=array.device,
        dtype=array.dtype,
    )
    limit_low, limit_high = torch.quantile(array, q2)

    # Clip new thresholds
    limit_high = limit_high.clamp(min=default, max=20000.0)
    limit_low = limit_low.clamp(min=0.0, max=500.0)
    limit_low = torch.where(
        median > (default / 2),
        limit_low,
        torch.tensor(0.0, device=array.device),
    )

    # Scale into [0, 1] (or multiply by 255 if you really need 0–255)
    array = (array - limit_low) / (limit_high - limit_low)

    return array
