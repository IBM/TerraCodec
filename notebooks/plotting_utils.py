import matplotlib.pyplot as plt
import numpy as np
import torch


def rgb_smooth_quantiles(array, tolerance=0.02, scaling=0.5, default=2000):
    """
    array: numpy array with dimensions [C, H, W]
    returns 0-1 scaled array
    """

    # Get scaling thresholds for smoothing the brightness
    limit_low, median, limit_high = np.quantile(
        array, q=[tolerance, 0.5, 1.0 - tolerance]
    )
    limit_high = limit_high.clip(
        default
    )  # Scale only pixels above default value
    limit_low = limit_low.clip(0, 1000)  # Scale only pixels below 1000
    limit_low = np.where(
        median > default / 2, limit_low, 0
    )  # Make image only darker if it is not dark already

    # Smooth very dark and bright values using linear scaling
    array = np.where(
        array >= limit_low, array, limit_low + (array - limit_low) * scaling
    )
    array = np.where(
        array <= limit_high, array, limit_high + (array - limit_high) * scaling
    )

    # Update scaling params using a 10th of the tolerance for max value
    limit_low, limit_high = np.quantile(
        array, q=[tolerance / 10, 1.0 - tolerance / 10]
    )
    limit_high = limit_high.clip(
        default, 20000
    )  # Scale only pixels above default value
    limit_low = limit_low.clip(0, 500)  # Scale only pixels below 500
    limit_low = np.where(
        median > default / 2, limit_low, 0
    )  # Make image only darker if it is not dark already

    # Scale data to 0-255
    array = (array - limit_low) / (limit_high - limit_low)

    return array


def s2_to_rgb(data, smooth_quantiles=True, default=2000):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    if len(data.shape) == 4:
        # Remove batch or time dim
        data = data[0]

    # Select RGB channels
    if data.shape[0] == 12 or data.shape[0] == 13:
        # assuming channel first
        rgb = data[[3, 2, 1]].transpose((1, 2, 0))
    else:
        # assuming channel last
        rgb = data[:, :, [3, 2, 1]]

    if smooth_quantiles:
        rgb = rgb_smooth_quantiles(rgb, default=default)
    else:
        rgb = rgb / default

    # to uint8
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)

    return rgb


def plot_s2(data, ax=None, smooth_quantiles=True, default=2000):
    rgb = s2_to_rgb(data, smooth_quantiles=smooth_quantiles)

    if ax is None:
        plt.imshow(rgb)
        plt.axis("off")
        plt.show()
    else:
        ax.imshow(rgb)
        ax.axis("off")
