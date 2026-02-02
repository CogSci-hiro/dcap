# =============================================================================
#                     ########################################
#                     #          GEOMETRY / LIMIT HELPERS     #
#                     ########################################
# =============================================================================
"""Geometry helpers for stable 2D projection plotting."""

from typing import Tuple

import numpy as np


# =============================================================================
#                     ########################################
#                     #               CONSTANTS               #
#                     ########################################
# =============================================================================
DEFAULT_PAD_FRACTION = 0.06
SINGLE_POINT_HALF_SPAN = 5.0  # coordinate units


def compute_equal_aspect_limits_2d(
    xy: np.ndarray,
    *,
    pad_fraction: float = DEFAULT_PAD_FRACTION,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute 2D axis limits with equal aspect.

    Parameters
    ----------
    xy
        Array of shape (n_points, 2).
    pad_fraction
        Fractional padding applied to the square bounding box.

    Returns
    -------
    xlim, ylim
        Each a (min, max) tuple.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (n, 2). Got {xy.shape}.")

    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    spans = maxs - mins

    max_span = float(np.max(spans))
    if not np.isfinite(max_span) or max_span <= 0.0:
        center = xy[0]
        half_span = SINGLE_POINT_HALF_SPAN
        xlim = (float(center[0] - half_span), float(center[0] + half_span))
        ylim = (float(center[1] - half_span), float(center[1] + half_span))
        return xlim, ylim

    center = (mins + maxs) / 2.0
    half_span = (max_span / 2.0) * (1.0 + pad_fraction)

    xlim = (float(center[0] - half_span), float(center[0] + half_span))
    ylim = (float(center[1] - half_span), float(center[1] + half_span))
    return xlim, ylim
