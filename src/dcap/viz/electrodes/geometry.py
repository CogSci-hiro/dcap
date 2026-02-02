# =============================================================================
#                     ########################################
#                     #          GEOMETRY / LIMIT HELPERS     #
#                     ########################################
# =============================================================================
"""Geometry helpers for stable 3D plotting."""

from typing import Tuple

import numpy as np


# =============================================================================
#                     ########################################
#                     #               CONSTANTS               #
#                     ########################################
# =============================================================================
DEFAULT_PAD_FRACTION = 0.05
SINGLE_POINT_HALF_SPAN = 5.0  # coordinate units; used when only one point exists


def compute_equal_aspect_limits(
    xyz: np.ndarray,
    *,
    pad_fraction: float = DEFAULT_PAD_FRACTION,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Compute axis limits that approximate equal aspect ratio in Matplotlib 3D.

    Parameters
    ----------
    xyz
        Array of shape (n_points, 3) with finite coordinates.
    pad_fraction
        Fractional padding applied to the cubic bounding box.

    Returns
    -------
    xlim, ylim, zlim
        Each a (min, max) tuple.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (n, 3). Got {xyz.shape}.")

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    spans = maxs - mins

    # Handle single-point / degenerate spans
    max_span = float(np.max(spans))
    if not np.isfinite(max_span) or max_span <= 0.0:
        center = xyz[0]
        half_span = SINGLE_POINT_HALF_SPAN
        xlim = (float(center[0] - half_span), float(center[0] + half_span))
        ylim = (float(center[1] - half_span), float(center[1] + half_span))
        zlim = (float(center[2] - half_span), float(center[2] + half_span))
        return xlim, ylim, zlim

    # Cube bounds centered on the midpoints
    center = (mins + maxs) / 2.0
    half_span = (max_span / 2.0) * (1.0 + pad_fraction)

    xlim = (float(center[0] - half_span), float(center[0] + half_span))
    ylim = (float(center[1] - half_span), float(center[1] + half_span))
    zlim = (float(center[2] - half_span), float(center[2] + half_span))
    return xlim, ylim, zlim
