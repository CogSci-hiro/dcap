from __future__ import annotations

import numpy as np

from dcap.features.types import DerivativeMode


def apply_derivative(
    *,
    x: np.ndarray,
    sfreq: float,
    mode: DerivativeMode,
) -> np.ndarray:
    """Apply derivative / abs-derivative along the last axis.

    Parameters
    ----------
    x
        Input array with time on last axis.
    sfreq
        Sampling frequency of `x`.
    mode
        "none" | "diff" | "absdiff"

    Returns
    -------
    np.ndarray
        Processed array, same shape as `x`.

    Usage example
    -------------
        y = apply_derivative(x=env, sfreq=100.0, mode="absdiff")
    """
    if mode == "none":
        return x

    # discrete time derivative: dx/dt ≈ (x[t] - x[t-1]) * sfreq
    dx = np.diff(x, axis=-1, prepend=x[..., :1]) * float(sfreq)
    if mode == "diff":
        return dx
    if mode == "absdiff":
        return np.abs(dx)

    raise ValueError(f"Unknown DerivativeMode: {mode}")
