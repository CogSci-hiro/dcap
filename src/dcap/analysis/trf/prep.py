# =============================================================================
#                         TRF analysis: preprocessing
# =============================================================================
#
# Array utilities used by TRF task adapters and backends:
# - z-scoring
# - polyphase resampling
# - cropping
# - stacking runs into (n_times, n_epochs, n_features) layout used by MNE
#
# =============================================================================

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class ZScoreConfig:
    """
    Z-score configuration.

    Parameters
    ----------
    axis : int
        Axis along which to compute mean/std.
    eps : float
        Small constant to avoid division by zero.

    Usage example
    -------------
        cfg = ZScoreConfig(axis=0, eps=1e-12)
    """

    axis: int = 0
    eps: float = 1e-12


# =============================================================================
# Z-scoring
# =============================================================================

def zscore(x: np.ndarray, config: ZScoreConfig = ZScoreConfig()) -> np.ndarray:
    """
    Z-score an array along a given axis.

    Parameters
    ----------
    x : ndarray
        Input array.
    config : ZScoreConfig
        Z-score settings.

    Returns
    -------
    x_z : ndarray
        Z-scored array, same shape as `x`.

    Usage example
    -------------
        x_z = zscore(x)
    """

    mean = np.mean(x, axis=config.axis, keepdims=True)
    std = np.std(x, axis=config.axis, ddof=0, keepdims=True)
    return (x - mean) / (std + config.eps)


# =============================================================================
# Resampling
# =============================================================================

def resample_poly_1d(x: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    """
    Resample a 1D signal using polyphase filtering.

    Parameters
    ----------
    x : ndarray, shape (n_times,)
        Signal sampled at `sfreq_in`.
    sfreq_in : float
        Input sampling frequency (Hz).
    sfreq_out : float
        Output sampling frequency (Hz).

    Returns
    -------
    x_rs : ndarray, shape (n_times_out,)
        Resampled signal.

    Usage example
    -------------
        x_rs = resample_poly_1d(x, sfreq_in=44100.0, sfreq_out=100.0)
    """

    if x.ndim != 1:
        raise ValueError("`x` must be 1D.")
    _validate_sfreq(sfreq_in, sfreq_out)
    up, down = _rational_approximation(sfreq_out / sfreq_in)
    return resample_poly(x, up=up, down=down)


def resample_poly_time_last(x: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    """
    Resample an array whose time axis is the first axis (axis=0).

    Supports:
    - (n_times,)
    - (n_times, n_features)
    - (n_times, n_epochs, n_features)

    Parameters
    ----------
    x : ndarray
        Input array with time axis=0.
    sfreq_in : float
        Input sampling frequency (Hz).
    sfreq_out : float
        Output sampling frequency (Hz).

    Returns
    -------
    x_rs : ndarray
        Resampled array with time axis still first.

    Usage example
    -------------
        X_rs = resample_poly_time_last(X, sfreq_in=1000.0, sfreq_out=100.0)
    """

    _validate_sfreq(sfreq_in, sfreq_out)

    if sfreq_in == sfreq_out:
        return x

    up, down = _rational_approximation(sfreq_out / sfreq_in)

    if x.ndim == 1:
        return resample_poly(x, up=up, down=down)

    if x.ndim == 2:
        # (time, feature)
        return np.vstack([resample_poly(x[:, idx], up=up, down=down) for idx in range(x.shape[1])]).T

    if x.ndim == 3:
        # (time, epoch, feature)
        n_epochs = x.shape[1]
        n_features = x.shape[2]
        out_epochs = []
        for epoch_idx in range(n_epochs):
            epoch = x[:, epoch_idx, :]  # (time, feature)
            epoch_rs = np.vstack(
                [resample_poly(epoch[:, feat_idx], up=up, down=down) for feat_idx in range(n_features)]
            ).T
            out_epochs.append(epoch_rs)
        return np.stack(out_epochs, axis=1)

    raise ValueError("`x` must be 1D, 2D, or 3D with time axis=0.")


def _validate_sfreq(sfreq_in: float, sfreq_out: float) -> None:
    if sfreq_in <= 0 or sfreq_out <= 0:
        raise ValueError("Sampling frequencies must be > 0.")


def _rational_approximation(ratio: float, max_denominator: int = 1000) -> Tuple[int, int]:
    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


# =============================================================================
# Cropping / stacking
# =============================================================================

def crop_by_samples(x: np.ndarray, start: int, stop: int) -> np.ndarray:
    """
    Crop along time axis=0.

    Parameters
    ----------
    x : ndarray
        Array with time axis=0.
    start : int
        Start sample (inclusive).
    stop : int
        Stop sample (exclusive).

    Returns
    -------
    x_crop : ndarray
        Cropped array.

    Usage example
    -------------
        x_crop = crop_by_samples(x, start=100, stop=500)
    """

    if start < 0 or stop < 0 or stop < start:
        raise ValueError("Invalid crop bounds.")
    if x.shape[0] < stop:
        raise ValueError("Crop stop exceeds array length.")
    return x[start:stop, ...]


def stack_time_epoch_feature(runs: Sequence[np.ndarray]) -> np.ndarray:
    """
    Stack a list of (n_times, n_features) arrays into (n_times, n_epochs, n_features).

    Parameters
    ----------
    runs : sequence of ndarray
        Each run must be shape (n_times, n_features).

    Returns
    -------
    stacked : ndarray, shape (n_times, n_epochs, n_features)

    Usage example
    -------------
        X = stack_time_epoch_feature([X_run1, X_run2, X_run3, X_run4])
    """

    if len(runs) == 0:
        raise ValueError("Need at least one run to stack.")

    first = runs[0]
    if first.ndim != 2:
        raise ValueError("Each run must have shape (n_times, n_features).")

    n_times, n_features = first.shape

    for idx, arr in enumerate(runs):
        if arr.ndim != 2:
            raise ValueError(f"Run {idx} is not 2D.")
        if arr.shape != (n_times, n_features):
            raise ValueError(
                "All runs must have the same shape. "
                f"Expected {(n_times, n_features)}, got {arr.shape} at run {idx}."
            )

    return np.stack(runs, axis=1)
