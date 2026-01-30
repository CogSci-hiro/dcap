# =============================================================================
#                 TRF analysis: lagged design matrix (X)
# =============================================================================
#
# Build a time-lagged (a.k.a. delayed) design matrix from a stimulus time series.
# Used for TRF / encoding models: y[t] ≈ Σ_k X[t, k] * w[k]
#
# Supports:
# - negative lags (stimulus leads response)
# - positive lags (stimulus lags response)
# - 1D (n_times,) or 2D (n_times, n_features) stimulus
#
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

# review
# =============================================================================
# Configuration
# =============================================================================

LagMode = Literal["valid", "same"]


@dataclass(frozen=True, slots=True)
class LagConfig:
    """
    Configuration for time lags.

    Parameters
    ----------
    tmin_ms : float
        Minimum lag in milliseconds. Negative values are allowed.
    tmax_ms : float
        Maximum lag in milliseconds. Must be >= tmin_ms.
    step_ms : float
        Lag step in milliseconds. Must be > 0.
    mode : {"valid", "same"}
        - "valid": trim edges so all lags are fully observed (recommended).
        - "same": keep original n_times and zero-pad edge values.
    """

    tmin_ms: float
    tmax_ms: float
    step_ms: float
    mode: LagMode = "valid"


# =============================================================================
# Public API
# =============================================================================

def make_lag_samples(
    sfreq: float,
    config: LagConfig,
) -> np.ndarray:
    """
    Create integer lag offsets in samples.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the stimulus (Hz).
    config : LagConfig
        Lag configuration in milliseconds.

    Returns
    -------
    lags_samp : ndarray of int, shape (n_lags,)
        Lags in samples, sorted ascending.

    Notes
    -----
    We use rounding to the nearest sample for each lag.

    Usage example
    -------------
        lcfg = LagConfig(tmin_ms=-100, tmax_ms=400, step_ms=10)
        lags = make_lag_samples(sfreq=100.0, config=lcfg)
    """

    if sfreq <= 0:
        raise ValueError("`sfreq` must be > 0.")
    if config.step_ms <= 0:
        raise ValueError("`step_ms` must be > 0.")
    if config.tmax_ms < config.tmin_ms:
        raise ValueError("`tmax_ms` must be >= `tmin_ms`.")

    lag_ms = np.arange(config.tmin_ms, config.tmax_ms + 1e-9, config.step_ms, dtype=float)
    lags_samp = np.rint(lag_ms * sfreq / 1000.0).astype(int)

    # Ensure strictly nondecreasing; duplicates can happen when step is < 1 sample.
    lags_samp = np.unique(lags_samp)

    return lags_samp


def build_lagged_design_matrix(
    stimulus: np.ndarray,
    sfreq: float,
    config: LagConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a lagged design matrix from stimulus time series.

    Parameters
    ----------
    stimulus : ndarray, shape (n_times,) or (n_times, n_features)
        Stimulus samples at `sfreq`.
    sfreq : float
        Sampling frequency of the stimulus (Hz).
    config : LagConfig
        Lag configuration.

    Returns
    -------
    X : ndarray, shape (n_rows, n_features * n_lags)
        Lagged design matrix. Columns are grouped by lag:
        [lag0_feat0..featF, lag1_feat0..featF, ...] with lags ascending.
    lags_samp : ndarray of int, shape (n_lags,)
        Lags in samples used to construct X (ascending).
    row_indices : ndarray of int, shape (n_rows,)
        Indices into the original time axis corresponding to each row of X.
        Useful for aligning the response: y_aligned = y[row_indices].

    Notes
    -----
    Convention:
    For a given lag `L` (in samples), the design matrix uses:
        X_row(t, L) = stimulus[t - L]
    So:
    - L > 0 uses earlier stimulus (past) to predict y[t]
    - L < 0 uses future stimulus (stimulus leads response)

    Mode behavior:
    - "valid": rows are restricted to times where all `t - L` are in-bounds.
    - "same": keeps all times and zero-pads any out-of-bounds samples.

    Usage example
    -------------
        lcfg = LagConfig(tmin_ms=-100, tmax_ms=400, step_ms=10, mode="valid")
        X, lags, idx = build_lagged_design_matrix(env, sfreq=100.0, config=lcfg)
    """

    stimulus_2d = _as_2d_stimulus(stimulus)
    n_times, n_features = stimulus_2d.shape

    lags_samp = make_lag_samples(sfreq=sfreq, config=config)
    n_lags = int(lags_samp.size)

    if n_times == 0:
        raise ValueError("`stimulus` must have at least one sample.")
    if n_lags == 0:
        raise ValueError("Lag range produced zero lags. Check tmin/tmax/step.")

    if config.mode == "valid":
        row_indices = _valid_row_indices(n_times=n_times, lags_samp=lags_samp)
        X = np.empty((row_indices.size, n_features * n_lags), dtype=stimulus_2d.dtype)

        for lag_idx, lag in enumerate(lags_samp):
            # For each target time t (row_indices), pull stimulus[t - lag]
            src_idx = row_indices - lag
            block = stimulus_2d[src_idx, :]
            col0 = lag_idx * n_features
            X[:, col0 : col0 + n_features] = block

        return X, lags_samp, row_indices

    if config.mode == "same":
        row_indices = np.arange(n_times, dtype=int)
        X = np.zeros((n_times, n_features * n_lags), dtype=stimulus_2d.dtype)

        for lag_idx, lag in enumerate(lags_samp):
            src_idx = row_indices - lag
            in_bounds = (src_idx >= 0) & (src_idx < n_times)

            col0 = lag_idx * n_features
            X[in_bounds, col0 : col0 + n_features] = stimulus_2d[src_idx[in_bounds], :]

        return X, lags_samp, row_indices

    raise ValueError(f"Unknown mode: {config.mode!r}. Expected 'valid' or 'same'.")


# =============================================================================
# Helpers
# =============================================================================

def _as_2d_stimulus(stimulus: np.ndarray) -> np.ndarray:
    if stimulus.ndim == 1:
        return stimulus[:, np.newaxis]
    if stimulus.ndim == 2:
        return stimulus
    raise ValueError("`stimulus` must be 1D (n_times,) or 2D (n_times, n_features).")


def _valid_row_indices(n_times: int, lags_samp: np.ndarray) -> np.ndarray:
    """
    Indices t such that all t - lag are within [0, n_times-1].
    """
    min_lag = int(lags_samp.min())
    max_lag = int(lags_samp.max())

    # Need: 0 <= t - lag <= n_times-1 for all lag
    # Worst cases:
    #   for max_lag: t - max_lag >= 0  => t >= max_lag
    #   for min_lag: t - min_lag <= n_times-1 => t <= n_times-1 + min_lag
    t_start = max_lag
    t_stop = (n_times - 1) + min_lag

    if t_stop < t_start:
        raise ValueError(
            "Lag window is too wide for the stimulus length in 'valid' mode. "
            f"n_times={n_times}, min_lag={min_lag}, max_lag={max_lag}."
        )

    return np.arange(t_start, t_stop + 1, dtype=int)
