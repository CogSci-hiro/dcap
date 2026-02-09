# =============================================================================
# TRF analysis: wrapper-level kernel prediction (backend-agnostic)
# =============================================================================

from __future__ import annotations

import numpy as np


def _valid_keep_slice(n_times: int, lags_samp: np.ndarray) -> slice:
    lags = np.asarray(lags_samp, dtype=int)
    min_lag = int(lags.min())
    max_lag = int(lags.max())
    start = max(0, max_lag)
    stop = n_times + min(0, min_lag)
    if stop <= start:
        raise ValueError("Time series is too short for requested lags in 'valid' mode.")
    return slice(start, stop)


def predict_from_kernel(
    X: np.ndarray,
    *,
    coef: np.ndarray,
    intercept: np.ndarray,
    lags_samp: np.ndarray,
    mode: str = "valid",
) -> np.ndarray:
    """Predict using an explicit TRF kernel (no backend required).

    Parameters
    ----------
    X
        Predictor array, shape (n_times, n_features).
    coef
        TRF kernel weights, shape (n_lags, n_features, n_outputs).
    intercept
        Intercept, shape (n_outputs,).
    lags_samp
        Lag samples, shape (n_lags,). Convention: y[t] depends on x[t - lag].
    mode
        "valid" fills predictions only where all lags are in-bounds; other
        samples are set to 0.
        "same" returns predictions for all time points with implicit zero
        padding outside bounds.

    Returns
    -------
    y_hat : ndarray
        Shape (n_times, n_outputs).

    Usage example
    -------------
        y_hat = predict_from_kernel(
            X,
            coef=coef,
            intercept=intercept,
            lags_samp=lags_samp,
            mode="valid",
        )
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_times, n_features), got shape {X.shape}")

    coef = np.asarray(coef)
    intercept = np.asarray(intercept)

    if coef.ndim != 3:
        raise ValueError(f"coef must be 3D (n_lags, n_features, n_outputs), got {coef.shape}")
    if intercept.ndim != 1 or intercept.shape[0] != coef.shape[2]:
        raise ValueError("intercept must be 1D with length n_outputs")

    n_times, n_features = X.shape
    n_lags, n_features_c, _ = coef.shape
    if n_features_c != n_features:
        raise ValueError(f"X has n_features={n_features} but coef expects {n_features_c}")

    lags = np.asarray(lags_samp, dtype=int)
    if lags.shape[0] != n_lags:
        raise ValueError("lags_samp length must match coef first dimension")

    y_hat = np.zeros((n_times, coef.shape[2]), dtype=np.result_type(X.dtype, coef.dtype, intercept.dtype))

    if mode not in ("valid", "same"):
        raise ValueError(f"Unknown mode {mode!r}")

    if mode == "valid":
        keep = _valid_keep_slice(n_times, lags)
        t0, t1 = int(keep.start), int(keep.stop)
        for j, lag in enumerate(lags):
            lag = int(lag)
            x_start = t0 - lag
            x_stop = t1 - lag
            y_hat[t0:t1, :] += X[x_start:x_stop, :] @ coef[j, :, :]
        y_hat[t0:t1, :] += intercept[None, :]
        return y_hat

    # same: implicit zero padding, compute overlapping regions per lag
    for j, lag in enumerate(lags):
        lag = int(lag)
        if lag >= 0:
            src_start = 0
            src_stop = n_times - lag
            dst_start = lag
            dst_stop = dst_start + (src_stop - src_start)
        else:
            src_start = -lag
            src_stop = n_times
            dst_start = 0
            dst_stop = src_stop - src_start
        if src_stop <= src_start:
            continue
        y_hat[dst_start:dst_stop, :] += X[src_start:src_stop, :] @ coef[j, :, :]
    y_hat += intercept[None, :]
    return y_hat
