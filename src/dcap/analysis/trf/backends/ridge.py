# =============================================================================
# TRF analysis: built-in ridge backend (explicit lag design)
# =============================================================================

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BackendFitResult, TrfBackend


def _build_lagged_matrix(X: np.ndarray, lags_samp: np.ndarray, *, mode: str) -> tuple[np.ndarray, slice]:
    """Build lagged design matrix.

    Returns
    -------
    X_lag : ndarray
        Shape (n_eff, n_features * n_lags).
    keep_slice : slice
        Slice over the original time axis corresponding to rows of X_lag / Y_eff.
    """
    n_times, n_features = X.shape
    lags = np.asarray(lags_samp, dtype=int)
    min_lag = int(lags.min())
    max_lag = int(lags.max())

    if mode not in ("valid", "same"):
        raise ValueError(f"Unknown mode {mode!r}")

    if mode == "valid":
        start = max(0, max_lag)
        stop = n_times + min(0, min_lag)
        if stop <= start:
            raise ValueError("Time series is too short for requested lags in 'valid' mode.")
        keep = slice(start, stop)
        n_eff = stop - start
    else:
        keep = slice(0, n_times)
        n_eff = n_times

    X_lag = np.zeros((n_eff, n_features * lags.size), dtype=X.dtype)
    for j, lag in enumerate(lags):
        if mode == "valid":
            # Row t corresponds to original time (keep.start + t)
            t0 = keep.start - lag
            t1 = keep.stop - lag
            X_lag[:, j * n_features : (j + 1) * n_features] = X[t0:t1, :]
        else:
            # same: pad with zeros where out-of-bounds
            src_start = max(0, -lag)
            src_stop = min(n_times, n_times - lag) if lag < 0 else min(n_times, n_times - lag)
            dst_start = max(0, lag)
            dst_stop = dst_start + (src_stop - src_start)
            if src_stop > src_start:
                X_lag[dst_start:dst_stop, j * n_features : (j + 1) * n_features] = X[src_start:src_stop, :]
    return X_lag, keep


class RidgeLagBackend:
    """Ridge regression backend with explicit lagged design matrix.

    This backend is backend-agnostic and does not depend on MNE. It is intended as
    the stable reference implementation.

    Notes
    -----
    - Intercept is always fitted (targets are centered implicitly).
    - Coef shape is (n_lags, n_features, n_outputs).
    """

    name = "ridge"

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        lags_samp: np.ndarray,
        alpha: float,
        sfreq: float,
        mode: str = "valid",
        **params: Any,
    ) -> BackendFitResult:
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have same n_times.")

        X_lag, keep = _build_lagged_matrix(X, lags_samp, mode=mode)
        Y_eff = Y[keep, :] if mode == "valid" else Y

        # Fit intercept by centering
        X_mean = X_lag.mean(axis=0, keepdims=True)
        Y_mean = Y_eff.mean(axis=0, keepdims=True)
        Xc = X_lag - X_mean
        Yc = Y_eff - Y_mean

        n_features_lag = Xc.shape[1]
        A = Xc.T @ Xc
        A.flat[:: n_features_lag + 1] += float(alpha)  # ridge on diagonal
        B = Xc.T @ Yc

        W = np.linalg.solve(A, B)  # (n_features_lag, n_outputs)

        intercept = (Y_mean - X_mean @ W).ravel()  # (n_outputs,)

        n_lags = int(len(lags_samp))
        n_features = int(X.shape[1])
        coef = W.reshape(n_lags, n_features, Y.shape[1])

        return BackendFitResult(coef=coef, intercept=intercept, extra={"mode": mode})

    def predict(
        self,
        fit: BackendFitResult,
        X: np.ndarray,
        *,
        lags_samp: np.ndarray,
        sfreq: float,
        mode: str = "valid",
        **params: Any,
    ) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array.")
        coef = np.asarray(fit.coef)
        intercept = np.asarray(fit.intercept)

        X_lag, keep = _build_lagged_matrix(X, lags_samp, mode=mode)
        W = coef.reshape(coef.shape[0] * coef.shape[1], coef.shape[2])
        Y_hat_eff = X_lag @ W + intercept[None, :]

        if mode == "valid":
            Y_hat = np.zeros((X.shape[0], Y_hat_eff.shape[1]), dtype=Y_hat_eff.dtype)
            Y_hat[keep, :] = Y_hat_eff
            return Y_hat
        return Y_hat_eff
