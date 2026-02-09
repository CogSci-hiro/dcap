# =============================================================================
# TRF analysis: scoring metrics
# =============================================================================

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

try:
    from scipy.stats import rankdata
except Exception:  # pragma: no cover
    rankdata = None


def _validate_real(y: np.ndarray, y_hat: np.ndarray, complex_handling: str) -> Tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(y) or np.iscomplexobj(y_hat):
        if complex_handling == "error":
            raise ValueError("Complex-valued targets/predictions require an explicit complex_handling policy.")
        if complex_handling == "real":
            return np.real(y), np.real(y_hat)
        if complex_handling == "imag":
            return np.imag(y), np.imag(y_hat)
        if complex_handling == "magnitude":
            return np.abs(y), np.abs(y_hat)
        raise ValueError(f"Unknown complex_handling={complex_handling!r}")
    return y, y_hat


def pearson_per_output(y: np.ndarray, y_hat: np.ndarray, *, complex_handling: str = "error") -> np.ndarray:
    """Pearson correlation per output channel.

    Parameters
    ----------
    y, y_hat
        Arrays of shape (n_times, n_outputs).

    Returns
    -------
    r : ndarray, shape (n_outputs,)
    """
    y, y_hat = _validate_real(y, y_hat, complex_handling)
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(f"y and y_hat must have same shape, got {y.shape} vs {y_hat.shape}")
    y0 = y - y.mean(axis=0, keepdims=True)
    y1 = y_hat - y_hat.mean(axis=0, keepdims=True)
    num = np.sum(y0 * y1, axis=0)
    den = np.sqrt(np.sum(y0 * y0, axis=0) * np.sum(y1 * y1, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    r = np.where(np.isfinite(r), r, 0.0)
    return r


def spearman_per_output(y: np.ndarray, y_hat: np.ndarray, *, complex_handling: str = "error") -> np.ndarray:
    """Spearman correlation per output channel (Pearson on ranks)."""
    if rankdata is None:
        raise ImportError("scipy is required for spearman scoring (scipy.stats.rankdata).")
    y, y_hat = _validate_real(y, y_hat, complex_handling)
    if y.shape != y_hat.shape:
        raise ValueError(f"y and y_hat must have same shape, got {y.shape} vs {y_hat.shape}")
    y_r = np.apply_along_axis(rankdata, 0, y)
    yh_r = np.apply_along_axis(rankdata, 0, y_hat)
    return pearson_per_output(y_r, yh_r, complex_handling="error")


def r2_per_output(y: np.ndarray, y_hat: np.ndarray, *, complex_handling: str = "error") -> np.ndarray:
    """Coefficient of determination per output."""
    y, y_hat = _validate_real(y, y_hat, complex_handling)
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(f"y and y_hat must have same shape, got {y.shape} vs {y_hat.shape}")
    ss_res = np.sum((y - y_hat) ** 2, axis=0)
    y_mean = y.mean(axis=0, keepdims=True)
    ss_tot = np.sum((y - y_mean) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)
    r2 = np.where(np.isfinite(r2), r2, 0.0)
    return r2


def aggregate_outputs(values: np.ndarray, *, agg: Literal["mean", "median"] = "mean") -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if agg == "mean":
        return float(values.mean())
    if agg == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown agg={agg!r}")


def metric_dispatch(name: str):
    if name == "pearson":
        return pearson_per_output
    if name == "spearman":
        return spearman_per_output
    if name == "r2":
        return r2_per_output
    raise ValueError(f"Unknown scoring {name!r}")
