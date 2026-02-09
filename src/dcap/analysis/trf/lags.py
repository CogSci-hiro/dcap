# =============================================================================
# TRF analysis: lag utilities
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


LagMode = Literal["valid", "same"]


@dataclass(frozen=True, slots=True)
class LagSpec:
    """Lag specification in seconds (sample step implied by ``sfreq``).

    Parameters
    ----------
    tmin_s, tmax_s
        Inclusive lag bounds in seconds. Negative means predictors precede the
        response (typical TRF convention).
    mode
        "valid" returns only time points where all lags are available.
        "same" returns the same length as the input by zero-padding.
    include_0
        If False, drop the 0-lag even if it lies within bounds.

    Usage example
    -------------
        lags = LagSpec(tmin_s=-0.1, tmax_s=0.4, mode="valid")
        lags_samp, lags_s = compute_lags(lags, sfreq=100.0)
    """

    tmin_s: float
    tmax_s: float
    mode: LagMode = "valid"
    include_0: bool = True


def compute_lags(lag_spec: LagSpec, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lag grid in samples and seconds for a given sampling rate."""
    if sfreq <= 0:
        raise ValueError(f"sfreq must be > 0, got {sfreq}")

    tmin_s = float(lag_spec.tmin_s)
    tmax_s = float(lag_spec.tmax_s)
    if tmax_s < tmin_s:
        raise ValueError(f"tmax_s must be >= tmin_s, got {tmin_s}..{tmax_s}")

    # sample step is 1 / sfreq
    t_step_s = 1.0 / float(sfreq)

    # inclusive endpoints on a sample grid
    n = int(np.floor((tmax_s - tmin_s) / t_step_s + 0.5))  # nearest
    lags_s = tmin_s + np.arange(n + 1, dtype=float) * t_step_s

    lags_samp = np.rint(lags_s * sfreq).astype(int)

    # Ensure monotonic unique (rounding can duplicate)
    uniq = np.unique(lags_samp)
    lags_samp = uniq
    lags_s = lags_samp.astype(float) / float(sfreq)

    if not lag_spec.include_0:
        mask = lags_samp != 0
        lags_samp = lags_samp[mask]
        lags_s = lags_s[mask]

    if lags_samp.size == 0:
        raise ValueError("Lag grid is empty after applying include_0=False / rounding.")

    return lags_samp, lags_s


def max_abs_lag_seconds(lag_spec: LagSpec) -> float:
    """Maximum absolute lag in seconds."""
    return float(max(abs(lag_spec.tmin_s), abs(lag_spec.tmax_s)))
