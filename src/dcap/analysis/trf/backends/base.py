# =============================================================================
# TRF analysis: backend interface
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class BackendFitResult:
    """Opaque backend fit result."""

    coef: np.ndarray
    intercept: np.ndarray
    extra: Dict[str, Any]


@runtime_checkable
class TrfBackend(Protocol):
    """Backend protocol.

    Backends fit TRFs given arrays and lag samples. CV, segmentation, and scoring
    live above the backend.
    """

    name: str

    def fit(
        self,
        X: np.ndarray,  # (n_times, n_features)
        Y: np.ndarray,  # (n_times, n_outputs)
        *,
        lags_samp: np.ndarray,
        alpha: float,
        sfreq: float,
        **params: Any,
    ) -> BackendFitResult: ...

    def predict(
        self,
        fit: BackendFitResult,
        X: np.ndarray,
        *,
        lags_samp: np.ndarray,
        sfreq: float,
        **params: Any,
    ) -> np.ndarray: ...
