# =============================================================================
#                  TRF backends: base interface (stable)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class BackendFitResult:
    """
    Backend-agnostic fit output.

    Attributes
    ----------
    coef_ : ndarray, shape (n_outputs, n_features)
        Model coefficients in the backend's feature space.
    intercept_ : ndarray, shape (n_outputs,)
        Intercept term.
    extra : Mapping[str, Any]
        Backend-specific extras (e.g., estimator object, CV details).
    """

    coef_: np.ndarray
    intercept_: np.ndarray
    extra: Mapping[str, Any]


class TrfBackend(Protocol):
    """
    Protocol for TRF backends.

    A backend is responsible for:
    - fitting weights given X, Y
    - predicting given X
    - exposing coefficients consistently
    """

    name: str

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        sfreq: float,
        lags_samp: np.ndarray,
        config: Any,
    ) -> BackendFitResult:
        ...

    def predict(
        self,
        X: np.ndarray,
        fit_result: BackendFitResult,
    ) -> np.ndarray:
        ...
