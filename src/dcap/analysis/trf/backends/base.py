# =============================================================================
#                   TRF backends: base interface (stable)
# =============================================================================

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class BackendFitResult:
    """
    Backend-agnostic fit output.

    Attributes
    ----------
    coef_ : ndarray
        Backend coefficient array.
    intercept_ : ndarray
        Backend intercept array.
    extra : mapping
        Backend-specific state required for prediction (e.g., fitted estimator).

    Usage example
    -------------
        result = BackendFitResult(coef_=coef, intercept_=intercept, extra={"estimator": obj})
    """

    coef_: np.ndarray
    intercept_: np.ndarray
    extra: Mapping[str, Any]


class TrfBackend(Protocol):
    """Protocol for TRF backends."""

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

    def predict(self, X: np.ndarray, fit_result: BackendFitResult) -> np.ndarray:
        ...
