# =============================================================================
#                           Analysis: TRF (types)
# =============================================================================
#
# Typed containers for Temporal Response Function (TRF) analysis.
#
# This subpackage focuses on *minimal* TRF analysis (e.g., speech envelope ->
# neural response) with an emphasis on stable APIs and testability.
#
# Notes
# -----
# - Keep this module dependency-light.
# - No plotting or report generation here.
#
# =============================================================================

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class EnvelopeConfig:
    """Configuration for speech-envelope extraction.

    Notes
    -----
    This config is intentionally generic. Concrete audio decoding / file I/O
    should live in higher-level adapters.

    Usage example
    -------------
        cfg = EnvelopeConfig(
            target_sfreq=200.0,
            method="hilbert",
            lowpass_hz=8.0,
        )
    """

    target_sfreq: float
    method: str = "hilbert"
    lowpass_hz: Optional[float] = 8.0
    rectify: bool = True


@dataclass(frozen=True, slots=True)
class LagConfig:
    """Configuration for time-lagged design matrix construction.

    Usage example
    -------------
        lag = LagConfig(tmin_s=-0.200, tmax_s=0.600, sfreq=200.0)
    """

    tmin_s: float
    tmax_s: float
    sfreq: float


@dataclass(frozen=True, slots=True)
class TrfFitConfig:
    """Configuration for TRF model fitting.

    Notes
    -----
    This is a minimal baseline (ridge regression, optional cross-validation).

    Usage example
    -------------
        cfg = TrfFitConfig(
            alpha=100.0,
            fit_intercept=True,
            standardize_X=True,
            standardize_y=False,
        )
    """

    alpha: float
    fit_intercept: bool = True
    standardize_X: bool = True
    standardize_y: bool = False
    cv_folds: Optional[int] = None
    random_state: Optional[int] = 0


@dataclass(frozen=True, slots=True)
class TrfResult:
    """Container for a fitted TRF model and its evaluation.

    Parameters
    ----------
    weights
        TRF weights with shape (n_features, n_lags, n_outputs) or
        (n_features * n_lags, n_outputs) depending on representation.
        The skeleton does not enforce a single convention; choose one and
        document it when implementing.
    lags_s
        Vector of lag times in seconds (shape: (n_lags,)).
    alpha
        Regularization strength used for ridge regression.
    metrics
        Optional metrics (e.g., correlation, R^2) by output channel.
    metadata
        Free-form metadata for provenance (e.g., preprocessing hash). Avoid
        sensitive identifiers in shareable artifacts.

    Usage example
    -------------
        result = TrfResult(
            weights=np.zeros((1, 10, 128)),
            lags_s=np.linspace(-0.2, 0.6, 10),
            alpha=100.0,
            metrics={"corr": np.zeros(128)},
            metadata={"sfreq": 200.0},
        )
    """

    weights: FloatArray
    lags_s: FloatArray
    alpha: float
    metrics: Optional[Mapping[str, FloatArray]] = None
    metadata: Optional[Mapping[str, Any]] = None
