# =============================================================================
#                     TRF analysis: fitting dispatcher
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dcap.analysis.trf.backends.registry import get_backend
from dcap.analysis.trf.backends.mne_rf import MneRfBackendConfig
from dcap.analysis.trf.design_matrix import LagConfig, make_lag_samples
from dcap.analysis.trf.types import TrfFitConfig


@dataclass(frozen=True, slots=True)
class TrfFitResult:
    """
    High-level TRF fit result (backend-agnostic wrapper).

    Attributes
    ----------
    backend : str
        Backend name used to fit the model.
    lags_samp : ndarray
        Lags used (samples).
    coef_ : ndarray
        Coefficients as returned by the backend.
    intercept_ : ndarray
        Intercept as returned by the backend.
    extra : Mapping[str, Any]
        Backend extras (may include fitted estimator object).
    """

    backend: str
    lags_samp: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    extra: Mapping[str, Any]


def fit_trf(
    X_unlagged: np.ndarray,
    Y: np.ndarray,
    *,
    sfreq: float,
    lag_config: LagConfig,
    fit_config: TrfFitConfig,
) -> TrfFitResult:
    """
    Fit TRF using the selected backend.

    Notes
    -----
    - `X_unlagged` is stimulus/features sampled at `sfreq`, shape (n_times, n_features).
    - `Y` is neural data sampled at `sfreq`, shape (n_times, n_outputs).
    - The backend is responsible for applying lags (or ignoring them if it expects lagged X).
    """

    if X_unlagged.ndim == 1:
        X_unlagged = X_unlagged[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if X_unlagged.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same n_times.")

    lags_samp = make_lag_samples(sfreq=sfreq, config=lag_config)

    backend = get_backend(fit_config.backend)
    backend_cfg = _make_backend_config(backend_name=backend.name, params=fit_config.backend_params)

    fit_out = backend.fit(
        X_unlagged,
        Y,
        sfreq=sfreq,
        lags_samp=lags_samp,
        config=backend_cfg,
    )

    return TrfFitResult(
        backend=backend.name,
        lags_samp=lags_samp,
        coef_=fit_out.coef_,
        intercept_=fit_out.intercept_,
        extra=dict(fit_out.extra),
    )


def predict_trf(
    X_unlagged: np.ndarray,
    fit_result: TrfFitResult,
) -> np.ndarray:
    """
    Predict neural responses using a fitted TRF model.

    Notes
    -----
    The backend is responsible for interpreting the contents of
    `fit_result.extra` (e.g. stored estimator object).
    """
    if X_unlagged.ndim == 1:
        X_unlagged = X_unlagged[:, np.newaxis]

    backend = get_backend(fit_result.backend)

    from dcap.analysis.trf.backends.base import BackendFitResult

    backend_fit_result = BackendFitResult(
        coef_=fit_result.coef_,
        intercept_=fit_result.intercept_,
        extra=fit_result.extra,
    )

    return backend.predict(X_unlagged, backend_fit_result)


def _make_backend_config(backend_name: str, params: Mapping[str, Any] | None) -> Any:
    """
    Convert `backend_params` dict into a backend-specific config object.
    """
    params = {} if params is None else dict(params)

    if backend_name == "mne-rf":
        return MneRfBackendConfig(
            alpha=float(params.get("alpha", 1.0)),
            estimator_kwargs=params.get("estimator_kwargs", {}) or {},
        )

    # Default: pass raw dict through (future backends can accept Mapping)
    return params
