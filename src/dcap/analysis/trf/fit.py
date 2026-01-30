# =============================================================================
#                     TRF analysis: fitting dispatcher
# =============================================================================

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dcap.analysis.trf.backends.base import BackendFitResult
from dcap.analysis.trf.backends.mne_rf import MneRfBackendConfig
from dcap.analysis.trf.backends.registry import get_backend
from dcap.analysis.trf.design_matrix import LagConfig, make_lag_samples


@dataclass(frozen=True, slots=True)
class TrfFitConfig:
    """
    Backend-agnostic TRF fit configuration.

    Parameters
    ----------
    backend : str
        Backend name, e.g. "mne-rf".
    backend_params : mapping, optional
        Backend-specific parameters.

    Usage example
    -------------
        cfg = TrfFitConfig(backend="mne-rf", backend_params={"alpha": 1.0})
    """

    backend: str = "mne-rf"
    backend_params: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrfFitResult:
    """
    High-level TRF fit result.

    Attributes
    ----------
    backend : str
        Backend name used.
    lags_samp : ndarray
        Lags used (samples).
    coef_ : ndarray
        Backend coefficients.
    intercept_ : ndarray
        Backend intercept.
    extra : mapping
        Backend-specific state required for prediction.

    Usage example
    -------------
        result = fit_trf(X, Y, sfreq=100.0, lag_config=lags, fit_config=cfg)
    """

    backend: str
    lags_samp: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    extra: Mapping[str, Any]


def fit_trf(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sfreq: float,
    lag_config: LagConfig,
    fit_config: TrfFitConfig,
) -> TrfFitResult:
    """
    Fit TRF using a selected backend.

    Parameters
    ----------
    X : ndarray
        Stimulus/features.
        Continuous: (n_times, n_features)
        Epoched: (n_times, n_epochs, n_features)
    Y : ndarray
        Responses.
        Continuous: (n_times, n_outputs)
        Epoched: (n_times, n_epochs, n_outputs)
    sfreq : float
        Sampling frequency (Hz).
    lag_config : LagConfig
        Lag configuration (ms) used to derive sample delays.
    fit_config : TrfFitConfig
        Backend selection + parameters.

    Returns
    -------
    result : TrfFitResult

    Usage example
    -------------
        result = fit_trf(
            X, Y,
            sfreq=100.0,
            lag_config=LagConfig(tmin_ms=-100, tmax_ms=400, step_ms=10),
            fit_config=TrfFitConfig(backend="mne-rf", backend_params={"alpha": 1.0}),
        )
    """

    if X.ndim not in (2, 3):
        raise ValueError("X must be 2D (time, feature) or 3D (time, epoch, feature).")
    if Y.ndim not in (2, 3):
        raise ValueError("Y must be 2D (time, output) or 3D (time, epoch, output).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same n_times.")
    if X.ndim == 3 and Y.ndim == 3 and X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same n_epochs for epoched inputs.")

    lags_samp = make_lag_samples(sfreq=sfreq, config=lag_config)

    backend = get_backend(fit_config.backend)
    backend_cfg = _make_backend_config(backend.name, fit_config.backend_params)

    fit_out = backend.fit(X, Y, sfreq=sfreq, lags_samp=lags_samp, config=backend_cfg)

    return TrfFitResult(
        backend=backend.name,
        lags_samp=lags_samp,
        coef_=fit_out.coef_,
        intercept_=fit_out.intercept_,
        extra=dict(fit_out.extra),
    )


def predict_trf(X: np.ndarray, fit_result: TrfFitResult) -> np.ndarray:
    """
    Predict responses using a fitted TRF model.

    Parameters
    ----------
    X : ndarray
        Stimulus/features, same shape convention as in fit.
    fit_result : TrfFitResult
        Result returned by `fit_trf`.

    Returns
    -------
    Y_hat : ndarray
        Predicted responses.

    Usage example
    -------------
        Y_hat = predict_trf(X, fit_result)
    """

    backend = get_backend(fit_result.backend)
    backend_fit_result = BackendFitResult(
        coef_=fit_result.coef_,
        intercept_=fit_result.intercept_,
        extra=fit_result.extra,
    )
    return backend.predict(X, backend_fit_result)


def fit_trf_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sfreq: float,
    lag_config: LagConfig,
    alpha: float = 1.0,
) -> TrfFitResult:
    """
    Convenience wrapper for ridge TRF using the default backend.

    Usage example
    -------------
        result = fit_trf_ridge(X, Y, sfreq=100.0, lag_config=lags, alpha=1.0)
    """

    return fit_trf(
        X,
        Y,
        sfreq=sfreq,
        lag_config=lag_config,
        fit_config=TrfFitConfig(backend="mne-rf", backend_params={"alpha": float(alpha)}),
    )


def _make_backend_config(backend_name: str, params: Mapping[str, Any] | None) -> Any:
    params_dict = {} if params is None else dict(params)

    if backend_name == "mne-rf":
        return MneRfBackendConfig(
            alpha=float(params_dict.get("alpha", 1.0)),
            estimator_kwargs=params_dict.get("estimator_kwargs", {}) or {},
        )

    return params_dict
