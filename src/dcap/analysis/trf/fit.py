# =============================================================================
#                     TRF analysis: fitting dispatcher
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dcap.analysis.trf.backends.base import BackendFitResult
from dcap.analysis.trf.backends.mne_rf import MneRfBackendConfig
from dcap.analysis.trf.backends.registry import get_backend
from dcap.analysis.trf.design_matrix import LagConfig, make_lag_samples


# =============================================================================
#                              Public dataclasses
# =============================================================================

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


@dataclass(frozen=True, slots=True)
class TrfRidgeCvResult:
    """
    Ridge CV result container.

    Attributes
    ----------
    best_alpha : float
        Alpha that maximized the CV score.
    mean_score_by_alpha : ndarray
        Shape (n_alphas,). Mean score across folds (and channels) for each alpha.
    fold_score_by_alpha : ndarray
        Shape (n_alphas, n_folds). Per-fold mean score (averaged across channels).
    alphas : ndarray
        Candidate alpha values, shape (n_alphas,).

    Usage example
    -------------
        cv = fit_trf_ridge_cv(X, Y, sfreq=100.0, lag_config=lags, alphas=[0.1, 1.0, 10.0])
        best_alpha = cv.best_alpha
    """

    best_alpha: float
    mean_score_by_alpha: np.ndarray
    fold_score_by_alpha: np.ndarray
    alphas: np.ndarray


# =============================================================================
#                               Core fit/predict
# =============================================================================

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


# =============================================================================
#                          Ridge CV (alpha selection)
# =============================================================================

def fit_trf_ridge_cv(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sfreq: float,
    lag_config: LagConfig,
    alphas: Sequence[float],
) -> TrfRidgeCvResult:
    """
    Select ridge alpha by leave-one-epoch-out cross-validation.

    This assumes epoched arrays where the epoch dimension corresponds to runs.

    Parameters
    ----------
    X : ndarray
        Shape (n_times, n_epochs, n_features).
    Y : ndarray
        Shape (n_times, n_epochs, n_outputs).
    sfreq : float
        Sampling frequency (Hz).
    lag_config : LagConfig
        Lag configuration (ms).
    alphas : sequence of float
        Candidate ridge alphas.

    Returns
    -------
    cv : TrfRidgeCvResult
        Contains best alpha + full CV curve.

    Notes
    -----
    - Scoring uses mean Pearson correlation across channels on the held-out epoch.
    - If you want a different CV scheme (e.g., KFold with shuffled epochs),
      we can add it, but for 4 runs LOO is the clean default.

    Usage example
    -------------
        cv = fit_trf_ridge_cv(X, Y, sfreq=100.0, lag_config=lags, alphas=[0.1, 1.0, 10.0])
        best_alpha = cv.best_alpha
    """
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError("fit_trf_ridge_cv requires epoched inputs: X,Y must be 3D.")
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must match on (n_times, n_epochs).")

    n_times, n_epochs, _ = X.shape
    _, _, n_outputs = Y.shape
    if n_epochs < 2:
        raise ValueError("Need at least 2 epochs to cross-validate.")

    alphas_arr = np.asarray([float(a) for a in alphas], dtype=float)
    if np.any(~np.isfinite(alphas_arr)) or np.any(alphas_arr <= 0):
        raise ValueError("All alphas must be finite and > 0.")

    n_alphas = alphas_arr.size
    fold_scores = np.zeros((n_alphas, n_epochs), dtype=float)

    # Leave-one-epoch-out (ideal for 'epoch == run')
    all_idx = np.arange(n_epochs)

    for fold, test_idx in enumerate(all_idx):
        train_idx = all_idx[all_idx != test_idx]

        X_train = X[:, train_idx, :]
        Y_train = Y[:, train_idx, :]
        X_test = X[:, test_idx : test_idx + 1, :]
        Y_test = Y[:, test_idx : test_idx + 1, :]

        for a_i, alpha in enumerate(alphas_arr):
            fit = fit_trf_ridge(X_train, Y_train, sfreq=sfreq, lag_config=lag_config, alpha=float(alpha))
            Y_hat = predict_trf(X_test, fit)

            # Score on held-out epoch, per channel correlation across time
            corr = _corr_per_channel(Y_test[:, 0, :], Y_hat[:, 0, :])  # (n_outputs,)
            # Mean across channels for this fold
            fold_scores[a_i, fold] = float(np.mean(corr)) if n_outputs > 0 else np.nan

    mean_scores = np.mean(fold_scores, axis=1)
    best_i = int(np.nanargmax(mean_scores))
    best_alpha = float(alphas_arr[best_i])

    return TrfRidgeCvResult(
        best_alpha=best_alpha,
        mean_score_by_alpha=mean_scores.astype(float),
        fold_score_by_alpha=fold_scores.astype(float),
        alphas=alphas_arr,
    )


def _corr_per_channel(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Pearson correlation per channel over time.

    Parameters
    ----------
    y, y_hat : ndarray
        Shape (n_times, n_outputs).

    Returns
    -------
    corr : ndarray
        Shape (n_outputs,).
    """
    y0 = y - np.mean(y, axis=0, keepdims=True)
    y1 = y_hat - np.mean(y_hat, axis=0, keepdims=True)
    denom = (np.std(y0, axis=0) * np.std(y1, axis=0)) + 1e-12
    corr = np.mean(y0 * y1, axis=0) / denom
    return corr.astype(np.float64)


def _make_backend_config(backend_name: str, params: Mapping[str, Any] | None) -> Any:
    params_dict = {} if params is None else dict(params)

    if backend_name == "mne-rf":
        return MneRfBackendConfig(
            alpha=float(params_dict.get("alpha", 1.0)),
            estimator_kwargs=params_dict.get("estimator_kwargs", {}) or {},
        )

    return params_dict
