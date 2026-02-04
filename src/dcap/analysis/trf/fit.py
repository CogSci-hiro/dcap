# =============================================================================
#                     TRF analysis: fitting dispatcher
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from dcap.analysis.trf.backends.base import BackendFitResult
from dcap.analysis.trf.backends.mne_rf import MneRfBackendConfig
from dcap.analysis.trf.backends.registry import get_backend
from dcap.analysis.trf.design_matrix import LagConfig, make_lag_samples


AlphaMode = Literal["shared", "per_channel"]
ScoreAgg = Literal["mean", "median"]


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
    alpha : float | ndarray
        Ridge alpha used. Scalar for shared-alpha fits; vector of shape (n_outputs,)
        for per-channel alpha fits.
    alpha_mode : {"shared", "per_channel"}
        Indicates whether the fit used one alpha for all outputs or one per output.
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
    alpha: Union[float, np.ndarray]
    alpha_mode: AlphaMode
    extra: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrfRidgeCvResult:
    """
    Ridge CV result container.

    Attributes
    ----------
    alpha_mode : {"shared", "per_channel"}
        Whether alpha selection was shared across channels or optimized per channel.
    best_alpha : float
        The alpha that maximized the aggregated CV score across channels (shared view).
        This is always provided for backward compatibility, even when alpha_mode="per_channel".
    best_alpha_by_output : ndarray | None
        Shape (n_outputs,). Only populated when alpha_mode="per_channel".
    mean_score_by_alpha : ndarray
        Shape (n_alphas,). Mean score across folds and channels for each alpha (aggregated).
    fold_score_by_alpha : ndarray
        Shape (n_alphas, n_folds). Per-fold aggregated score (averaged across channels).
    score_by_alpha_by_output : ndarray | None
        Shape (n_alphas, n_outputs). Mean score across folds for each (alpha, output).
        Only populated when alpha_mode="per_channel".
    alphas : ndarray
        Candidate alpha values, shape (n_alphas,).

    Usage example
    -------------
        cv = fit_trf_ridge_cv(
            X, Y, sfreq=100.0, lag_config=lags, alphas=[0.1, 1.0, 10.0],
            alpha_mode="per_channel",
        )
        alpha_vec = cv.best_alpha_by_output
    """

    alpha_mode: AlphaMode
    best_alpha: float
    best_alpha_by_output: Optional[np.ndarray]
    mean_score_by_alpha: np.ndarray
    fold_score_by_alpha: np.ndarray
    score_by_alpha_by_output: Optional[np.ndarray]
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

    # We treat this path as "shared alpha" because the backend config carries a scalar alpha.
    alpha_used = float(getattr(backend_cfg, "alpha", 1.0))

    return TrfFitResult(
        backend=backend.name,
        lags_samp=lags_samp,
        coef_=fit_out.coef_,
        intercept_=fit_out.intercept_,
        alpha=alpha_used,
        alpha_mode="shared",
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
        Result returned by `fit_trf` or `fit_trf_ridge`.

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


# =============================================================================
#                            Ridge convenience wrappers
# =============================================================================

def fit_trf_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sfreq: float,
    lag_config: LagConfig,
    alpha: Union[float, np.ndarray] = 1.0,
) -> TrfFitResult:
    """
    Convenience wrapper for ridge TRF using the default backend ("mne-rf").

    Parameters
    ----------
    X, Y : ndarray
        Same conventions as `fit_trf`.
    sfreq : float
        Sampling frequency (Hz).
    lag_config : LagConfig
        Lag configuration (ms).
    alpha : float | ndarray
        Ridge alpha. If float, a single shared alpha is used across outputs.
        If ndarray, must have shape (n_outputs,) and one alpha will be used per output.

    Returns
    -------
    TrfFitResult

    Notes
    -----
    Per-channel alpha fitting is implemented by fitting one single-output model per
    channel and then merging coefficients + intercepts into a single result object.

    Usage example
    -------------
        # shared alpha
        fit = fit_trf_ridge(X, Y, sfreq=100.0, lag_config=lags, alpha=1.0)

        # per-channel alpha
        alpha_vec = np.full((Y.shape[-1],), 10.0)
        fit = fit_trf_ridge(X, Y, sfreq=100.0, lag_config=lags, alpha=alpha_vec)
    """
    alpha_arr = np.asarray(alpha, dtype=float)

    if alpha_arr.ndim == 0:
        # Shared alpha: keep the simple fast path
        return fit_trf(
            X,
            Y,
            sfreq=sfreq,
            lag_config=lag_config,
            fit_config=TrfFitConfig(backend="mne-rf", backend_params={"alpha": float(alpha_arr)}),
        )

    if alpha_arr.ndim != 1:
        raise ValueError("alpha must be a scalar or a 1D array of shape (n_outputs,).")

    n_outputs = int(Y.shape[-1])
    if alpha_arr.shape[0] != n_outputs:
        raise ValueError(f"alpha vector must have shape (n_outputs,) == ({n_outputs},).")
    if np.any(~np.isfinite(alpha_arr)) or np.any(alpha_arr <= 0):
        raise ValueError("All per-channel alphas must be finite and > 0.")

    # Fit one model per output channel and merge
    merged = _fit_trf_ridge_per_channel(
        X=X,
        Y=Y,
        sfreq=sfreq,
        lag_config=lag_config,
        alpha_by_output=alpha_arr.astype(float),
    )
    return merged


def _fit_trf_ridge_per_channel(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    sfreq: float,
    lag_config: LagConfig,
    alpha_by_output: np.ndarray,
) -> TrfFitResult:
    """
    Internal helper: fit one TRF per output channel and merge results.

    Parameters
    ----------
    alpha_by_output : ndarray
        Shape (n_outputs,).

    Returns
    -------
    TrfFitResult
        coef_ / intercept_ are merged into a single tensor following the backend
        coefficient shape for multi-output. The backend estimator stored in `extra`
        is not used for prediction in this mode; prediction uses an explicit
        per-channel estimator list stored in extra["estimators_by_output"].
    """
    lags_samp = make_lag_samples(sfreq=sfreq, config=lag_config)
    backend = get_backend("mne-rf")

    # We store one estimator per output channel, because MNE requires it for predict().
    estimators_by_output: list[Any] = []

    coef_list: list[np.ndarray] = []
    intercept_list: list[np.ndarray] = []

    for out_i, a in enumerate(alpha_by_output):
        cfg = _make_backend_config("mne-rf", {"alpha": float(a)})

        # Slice single output, preserving dimensionality expected by backend
        if Y.ndim == 2:
            Y_one = Y[:, out_i : out_i + 1]
        else:
            Y_one = Y[:, :, out_i : out_i + 1]

        fit_out = backend.fit(X, Y_one, sfreq=sfreq, lags_samp=lags_samp, config=cfg)

        coef_list.append(np.asarray(fit_out.coef_, dtype=float))
        intercept_list.append(np.asarray(fit_out.intercept_, dtype=float))

        est = fit_out.extra.get("estimator", None)
        if est is None:
            raise ValueError("Backend did not return extra['estimator']; cannot support per-channel prediction.")
        estimators_by_output.append(est)

    # Merge coefficients:
    # MNE typically gives coef_ as (n_features, n_outputs, n_delays) for multi-output fits.
    # For single-output fits it is commonly (n_features, 1, n_delays). We concatenate on axis=1.
    coef0 = coef_list[0]
    if coef0.ndim != 3:
        raise ValueError(f"Expected backend coef_ to be 3D, got shape {coef0.shape}.")

    coef_merged = np.concatenate(coef_list, axis=1)

    # Merge intercepts: each is (1,) or (1,). We flatten to (n_outputs,)
    intercept_merged = np.concatenate([i.reshape(-1) for i in intercept_list], axis=0)

    return TrfFitResult(
        backend="mne-rf",
        lags_samp=lags_samp,
        coef_=coef_merged,
        intercept_=intercept_merged,
        alpha=alpha_by_output.astype(float),
        alpha_mode="per_channel",
        extra={
            # Predict path will use these instead of a single estimator
            "estimators_by_output": estimators_by_output,
            "tmin": float((lags_samp / float(sfreq)).min()),
            "tmax": float((lags_samp / float(sfreq)).max()),
            "sfreq": float(sfreq),
        },
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
    alpha_mode: AlphaMode = "shared",
    score_agg: ScoreAgg = "mean",
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
    alpha_mode : {"shared", "per_channel"}
        - "shared": choose a single alpha using aggregated score across channels.
        - "per_channel": choose alpha independently per output channel.
    score_agg : {"mean", "median"}
        Aggregation across channels for the shared-alpha view.

    Returns
    -------
    cv : TrfRidgeCvResult
        Contains best alpha + full CV curve. In per-channel mode, also contains
        `best_alpha_by_output` and `score_by_alpha_by_output`.

    Notes
    -----
    Scoring uses Pearson correlation per channel on held-out epoch, then aggregates.

    Usage example
    -------------
        cv = fit_trf_ridge_cv(
            X, Y, sfreq=100.0, lag_config=lags, alphas=[0.1, 1.0, 10.0],
            alpha_mode="per_channel",
        )
        alpha_vec = cv.best_alpha_by_output
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

    if alpha_mode not in ("shared", "per_channel"):
        raise ValueError("alpha_mode must be 'shared' or 'per_channel'.")
    if score_agg not in ("mean", "median"):
        raise ValueError("score_agg must be 'mean' or 'median'.")

    n_alphas = int(alphas_arr.size)

    # Aggregated fold scores (backward-compatible): (n_alphas, n_folds)
    fold_scores_agg = np.zeros((n_alphas, n_epochs), dtype=float)

    # Per-output fold scores: (n_alphas, n_folds, n_outputs)
    fold_scores_by_output = np.zeros((n_alphas, n_epochs, n_outputs), dtype=float)

    # Leave-one-epoch-out
    all_idx = np.arange(n_epochs)

    for fold, test_idx in enumerate(all_idx):
        train_idx = all_idx[all_idx != test_idx]

        X_train = X[:, train_idx, :]
        Y_train = Y[:, train_idx, :]
        X_test = X[:, test_idx : test_idx + 1, :]
        Y_test = Y[:, test_idx : test_idx + 1, :]

        for a_i, a in enumerate(alphas_arr):
            fit = fit_trf_ridge(X_train, Y_train, sfreq=sfreq, lag_config=lag_config, alpha=float(a))
            Y_hat = predict_trf(X_test, fit)

            corr = _corr_per_channel(Y_test[:, 0, :], Y_hat[:, 0, :])  # (n_outputs,)
            fold_scores_by_output[a_i, fold, :] = corr

            if score_agg == "mean":
                fold_scores_agg[a_i, fold] = float(np.mean(corr)) if n_outputs > 0 else np.nan
            else:
                fold_scores_agg[a_i, fold] = float(np.median(corr)) if n_outputs > 0 else np.nan

    mean_score_by_alpha = np.mean(fold_scores_agg, axis=1)

    # Shared-alpha best (always computed for compatibility / summary)
    best_i_shared = int(np.nanargmax(mean_score_by_alpha))
    best_alpha_shared = float(alphas_arr[best_i_shared])

    if alpha_mode == "shared":
        return TrfRidgeCvResult(
            alpha_mode="shared",
            best_alpha=best_alpha_shared,
            best_alpha_by_output=None,
            mean_score_by_alpha=mean_score_by_alpha.astype(float),
            fold_score_by_alpha=fold_scores_agg.astype(float),
            score_by_alpha_by_output=None,
            alphas=alphas_arr,
        )

    # Per-channel alpha selection
    score_by_alpha_by_output = np.mean(fold_scores_by_output, axis=1)  # (n_alphas, n_outputs)
    best_idx_by_output = np.nanargmax(score_by_alpha_by_output, axis=0)  # (n_outputs,)
    best_alpha_by_output = alphas_arr[best_idx_by_output].astype(float)

    return TrfRidgeCvResult(
        alpha_mode="per_channel",
        best_alpha=best_alpha_shared,
        best_alpha_by_output=best_alpha_by_output,
        mean_score_by_alpha=mean_score_by_alpha.astype(float),
        fold_score_by_alpha=fold_scores_agg.astype(float),
        score_by_alpha_by_output=score_by_alpha_by_output.astype(float),
        alphas=alphas_arr,
    )


# =============================================================================
#                         Prediction helpers (per-channel mode)
# =============================================================================

def _predict_trf_per_channel_estimators(X: np.ndarray, estimators_by_output: Sequence[Any]) -> np.ndarray:
    """
    Predict with a list of single-output estimators (one per output).

    This is only needed if a backend requires a fitted estimator object for predict().
    MNE's ReceptiveField is one such backend.

    Parameters
    ----------
    X : ndarray
        Input regressors.
    estimators_by_output : sequence
        One fitted estimator per output channel.

    Returns
    -------
    Y_hat : ndarray
        Predicted output with shape matching multi-output prediction.
    """
    preds: list[np.ndarray] = []
    for est in estimators_by_output:
        y_hat = np.asarray(est.predict(X), dtype=float)
        preds.append(y_hat)

    # Each y_hat is (n_times, 1) or (n_times, n_epochs, 1)
    Y_hat = np.concatenate(preds, axis=-1)
    return Y_hat


# Monkeypatch predict_trf to support per-channel merged results without changing public signature.
# We keep it explicit and contained here.
def predict_trf(X: np.ndarray, fit_result: TrfFitResult) -> np.ndarray:  # type: ignore[override]
    """
    Predict responses using a fitted TRF model (shared or per-channel alpha).

    If fit_result.alpha_mode == "per_channel", prediction uses the stored
    per-output estimators in extra["estimators_by_output"].
    """
    if fit_result.alpha_mode == "per_channel":
        estimators = fit_result.extra.get("estimators_by_output", None)
        if estimators is None:
            raise ValueError("Per-channel TRF fit is missing extra['estimators_by_output'] for prediction.")
        return _predict_trf_per_channel_estimators(X, estimators)

    backend = get_backend(fit_result.backend)
    backend_fit_result = BackendFitResult(
        coef_=fit_result.coef_,
        intercept_=fit_result.intercept_,
        extra=fit_result.extra,
    )
    return backend.predict(X, backend_fit_result)


# =============================================================================
#                               Utilities
# =============================================================================

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
