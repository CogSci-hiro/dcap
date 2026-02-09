# =============================================================================
# TRF analysis: MNE ReceptiveField backend (optional)
# =============================================================================

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BackendFitResult


class MneRfBackend:
    """Backend wrapper around ``mne.decoding.ReceptiveField``.

    Notes
    -----
    This backend is optional and requires MNE. It is mainly useful for parity
    with existing workflows. The refactor's stable reference backend is
    ``RidgeLagBackend``.
    """

    name = "mne-rf"

    def __init__(self) -> None:
        try:
            import mne  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError("MNE is required for the 'mne-rf' backend.") from e

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        lags_samp: np.ndarray,
        alpha: float,
        sfreq: float,
        tmin_s: float | None = None,
        tmax_s: float | None = None,
        **params: Any,
    ) -> BackendFitResult:
        import mne

        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have same n_times.")

        # Convert lags_samp to tmin/tmax if not provided
        if tmin_s is None or tmax_s is None:
            tmin_s = float(lags_samp.min()) / float(sfreq)
            tmax_s = float(lags_samp.max()) / float(sfreq)

        estimator = mne.decoding.ReceptiveField(
            tmin=float(tmin_s),
            tmax=float(tmax_s),
            sfreq=float(sfreq),
            estimator=float(alpha),
            **params,
        )
        estimator.fit(X, Y)

        coef = np.asarray(estimator.coef_)  # (n_outputs, n_features, n_lags) in MNE
        intercept = np.asarray(estimator.intercept_).ravel()

        # Convert to canonical (n_lags, n_features, n_outputs)
        coef = np.transpose(coef, (2, 1, 0))

        return BackendFitResult(coef=coef, intercept=intercept, extra={"estimator": estimator})

    def predict(
        self,
        fit: BackendFitResult,
        X: np.ndarray,
        *,
        lags_samp: np.ndarray,
        sfreq: float,
        **params: Any,
    ) -> np.ndarray:
        estimator = fit.extra.get("estimator", None)
        if estimator is None:
            raise ValueError("Missing estimator in backend fit result; cannot predict.")
        X = np.asarray(X)
        return np.asarray(estimator.predict(X))
