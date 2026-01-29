# =============================================================================
#                TRF backend: MNE receptive field wrapper
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dcap.analysis.trf.backends.base import BackendFitResult


@dataclass(frozen=True, slots=True)
class MneRfBackendConfig:
    """
    Backend-specific configuration for the MNE ReceptiveField backend.

    Parameters
    ----------
    alpha : float
        Ridge regularization parameter (passed to MNE estimator).
    estimator_kwargs : Mapping[str, Any]
        Extra kwargs forwarded to MNE estimator constructor.
    """

    alpha: float = 1.0
    estimator_kwargs: Mapping[str, Any] = None  # type: ignore[assignment]


class MneReceptiveFieldBackend:
    """
    MNE backend using mne.decoding.ReceptiveField (if available).

    Notes
    -----
    This backend expects:
    - X: shape (n_times, n_features)
    - Y: shape (n_times, n_outputs)
    """

    name = "mne-rf"

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        sfreq: float,
        lags_samp: np.ndarray,
        config: MneRfBackendConfig,
    ) -> BackendFitResult:
        try:
            import mne  # noqa: F401
            from mne.decoding import ReceptiveField  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "MNE backend requested but mne-python (and mne.decoding.ReceptiveField) "
                "is not available in this environment."
            ) from exc

        if config.estimator_kwargs is None:
            estimator_kwargs: Mapping[str, Any] = {}
        else:
            estimator_kwargs = config.estimator_kwargs

        # Convert lags in samples -> seconds.
        lags_sec = lags_samp.astype(float) / float(sfreq)

        # MNE expects tmin/tmax; it builds internal lagged X itself.
        # We set tmin/tmax from our lags array and feed the *unlagged* X.
        tmin = float(lags_sec.min())
        tmax = float(lags_sec.max())

        rf = ReceptiveField(
            tmin=tmin,
            tmax=tmax,
            sfreq=float(sfreq),
            estimator="ridge",
            alpha=float(config.alpha),
            **estimator_kwargs,
        )

        rf.fit(X, Y)

        # rf.coef_ is typically (n_outputs, n_features, n_delays)
        coef = np.asarray(rf.coef_, dtype=float)
        intercept = np.asarray(getattr(rf, "intercept_", np.zeros(Y.shape[1])), dtype=float)

        return BackendFitResult(
            coef_=coef,
            intercept_=intercept,
            extra={"estimator": rf, "tmin": tmin, "tmax": tmax},
        )

    def predict(
        self,
        X: np.ndarray,
        fit_result: BackendFitResult,
    ) -> np.ndarray:
        rf = fit_result.extra.get("estimator", None)
        if rf is None:
            raise ValueError("MNE backend predict requires stored estimator in fit_result.extra['estimator'].")

        Y_hat = rf.predict(X)
        return np.asarray(Y_hat, dtype=float)
