# =============================================================================
#                TRF backend: MNE ReceptiveField wrapper
# =============================================================================

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dcap.analysis.trf.backends.base import BackendFitResult


@dataclass(frozen=True, slots=True)
class MneRfBackendConfig:
    """
    Backend-specific configuration for MNE ReceptiveField.

    Parameters
    ----------
    alpha : float
        Ridge regularization parameter.
    estimator_kwargs : mapping
        Extra kwargs forwarded to MNE ReceptiveField constructor.

    Usage example
    -------------
        cfg = MneRfBackendConfig(alpha=1.0, estimator_kwargs={})
    """

    alpha: float = 1.0
    estimator_kwargs: Mapping[str, Any] | None = None


class MneReceptiveFieldBackend:
    """
    MNE backend using `mne.decoding.ReceptiveField`.

    Expected shapes
    ---------------
    - Continuous:
        X: (n_times, n_features)
        Y: (n_times, n_outputs)
    - Epoched:
        X: (n_times, n_epochs, n_features)
        Y: (n_times, n_epochs, n_outputs)

    Notes
    -----
    This backend relies on the fitted estimator stored in `extra["estimator"]` for prediction.
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
            from mne.decoding import ReceptiveField  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "MNE backend requested but mne-python (and mne.decoding.ReceptiveField) is not available."
            ) from exc

        if config.estimator_kwargs is None:
            estimator_kwargs: Mapping[str, Any] = {}
        else:
            estimator_kwargs = config.estimator_kwargs

        lags_sec = lags_samp.astype(float) / float(sfreq)
        tmin = float(lags_sec.min())
        tmax = float(lags_sec.max())

        rf = ReceptiveField(
            tmin=tmin,
            tmax=tmax,
            sfreq=float(sfreq),
            estimator=float(config.alpha),
            **estimator_kwargs,
        )

        rf.fit(X, Y)

        coef = np.asarray(rf.coef_, dtype=float)
        intercept = np.asarray(getattr(rf, "intercept_", np.zeros(Y.shape[-1])), dtype=float)

        return BackendFitResult(
            coef_=coef,
            intercept_=intercept,
            extra={"estimator": rf, "tmin": tmin, "tmax": tmax},
        )

    def predict(self, X: np.ndarray, fit_result: BackendFitResult) -> np.ndarray:
        estimator = fit_result.extra.get("estimator", None)
        if estimator is None:
            raise ValueError("MNE backend requires extra['estimator'] for prediction.")
        Y_hat = estimator.predict(X)
        return np.asarray(Y_hat, dtype=float)
