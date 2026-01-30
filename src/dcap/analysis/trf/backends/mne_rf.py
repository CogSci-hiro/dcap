# =============================================================================
#                TRF backend: MNE ReceptiveField wrapper
# =============================================================================
#
# This module provides a thin, explicit wrapper around
# `mne.decoding.ReceptiveField` so it can be used as a backend
# within the DCAP TRF abstraction layer.
#
# Design goals
# ------------
# - Keep MNE as an *optional* dependency (imported lazily).
# - Normalize coefficient / intercept outputs to a backend-agnostic format.
# - Store the fitted estimator so prediction semantics remain identical to MNE.
# - Avoid leaking MNE objects outside the backend boundary.
#
# This backend intentionally performs *no* reshaping or validation beyond
# what MNE itself enforces; upstream code is responsible for providing
# correctly shaped arrays.
#

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dcap.analysis.trf.backends.base import BackendFitResult


@dataclass(frozen=True, slots=True)
class MneRfBackendConfig:
    """
    Backend-specific configuration for the MNE ReceptiveField estimator.

    This config mirrors only the *minimal* set of parameters that are
    backend-specific. All task-level or analysis-level parameters
    (lags, normalization, cross-validation, etc.) live upstream.

    Parameters
    ----------
    alpha : float
        Ridge regularization parameter passed to MNE's ReceptiveField.
        This controls the L2 penalty strength.
    estimator_kwargs : mapping | None
        Optional extra keyword arguments forwarded verbatim to the
        `mne.decoding.ReceptiveField` constructor.

        This allows advanced users to access MNE-specific features
        without expanding the DCAP public API.

    Usage example
    -------------
        cfg = MneRfBackendConfig(
            alpha=1.0,
            estimator_kwargs=dict(n_jobs=4),
        )
    """

    alpha: float = 1.0
    estimator_kwargs: Mapping[str, Any] | None = None


class MneReceptiveFieldBackend:
    """
    TRF backend based on `mne.decoding.ReceptiveField`.

    This backend implements the DCAP TRF backend interface by delegating
    model fitting and prediction entirely to MNE, while adapting inputs
    and outputs to DCAP's standardized `BackendFitResult`.

    Expected input shapes
    ---------------------
    Continuous data:
        X : (n_times, n_features)
        Y : (n_times, n_outputs)

    Epoched data:
        X : (n_times, n_epochs, n_features)
        Y : (n_times, n_epochs, n_outputs)

    Shape conventions follow MNE's ReceptiveField API exactly.
    No implicit reshaping or axis permutation is performed.

    Notes
    -----
    - The fitted MNE estimator is stored in `fit_result.extra["estimator"]`.
    - Prediction *requires* this estimator; coefficients alone are not
      sufficient because MNE internally manages lagged design matrices.
    - Coefficients returned here are exposed mainly for inspection,
      visualization, or downstream summaries.
    """

    # Stable backend identifier used in registries / dispatch
    name = "mne-rf"

    @staticmethod
    def fit(
        X: np.ndarray,
        Y: np.ndarray,
        *,
        sfreq: float,
        lags_samp: np.ndarray,
        config: MneRfBackendConfig,
    ) -> BackendFitResult:
        """
        Fit a temporal receptive field model using MNE.

        Parameters
        ----------
        X : ndarray
            Stimulus / regressor array following MNE shape conventions.
        Y : ndarray
            Neural response array following MNE shape conventions.
        sfreq : float
            Sampling frequency in Hz.
        lags_samp : ndarray
            Array of lag values in *samples*. These are converted internally
            to seconds for MNE.
        config : MneRfBackendConfig
            Backend-specific configuration.

        Returns
        -------
        BackendFitResult
            Container holding coefficients, intercept, and backend-specific
            objects required for prediction.
        """
        # Lazy import to keep MNE an optional dependency
        try:
            from mne.decoding import ReceptiveField  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "MNE backend requested but mne-python "
                "(and mne.decoding.ReceptiveField) is not available."
            ) from exc

        # Normalize estimator kwargs
        if config.estimator_kwargs is None:
            estimator_kwargs: Mapping[str, Any] = {}
        else:
            estimator_kwargs = config.estimator_kwargs

        # Convert lags from samples to seconds as required by MNE
        lags_sec = lags_samp.astype(float) / float(sfreq)
        tmin = float(lags_sec.min())
        tmax = float(lags_sec.max())

        # Construct the MNE ReceptiveField estimator
        # Note: `estimator` corresponds to the ridge parameter (alpha)
        rf = ReceptiveField(
            tmin=tmin,
            tmax=tmax,
            sfreq=float(sfreq),
            estimator=float(config.alpha),
            **estimator_kwargs,
        )

        # Fit the TRF model
        rf.fit(X, Y)

        # Extract learned parameters
        # Shapes are backend-specific but cast to ndarray for consistency
        coef = np.asarray(rf.coef_, dtype=float)

        # Intercept handling is defensive: some MNE versions may omit it
        intercept = np.asarray(
            getattr(rf, "intercept_", np.zeros(Y.shape[-1])),
            dtype=float,
        )

        # Store the estimator itself for later prediction
        return BackendFitResult(
            coef_=coef,
            intercept_=intercept,
            extra={
                "estimator": rf,
                "tmin": tmin,
                "tmax": tmax,
            },
        )

    @staticmethod
    def predict(X: np.ndarray, fit_result: BackendFitResult) -> np.ndarray:
        """
        Generate predictions using a fitted MNE ReceptiveField model.

        Parameters
        ----------
        X : ndarray
            Input regressor array with the same shape convention used at fit time.
        fit_result : BackendFitResult
            Result object returned by `fit`.

        Returns
        -------
        ndarray
            Predicted neural response with the same shape semantics as `Y`.

        Raises
        ------
        ValueError
            If the fitted estimator is missing from `fit_result.extra`.
        """
        estimator = fit_result.extra.get("estimator", None)
        if estimator is None:
            raise ValueError(
                "MNE backend requires extra['estimator'] for prediction."
            )

        Y_hat = estimator.predict(X)
        return np.asarray(Y_hat, dtype=float)
