# =============================================================================
#                        Analysis: TRF (model fitting)
# =============================================================================
#
# Minimal TRF fitting API.
#
# The intended baseline model is ridge regression on a lagged stimulus matrix:
#   y(t) ~ sum_{lag} w(lag) * x(t - lag)
#
# =============================================================================

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dcap.analysis.trf.design_matrix import build_lagged_design_matrix
from dcap.analysis.trf.types import LagConfig, TrfFitConfig, TrfResult


FloatArray = NDArray[np.floating]


def fit_trf_ridge(
    x: FloatArray,
    y: FloatArray,
    lag_cfg: LagConfig,
    fit_cfg: TrfFitConfig,
) -> TrfResult:
    """Fit a ridge-regularized TRF model.

    Parameters
    ----------
    x
        Stimulus feature(s), shape (n_samples,) or (n_samples, n_features).
    y
        Neural data, shape (n_samples,) or (n_samples, n_outputs).
    lag_cfg
        Time-lag settings used to construct the design matrix.
    fit_cfg
        Model fitting configuration.

    Returns
    -------
    result
        Fitted TRF result container.

    Notes
    -----
    - This skeleton leaves scaling/standardization and CV unimplemented.
    - Decide and document your weight shape convention when implementing.

    Usage example
    -------------
        import numpy as np
        from dcap.analysis.trf import LagConfig, TrfFitConfig, fit_trf_ridge

        x = np.random.randn(2000).astype(float)
        y = np.random.randn(2000, 16).astype(float)
        result = fit_trf_ridge(
            x=x,
            y=y,
            lag_cfg=LagConfig(-0.2, 0.6, 200.0),
            fit_cfg=TrfFitConfig(alpha=100.0),
        )
    """
    _validate_xy(x=x, y=y)

    # Build lagged design matrix (implementation pending)
    X_lagged, lags_s = build_lagged_design_matrix(x=x, cfg=lag_cfg)

    raise NotImplementedError(
        "TODO: implement ridge regression fit and return TrfResult. "
        "Likely steps: standardize, add intercept, solve ridge, compute metrics."
    )


def predict_trf(
    x: FloatArray,
    result: TrfResult,
    lag_cfg: LagConfig,
) -> FloatArray:
    """Predict neural response using a fitted TRF model.

    Parameters
    ----------
    x
        Stimulus feature(s), shape (n_samples,) or (n_samples, n_features).
    result
        Fitted TRF model container.
    lag_cfg
        Lag settings matching those used for fitting.

    Returns
    -------
    y_hat
        Predicted neural response, shape (n_samples, n_outputs).

    Usage example
    -------------
        y_hat = predict_trf(x, result, lag_cfg)
    """
    _validate_x_for_predict(x=x)

    # Build lagged design matrix (implementation pending)
    X_lagged, _ = build_lagged_design_matrix(x=x, cfg=lag_cfg)

    raise NotImplementedError("TODO: implement prediction.")  # noqa: TRY003


def _validate_xy(x: FloatArray, y: FloatArray) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same n_samples, got {x.shape[0]} vs {y.shape[0]}.")
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1D or 2D, got shape={x.shape!r}.")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1D or 2D, got shape={y.shape!r}.")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"x must be floating dtype, got dtype={x.dtype!r}.")
    if not np.issubdtype(y.dtype, np.floating):
        raise TypeError(f"y must be floating dtype, got dtype={y.dtype!r}.")


def _validate_x_for_predict(x: FloatArray) -> None:
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1D or 2D, got shape={x.shape!r}.")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"x must be floating dtype, got dtype={x.dtype!r}.")
