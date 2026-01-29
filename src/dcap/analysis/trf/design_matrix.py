# =============================================================================
#                    Analysis: TRF (lagged design matrix)
# =============================================================================
#
# Utilities to construct time-lagged design matrices for TRF models.
#
# =============================================================================

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from dcap.analysis.trf.types import LagConfig


FloatArray = NDArray[np.floating]


def build_lagged_design_matrix(
    x: FloatArray,
    cfg: LagConfig,
) -> Tuple[FloatArray, FloatArray]:
    """Build a lagged (time-embedded) design matrix.

    Parameters
    ----------
    x
        Input feature time series, shape (n_samples,) or (n_samples, n_features).
        For a speech envelope baseline, `x` is typically 1D.
    cfg
        Lag configuration.

    Returns
    -------
    X_lagged
        Lagged design matrix, typically shape (n_samples, n_features * n_lags),
        but the exact convention is left to the implementation.
    lags_s
        Lag vector in seconds, shape (n_lags,).

    Notes
    -----
    Typical TRF conventions include:
    - positive lags: stimulus precedes response
    - negative lags: response precedes stimulus (often used for acausal filters)

    Usage example
    -------------
        import numpy as np
        from dcap.analysis.trf import LagConfig, build_lagged_design_matrix

        x = np.random.randn(2000).astype(float)  # 10 s at 200 Hz
        X, lags_s = build_lagged_design_matrix(x, LagConfig(-0.2, 0.6, 200.0))
    """
    _validate_x(x=x)
    _validate_lag_config(cfg=cfg)

    raise NotImplementedError(
        "TODO: implement lagged design matrix construction (time embedding). "
        "Choose and document a convention for shapes and edge handling."
    )


def _validate_x(x: FloatArray) -> None:
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1D or 2D, got shape={x.shape!r}.")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"x must be floating dtype, got dtype={x.dtype!r}.")


def _validate_lag_config(cfg: LagConfig) -> None:
    if cfg.tmax_s < cfg.tmin_s:
        raise ValueError(f"tmax_s must be >= tmin_s, got {cfg.tmax_s} < {cfg.tmin_s}.")
    if cfg.sfreq <= 0 or not np.isfinite(cfg.sfreq):
        raise ValueError(f"sfreq must be a finite positive number, got {cfg.sfreq!r}.")
