# =============================================================================
# =============================================================================
#                           ########################
#                           #   TRF I/O CONTRACTS  #
#                           ########################
# =============================================================================
# =============================================================================
#
# Minimal stable dataclasses to decouple TRF computation from upstream
# preprocessing and downstream reporting.
#
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TRFConfig:
    """
    Configuration for TRF computation.

    Attributes
    ----------
    model_name
        Human-readable model label (e.g., "mTRF", "ridge").
    tmin_s
        Minimum lag in seconds.
    tmax_s
        Maximum lag in seconds.
    alpha
        Regularization strength (if applicable).

    Usage example
    -------------
        cfg = TRFConfig(model_name="ridge", tmin_sec=-0.2, tmax_sec=0.8, alpha=1.0)
    """

    backend: str = "mne-rf"
    model_name: str = "ridge"
    tmin_s: float = -0.2
    tmax_s: float = 0.8
    alpha: float = 1.0


@dataclass(frozen=True)
class TRFInput:
    """
    Inputs required to compute a TRF.

    Attributes
    ----------
    signal_raw
        Raw object containing the response signal (often an envelope time series).
    events_df
        Event table with at least `onset_sec` and a categorical label column.
    predictors
        Optional extra predictors (design matrix or other objects), analysis-defined.

    Usage example
    -------------
        trf_input = TRFInput(signal_raw=env_raw, events_df=events_df)
    """

    signal_raw: mne.io.BaseRaw
    events_df: pd.DataFrame
    predictors: Optional[Any] = None


@dataclass(frozen=True)
class TRFResult:
    """
    Outputs of a TRF computation.

    Attributes
    ----------
    model_name
        Name of the model used.
    coefficients
        TRF coefficients (shape analysis-specific).
    times_sec
        Time lags in seconds corresponding to coefficients.
    metrics
        Scalar summary metrics for reporting (e.g., r, mse).
    extra
        Optional extra information (e.g., per-channel metrics).

    Usage example
    -------------
        result = TRFResult(
            model_name="ridge",
            coefficients=coef,
            times_sec=times,
            metrics={"r_mean": 0.12},
        )
    """

    model_name: str
    coefficients: np.ndarray
    times_sec: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
