from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import mne
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TRFConfig:
    model_name: str = "ridge"
    tmin_sec: float = -0.2
    tmax_sec: float = 0.8
    alpha: float = 1.0

@dataclass(frozen=True)
class TRFInput:
    signal_raw: mne.io.BaseRaw
    events_df: pd.DataFrame
    predictors: Optional[Any] = None

@dataclass(frozen=True)
class TRFResult:
    model_name: str
    coefficients: np.ndarray
    times_sec: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
