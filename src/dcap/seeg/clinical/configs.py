from dataclasses import dataclass
from typing import Literal, Optional

AnalysisView = Literal["original", "car", "bipolar", "laplacian", "wm_ref"]

@dataclass(frozen=True)
class ClinicalAnalysisConfig:
    analysis_view: AnalysisView = "original"

@dataclass(frozen=True, slots=True)
class ClinicalTrfConfig:
    enabled: bool = False

    # Backend selection
    backend: str = "mne-rf"

    # Lags (ms)
    tmin_ms: float = -100.0
    tmax_ms: float = 400.0
    step_ms: float = 10.0

    # Regularization
    alpha: float = 1.0

    # Envelope extraction (if used by task adapter)
    envelope_low_hz: float = 70.0
    envelope_high_hz: float = 150.0
    envelope_hilbert: bool = True

    # Optional: TRF scoring config (if you add later)
    cv_folds: Optional[int] = None
