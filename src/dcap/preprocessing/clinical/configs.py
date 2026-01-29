from dataclasses import dataclass
from typing import Literal

AnalysisView = Literal["original", "car", "bipolar", "laplacian", "wm_ref"]

@dataclass(frozen=True)
class ClinicalAnalysisConfig:
    analysis_view: AnalysisView = "original"
