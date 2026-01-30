from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import mne

from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext
from dcap.seeg.clinical.qc import ClinicalQcSummary

@dataclass(frozen=True)
class ClinicalAnalysisNotes:
    items: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class ClinicalAnalysisBundle:
    subject_id: str
    session_id: Optional[str]
    run_id: Optional[str]
    raw_views: Mapping[str, mne.io.BaseRaw]
    preprocessing_artifacts: Sequence[BlockArtifact]
    preprocessing_context: PreprocContext
    envelopes: Optional[Mapping[str, mne.io.BaseRaw]] = None
    trf_result: Optional[Any] = None
    notes: ClinicalAnalysisNotes = field(default_factory=ClinicalAnalysisNotes)
    qc: Optional[ClinicalQcSummary] = None
