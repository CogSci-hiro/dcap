from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SorciereRecordingUnit:
    subject_bids: str
    dcap_id: str
    session: Optional[str]
    run: str
    vhdr_path: Path


@dataclass(frozen=True)
class RawTriggerCandidate:
    description: str
    event_code: int
    onset_samples: np.ndarray


@dataclass(frozen=True)
class SorciereAlignmentResult:
    selected_description: str
    selected_event_code: int
    delay_s: float
    stimulus_start_s: float
    matched_hits: int
    reference_onsets_s: np.ndarray
    raw_onsets_s: np.ndarray
    annotation_origin_in_reference_s: float
    reference_duration_s: Optional[float]
    candidate_count: int
