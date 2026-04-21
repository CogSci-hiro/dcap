from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class NamingRecordingUnit:
    subject_bids: str
    dcap_id: str
    session: Optional[str]
    run: str
    vhdr_path: Path
    vmrk_path: Path
    log_dir: Path
    log_path: Path
    sequence_id: str


@dataclass(frozen=True)
class PresentationLogRow:
    subject: str
    trial: int
    event_type: str
    code: str
    time_ms: int


@dataclass(frozen=True)
class MarkerRow:
    index: int
    kind: str
    description: str
    sample: int
    size: int
    channel: int
    code: Optional[int]


@dataclass(frozen=True)
class ResponseAudioFile:
    stimulus_id: str
    path: Path
    timestamp: datetime


@dataclass(frozen=True)
class NamingTrial:
    trial_index: int
    stimulus_id: str
    stimulus_file: str
    sequence_trigger_code: int
    recorded_picture_code: Optional[int]
    stimulus_catalog_id: int
    recording_onset_ms: int
    picture_onset_ms: int
    isi_onset_ms: int
    recording_sample: int
    picture_sample: int
    isi_sample: int
    response_audio_path: Optional[Path]
    response_audio_timestamp: Optional[datetime]


@dataclass(frozen=True)
class NamingAuxEvent:
    event_type: str
    onset_ms: int
    sample: int
    marker_code: Optional[int]


@dataclass(frozen=True)
class NamingAlignmentSummary:
    sequence_id: str
    n_trials: int
    log_marker_offset_s: float
    recording_to_picture_mae_s: float
    picture_to_isi_mae_s: float
    wav_matches_all_trials: bool
    wav_order_is_monotonic: bool
    n_pause_events: int
    n_button_presses: int
    n_recorded_code_mismatches: int
    n_interpolated_picture_onsets: int
