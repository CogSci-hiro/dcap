# =============================================================================
#                           BIDS: Task interface
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import mne
import numpy as np
from mne_bids import BIDSPath


@dataclass(frozen=True)
class RecordingUnit:
    """
    One convertable unit (typically one run).

    Usage example
    -------------
        unit = RecordingUnit(
            run="1",
            raw_path=Path("conversation_1.vhdr"),
            audio_path=Path("conversation_1.wav"),
            video_path=Path("conversation_1.asf"),
        )
    """
    run: Optional[str]
    raw_path: Path
    audio_path: Optional[Path]
    video_path: Optional[Path]


@dataclass(frozen=True)
class PreparedEvents:
    """
    Events prepared for MNE-BIDS.

    Usage example
    -------------
        prepared = PreparedEvents(events=events, event_id=event_id)
    """
    events: Optional[np.ndarray]
    event_id: Optional[dict[str, int]]


class BidsTask(Protocol):
    """
    Protocol for task-specific conversion logic.

    A task implementation provides discovery, raw loading, and optional hooks.
    """

    name: str  # e.g., "diapix"

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        ...

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        ...

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        ...

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        ...
