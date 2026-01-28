# dcap/bids/tasks/diapix/models.py
# =============================================================================
#                               DIAPIX MODELS
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DiapixRecordingUnit:
    """
    A single Diapix recording unit (one run).

    Attributes
    ----------
    subject_bids
        BIDS subject label without "sub-" prefix (e.g., "NicEle").
    session
        Optional session label without "ses-" prefix.
    run
        Run number as string (e.g., "1", "2").
    vhdr_path
        BrainVision header file.
    wav_path
        Raw audio WAV (uncropped).
    video_path
        Optional ASF video.

    Usage example
    -------------
        unit = DiapixRecordingUnit(
            subject_bids="NicEle",
            session=None,
            run="1",
            vhdr_path=Path("conversation_1.vhdr"),
            wav_path=Path("conversation_1.wav"),
            video_path=None,
        )
    """

    subject_bids: str
    session: Optional[str]
    run: str
    vhdr_path: Path
    wav_path: Path
    video_path: Optional[Path]
