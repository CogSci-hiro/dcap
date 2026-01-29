from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DiapixRecordingUnit:
    """
    A single Diapix recording unit (one run).

    Attributes
    ----------
    dcap_id
        Private DCAP subject identifier (e.g., "Nic-Ele"). Used to look up
        private metadata (trigger mapping, audio onset table). Must never be
        written into BIDS outputs.
    run
        Run number as string (e.g., "1", "2"). Canonical form is no leading
        zeros (i.e., "1" not "01").
    vhdr_path
        BrainVision header file for iEEG.
    wav_path
        Raw audio WAV (uncropped), if available.
    video_path
        Optional ASF video file, if available.

    Usage example
    -------------
        unit = DiapixRecordingUnit(
            dcap_id="Nic-Ele",
            run="1",
            vhdr_path=Path("conversation_1.vhdr"),
            wav_path=Path("conversation_1.wav"),
            video_path=None,
        )
    """

    subject_bids: str
    dcap_id: str
    session: Optional[str]
    run: str
    vhdr_path: Path
    wav_path: Optional[Path]
    video_path: Optional[Path]


@dataclass(frozen=True)
class DiapixTiming:
    """
    Timing constants for Diapix.

    Attributes
    ----------
    conversation_duration_s
        Expected conversation duration (seconds).
    start_delay_s
        Offset from the first sync reference to the actual conversation start.
        (In your old code: START_DELAY = 4)
    """

    conversation_duration_s: float = 4 * 60
    start_delay_s: float = 4.0
