# =============================================================================
#                          BIDS: Conversion config
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DiapixTimingConfig:
    """
    Timing conventions for the Diapix-style recordings.

    Parameters
    ----------
    start_delay_s
        Delay between start-of-audio file and actual task start marker (seconds).
    conversation_duration_s
        Duration of the task segment to retain (seconds).

    Usage example
    -------------
        timing = DiapixTimingConfig(start_delay_s=4.0, conversation_duration_s=240.0)
    """
    start_delay_s: float
    conversation_duration_s: float


@dataclass(frozen=True)
class BidsConvertConfig:
    """
    End-to-end configuration for converting source data to BIDS.

    Notes
    -----
    This config is intentionally explicit: no hidden globals.

    Usage example
    -------------
        cfg = BidsConvertConfig(
            source_root=Path("sourcedata/Nic-Ele"),
            bids_root=Path("bids"),
            subject="NicEle",
            session=None,
            task="diapix",
            datatype="ieeg",
            run=None,
            line_freq=50.0,
            channels_tsv=Path("config/channels.tsv"),
            audio_onsets_tsv=Path("config/audio_onsets.tsv"),
            stim_wav=Path("config/beeps.wav"),
            subjects_dir=Path("sourcedata/subjects_dir"),
            original_id="Nic-Ele",
            atlas_file=Path("sourcedata/subjects_dir/Nic-Ele/elec_recon/elec2atlas.mat"),
            timing=DiapixTimingConfig(start_delay_s=4.0, conversation_duration_s=240.0),
            overwrite=False,
            dry_run=True,
        )
    """

    source_root: Path
    bids_root: Path

    subject: str
    session: Optional[str]
    task: str
    datatype: str
    run: Optional[str]

    line_freq: float

    channels_tsv: Optional[Path]
    audio_onsets_tsv: Optional[Path]
    stim_wav: Optional[Path]

    subjects_dir: Optional[Path]
    original_id: Optional[str]
    atlas_file: Optional[Path]

    timing: DiapixTimingConfig

    overwrite: bool
    dry_run: bool
