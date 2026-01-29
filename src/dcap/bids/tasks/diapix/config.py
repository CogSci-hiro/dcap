# dcap/bids/tasks/diapix/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class DiapixTimingConfig:
    """
    Task-specific timing parameters for Diapix.

    Usage example
    -------------
        timing = DiapixTimingConfig(
            start_delay_s=4.0,
            conversation_duration_s=240.0,
        )
    """
    start_delay_s: float = 4.0
    conversation_duration_s: float = 4.0 * 60.0


@dataclass(frozen=True, slots=True)
class DiapixAssetsConfig:
    """
    Task assets needed for Diapix conversion.

    Usage example
    -------------
        assets = DiapixAssetsConfig(
            audio_onsets_tsv=Path("audio_onsets.tsv"),
            stim_wav=Path("beeps.wav"),
            atlas_path=Path("elec2atlas.mat"),
        )
    """
    audio_onsets_tsv: Path
    stim_wav: Path
    atlas_path: Path
