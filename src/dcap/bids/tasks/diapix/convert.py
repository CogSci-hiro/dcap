# =============================================================================
#                      Library: Diapix BIDS conversion
# =============================================================================
#
# No CLI logic here. This module:
# - defines a task-level library config
# - builds core config + task implementation
# - calls the single core entry point: convert_subject()
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dcap.bids.core.config import BidsCoreConfig
from dcap.bids.core.converter import convert_subject
from dcap.bids.tasks.diapix.task import DiapixTask


@dataclass(frozen=True, slots=True)
class DiapixConvertConfig:
    """
    Library configuration for Diapix BIDS conversion.

    Usage example
    -------------
        cfg = DiapixConvertConfig(
            source_root=Path("sourcedata/NicEle"),
            bids_root=Path("bids"),
            subject="NicEle",
            session=None,
            overwrite=False,
            dry_run=False,
            preload_raw=True,
            line_freq_hz=50.0,
            audio_onsets_tsv=Path("audio_onsets.tsv"),
            stim_wav=Path("beeps.wav"),
            atlas_path=Path("elec2atlas.mat"),
        )
    """

    source_root: Path
    bids_root: Path
    subject: str
    session: Optional[str]

    overwrite: bool
    dry_run: bool
    preload_raw: bool
    line_freq_hz: float

    audio_onsets_tsv: Path
    stim_wav: Path
    atlas_path: Path


def convert_diapix(cfg: DiapixConvertConfig) -> int:
    """
    Convert Diapix data into BIDS using the task-agnostic core.

    Returns
    -------
    int
        Exit code (0 = success).

    Usage example
    -------------
        exit_code = convert_diapix(cfg)
    """
    core_cfg = BidsCoreConfig(
        source_root=cfg.source_root,
        bids_root=cfg.bids_root,
        subject=cfg.subject,
        session=cfg.session,
        datatype="ieeg",
        overwrite=cfg.overwrite,
        dry_run=cfg.dry_run,
        preload_raw=cfg.preload_raw,
        line_freq=cfg.line_freq_hz,
    )

    task = DiapixTask(
        subject_bids=cfg.subject,
        audio_onsets_tsv=cfg.audio_onsets_tsv,
        stim_wav=cfg.stim_wav,
        atlas_path=cfg.atlas_path,
    )

    _ = convert_subject(cfg=core_cfg, task=task)
    return 0
