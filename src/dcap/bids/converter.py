# =============================================================================
#                       DCAP: Source → BIDS conversion
# =============================================================================
# > High-level orchestration for converting source recordings into BIDS with MNE-BIDS.
# > Dataset-specific logic should live in `heuristics.py` and `metadata.py`.

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mne
from mne_bids import BIDSPath, write_raw_bids

from dcap_bids.heuristics import SourceItem, discover_source_items
from dcap_bids.io import load_source_raw
from dcap_bids.metadata import (
    BidsSidecars,
    build_bids_path,
    infer_subject_session_run,
    make_bids_sidecars,
)
from dcap_bids.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ConvertConfig:
    """
    Configuration for conversion.

    Parameters
    ----------
    source_root
        Root directory containing messy clinical source recordings.
    bids_root
        Output BIDS root directory.
    subject
        BIDS subject label. Accepts '001' or 'sub-001' (normalized internally).
    session
        Optional session label. Accepts '01' or 'ses-01' (normalized internally).
    task
        BIDS task label (e.g., 'conversation').
    datatype
        One of {'ieeg', 'eeg', 'meg'}.
    run
        Optional run label.
    line_freq
        Power line frequency in Hz (typically 50 in EU, 60 in US).
    overwrite
        Overwrite existing BIDS outputs if present.
    dry_run
        If True, do not write outputs; just log what would happen.

    Usage example
        cfg = ConvertConfig(
            source_root=Path("./source"),
            bids_root=Path("./bids"),
            subject="001",
            session="01",
            task="conversation",
            datatype="ieeg",
            run="01",
            line_freq=50.0,
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
    overwrite: bool
    dry_run: bool


def convert_all(cfg: ConvertConfig) -> None:
    """
    Convert all discovered source items to BIDS.

    This is the main orchestration function:
      1) discover source recordings
      2) load each recording into an MNE Raw
      3) compute metadata + BIDSPath
      4) write to BIDS using MNE-BIDS

    Parameters
    ----------
    cfg
        Conversion configuration.

    Usage example
        convert_all(cfg)
    """
    cfg.bids_root.mkdir(parents=True, exist_ok=True)

    items = list(discover_source_items(cfg.source_root))
    LOGGER.info("Discovered %d source item(s) under %s", len(items), cfg.source_root)

    for item in items:
        _convert_one(item=item, cfg=cfg)


def _convert_one(item: SourceItem, cfg: ConvertConfig) -> None:
    """
    Convert a single source item to BIDS.

    Parameters
    ----------
    item
        A discovered source item (path + hints).
    cfg
        Conversion configuration.

    Usage example
        _convert_one(item, cfg)
    """
    subject, session, run = infer_subject_session_run(
        subject=cfg.subject,
        session=cfg.session,
        run=cfg.run,
        item=item,
    )

    bids_path = build_bids_path(
        bids_root=cfg.bids_root,
        subject=subject,
        session=session,
        task=cfg.task,
        datatype=cfg.datatype,
        run=run,
    )

    LOGGER.info("Converting: %s → %s", item.source_path, bids_path)

    raw = load_source_raw(item)
    _apply_minimal_normalization(raw=raw, line_freq=cfg.line_freq)

    sidecars = make_bids_sidecars(item=item, raw=raw, task=cfg.task, datatype=cfg.datatype)

    if cfg.dry_run:
        LOGGER.info("[DRY RUN] Would write BIDS for %s", bids_path)
        return

    # Write the data and sidecars using MNE-BIDS.
    # Note: allow_preload=False lets MNE-BIDS decide; dataset-specific tuning can go later.
    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        overwrite=cfg.overwrite,
        format="auto",
        allow_preload=False,
        anonymize=None,  # IMPORTANT: keep as None until you implement a safe de-ID policy
        events=None,
        event_id=None,
        verbose=False,
    )

    # Sidecars: write extra metadata *after* write_raw_bids so file tree exists.
    sidecars.write_all(bids_path=bids_path)

    LOGGER.info("Done: %s", bids_path)


def _apply_minimal_normalization(raw: mne.io.BaseRaw, line_freq: float) -> None:
    """
    Apply minimal normalization to Raw before writing BIDS.

    Parameters
    ----------
    raw
        The MNE Raw object to be modified in-place.
    line_freq
        Power line frequency in Hz.

    Notes
    -----
    Keep this intentionally tiny. Anything more opinionated should live in preprocessing,
    not in the "format conversion" step.

    Usage example
        _apply_minimal_normalization(raw, line_freq=50.0)
    """
    raw.info["line_freq"] = float(line_freq)

    # Optional: enforce a meas_date policy (you'll likely want to blank it for de-ID).
    # raw.set_meas_date(None)

    # Optional: channel type mapping could go here, but usually belongs in heuristics.
    # raw.set_channel_types({...})
