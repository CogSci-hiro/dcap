# src/dcap/bids/core/config.py
# =============================================================================
#                         BIDS Core: Conversion config
# =============================================================================
#
# Task-agnostic configuration objects for BIDS conversion.
#
# These configs describe *how* conversion should happen (paths, flags, global
# conventions), but not *what* a task is. Task-specific parameters must live
# under dcap.bids.tasks.<task>.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =============================================================================
# Core conversion config
# =============================================================================

@dataclass(frozen=True)
class BidsCoreConfig:
    """
    Core, task-agnostic configuration for BIDS conversion.

    Parameters
    ----------
    source_root
        Root directory containing source data for a single subject.
        Interpretation of its contents is task-dependent.
    bids_root
        Root directory of the BIDS dataset to write to.
    subject
        BIDS subject label (with or without "sub-" prefix).
    session
        Optional BIDS session label (with or without "ses-" prefix).
    datatype
        BIDS datatype ("ieeg", "eeg", "meg", "anat", ...).
    overwrite
        Whether to overwrite existing BIDS outputs where supported.
    dry_run
        If True, do not write any files; only execute logic up to the write step.
    preload_raw
        Whether to preload raw data into memory before writing.
        Some transforms (e.g., padding, montage operations) may require this.
    line_freq
        Power line frequency in Hz (e.g., 50 in EU, 60 in US).

    Usage example
    -------------
        cfg = BidsCoreConfig(
            source_root=Path("sourcedata/Nic-Ele"),
            bids_root=Path("bids"),
            subject="NicEle",
            session=None,
            datatype="ieeg",
            overwrite=False,
            dry_run=True,
            preload_raw=True,
            line_freq=50.0,
        )
    """

    source_root: Path
    bids_root: Path

    subject: str
    session: Optional[str]
    datatype: str

    overwrite: bool
    dry_run: bool
    preload_raw: bool
    line_freq: float


# =============================================================================
# Optional anatomy config (core-level, task-independent)
# =============================================================================

@dataclass(frozen=True)
class BidsAnatConfig:
    """
    Optional configuration for anatomy writing.

    This config is intentionally separate from `BidsCoreConfig` so that anatomy
    can be:
      - run as part of a task
      - run once per subject
      - or exposed as a separate CLI command later

    Parameters
    ----------
    subjects_dir
        FreeSurfer SUBJECTS_DIR containing reconstructions.
    original_id
        Subject identifier under SUBJECTS_DIR (often original clinical ID).
    bids_subject
        BIDS subject label without "sub-" prefix.
    session
        Optional BIDS session label without "ses-" prefix.
    deface
        Whether to deface the anatomical image.
    overwrite
        Whether to overwrite existing outputs where supported.

    Usage example
    -------------
        anat_cfg = BidsAnatConfig(
            subjects_dir=Path("sourcedata/subjects_dir"),
            original_id="Nic-Ele",
            bids_subject="NicEle",
            session=None,
            deface=False,
            overwrite=True,
        )
    """

    subjects_dir: Path
    original_id: str
    bids_subject: str
    session: Optional[str]

    deface: bool
    overwrite: bool
