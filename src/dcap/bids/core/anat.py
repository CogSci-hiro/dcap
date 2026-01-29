# src/dcap/bids/core/anat.py
# =============================================================================
#                            BIDS Core: Anatomy
# =============================================================================
#
# Task-independent anatomy utilities:
# - Write ACPC-aligned T1w into BIDS using MNE-BIDS `write_anat`
# - Copy electrode reconstruction outputs into BIDS derivatives
#
# This module must remain task-agnostic. Task-specific decisions (e.g., whether
# to write anatomy at all, or which derivative folders to copy) should be made
# by tasks or higher-level orchestration code.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mne_bids import BIDSPath, write_anat

# =============================================================================
# Constants
# =============================================================================

DEFAULT_T1_MGZ_RELATIVE = Path("mri") / "T1.mgz"
DEFAULT_ELEC_RECON_RELATIVE = Path("elec_recon")


# =============================================================================
# Public config
# =============================================================================

@dataclass(frozen=True)
class AnatWriteConfig:
    """
    Configuration for writing anatomy and copying recon derivatives.

    Parameters
    ----------
    subjects_dir
        FreeSurfer SUBJECTS_DIR containing per-subject recon outputs.
    original_id
        Subject directory name under subjects_dir (often original clinical ID, may include hyphens).
    bids_root
        Output BIDS root.
    bids_subject
        BIDS subject label WITHOUT the "sub-" prefix (e.g., "NicEle" or "001").
    session
        Optional BIDS session label WITHOUT "ses-" prefix (e.g., "01").
        If provided, the T1w will be written under that session. If None, writes
        subject-level anat (no session folder).
    deface
        Whether to deface the anatomical image (passed to mne_bids.write_anat).
    overwrite
        Whether to overwrite existing outputs where supported.
    t1_mgz_path
        Optional explicit path to the FreeSurfer T1.mgz. If None, uses:
        subjects_dir / original_id / "mri/T1.mgz"
    elec_recon_dir
        Optional explicit path to electrode recon folder. If None, uses:
        subjects_dir / original_id / "elec_recon"
    derivatives_name
        Derivatives folder name under BIDS root (default: "elec_recon").
    copy_elec_recon
        If True, copy electrode recon folder to BIDS derivatives.

    Usage example
    -------------
        cfg = AnatWriteConfig(
            subjects_dir=Path("sourcedata/subjects_dir"),
            original_id="Nic-Ele",
            bids_root=Path("bids"),
            bids_subject="NicEle",
            session=None,
            deface=False,
            overwrite=True,
        )
        write_anat_and_derivatives(cfg)
    """

    subjects_dir: Path
    original_id: str
    bids_root: Path
    bids_subject: str
    session: Optional[str]
    deface: bool
    overwrite: bool
    t1_mgz_path: Optional[Path] = None
    elec_recon_dir: Optional[Path] = None
    derivatives_name: str = "elec_recon"
    copy_elec_recon: bool = True


# =============================================================================
# Helpers
# =============================================================================

def _resolve_t1_path(cfg: AnatWriteConfig) -> Path:
    """Prefer explicitly provided T1 path; otherwise fall back to default FreeSurfer location"""
    if cfg.t1_mgz_path is not None:
        return cfg.t1_mgz_path
    return cfg.subjects_dir / cfg.original_id / DEFAULT_T1_MGZ_RELATIVE


def _resolve_elec_recon_dir(cfg: AnatWriteConfig) -> Path:
    """Prefer explicitly provided electrode reconstruction directory; otherwise use default layout"""
    if cfg.elec_recon_dir is not None:
        return cfg.elec_recon_dir
    return cfg.subjects_dir / cfg.original_id / DEFAULT_ELEC_RECON_RELATIVE


def _bids_anat_path(cfg: AnatWriteConfig) -> BIDSPath:
    """Construct BIDS T1w anatomical path from config"""
    return BIDSPath(
        root=cfg.bids_root,
        subject=cfg.bids_subject,
        session=cfg.session,
        datatype="anat",
        suffix="T1w",
        extension=".nii.gz",
    )


# =============================================================================
# Public API
# =============================================================================

def write_anat_and_derivatives(cfg: AnatWriteConfig) -> None:
    """
    Write T1w anatomy into BIDS and optionally copy electrode recon derivatives.

    Parameters
    ----------
    cfg
        Anatomy writing configuration.

    Returns
    -------
    None

    Notes
    -----
    - This function writes a BIDS-compliant T1w file via MNE-BIDS.
    - If `copy_elec_recon` is True, it copies a FreeSurfer `elec_recon/` folder
      into: <bids_root>/derivatives/<derivatives_name>/sub-<subject>/...
    - This function does NOT implement any de-identification policy beyond the
      `deface` toggle. Decide and audit your de-ID policy elsewhere.

    Usage example
    -------------
        cfg = AnatWriteConfig(
            subjects_dir=Path("sourcedata/subjects_dir"),
            original_id="Nic-Ele",
            bids_root=Path("bids"),
            bids_subject="NicEle",
            session=None,
            deface=False,
            overwrite=True,
        )
        write_anat_and_derivatives(cfg)
    """
    cfg.bids_root.mkdir(parents=True, exist_ok=True)

    t1_path = _resolve_t1_path(cfg)
    if not t1_path.exists():
        raise FileNotFoundError(f"T1 file not found: {t1_path}")

    bids_path = _bids_anat_path(cfg)

    # Write ACPC-aligned T1w into BIDS
    write_anat(
        t1_path,
        bids_path,
        deface=cfg.deface,
        landmarks=None,
        overwrite=cfg.overwrite,
    )

    if not cfg.copy_elec_recon:
        return

    recon_src = _resolve_elec_recon_dir(cfg)
    if not recon_src.exists():
        # Keep this non-fatal by default: many datasets may not have recon ready yet.
        return

    recon_dst = (
        cfg.bids_root
        / "derivatives"
        / cfg.derivatives_name
        / f"sub-{cfg.bids_subject}"
    )
    recon_dst.mkdir(parents=True, exist_ok=True)

    shutil.copytree(recon_src, recon_dst, dirs_exist_ok=True)
