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

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from mne_bids import BIDSPath, write_anat
import numpy as np
import pandas as pd

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


_ATLAS_KEYS_TO_SKIP = {"cfg", "coi"}


def _as_label_vector(value: Any, n_contacts: int) -> np.ndarray:
    """
    Convert a MATLAB-imported label field into a 1D array of length n_contacts.

    Supports:
    - cellstr / object array of strings
    - MATLAB char matrix: shape (n_contacts, max_len) or (max_len, n_contacts)

    Returns
    -------
    labels : np.ndarray
        Shape (n_contacts,), dtype=str
    """
    arr = np.asarray(value)

    # Case 1: already list-like / object-like strings
    if arr.ndim == 1:
        return arr.astype(str).reshape(-1)

    # Case 2: MATLAB char matrix (2D) -> join characters per row/col
    if arr.ndim == 2:
        # If it's a typical char matrix, elements are single-character strings/bytes.
        # Try interpreting rows as labels first.
        if arr.shape[0] == n_contacts:
            labels = ["".join(map(str, row)).strip() for row in arr]
            return np.asarray(labels, dtype=str)

        # Or columns as labels
        if arr.shape[1] == n_contacts:
            labels = ["".join(map(str, col)).strip() for col in arr.T]
            return np.asarray(labels, dtype=str)

    # Fallback: stringify and hope, but it likely won't match n_contacts
    return arr.astype(str).reshape(-1)


def _as_1d_str_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    return arr.astype(str).reshape(-1)


def _as_2d_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected array of shape (n, 3), got {arr.shape}.")
    return arr


def _extract_prob_per_contact(prob: Any, n_contacts: int) -> Optional[np.ndarray]:
    if prob is None:
        return None

    prob_arr = np.asarray(prob, dtype=float)

    if prob_arr.ndim == 1 and prob_arr.shape[0] == n_contacts:
        return prob_arr

    if prob_arr.ndim == 2 and prob_arr.shape[0] == n_contacts:
        # Conservative: keep the max probability per contact
        return prob_arr.max(axis=1)

    return None


def _resolve_derivatives_subject_dir(
    *,
    bids_root: Path,
    derivatives_name: str,
    bids_subject: str,
    session: Optional[str],
) -> Path:
    """
    Resolve derivatives directory for a subject (and optional session).

    Notes
    -----
    BIDS derivatives commonly nest session folders, but many pipelines don't.
    We'll support an optional session nesting to stay future-proof.

    Usage example
    -------------
        out_dir = _resolve_derivatives_subject_dir(
            bids_root=Path("bids"),
            derivatives_name="elec_recon",
            bids_subject="NicEle",
            session="01",
        )
    """
    base = bids_root / "derivatives" / derivatives_name / f"sub-{bids_subject}"
    if session is None:
        return base
    return base / f"ses-{session}"


def _build_electrodes_tsv_name(
    *,
    bids_subject: str,
    session: Optional[str],
    coords_space: str,
) -> str:
    """
    Build a derivatives electrodes TSV filename.

    Usage example
    -------------
        fname = _build_electrodes_tsv_name(bids_subject="NicEle", session=None, coords_space="MNI152")
    """
    parts = [f"sub-{bids_subject}"]
    if session is not None:
        parts.append(f"ses-{session}")
    parts.append(f"space-{coords_space}")
    parts.append("electrodes.tsv")
    return "_".join(parts)


def _extract_bids_electrodes_df(
    electrodes_df: pd.DataFrame,
    *,
    coords_space: str,
) -> pd.DataFrame:
    """
    Convert a parsed electrodes_df into BIDS-like electrodes.tsv columns.

    Parameters
    ----------
    electrodes_df
        DataFrame as returned by `parse_elec2atlas_payload`.
        Expected columns include:
        - contact
        - x_mni, y_mni, z_mni
        - x_ori, y_ori, z_ori
    coords_space
        "MNI152" to export x_mni/y_mni/z_mni, or "orig" to export x_ori/y_ori/z_ori.

    Returns
    -------
    pd.DataFrame
        Columns: name, x, y, z

    Usage example
    -------------
        out_df = _extract_bids_electrodes_df(electrodes_df, coords_space="MNI152")
    """
    if coords_space.lower() in {"mni", "mni152", "mni152nl", "mni152nlin2009c"}:
        x_col, y_col, z_col = "x_mni", "y_mni", "z_mni"
    else:
        x_col, y_col, z_col = "x_ori", "y_ori", "z_ori"

    for col in ("contact", x_col, y_col, z_col):
        if col not in electrodes_df.columns:
            raise KeyError(f"Missing required column in electrodes_df: {col}")

    out = pd.DataFrame(
        {
            "name": electrodes_df["contact"].astype(str),
            "x": pd.to_numeric(electrodes_df[x_col], errors="coerce"),
            "y": pd.to_numeric(electrodes_df[y_col], errors="coerce"),
            "z": pd.to_numeric(electrodes_df[z_col], errors="coerce"),
        }
    )

    # Drop contacts with invalid coordinates
    out = out.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    return out


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

    # Optional: export normalized electrodes.tsv for downstream plotting/reporting.
    # This stays task-agnostic: it just normalizes a common recon output if present.
    export_electrodes_tsv_from_elec2atlas_mat(
        bids_root=cfg.bids_root,
        derivatives_name=cfg.derivatives_name,
        bids_subject=cfg.bids_subject,
        session=cfg.session,
        elec_recon_dir=recon_src,
        coords_space="MNI152",
        elec_mat_name="elec2atlas.mat",
        overwrite=cfg.overwrite,
    )


def export_electrodes_tsv_from_elec2atlas_mat(
    *,
    bids_root: Path,
    derivatives_name: str,
    bids_subject: str,
    session: Optional[str],
    elec_recon_dir: Path,
    coords_space: str,
    elec_mat_name: str = "elec2atlas.mat",
    overwrite: bool = True,
) -> Optional[Path]:
    """
    Export a BIDS-like electrodes.tsv from an elec2atlas MAT file.

    This reads a subject-level localization MAT file (produced by elec recon)
    and writes a normalized TSV suitable for downstream plotting/reporting.

    Parameters
    ----------
    bids_root
        BIDS root containing the derivatives folder.
    derivatives_name
        Derivatives folder name (e.g., "elec_recon" or "dcap_electrodes").
    bids_subject
        Subject label without "sub-".
    session
        Optional session label without "ses-".
    elec_recon_dir
        Source directory containing the MAT file.
    coords_space
        Coordinate space label used in the output filename (e.g., "MNI152", "orig").
    elec_mat_name
        MAT filename within elec_recon_dir (default: "elec2atlas.mat").
    overwrite
        Overwrite existing TSV if present.

    Returns
    -------
    Optional[Path]
        Path to the written TSV, or None if MAT file is missing.

    Usage example
    -------------
        out_path = export_electrodes_tsv_from_elec2atlas_mat(
            bids_root=Path("bids"),
            derivatives_name="elec_recon",
            bids_subject="NicEle",
            session=None,
            elec_recon_dir=Path("sourcedata/subjects_dir/Nic-Ele/elec_recon"),
            coords_space="MNI152",
        )
    """
    mat_path = elec_recon_dir / elec_mat_name
    if not mat_path.exists():
        return None

    # Local import to keep mat73 an optional dependency for users without elec recon
    import mat73  # type: ignore[import-not-found]

    payload: Mapping[str, Any] = mat73.loadmat(mat_path)

    # You already have this parser; reuse it here.
    electrodes_df, _atlas_df = parse_elec2atlas_payload(payload, keep_atlas_table=False)

    bids_like_df = _extract_bids_electrodes_df(electrodes_df, coords_space=coords_space)

    out_dir = _resolve_derivatives_subject_dir(
        bids_root=bids_root,
        derivatives_name=derivatives_name,
        bids_subject=bids_subject,
        session=session,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = _build_electrodes_tsv_name(
        bids_subject=bids_subject,
        session=session,
        coords_space=coords_space,
    )
    out_path = out_dir / out_name

    if out_path.exists() and not overwrite:
        return out_path

    bids_like_df.to_csv(out_path, sep="\t", index=False)
    return out_path


def export_subject_anat_electrodes_from_elec2atlas_mat(
    *,
    bids_root: Path,
    bids_subject: str,
    elec2atlas_mat_path: Path,
    overwrite: bool = True,
    coords_space: str = "MNI152",
    coords_units: str = "mm",
) -> Optional[tuple[Path, Path]]:
    """
    Export subject-level `anat/` electrodes.tsv + coordsystem.json from elec2atlas.

    Returns None when the MAT file is absent.
    """
    mat_path = Path(elec2atlas_mat_path)
    if not mat_path.exists():
        return None

    import mat73  # type: ignore[import-not-found]

    payload: Mapping[str, Any] = mat73.loadmat(mat_path)
    electrodes_df, _ = parse_elec2atlas_payload(payload, keep_atlas_table=False)
    bids_like_df = _extract_bids_electrodes_df(electrodes_df, coords_space=coords_space)

    subject_bare = bids_subject[4:] if str(bids_subject).startswith("sub-") else str(bids_subject)
    anat_dir = Path(bids_root) / f"sub-{subject_bare}" / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)

    electrodes_tsv = anat_dir / f"sub-{subject_bare}_electrodes.tsv"
    coordsystem_json = anat_dir / f"sub-{subject_bare}_coordsystem.json"

    if not electrodes_tsv.exists() or overwrite:
        bids_like_df.to_csv(electrodes_tsv, sep="\t", index=False)

    if not coordsystem_json.exists() or overwrite:
        payload_json = {
            "iEEGCoordinateSystem": coords_space,
            "iEEGCoordinateUnits": coords_units,
            "iEEGCoordinateSystemDescription": "Coordinates exported from elec_recon/elec2atlas.mat",
        }
        coordsystem_json.write_text(json.dumps(payload_json, indent=2) + "\n", encoding="utf-8")

    return electrodes_tsv, coordsystem_json


def parse_elec2atlas_payload(
    payload: Mapping[str, Any],
    *,
    keep_atlas_table: bool = True,
    wide_atlas_columns: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Parse elec2atlas MAT payload into electrode coordinates + atlas assignments.

    Parameters
    ----------
    payload
        Dict produced by scipy.io.loadmat(..., simplify_cells=True).
    keep_atlas_table
        If True, return a long-form atlas table with one row per (contact, atlas).
    wide_atlas_columns
        If True, also add wide columns `atlas_<Atlas>` and optional `prob_<Atlas>`
        into electrodes_df. (This is convenient but can get wide.)

    Returns
    -------
    electrodes_df
        One row per contact with coordinates in MNI and original space.
    atlas_df
        Long-form table with columns: contact, atlas, label, prob (optional).
        None if keep_atlas_table is False.

    Usage example
    -------------
        electrodes_df, atlas_df = parse_elec2atlas_payload(mat, keep_atlas_table=True)
    """
    if "coi" not in payload:
        raise KeyError("Expected top-level key 'coi' in elec2atlas payload.")

    coi = payload["coi"]
    if not isinstance(coi, dict):
        raise TypeError(f"Expected payload['coi'] to be dict, got {type(coi)}.")

    contacts = _as_1d_str_array(coi["label"])
    n_contacts = int(contacts.shape[0])

    elecpos_mni = _as_2d_float_array(coi["elecpos_mni"])
    elecpos_ori = _as_2d_float_array(coi["elecpos_ori"])

    if elecpos_mni.shape[0] != n_contacts:
        raise ValueError("coi['elecpos_mni'] length does not match number of contacts.")
    if elecpos_ori.shape[0] != n_contacts:
        raise ValueError("coi['elecpos_ori'] length does not match number of contacts.")

    electrodes_df = pd.DataFrame(
        {
            "contact": contacts,
            "x_mni": elecpos_mni[:, 0],
            "y_mni": elecpos_mni[:, 1],
            "z_mni": elecpos_mni[:, 2],
            "x_ori": elecpos_ori[:, 0],
            "y_ori": elecpos_ori[:, 1],
            "z_ori": elecpos_ori[:, 2],
        }
    )

    atlas_rows = []

    for atlas_name, atlas in payload.items():
        if atlas_name in _ATLAS_KEYS_TO_SKIP:
            continue
        if not isinstance(atlas, dict):
            continue
        if "label" not in atlas:
            continue

        labels = _as_label_vector(atlas["label"], n_contacts)
        if labels.shape[0] != n_contacts:
            continue

        prob_per_contact = _extract_prob_per_contact(atlas.get("prob", None), n_contacts)

        if keep_atlas_table:
            if prob_per_contact is None:
                atlas_rows.extend(
                    {"contact": contacts[i], "atlas": atlas_name, "label": labels[i]}
                    for i in range(n_contacts)
                )
            else:
                atlas_rows.extend(
                    {
                        "contact": contacts[i],
                        "atlas": atlas_name,
                        "label": labels[i],
                        "prob": float(prob_per_contact[i]),
                    }
                    for i in range(n_contacts)
                )

        if wide_atlas_columns:
            electrodes_df[f"atlas_{atlas_name}"] = labels
            if prob_per_contact is not None:
                electrodes_df[f"prob_{atlas_name}"] = prob_per_contact

    atlas_df = None
    if keep_atlas_table:
        atlas_df = pd.DataFrame(atlas_rows)
        # Keep a stable column order
        if "prob" in atlas_df.columns:
            atlas_df = atlas_df[["contact", "atlas", "label", "prob"]]
        else:
            atlas_df = atlas_df[["contact", "atlas", "label"]]

    return electrodes_df, atlas_df
