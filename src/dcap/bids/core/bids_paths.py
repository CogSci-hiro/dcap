# src/dcap/bids/core/bids_paths.py
# =============================================================================
#                         BIDS Core: BIDSPath helpers
# =============================================================================
#
# Task-agnostic helpers for constructing BIDSPath objects and normalizing
# BIDS entity labels (subject/session/run/etc.).
#
# This module must remain free of task-specific assumptions.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from typing import Optional

from mne_bids import BIDSPath


# =============================================================================
# Label normalization
# =============================================================================

def normalize_bids_label(value: Optional[str], prefix: str) -> Optional[str]:
    """
    Normalize a BIDS label by removing an optional prefix.

    Examples
    --------
    - ("sub-001", "sub") -> "001"
    - ("001", "sub")     -> "001"
    - (None, "ses")      -> None

    Parameters
    ----------
    value
        Input label, possibly already prefixed.
    prefix
        The BIDS entity prefix without hyphen (e.g., "sub", "ses", "run").

    Returns
    -------
    Optional[str]
        Normalized label without prefix.

    Usage example
    -------------
        subject = normalize_bids_label("sub-001", "sub")  # "001"
        session = normalize_bids_label("01", "ses")       # "01"
    """
    if value is None:
        return None

    value_str = str(value).strip()
    if value_str == "":
        return None

    prefixed = f"{prefix}-"
    if value_str.startswith(prefixed):
        return value_str[len(prefixed):]

    return value_str


# =============================================================================
# BIDSPath construction
# =============================================================================

def build_bids_path(
    bids_root: Path,
    subject: str,
    session: Optional[str],
    task: Optional[str],
    datatype: str,
    run: Optional[str],
    suffix: Optional[str] = None,
    extension: Optional[str] = None,
) -> BIDSPath:
    """
    Construct an MNE-BIDS BIDSPath with normalized labels.

    Parameters
    ----------
    bids_root
        Root directory of the BIDS dataset.
    subject
        BIDS subject label (with or without "sub-" prefix).
    session
        Optional session label (with or without "ses-" prefix).
    task
        Optional task label.
    datatype
        One of {"ieeg", "eeg", "meg", "anat", ...}.
    run
        Optional run label (with or without "run-" prefix).
    suffix
        Optional suffix (e.g., "ieeg", "eeg", "meg", "T1w"). Often inferred by MNE-BIDS.
    extension
        Optional extension including dot (e.g., ".edf", ".fif", ".vhdr", ".nii.gz").

    Returns
    -------
    mne_bids.BIDSPath
        Constructed BIDSPath.

    Notes
    -----
    - MNE-BIDS generally infers some filename parts from datatype, but explicitly
      setting suffix/extension can be useful for anat or specific formats.

    Usage example
    -------------
        bids_path = build_bids_path(
            bids_root=Path("bids"),
            subject="sub-001",
            session="ses-01",
            task="diapix",
            datatype="ieeg",
            run="run-1",
            suffix="ieeg",
            extension=".edf",
        )
    """
    subject_norm = normalize_bids_label(subject, "sub")
    session_norm = normalize_bids_label(session, "ses") if session is not None else None
    run_norm = normalize_bids_label(run, "run") if run is not None else None

    if subject_norm is None:
        raise ValueError("Subject label is required to build a BIDSPath.")

    return BIDSPath(
        root=bids_root,
        subject=subject_norm,
        session=session_norm,
        task=task,
        run=run_norm,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
    )


def build_bids_file_path(
    *,
    bids_root: Path,
    subject: str,
    session: Optional[str],
    task: Optional[str],
    datatype: str,
    run: Optional[str],
    suffix: str,
    extension: str,
) -> Path:
    """
    Build a filesystem path for BIDS-like files with extensions unsupported by MNE-BIDS.

    This reuses `build_bids_path()` for entity normalization and filename stem construction,
    then swaps to the desired extension on the resulting path.
    """
    if not extension.startswith("."):
        raise ValueError(f"Extension must start with '.', got {extension!r}")

    # MNE-BIDS validates suffixes/extensions and rejects media labels like
    # "audio"/"video" and extensions like ".wav"/".mp4". We use an allowed
    # anchor path for entity normalization, then rewrite the terminal suffix.
    anchor_suffix = suffix
    if datatype == "beh" and suffix in {"audio", "video"}:
        anchor_suffix = "beh"

    anchor = build_bids_path(
        bids_root=bids_root,
        subject=subject,
        session=session,
        task=task,
        datatype=datatype,
        run=run,
        suffix=anchor_suffix,
        extension=".json",
    )
    out_path = Path(anchor.fpath)
    if anchor_suffix != suffix:
        stem = out_path.stem
        token = f"_{anchor_suffix}"
        if stem.endswith(token):
            stem = stem[: -len(token)] + f"_{suffix}"
            out_path = out_path.with_name(stem + out_path.suffix)
        else:
            raise ValueError(f"Unexpected anchor filename when rewriting suffix: {out_path.name}")
    return out_path.with_suffix(extension)
