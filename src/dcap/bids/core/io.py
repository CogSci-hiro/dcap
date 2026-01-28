# src/dcap/bids/core/io.py
# =============================================================================
#                             BIDS Core: I/O
# =============================================================================
#
# Task-agnostic I/O utilities:
# - small file hygiene fixes (e.g., BrainVision header byte quirks)
# - minimal raw loaders by explicit format (called by tasks)
#
# This module must not encode task policy (no filename conventions, no run logic).
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

import codecs
from pathlib import Path
from typing import Literal

import mne


# =============================================================================
# File hygiene
# =============================================================================

def fix_brainvision_header_utf8(vhdr_path: Path) -> None:
    """
    Fix common BrainVision header encoding issues in-place.

    Some BrainVision headers contain the byte 0xB5 (µ), which can break UTF-8
    parsing depending on downstream tooling. This function replaces that byte
    with ASCII 'u'.

    Parameters
    ----------
    vhdr_path
        Path to a .vhdr file.

    Returns
    -------
    None

    Usage example
    -------------
        fix_brainvision_header_utf8(Path("conversation_1.vhdr"))
    """
    if not vhdr_path.exists():
        raise FileNotFoundError(f"BrainVision header not found: {vhdr_path}")

    with codecs.open(vhdr_path, "rb") as f:
        data = f.read()

    fixed = data.replace(b"\xb5", b"u")
    if fixed == data:
        return

    with open(vhdr_path, "wb") as f:
        f.write(fixed)


# =============================================================================
# Raw loaders (task calls explicitly)
# =============================================================================

RawFormat = Literal["brainvision", "edf", "fif"]


def load_raw(path: Path, raw_format: RawFormat, preload: bool) -> mne.io.BaseRaw:
    """
    Load raw data in a task-agnostic way.

    Notes
    -----
    Tasks should call this with an explicit format; do not guess formats here.
    Tasks can also bypass this helper if they need specialized loaders.

    Parameters
    ----------
    path
        Path to the raw file (e.g., .vhdr, .edf, .fif).
    raw_format
        One of {"brainvision", "edf", "fif"}.
    preload
        Whether to preload data.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw object.

    Usage example
    -------------
        raw = load_raw(Path("conversation_1.vhdr"), raw_format="brainvision", preload=True)
    """
    if raw_format == "brainvision":
        return load_raw_brainvision(path, preload=preload)

    if raw_format == "edf":
        return load_raw_edf(path, preload=preload)

    if raw_format == "fif":
        return load_raw_fif(path, preload=preload)

    raise ValueError(f"Unsupported raw_format: {raw_format}")


def load_raw_brainvision(vhdr_path: Path, preload: bool) -> mne.io.BaseRaw:
    """
    Load BrainVision format.

    Parameters
    ----------
    vhdr_path
        Path to the .vhdr file.
    preload
        Whether to preload data.

    Returns
    -------
    mne.io.BaseRaw

    Usage example
    -------------
        raw = load_raw_brainvision(Path("conversation_1.vhdr"), preload=False)
    """
    fix_brainvision_header_utf8(vhdr_path)
    return mne.io.read_raw_brainvision(vhdr_path, preload=preload, verbose=False)


def load_raw_edf(edf_path: Path, preload: bool) -> mne.io.BaseRaw:
    """
    Load EDF format.

    Parameters
    ----------
    edf_path
        Path to the .edf file.
    preload
        Whether to preload data.

    Returns
    -------
    mne.io.BaseRaw

    Usage example
    -------------
        raw = load_raw_edf(Path("recording.edf"), preload=False)
    """
    return mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)


def load_raw_fif(fif_path: Path, preload: bool) -> mne.io.BaseRaw:
    """
    Load FIF format.

    Parameters
    ----------
    fif_path
        Path to the .fif file.
    preload
        Whether to preload data.

    Returns
    -------
    mne.io.BaseRaw

    Usage example
    -------------
        raw = load_raw_fif(Path("raw.fif"), preload=False)
    """
    return mne.io.read_raw_fif(fif_path, preload=preload, verbose=False)
