# src/dcap/bids/core/sidecars.py
# =============================================================================
#                          BIDS Core: Sidecar helpers
# =============================================================================
#
# Task-agnostic helpers for constructing and writing sidecar JSON files.
#
# Notes
# -----
# - MNE-BIDS writes many required sidecars automatically.
# - This module is for *optional* additional JSON content that tasks may want to
#   add (e.g., extra Manufacturer fields, custom metadata fields, etc.).
# - Core does not decide *what* fields should exist; tasks/orchestrators do.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mne_bids import BIDSPath


# =============================================================================
# Public containers
# =============================================================================

@dataclass(frozen=True)
class JsonSidecar:
    """
    A JSON sidecar payload intended to be written next to a BIDS recording.

    Parameters
    ----------
    content
        JSON-serializable mapping.
    extension
        File extension to use. Defaults to ".json".

    Usage example
    -------------
        sidecar = JsonSidecar(content={"TaskName": "diapix"})
        sidecar.write(bids_path)
    """

    content: Mapping[str, Any]
    extension: str = ".json"

    def write(self, bids_path: BIDSPath, overwrite: bool) -> Path:
        """
        Write the sidecar to disk.

        Parameters
        ----------
        bids_path
            BIDSPath for the recording. The sidecar is written "next to" this path.
        overwrite
            If False and file exists, raises FileExistsError.

        Returns
        -------
        Path
            Path to the written sidecar file.

        Usage example
        -------------
            written_path = sidecar.write(bids_path, overwrite=True)
        """
        target = _sidecar_path_for_recording(bids_path=bids_path, extension=self.extension)
        _write_json(target, content=dict(self.content), overwrite=overwrite)
        return target


# =============================================================================
# Public helpers
# =============================================================================

def merge_json_mappings(*mappings: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Merge multiple JSON-like mappings, later mappings override earlier ones.

    Parameters
    ----------
    *mappings
        Any number of mappings (or None). None values are ignored.

    Returns
    -------
    dict[str, Any]
        Merged dictionary.

    Usage example
    -------------
        base = {"TaskName": "diapix", "Manufacturer": "n/a"}
        extra = {"Manufacturer": "Acme"}
        merged = merge_json_mappings(base, extra)
    """
    out: dict[str, Any] = {}
    for mapping in mappings:
        if mapping is None:
            continue
        for key, value in mapping.items():
            out[str(key)] = value
    return out


def remove_none_values(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a copy of `mapping` with keys removed where value is None.

    Usage example
    -------------
        clean = remove_none_values({"a": 1, "b": None})  # {"a": 1}
    """
    return {str(k): v for k, v in mapping.items() if v is not None}


# =============================================================================
# Internal helpers
# =============================================================================

def _sidecar_path_for_recording(bids_path: BIDSPath, extension: str) -> Path:
    """
    Compute the sidecar JSON path for a recording-level BIDSPath.

    Parameters
    ----------
    bids_path
        BIDSPath for the recording (e.g., ieeg).
    extension
        Extension including leading dot (e.g., ".json").

    Returns
    -------
    Path
        The full filesystem path for the sidecar.

    Usage example
    -------------
        json_path = _sidecar_path_for_recording(bids_path, ".json")
    """
    if not extension.startswith("."):
        raise ValueError(f"extension must start with '.', got: {extension}")

    # Make sure the base directory exists.
    bids_path.mkdir()

    sidecar_bids_path = bids_path.copy().update(extension=extension)
    if sidecar_bids_path.fpath is None:
        raise RuntimeError("MNE-BIDS did not produce a valid file path for sidecar.")
    return Path(sidecar_bids_path.fpath)


def _write_json(path: Path, content: Mapping[str, Any], overwrite: bool) -> None:
    """
    Write JSON to disk with pretty formatting.

    Parameters
    ----------
    path
        Target path.
    content
        JSON-serializable mapping.
    overwrite
        Whether to overwrite if file exists.

    Usage example
    -------------
        _write_json(Path("x.json"), {"a": 1}, overwrite=True)
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"Sidecar already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(content), f, indent=2, ensure_ascii=False)
        f.write("\n")
