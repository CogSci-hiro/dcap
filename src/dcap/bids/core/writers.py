# src/dcap/bids/core/writers.py
# =============================================================================
#                         BIDS Core: Writing utilities
# =============================================================================
#
# Thin, task-agnostic wrappers around MNE-BIDS writing.
#
# Goals
# -----
# - Centralize write_raw_bids defaults (overwrite/dry-run handling happens elsewhere)
# - Provide small helpers for common patterns (e.g., making sure directories exist)
# - Keep *policy* out: tasks decide events/sidecars, core decides mechanics
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne
import numpy as np
from mne_bids import BIDSPath, write_raw_bids


# =============================================================================
# Public config
# =============================================================================

@dataclass(frozen=True)
class RawBidsWriteOptions:
    """
    Options controlling how MNE-BIDS writes raw recordings.

    Parameters
    ----------
    overwrite
        Whether to overwrite existing outputs.
    allow_preload
        Whether MNE-BIDS may operate on preloaded data (should match how raw was loaded).
    format
        File format passed to MNE-BIDS. Use "auto" unless you have a strong reason.
        Examples: "auto", "EDF", "FIF".
    anonymize
        Anonymization dict passed to MNE-BIDS (or None). Keep as None unless you
        have a deliberate, audited de-ID policy.

    Usage example
    -------------
        opts = RawBidsWriteOptions(overwrite=True, allow_preload=True, format="auto", anonymize=None)
    """

    overwrite: bool
    allow_preload: bool
    format: str
    anonymize: Optional[dict]


# =============================================================================
# Public API
# =============================================================================

def write_raw_recording(
    raw: mne.io.BaseRaw,
    bids_path: BIDSPath,
    events: Optional[np.ndarray],
    event_id: Optional[dict[str, int]],
    options: RawBidsWriteOptions,
) -> None:
    """
    Write a raw recording to BIDS via MNE-BIDS.

    Parameters
    ----------
    raw
        MNE Raw object.
    bids_path
        Target BIDSPath (recording-level).
    events
        Optional MNE events array (n_events, 3) or None.
    event_id
        Optional mapping of event name -> integer code, or None.
    options
        Write options.

    Returns
    -------
    None

    Notes
    -----
    - This is intentionally thin. The core converter decides whether to call
      this (e.g., dry-run) and tasks decide what events/sidecars to provide.

    Usage example
    -------------
        opts = RawBidsWriteOptions(overwrite=True, allow_preload=True, format="auto", anonymize=None)
        write_raw_recording(raw, bids_path, events, event_id, opts)
    """
    _validate_events_pair(events=events, event_id=event_id)

    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        events=events,
        event_id=event_id,
        overwrite=options.overwrite,
        format=options.format,
        allow_preload=options.allow_preload,
        anonymize=options.anonymize,
        verbose=False,
    )


def ensure_bids_root(bids_root: Path) -> None:
    """
    Ensure the BIDS root directory exists.

    Usage example
    -------------
        ensure_bids_root(Path("bids"))
    """
    bids_root.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Internal helpers
# =============================================================================

def _validate_events_pair(events: Optional[np.ndarray], event_id: Optional[dict[str, int]]) -> None:
    """
    Validate that events and event_id are consistently provided.

    Usage example
    -------------
        _validate_events_pair(events=None, event_id=None)
    """
    if events is None and event_id is None:
        return

    if events is None or event_id is None:
        raise ValueError("Must provide both (events, event_id) or neither.")

    events_arr = np.asarray(events)
    if events_arr.ndim != 2 or events_arr.shape[1] != 3:
        raise ValueError(f"events must have shape (n_events, 3), got {events_arr.shape}")

    # Ensure int dtype for MNE-style events arrays
    if events_arr.dtype.kind not in {"i", "u"}:
        raise ValueError(f"events dtype must be integer, got {events_arr.dtype}")

    if len(event_id) == 0:
        raise ValueError("event_id must not be empty when events are provided.")
