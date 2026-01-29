# src/dcap/bids/core/converter.py
# =============================================================================
#                       BIDS Core: Task-agnostic converter
# =============================================================================
#
# This module implements the task-agnostic conversion engine:
# - iterate over task-discovered recording units
# - construct BIDSPath
# - call task hooks to load raw and prepare events
# - write raw to BIDS via MNE-BIDS
#
# Absolutely no task-specific logic belongs here.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

from dataclasses import dataclass
from typing import Optional, Sequence

import mne
import numpy as np
from mne_bids import BIDSPath, write_raw_bids

from dcap.bids.core.bids_paths import build_bids_path, normalize_bids_label
from dcap.bids.core.config import BidsCoreConfig
from dcap.bids.core.transforms import apply_line_frequency
from dcap.bids.tasks.base import BidsTask, PreparedEvents, RecordingUnit


# =============================================================================
# Public result types
# =============================================================================

@dataclass(frozen=True)
class ConvertedItem:
    """
    Result record for one converted recording unit.

    Parameters
    ----------
    unit
        The input recording unit.
    bids_path
        BIDSPath used for writing.
    wrote_files
        Whether this unit was actually written (False when dry_run=True).

    Usage example
    -------------
        result = ConvertedItem(unit=unit, bids_path=bids_path, wrote_files=True)
    """

    unit: RecordingUnit
    bids_path: BIDSPath
    wrote_files: bool


# =============================================================================
# Core conversion engine
# =============================================================================

def convert_subject(
    cfg: BidsCoreConfig,
    task: BidsTask,
) -> list[ConvertedItem]:
    """
    Convert all recording units discovered by a task into BIDS.

    Parameters
    ----------
    cfg
        Task-agnostic core configuration.
    task
        Task implementation (discovery, loading, event policy, post-write hooks).

    Returns
    -------
    list[ConvertedItem]
        A list of per-unit conversion results.

    Usage example
    -------------
        task_impl = get_task("diapix")
        results = convert_subject(cfg, task_impl)
        for r in results:
            print(r.bids_path)
    """
    cfg.bids_root.mkdir(parents=True, exist_ok=True)

    units: Sequence[RecordingUnit] = task.discover(cfg.source_root)

    subject_norm = normalize_bids_label(cfg.subject, "sub")
    if subject_norm is None:
        raise ValueError("cfg.subject must be provided.")

    session_norm = normalize_bids_label(cfg.session, "ses") if cfg.session is not None else None

    results: list[ConvertedItem] = []

    for unit in units:
        bids_path = build_bids_path(
            bids_root=cfg.bids_root,
            subject=subject_norm,
            session=session_norm,
            task=task.name,
            datatype=cfg.datatype,
            run=unit.run,
        )

        raw = task.load_raw(unit=unit, preload=cfg.preload_raw)
        _validate_raw(raw, bids_path=bids_path)

        apply_line_frequency(raw, line_freq_hz=cfg.line_freq)

        prepared = task.prepare_events(raw=raw, unit=unit, bids_path=bids_path)
        events, event_id = _sanitize_prepared_events(prepared)

        # MNE-BIDS may raise if annotations extend outside data range; tasks can
        # preserve annotations if they want by writing a compatible events.tsv,
        # but the safe default here is to clear them.
        raw.set_annotations(None)

        wrote_files = False
        if not cfg.dry_run:
            write_raw_bids(
                raw=raw,
                bids_path=bids_path,
                events=events,
                event_id=event_id,
                overwrite=cfg.overwrite,
                format="auto",
                allow_preload=cfg.preload_raw,
                anonymize=None,
                verbose=False,
            )
            wrote_files = True

        task.post_write(unit=unit, bids_path=bids_path)

        results.append(ConvertedItem(unit=unit, bids_path=bids_path, wrote_files=wrote_files))

    return results


# =============================================================================
# Internal helpers
# =============================================================================

def _validate_raw(raw: mne.io.BaseRaw, bids_path: BIDSPath) -> None:
    """
    Basic raw sanity checks.

    Parameters
    ----------
    raw
        Raw object loaded by the task.
    bids_path
        BIDSPath being targeted.

    Usage example
    -------------
        _validate_raw(raw, bids_path)
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"Task loader returned unexpected type for {bids_path}: {type(raw)}")

    if raw.info.get("sfreq", 0.0) <= 0:
        raise ValueError(f"Raw has invalid sampling rate for {bids_path}: {raw.info.get('sfreq')}")

    if raw.n_times <= 0:
        raise ValueError(f"Raw has no samples for {bids_path}.")


def _sanitize_prepared_events(prepared: PreparedEvents) -> tuple[Optional[np.ndarray], Optional[dict[str, int]]]:
    """
    Sanitize task-provided events for MNE-BIDS.

    Parameters
    ----------
    prepared
        Task-produced events container.

    Returns
    -------
    events, event_id
        Either both None, or both valid objects.

    Notes
    -----
    MNE-BIDS expects that if `events` is provided, `event_id` is also provided.

    Usage example
    -------------
        events, event_id = _sanitize_prepared_events(prepared)
    """
    if prepared.events is None and prepared.event_id is None:
        return None, None

    if prepared.events is None or prepared.event_id is None:
        raise ValueError("Task returned only one of (events, event_id); must return both or neither.")

    events = np.asarray(prepared.events, dtype=int)
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(f"events must have shape (n_events, 3), got {events.shape}")

    # event_id keys should be strings, values ints
    event_id: dict[str, int] = {}
    for key, value in prepared.event_id.items():
        event_id[str(key)] = int(value)

    return events, event_id
