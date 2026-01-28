# =============================================================================
#                           DCAP: BIDS metadata
# =============================================================================
# > Construct BIDSPath and sidecar metadata (participants, scans, *_ieeg.json, etc.).
# > Start minimal; expand as you formalize your dataset.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json

import mne
from mne_bids import BIDSPath

from dcap_bids.heuristics import SourceItem
from dcap_bids.logging import get_logger

LOGGER = get_logger(__name__)


def _normalize_bids_label(value: Optional[str], prefix: str) -> Optional[str]:
    """
    Normalize a label like '01' or 'ses-01' into plain '01' (BIDSPath adds prefixes).

    Usage example
        _normalize_bids_label("ses-01", "ses")  # -> "01"
    """
    if value is None:
        return None
    value_str = str(value)
    if value_str.startswith(prefix + "-"):
        return value_str.split("-", 1)[1]
    return value_str


def infer_subject_session_run(
    subject: str,
    session: Optional[str],
    run: Optional[str],
    item: SourceItem,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Infer subject/session/run from CLI args plus optional hints on the SourceItem.

    Parameters
    ----------
    subject
        Subject label from CLI.
    session
        Session label from CLI.
    run
        Run label from CLI.
    item
        Source item which may contain hints.

    Returns
    -------
    subject, session, run
        Normalized labels.

    Usage example
        subject, session, run = infer_subject_session_run("001", "01", None, item)
    """
    subject_norm = _normalize_bids_label(subject, "sub") or item.subject_hint
    session_norm = _normalize_bids_label(session, "ses") if session is not None else item.session_hint
    run_norm = _normalize_bids_label(run, "run") if run is not None else item.run_hint

    if subject_norm is None:
        raise ValueError("Could not infer subject label (provide --subject or implement hints).")

    return subject_norm, session_norm, run_norm


def build_bids_path(
    bids_root: Path,
    subject: str,
    session: Optional[str],
    task: str,
    datatype: str,
    run: Optional[str],
) -> BIDSPath:
    """
    Build a BIDSPath for writing.

    Usage example
        bids_path = build_bids_path(Path("./bids"), "001", "01", "conversation", "ieeg", "01")
    """
    return BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype=datatype,
    )


@dataclass
class BidsSidecars:
    """
    Bundle of sidecar metadata writers.

    Notes
    -----
    Keep sensitive identifiers OUT of BIDS unless you have a deliberate de-ID strategy.

    Usage example
        sidecars = make_bids_sidecars(item, raw, task="conversation", datatype="ieeg")
        sidecars.write_all(bids_path)
    """

    modality_json: Dict[str, Any]

    def write_all(self, bids_path: BIDSPath) -> None:
        """
        Write sidecar files next to the data in the BIDS tree.

        Usage example
            sidecars.write_all(bids_path)
        """
        bids_path.mkdir()

        # Decide which JSON filename to use based on datatype.
        # For iEEG: *_ieeg.json
        # For EEG:  *_eeg.json
        # For MEG:  *_meg.json
        json_path = bids_path.copy().update(extension=".json")
        LOGGER.info("Writing sidecar JSON: %s", json_path.fpath)

        with open(json_path.fpath, "w", encoding="utf-8") as f:
            json.dump(self.modality_json, f, indent=2, ensure_ascii=False)


def make_bids_sidecars(item: SourceItem, raw: mne.io.BaseRaw, task: str, datatype: str) -> BidsSidecars:
    """
    Create minimal sidecar JSON content.

    Parameters
    ----------
    item
        Source item being converted.
    raw
        Loaded MNE Raw.
    task
        BIDS task label.
    datatype
        BIDS datatype: 'ieeg', 'eeg', or 'meg'.

    Returns
    -------
    BidsSidecars
        Sidecar bundle.

    Usage example
        sidecars = make_bids_sidecars(item, raw, task="conversation", datatype="ieeg")
    """
    # Very minimal, safe defaults. Expand with your acquisition specifics.
    modality_json: Dict[str, Any] = {
        "TaskName": task,
        "PowerLineFrequency": raw.info.get("line_freq", None),
        "SamplingFrequency": float(raw.info["sfreq"]),
        "SoftwareFilters": "n/a",
        "Manufacturer": "n/a",
        "ManufacturersModelName": "n/a",
        "iEEGReference": "n/a" if datatype == "ieeg" else None,
        "SourceFile": str(item.source_path.name),
    }

    # Clean out None values (nice for eeg/meg).
    modality_json = {k: v for k, v in modality_json.items() if v is not None}

    return BidsSidecars(modality_json=modality_json)
