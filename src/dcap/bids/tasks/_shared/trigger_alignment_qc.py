# =============================================================================
#                     ########################################
#                     #     TRIGGER ALIGNMENT QC ARTIFACT    #
#                     ########################################
# =============================================================================
"""Write trigger-alignment QC JSON artifacts to a BIDS derivatives dataset."""

import json
from pathlib import Path
from typing import Any, Mapping, Optional


def ensure_dcap_qc_dataset_description(*, bids_root: Path, dcap_version: str) -> None:
    """Ensure derivatives/dcap-qc/dataset_description.json exists."""
    out_dir = bids_root / "derivatives" / "dcap-qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    desc_path = out_dir / "dataset_description.json"
    if desc_path.exists():
        return

    payload: dict[str, Any] = {
        "Name": "DCAP QC",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "dcap",
                "Version": dcap_version,
                "Description": "QC artifacts generated during DCAP BIDS conversion",
            }
        ],
    }
    _write_json(desc_path, payload)


def write_trigger_alignment_qc_json(
    *,
    bids_root: Path,
    subject: str,
    session: Optional[str],
    datatype: str,
    filename_stem: str,
    payload: Mapping[str, Any],
    dcap_version: str,
) -> Path:
    """Write a trigger-alignment QC JSON to derivatives/dcap-qc and return the path.

    Parameters
    ----------
    filename_stem
        The BIDS filename without extension, e.g. "sub-003_task-diapix_run-01_ieeg".
        The output becomes "<stem>_desc-triggeralign_qc.json".
    """
    ensure_dcap_qc_dataset_description(bids_root=bids_root, dcap_version=dcap_version)

    out_dir = bids_root / "derivatives" / "dcap-qc" / subject
    if session is not None:
        out_dir = out_dir / session
    out_dir = out_dir / datatype
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{filename_stem}_desc-triggeralign_qc.json"
    _write_json(out_path, dict(payload))
    return out_path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        # Numpy arrays / scalars
        try:
            import numpy as np  # local import to avoid hard dependency at import time

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        except Exception:
            pass

        # Path objects
        if isinstance(obj, Path):
            return str(obj)

        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_default)
        f.write("\n")

