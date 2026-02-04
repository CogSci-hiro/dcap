# =============================================================================
#                     ########################################
#                     #        BIDS CORE: SUBJECT MAP        #
#                     ########################################
# =============================================================================
"""Load subject mapping YAML used to avoid manual ID mixups in CLI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class SubjectMappingEntry:
    dataset_id: str
    bids_subject: str  # bare, e.g. "001"
    original_id: str   # FreeSurfer folder name


def load_subject_mapping_entry(
    *,
    mapping_yaml: Path,
    dataset_id: str,
    bids_subject: str,
) -> SubjectMappingEntry:
    """
    Resolve a subject entry from the mapping YAML.

    Parameters
    ----------
    mapping_yaml
        Path to mapping YAML file.
    dataset_id
        Dataset identifier (e.g., "Timone2025").
    bids_subject
        BIDS subject label (either "sub-001" or "001").

    Returns
    -------
    SubjectMappingEntry

    Raises
    ------
    FileNotFoundError
        If mapping file missing.
    ValueError
        If dataset/version/subject not found or invalid.

    Usage example
    -------------
        entry = load_subject_mapping_entry(
            mapping_yaml=Path("subject_map.yaml"),
            dataset_id="Timone2025",
            bids_subject="sub-001",
        )
        # entry.original_id -> FreeSurfer folder name
    """
    mapping_yaml = Path(mapping_yaml).expanduser().resolve()
    if not mapping_yaml.exists():
        raise FileNotFoundError(f"Mapping YAML not found: {mapping_yaml}")

    payload = _load_yaml(mapping_yaml)
    version = payload.get("version", None)
    if version != 1:
        raise ValueError(f"Unsupported mapping YAML version: {version!r} (expected 1).")

    datasets = payload.get("datasets", None)
    if not isinstance(datasets, dict):
        raise ValueError("Expected top-level key 'datasets' to be a mapping.")

    ds = datasets.get(dataset_id, None)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id!r} not found in mapping YAML.")

    # Support both:
    # datasets: { Timone2025: { subjects: [ ... ] } }
    # and:
    # datasets: { Timone2025: [ ... ] }
    subjects = None
    if isinstance(ds, dict):
        subjects = ds.get("subjects", None)
        if subjects is None:
            # allow list-less dict only if it contains direct list under dataset key (rare)
            pass
    if subjects is None:
        subjects = ds
    if not isinstance(subjects, list):
        raise ValueError(f"Expected datasets.{dataset_id} subjects to be a list.")

    bids_subject_bare = _strip_prefix(bids_subject, "sub")

    for row in subjects:
        if not isinstance(row, dict):
            continue
        row_bids = _strip_prefix(str(row.get("bids_subject", "")).strip(), "sub")
        if row_bids == bids_subject_bare:
            # You called it dcap_id, but AnatWriteConfig expects original_id (FS folder).
            original_id = str(row.get("dcap_id", "")).strip()
            if not original_id:
                raise ValueError(
                    f"Entry found for {dataset_id}/{bids_subject}, but 'dcap_id' is empty."
                )
            return SubjectMappingEntry(
                dataset_id=dataset_id,
                bids_subject=bids_subject_bare,
                original_id=original_id,
            )

    raise ValueError(f"No entry for dataset={dataset_id!r}, bids_subject={bids_subject!r} in mapping YAML.")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML top-level mapping, got {type(obj)}.")
    return obj


def _strip_prefix(value: str, prefix: str) -> str:
    s = str(value).strip()
    token = f"{prefix}-"
    return s[len(token):] if s.startswith(token) else s
