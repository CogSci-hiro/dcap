# =============================================================================
#                    BIDS Tasks: Subject re-identification map
# =============================================================================
#
# WARNING:
# - Contains sensitive identifiers.
# - DO NOT commit mapping files to Git.
# - Intended to be loaded from $DCAP_PRIVATE_ROOT.
#
# This module provides parsing + lookup only.
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


# =============================================================================
# Models
# =============================================================================

@dataclass(frozen=True, slots=True)
class SubjectMapEntry:
    """
    One subject mapping entry.

    Usage example
    -------------
        e = SubjectMapEntry(
            bids_subject="sub-001",
            dcap_id="Nic-Ele",
            site="Timone",
            implant_date="",
            notes="child",
        )
    """

    bids_subject: str
    dcap_id: str
    site: str
    implant_date: str
    notes: str


@dataclass(frozen=True, slots=True)
class SubjectReidMap:
    """
    Parsed subject re-identification mapping.

    Usage example
    -------------
        mapping = load_subject_reid_map(Path("/private/subject_map.yml"))
        dcap_id = mapping.resolve_dcap_id(dataset_id="Timone2025", bids_subject="sub-001")
    """

    version: int
    notes: str
    datasets: Mapping[str, tuple[SubjectMapEntry, ...]]

    def resolve_dcap_id(self, *, dataset_id: str, bids_subject: str) -> str:
        """
        Resolve BIDS subject to private clinical identifier.

        Parameters
        ----------
        dataset_id
            Dataset key in YAML under `datasets:`.
        bids_subject
            BIDS subject label. Accepts "sub-001" or "001".

        Returns
        -------
        str
            dcap_id / clinical identifier string.

        Usage example
        -------------
            dcap_id = mapping.resolve_dcap_id(dataset_id="Timone2025", bids_subject="sub-003")
        """
        dataset_key = str(dataset_id).strip()
        subject_key = _normalize_bids_subject(bids_subject)

        if dataset_key not in self.datasets:
            available = ", ".join(sorted(self.datasets.keys()))
            raise KeyError(f"Unknown dataset_id={dataset_key!r}. Available: {available}")

        for entry in self.datasets[dataset_key]:
            if _normalize_bids_subject(entry.bids_subject) == subject_key:
                return entry.dcap_id

        raise KeyError(f"No mapping found for dataset_id={dataset_key!r}, bids_subject={subject_key!r}.")


# =============================================================================
# Public API
# =============================================================================

def load_subject_reid_map(yaml_path: Path) -> SubjectReidMap:
    """
    Load a private subject re-identification map YAML file.

    Parameters
    ----------
    yaml_path
        Path to the YAML file.

    Returns
    -------
    SubjectReidMap
        Parsed mapping object.

    Usage example
    -------------
        mapping = load_subject_reid_map(Path("/private/subject_reid.yml"))
    """
    yaml_path = Path(yaml_path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Subject map YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError("Subject map YAML must parse to a dict at top level.")

    version = int(payload.get("version", 0))
    if version != 1:
        raise ValueError(f"Unsupported subject map version={version}. Expected 1.")

    notes = str(payload.get("notes", ""))

    datasets_raw = payload.get("datasets", None)
    if not isinstance(datasets_raw, dict):
        raise ValueError("Subject map YAML must contain a dict field 'datasets'.")

    datasets: Dict[str, tuple[SubjectMapEntry, ...]] = {}
    for dataset_id, entries_raw in datasets_raw.items():
        if not isinstance(entries_raw, list):
            raise ValueError(f"datasets.{dataset_id} must be a list of entries.")

        entries: list[SubjectMapEntry] = []
        for row in entries_raw:
            if not isinstance(row, dict):
                raise ValueError(f"datasets.{dataset_id} contains a non-dict entry: {row!r}")

            entries.append(
                SubjectMapEntry(
                    bids_subject=str(row.get("bids_subject", "")).strip(),
                    dcap_id=str(row.get("dcap_id", "")).strip(),
                    site=str(row.get("site", "")).strip(),
                    implant_date=str(row.get("implant_date", "")).strip(),
                    notes=str(row.get("notes", "")).strip(),
                )
            )

        datasets[str(dataset_id).strip()] = tuple(entries)

    return SubjectReidMap(version=version, notes=notes, datasets=datasets)


def _normalize_bids_subject(bids_subject: str) -> str:
    s = str(bids_subject).strip()
    if s.startswith("sub-"):
        return s
    return f"sub-{s}"
