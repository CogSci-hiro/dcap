# dcap/registry/build.py
# =============================================================================
#                     Registry: public registry builder
# =============================================================================

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import csv
import yaml


# =============================================================================
# Constants
# =============================================================================

PUBLIC_REGISTRY_COLUMNS: Tuple[str, ...] = (
    "dataset_id",
    "subject",
    "session",
    "acquisition_id",
    "protocol_id",
    "task",
    "age",
    "sex",
    "record_id",
)


# =============================================================================
# YAML helpers
# =============================================================================

def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping (dict): {path}")
    return data


def _write_tsv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _normalize_subject_keys(dataset_entries: Any) -> Dict[str, str]:
    """
    Expected by your validator:

    subject_keys.yaml:
      datasets:
        Timone2025:
          - bids_subject: sub-001
            dcap_id: D001
    """
    if not isinstance(dataset_entries, list):
        raise ValueError("subject_keys.yaml.datasets[dataset_id] must be a list of mappings")

    mapping: Dict[str, str] = {}
    for entry in dataset_entries:
        if not isinstance(entry, dict):
            raise ValueError("subject_keys entry must be a mapping (dict)")
        bids_subject = str(entry.get("bids_subject", "")).strip()
        dcap_id = str(entry.get("dcap_id", "")).strip()
        if not bids_subject or not dcap_id:
            raise ValueError(f"Invalid subject_keys entry: {entry}")
        mapping[bids_subject] = dcap_id
    return mapping


# =============================================================================
# Protocol matching
# =============================================================================

def _protocols_for_session(subject_data: Dict[str, Any], session: str) -> List[Dict[str, str]]:
    """
    Return protocol descriptors applicable to a given session.

    If none match, return empty list.
    """
    protocols = subject_data.get("protocols", [])
    if not isinstance(protocols, list):
        return []

    matched: List[Dict[str, str]] = []
    for proto in protocols:
        if not isinstance(proto, dict):
            continue
        sessions = proto.get("sessions", [])
        if not isinstance(sessions, list):
            continue
        if session in [str(s).strip() for s in sessions if isinstance(s, (str, int))]:
            protocol_id = str(proto.get("protocol_id", "")).strip()
            task = str(proto.get("task", "")).strip()
            matched.append({"protocol_id": protocol_id, "task": task})

    # Deterministic ordering
    matched.sort(key=lambda d: (d.get("protocol_id", ""), d.get("task", "")))
    return matched


# =============================================================================
# Date helpers
# =============================================================================


def _parse_iso_date(value: str) -> Optional[dt.date]:
    value = value.strip()
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _compute_age_years(*, date_of_birth: Optional[dt.date], reference_date: Optional[dt.date]) -> str:
    if date_of_birth is None or reference_date is None:
        return ""
    if reference_date < date_of_birth:
        return ""
    years = reference_date.year - date_of_birth.year
    if (reference_date.month, reference_date.day) < (date_of_birth.month, date_of_birth.day):
        years -= 1
    return str(years)


# =============================================================================
# Public builder
# =============================================================================

def build_public_registry(
    *,
    public_registry_out: Path,
    private_root: Path,
    dataset_id: str,
) -> Path:
    """
    Build a sanitized public registry TSV *from YAML only*.

    Output columns (exact):
      dataset_id, subject, dcap_id, session, acquisition_id, protocol_id, task, record_id

    record_id:
      f"{dataset_id}|{subject}|{session}|{acquisition_id}"
    """
    private_root = private_root.expanduser().resolve()

    subject_keys_path = private_root / "subject_keys.yaml"
    subjects_dir = private_root / "subjects"

    subject_keys = _read_yaml(subject_keys_path)
    datasets_block = subject_keys.get("datasets")
    if not isinstance(datasets_block, dict):
        raise ValueError("subject_keys.yaml must contain a top-level 'datasets' mapping.")

    if dataset_id not in datasets_block:
        available = sorted(datasets_block.keys())
        raise ValueError(
            f"dataset_id={dataset_id!r} not found in subject_keys.yaml.datasets. Available: {available}"
        )

    subject_to_dcap = _normalize_subject_keys(datasets_block[dataset_id])

    rows: List[Dict[str, Any]] = []

    for bids_subject, dcap_id in sorted(subject_to_dcap.items()):
        subject_file = subjects_dir / f"{bids_subject}.yaml"
        if not subject_file.exists():
            raise FileNotFoundError(f"Missing private subject file: {subject_file}")

        subject_data = _read_yaml(subject_file)

        subject_in_file = str(subject_data.get("subject", "")).strip()
        dataset_in_file = str(subject_data.get("dataset_id", "")).strip()

        if subject_in_file and subject_in_file != bids_subject:
            raise ValueError(
                f"Subject mismatch in {subject_file.name}: subject={subject_in_file!r} expected {bids_subject!r}"
            )
        if dataset_in_file and dataset_in_file != dataset_id:
            raise ValueError(
                f"Dataset mismatch in {subject_file.name}: dataset_id={dataset_in_file!r} building {dataset_id!r}"
            )

        acquisitions = subject_data.get("acquisitions", [])
        if acquisitions is None:
            acquisitions = []
        if not isinstance(acquisitions, list):
            raise ValueError(f"Expected 'acquisitions' list in {subject_file}")

        for acq in acquisitions:
            if not isinstance(acq, dict):
                continue

            acquisition_id = str(acq.get("acquisition_id", "")).strip()
            session = str(acq.get("session", "")).strip()

            # ---------------------------------------------------------------------
            # Public demographics (derived, non-identifying)
            # ---------------------------------------------------------------------
            identity = subject_data.get("identity", {})
            sex = ""
            dob = None

            if isinstance(identity, dict):
                sex = str(identity.get("sex", "")).strip()
                dob_raw = str(identity.get("date_of_birth", "")).strip()
                dob = _parse_iso_date(dob_raw)

            acq_date = _parse_iso_date(str(acq.get("date", "")).strip())
            age_years = _compute_age_years(
                date_of_birth=dob,
                reference_date=acq_date,
            )

            if not acquisition_id or not session:
                raise ValueError(
                    f"Acquisition missing acquisition_id/session in {subject_file.name}: {acq}"
                )

            matched_protocols = _protocols_for_session(subject_data, session)

            # If protocols match this session, emit one row per protocol (deterministic).
            # If none match, still emit a row (protocol_id/task empty) — columns exist either way.
            if matched_protocols:
                for mp in matched_protocols:
                    protocol_id = mp.get("protocol_id", "")
                    task = mp.get("task", "")
                    record_id = f"{dataset_id}|{bids_subject}|{session}|{acquisition_id}|{protocol_id}"

                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "subject": bids_subject,
                            "session": session,
                            "acquisition_id": acquisition_id,
                            "protocol_id": protocol_id,
                            "task": task,
                            "sex": sex,
                            "age": age_years,
                            "record_id": record_id,
                        }
                    )
            else:
                protocol_id_for_id = protocol_id if protocol_id else "none"
                record_id = f"{dataset_id}|{bids_subject}|{session}|{acquisition_id}|{protocol_id_for_id}"
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "subject": bids_subject,
                        "session": session,
                        "acquisition_id": acquisition_id,
                        "protocol_id": "",
                        "task": "",
                        "record_id": record_id,
                    }
                )

    _write_tsv(public_registry_out, rows=rows, fieldnames=PUBLIC_REGISTRY_COLUMNS)
    return public_registry_out
