# =============================================================================
#                      Registry: metadata validation (library)
# =============================================================================
#
# Registry-scoped validation for:
# - registry_public.tsv (shareable)
# - $DCAP_PRIVATE_ROOT/subject_keys.yaml (private)
# - $DCAP_PRIVATE_ROOT/subjects/sub-*.yaml (private)
# - $DCAP_PRIVATE_ROOT/registry_private.tsv (private, optional)
#
# This module intentionally avoids building a generic validation framework.
# It implements the minimum set of checks needed to keep the metadata layer:
# - well-formed
# - internally consistent
# - privacy-safe by construction
#
# REVIEW
# =============================================================================

from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import re
import csv
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import yaml


# =============================================================================
# Constants
# =============================================================================

IssueLevel = Literal["error", "warning"]

PUBLIC_REGISTRY_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "dataset_id",
    "subject",
    "dcap_id",
    "session",
    "acquisition_id",
    "protocol_id",
    "task",
    "record_id",
)

PRIVATE_REGISTRY_COLUMNS: Tuple[str, ...] = (
    "record_id",
    "dcap_id",
    "exclude_reason",
    "review_date",
    "notes",
)

SUBJECT_PATTERN = re.compile(r"^sub-\d{3}$")
SESSION_PATTERN = re.compile(r"^ses-[A-Za-z0-9]+$")

# Note: Keep this list intentionally small; we only guard obvious privacy leaks.
PUBLIC_FORBIDDEN_COLUMN_HINTS: Tuple[str, ...] = (
    "name",
    "date_of_birth",
    "dob",
    "local_subject_id",
    "mrn",
    "medication",
    "notes_private",
)


# =============================================================================
# Data model
# =============================================================================

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """
    A single validation issue.

    Usage example
    -------------
        issue = ValidationIssue(
            level="error",
            location="subjects/sub-001.yaml:identity.date_of_birth",
            message="invalid ISO date (YYYY-MM-DD)",
        )
    """

    level: IssueLevel
    location: str
    message: str


# =============================================================================
# Public API
# =============================================================================

def resolve_private_root(private_root: str) -> Optional[Path]:
    """
    Resolve private root from CLI-style argument.

    Parameters
    ----------
    private_root
        One of:
        - 'env'  -> read DCAP_PRIVATE_ROOT
        - 'none' -> skip private checks
        - path   -> explicit directory

    Returns
    -------
    pathlib.Path | None
        Private root directory path, or None if skipping or unavailable.

    Usage example
    -------------
        root = resolve_private_root("env")
    """
    raw = private_root.strip()
    if raw.lower() in {"none", "null", "skip"}:
        return None
    if raw.lower() == "env":
        env_val = os.environ.get("DCAP_PRIVATE_ROOT", "").strip()
        return Path(env_val).expanduser().resolve() if env_val else None
    return Path(raw).expanduser().resolve()


def validate_registry(
    *,
    public_registry: Path,
    private_root: Optional[Path],
    strict: bool,
) -> int:
    """
    Validate registry + private metadata files and print a report.

    Parameters
    ----------
    public_registry
        Path to registry_public.tsv.
    private_root
        Private root path. If None, private checks are skipped.
    strict
        If True, warnings cause non-zero exit.

    Returns
    -------
    int
        Exit code:
        - 0: OK (warnings allowed unless strict=True)
        - 2: Errors present
        - 3: Warnings present and strict=True

    Usage example
    -------------
        exit_code = validate_registry(
            public_registry=Path("registry_public.tsv"),
            private_root=Path("~/.dcap_private").expanduser(),
            strict=False,
        )
    """
    all_issues: List[ValidationIssue] = []

    public_issues = _validate_registry_public_tsv(public_registry)
    _print_report(f"Public registry: {public_registry}", public_issues)
    all_issues.extend(public_issues)

    private_issues: List[ValidationIssue] = []
    if private_root is None:
        print("\nPrivate metadata: skipped")
    else:
        if not private_root.exists():
            print(f"\nPrivate metadata root not found (skipping): {private_root}")
        else:
            private_issues.extend(_validate_private_root(private_root))
            _print_report(f"Private metadata root: {private_root}", private_issues)
            all_issues.extend(private_issues)

    n_errors = sum(1 for i in all_issues if i.level == "error")
    n_warnings = sum(1 for i in all_issues if i.level == "warning")

    print("\nSummary")
    print(f"  Errors:   {n_errors}")
    print(f"  Warnings: {n_warnings}")

    if n_errors > 0:
        return 2
    if strict and n_warnings > 0:
        return 3
    return 0


# =============================================================================
# Private root validation
# =============================================================================

def _validate_private_root(private_root: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # subject_keys.yaml
    subject_keys_path = private_root / "subject_keys.yaml"
    issues.extend(_validate_subject_keys_yaml(subject_keys_path))

    # subjects/sub-*.yaml
    subjects_dir = private_root / "subjects"
    issues.extend(_validate_subjects_dir(subjects_dir))

    # registry_private.tsv (optional)
    private_registry_path = private_root / "registry_private.tsv"
    if private_registry_path.exists():
        issues.extend(_validate_registry_private_tsv(private_registry_path))
    else:
        issues.append(
            ValidationIssue(
                level="warning",
                location=str(private_registry_path),
                message="file not found (optional) — OK if no run-level decisions yet",
            )
        )

    return issues


def _validate_subjects_dir(subjects_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    if not subjects_dir.exists():
        return [
            ValidationIssue(
                level="warning",
                location=str(subjects_dir),
                message="subjects directory not found",
            )
        ]

    subject_files = sorted(subjects_dir.glob("sub-*.yaml"))
    if not subject_files:
        return [
            ValidationIssue(
                level="warning",
                location=str(subjects_dir),
                message="no subject YAML files found (expected sub-*.yaml)",
            )
        ]

    for path in subject_files:
        issues.extend(_validate_subject_yaml(path))

    return issues


# =============================================================================
# File: subject_keys.yaml
# =============================================================================

def _validate_subject_keys_yaml(path: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    if not path.exists():
        return [ValidationIssue(level="warning", location=str(path), message="file not found")]

    data = _load_yaml(path, issues)
    if data is None:
        return issues

    if not isinstance(data, dict):
        issues.append(ValidationIssue(level="error", location=str(path), message="YAML must be a mapping (dict)"))
        return issues

    if "datasets" not in data:
        issues.append(ValidationIssue(level="error", location=f"{path.name}:datasets", message="missing required key"))
        return issues

    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        issues.append(
            ValidationIssue(level="error", location=f"{path.name}:datasets", message="datasets must be a mapping (dict)")
        )
        return issues

    for dataset_id, entries in datasets.items():
        loc_ds = f"{path.name}:datasets.{dataset_id}"
        if not isinstance(entries, list):
            issues.append(ValidationIssue(level="error", location=loc_ds, message="dataset entries must be a list"))
            continue

        seen: set[str] = set()

        for idx, entry in enumerate(entries):
            loc = f"{loc_ds}[{idx}]"
            if not isinstance(entry, dict):
                issues.append(ValidationIssue(level="error", location=loc, message="entry must be a mapping (dict)"))
                continue

            bids_subject = entry.get("bids_subject")
            dcap_id = entry.get("dcap_id")

            if not _is_non_empty_str(bids_subject):
                issues.append(ValidationIssue(level="error", location=f"{loc}.bids_subject", message="missing required field"))
            elif not SUBJECT_PATTERN.match(str(bids_subject)):
                issues.append(ValidationIssue(level="error", location=f"{loc}.bids_subject", message="must match sub-###"))

            if not _is_non_empty_str(dcap_id):
                issues.append(ValidationIssue(level="error", location=f"{loc}.dcap_id", message="missing required field"))

            if _is_non_empty_str(bids_subject):
                if str(bids_subject) in seen:
                    issues.append(
                        ValidationIssue(level="error", location=f"{loc}.bids_subject", message="duplicate bids_subject within dataset")
                    )
                seen.add(str(bids_subject))

            implant_date = entry.get("implant_date")
            if _is_non_empty_str(implant_date) and _parse_iso_date(str(implant_date)) is None:
                issues.append(ValidationIssue(level="error", location=f"{loc}.implant_date", message="invalid ISO date (YYYY-MM-DD)"))

    return issues


# =============================================================================
# File: subjects/sub-XXX.yaml
# =============================================================================

def _validate_subject_yaml(path: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    data = _load_yaml(path, issues)
    if data is None:
        return issues

    if not isinstance(data, dict):
        issues.append(ValidationIssue(level="error", location=str(path), message="YAML must be a mapping (dict)"))
        return issues

    # Required top-level keys
    subject = data.get("subject")
    dataset_id = data.get("dataset_id")

    if not _is_non_empty_str(subject):
        issues.append(ValidationIssue(level="error", location=f"{path.name}:subject", message="missing required key"))
    elif not SUBJECT_PATTERN.match(str(subject)):
        issues.append(ValidationIssue(level="error", location=f"{path.name}:subject", message="must match sub-###"))

    if not _is_non_empty_str(dataset_id):
        issues.append(ValidationIssue(level="error", location=f"{path.name}:dataset_id", message="missing required key"))

    # Filename sanity
    expected_name = f"{subject}.yaml" if _is_non_empty_str(subject) else None
    if expected_name is not None and path.name != expected_name:
        issues.append(
            ValidationIssue(
                level="warning",
                location=str(path),
                message=f"filename should match subject field: expected {expected_name}",
            )
        )

    # identity
    identity = data.get("identity")
    if identity is not None and not isinstance(identity, dict):
        issues.append(ValidationIssue(level="error", location=f"{path.name}:identity", message="must be a mapping (dict)"))
    if isinstance(identity, dict):
        dob = identity.get("date_of_birth")
        if _is_non_empty_str(dob) and _parse_iso_date(str(dob)) is None:
            issues.append(
                ValidationIssue(
                    level="error",
                    location=f"{path.name}:identity.date_of_birth",
                    message="invalid ISO date (YYYY-MM-DD)",
                )
            )

        sex = identity.get("sex")
        if _is_non_empty_str(sex):
            allowed_sex = {"male", "female", "other", "unknown"}
            if str(sex) not in allowed_sex:
                issues.append(
                    ValidationIssue(
                        level="error",
                        location=f"{path.name}:identity.sex",
                        message=f"must be one of {sorted(allowed_sex)}",
                    )
                )

        handedness = identity.get("handedness")
        if _is_non_empty_str(handedness):
            allowed_hand = {"right", "left", "unknown"}
            if str(handedness) not in allowed_hand:
                issues.append(
                    ValidationIssue(
                        level="error",
                        location=f"{path.name}:identity.handedness",
                        message=f"must be one of {sorted(allowed_hand)} (or empty)",
                    )
                )

    # acquisitions
    acquisitions = data.get("acquisitions")
    if acquisitions is not None and not isinstance(acquisitions, list):
        issues.append(ValidationIssue(level="error", location=f"{path.name}:acquisitions", message="must be a list"))
    if isinstance(acquisitions, list):
        seen_acq: set[str] = set()
        for i, item in enumerate(acquisitions):
            loc = f"{path.name}:acquisitions[{i}]"
            if not isinstance(item, dict):
                issues.append(ValidationIssue(level="error", location=loc, message="entry must be a mapping (dict)"))
                continue

            acq_id = item.get("acquisition_id")
            if not _is_non_empty_str(acq_id):
                issues.append(
                    ValidationIssue(
                        level="error",
                        location=f"{loc}.acquisition_id",
                        message="missing required field",
                    )
                )
            else:
                if str(acq_id) in seen_acq:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            location=f"{loc}.acquisition_id",
                            message="duplicate acquisition_id (must be unique per subject)",
                        )
                    )
                seen_acq.add(str(acq_id))

            date = item.get("date")
            if _is_non_empty_str(date) and _parse_iso_date(str(date)) is None:
                issues.append(
                    ValidationIssue(level="error", location=f"{loc}.date", message="invalid ISO date (YYYY-MM-DD)"))

            session = item.get("session")
            if not _is_non_empty_str(session):
                issues.append(
                    ValidationIssue(level="error", location=f"{loc}.session", message="missing required field"))
            elif not SESSION_PATTERN.match(str(session)):
                issues.append(
                    ValidationIssue(
                        level="warning",
                        location=f"{loc}.session",
                        message="session should look like ses-01 / ses-XYZ",
                    )
                )

            # medication (light checks) — PER ACQUISITION
            medication_scale = item.get("medication")
            if medication_scale is not None and medication_scale != "":
                try:
                    medication_int = int(medication_scale)
                except (TypeError, ValueError):
                    issues.append(
                        ValidationIssue(
                            level="error",
                            location=f"{loc}.medication",
                            message="medication must be an integer in [0, 1, 2, 3] (or empty)",
                        )
                    )
                else:
                    if medication_int not in {0, 1, 2, 3}:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                location=f"{loc}.medication",
                                message="medication must be in [0, 1, 2, 3]",
                            )
                        )

    return issues


# =============================================================================
# File: registry_private.tsv
# =============================================================================

def _validate_registry_private_tsv(path: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    rows, header = _read_tsv(path, issues)
    if rows is None or header is None:
        return issues

    if tuple(header) != PRIVATE_REGISTRY_COLUMNS:
        issues.append(
            ValidationIssue(
                level="error",
                location=f"{path.name}:header",
                message=f"header must be exactly: {list(PRIVATE_REGISTRY_COLUMNS)}",
            )
        )
        # Continue anyway to provide more feedback.

    seen_record_id: set[str] = set()

    for idx, row in enumerate(rows, start=2):
        loc_row = f"{path.name}:row={idx}"
        record_id = row.get("record_id", "")
        dcap_id = row.get("dcap_id", "")
        review_date = row.get("review_date", "")

        if not _is_non_empty_str(record_id):
            issues.append(ValidationIssue(level="error", location=f"{loc_row}:record_id", message="missing required value"))
        else:
            if record_id in seen_record_id:
                issues.append(ValidationIssue(level="error", location=f"{loc_row}:record_id", message="duplicate record_id"))
            seen_record_id.add(record_id)

        if not _is_non_empty_str(dcap_id):
            issues.append(ValidationIssue(level="error", location=f"{loc_row}:dcap_id", message="missing required value"))

        if _is_non_empty_str(review_date) and _parse_iso_date(review_date) is None:
            issues.append(ValidationIssue(level="error", location=f"{loc_row}:review_date", message="invalid ISO date (YYYY-MM-DD)"))

    return issues


# =============================================================================
# File: registry_public.tsv
# =============================================================================

def _validate_registry_public_tsv(path: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    rows, header = _read_tsv(path, issues)
    if rows is None or header is None:
        return issues

    missing = [c for c in PUBLIC_REGISTRY_REQUIRED_COLUMNS if c not in header]
    if missing:
        issues.append(
            ValidationIssue(
                level="error",
                location=f"{path.name}:header",
                message=f"missing required columns: {missing}",
            )
        )

    # Privacy guardrails (very shallow but catches obvious mistakes)
    for col in header:
        for hint in PUBLIC_FORBIDDEN_COLUMN_HINTS:
            if hint.lower() in col.lower():
                issues.append(
                    ValidationIssue(
                        level="warning",
                        location=f"{path.name}:header",
                        message=f"column name looks private/sensitive: {col!r}",
                    )
                )

    seen_record_id: set[str] = set()

    for idx, row in enumerate(rows, start=2):
        loc_row = f"{path.name}:row={idx}"

        record_id = row.get("record_id", "")
        dataset_id = row.get("dataset_id", "")
        subject = row.get("subject", "")
        session = row.get("session", "")
        acquisition_id = row.get("acquisition_id", "")
        protocol_id = row.get("protocol_id", "")
        task = row.get("task", "")

        if _is_non_empty_str(subject) and not SUBJECT_PATTERN.match(subject):
            issues.append(
                ValidationIssue(
                    level="error",
                    location=f"{loc_row}:subject",
                    message="must match sub-###",
                )
            )

        if _is_non_empty_str(session) and not SESSION_PATTERN.match(session):
            issues.append(
                ValidationIssue(
                    level="warning",
                    location=f"{loc_row}:session",
                    message="session should look like ses-01 / ses-XYZ",
                )
            )

        if not _is_non_empty_str(record_id):
            issues.append(
                ValidationIssue(
                    level="error",
                    location=f"{loc_row}:record_id",
                    message="missing required value",
                )
            )
            continue

        if record_id in seen_record_id:
            issues.append(
                ValidationIssue(
                    level="error",
                    location=f"{loc_row}:record_id",
                    message="duplicate record_id",
                )
            )
        else:
            seen_record_id.add(record_id)

        # Consistency check (only if the required parts exist)
        if all(_is_non_empty_str(x) for x in (dataset_id, subject, session, acquisition_id)):
            protocol_id_for_id = protocol_id if _is_non_empty_str(protocol_id) else "none"
            expected = f"{dataset_id}|{subject}|{session}|{acquisition_id}|{protocol_id_for_id}"
            if record_id != expected:
                issues.append(
                    ValidationIssue(
                        level="error",
                        location=f"{loc_row}:record_id",
                        message=f"record_id mismatch; expected {expected!r}",
                    )
                )

    return issues



# =============================================================================
# I/O helpers
# =============================================================================

def _load_yaml(path: Path, issues: List[ValidationIssue]) -> Optional[Any]:
    if not path.exists():
        issues.append(ValidationIssue(level="warning", location=str(path), message="file not found"))
        return None

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        issues.append(ValidationIssue(level="error", location=str(path), message=f"failed to parse YAML: {exc}"))
        return None


def _read_tsv(path: Path, issues: List[ValidationIssue]) -> Tuple[Optional[List[Dict[str, str]]], Optional[List[str]]]:
    if not path.exists():
        issues.append(ValidationIssue(level="error", location=str(path), message="file not found"))
        return None, None

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            header = list(reader.fieldnames) if reader.fieldnames is not None else []
            rows = []
            for row in reader:
                # Normalize None -> "" for safety
                rows.append({k: (v if v is not None else "") for k, v in row.items()})  # noqa
            return rows, header
    except Exception as exc:  # noqa: BLE001
        issues.append(ValidationIssue(level="error", location=str(path), message=f"failed to read TSV: {exc}"))
        return None, None


# =============================================================================
# Small utilities
# =============================================================================

def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _parse_iso_date(value: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _print_report(title: str, issues: Sequence[ValidationIssue]) -> None:
    print(f"\n{title}")
    if not issues:
        print("  OK")
        return
    for issue in issues:
        print(f"  [{issue.level.upper()}] {issue.location}: {issue.message}")
