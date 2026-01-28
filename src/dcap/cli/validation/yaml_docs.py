# =============================================================================
#                  Validation: YAML document validation engine
# =============================================================================
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from dcap.validation.errors import ValidationIssue
from dcap.validation.utils import is_non_empty_string, matches_pattern, parse_iso_date


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_subject_keys_yaml(
    yaml_path: Path,
    spec: Dict[str, Any],
) -> List[ValidationIssue]:
    """
    Validate a subject_keys.yaml file against the dcap schema spec.

    Parameters
    ----------
    yaml_path
        Path to subject_keys.yaml.
    spec
        Parsed schema dictionary.

    Returns
    -------
    list of ValidationIssue
        Issues found.

    Usage example
    -------------
        issues = validate_subject_keys_yaml(Path("subject_keys.yaml"), spec)
    """
    issues: List[ValidationIssue] = []

    if not yaml_path.exists():
        return [ValidationIssue(level="error", location=str(yaml_path), message="file not found")]

    try:
        data = _load_yaml(yaml_path)
    except Exception as exc:
        return [ValidationIssue(level="error", location=str(yaml_path), message=f"failed to parse YAML: {exc}")]

    if not isinstance(data, dict):
        return [ValidationIssue(level="error", location=str(yaml_path), message="YAML must be a mapping (dict)")]

    required_top = spec.get("required_top_level_keys", []) or []
    for key in required_top:
        if key not in data:
            issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:{key}", message="missing required top-level key"))

    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:datasets", message="datasets must be a mapping (dict)"))
        return issues

    entry_schema = spec.get("entry_schema", {}) or {}
    required_keys = entry_schema.get("required_keys", []) or []
    fields = (entry_schema.get("fields", {}) or {})

    bids_subject_pattern = (fields.get("bids_subject", {}) or {}).get("constraints", [])
    local_id_constraints = (fields.get("local_subject_id", {}) or {}).get("constraints", [])
    # pattern is stored under constraints list in our custom spec
    bids_pattern = None
    for c in bids_subject_pattern:
        if isinstance(c, dict) and "pattern" in c:
            bids_pattern = str(c["pattern"])
            break

    for dataset_id, entries in datasets.items():
        if not isinstance(entries, list):
            issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:datasets.{dataset_id}", message="dataset entries must be a list"))
            continue

        seen_subjects: set[str] = set()

        for idx, entry in enumerate(entries):
            loc = f"{yaml_path.name}:datasets.{dataset_id}[{idx}]"
            if not isinstance(entry, dict):
                issues.append(ValidationIssue(level="error", location=loc, message="entry must be a mapping (dict)"))
                continue

            for k in required_keys:
                if k not in entry or not is_non_empty_string(entry.get(k)):
                    issues.append(ValidationIssue(level="error", location=f"{loc}.{k}", message="missing required field"))

            bids_subject = entry.get("bids_subject")
            if bids_pattern and is_non_empty_string(bids_subject) and not matches_pattern(bids_subject, bids_pattern):
                issues.append(ValidationIssue(level="error", location=f"{loc}.bids_subject", message=f"must match pattern {bids_pattern!r}"))

            if is_non_empty_string(bids_subject):
                if bids_subject in seen_subjects:
                    issues.append(ValidationIssue(level="error", location=f"{loc}.bids_subject", message="duplicate bids_subject within dataset"))
                seen_subjects.add(bids_subject)

            implant_date = entry.get("implant_date")
            if is_non_empty_string(implant_date) and parse_iso_date(implant_date) is None:
                issues.append(ValidationIssue(level="error", location=f"{loc}.implant_date", message="invalid ISO date (YYYY-MM-DD)"))

    return issues


def validate_subject_yaml(
    yaml_path: Path,
    spec: Dict[str, Any],
) -> List[ValidationIssue]:
    """
    Validate a subject-level YAML (subjects/sub-XXX.yaml) against the dcap schema spec.

    Parameters
    ----------
    yaml_path
        Path to subject YAML.
    spec
        Parsed schema dictionary.

    Returns
    -------
    list of ValidationIssue
        Issues found.

    Usage example
    -------------
        issues = validate_subject_yaml(Path("subjects/sub-001.yaml"), spec)
    """
    issues: List[ValidationIssue] = []

    if not yaml_path.exists():
        return [ValidationIssue(level="error", location=str(yaml_path), message="file not found")]

    try:
        data = _load_yaml(yaml_path)
    except Exception as exc:
        return [ValidationIssue(level="error", location=str(yaml_path), message=f"failed to parse YAML: {exc}")]

    if not isinstance(data, dict):
        return [ValidationIssue(level="error", location=str(yaml_path), message="YAML must be a mapping (dict)")]

    required_top = spec.get("required_top_level_keys", []) or []
    for key in required_top:
        if key not in data:
            issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:{key}", message="missing required top-level key"))

    # subject pattern
    subject_constraints = (spec.get("constraints", {}) or {}).get("subject", {})
    subj_pattern = subject_constraints.get("pattern")
    subject = data.get("subject")
    if subj_pattern and is_non_empty_string(subject) and not matches_pattern(subject, str(subj_pattern)):
        issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:subject", message=f"must match pattern {subj_pattern!r}"))

    # Top-level keys allowed
    allowed = set(spec.get("required_top_level_keys", []) or []) | set(spec.get("optional_top_level_keys", []) or [])
    extra_top = sorted(set(data.keys()) - allowed)
    if extra_top:
        issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:header", message=f"unexpected top-level keys: {extra_top}"))

    sections = spec.get("sections", {}) or {}

    # identity
    identity_spec = sections.get("identity", {})
    identity = data.get("identity")
    if identity is not None and not isinstance(identity, dict):
        issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:identity", message="identity must be a mapping (dict)"))
    if isinstance(identity, dict):
        sex = identity.get("sex")
        allowed_sex = ((identity_spec.get("fields", {}) or {}).get("sex", {}) or {}).get("allowed", [])
        if is_non_empty_string(sex) and allowed_sex and sex not in set(allowed_sex):
            issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:identity.sex", message=f"must be one of {sorted(set(allowed_sex))}"))

        dob = identity.get("date_of_birth")
        if is_non_empty_string(dob) and parse_iso_date(dob) is None:
            issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:identity.date_of_birth", message="invalid ISO date (YYYY-MM-DD)"))

    # acquisitions
    acquisitions_spec = sections.get("acquisitions", {})
    acquisitions = data.get("acquisitions")
    if acquisitions is not None and not isinstance(acquisitions, list):
        issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:acquisitions", message="acquisitions must be a list"))
    if isinstance(acquisitions, list):
        seen_acq: set[str] = set()
        item_schema = (acquisitions_spec.get("item_schema", {}) or {})
        required = item_schema.get("required_keys", []) or []
        fields = item_schema.get("fields", {}) or {}
        session_pattern = ((fields.get("session", {}) or {}).get("pattern"))
        for idx, item in enumerate(acquisitions):
            loc = f"{yaml_path.name}:acquisitions[{idx}]"
            if not isinstance(item, dict):
                issues.append(ValidationIssue(level="error", location=loc, message="acquisition entry must be a mapping (dict)"))
                continue
            for k in required:
                if k not in item or not is_non_empty_string(item.get(k)):
                    issues.append(ValidationIssue(level="error", location=f"{loc}.{k}", message="missing required field"))
            acq_id = item.get("acquisition_id")
            if is_non_empty_string(acq_id):
                if acq_id in seen_acq:
                    issues.append(ValidationIssue(level="error", location=f"{loc}.acquisition_id", message="duplicate acquisition_id"))
                seen_acq.add(acq_id)
            date = item.get("date")
            if is_non_empty_string(date) and parse_iso_date(date) is None:
                issues.append(ValidationIssue(level="error", location=f"{loc}.date", message="invalid ISO date (YYYY-MM-DD)"))
            session = item.get("session")
            if session_pattern and is_non_empty_string(session) and not matches_pattern(session, str(session_pattern)):
                issues.append(ValidationIssue(level="warning", location=f"{loc}.session", message=f"should match pattern {session_pattern!r}"))

    # protocols
    protocols_spec = sections.get("protocols", {})
    protocols = data.get("protocols")
    if protocols is not None and not isinstance(protocols, list):
        issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:protocols", message="protocols must be a list"))
    if isinstance(protocols, list):
        seen_pid: set[str] = set()
        item_schema = (protocols_spec.get("item_schema", {}) or {})
        required = item_schema.get("required_keys", []) or []
        fields = item_schema.get("fields", {}) or {}
        sessions_item_pattern = (((fields.get("sessions", {}) or {}).get("item_schema", {}) or {}).get("pattern"))
        for idx, item in enumerate(protocols):
            loc = f"{yaml_path.name}:protocols[{idx}]"
            if not isinstance(item, dict):
                issues.append(ValidationIssue(level="error", location=loc, message="protocol entry must be a mapping (dict)"))
                continue
            for k in required:
                if k not in item:
                    issues.append(ValidationIssue(level="error", location=f"{loc}.{k}", message="missing required field"))
            pid = item.get("protocol_id")
            if is_non_empty_string(pid):
                if pid in seen_pid:
                    issues.append(ValidationIssue(level="error", location=f"{loc}.protocol_id", message="duplicate protocol_id"))
                seen_pid.add(pid)

            sessions = item.get("sessions")
            if not isinstance(sessions, list) or len(sessions) == 0:
                issues.append(ValidationIssue(level="error", location=f"{loc}.sessions", message="sessions must be a non-empty list"))
            elif sessions_item_pattern:
                for j, s in enumerate(sessions):
                    if is_non_empty_string(s) and not matches_pattern(s, str(sessions_item_pattern)):
                        issues.append(ValidationIssue(level="warning", location=f"{loc}.sessions[{j}]", message=f"should match pattern {sessions_item_pattern!r}"))

    # medication (light checks)
    medication_spec = sections.get("medication", {})
    medication = data.get("medication")
    if medication is not None and not isinstance(medication, list):
        issues.append(ValidationIssue(level="error", location=f"{yaml_path.name}:medication", message="medication must be a list"))
    if isinstance(medication, list):
        item_schema = (medication_spec.get("item_schema", {}) or {})
        required = item_schema.get("required_keys", []) or []
        for idx, item in enumerate(medication):
            loc = f"{yaml_path.name}:medication[{idx}]"
            if not isinstance(item, dict):
                issues.append(ValidationIssue(level="error", location=loc, message="medication entry must be a mapping (dict)"))
                continue
            for k in required:
                if k not in item:
                    issues.append(ValidationIssue(level="error", location=f"{loc}.{k}", message="missing required field"))
            start = item.get("start_date")
            if is_non_empty_string(start) and parse_iso_date(start) is None:
                issues.append(ValidationIssue(level="error", location=f"{loc}.start_date", message="invalid ISO date (YYYY-MM-DD)"))
            end = item.get("end_date")
            if is_non_empty_string(end) and parse_iso_date(end) is None:
                issues.append(ValidationIssue(level="error", location=f"{loc}.end_date", message="invalid ISO date (YYYY-MM-DD)"))

    return issues
