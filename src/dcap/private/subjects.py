# =============================================================================
#                         Private: subject YAML format
# =============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re
import datetime as _dt

import yaml


_SUBJECT_RE = re.compile(r"^sub-\d{3}$")
_SESSION_RE = re.compile(r"^ses-[A-Za-z0-9]+$")


@dataclass(frozen=True, slots=True)
class SubjectYamlValidationIssue:
    """
    Validation issue for a subject YAML file.

    Usage example
    -------------
        issue = SubjectYamlValidationIssue(
            level="error",
            path="protocols[0].sessions",
            message="sessions must be a non-empty list",
        )
    """

    level: str  # "error" | "warning"
    path: str
    message: str


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        _dt.date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _expect(condition: bool, issues: List[SubjectYamlValidationIssue], *, level: str, path: str, message: str) -> None:
    if not condition:
        issues.append(SubjectYamlValidationIssue(level=level, path=path, message=message))


def load_subject_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a subject YAML file.

    Parameters
    ----------
    path
        Path to YAML file.

    Returns
    -------
    dict
        Parsed YAML data.

    Usage example
    -------------
        data = load_subject_yaml(Path("~/.dcap_private/subjects/sub-001.yaml").expanduser())
    """
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Subject YAML must parse to a dict, got {type(data)} at {path}")
    return data


def validate_subject_yaml(data: Dict[str, Any]) -> List[SubjectYamlValidationIssue]:
    """
    Validate a subject YAML dict.

    This is intentionally lightweight:
    - strict on required keys and key fields
    - lenient on optional extra fields inside sections

    Parameters
    ----------
    data
        Parsed subject YAML dict.

    Returns
    -------
    list of SubjectYamlValidationIssue
        Validation issues (errors and warnings).

    Usage example
    -------------
        data = load_subject_yaml(Path("sub-001.yaml"))
        issues = validate_subject_yaml(data)
        errors = [i for i in issues if i.level == "error"]
    """
    issues: List[SubjectYamlValidationIssue] = []

    allowed_top = {
        "subject",
        "dataset_id",
        "identity",
        "implantation",
        "acquisitions",
        "medication",
        "protocols",
        "notes",
    }
    extra_top = sorted(set(data.keys()) - allowed_top)
    _expect(
        not extra_top,
        issues,
        level="error",
        path="",
        message=f"Unexpected top-level keys: {extra_top}. Allowed: {sorted(allowed_top)}",
    )

    subject = data.get("subject")
    _expect(isinstance(subject, str), issues, level="error", path="subject", message="subject must be a string like sub-001")
    if isinstance(subject, str):
        _expect(bool(_SUBJECT_RE.match(subject)), issues, level="error", path="subject", message="subject must match sub-XXX (3 digits)")

    dataset_id = data.get("dataset_id")
    _expect(isinstance(dataset_id, str) and len(dataset_id) > 0, issues, level="error", path="dataset_id", message="dataset_id must be a non-empty string")

    # identity
    identity = data.get("identity")
    if identity is not None:
        _expect(isinstance(identity, dict), issues, level="error", path="identity", message="identity must be a dict")
        if isinstance(identity, dict):
            sex = identity.get("sex")
            if sex is not None:
                _expect(
                    sex in {"male", "female", "other", "unknown"},
                    issues,
                    level="error",
                    path="identity.sex",
                    message="sex must be one of: male, female, other, unknown",
                )

            dob = identity.get("date_of_birth")
            if dob is not None:
                _expect(_is_iso_date(dob), issues, level="error", path="identity.date_of_birth", message="date_of_birth must be ISO YYYY-MM-DD")

    # acquisitions
    acquisitions = data.get("acquisitions")
    if acquisitions is not None:
        _expect(isinstance(acquisitions, list), issues, level="error", path="acquisitions", message="acquisitions must be a list")
        if isinstance(acquisitions, list):
            seen_ids: set[str] = set()
            for i, item in enumerate(acquisitions):
                pfx = f"acquisitions[{i}]"
                _expect(isinstance(item, dict), issues, level="error", path=pfx, message="each acquisition must be a dict")
                if not isinstance(item, dict):
                    continue

                acq_id = item.get("acquisition_id")
                _expect(isinstance(acq_id, str) and len(acq_id) > 0, issues, level="error", path=f"{pfx}.acquisition_id", message="acquisition_id required")
                if isinstance(acq_id, str) and acq_id:
                    if acq_id in seen_ids:
                        issues.append(SubjectYamlValidationIssue(level="error", path=f"{pfx}.acquisition_id", message=f"duplicate acquisition_id: {acq_id!r}"))
                    seen_ids.add(acq_id)

                date = item.get("date")
                _expect(_is_iso_date(date), issues, level="error", path=f"{pfx}.date", message="date must be ISO YYYY-MM-DD")

                session = item.get("session")
                _expect(isinstance(session, str) and bool(_SESSION_RE.match(session)), issues, level="warning", path=f"{pfx}.session", message="session should look like ses-01")

                place = item.get("place")
                _expect(isinstance(place, str) and len(place) > 0, issues, level="warning", path=f"{pfx}.place", message="place should be a non-empty string")

    # medication
    medication = data.get("medication")
    if medication is not None:
        _expect(isinstance(medication, list), issues, level="error", path="medication", message="medication must be a list")
        if isinstance(medication, list):
            for i, item in enumerate(medication):
                pfx = f"medication[{i}]"
                _expect(isinstance(item, dict), issues, level="error", path=pfx, message="each medication entry must be a dict")
                if not isinstance(item, dict):
                    continue

                start = item.get("start_date")
                _expect(_is_iso_date(start), issues, level="error", path=f"{pfx}.start_date", message="start_date must be ISO YYYY-MM-DD")

                end = item.get("end_date")
                if end is not None:
                    _expect(_is_iso_date(end), issues, level="error", path=f"{pfx}.end_date", message="end_date must be ISO YYYY-MM-DD")

                drugs = item.get("drugs")
                _expect(isinstance(drugs, list) and len(drugs) > 0, issues, level="warning", path=f"{pfx}.drugs", message="drugs should be a non-empty list")
                if isinstance(drugs, list):
                    for j, d in enumerate(drugs):
                        dp = f"{pfx}.drugs[{j}]"
                        _expect(isinstance(d, dict), issues, level="error", path=dp, message="each drug must be a dict")
                        if isinstance(d, dict):
                            name = d.get("name")
                            _expect(isinstance(name, str) and len(name) > 0, issues, level="error", path=f"{dp}.name", message="drug name required")

    # protocols
    protocols = data.get("protocols")
    if protocols is not None:
        _expect(isinstance(protocols, list), issues, level="error", path="protocols", message="protocols must be a list")
        if isinstance(protocols, list):
            seen_pids: set[str] = set()
            for i, item in enumerate(protocols):
                pfx = f"protocols[{i}]"
                _expect(isinstance(item, dict), issues, level="error", path=pfx, message="each protocol must be a dict")
                if not isinstance(item, dict):
                    continue

                pid = item.get("protocol_id")
                _expect(isinstance(pid, str) and len(pid) > 0, issues, level="error", path=f"{pfx}.protocol_id", message="protocol_id required")
                if isinstance(pid, str) and pid:
                    if pid in seen_pids:
                        issues.append(SubjectYamlValidationIssue(level="error", path=f"{pfx}.protocol_id", message=f"duplicate protocol_id: {pid!r}"))
                    seen_pids.add(pid)

                task = item.get("task")
                _expect(isinstance(task, str) and len(task) > 0, issues, level="warning", path=f"{pfx}.task", message="task should be a non-empty string")

                sessions = item.get("sessions")
                _expect(isinstance(sessions, list) and len(sessions) > 0, issues, level="error", path=f"{pfx}.sessions", message="sessions must be a non-empty list")
                if isinstance(sessions, list):
                    for j, s in enumerate(sessions):
                        _expect(isinstance(s, str) and bool(_SESSION_RE.match(s)), issues, level="warning", path=f"{pfx}.sessions[{j}]", message="session should look like ses-01")

    return issues
