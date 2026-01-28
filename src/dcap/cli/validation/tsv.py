# =============================================================================
#                    Validation: TSV table validation engine
# =============================================================================
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dcap.validation.errors import ValidationIssue
from dcap.validation.utils import is_non_empty_string, matches_pattern, parse_iso_date, to_bool


def validate_tsv_against_spec(
    tsv_path: Path,
    spec: Dict[str, Any],
) -> List[ValidationIssue]:
    """
    Validate a TSV file against a dcap table spec.

    Parameters
    ----------
    tsv_path
        Path to TSV file.
    spec
        Parsed schema spec (dict) with keys:
        - columns: mapping of column specs
        - primary_key: column name
        - optional_columns: optional mapping
        - rules: list of rules

    Returns
    -------
    list of ValidationIssue
        Validation issues.

    Notes
    -----
    This validator is intentionally simple and tailored to dcap's schema YAMLs.
    It checks:
    - required columns exist
    - no unexpected columns (unless declared in optional_columns)
    - basic type checks (string/date/number/boolean)
    - regex patterns for selected columns
    - primary key uniqueness (if declared)
    - rule: record_id_consistency (if declared)

    Usage example
    -------------
        issues = validate_tsv_against_spec(
            Path("registry_public.tsv"),
            spec,
        )
    """
    issues: List[ValidationIssue] = []

    if not tsv_path.exists():
        return [ValidationIssue(level="error", location=str(tsv_path), message="file not found")]

    try:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:
        return [ValidationIssue(level="error", location=str(tsv_path), message=f"failed to read TSV: {exc}")]

    declared_columns: Dict[str, Any] = dict(spec.get("columns", {}) or {})
    optional_columns: Dict[str, Any] = dict(spec.get("optional_columns", {}) or {})
    primary_key: Optional[str] = spec.get("primary_key")

    # Required columns present
    required_cols = [name for name, colspec in declared_columns.items() if bool(colspec.get("required", False))]
    for col in required_cols:
        if col not in df.columns:
            issues.append(ValidationIssue(level="error", location=f"{tsv_path.name}:{col}", message="missing required column"))

    # Unexpected columns
    allowed = set(declared_columns.keys()) | set(optional_columns.keys())
    unexpected = [c for c in df.columns if c not in allowed]
    if unexpected:
        issues.append(
            ValidationIssue(
                level="error",
                location=f"{tsv_path.name}:header",
                message=f"unexpected columns: {unexpected}",
            )
        )

    # Column-level checks
    for col, colspec in declared_columns.items():
        if col not in df.columns:
            continue

        col_type = str(colspec.get("type", "string")).lower()
        constraints = colspec.get("constraints", []) or []

        values = df[col].tolist()

        for row_idx, raw in enumerate(values, start=2):  # header is row 1
            location = f"{tsv_path.name}:row={row_idx}:{col}"
            value: Any = raw

            if colspec.get("required", False):
                if not is_non_empty_string(value):
                    issues.append(ValidationIssue(level="error", location=location, message="required value missing"))
                    continue

            if not is_non_empty_string(value):
                # Empty is allowed for optional fields; skip type checks
                continue

            # Type checks (minimal)
            if col_type == "string":
                pass
            elif col_type == "date":
                if parse_iso_date(value) is None:
                    issues.append(ValidationIssue(level="error", location=location, message="invalid ISO date (YYYY-MM-DD)"))
            elif col_type in {"number", "float"}:
                try:
                    float(value)
                except ValueError:
                    issues.append(ValidationIssue(level="error", location=location, message="invalid number"))
            elif col_type in {"integer", "int"}:
                try:
                    int(float(value))
                except ValueError:
                    issues.append(ValidationIssue(level="error", location=location, message="invalid integer"))
            elif col_type in {"boolean", "bool"}:
                if to_bool(value) is None:
                    issues.append(ValidationIssue(level="error", location=location, message="invalid boolean"))
            else:
                issues.append(ValidationIssue(level="warning", location=location, message=f"unknown declared type: {col_type!r}"))

            # Constraints: non_empty, pattern
            for c in constraints:
                if isinstance(c, dict) and "non_empty" in c:
                    if bool(c["non_empty"]) and not is_non_empty_string(value):
                        issues.append(ValidationIssue(level="error", location=location, message="must be non-empty"))
                if isinstance(c, dict) and "pattern" in c:
                    pattern = str(c["pattern"])
                    if not matches_pattern(value, pattern):
                        issues.append(ValidationIssue(level="error", location=location, message=f"must match pattern {pattern!r}"))

        # Pattern at top-level constraints too (common in our specs)
        if "constraints" not in colspec and "pattern" in colspec:
            pattern = str(colspec["pattern"])
            for row_idx, value in enumerate(values, start=2):
                if is_non_empty_string(value) and not matches_pattern(value, pattern):
                    issues.append(
                        ValidationIssue(
                            level="error",
                            location=f"{tsv_path.name}:row={row_idx}:{col}",
                            message=f"must match pattern {pattern!r}",
                        )
                    )

    # Primary key uniqueness
    if primary_key and primary_key in df.columns:
        pk = df[primary_key].tolist()
        seen: Dict[str, int] = {}
        for i, v in enumerate(pk, start=2):
            if not is_non_empty_string(v):
                continue
            if v in seen:
                issues.append(
                    ValidationIssue(
                        level="error",
                        location=f"{tsv_path.name}:row={i}:{primary_key}",
                        message=f"duplicate primary key (also at row {seen[v]})",
                    )
                )
            else:
                seen[v] = i

    # Rule: record_id_consistency
    rules = spec.get("rules", []) or []
    rule_names = {r.get("name") for r in rules if isinstance(r, dict) and "name" in r}

    if "record_id_consistency" in rule_names:
        required_for_record_id = ["dataset_id", "subject", "session", "task", "run", "datatype", "record_id"]
        if all(c in df.columns for c in required_for_record_id):
            for row_idx, row in enumerate(df[required_for_record_id].itertuples(index=False, name=None), start=2):
                dataset_id, subject, session, task, run, datatype, record_id = row
                expected = f"{dataset_id}|{subject}|{session}|{task}|{run}|{datatype}"
                if is_non_empty_string(record_id) and record_id != expected:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            location=f"{tsv_path.name}:row={row_idx}:record_id",
                            message=f"record_id mismatch; expected {expected!r}",
                        )
                    )

    return issues
