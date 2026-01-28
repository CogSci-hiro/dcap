# =============================================================================
#                    Registry: validate everything (library)
# =============================================================================
#
# Library entry point for validating:
# - registry_public.tsv
# - registry_private.tsv (optional)
# - subject_keys.yaml (optional)
# - subjects/sub-*.yaml (optional)
#
# REVIEW
# =============================================================================

import os
from pathlib import Path
from typing import Optional

from dcap.validation.errors import ValidationIssue
from dcap.validation.spec import load_spec
from dcap.validation.tsv import validate_tsv_against_spec
from dcap.validation.yaml_docs import validate_subject_keys_yaml, validate_subject_yaml


# =============================================================================
# Defaults
# =============================================================================

SPEC_PUBLIC = "SPEC_registry_public.schema.yaml"
SPEC_PRIVATE = "SPEC_registry_private.schema.yaml"
SPEC_SUBJECT_KEYS = "SPEC_subject_keys.schema.yaml"
SPEC_SUBJECT = "SPEC_subject.schema.yaml"

PRIVATE_REGISTRY_NAME = "registry_private.tsv"
SUBJECT_KEYS_NAME = "subject_keys.yaml"
SUBJECTS_DIRNAME = "subjects"


def validate_everything(
    *,
    public_registry: Path,
    spec_dir: Path,
    private_root: Optional[Path],
    strict: bool,
) -> int:
    """
    Validate public + (optional) private metadata and print a report.

    Returns
    -------
    int
        Exit code:
        - 0: OK (warnings allowed unless strict)
        - 2: errors present
        - 3: warnings present and strict=True

    Usage example
    -------------
        code = validate_everything(
            public_registry=Path("registry_public.tsv"),
            spec_dir=Path("docs"),
            private_root=Path("~/.dcap_private").expanduser(),
            strict=False,
        )
    """
    spec_public = load_spec(spec_dir / SPEC_PUBLIC)
    spec_private = load_spec(spec_dir / SPEC_PRIVATE)
    spec_subject_keys = load_spec(spec_dir / SPEC_SUBJECT_KEYS)
    spec_subject = load_spec(spec_dir / SPEC_SUBJECT)

    issues: list[ValidationIssue] = []

    public_issues = validate_tsv_against_spec(public_registry, spec_public)
    _print_report(f"Public registry: {public_registry}", public_issues)
    issues.extend(public_issues)

    if private_root is None:
        print("\nPrivate metadata: skipped")
    elif not private_root.exists():
        print(f"\nPrivate metadata root not found (skipping): {private_root}")
    else:
        private_registry_path = private_root / PRIVATE_REGISTRY_NAME
        subject_keys_path = private_root / SUBJECT_KEYS_NAME
        subjects_dir = private_root / SUBJECTS_DIRNAME

        pr_issues = validate_tsv_against_spec(private_registry_path, spec_private)
        _print_report(f"Private run-level registry: {private_registry_path}", pr_issues)
        issues.extend(pr_issues)

        sk_issues = validate_subject_keys_yaml(subject_keys_path, spec_subject_keys)
        _print_report(f"Subject re-ID map: {subject_keys_path}", sk_issues)
        issues.extend(sk_issues)

        subj_issues: list[ValidationIssue] = []
        if subjects_dir.exists():
            subject_files = sorted(subjects_dir.glob("sub-*.yaml"))
            if not subject_files:
                subj_issues.append(
                    ValidationIssue(
                        level="warning",
                        location=str(subjects_dir),
                        message="no subject YAML files found (sub-*.yaml)",
                    )
                )
            for f in subject_files:
                subj_issues.extend(validate_subject_yaml(f, spec_subject))
        else:
            subj_issues.append(
                ValidationIssue(
                    level="warning",
                    location=str(subjects_dir),
                    message="subjects directory not found",
                )
            )

        _print_report(f"Subject YAMLs: {subjects_dir}/sub-*.yaml", subj_issues)
        issues.extend(subj_issues)

    n_err = sum(1 for i in issues if i.level == "error")
    n_warn = sum(1 for i in issues if i.level == "warning")

    print("\nSummary")
    print(f"  Errors:   {n_err}")
    print(f"  Warnings: {n_warn}")

    if n_err > 0:
        return 2
    if strict and n_warn > 0:
        return 3
    return 0


def resolve_private_root(private_root_arg: str) -> Optional[Path]:
    """
    Resolve private root from CLI-style argument.

    Accepts:
    - 'env'  -> DCAP_PRIVATE_ROOT
    - 'none' -> skip private checks
    - path   -> explicit directory

    Usage example
    -------------
        root = resolve_private_root("env")
    """
    arg = private_root_arg.strip()
    if arg.lower() in {"none", "null", "skip"}:
        return None
    if arg.lower() == "env":
        env_val = os.environ.get("DCAP_PRIVATE_ROOT", "").strip()
        return Path(env_val).expanduser().resolve() if env_val else None
    return Path(arg).expanduser().resolve()


def _print_report(title: str, issues: list[ValidationIssue]) -> None:
    print(f"\n{title}")
    if not issues:
        print("  OK")
        return
    for issue in issues:
        print(f"  [{issue.level.upper()}] {issue.location}: {issue.message}")
