# =============================================================================
#                      CLI: validate registries + private metadata
# =============================================================================
import argparse
import os
from pathlib import Path
from typing import List, Optional

from dcap.validation.errors import ValidationIssue
from dcap.validation.spec import load_spec
from dcap.validation.tsv import validate_tsv_against_spec
from dcap.validation.yaml_docs import validate_subject_keys_yaml, validate_subject_yaml


# -----------------------------------------------------------------------------
# Defaults (filenames)
# -----------------------------------------------------------------------------
DEFAULT_SPEC_PUBLIC = "SPEC_registry_public.schema.yaml"
DEFAULT_SPEC_PRIVATE = "SPEC_registry_private.schema.yaml"
DEFAULT_SPEC_SUBJECT_KEYS = "SPEC_subject_keys.schema.yaml"
DEFAULT_SPEC_SUBJECT = "SPEC_subject.schema.yaml"

DEFAULT_PRIVATE_REGISTRY_NAME = "registry_private.tsv"
DEFAULT_SUBJECT_KEYS_NAME = "subject_keys.yaml"
DEFAULT_SUBJECTS_DIRNAME = "subjects"


def _format_issue(issue: ValidationIssue) -> str:
    return f"[{issue.level.upper()}] {issue.location}: {issue.message}"


def _load_spec_from_dir(spec_dir: Path, filename: str) -> Path:
    path = spec_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    return path


def _print_report(title: str, issues: List[ValidationIssue]) -> None:
    print(f"\n{title}")
    if not issues:
        print("  OK")
        return
    for i in issues:
        print(f"  {_format_issue(i)}")


def _count_levels(issues: List[ValidationIssue]) -> tuple[int, int]:
    n_err = sum(1 for i in issues if i.level == "error")
    n_warn = sum(1 for i in issues if i.level == "warning")
    return n_err, n_warn


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dcap-check",
        description="Validate dcap public registry + private metadata (TSV/YAML) against schema specs.",
    )

    # Public registry
    parser.add_argument(
        "--public-registry",
        type=Path,
        required=True,
        help="Path to registry_public.tsv",
    )

    # Private root
    parser.add_argument(
        "--private-root",
        type=str,
        default="env",
        help="Private root directory. Use 'env' to read DCAP_PRIVATE_ROOT, or provide a path. Use 'none' to skip.",
    )

    # Specs
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=Path("."),
        help="Directory containing schema YAML files (defaults to current directory).",
    )
    parser.add_argument("--spec-public", type=str, default=DEFAULT_SPEC_PUBLIC)
    parser.add_argument("--spec-private", type=str, default=DEFAULT_SPEC_PRIVATE)
    parser.add_argument("--spec-subject-keys", type=str, default=DEFAULT_SPEC_SUBJECT_KEYS)
    parser.add_argument("--spec-subject", type=str, default=DEFAULT_SPEC_SUBJECT)

    # Behavior
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit if any warning).",
    )

    args = parser.parse_args(argv)

    spec_dir: Path = args.spec_dir

    public_registry_path: Path = args.public_registry

    private_root_arg: str = args.private_root.strip()
    private_root: Optional[Path]
    if private_root_arg.lower() in {"none", "null", "skip"}:
        private_root = None
    elif private_root_arg.lower() == "env":
        env_val = os.environ.get("DCAP_PRIVATE_ROOT", "").strip()
        private_root = Path(env_val).expanduser().resolve() if env_val else None
    else:
        private_root = Path(private_root_arg).expanduser().resolve()

    # Load specs
    spec_public_path = _load_spec_from_dir(spec_dir, args.spec_public)
    spec_private_path = _load_spec_from_dir(spec_dir, args.spec_private)
    spec_subject_keys_path = _load_spec_from_dir(spec_dir, args.spec_subject_keys)
    spec_subject_path = _load_spec_from_dir(spec_dir, args.spec_subject)

    spec_public = load_spec(spec_public_path)
    spec_private = load_spec(spec_private_path)
    spec_subject_keys = load_spec(spec_subject_keys_path)
    spec_subject = load_spec(spec_subject_path)

    # Validate public registry
    public_issues = validate_tsv_against_spec(public_registry_path, spec_public)
    _print_report(f"Public registry: {public_registry_path}", public_issues)

    # Validate private metadata (optional)
    private_registry_issues: List[ValidationIssue] = []
    subject_keys_issues: List[ValidationIssue] = []
    subject_files_issues: List[ValidationIssue] = []

    if private_root is None:
        print("\nPrivate metadata: skipped")
    else:
        if not private_root.exists():
            print(f"\nPrivate metadata root not found (skipping): {private_root}")
        else:
            private_registry_path = private_root / DEFAULT_PRIVATE_REGISTRY_NAME
            subject_keys_path = private_root / DEFAULT_SUBJECT_KEYS_NAME
            subjects_dir = private_root / DEFAULT_SUBJECTS_DIRNAME

            private_registry_issues = validate_tsv_against_spec(private_registry_path, spec_private)
            _print_report(f"Private run-level registry: {private_registry_path}", private_registry_issues)

            subject_keys_issues = validate_subject_keys_yaml(subject_keys_path, spec_subject_keys)
            _print_report(f"Subject re-ID map: {subject_keys_path}", subject_keys_issues)

            if subjects_dir.exists():
                subject_files = sorted(subjects_dir.glob("sub-*.yaml"))
                if not subject_files:
                    subject_files_issues.append(
                        ValidationIssue(
                            level="warning",
                            location=str(subjects_dir),
                            message="no subject YAML files found (sub-*.yaml)",
                        )
                    )
                for f in subject_files:
                    subject_files_issues.extend(validate_subject_yaml(f, spec_subject))
            else:
                subject_files_issues.append(
                    ValidationIssue(
                        level="warning",
                        location=str(subjects_dir),
                        message="subjects directory not found",
                    )
                )

            _print_report(f"Subject YAMLs: {subjects_dir}/sub-*.yaml", subject_files_issues)

    # Summary
    all_issues = public_issues + private_registry_issues + subject_keys_issues + subject_files_issues
    n_err, n_warn = _count_levels(all_issues)

    print("\nSummary")
    print(f"  Errors:   {n_err}")
    print(f"  Warnings: {n_warn}")

    if n_err > 0:
        return 2
    if args.strict and n_warn > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
