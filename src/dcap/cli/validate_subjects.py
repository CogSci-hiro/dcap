# =============================================================================
#                         CLI: validate subject YAMLs
# =============================================================================
import argparse
from pathlib import Path
from typing import List

from dcap.private.subjects import load_subject_yaml, validate_subject_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dcap private subject YAML files.")
    parser.add_argument("--subjects-dir", type=Path, required=True, help="Directory containing sub-XXX.yaml files")
    args = parser.parse_args()

    subjects_dir: Path = args.subjects_dir
    paths = sorted(subjects_dir.glob("sub-*.yaml"))

    if not paths:
        print(f"No subject YAML files found in: {subjects_dir}")
        return 1

    total_errors = 0
    for path in paths:
        data = load_subject_yaml(path)
        issues = validate_subject_yaml(data)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]

        if issues:
            print(f"\n{path.name}")
            for issue in issues:
                print(f"  [{issue.level.upper()}] {issue.path}: {issue.message}")

        total_errors += len(errors)

    if total_errors > 0:
        print(f"\nValidation failed with {total_errors} error(s).")
        return 2

    print("\nAll subject YAML files passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
