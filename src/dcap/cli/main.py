"""
Command-line interface entry point for dcap.
"""
import argparse
from pathlib import Path
from typing import Sequence

from dcap.registry.loader import (
    load_registry_table,
    resolve_registry_sources,
    validate_private_registry,
    validate_public_registry,
)
from dcap.registry.sanitize import make_public_registry


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser.

    Returns
    -------
    parser
        Argument parser for the `dcap` command.

    Usage example
    ------------
        from dcap.cli.main import build_parser

        parser = build_parser()
        _ = parser.parse_args(["--help"])
    """
    parser = argparse.ArgumentParser(prog="dcap", description="dcap command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Top-level placeholders
    subparsers.add_parser("bids-convert", help="Convert raw data to BIDS (placeholder)")
    subparsers.add_parser("qc-run", help="Run QC/validation (placeholder)")

    # Registry command group
    reg = subparsers.add_parser("registry", help="Registry utilities")
    reg_sub = reg.add_subparsers(dest="registry_cmd", required=False)

    reg_validate = reg_sub.add_parser("validate", help="Validate public/private registry tables")
    reg_validate.add_argument("--public", type=Path, required=True, help="Path to public registry (CSV/Parquet)")
    reg_validate.add_argument(
        "--private",
        type=Path,
        default=None,
        help="Optional path to private registry (CSV/Parquet). If omitted, uses DCAP_PRIVATE_ROOT if available.",
    )
    reg_validate.add_argument("--private-prefix", type=str, default="private__", help="Prefix for private columns")
    reg_validate.add_argument("--no-merge", action="store_true", help="Only validate; do not attempt merge")

    reg_merge = reg_sub.add_parser("merge", help="Load and merge public + private registry safely")
    reg_merge.add_argument("--public", type=Path, required=True, help="Path to public registry (CSV/Parquet)")
    reg_merge.add_argument("--private-prefix", type=str, default="private__", help="Prefix for private columns")

    reg_init = reg_sub.add_parser("init-templates", help="Copy registry templates to chosen locations")
    reg_init.add_argument(
        "--public-out",
        type=Path,
        required=True,
        help="Output path for the public registry template (CSV).",
    )
    reg_init.add_argument(
        "--private-out",
        type=Path,
        required=True,
        help="Output path for the private registry template (CSV).",
    )
    reg_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )

    reg_make_public = reg_sub.add_parser("make-public", help="Generate a sanitized public registry from a private registry")
    reg_make_public.add_argument("--private", type=Path, required=True, help="Path to private registry (CSV/Parquet)")
    reg_make_public.add_argument("--out", type=Path, required=True, help="Output public registry path (CSV/Parquet)")
    reg_make_public.add_argument("--bids-root", type=str, required=True, help="BIDS root path to record in public registry")
    reg_make_public.add_argument(
        "--default-qc-status",
        type=str,
        default="unknown",
        help="QC status used if private registry lacks qc_status (pass/fail/review/unknown)",
    )
    reg_make_public.add_argument(
        "--subject-map",
        type=Path,
        default=None,
        help="Optional mapping table with columns subject_key, subject (CSV/Parquet)",
    )
    reg_make_public.add_argument("--subject-key-column", type=str, default="subject_key", help="Subject key column name")
    reg_make_public.add_argument("--subject-column", type=str, default="subject", help="Anonymized subject column name")

    return parser


def _read_table(path: Path):
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_csv_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cmd_registry_validate(args: argparse.Namespace) -> int:
    public_path: Path = args.public
    private_path: Path | None = args.private

    public_df = _read_table(public_path)
    report = validate_public_registry(public_df)
    if not report.ok:
        print("Public registry: INVALID")
        for e in report.errors:
            print(f"- {e}")
        return 2
    print("Public registry: OK")

    if private_path is not None and private_path.exists():
        private_df = _read_table(private_path)
        report = validate_private_registry(private_df)
        if not report.ok:
            print("Private registry: INVALID")
            for e in report.errors:
                print(f"- {e}")
            return 2
        print("Private registry: OK")
    else:
        print("Private registry: (not provided or not found)")

    if not bool(args.no_merge):
        sources = resolve_registry_sources(public_path)
        if private_path is not None and private_path.exists():
            sources = type(sources)(public_registry_path=public_path, private_registry_path=private_path)
        _ = load_registry_table(sources, private_prefix=str(args.private_prefix), validate=True)
        print("Merge: OK")

    return 0


def _cmd_registry_merge(args: argparse.Namespace) -> int:
    public_path: Path = args.public
    sources = resolve_registry_sources(public_path)
    merged = load_registry_table(sources, private_prefix=str(args.private_prefix), validate=True)
    print(f"Merged registry rows: {len(merged)}")
    print("Columns:")
    for c in merged.columns:
        print(f"- {c}")
    return 0


def _cmd_registry_init_templates(args: argparse.Namespace) -> int:
    from importlib import resources

    public_out: Path = args.public_out
    private_out: Path = args.private_out
    force: bool = bool(args.force)

    for out in (public_out, private_out):
        if out.exists() and not force:
            print(f"Refusing to overwrite existing file: {out} (use --force to override)")
            return 3

    public_text = resources.files("dcap.registry.templates").joinpath("registry_public_TEMPLATE.csv").read_text("utf-8")
    private_text = (
        resources.files("dcap.registry.templates").joinpath("registry_private_TEMPLATE.csv").read_text("utf-8")
    )

    _write_csv_text(public_text, public_out)
    _write_csv_text(private_text, private_out)

    print(f"Wrote public template: {public_out}")
    print(f"Wrote private template: {private_out}")
    return 0


def _cmd_registry_make_public(args: argparse.Namespace) -> int:
    make_public_registry(
        private_registry_path=args.private,
        output_path=args.out,
        bids_root=str(args.bids_root),
        default_qc_status=str(args.default_qc_status),
        subject_map_path=args.subject_map,
        subject_key_column=str(args.subject_key_column),
        subject_column=str(args.subject_column),
    )
    print(f"Wrote public registry: {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the `dcap` CLI.

    Parameters
    ----------
    argv
        Optional sequence of CLI arguments. If None, uses sys.argv.

    Returns
    -------
    exit_code
        Exit code (0 indicates success).

    Usage example
    ------------
        from dcap.cli.main import main

        exit_code = main(
            [
                "registry",
                "make-public",
                "--private",
                "~/.dcap_private/registry_private.csv",
                "--out",
                "registry_public.csv",
                "--bids-root",
                "/data/bids/conversation",
            ]
        )
        assert exit_code == 0
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "registry" and args.registry_cmd == "validate":
        return _cmd_registry_validate(args)
    if args.command == "registry" and args.registry_cmd == "merge":
        return _cmd_registry_merge(args)
    if args.command == "registry" and args.registry_cmd == "init-templates":
        return _cmd_registry_init_templates(args)
    if args.command == "registry" and args.registry_cmd == "make-public":
        return _cmd_registry_make_public(args)

    return 0
