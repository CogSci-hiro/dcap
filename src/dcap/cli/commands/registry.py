# =============================================================================
#                          CLI: registry (group)
# =============================================================================
#
# No registry logic here. This module:
# - defines arguments
# - converts argparse namespace -> library config objects
# - calls a single library entry point
#
# REVIEW
# =============================================================================

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from dcap.registry.build import build_public_registry
from dcap.registry.validate import resolve_private_root, validate_registry
from dcap.registry.view import build_registry_view, write_registry_view_tsv
from dcap.registry.products import write_availability_index_by_task


# =============================================================================
# Config
# =============================================================================

PrivateRootMode = Literal["env", "none", "path"]


@dataclass(frozen=True, slots=True)
class RegistryValidateCliConfig:
    """
    CLI configuration for `dcap registry validate`.

    Usage example
    -------------
        cfg = RegistryValidateCliConfig(
            public_registry=Path("registry_public.tsv"),
            spec_dir=Path("docs"),
            private_root_mode="env",
            private_root_path=None,
            strict=False,
        )
    """

    public_registry: Path
    spec_dir: Path
    private_root_mode: PrivateRootMode
    private_root_path: Optional[Path]
    strict: bool


# =============================================================================
# argparse wiring
# =============================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `registry` command group.

    Usage example
    -------------
        dcap registry --help
    """
    registry_parser = subparsers.add_parser("registry", help="Registry utilities")
    registry_sub = registry_parser.add_subparsers(dest="registry_cmd", required=True)

    _add_registry_validate(registry_sub)
    _add_build_public_subcommand(registry_sub)
    _add_view_subcommand(registry_sub)
    _add_export_availability_subcommand(registry_sub)


def _add_registry_validate(subparsers) -> None:  # noqa: ANN001
    """
    Register `dcap registry validate`.

    Usage example
    -------------
        dcap registry validate --public-registry registry_public.tsv --spec-dir docs
    """
    parser = subparsers.add_parser(
        "validate",
        help="Validate registry + private metadata against schema specs.",
    )

    parser.add_argument(
        "--public-registry",
        type=Path,
        required=True,
        help="Path to registry_public.tsv",
    )

    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=Path("."),
        help="Directory containing schema YAML files (default: current directory).",
    )

    parser.add_argument(
        "--private-root",
        type=str,
        default="env",
        help="Private root: 'env' (DCAP_PRIVATE_ROOT), 'none' (skip), or a path.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit if any warning).",
    )


def _add_view_subcommand(subparsers: Any) -> None:
    p = subparsers.add_parser("view", help="Build registry view (public + optional private overlays)")
    p.add_argument("--public-registry", dest="public_registry", type=Path, required=True)
    p.add_argument(
        "--private-root",
        type=str,
        default="env",
        help="Private root: 'env' (DCAP_PRIVATE_ROOT), 'none' (skip), or an explicit path",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output TSV path for the full view (local-only; may contain private fields)",
    )
    p.add_argument("--summary-only", action="store_true", help="Only print summary counts")
    p.set_defaults(registry_cmd="view")


def _add_export_availability_subcommand(subparsers: Any) -> None:
    p = subparsers.add_parser("export-availability", help="Export sanitized availability indices (shareable)")
    p.add_argument("--public-registry", dest="public_registry", type=Path, required=True)
    p.add_argument(
        "--private-root",
        type=str,
        default="env",
        help="Private root: 'env' (DCAP_PRIVATE_ROOT), 'none' (skip), or an explicit path",
    )
    p.add_argument("--out", type=Path, required=True, help="Output TSV path (sanitized)")
    p.set_defaults(registry_cmd="export-availability")


# =============================================================================
# Execution
# =============================================================================

def run(args) -> int:
    if args.registry_cmd == "validate":
        return validate_registry(
            public_registry=Path(args.public_registry),
            private_root=Path(args.private_root) if args.private_root is not None else None,
            strict=bool(args.strict),
        )

    if args.registry_cmd == "build-public":
        out_path = build_public_registry(
            public_registry_out=Path(args.out),
            private_root=Path(args.private_root),
            dataset_id=args.dataset_id,
        )
        if args.validate:
            return validate_registry(
                public_registry=out_path,
                private_root=Path(args.private_root),
                strict=bool(args.strict),
            )
        return 0

    if args.registry_cmd == "view":
        private_root = resolve_private_root(args.private_root)
        private_registry = None

        if private_root is not None:
            candidate = private_root / "registry_private.tsv"
            private_registry = candidate if candidate.exists() else None

        view_rows = build_registry_view(
            public_registry=args.public_registry,
            private_registry=private_registry,
        )

        total = len(view_rows)
        excluded = sum(1 for r in view_rows if bool(r.get("excluded", False)))
        usable = total - excluded

        print("Registry view summary")
        print(f"  Total records:   {total}")
        print(f"  Excluded records:{excluded}")
        print(f"  Usable records:  {usable}")

        if args.summary_only:
            return 0

        if args.out is not None:
            write_registry_view_tsv(view_rows=view_rows, out_tsv=args.out)
            print(f"\nWrote registry view TSV: {args.out}")

        return 0

    if args.registry_cmd == "export-availability":
        private_root = resolve_private_root(args.private_root)
        private_registry = None

        if private_root is not None:
            candidate = private_root / "registry_private.tsv"
            private_registry = candidate if candidate.exists() else None

        view_rows = build_registry_view(
            public_registry=args.public_registry,
            private_registry=private_registry,
        )

        write_availability_index_by_task(
            registry_view_rows=view_rows,
            out_tsv=args.out,
        )
        print(f"Wrote availability index TSV: {args.out}")
        return 0

    raise RuntimeError(f"Unknown registry subcommand: {args.registry_cmd!r}")


def _parse_validate_args(args) -> RegistryValidateCliConfig:  # noqa: ANN001
    private_root_raw = str(args.private_root).strip()

    if private_root_raw.lower() in {"none", "null", "skip"}:
        private_root_mode: PrivateRootMode = "none"
        private_root_path = None
    elif private_root_raw.lower() == "env":
        private_root_mode = "env"
        private_root_path = None
    else:
        private_root_mode = "path"
        private_root_path = Path(private_root_raw).expanduser().resolve()

    return RegistryValidateCliConfig(
        public_registry=Path(args.public_registry),
        spec_dir=Path(args.spec_dir),
        private_root_mode=private_root_mode,
        private_root_path=private_root_path,
        strict=bool(args.strict),
    )


def _add_build_public_subcommand(subparsers: Any) -> None:
    p = subparsers.add_parser("build-public", help="Build sanitized public registry TSV")
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--dataset-id", type=str, default=None)
    p.add_argument("--out", type=Path, required=True, help="Output public TSV path")
    p.add_argument("--validate", action="store_true", help="Run validation after building")
    p.add_argument("--strict", action="store_true", help="Stricter builder behavior")
    p.set_defaults(func=_run_build_public)


def _add_validate_subcommand(subparsers: Any) -> None:
    p = subparsers.add_parser("validate", help="Validate registry inputs")
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--public-registry", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=_run_validate)


def _run_build_public(args: argparse.Namespace) -> int:
    out_path = build_public_registry(
        public_registry_out=args.out,
        private_root=args.private_root,
        dataset_id=args.dataset_id,
        strict=args.strict,
    )

    if args.validate:
        return validate_registry(
            public_registry=out_path,
            private_root=args.private_root,
            strict=args.strict,
        )

    return 0


def _run_validate(args: argparse.Namespace) -> int:
    return validate_registry(
        public_registry=args.public_registry,
        private_root=args.private_root,
        strict=args.strict,
    )
