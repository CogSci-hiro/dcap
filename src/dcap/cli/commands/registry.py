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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dcap.registry.validate import resolve_private_root, validate_everything


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

def add_subparser(subparsers) -> None:  # noqa: ANN001
    """
    Register the `registry` command group.

    Usage example
    -------------
        dcap registry --help
    """
    parser = subparsers.add_parser(
        "registry",
        help="Registry utilities (validate, scan-bids, export-sanitized, ...).",
    )

    registry_subparsers = parser.add_subparsers(
        dest="registry_command",
        required=True,
        metavar="<registry-command>",
    )

    _add_registry_validate(registry_subparsers)


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


# =============================================================================
# Execution
# =============================================================================

def run(args) -> None:  # noqa: ANN001
    """
    Execute `dcap registry ...`.

    Usage example
    -------------
        # See dcap.cli.main usage example
    """
    private_root = resolve_private_root(str(args.private_root))
    exit_code = validate_everything(
        public_registry=Path(args.public_registry),
        spec_dir=Path(args.spec_dir),
        private_root=private_root,
        strict=bool(args.strict),
    )
    raise SystemExit(int(exit_code))


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
