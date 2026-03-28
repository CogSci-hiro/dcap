# =============================================================================
#                                   CLI
# =============================================================================
#
# Entry point for the `dcap` command-line interface.
#
# This module is a thin dispatcher:
# - parse global + subcommand arguments
# - call a single library function per subcommand
#
#
# =============================================================================
# Imports
# =============================================================================

import argparse
import importlib
from typing import Sequence

from dcap.cli.cli_types import CliCommand


# =============================================================================
# Command registry
# =============================================================================

_COMMAND_MODULES: dict[str, str] = {
    "registry": "dcap.cli.commands.registry",
    "bids-anat": "dcap.cli.commands.bids.bids_anat",
    "bids-convert": "dcap.cli.commands.bids.bids_convert",
    "report": "dcap.cli.commands.viz.report",
    "seeg-clinical-report": "dcap.cli.commands.viz.seeg_clinical_report",
    "preprocess": "dcap.cli.commands.preprocess",
}


def _load_command_module(command_name: str) -> CliCommand:
    module_path = _COMMAND_MODULES.get(command_name)
    if module_path is None:
        raise RuntimeError(f"Unknown command: {command_name}")
    return importlib.import_module(module_path)


# =============================================================================
# Argument parsing
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.

    Usage example
    -------------
        dcap bids-convert --help
        dcap registry validate --help
    """
    parser = argparse.ArgumentParser(
        prog="dcap",
        description="DCAP: sEEG data processing",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for name in _COMMAND_MODULES:
        module = _load_command_module(name)
        if not hasattr(module, "add_subparser"):
            raise RuntimeError(f"CLI command module for '{name}' is missing add_subparser().")
        module.add_subparser(subparsers)

    return parser


def build_command_parser(command_name: str) -> argparse.ArgumentParser:
    """
    Build a parser that only loads the requested subcommand module.

    This avoids import-time failures in unrelated commands from blocking the
    command the user actually asked to run.
    """
    parser = argparse.ArgumentParser(
        prog="dcap",
        description="DCAP: sEEG data processing",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    module = _load_command_module(command_name)
    if not hasattr(module, "add_subparser"):
        raise RuntimeError(f"CLI command module for '{command_name}' is missing add_subparser().")
    module.add_subparser(subparsers)
    return parser


# =============================================================================
# Entry point
# =============================================================================

def main(argv: Sequence[str] | None = None) -> None:
    """
    CLI entry point.

    Parameters
    ----------
    argv
        Optional argv for testing. If None, reads from sys.argv.

    Returns
    -------
    None

    Usage example
    -------------
        main(["registry", "validate", "--public-registry", "registry_public.tsv"])
    """
    argv_list = list(argv) if argv is not None else None

    if argv_list is None:
        import sys

        argv_list = sys.argv[1:]

    if len(argv_list) == 0:
        parser = build_arg_parser()
        args = parser.parse_args(argv_list)
    else:
        first = str(argv_list[0]).strip()
        if first in {"-h", "--help"}:
            parser = build_arg_parser()
            args = parser.parse_args(argv_list)
        else:
            parser = build_command_parser(first)
            args = parser.parse_args(argv_list)

    command_name = str(args.command)
    module = _load_command_module(command_name)

    if not hasattr(module, "run"):
        raise RuntimeError(f"CLI command module for '{command_name}' is missing run().")

    module.run(args)


if __name__ == "__main__":
    main()
