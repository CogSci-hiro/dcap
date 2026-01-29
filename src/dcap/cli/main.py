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
from typing import Dict, Sequence

from dcap.cli.cli_types import CliCommand

from dcap.cli.commands import registry as cmd_registry
from dcap.cli.commands import bids_anat as cmd_bids_anat
from dcap.cli.commands import bids_convert as cmd_bids_convert
from dcap.cli.commands.viz import report as cmd_report


# =============================================================================
# Command registry
# =============================================================================

_COMMANDS: Dict[str, CliCommand] = {
    "registry": cmd_registry,
    "bids-anat": cmd_bids_anat,
    "bids-convert": cmd_bids_convert,
    "report": cmd_report
}


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

    for name, module in _COMMANDS.items():
        if not hasattr(module, "add_subparser"):
            raise RuntimeError(f"CLI command module for '{name}' is missing add_subparser().")
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
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    command_name = str(args.command)
    module = _COMMANDS.get(command_name)
    if module is None:
        raise RuntimeError(f"Unknown command: {command_name}")

    if not hasattr(module, "run"):
        raise RuntimeError(f"CLI command module for '{command_name}' is missing run().")

    module.run(args)


if __name__ == "__main__":
    main()
