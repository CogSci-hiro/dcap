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
from dcap.cli.commands.bids import bids_anat as cmd_bids_anat, bids_convert as cmd_bids_convert
from dcap.cli.commands.viz import report as cmd_report
from dcap.cli.commands import preprocess as cmd_preprocess

try:
    from dcap.cli.commands.viz import seeg_clinical_report as cmd_seeg_clinical_report
except ImportError:
    # Optional command path can be unavailable during TRF/analysis deprecation.
    cmd_seeg_clinical_report = None


# =============================================================================
# Command registry
# =============================================================================

_COMMANDS: Dict[str, CliCommand] = {
    "registry": cmd_registry,
    "bids-anat": cmd_bids_anat,
    "bids-convert": cmd_bids_convert,
    "report": cmd_report,
    "preprocess": cmd_preprocess
}

if cmd_seeg_clinical_report is not None:
    _COMMANDS["seeg-clinical-report"] = cmd_seeg_clinical_report


def _iter_argparse_commands() -> Dict[str, CliCommand]:
    """
    Return only CLI command modules that implement the argparse command contract.
    """
    valid: Dict[str, CliCommand] = {}
    for name, module in _COMMANDS.items():
        if hasattr(module, "add_subparser") and hasattr(module, "run"):
            valid[name] = module
    return valid


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

    for name, module in _iter_argparse_commands().items():
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
    module = _iter_argparse_commands().get(command_name)
    if module is None:
        raise RuntimeError(f"Unknown command: {command_name}")

    module.run(args)


if __name__ == "__main__":
    main()
