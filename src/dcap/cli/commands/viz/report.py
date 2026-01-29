# =============================================================================
#                              CLI: report
# =============================================================================
#
# `dcap report ...` command group.
#
# This command is a dispatcher for report subcommands:
# - dataset
# - task
# - patient
# - qc
#
# Each subcommand should live in its own module and expose:
# - add_subparser(subparsers)
# - run(args)
#
# =============================================================================

import argparse
from typing import Any, Dict

from dcap.cli.cli_types import CliCommand

from dcap.cli.commands.viz import report_dataset as cmd_report_dataset
from dcap.cli.commands.viz import report_task as cmd_report_task
from dcap.cli.commands.viz import report_patient as cmd_report_patient
from dcap.cli.commands.viz import report_qc as cmd_report_qc


# =============================================================================
# Subcommand registry
# =============================================================================

_SUBCOMMANDS: Dict[str, CliCommand] = {
    "dataset": cmd_report_dataset,
    "task": cmd_report_task,
    "patient": cmd_report_patient,
    "qc": cmd_report_qc,
}


# =============================================================================
# Parser wiring
# =============================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register `dcap report` and its subcommands.
    """
    parser = subparsers.add_parser(
        "report",
        help="Generate reports (dataset, task, patient, qc)",
    )
    report_subparsers = parser.add_subparsers(
        dest="report_kind",
        required=True,
        metavar="<report>",
    )

    for name, module in _SUBCOMMANDS.items():
        if not hasattr(module, "add_subparser"):
            raise RuntimeError(
                f"Report subcommand module for '{name}' is missing add_subparser()."
            )
        module.add_subparser(report_subparsers)

    parser.set_defaults(_report_parent="report")


# =============================================================================
# Execution
# =============================================================================

def run(args: argparse.Namespace) -> None:
    """
    Dispatch to the selected report subcommand.
    """
    report_kind = getattr(args, "report_kind", None)
    if report_kind is None:
        raise RuntimeError("Missing report_kind. Did parsing fail?")

    module = _SUBCOMMANDS.get(str(report_kind))
    if module is None:
        raise RuntimeError(f"Unknown report subcommand: {report_kind}")

    if not hasattr(module, "run"):
        raise RuntimeError(f"Report subcommand module for '{report_kind}' is missing run().")

    module.run(args)
