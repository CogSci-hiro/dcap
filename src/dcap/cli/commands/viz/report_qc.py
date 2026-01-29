# =============================================================================
#                                 CLI Command
# =============================================================================
"""Generate QC report (validation gates + TRF baseline sanity checks).

Usage example
    dcap report qc --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dcap.viz import api as viz_api


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qc", help="Generate QC report (validation gates + TRF baseline sanity checks).")
    _configure_parser(parser)
    parser.set_defaults(func=run)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    add_parser(subparsers)



def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")


def run(args: argparse.Namespace) -> None:
    viz_api.make_qc_report(out_dir=args.out_dir)
