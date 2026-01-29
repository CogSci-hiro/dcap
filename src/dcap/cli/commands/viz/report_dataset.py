# =============================================================================
#                                 CLI Command
# =============================================================================
"""Generate dataset report (dataset-wide overview + QC snapshot).

Usage example
    dcap report dataset --help
"""

import argparse
from pathlib import Path
from typing import Any

from dcap.viz import api as viz_api


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("dataset", help="Generate dataset report (dataset-wide overview + QC snapshot).")
    _configure_parser(parser)
    parser.set_defaults(func=run)


def add_subparser(subparsers: Any) -> None:
    add_parser(subparsers)


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")


def run(args: argparse.Namespace) -> None:
    viz_api.make_dataset_report(out_dir=args.out_dir)
