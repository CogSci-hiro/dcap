# =============================================================================
#                                 CLI Command
# =============================================================================
"""Generate patient (subject) report.

Usage example
    dcap report patient --subject sub-001 --out-dir ./out
"""


import argparse
from pathlib import Path
from typing import Any

from dcap.viz import api as viz_api


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("patient", help="Generate patient report")
    _configure_parser(parser)
    parser.set_defaults(func=run)


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., sub-001)")
    parser.add_argument("--mode", type=str, default="clinical", choices=["clinical", "research"], help="Report mode")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")


def run(args: argparse.Namespace) -> None:
    viz_api.make_patient_report(subject=args.subject, out_dir=args.out_dir, mode=args.mode)
