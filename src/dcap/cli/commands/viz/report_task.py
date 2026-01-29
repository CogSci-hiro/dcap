# =============================================================================
#                                 CLI Command
# =============================================================================
"""Generate task report (task-scoped overview + validation).

Usage example
    dcap report task --task conversation --out-dir ./out
"""

import argparse
from pathlib import Path
from typing import Any

from dcap.viz import api as viz_api


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("task", help="Generate task report")
    _configure_parser(parser)
    parser.set_defaults(func=run)

def add_subparser(subparsers: Any) -> None:
    add_parser(subparsers)


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., conversation)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")


def run(args: argparse.Namespace) -> None:
    viz_api.make_task_report(task=args.task, out_dir=args.out_dir)
