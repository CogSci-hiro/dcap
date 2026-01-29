# =============================================================================
#                                 CLI Command
# =============================================================================
"""Generate validation figure bundle (integrity + sampling + channels + events + annotations).

Usage example
    dcap viz validate --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dcap.viz import api as viz_api


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate", help="Generate validation figure bundle (integrity + sampling + channels + events + annotations).")
    _configure_parser(parser)
    parser.set_defaults(func=run)


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")


def run(args: argparse.Namespace) -> None:
    viz_api.make_validation_bundle(out_dir=args.out_dir)
