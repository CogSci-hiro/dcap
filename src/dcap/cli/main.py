"""
Command-line interface entry point for dcap.
"""
import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser.

    Returns
    -------
    parser
        Argument parser for the `dcap` command.

    Usage example
    ------------
        from dcap.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["--help"])
    """
    parser = argparse.ArgumentParser(prog="dcap", description="dcap command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Placeholder subcommands
    subparsers.add_parser("bids-convert", help="Convert raw data to BIDS (placeholder)")
    subparsers.add_parser("qc-run", help="Run QC/validation (placeholder)")
    subparsers.add_parser("registry", help="Query dataset registry (placeholder)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the `dcap` CLI.

    Parameters
    ----------
    argv
        Optional sequence of CLI arguments. If None, uses sys.argv.

    Returns
    -------
    exit_code
        Exit code (0 indicates success).

    Usage example
    ------------
        from dcap.cli.main import main

        exit_code = main(["--help"])
        assert exit_code == 0
    """
    parser = build_parser()
    _ = parser.parse_args(list(argv) if argv is not None else None)

    # Skeleton: no behavior yet.
    return 0
