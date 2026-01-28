# =============================================================================
#                              DCAP: BIDS CLI
# =============================================================================
# > Command-line entry point for converting source recordings to BIDS using MNE-BIDS.

import argparse
from pathlib import Path
from typing import Optional

from dcap_bids.converter import ConvertConfig, convert_all
from dcap_bids.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Usage example
        python -m dcap_bids.cli convert \
            --source-root /path/to/source \
            --bids-root /path/to/bids \
            --subject sub-001 \
            --session ses-01 \
            --task conversation \
            --datatype ieeg \
            --dry-run
    """
    parser = argparse.ArgumentParser(
        prog="dcap-bids",
        description="Convert messy clinical source data to BIDS using MNE-BIDS.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert source data to BIDS.")
    convert_parser.add_argument("--source-root", type=Path, required=True)
    convert_parser.add_argument("--bids-root", type=Path, required=True)

    convert_parser.add_argument("--subject", type=str, required=True, help="BIDS subject label (e.g., 'sub-001' or '001').")
    convert_parser.add_argument("--session", type=str, default=None, help="BIDS session label (e.g., '01' or 'ses-01').")
    convert_parser.add_argument("--task", type=str, required=True)
    convert_parser.add_argument("--datatype", type=str, required=True, choices=["ieeg", "eeg", "meg"])

    convert_parser.add_argument("--run", type=str, default=None, help="Optional run label (e.g., '01').")

    convert_parser.add_argument("--line-freq", type=float, default=50.0, help="Power line frequency (Hz).")
    convert_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS files.")
    convert_parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; just log actions.")

    convert_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main CLI entry point.

    Parameters
    ----------
    argv
        Optional argument vector; mainly for testing.

    Returns
    -------
    int
        Process exit code.

    Usage example
        dcap-bids convert --source-root ./source --bids-root ./bids \
            --subject 001 --session 01 --task conversation --datatype ieeg
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(log_level=args.log_level)

    if args.command == "convert":
        cfg = ConvertConfig(
            source_root=args.source_root,
            bids_root=args.bids_root,
            subject=args.subject,
            session=args.session,
            task=args.task,
            datatype=args.datatype,
            run=args.run,
            line_freq=args.line_freq,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        convert_all(cfg)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
