# =============================================================================
#                           CLI: bids-convert
# =============================================================================
#
# No BIDS logic here. This module:
# - defines arguments
# - converts argparse namespace -> library config objects
# - calls a single library entry point
#
# REVIEW
# =============================================================================

from pathlib import Path

from dcap.bids.config import BidsConvertConfig, DiapixTimingConfig
from dcap.bids.converter import convert_subject_to_bids


def add_subparser(subparsers) -> None:  # noqa: ANN001
    """
    Register the `bids-convert` subcommand.

    Usage example
    -------------
        dcap bids-convert --help
    """
    parser = subparsers.add_parser(
        "bids-convert",
        help="Convert messy clinical source recordings to BIDS using MNE-BIDS.",
    )

    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)

    parser.add_argument("--subject", type=str, required=True, help="BIDS subject label (e.g., NicEle or 001).")
    parser.add_argument("--session", type=str, default=None, help="Optional BIDS session label (e.g., 01).")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--datatype", type=str, required=True, choices=["ieeg", "eeg", "meg"])
    parser.add_argument("--run", type=str, default=None, help="Optional run label override (e.g., 01).")

    parser.add_argument("--line-freq", type=float, default=50.0)

    parser.add_argument("--channels-tsv", type=Path, required=False, default=None)
    parser.add_argument("--audio-onsets-tsv", type=Path, required=False, default=None)
    parser.add_argument("--stim-wav", type=Path, required=False, default=None)

    parser.add_argument("--subjects-dir", type=Path, required=False, default=None)
    parser.add_argument("--original-id", type=str, required=False, default=None)
    parser.add_argument("--atlas-file", type=Path, required=False, default=None)

    parser.add_argument("--start-delay-s", type=float, default=4.0)
    parser.add_argument("--conversation-duration-s", type=float, default=240.0)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def run(args) -> None:  # noqa: ANN001
    """
    Execute `bids-convert`.

    Usage example
    -------------
        # See dcap.cli.main usage example
    """
    timing = DiapixTimingConfig(
        start_delay_s=float(args.start_delay_s),
        conversation_duration_s=float(args.conversation_duration_s),
    )

    cfg = BidsConvertConfig(
        source_root=Path(args.source_root),
        bids_root=Path(args.bids_root),
        subject=str(args.subject),
        session=str(args.session) if args.session is not None else None,
        task=str(args.task),
        datatype=str(args.datatype),
        run=str(args.run) if args.run is not None else None,
        line_freq=float(args.line_freq),
        channels_tsv=Path(args.channels_tsv) if args.channels_tsv is not None else None,
        audio_onsets_tsv=Path(args.audio_onsets_tsv) if args.audio_onsets_tsv is not None else None,
        stim_wav=Path(args.stim_wav) if args.stim_wav is not None else None,
        subjects_dir=Path(args.subjects_dir) if args.subjects_dir is not None else None,
        original_id=str(args.original_id) if args.original_id is not None else None,
        atlas_file=Path(args.atlas_file) if args.atlas_file is not None else None,
        timing=timing,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    convert_subject_to_bids(cfg)
