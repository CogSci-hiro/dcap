# =============================================================================
#                           CLI: bids (group)
# =============================================================================
#
# No conversion logic here. This module:
# - defines arguments
# - converts argparse namespace -> library config objects
# - calls a single library entry point
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dcap.bids.tasks.diapix.convert import DiapixConvertConfig, convert_diapix


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True, slots=True)
class DiapixConvertCliConfig:
    """
    CLI configuration for `dcap bids diapix`.

    Usage example
    -------------
        cfg = DiapixConvertCliConfig(
            source_root=Path("sourcedata/NicEle"),
            bids_root=Path("bids"),
            subject="NicEle",
            session=None,
            audio_onsets_tsv=Path("audio_onsets.tsv"),
            stim_wav=Path("beeps.wav"),
            atlas_path=Path("elec2atlas.mat"),
            overwrite=False,
            dry_run=False,
            preload_raw=True,
            line_freq_hz=50.0,
        )
    """

    source_root: Path
    bids_root: Path
    subject: str
    session: Optional[str]
    audio_onsets_tsv: Path
    stim_wav: Path
    atlas_path: Path
    overwrite: bool
    dry_run: bool
    preload_raw: bool
    line_freq_hz: float


# =============================================================================
# argparse wiring
# =============================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `bids` command group.

    Usage example
    -------------
        dcap bids --help
    """
    bids_parser = subparsers.add_parser("bids", help="BIDS conversion utilities")
    bids_sub = bids_parser.add_subparsers(dest="bids_cmd", required=True)

    _add_diapix_convert(bids_sub)


def _add_diapix_convert(subparsers: Any) -> None:
    """
    Register `dcap bids diapix`.

    Usage example
    -------------
        dcap bids diapix --source-root sourcedata/NicEle --bids-root bids --subject NicEle \
          --audio-onsets-tsv audio_onsets.tsv --stim-wav beeps.wav --atlas-path elec2atlas.mat
    """
    p = subparsers.add_parser("diapix", help="Convert Diapix data to BIDS (iEEG + task artifacts)")

    p.add_argument("--source-root", type=Path, required=True, help="Directory containing conversation_*.vhdr/wav/asf")
    p.add_argument("--bids-root", type=Path, required=True, help="BIDS output root")
    p.add_argument("--subject", type=str, required=True, help="BIDS subject label without 'sub-'")
    p.add_argument("--session", type=str, default=None, help="Optional BIDS session label without 'ses-'")

    p.add_argument("--audio-onsets-tsv", type=Path, required=True, help="Path to audio_onsets.tsv")
    p.add_argument("--stim-wav", type=Path, required=True, help="Path to trigger reference WAV")
    p.add_argument("--atlas-path", type=Path, required=True, help="Path to elec2atlas.mat")

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true", help="Run checks but do not write outputs")
    p.add_argument("--preload-raw", action="store_true", help="Preload raw when reading BrainVision")
    p.add_argument("--line-freq-hz", type=float, default=50.0, help="Line frequency (default: 50 Hz)")

    # If your main() dispatch uses args.<group>_cmd checks (like registry_cmd), keep it consistent:
    # we rely on run(args) below checking args.bids_cmd == "diapix".


# =============================================================================
# Execution
# =============================================================================

def run(args) -> int:
    if args.bids_cmd == "diapix":
        cli_cfg = _parse_diapix_args(args)
        lib_cfg = DiapixConvertConfig(
            source_root=cli_cfg.source_root,
            bids_root=cli_cfg.bids_root,
            subject=cli_cfg.subject,
            session=cli_cfg.session,
            overwrite=cli_cfg.overwrite,
            dry_run=cli_cfg.dry_run,
            preload_raw=cli_cfg.preload_raw,
            line_freq_hz=cli_cfg.line_freq_hz,
            audio_onsets_tsv=cli_cfg.audio_onsets_tsv,
            stim_wav=cli_cfg.stim_wav,
            atlas_path=cli_cfg.atlas_path,
        )
        return convert_diapix(lib_cfg)

    raise RuntimeError(f"Unknown bids subcommand: {args.bids_cmd!r}")


def _parse_diapix_args(args) -> DiapixConvertCliConfig:  # noqa: ANN001
    return DiapixConvertCliConfig(
        source_root=Path(args.source_root).expanduser().resolve(),
        bids_root=Path(args.bids_root).expanduser().resolve(),
        subject=str(args.subject).strip(),
        session=str(args.session).strip() if args.session is not None else None,
        audio_onsets_tsv=Path(args.audio_onsets_tsv).expanduser().resolve(),
        stim_wav=Path(args.stim_wav).expanduser().resolve(),
        atlas_path=Path(args.atlas_path).expanduser().resolve(),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        preload_raw=bool(args.preload_raw),
        line_freq_hz=float(args.line_freq_hz),
    )
