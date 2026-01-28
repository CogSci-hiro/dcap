# =============================================================================
#                           CLI: bids (group)
# =============================================================================
#
# Generic CLI for all BIDS tasks:
# - parse common core args
# - choose task by name
# - build task via task registry
# - call core.convert_subject()
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dcap.bids.core.config import BidsCoreConfig
from dcap.bids.core.converter import convert_subject
from dcap.bids.tasks.registry import TaskFactoryContext, resolve_task


@dataclass(frozen=True, slots=True)
class BidsConvertCliConfig:
    """
    CLI configuration for `dcap bids convert`.

    Usage example
    -------------
        cfg = BidsConvertCliConfig(
            source_root=Path("sourcedata/NicEle"),
            bids_root=Path("bids"),
            subject="NicEle",
            session=None,
            task="diapix",
            task_assets_dir=Path("assets/diapix"),
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

    task: str
    task_assets_dir: Optional[Path]

    overwrite: bool
    dry_run: bool
    preload_raw: bool
    line_freq_hz: float


def add_subparser(subparsers: Any) -> None:
    bids_parser = subparsers.add_parser("bids", help="BIDS conversion utilities")
    bids_sub = bids_parser.add_subparsers(dest="bids_cmd", required=True)

    _add_convert(bids_sub)


def _add_convert(subparsers: Any) -> None:
    p = subparsers.add_parser("convert", help="Convert sourcedata to BIDS using a selected task adapter")

    # Core inputs
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--bids-root", type=Path, required=True)
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--session", type=str, default=None)

    # Task selection
    p.add_argument("--task", type=str, required=True, help="Task name (e.g., diapix)")
    p.add_argument(
        "--task-assets-dir",
        type=Path,
        default=None,
        help="Optional task assets directory (task-specific auxiliary files).",
    )

    # Safety/runtime
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preload-raw", action="store_true")
    p.add_argument("--line-freq-hz", type=float, default=50.0)


def run(args) -> int:
    if args.bids_cmd != "convert":
        raise RuntimeError(f"Unknown bids subcommand: {args.bids_cmd!r}")

    cfg = _parse_convert_args(args)

    core_cfg = BidsCoreConfig(
        source_root=cfg.source_root,
        bids_root=cfg.bids_root,
        subject=cfg.subject,
        session=cfg.session,
        datatype="ieeg",
        overwrite=cfg.overwrite,
        dry_run=cfg.dry_run,
        preload_raw=cfg.preload_raw,
        line_freq=cfg.line_freq_hz,
    )

    task_ctx = TaskFactoryContext(
        subject=cfg.subject,
        session=cfg.session,
        task_assets_dir=cfg.task_assets_dir,
    )
    task = resolve_task(cfg.task, task_ctx)

    _ = convert_subject(cfg=core_cfg, task=task)
    return 0


def _parse_convert_args(args) -> BidsConvertCliConfig:  # noqa: ANN001
    return BidsConvertCliConfig(
        source_root=Path(args.source_root).expanduser().resolve(),
        bids_root=Path(args.bids_root).expanduser().resolve(),
        subject=str(args.subject).strip(),
        session=str(args.session).strip() if args.session is not None else None,
        task=str(args.task).strip(),
        task_assets_dir=Path(args.task_assets_dir).expanduser().resolve() if args.task_assets_dir is not None else None,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        preload_raw=bool(args.preload_raw),
        line_freq_hz=float(args.line_freq_hz),
    )
