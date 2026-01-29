# =============================================================================
#                           CLI: bids (group)
# =============================================================================
#
# No conversion logic here. This module:
# - defines arguments
# - converts argparse namespace -> config objects
# - calls a single library entry point
#
# =============================================================================

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

from dcap.bids.core.config import BidsCoreConfig
from dcap.bids.core.converter import convert_subject
from dcap.bids.tasks.registry import TaskFactoryContext, resolve_task
from dcap.registry.validate import resolve_private_root


@dataclass(frozen=True, slots=True)
class BidsConvertCliConfig:
    """
    CLI configuration for `dcap bids convert`.

    Usage example
    -------------
        cfg = BidsConvertCliConfig(
            source_root=Path("sourcedata/sub-001"),
            bids_root=Path("bids"),
            bids_subject="sub-001",
            session=None,
            dataset_id="Timone2025",
            task="diapix",
            private_root_mode="env",
            private_root_path=None,
            subject_map_yaml=None,
            task_assets_dir=Path("/private/assets/diapix"),
            overwrite=False,
            dry_run=False,
            preload_raw=True,
            line_freq_hz=50.0,
        )
    """

    source_root: Path
    bids_root: Path
    bids_subject: str
    session: Optional[str]

    dataset_id: str
    task: str

    private_root_path: Optional[Path]
    subject_map_yaml: Optional[Path]
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
    p.add_argument("--source-root", type=Path, required=True, help="Task-discovery root (task-specific layout)")
    p.add_argument("--bids-root", type=Path, required=True, help="BIDS output root")

    # BIDS-facing identity
    p.add_argument(
        "--subject",
        type=str,
        required=True,
        help="BIDS subject label (target), accepts '001' or 'sub-001'",
    )
    p.add_argument("--session", type=str, default=None, help="Optional BIDS session label without 'ses-'")

    # Task selection
    p.add_argument("--task", type=str, required=True, help="Task name (e.g., diapix)")
    p.add_argument("--dataset-id", type=str, required=True, help="Dataset identifier (must match subject map YAML)")

    p.add_argument(
        "--private-root",
        type=Path,
        required=False,
        default=None,
        help="Path to private metadata root (defaults to $DCAP_PRIVATE_ROOT).",
    )
    p.add_argument(
        "--subject-map-yaml",
        type=Path,
        default=None,
        help="Optional explicit path to subject re-identification YAML (overrides --private-root).",
    )

    # Task assets
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


def run(args: argparse.Namespace) -> int:
    if args.bids_cmd != "convert":
        raise RuntimeError(f"Unknown bids subcommand: {args.bids_cmd!r}")

    cfg = _parse_convert_args(args)

    private_root = resolve_private_root(str(cfg.private_root_path))

    core_cfg = BidsCoreConfig(
        source_root=cfg.source_root,
        bids_root=cfg.bids_root,
        subject=cfg.bids_subject,   # core will normalize sub- prefix etc.
        session=cfg.session,
        datatype="ieeg",
        overwrite=cfg.overwrite,
        dry_run=cfg.dry_run,
        preload_raw=cfg.preload_raw,
        line_freq=cfg.line_freq_hz,
    )

    task_ctx = TaskFactoryContext(
        task_name=cfg.task,
        dataset_id=cfg.dataset_id,
        bids_subject=cfg.bids_subject,
        session=cfg.session,
        private_root=private_root,
        subject_map_yaml=cfg.subject_map_yaml,
        task_assets_dir=cfg.task_assets_dir,
    )
    task = resolve_task(task_ctx)

    _ = convert_subject(cfg=core_cfg, task=task)
    return 0


def _parse_convert_args(args: argparse.Namespace) -> BidsConvertCliConfig:
    private_root = args.private_root
    if private_root is None:
        env = os.environ.get("DCAP_PRIVATE_ROOT")
        if env is None:
            raise ValueError(
                "BIDS conversion requires private metadata. "
                "Provide --private-root or set $DCAP_PRIVATE_ROOT."
            )
        private_root = Path(env)

    private_root = private_root.expanduser().resolve()

    return BidsConvertCliConfig(
        source_root=Path(args.source_root).expanduser().resolve(),
        bids_root=Path(args.bids_root).expanduser().resolve(),
        bids_subject=str(args.subject).strip(),
        session=str(args.session).strip() if args.session is not None else None,
        dataset_id=str(args.dataset_id).strip(),
        task=str(args.task).strip(),
        private_root_path=private_root,
        subject_map_yaml=Path(args.subject_map_yaml).expanduser().resolve() if args.subject_map_yaml is not None else None,
        task_assets_dir=Path(args.task_assets_dir).expanduser().resolve() if args.task_assets_dir is not None else None,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        preload_raw=bool(args.preload_raw),
        line_freq_hz=float(args.line_freq_hz),
    )
