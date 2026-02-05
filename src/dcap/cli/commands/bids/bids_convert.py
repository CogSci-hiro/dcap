# src/dcap/cli/commands/bids_convert.py
# =============================================================================
#                            CLI: bids-convert
# =============================================================================
#
# Thin command:
# - parse args
# - resolve private root + subject map
# - build core config
# - resolve task via registry
# - call core converter
#
# REVIEW
# =============================================================================

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dcap.bids.core.config import BidsCoreConfig
from dcap.bids.core.converter import convert_subject
from dcap.bids.tasks.registry import TaskFactoryContext, resolve_task
from dcap.registry.validate import resolve_private_root
from dcap.bids.core.subject_mapping import load_subject_mapping_entry


@dataclass(frozen=True, slots=True)
class BidsConvertCliConfig:
    """
    CLI configuration for `dcap bids-convert`.

    Usage example
    -------------
        cfg = BidsConvertCliConfig(
            source_root=Path("sourcedata/sub-001"),
            bids_root=Path("bids"),
            bids_subject="sub-001",
            session=None,
            dataset_id="Timone2025",
            task="diapix",
            private_root_path=None,
            subject_map_yaml=None,
            task_assets_dir=Path("/private/assets/diapix"),
            overwrite=False,
            dry_run=False,
            preload_raw=True,
            line_freq_hz=50.0,
            datatype="ieeg",
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
    datatype: str


def add_subparser(subparsers: Any) -> None:
    """
    Register the bids-convert subcommand.

    Usage example
    -------------
        dcap bids-convert --task diapix --dataset-id Timone2025 \
          --source-root sourcedata/sub-001 --bids-root bids --subject sub-001 \
          --private-root /path/to/private --task-assets-dir /path/to/assets \
          --overwrite --preload-raw --line-freq-hz 50
    """
    p = subparsers.add_parser(
        "bids-convert",
        help="Convert sourcedata to BIDS using a selected task adapter (task-level runs only).",
    )

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

    # Private metadata root / subject mapping
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

    # Task assets (aux files like stim wav, onset TSV, atlas path, etc.)
    p.add_argument(
        "--task-assets-dir",
        type=Path,
        default=None,
        help="Optional task assets directory (task-specific auxiliary files).",
    )

    # Safety/runtime
    p.add_argument("--datatype", type=str, default="ieeg", help='BIDS datatype, e.g. "ieeg".')
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preload-raw", action="store_true")
    p.add_argument("--line-freq-hz", type=float, default=50.0)


def run(args: argparse.Namespace) -> int:
    """
    Run the bids-convert command.

    Parameters
    ----------
    args
        Parsed CLI args.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    cfg = _parse_args(args)

    private_root = resolve_private_root(str(cfg.private_root_path))

    # Resolve subject mapping YAML (if user didn’t pass it explicitly, default under private_root)
    subject_map_yaml = cfg.subject_map_yaml
    if subject_map_yaml is None:
        subject_map_yaml = private_root / "subject_keys.yaml"  # adjust filename if yours differs

    entry = load_subject_mapping_entry(
        mapping_yaml=subject_map_yaml,
        dataset_id=cfg.dataset_id,
        bids_subject=cfg.bids_subject,
    )
    dcap_id = entry.original_id

    resolved_source_root = cfg.source_root / dcap_id
    if not resolved_source_root.exists():
        raise FileNotFoundError(
            f"Resolved source_root does not exist: {resolved_source_root} "
            f"(source_root={cfg.source_root}, dcap_id={dcap_id})"
        )

    core_cfg = BidsCoreConfig(
        source_root=resolved_source_root,
        bids_root=cfg.bids_root,
        subject=cfg.bids_subject,
        session=cfg.session,
        datatype=cfg.datatype,
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


def _parse_args(args: argparse.Namespace) -> BidsConvertCliConfig:
    private_root = args.private_root
    if private_root is None:
        env = os.environ.get("DCAP_PRIVATE_ROOT")
        if env is None:
            raise ValueError(
                "BIDS conversion requires private metadata. "
                "Provide --private-root or set $DCAP_PRIVATE_ROOT."
            )
        private_root = Path(env)

    private_root = Path(private_root).expanduser().resolve()

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
        datatype=str(args.datatype).strip(),
    )
