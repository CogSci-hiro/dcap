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
import shutil
from typing import Any, Optional
import warnings

from dcap.bids.core.anat import export_subject_anat_electrodes_from_elec2atlas_mat
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
            source_root=Path("sourcedata"),
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
    subjects_dir: Path
    bids_root: Path
    bids_subject: str
    session: Optional[str]

    dataset_id: str
    task: str

    private_root_path: Optional[Path]
    subject_map_yaml: Optional[Path]
    task_assets_dir: Optional[Path]
    task_trigger_id: Optional[int]

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
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Sourcedata root containing subjects/, assets_dir/, and optional stimuli/",
    )
    p.add_argument("--subjects-dir", type=Path, required=True, help="Anatomy/recon sourcedata root")
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
        help="Optional task assets directory (defaults to <source-root>/assets_dir).",
    )
    p.add_argument(
        "--task-trigger-id",
        type=int,
        default=None,
        help="Optional task-specific trigger code override (currently used by sorciere).",
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

    sourcedata_root = cfg.source_root
    if not sourcedata_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {sourcedata_root}")

    resolved_source_root = sourcedata_root / "subjects"
    if not resolved_source_root.exists():
        raise FileNotFoundError(
            f"Expected subjects/ under source_root, but not found: {resolved_source_root}"
        )

    task_assets_dir = cfg.task_assets_dir if cfg.task_assets_dir is not None else (sourcedata_root / "assets_dir")

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
        task_assets_dir=task_assets_dir,
        task_trigger_id=cfg.task_trigger_id,
    )
    task = resolve_task(task_ctx)

    _ = convert_subject(cfg=core_cfg, task=task)
    if not cfg.dry_run:
        _copy_stimuli_once(source_root=cfg.source_root, bids_root=cfg.bids_root)
        _export_anat_electrodes_if_present(
            bids_root=cfg.bids_root,
            bids_subject=cfg.bids_subject,
            subjects_dir=cfg.subjects_dir,
            source_subject_id=dcap_id,
            overwrite=cfg.overwrite,
        )
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
        subjects_dir=Path(args.subjects_dir).expanduser().resolve(),
        bids_root=Path(args.bids_root).expanduser().resolve(),
        bids_subject=str(args.subject).strip(),
        session=str(args.session).strip() if args.session is not None else None,
        dataset_id=str(args.dataset_id).strip(),
        task=str(args.task).strip(),
        private_root_path=private_root,
        subject_map_yaml=Path(args.subject_map_yaml).expanduser().resolve() if args.subject_map_yaml is not None else None,
        task_assets_dir=Path(args.task_assets_dir).expanduser().resolve() if args.task_assets_dir is not None else None,
        task_trigger_id=int(args.task_trigger_id) if args.task_trigger_id is not None else None,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        preload_raw=bool(args.preload_raw),
        line_freq_hz=float(args.line_freq_hz),
        datatype=str(args.datatype).strip(),
    )


def _copy_stimuli_once(*, source_root: Path, bids_root: Path) -> None:
    stimuli_src = Path(source_root).resolve() / "stimuli"
    if not stimuli_src.exists() or not stimuli_src.is_dir():
        return
    stimuli_dst = Path(bids_root).resolve() / "stimuli"
    if stimuli_dst.exists():
        return
    shutil.copytree(stimuli_src, stimuli_dst)


def _export_anat_electrodes_if_present(
    *,
    bids_root: Path,
    bids_subject: str,
    subjects_dir: Path,
    source_subject_id: str,
    overwrite: bool,
) -> None:
    elec2atlas_mat = Path(subjects_dir).resolve() / str(source_subject_id).strip() / "elec_recon" / "elec2atlas.mat"
    result = export_subject_anat_electrodes_from_elec2atlas_mat(
        bids_root=Path(bids_root).resolve(),
        bids_subject=bids_subject,
        elec2atlas_mat_path=elec2atlas_mat,
        overwrite=overwrite,
    )
    if result is None:
        warnings.warn(f"Anatomical electrode recon MAT not found (continuing): {elec2atlas_mat}")
