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
import shutil
from typing import Any, Optional
import warnings

from dcap.bids.core.config import BidsCoreConfig
from dcap.bids.core.converter import convert_subject
from dcap.bids.core.anat import export_subject_anat_electrodes_from_elec2atlas_mat
from dcap.bids.tasks.registry import TaskFactoryContext, resolve_task
from dcap.registry.validate import resolve_private_root
from dcap.bids.core.subject_mapping import load_subject_mapping_entry


@dataclass(frozen=True, slots=True)
class BidsConvertCliConfig:
    """
    CLI configuration for `dcap bids convert`.

    Usage example
    -------------
        cfg = BidsConvertCliConfig(
            source_root=Path("sourcedata"),
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
    subjects_dir: Path
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
        help="Optional task assets directory (defaults to <source-root>/assets_dir).",
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

    sourcedata_root = cfg.source_root
    if not sourcedata_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {sourcedata_root}")

    subjects_root = sourcedata_root / "subjects"
    if not subjects_root.exists():
        raise FileNotFoundError(f"Expected subjects/ under source_root, but not found: {subjects_root}")

    task_assets_dir = cfg.task_assets_dir if cfg.task_assets_dir is not None else (sourcedata_root / "assets_dir")

    core_cfg = BidsCoreConfig(
        source_root=subjects_root,
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
        task_assets_dir=task_assets_dir,
    )
    task = resolve_task(task_ctx)

    _ = convert_subject(cfg=core_cfg, task=task)
    if not cfg.dry_run:
        _copy_stimuli_once(source_root=cfg.source_root, bids_root=cfg.bids_root)
        _export_anat_electrodes_if_present(
            bids_root=cfg.bids_root,
            bids_subject=cfg.bids_subject,
            subjects_dir=cfg.subjects_dir,
            source_subject_id=_resolve_source_subject_id(cfg),
            overwrite=cfg.overwrite,
        )
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
        subjects_dir=Path(args.subjects_dir).expanduser().resolve(),
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
    source_subject_id: Optional[str],
    overwrite: bool,
) -> None:
    subject_folder = str(source_subject_id).strip() if source_subject_id else str(bids_subject).strip()
    elec2atlas_mat = Path(subjects_dir).resolve() / subject_folder / "elec_recon" / "elec2atlas.mat"

    result = export_subject_anat_electrodes_from_elec2atlas_mat(
        bids_root=Path(bids_root).resolve(),
        bids_subject=bids_subject,
        elec2atlas_mat_path=elec2atlas_mat,
        overwrite=overwrite,
    )
    if result is None:
        warnings.warn(f"Anatomical electrode recon MAT not found (continuing): {elec2atlas_mat}")


def _resolve_source_subject_id(cfg: BidsConvertCliConfig) -> Optional[str]:
    if cfg.subject_map_yaml is None and cfg.private_root_path is None:
        return None
    try:
        private_root = resolve_private_root(str(cfg.private_root_path)) if cfg.private_root_path is not None else None
        mapping_yaml = cfg.subject_map_yaml
        if mapping_yaml is None and private_root is not None:
            mapping_yaml = private_root / "subject_keys.yaml"
        if mapping_yaml is None:
            return None
        entry = load_subject_mapping_entry(
            mapping_yaml=mapping_yaml,
            dataset_id=cfg.dataset_id,
            bids_subject=cfg.bids_subject,
        )
        return entry.original_id
    except Exception:
        return None
