# dcap/bids/tasks/diapix/convert.py
# =============================================================================
#                 Library: Diapix conversion (via task registry)
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dcap.bids.core.config import BidsCoreConfig
from dcap.bids.core.converter import convert_subject
from dcap.bids.tasks.registry import TaskFactoryContext, resolve_task


@dataclass(frozen=True, slots=True)
class DiapixConvertConfig:
    source_root: Path
    bids_root: Path
    dataset_id: str
    subject: str                 # BIDS subject, e.g. "sub-001" or "001"
    session: Optional[str]

    private_root: Optional[Path]         # resolved path or None if using subject_map_yaml
    subject_map_yaml: Optional[Path]     # explicit override

    task_assets_dir: Path                # diapix assets directory

    overwrite: bool
    dry_run: bool
    preload_raw: bool
    line_freq_hz: float


def convert_diapix(cfg: DiapixConvertConfig) -> int:
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
        task_name="diapix",
        dataset_id=cfg.dataset_id,
        bids_subject=cfg.subject,
        session=cfg.session,
        private_root=cfg.private_root,
        subject_map_yaml=cfg.subject_map_yaml,
        task_assets_dir=cfg.task_assets_dir,
    )
    task = resolve_task(task_ctx)

    _ = convert_subject(cfg=core_cfg, task=task)
    return 0
