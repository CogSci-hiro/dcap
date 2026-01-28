# =============================================================================
#                        BIDS Core: Task-agnostic converter
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mne_bids import write_raw_bids

from dcap.bids.core.bids_paths import build_bids_path, normalize_label
from dcap.bids.core.transforms import apply_line_frequency
from dcap.bids.tasks.base import BidsTask


@dataclass(frozen=True)
class ConvertConfig:
    source_root: Path
    bids_root: Path
    subject: str
    session: Optional[str]
    datatype: str
    task: str
    overwrite: bool
    dry_run: bool
    line_freq: float
    preload_raw: bool


def convert_subject(cfg: ConvertConfig, task_impl: BidsTask) -> None:
    units = task_impl.discover(cfg.source_root)

    subject = normalize_label(cfg.subject, "sub")
    session = normalize_label(cfg.session, "ses") if cfg.session is not None else None

    for unit in units:
        bids_path = build_bids_path(
            bids_root=cfg.bids_root,
            subject=subject,
            session=session,
            task=task_impl.name,  # authoritative
            datatype=cfg.datatype,
            run=unit.run,
        )

        raw = task_impl.load_raw(unit, preload=cfg.preload_raw)
        apply_line_frequency(raw, cfg.line_freq)

        prepared = task_impl.prepare_events(raw=raw, unit=unit, bids_path=bids_path)

        raw.set_annotations(None)

        if cfg.dry_run:
            continue

        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            events=prepared.events,
            event_id=prepared.event_id,
            overwrite=cfg.overwrite,
            format="auto",
            allow_preload=cfg.preload_raw,
            anonymize=None,
            verbose=False,
        )

        task_impl.post_write(unit=unit, bids_path=bids_path)
