from pathlib import Path
import shutil
from typing import Optional, Sequence

import mne
from mne_bids import BIDSPath

from dcap.bids.core.bids_paths import build_bids_file_path
from dcap.bids.core.io import load_raw
from dcap.bids.tasks.base import BidsTask, RecordingUnit, discover_recording_units_from_task_layout
from dcap.bids.tasks.naming.events import prepare_naming_events


class NamingTask(BidsTask):
    """
    Picture naming task adapter.

    Current implementation assumptions
    ----------------------------------
    - Event semantics are inferred and should be considered provisional.
    - Discovery uses `task-naming/run-*` subfolders with exactly one EEG file per run.
    """

    name: str = "naming"

    def __init__(
        self,
        *,
        bids_subject: str,
        dcap_id: Optional[str],
        session: Optional[str],
    ) -> None:
        self._bids_subject = str(bids_subject).strip()
        self._dcap_id = None if dcap_id is None else str(dcap_id).strip()
        self._session = session

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        return discover_recording_units_from_task_layout(
            source_root=source_root,
            bids_subject=self._bids_subject,
            task_name=self.name,
            source_subject_id=self._dcap_id,
        )

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        raw_path = Path(unit.raw_path)
        suffix = raw_path.suffix.lower()

        if suffix == ".vhdr":
            return load_raw(raw_path, raw_format="brainvision", preload=preload)
        if suffix == ".edf":
            return load_raw(raw_path, raw_format="edf", preload=preload)
        if suffix == ".fif":
            return load_raw(raw_path, raw_format="fif", preload=preload)

        raise ValueError(f"Unsupported naming raw format: {raw_path}")

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath):
        _ = (unit, bids_path)  # naming events depend only on raw annotations for now
        return prepare_naming_events(raw)

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        if unit.audio_path is not None:
            audio_out = build_bids_file_path(
                bids_root=Path(bids_path.root),
                subject=self._bids_subject,
                session=self._session,
                task=self.name,
                datatype="beh",
                run=unit.run,
                suffix="audio",
                extension=unit.audio_path.suffix.lower(),
            )
            audio_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(unit.audio_path, audio_out)
        return None
