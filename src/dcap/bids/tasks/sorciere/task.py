from pathlib import Path
import shutil
from typing import Any, Optional, Sequence

import mne
from mne_bids import BIDSPath

from dcap.bids.core.bids_paths import build_bids_file_path
from dcap.bids.core.io import load_raw
from dcap.bids.core.events import PreparedEvents
from dcap.bids.tasks.base import BidsTask, RecordingUnit, discover_recording_units_from_task_layout
from dcap.bids.tasks.sorciere.events import prepare_sorciere_events
from dcap.bids.tasks.sorciere.models import SorciereTiming


class SorciereTask(BidsTask):
    """
    Sorciere (passive listening) task adapter.

    Assumptions
    -----------
    - Runs are discovered from `task-sorciere/run-*` subfolders.
    - A single shared stimulus WAV is provided in `task_assets_dir`.
    - Trigger-train synchronization uses the same random-train mechanism as Diapix.
    """

    name: str = "sorciere"

    def __init__(
        self,
        *,
        bids_subject: str,
        dcap_id: str,
        session: Optional[str],
        stim_wav: Path,
        trigger_id: int = 10004,
        timing: Optional[SorciereTiming] = None,
    ) -> None:
        self._bids_subject = str(bids_subject).strip()
        self._dcap_id = str(dcap_id).strip()
        self._session = session
        self._stim_wav = Path(stim_wav).expanduser().resolve()
        self._trigger_id = int(trigger_id)
        self._timing = timing if timing is not None else SorciereTiming()

        self._alignment_cache: dict[str, dict[str, Any]] = {}
        self._stimulus_copied = False

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
        raise ValueError(f"Unsupported Sorciere raw format: {raw_path}")

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        prepared, alignment = prepare_sorciere_events(
            raw=raw,
            stim_wav=self._stim_wav,
            trigger_id=self._trigger_id,
            stimulus_start_delay_s=self._timing.stimulus_start_delay_s,
        )
        self._alignment_cache[str(unit.eeg_path)] = alignment
        return prepared

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

        if unit.video_path is not None:
            video_out = build_bids_file_path(
                bids_root=Path(bids_path.root),
                subject=self._bids_subject,
                session=self._session,
                task=self.name,
                datatype="beh",
                run=unit.run,
                suffix="video",
                extension=unit.video_path.suffix.lower(),
            )
            video_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(unit.video_path, video_out)

        if self._stimulus_copied:
            return

        stimuli_dir = Path(bids_path.root) / "stimuli"
        stimuli_dir.mkdir(parents=True, exist_ok=True)
        dst = stimuli_dir / self._stim_wav.name

        if not dst.exists():
            shutil.copy2(self._stim_wav, dst)

        self._stimulus_copied = True
