from pathlib import Path
from shutil import copy2
from typing import Any, Optional, Sequence

import mne
import numpy as np
from mne_bids import BIDSPath

from dcap.bids.core.events import make_events
from dcap.bids.core.io import load_raw_brainvision
from dcap.bids.tasks.base import BidsTask, PreparedEvents, RecordingUnit
from dcap.bids.tasks.diapix.task import infer_bids_root_from_written_file
from dcap.bids.tasks.naming.models import NamingRecordingUnit, NamingTrial
from dcap.bids.tasks.naming.presentation import (
    build_naming_alignment,
    infer_sequence_id_from_log_dir,
    make_bids_event_rows,
    write_alignment_summary_json,
    write_events_tsv,
    write_stimulus_lookup_tsv,
)


_EVENT_ID = {
    "recording_onset": 1,
    "picture_onset": 2,
    "isi_onset": 3,
    "pause_onset": 4,
    "button_press": 5,
    "run_start": 6,
}


class NamingTask(BidsTask):
    name: str = "naming"

    def __init__(
        self,
        *,
        bids_subject: str,
        dcap_id: str,
        session: Optional[str],
        presentation_root: Path,
    ) -> None:
        self._bids_subject = str(bids_subject).strip()
        self._bids_subject_bare = _strip_sub_prefix(self._bids_subject)
        self._dcap_id = str(dcap_id).strip()
        self._session = session
        self._presentation_root = Path(presentation_root).expanduser().resolve()
        self._cache: dict[str, dict[str, Any]] = {}

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        source_root = Path(source_root).expanduser().resolve()
        task_root = source_root
        vhdr_paths = sorted(task_root.glob("*.vhdr"))
        if len(vhdr_paths) == 0 and (source_root / "task-naming").exists():
            task_root = source_root / "task-naming"
            vhdr_paths = sorted(task_root.glob("*.vhdr"))
        if len(vhdr_paths) != 1:
            raise FileNotFoundError(f"Expected exactly one .vhdr file in {task_root}, found {len(vhdr_paths)}.")

        vhdr_path = vhdr_paths[0]
        vmrk_path = task_root / Path(vhdr_path).with_suffix(".vmrk").name
        if not vmrk_path.exists():
            raise FileNotFoundError(f"Missing .vmrk alongside {vhdr_path}: {vmrk_path}")

        log_dirs = sorted(path for path in task_root.glob("seq_*") if path.is_dir())
        if len(log_dirs) != 1:
            raise FileNotFoundError(f"Expected exactly one seq_* logfile directory in {task_root}, found {len(log_dirs)}.")
        log_dir = log_dirs[0]

        log_paths = sorted(log_dir.glob("*.log"))
        if len(log_paths) != 1:
            raise FileNotFoundError(f"Expected exactly one .log file in {log_dir}, found {len(log_paths)}.")
        log_path = log_paths[0]

        sequence_id = infer_sequence_id_from_log_dir(log_dir)
        return [
            NamingRecordingUnit(
                subject_bids=self._bids_subject_bare,
                dcap_id=self._dcap_id,
                session=self._session,
                run="1",
                vhdr_path=vhdr_path,
                vmrk_path=vmrk_path,
                log_dir=log_dir,
                log_path=log_path,
                sequence_id=sequence_id,
            )
        ]

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        assert isinstance(unit, NamingRecordingUnit)
        return load_raw_brainvision(unit.vhdr_path, preload=preload)

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        assert isinstance(unit, NamingRecordingUnit)
        trials, aux_events, summary = build_naming_alignment(
            presentation_root=self._presentation_root,
            log_dir=unit.log_dir,
            log_path=unit.log_path,
            vmrk_path=unit.vmrk_path,
            sfreq=float(raw.info["sfreq"]),
        )
        event_rows = make_bids_event_rows(trials=trials, aux_events=aux_events, sfreq=float(raw.info["sfreq"]))

        self._cache[str(unit.vhdr_path)] = {
            "trials": trials,
            "event_rows": event_rows,
            "summary": summary,
        }

        events = np.array(
            [
                [trial.recording_sample, 0, _EVENT_ID["recording_onset"]]
                for trial in trials
            ]
            + [
                [trial.picture_sample, 0, _EVENT_ID["picture_onset"]]
                for trial in trials
            ]
            + [
                [trial.isi_sample, 0, _EVENT_ID["isi_onset"]]
                for trial in trials
            ],
            dtype=int,
        )
        events = events[np.argsort(events[:, 0])]
        return make_events(events, _EVENT_ID)

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        assert isinstance(unit, NamingRecordingUnit)
        cached = self._cache.get(str(unit.vhdr_path))
        if cached is None:
            return

        trials: list[NamingTrial] = cached["trials"]
        event_rows: list[dict[str, object]] = cached["event_rows"]
        summary = cached["summary"]

        audio_dir = bids_path.root / f"sub-{unit.subject_bids}" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        response_audio_map: dict[int, str] = {}
        for trial in trials:
            if trial.response_audio_path is None:
                continue
            copied_name = (
                f"sub-{unit.subject_bids}_task-naming_run-{unit.run}_trial-{trial.trial_index:03d}_response.wav"
            )
            copied_path = audio_dir / copied_name
            copy2(trial.response_audio_path, copied_path)
            response_audio_map[trial.trial_index] = f"audio/{copied_name}"

        for row in event_rows:
            trial_index = row.get("trial_index")
            if isinstance(trial_index, int):
                row["response_audio_file"] = response_audio_map.get(trial_index, "")

        base_dir = bids_path.directory
        base_name = bids_path.basename
        if base_name is None:
            raise ValueError(f"Could not determine BIDS basename for naming events: {bids_path}")
        events_path = base_dir / f"{base_name}_events.tsv"
        write_events_tsv(events_path, event_rows)

        bids_root = infer_bids_root_from_written_file(bids_path.fpath)
        deriv_dir = bids_root / "derivatives" / "dcap" / f"sub-{unit.subject_bids}" / "naming"
        write_stimulus_lookup_tsv(
            deriv_dir / f"sub-{unit.subject_bids}_task-naming_run-{unit.run}_stimulus_lookup.tsv",
            trials,
        )
        write_alignment_summary_json(
            deriv_dir / f"sub-{unit.subject_bids}_task-naming_run-{unit.run}_alignment_summary.json",
            summary,
        )


def _strip_sub_prefix(bids_subject: str) -> str:
    s = str(bids_subject).strip()
    return s[len("sub-"):] if s.startswith("sub-") else s
