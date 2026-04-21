from pathlib import Path
from typing import Any, Optional, Sequence

import mne
import numpy as np
from mne_bids import BIDSPath

from dcap.bids.core.io import load_raw_brainvision
from dcap.bids.tasks._shared.trigger_alignment_qc import write_trigger_alignment_qc_json
from dcap.bids.tasks.base import BidsTask, PreparedEvents, RecordingUnit
from dcap.bids.tasks.sorciere.alignment import align_sorciere_raw
from dcap.bids.tasks.sorciere.models import SorciereAlignmentResult, SorciereRecordingUnit


_EVENT_ID = {
    "stimulus_start": 1,
    "stimulus_end": 2,
}


class SorciereTask(BidsTask):
    name: str = "sorciere"

    def __init__(
        self,
        *,
        bids_subject: str,
        dcap_id: str,
        session: Optional[str],
        reference_audio_path: Path,
        annotation_origin_in_reference_s: float = 3.0,
    ) -> None:
        self._bids_subject = str(bids_subject).strip()
        self._bids_subject_bare = _strip_sub_prefix(self._bids_subject)
        self._dcap_id = str(dcap_id).strip()
        self._session = session
        self._reference_audio_path = Path(reference_audio_path).expanduser().resolve()
        self._annotation_origin_in_reference_s = float(annotation_origin_in_reference_s)
        self._alignment_cache: dict[str, SorciereAlignmentResult] = {}

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        source_root = Path(source_root).expanduser().resolve()
        task_root = source_root
        vhdr_paths = sorted(task_root.glob("*.vhdr"))
        if len(vhdr_paths) == 0 and (source_root / "task-sorciere").exists():
            task_root = source_root / "task-sorciere"
            vhdr_paths = sorted(task_root.glob("*.vhdr"))
        if len(vhdr_paths) != 1:
            raise FileNotFoundError(f"Expected exactly one .vhdr file in {task_root}, found {len(vhdr_paths)}.")

        return [
            SorciereRecordingUnit(
                subject_bids=self._bids_subject_bare,
                dcap_id=self._dcap_id,
                session=self._session,
                run="01",
                vhdr_path=vhdr_paths[0],
            )
        ]

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        assert isinstance(unit, SorciereRecordingUnit)
        return load_raw_brainvision(unit.vhdr_path, preload=preload)

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        assert isinstance(unit, SorciereRecordingUnit)

        alignment = align_sorciere_raw(
            raw=raw,
            reference_audio_path=self._reference_audio_path,
            annotation_origin_in_reference_s=self._annotation_origin_in_reference_s,
        )
        self._alignment_cache[str(unit.vhdr_path)] = alignment

        sfreq = float(raw.info["sfreq"])
        stimulus_start_sample = int(np.rint(alignment.stimulus_start_s * sfreq))
        if stimulus_start_sample < 0:
            raise ValueError(
                f"Computed Sorciere stimulus start precedes recording: {alignment.stimulus_start_s:.6f} s."
            )
        if stimulus_start_sample >= raw.n_times:
            raise ValueError(
                f"Computed Sorciere stimulus start exceeds recording length: {alignment.stimulus_start_s:.6f} s."
            )
        if alignment.reference_duration_s is None:
            raise ValueError("Sorciere reference duration is required to compute stimulus_end.")

        stimulus_end_s = alignment.stimulus_start_s + alignment.reference_duration_s - alignment.annotation_origin_in_reference_s
        stimulus_end_sample = int(np.rint(stimulus_end_s * sfreq))
        if stimulus_end_sample <= stimulus_start_sample:
            raise ValueError(
                "Computed Sorciere stimulus end does not occur after stimulus start: "
                f"start={alignment.stimulus_start_s:.6f}s end={stimulus_end_s:.6f}s."
            )
        if stimulus_end_sample >= raw.n_times:
            stimulus_end_sample = int(raw.n_times - 1)

        events = np.array(
            [
                [stimulus_start_sample, 0, _EVENT_ID["stimulus_start"]],
                [stimulus_end_sample, 0, _EVENT_ID["stimulus_end"]],
            ],
            dtype=int,
        )
        return PreparedEvents(events=events, event_id=dict(_EVENT_ID))

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        assert isinstance(unit, SorciereRecordingUnit)
        if bids_path.fpath is None or not bids_path.fpath.exists():
            return
        alignment = self._alignment_cache.get(str(unit.vhdr_path))
        if alignment is None:
            return

        payload: dict[str, Any] = {
            "selected_description": alignment.selected_description,
            "selected_event_code": alignment.selected_event_code,
            "delay_s": alignment.delay_s,
            "stimulus_start_s": alignment.stimulus_start_s,
            "annotation_origin_in_reference_s": alignment.annotation_origin_in_reference_s,
            "matched_hits": alignment.matched_hits,
            "reference_duration_s": alignment.reference_duration_s,
            "candidate_count": alignment.candidate_count,
            "reference_audio_path": self._reference_audio_path,
            "reference_onsets_s": alignment.reference_onsets_s,
            "raw_onsets_s": alignment.raw_onsets_s,
        }

        _ = write_trigger_alignment_qc_json(
            bids_root=_infer_bids_root_from_written_file(bids_path.fpath),
            subject=unit.subject_bids,
            session=self._session,
            datatype=bids_path.datatype,
            filename_stem=bids_path.fpath.stem,
            payload=payload,
            dcap_version="0.0.0",
        )


def _strip_sub_prefix(bids_subject: str) -> str:
    s = str(bids_subject).strip()
    return s[len("sub-"):] if s.startswith("sub-") else s


def _infer_bids_root_from_written_file(written_file: Path) -> Path:
    parts = written_file.parts
    for idx, part in enumerate(parts):
        if part.startswith("sub-"):
            return Path(*parts[:idx])
    raise ValueError(f"Could not infer BIDS root from path: {written_file}")
