# dcap/bids/tasks/diapix/task.py
# =============================================================================
#                                DIAPIX TASK
# =============================================================================

from pathlib import Path
from typing import Any, Optional, Sequence

import mne
from mne_bids import BIDSPath
import numpy as np

from dcap.bids.tasks.base import BidsTask, PreparedEvents, RecordingUnit
from dcap.bids.tasks.diapix.models import DiapixRecordingUnit, DiapixTiming
from dcap.bids.tasks.diapix.heuristics import ensure_vhdr_utf8
from dcap.bids.tasks.diapix.audio import crop_and_normalize_audio
from dcap.bids.tasks.diapix.events import prepare_diapix_events
from dcap.bids.tasks.diapix.triggers import _TRIGGER_ID_MAP
from dcap.bids.tasks._shared.trigger_alignment_qc import (  # noqa
    write_trigger_alignment_qc_json,
)


class DiapixTask(BidsTask):
    """
    Diapix task adapter.

    Notes
    -----
    - `dcap_id` is private and must never be written into BIDS outputs.
    - `bids_subject` is BIDS-facing and may be "sub-001" or "001".

    Usage example
    -------------
        task = DiapixTask(
            bids_subject="sub-001",
            dcap_id="Nic-Ele",
            session=None,
            audio_onsets_tsv=Path("audio_onsets.tsv"),
            stim_wav=Path("beeps.wav"),
            atlas_path=Path("elec2atlas.mat"),
        )
    """

    name: str = "diapix"

    def __init__(
            self,
            *,
            bids_subject: str,
            dcap_id: str,
            session: Optional[str],
            audio_onsets_tsv: Path,
            stim_wav: Path,
            atlas_path: Path,
            timing: Optional[DiapixTiming] = None,
    ) -> None:
        self._bids_subject = str(bids_subject).strip()
        self._bids_subject_bare = _strip_sub_prefix(self._bids_subject)

        self._dcap_id = str(dcap_id).strip()
        self._session = session

        self._audio_onsets_tsv = Path(audio_onsets_tsv).expanduser().resolve()
        self._stim_wav = Path(stim_wav).expanduser().resolve()
        self._atlas_path = Path(atlas_path).expanduser().resolve()

        self._timing = timing if timing is not None else DiapixTiming()

        self._trigger_alignment_qc_cache: dict[str, dict[str, Any]] = {}

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        source_root = Path(source_root).expanduser().resolve()
        vhdr_paths = sorted(source_root.glob("conversation_*.vhdr"))
        if len(vhdr_paths) == 0:
            raise FileNotFoundError(f"No conversation_*.vhdr files found in {source_root}")

        units: list[DiapixRecordingUnit] = []
        for vhdr_path in vhdr_paths:
            run = vhdr_path.stem.split("_")[-1]
            wav_path = source_root / f"conversation_{run}.wav"
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing WAV for run={run}: {wav_path}")

            video_path = source_root / f"conversation_{run}.asf"
            units.append(
                DiapixRecordingUnit(
                    subject_bids=self._bids_subject_bare,
                    dcap_id=self._dcap_id,
                    session=self._session,
                    run=run,
                    vhdr_path=vhdr_path,
                    wav_path=wav_path,
                    video_path=video_path if video_path.exists() else None,
                )
            )

        return units

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        assert isinstance(unit, DiapixRecordingUnit)
        ensure_vhdr_utf8(unit.vhdr_path)

        raw = mne.io.read_raw_brainvision(unit.vhdr_path, preload=preload, verbose=False)

        # TODO: add montage from self._atlas_path (task-specific)
        # TODO: apply channel renaming/types (task-specific policy)

        # IMPORTANT:
        # If you ever need to pad raw to avoid negative conversation_start, do it here,
        # not in prepare_events, because core writes the raw object it gets from here.

        return raw

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        assert isinstance(unit, DiapixRecordingUnit)

        cache_key = str(unit.vhdr_path)

        trigger_id = get_trigger_id(self._dcap_id, run=unit.run)

        prepared, alignment = prepare_diapix_events(
            raw=raw,
            subject_bids=unit.subject_bids,
            run=unit.run,
            stim_wav=self._stim_wav,
            trigger_id=trigger_id,
        )
        self._trigger_alignment_qc_cache[cache_key] = alignment

        return prepared

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        assert isinstance(unit, DiapixRecordingUnit)

        # Audio
        audio_dir = bids_path.root / f"sub-{unit.subject_bids}" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_out_path = audio_dir / f"sub-{unit.subject_bids}_task-diapix_run-{unit.run}.wav"

        crop_and_normalize_audio(
            src_wav=unit.wav_path,
            dst_wav=audio_out_path,
            audio_onsets_tsv=self._audio_onsets_tsv,
            dcap_id=self._dcap_id,
            run=str(unit.run),
            duration_s=self._timing.conversation_duration_s,
        )

        # Video
        if unit.video_path is not None:
            video_dir = bids_path.root / f"sub-{unit.subject_bids}" / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_out = video_dir / f"sub-{unit.subject_bids}_task-diapix_run-{unit.run}.asf"
            video_out.write_bytes(unit.video_path.read_bytes())

        # Trigger alignment JSON creation
        cache_key = str(unit.vhdr_path)
        payload = self._trigger_alignment_qc_cache.get(cache_key)
        if payload is None:
            return

        _ = write_trigger_alignment_qc_json(
            bids_root=infer_bids_root_from_written_file(bids_path.fpath),
            subject=unit.subject_bids,
            session=getattr(unit, "session_bids", None),
            datatype=bids_path.datatype,
            filename_stem=bids_path.fpath.stem,
            payload=payload,
            dcap_version="0.0.0",  # replace with your version getter
        )


def _strip_sub_prefix(bids_subject: str) -> str:
    s = str(bids_subject).strip()
    return s[len("sub-"):] if s.startswith("sub-") else s


def get_trigger_id(dcap_id: str, run: str) -> int:
    try:
        return _TRIGGER_ID_MAP[dcap_id][run]
    except KeyError as exc:
        raise KeyError(
            f"No trigger_id defined for Diapix "
            f"(dcap_id={dcap_id}, run={run})"
        ) from exc


def infer_bids_root_from_written_file(written_file: Path) -> Path:
    """Infer BIDS root from a file path like <bids_root>/sub-XXX/..."""
    parts = written_file.parts
    for idx, part in enumerate(parts):
        if part.startswith("sub-"):
            return Path(*parts[:idx])  # everything before sub-XXX
    raise ValueError(f"Could not infer BIDS root from path: {written_file}")


