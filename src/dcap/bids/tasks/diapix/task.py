# dcap/bids/tasks/diapix/task.py
# =============================================================================
#                                DIAPIX TASK
# =============================================================================

from pathlib import Path
import shutil
from typing import Any, Optional, Sequence

import mne
from mne_bids import BIDSPath

from dcap.bids.core.bids_paths import build_bids_file_path
from dcap.bids.core.io import load_raw
from dcap.bids.tasks.base import BidsTask, PreparedEvents, RecordingUnit
from dcap.bids.tasks.base import discover_recording_units_from_task_layout
from dcap.bids.tasks.diapix.models import DiapixTiming
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
            atlas_path: Optional[Path] = None,
            timing: Optional[DiapixTiming] = None,
    ) -> None:
        self._bids_subject = str(bids_subject).strip()
        self._bids_subject_bare = _strip_sub_prefix(self._bids_subject)

        self._dcap_id = str(dcap_id).strip()
        self._session = session

        self._audio_onsets_tsv = Path(audio_onsets_tsv).expanduser().resolve()
        self._stim_wav = Path(stim_wav).expanduser().resolve()
        self._atlas_path = None if atlas_path is None else Path(atlas_path).expanduser().resolve()

        self._timing = timing if timing is not None else DiapixTiming()

        self._trigger_alignment_qc_cache: dict[str, dict[str, Any]] = {}

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        return discover_recording_units_from_task_layout(
            source_root=source_root,
            bids_subject=self._bids_subject,
            task_name=self.name,
            source_subject_id=self._dcap_id,
        )

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        suffix = unit.eeg_path.suffix.lower()
        if suffix == ".vhdr":
            ensure_vhdr_utf8(unit.eeg_path)
            raw = mne.io.read_raw_brainvision(unit.eeg_path, preload=preload, verbose=False)
        elif suffix == ".edf":
            raw = load_raw(unit.eeg_path, raw_format="edf", preload=preload)
        elif suffix == ".fif":
            raw = load_raw(unit.eeg_path, raw_format="fif", preload=preload)
        else:
            raise ValueError(f"Unsupported Diapix raw format: {unit.eeg_path}")

        # TODO: add montage from self._atlas_path (task-specific)
        # TODO: apply channel renaming/types (task-specific policy)

        # IMPORTANT:
        # If you ever need to pad raw to avoid negative conversation_start, do it here,
        # not in prepare_events, because core writes the raw object it gets from here.

        return raw

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        cache_key = str(unit.eeg_path)

        trigger_id = get_trigger_id(self._dcap_id, run=unit.run)

        prepared, alignment = prepare_diapix_events(
            raw=raw,
            subject_bids=self._bids_subject_bare,
            run=unit.run,
            stim_wav=self._stim_wav,
            trigger_id=trigger_id,
        )
        self._trigger_alignment_qc_cache[cache_key] = alignment

        return prepared

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        # Audio
        if unit.audio_path is not None:
            audio_out_path = build_bids_file_path(
                bids_root=Path(bids_path.root),
                subject=self._bids_subject,
                session=self._session,
                task=self.name,
                datatype="beh",
                run=unit.run,
                suffix="audio",
                extension=".wav",
            )
            audio_out_path.parent.mkdir(parents=True, exist_ok=True)

            crop_and_normalize_audio(
                src_wav=unit.audio_path,
                dst_wav=audio_out_path,
                audio_onsets_tsv=self._audio_onsets_tsv,
                dcap_id=self._dcap_id,
                run=str(unit.run),
                duration_s=self._timing.conversation_duration_s,
            )

        # Video
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

        # Trigger alignment JSON creation
        cache_key = str(unit.eeg_path)
        payload = self._trigger_alignment_qc_cache.get(cache_key)
        if payload is None:
            return

        _ = write_trigger_alignment_qc_json(
            bids_root=infer_bids_root_from_written_file(bids_path.fpath),
            subject=self._bids_subject_bare,
            session=self._session,
            datatype=bids_path.datatype,
            filename_stem=bids_path.fpath.stem,
            payload=payload,
            dcap_version="0.0.0",  # replace with your version getter
        )


def _strip_sub_prefix(bids_subject: str) -> str:
    s = str(bids_subject).strip()
    return s[len("sub-"):] if s.startswith("sub-") else s


def get_trigger_id(dcap_id: str, run: str) -> int:
    run_key = str(run).strip()
    normalized_candidates = [run_key]
    if run_key.isdigit():
        normalized_candidates.append(str(int(run_key)))

    try:
        per_subject = _TRIGGER_ID_MAP[dcap_id]
    except KeyError as exc:
        raise KeyError(
            f"No trigger_id mapping defined for Diapix subject dcap_id={dcap_id}"
        ) from exc

    for candidate in normalized_candidates:
        if candidate in per_subject:
            return per_subject[candidate]

    raise KeyError(
        f"No trigger_id defined for Diapix "
        f"(dcap_id={dcap_id}, run={run})"
    )


def infer_bids_root_from_written_file(written_file: Path) -> Path:
    """Infer BIDS root from a file path like <bids_root>/sub-XXX/..."""
    parts = written_file.parts
    for idx, part in enumerate(parts):
        if part.startswith("sub-"):
            return Path(*parts[:idx])  # everything before sub-XXX
    raise ValueError(f"Could not infer BIDS root from path: {written_file}")
