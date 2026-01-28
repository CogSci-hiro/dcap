# dcap/bids/tasks/diapix/task.py
# =============================================================================
#                                DIAPIX TASK
# =============================================================================

from pathlib import Path
from typing import Dict, Final, Optional, Sequence

import mne
import pandas as pd
from mne_bids import BIDSPath

from dcap.bids.core.events import PreparedEvents
from dcap.bids.core.transforms import (
    apply_line_frequency,
    drop_channels_defensively,
    rename_channels_defensively,
    set_channel_types_defensively,
)
from dcap.bids.tasks.diapix.audio import crop_and_normalize_audio
from dcap.bids.tasks.diapix.events import prepare_diapix_events
from dcap.bids.tasks.diapix.heuristics import ensure_vhdr_utf8
from dcap.bids.tasks.diapix.models import DiapixRecordingUnit
from dcap.bids.tasks.diapix.sidecars import build_task_sidecar_fields


LINE_FREQ_HZ: Final[float] = 50.0

# Your old script had per-subject/run trigger mapping hard-coded.
# Keep it here for now; you can move it to a TSV/YAML later.
TRIGGER_ID_BY_SUBJECT_RUN: Final[Dict[str, Dict[str, int]]] = {
    "BacJul": {"1": 10005, "2": 10004, "3": 10006, "4": 10005},
}


class DiapixTask:
    """
    Task adapter for Diapix.

    Implements the BidsTask contract expected by the core conversion engine.

    Usage example
    -------------
        task = DiapixTask(
            subject_bids="NicEle",
            audio_onsets_tsv=Path("audio_onsets.tsv"),
            stim_wav=Path("beeps.wav"),
            atlas_path=Path("elec2atlas.mat"),
        )
    """

    name: str = "diapix"

    def __init__(
        self,
        *,
        subject_bids: str,
        audio_onsets_tsv: Path,
        stim_wav: Path,
        atlas_path: Path,
    ) -> None:
        self._subject_bids = subject_bids
        self._audio_onsets_tsv = audio_onsets_tsv
        self._stim_wav = stim_wav
        self._atlas_path = atlas_path

    # -------------------------------------------------------------------------
    # Contract: discover
    # -------------------------------------------------------------------------

    def discover(self, source_root: Path) -> Sequence[DiapixRecordingUnit]:
        """
        Discover all Diapix runs in a flat subject directory.

        Expected filenames
        ------------------
        conversation_<RUN>.vhdr
        conversation_<RUN>.wav
        conversation_<RUN>.asf (optional)

        Usage example
        -------------
            units = task.discover(Path("/path/to/patient_dir"))
        """
        if not source_root.exists():
            raise FileNotFoundError(f"Missing source_root: {source_root}")

        vhdr_paths = sorted(source_root.glob("conversation_*.vhdr"))
        units: list[DiapixRecordingUnit] = []

        for vhdr_path in vhdr_paths:
            run = vhdr_path.stem.split("_")[-1]  # conversation_<RUN>
            wav_path = source_root / f"conversation_{run}.wav"
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing WAV for run={run}: {wav_path}")

            video_path = source_root / f"conversation_{run}.asf"
            video = video_path if video_path.exists() else None

            units.append(
                DiapixRecordingUnit(
                    subject_bids=self._subject_bids,
                    session=None,
                    run=run,
                    vhdr_path=vhdr_path,
                    wav_path=wav_path,
                    video_path=video,
                )
            )

        if len(units) == 0:
            raise FileNotFoundError(f"No conversation_*.vhdr files found in {source_root}")

        return units

    # -------------------------------------------------------------------------
    # Contract: load_raw
    # -------------------------------------------------------------------------

    def load_raw(self, unit: DiapixRecordingUnit, preload: bool) -> mne.io.BaseRaw:
        """
        Load raw and apply task-level transforms (but not writing).

        Usage example
        -------------
            raw = task.load_raw(unit, preload=True)
        """
        ensure_vhdr_utf8(unit.vhdr_path)

        raw = mne.io.read_raw_brainvision(unit.vhdr_path, preload=preload, verbose=False)

        # Core-transform wrappers (task decides inputs; core performs safe ops)
        apply_line_frequency(raw, line_freq_hz=LINE_FREQ_HZ)

        # Channel rename mapping from TSV (task policy)
        # Your old script used CHANNELS_FILE filtered by subject.
        # Here we assume you provide that TSV somewhere task-level (or via registry).
        # If you already have a core registry for this, swap this method to load from it.
        # For now: optional/no-op unless you wire the TSV in.
        # rename_channels_defensively(raw, mapping)

        # Channel typing (task policy, core safe setter)
        channel_types = {ch: "seeg" for ch in raw.ch_names}
        if "ECG" in raw.ch_names:
            channel_types["ECG"] = "ecg"
        set_channel_types_defensively(raw, channel_types)

        # Drop NULL electrodes (task policy)
        drop_channels_defensively(raw, pattern=r".*NULL.*")

        # Montage / coordinates are very project-specific.
        # Keep it in the task layer, but if you have a generic helper, call it here.
        raw = self._add_montage_from_elec2atlas(raw, atlas_path=self._atlas_path)

        return raw

    def _add_montage_from_elec2atlas(self, raw: mne.io.BaseRaw, *, atlas_path: Path) -> mne.io.BaseRaw:
        """
        Add monopolar montage using subject-specific elec2atlas.mat.

        Note
        ----
        This is intentionally task-local (project-specific recon format).

        Usage example
        -------------
            raw = self._add_montage_from_elec2atlas(raw, atlas_path=Path("elec2atlas.mat"))
        """
        # TODO: integrate your existing montage builder here.
        # e.g.:
        #   atlas = mat73.loadmat(atlas_path)
        #   coords = get_mni_mono_coordinates(atlas)
        #   montage, _, _ = get_montage(raw, coords, montage_type="monopolar")
        #   raw.set_montage(montage)
        #
        # Keeping as a stub so core integration is unblocked.
        return raw

    # -------------------------------------------------------------------------
    # Contract: prepare_events
    # -------------------------------------------------------------------------

    def prepare_events(self, raw: mne.io.BaseRaw, unit: DiapixRecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        """
        Create PreparedEvents for this unit.

        Usage example
        -------------
            prepared = task.prepare_events(raw, unit, bids_path)
        """
        trigger_id = self._get_trigger_id(subject_bids=unit.subject_bids, run=unit.run)

        prepared, raw_out = prepare_diapix_events(
            raw=raw,
            subject_bids=unit.subject_bids,
            run=unit.run,
            stim_wav=self._stim_wav,
            trigger_id=trigger_id,
        )

        # IMPORTANT: if prepare_diapix_events pads raw, core needs to write that padded raw.
        # The cleanest pattern: do padding inside load_raw/prepare_events BEFORE core writes.
        #
        # If your core converter passes the same raw instance onward, we can update in-place.
        # Otherwise, you may need to change core contract to allow prepare_events to return
        # (PreparedEvents, BaseRaw). But your handoff said the interface is frozen.
        #
        # So: keep padding out of prepare_events if your core doesn’t support raw replacement.
        # For now we assert no replacement happened (or you refactor padding earlier).
        if raw_out is not raw:
            raise RuntimeError(
                "prepare_events produced a new raw (padding). "
                "Move padding into load_raw(), or extend core contract carefully."
            )

        return prepared

    def _get_trigger_id(self, *, subject_bids: str, run: str) -> int:
        if subject_bids in TRIGGER_ID_BY_SUBJECT_RUN and run in TRIGGER_ID_BY_SUBJECT_RUN[subject_bids]:
            return TRIGGER_ID_BY_SUBJECT_RUN[subject_bids][run]
        raise KeyError(f"No trigger_id mapping for subject={subject_bids}, run={run}")

    # -------------------------------------------------------------------------
    # Contract: post_write
    # -------------------------------------------------------------------------

    def post_write(self, unit: DiapixRecordingUnit, bids_path: BIDSPath) -> None:
        """
        Copy/crop task artifacts after core writes iEEG.

        Policy
        ------
        - audio: crop + normalize into sub-<id>/audio/
        - video: copy into sub-<id>/video/ if present
        - sidecars: optional task fields merged next to iEEG recording (if core helper exists)

        Usage example
        -------------
            task.post_write(unit, bids_path)
        """
        # Audio
        audio_dir = bids_path.root / f"sub-{unit.subject_bids}" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_out = audio_dir / f"sub-{unit.subject_bids}_task-diapix_run-{unit.run}.wav"

        crop_and_normalize_audio(
            src_wav=unit.wav_path,
            dst_wav=audio_out,
            audio_onsets_tsv=self._audio_onsets_tsv,
            subject_bids=unit.subject_bids,
            run=unit.run,
        )

        # Video
        if unit.video_path is not None:
            video_dir = bids_path.root / f"sub-{unit.subject_bids}" / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_out = video_dir / f"sub-{unit.subject_bids}_task-diapix_run-{unit.run}.asf"
            video_out.write_bytes(unit.video_path.read_bytes())

        # Optional: sidecar merge/write (only if you have a core helper)
        # fields = build_task_sidecar_fields()
        # core.sidecars.merge_and_write(bids_path, fields)
