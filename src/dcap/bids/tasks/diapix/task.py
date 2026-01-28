from dcap.bids.tasks.base import BidsTask, RecordingUnit, PreparedEvents
from dcap.bids.core.sync import estimate_delay_seconds
from dcap.bids.tasks.diapix.audio import crop_and_write_audio
from dcap.bids.tasks.diapix.events import build_conversation_events


class DiapixTask:
    name = "diapix"

    def discover(self, source_root: Path) -> list[RecordingUnit]:
        return discover_diapix_units(source_root)

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        raw = load_brainvision(unit.raw_path, preload=preload)
        apply_channel_policy(raw)
        apply_montage(raw)
        return raw

    def prepare_events(self, raw, unit, bids_path) -> PreparedEvents:
        triggers = extract_trigger_events(raw, unit)
        delay_s = estimate_delay_seconds(
            raw_triggers=triggers,
            sfreq=raw.info["sfreq"],
            stim_wav_path=self.stim_wav,
        )
        events, event_id = build_conversation_events(raw, delay_s, self.timing)
        return PreparedEvents(events=events, event_id=event_id)

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        if unit.audio_path is not None:
            crop_and_write_audio(
                src_wav=unit.audio_path,
                dst_wav=self._audio_out_path(bids_path),
                onset_s=self._audio_onset(unit),
                duration_s=self.timing.conversation_duration_s,
            )

        if unit.video_path is not None:
            copy_video(unit.video_path, bids_path)
