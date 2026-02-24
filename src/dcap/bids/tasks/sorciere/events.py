from pathlib import Path
from typing import Any, Final, Tuple

import mne
import numpy as np
from scipy.io import wavfile

from dcap.bids.core.events import PreparedEvents
from dcap.bids.core.sync import (
    estimate_delay_by_onset_hits,
    match_interval_sequences_delay_mad,
    onsets_and_intervals_from_samples,
)


DEFAULT_EVENT_ID: Final[dict[str, int]] = {
    "stimulus_start": 1,
    "stimulus_end": 2,
}


def prepare_sorciere_events(
    *,
    raw: mne.io.BaseRaw,
    stim_wav: Path,
    trigger_id: int,
    stimulus_start_delay_s: float = 0.0,
    max_fallback_offset: int = 30,
) -> Tuple[PreparedEvents, dict[str, Any]]:
    """
    Create Sorciere stimulus start/end events by aligning raw triggers to the reference WAV.
    """
    events, _ = mne.events_from_annotations(raw, verbose=False)
    sfreq = float(raw.info["sfreq"])

    trigger_events = events[events[:, 2] == int(trigger_id)]
    if trigger_events.size == 0:
        raise ValueError(f"No triggers found for trigger_id={trigger_id}.")

    delay_s, wav_onsets_s, raw_onsets_s, stim_duration_s = _compute_delay_seconds_and_duration(
        raw_trigger_events=trigger_events,
        sfreq=sfreq,
        stim_wav=stim_wav,
        max_fallback_offset=max_fallback_offset,
    )

    start_s = float(delay_s + stimulus_start_delay_s)
    pad_required_s = float(max(0.0, -start_s))
    start_s_clamped = max(0.0, start_s)
    start_sample = int(start_s_clamped * sfreq)

    planned_end_sample = start_sample + int(stim_duration_s * sfreq)
    file_end_sample = int(raw.n_times - 1)
    end_sample = min(planned_end_sample, file_end_sample)

    out_events = np.zeros((2, 3), dtype=int)
    out_events[0, 0] = start_sample
    out_events[0, 2] = DEFAULT_EVENT_ID["stimulus_start"]
    out_events[1, 0] = end_sample
    out_events[1, 2] = DEFAULT_EVENT_ID["stimulus_end"]

    prepared = PreparedEvents(events=out_events, event_id=DEFAULT_EVENT_ID.copy())
    alignment: dict[str, Any] = {
        "delay_s": float(delay_s),
        "stimulus_start_s": float(start_s),
        "stimulus_duration_s": float(stim_duration_s),
        "pad_required_s": float(pad_required_s),
        "wav_onsets_s": wav_onsets_s,
        "raw_onsets_s": raw_onsets_s,
        "stimulus_end_sample_planned": int(planned_end_sample),
        "stimulus_end_sample_actual": int(end_sample),
        "stimulus_window_is_full": bool(end_sample == planned_end_sample),
    }
    return prepared, alignment


def _compute_delay_seconds_and_duration(
    *,
    raw_trigger_events: np.ndarray,
    sfreq: float,
    stim_wav: Path,
    max_fallback_offset: int,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    wav_onsets, wav_intervals, stim_duration_s = _get_wav_trigger_onsets_and_intervals(stim_wav=stim_wav)
    raw_onsets, raw_intervals = onsets_and_intervals_from_samples(
        sample_indices=raw_trigger_events[:, 0],
        sfreq=sfreq,
    )

    tolerance = 1.0 / sfreq
    try:
        delay_s = match_interval_sequences_delay_mad(
            reference_onsets_s=wav_onsets,
            reference_intervals_s=wav_intervals,
            target_onsets_s=raw_onsets,
            target_intervals_s=raw_intervals,
            tolerance_s=tolerance,
        )
    except ValueError:
        delay_s = estimate_delay_by_onset_hits(
            reference_onsets_s=wav_onsets,
            target_onsets_s=raw_onsets,
            max_offset=max_fallback_offset,
            match_tol_s=max(2.0 * tolerance, 0.01),
        )

    return float(delay_s), wav_onsets, raw_onsets, float(stim_duration_s)


def _get_wav_trigger_onsets_and_intervals(
    *,
    stim_wav: Path,
    threshold: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    sr, wav = wavfile.read(stim_wav)
    if wav.ndim != 2 or wav.shape[1] < 2:
        raise ValueError(f"Expected at least 2-channel WAV for trigger extraction: {stim_wav}")

    triggers = wav[:, 1].astype(float)
    binary = np.zeros_like(triggers, dtype=int)
    binary[triggers > threshold] = 1

    edges = np.diff(binary)
    edges[edges < 0] = 0
    onset_samples = np.where(edges == 1)[0]

    if onset_samples.size < 2:
        raise ValueError(f"Not enough trigger onsets in stim WAV: {stim_wav}")

    onsets_s = onset_samples.astype(float) / float(sr)
    intervals_s = np.diff(onsets_s)
    stim_duration_s = float(wav.shape[0]) / float(sr)

    # Keep onset/interval lengths aligned for delay matching.
    return onsets_s[:-1], intervals_s, stim_duration_s

