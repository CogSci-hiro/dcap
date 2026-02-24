# dcap/bids/tasks/diapix/events.py
# =============================================================================
#                               DIAPIX EVENTS
# =============================================================================

from pathlib import Path
from typing import Any, Dict, Final, Tuple

import mne
import numpy as np
from scipy.io import wavfile

from dcap.bids.core.events import PreparedEvents
from dcap.bids.core.sync import (
    count_onset_matches as _core_count_onset_matches,
    estimate_delay_by_onset_hits as _core_estimate_delay_by_onset_hits,
    match_interval_sequences_delay_mad as _core_match_interval_sequences_delay_mad,
)


LINE_FREQ_HZ: Final[float] = 50.0  # France
START_DELAY_S: Final[float] = 4.0
CONVERSATION_DURATION_S: Final[float] = 4.0 * 60.0

DEFAULT_EVENTS_DICT: Final[Dict[str, int]] = {
    "conversation_start": 1,
    "conversation_end": 2,
}


def prepare_diapix_events(
    *,
    raw: mne.io.BaseRaw,
    subject_bids: str,
    run: str,
    stim_wav: Path,
    trigger_id: int,
) -> Tuple[PreparedEvents, dict[str, Any]]:
    """
    Create Diapix events for BIDS writing.

    Steps
    -----
    1) read trigger onsets from raw annotations
    2) compute delay between stim WAV and raw trigger train
    3) create conversation_start/end samples
    4) if needed, pad the raw at the beginning to avoid negative start indices

    Parameters
    ----------
    raw
        Loaded BrainVision raw.
    subject_bids
        BIDS subject label without "sub-".
    run
        Run number.
    stim_wav
        Reference WAV containing the trigger train.
    trigger_id
        Annotation event code for the trigger channel.

    Returns
    -------
    prepared_events
        PreparedEvents container (events + event_id).
    raw_out
        Possibly padded raw.

    Usage example
    -------------
        prepared, raw = prepare_diapix_events(
            raw=raw,
            subject_bids="NicEle",
            run="1",
            stim_wav=Path("beeps.wav"),
            trigger_id=10005,
        )
    """
    orig_events, orig_id = mne.events_from_annotations(raw, verbose=False)
    sfreq = float(raw.info["sfreq"])

    trigger_events = orig_events[orig_events[:, 2] == trigger_id]
    if trigger_events.size == 0:
        raise ValueError(
            f"No triggers found for trigger_id={trigger_id} (subject={subject_bids}, run={run})."
        )

    delay_s, wav_onsets_s, raw_onsets_s = _compute_delay_seconds(
        raw_trigger_events=trigger_events,
        sfreq=sfreq,
        stim_wav=stim_wav,
    )

    conversation_start_s = float(START_DELAY_S + delay_s)

    # How much data is missing *before* the recording starts (if any)
    pad_required_s = float(max(0.0, -conversation_start_s))

    # We are NOT allowed to pad raw here (core contract), so we clamp events to valid range.
    conversation_start_s_clamped = max(0.0, conversation_start_s)
    conversation_start_sample = int(conversation_start_s_clamped * sfreq)

    planned_conversation_end_sample = conversation_start_sample + int(CONVERSATION_DURATION_S * sfreq)
    file_end_sample = int(raw.n_times - 1)
    conversation_end_sample = min(planned_conversation_end_sample, file_end_sample)

    events = np.zeros((2, 3), dtype=int)
    events[0, 0] = conversation_start_sample
    events[0, 2] = DEFAULT_EVENTS_DICT["conversation_start"]
    events[1, 0] = conversation_end_sample
    events[1, 2] = DEFAULT_EVENTS_DICT["conversation_end"]

    event_id = {k: v for k, v in DEFAULT_EVENTS_DICT.items()}
    prepared = PreparedEvents(events=events, event_id=event_id)

    alignment: dict[str, Any] = {
        "delay_s": float(delay_s),
        "conversation_start_s": float(conversation_start_s),
        "pad_required_s": float(pad_required_s),
        "wav_onsets_s": wav_onsets_s,  # np.ndarray
        "raw_onsets_s": raw_onsets_s,  # np.ndarray
        "conversation_end_sample_planned": int(planned_conversation_end_sample),
        "conversation_end_sample_actual": int(conversation_end_sample),
        "conversation_window_is_full": bool(conversation_end_sample == planned_conversation_end_sample),
    }
    return prepared, alignment


def _pad_raw_at_start(raw: mne.io.BaseRaw, *, pad_samples: int) -> mne.io.BaseRaw:
    data = raw.get_data()  # already loaded by task.load_raw(preload=True) in most cases
    padded = np.zeros((data.shape[0], data.shape[1] + pad_samples), dtype=data.dtype)
    padded[:, pad_samples:] = data
    return mne.io.RawArray(padded, raw.info, verbose=False)


def _get_wav_trigger_onsets_and_intervals(*, stim_wav: Path, threshold: float = 10_000.0) -> Tuple[np.ndarray, np.ndarray]:
    sr, wav = wavfile.read(stim_wav)
    if wav.ndim != 2 or wav.shape[1] < 2:
        raise ValueError(f"Expected at least 2-channel WAV for trigger extraction: {stim_wav}")

    triggers = wav[:, 1].astype(float)

    binary = np.zeros_like(triggers, dtype=int)
    binary[triggers > threshold] = 1

    edges = np.diff(binary)
    edges[edges < 0] = 0

    onset_samples = np.where(edges == 1)[0]
    onsets_s = onset_samples / float(sr)

    intervals_s = np.diff(onsets_s)
    return onsets_s[:-1], intervals_s


def _get_raw_trigger_onsets_and_intervals(*, raw_trigger_events: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    onsets_s = raw_trigger_events[:, 0].astype(float) / sfreq
    intervals_s = np.diff(onsets_s)
    return onsets_s[:-1], intervals_s


def _compute_delay_seconds(*,
                           raw_trigger_events: np.ndarray,
                           sfreq: float, stim_wav: Path) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate delay between WAV trigger train and raw trigger train via interval matching.

    Note
    ----
    This is exactly the sort of logic you *could* migrate into `dcap.bids.core.sync`
    if you want it reusable across tasks — but it's still "task-provided arrays"
    style, so keeping it here is fine.

    Returns
    -------
    delay_s
        Positive means WAV starts later inside the raw.
    """
    wav_onsets, wav_intervals = _get_wav_trigger_onsets_and_intervals(stim_wav=stim_wav)
    raw_onsets, raw_intervals = _get_raw_trigger_onsets_and_intervals(raw_trigger_events=raw_trigger_events, sfreq=sfreq)

    tolerance = 1.0 / sfreq

    try:
        delay_s = _match_intervals_delay_mad(
            raw_onsets=raw_onsets,
            raw_intervals=raw_intervals,
            wav_onsets=wav_onsets,
            wav_intervals=wav_intervals,
            tolerance=tolerance,
        )
    except ValueError:
        # Fallback: onset-only matching (handles missing/extra triggers)
        delay_s = _estimate_delay_by_onset_hits(
            wav_onsets=wav_onsets,
            raw_onsets=raw_onsets,
            max_offset=30,
            match_tol_s=max(2.0 * tolerance, 0.01),
        )

    return delay_s, wav_onsets, raw_onsets


def _match_intervals_delay_mad(
    *,
    raw_onsets: np.ndarray,
    raw_intervals: np.ndarray,
    wav_onsets: np.ndarray,
    wav_intervals: np.ndarray,
    tolerance: float,
) -> float:
    """Diapix wrapper around shared interval-matching delay estimation."""
    return _core_match_interval_sequences_delay_mad(
        reference_onsets_s=wav_onsets,
        reference_intervals_s=wav_intervals,
        target_onsets_s=raw_onsets,
        target_intervals_s=raw_intervals,
        tolerance_s=tolerance,
    )


def _count_onset_matches(
    *,
    wav_onsets_s: np.ndarray,
    raw_onsets_s: np.ndarray,
    delay_s: float,
    match_tol_s: float,
) -> int:
    """Diapix wrapper around shared onset-match counting."""
    return _core_count_onset_matches(
        reference_onsets_s=wav_onsets_s,
        target_onsets_s=raw_onsets_s,
        delay_s=delay_s,
        match_tol_s=match_tol_s,
    )


def _estimate_delay_by_onset_hits(
    *,
    wav_onsets: np.ndarray,
    raw_onsets: np.ndarray,
    max_offset: int,
    match_tol_s: float,
) -> float:
    return _core_estimate_delay_by_onset_hits(
        reference_onsets_s=wav_onsets,
        target_onsets_s=raw_onsets,
        max_offset=max_offset,
        match_tol_s=match_tol_s,
    )
