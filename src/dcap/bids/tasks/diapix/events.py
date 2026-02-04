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

    conversation_end_sample = conversation_start_sample + int(CONVERSATION_DURATION_S * sfreq)

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

    return _match_intervals_delay_mad(
        raw_onsets=raw_onsets,
        raw_intervals=raw_intervals,
        wav_onsets=wav_onsets,
        wav_intervals=wav_intervals,
        tolerance=0.005,
    ), wav_onsets, raw_onsets


def _match_intervals_delay_mad(
    *,
    raw_onsets: np.ndarray,
    raw_intervals: np.ndarray,
    wav_onsets: np.ndarray,
    wav_intervals: np.ndarray,
    tolerance: float,
) -> float:
    """Match trigger-train intervals with offset tolerance and robustly estimate onset delay.

    Why this exists
    ---------------
    In some runs, the beginning of the trigger train may be missing on either side
    (e.g., missing early raw triggers). A "match-from-index-0" strategy can then
    latch onto the wrong correspondence and yield a wildly wrong delay (often
    negative enough to require padding).

    This implementation explicitly searches over small offsets (i, j) and chooses
    the alignment window with minimum RMSE between interval sequences:

        E(i, j) = sqrt(mean_k (wav_intervals[i+k] - raw_intervals[j+k])^2)

    Once the best (i, j) is found, we compute onset differences within the matched
    window and apply a MAD-based outlier filter.
    """
    if wav_intervals.size == 0 or raw_intervals.size == 0:
        raise ValueError("Empty trigger interval sequence; cannot estimate delay.")

    max_window = 12
    min_window = 5
    window_len = int(min(max_window, wav_intervals.size, raw_intervals.size))
    if window_len < min_window:
        raise ValueError(
            f"Not enough trigger intervals to match (wav={wav_intervals.size}, raw={raw_intervals.size})."
        )

    max_offset = 30
    max_wav_start = int(min(max_offset, wav_intervals.size - window_len))
    max_raw_start = int(min(max_offset, raw_intervals.size - window_len))

    best_rmse = float("inf")
    best_hits = -1
    best_wav_start = None
    best_raw_start = None
    best_delay = None

    for wav_start in range(max_wav_start + 1):
        wav_win = wav_intervals[wav_start: wav_start + window_len]
        for raw_start in range(max_raw_start + 1):
            raw_win = raw_intervals[raw_start: raw_start + window_len]
            rmse = float(np.sqrt(np.mean((wav_win - raw_win) ** 2)))

            # Reject bad interval matches early
            if rmse > tolerance:
                continue

            # IMPORTANT: keep your existing delay sign convention
            delay = float(raw_onsets[raw_start] - wav_onsets[wav_start])

            hits = _count_onset_matches(
                wav_onsets_s=wav_onsets,
                raw_onsets_s=raw_onsets,
                delay_s=delay,
                match_tol_s=max(2.0 * tolerance, 0.01),
            )

            # Prefer alignments that explain MORE onsets
            if (hits > best_hits) or (hits == best_hits and rmse < best_rmse):
                best_hits = hits
                best_rmse = rmse
                best_wav_start = wav_start
                best_raw_start = raw_start
                best_delay = delay

    if best_wav_start is None or best_raw_start is None or best_delay is None:
        raise ValueError(
            "Could not find an interval alignment that also yields consistent onset alignment."
        )

    wav_idx = int(best_wav_start)
    raw_idx = int(best_raw_start)
    diffs = raw_onsets[raw_idx : raw_idx + window_len] - wav_onsets[wav_idx : wav_idx + window_len]

    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))

    if mad == 0.0:
        return median

    modified_z = 0.6745 * (diffs - median) / mad
    filtered = diffs[np.abs(modified_z) < 3.5]
    if filtered.size == 0:
        return median
    return float(filtered.mean())


def _count_onset_matches(
    *,
    wav_onsets_s: np.ndarray,
    raw_onsets_s: np.ndarray,
    delay_s: float,
    match_tol_s: float,
) -> int:
    """Count how many WAV onsets align to RAW onsets under a proposed delay.

    We treat shifted WAV onsets as predictions of RAW onsets:
        wav_onsets_s + delay_s ≈ raw_onsets_s

    Uses a two-pointer scan (both arrays sorted).
    """
    if wav_onsets_s.size == 0 or raw_onsets_s.size == 0:
        return 0

    shifted = wav_onsets_s + delay_s
    i = 0
    j = 0
    hits = 0

    while i < shifted.size and j < raw_onsets_s.size:
        dt = shifted[i] - raw_onsets_s[j]
        if abs(dt) <= match_tol_s:
            hits += 1
            i += 1
            j += 1
        elif dt < -match_tol_s:
            i += 1
        else:
            j += 1

    return hits
