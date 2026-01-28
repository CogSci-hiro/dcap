# =============================================================================
#                     BIDS: Audio ↔ Raw synchronization
# =============================================================================
#
# Estimate delay between raw trigger beeps and WAV trigger channel.
#
# REVIEW
# =============================================================================

from pathlib import Path
from typing import Tuple

import numpy as np

from dcap.bids.core.io import load_wav


def get_wav_trigger_onsets_and_intervals(
    wav_path: Path,
    trigger_channel_index: int = 1,
    threshold: float = 10_000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract trigger onsets (seconds) and inter-onset intervals from a WAV file.

    Parameters
    ----------
    wav_path
        Path to WAV file.
    trigger_channel_index
        Index of trigger channel in stereo WAV (default: 1).
    threshold
        Threshold for trigger detection.

    Returns
    -------
    onsets_s
        Trigger onset times in seconds (excluding the last onset for interval alignment).
    intervals_s
        Intervals between consecutive onsets in seconds.

    Usage example
    -------------
        onsets_s, intervals_s = get_wav_trigger_onsets_and_intervals(Path("beeps.wav"))
    """
    sr, wav = load_wav(wav_path)
    if wav.ndim == 1:
        raise ValueError("Expected multi-channel WAV for trigger extraction.")
    if trigger_channel_index >= wav.shape[1]:
        raise ValueError("trigger_channel_index out of range for WAV channels.")

    trigger = wav[:, trigger_channel_index]

    binary = np.zeros_like(trigger, dtype=int)
    binary[trigger > threshold] = 1

    diff = np.diff(binary)
    diff[diff < 0] = 0

    onset_samples = np.where(diff == 1)[0]
    onsets = onset_samples / float(sr)

    if onsets.size < 2:
        raise ValueError("Not enough trigger onsets detected in WAV to compute intervals.")

    intervals = np.diff(onsets)
    return onsets[:-1], intervals


def get_raw_trigger_onsets_and_intervals(triggers: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw trigger events (samples) to onsets (seconds) and intervals.

    Parameters
    ----------
    triggers
        MNE events array subset (n_events, 3) for the trigger of interest.
    sfreq
        Sampling frequency of raw.

    Returns
    -------
    onsets_s
        Onsets in seconds (excluding the last onset for interval alignment).
    intervals_s
        Intervals between consecutive onsets in seconds.

    Usage example
    -------------
        onsets_s, intervals_s = get_raw_trigger_onsets_and_intervals(triggers, sfreq=2048.0)
    """
    onsets = triggers[:, 0].astype(float) / float(sfreq)
    if onsets.size < 2:
        raise ValueError("Not enough trigger events in raw to compute intervals.")
    intervals = np.diff(onsets)
    return onsets[:-1], intervals


def robust_mean_without_outliers_mad(values: np.ndarray, thresh: float = 3.5) -> float:
    """
    Robust mean using MAD-based outlier filtering.

    Usage example
    -------------
        x = np.array([0.1, 0.11, 0.09, 5.0])
        robust = robust_mean_without_outliers_mad(x)
    """
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return float(median)

    modified_z = 0.6745 * (values - median) / mad
    filtered = values[np.abs(modified_z) < thresh]
    if filtered.size == 0:
        return float(median)
    return float(filtered.mean())


def estimate_delay_seconds(raw_triggers: np.ndarray, sfreq: float, stim_wav_path: Path, tolerance_s: float = 0.005) -> float:
    """
    Estimate time offset between raw triggers and WAV triggers.

    Positive delay means: WAV starts later inside the raw timeline (wav is "shifted right").

    Parameters
    ----------
    raw_triggers
        Trigger events (n_events, 3) for the trigger of interest.
    sfreq
        Raw sampling rate.
    stim_wav_path
        Reference WAV containing the trigger train.
    tolerance_s
        Interval-matching tolerance.

    Returns
    -------
    delay_s
        Estimated delay in seconds.

    Usage example
    -------------
        delay_s = estimate_delay_seconds(raw_triggers, raw.info["sfreq"], Path("beeps.wav"))
    """
    wav_onsets, wav_intervals = get_wav_trigger_onsets_and_intervals(stim_wav_path)
    raw_onsets, raw_intervals = get_raw_trigger_onsets_and_intervals(raw_triggers, sfreq)

    onset_diffs: list[float] = []
    for raw_idx, raw_interval in enumerate(raw_intervals):
        wav_idx = int(np.argmin(np.abs(wav_intervals - raw_interval)))
        interval_diff = float(np.abs(raw_interval - wav_intervals[wav_idx]))
        if interval_diff > tolerance_s:
            continue
        onset_diffs.append(float(raw_onsets[raw_idx] - wav_onsets[wav_idx]))

    if len(onset_diffs) == 0:
        raise ValueError("Could not match enough trigger intervals to estimate delay.")

    return robust_mean_without_outliers_mad(np.asarray(onset_diffs))
