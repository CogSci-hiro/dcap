# src/dcap/bids/core/sync.py
# =============================================================================
#                    BIDS Core: Temporal synchronization
# =============================================================================
#
# Task-agnostic utilities for estimating temporal offsets between two event
# streams based on inter-event intervals.
#
# This module implements *generic* logic:
# - extract onset times + intervals
# - robustly match interval patterns
# - estimate a constant temporal offset
#
# It does NOT:
# - assume audio
# - assume specific trigger channels
# - assume specific event semantics
#
# Tasks decide what the two event streams are.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

from typing import Tuple

import numpy as np


# =============================================================================
# Public API
# =============================================================================

def estimate_constant_delay(
    reference_onsets_s: np.ndarray,
    reference_intervals_s: np.ndarray,
    target_onsets_s: np.ndarray,
    target_intervals_s: np.ndarray,
    tolerance_s: float = 0.005,
) -> float:
    """
    Estimate a constant temporal offset between two event streams.

    The method:
    - matches inter-event intervals between two sequences
    - computes onset differences for matched intervals
    - returns a robust mean offset (MAD-filtered)

    Interpretation
    --------------
    Returned value `delay_s` satisfies approximately:

        target_onset ≈ reference_onset + delay_s

    Positive delay means the *target* stream occurs later in time.

    Parameters
    ----------
    reference_onsets_s
        Onset times (seconds) for the reference stream.
    reference_intervals_s
        Intervals between consecutive reference onsets (seconds).
    target_onsets_s
        Onset times (seconds) for the target stream.
    target_intervals_s
        Intervals between consecutive target onsets (seconds).
    tolerance_s
        Maximum allowed interval mismatch for matching (seconds).

    Returns
    -------
    float
        Estimated constant delay in seconds.

    Raises
    ------
    ValueError
        If no reliable matches can be found.

    Usage example
    -------------
        delay_s = estimate_constant_delay(
            reference_onsets_s=wav_onsets,
            reference_intervals_s=wav_intervals,
            target_onsets_s=raw_onsets,
            target_intervals_s=raw_intervals,
        )
    """
    reference_onsets_s = np.asarray(reference_onsets_s, dtype=float)
    reference_intervals_s = np.asarray(reference_intervals_s, dtype=float)
    target_onsets_s = np.asarray(target_onsets_s, dtype=float)
    target_intervals_s = np.asarray(target_intervals_s, dtype=float)

    if reference_intervals_s.size == 0 or target_intervals_s.size == 0:
        raise ValueError("Interval arrays must be non-empty.")

    onset_differences: list[float] = []

    for ref_idx, ref_interval in enumerate(reference_intervals_s):
        # Find closest matching interval in target
        tgt_idx = int(np.argmin(np.abs(target_intervals_s - ref_interval)))
        interval_diff = abs(ref_interval - target_intervals_s[tgt_idx])

        if interval_diff > tolerance_s:
            continue

        onset_diff = target_onsets_s[tgt_idx] - reference_onsets_s[ref_idx]
        onset_differences.append(float(onset_diff))

    if not onset_differences:
        raise ValueError("No matching intervals found; cannot estimate delay.")

    return _robust_mean_mad(np.asarray(onset_differences))


# =============================================================================
# Helper utilities
# =============================================================================

def onsets_and_intervals_from_samples(
    sample_indices: np.ndarray,
    sfreq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sample indices to onset times and inter-onset intervals.

    Parameters
    ----------
    sample_indices
        Array of sample indices (integers).
    sfreq
        Sampling frequency (Hz).

    Returns
    -------
    onsets_s
        Onset times in seconds (excluding last onset for interval alignment).
    intervals_s
        Inter-onset intervals in seconds.

    Usage example
    -------------
        onsets_s, intervals_s = onsets_and_intervals_from_samples(events[:, 0], sfreq)
    """
    sample_indices = np.asarray(sample_indices, dtype=float)
    if sample_indices.size < 2:
        raise ValueError("At least two sample indices are required.")

    onsets = sample_indices / float(sfreq)
    intervals = np.diff(onsets)
    return onsets[:-1], intervals


def _robust_mean_mad(values: np.ndarray, z_thresh: float = 3.5) -> float:
    """
    Compute a robust mean using Median Absolute Deviation (MAD) filtering.

    Parameters
    ----------
    values
        Input array.
    z_thresh
        Modified z-score threshold.

    Returns
    -------
    float
        Robust mean estimate.

    Usage example
    -------------
        mean = _robust_mean_mad(np.array([0.1, 0.11, 0.09, 5.0]))
    """
    values = np.asarray(values, dtype=float)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        return float(median)

    modified_z = 0.6745 * (values - median) / mad
    filtered = values[np.abs(modified_z) < z_thresh]

    if filtered.size == 0:
        return float(median)

    return float(filtered.mean())
