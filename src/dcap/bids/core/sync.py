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


def count_onset_matches(
    *,
    reference_onsets_s: np.ndarray,
    target_onsets_s: np.ndarray,
    delay_s: float,
    match_tol_s: float,
) -> int:
    """
    Count aligned onset pairs under a proposed constant delay.

    Interpretation
    --------------
    We treat shifted reference onsets as predictions of target onsets:

        reference_onsets_s + delay_s ~= target_onsets_s

    Parameters
    ----------
    reference_onsets_s
        Reference onset times (seconds), sorted ascending.
    target_onsets_s
        Target onset times (seconds), sorted ascending.
    delay_s
        Proposed constant delay (target minus reference).
    match_tol_s
        Absolute tolerance for a match.
    """
    if reference_onsets_s.size == 0 or target_onsets_s.size == 0:
        return 0

    shifted = np.asarray(reference_onsets_s, dtype=float) + float(delay_s)
    target = np.asarray(target_onsets_s, dtype=float)

    i = 0
    j = 0
    hits = 0
    while i < shifted.size and j < target.size:
        dt = float(shifted[i] - target[j])
        if abs(dt) <= match_tol_s:
            hits += 1
            i += 1
            j += 1
        elif dt < -match_tol_s:
            i += 1
        else:
            j += 1

    return hits


def estimate_delay_by_onset_hits(
    *,
    reference_onsets_s: np.ndarray,
    target_onsets_s: np.ndarray,
    max_offset: int,
    match_tol_s: float,
) -> float:
    """
    Estimate delay via onset-only matching, allowing missing leading events.

    Returns
    -------
    float
        Delay in seconds (target minus reference).
    """
    reference_onsets = np.asarray(reference_onsets_s, dtype=float)
    target_onsets = np.asarray(target_onsets_s, dtype=float)

    if reference_onsets.size == 0 or target_onsets.size == 0:
        raise ValueError("Empty onset sequence; cannot estimate delay.")

    max_ref = int(min(max_offset, reference_onsets.size - 1))
    max_target = int(min(max_offset, target_onsets.size - 1))

    best_hits = -1
    best_delay = None

    for i in range(max_ref + 1):
        for j in range(max_target + 1):
            delay = float(target_onsets[j] - reference_onsets[i])
            hits = count_onset_matches(
                reference_onsets_s=reference_onsets,
                target_onsets_s=target_onsets,
                delay_s=delay,
                match_tol_s=match_tol_s,
            )
            if hits > best_hits:
                best_hits = hits
                best_delay = delay

    if best_delay is None or best_hits <= 0:
        raise ValueError("Could not estimate delay from onset matches (no matches found).")

    shifted = reference_onsets + best_delay
    i = 0
    j = 0
    diffs: list[float] = []
    while i < shifted.size and j < target_onsets.size:
        dt = float(shifted[i] - target_onsets[j])
        if abs(dt) <= match_tol_s:
            diffs.append(float(target_onsets[j] - reference_onsets[i]))
            i += 1
            j += 1
        elif dt < -match_tol_s:
            i += 1
        else:
            j += 1

    if len(diffs) == 0:
        return float(best_delay)

    return float(np.median(np.asarray(diffs, dtype=float)))


def match_interval_sequences_delay_mad(
    *,
    reference_onsets_s: np.ndarray,
    reference_intervals_s: np.ndarray,
    target_onsets_s: np.ndarray,
    target_intervals_s: np.ndarray,
    tolerance_s: float,
) -> float:
    """
    Robustly estimate delay by matching interval windows and filtering onset diffs.

    Returns
    -------
    float
        Delay in seconds (target minus reference).
    """
    reference_onsets = np.asarray(reference_onsets_s, dtype=float)
    reference_intervals = np.asarray(reference_intervals_s, dtype=float)
    target_onsets = np.asarray(target_onsets_s, dtype=float)
    target_intervals = np.asarray(target_intervals_s, dtype=float)

    if reference_intervals.size == 0 or target_intervals.size == 0:
        raise ValueError("Empty trigger interval sequence; cannot estimate delay.")

    max_window = 12
    min_window = 5
    window_len = int(min(max_window, reference_intervals.size, target_intervals.size))
    if window_len < min_window:
        raise ValueError(
            "Not enough trigger intervals to match "
            f"(reference={reference_intervals.size}, target={target_intervals.size})."
        )

    max_offset = 30
    max_ref_start = int(min(max_offset, reference_intervals.size - window_len))
    max_target_start = int(min(max_offset, target_intervals.size - window_len))

    best_rmse = float("inf")
    best_hits = -1
    best_ref_start = None
    best_target_start = None
    best_delay = None

    best_rmse_any = float("inf")
    best_ref_start_any = None
    best_target_start_any = None
    best_delay_any = None

    for ref_start in range(max_ref_start + 1):
        ref_win = reference_intervals[ref_start: ref_start + window_len]
        for target_start in range(max_target_start + 1):
            target_win = target_intervals[target_start: target_start + window_len]
            rmse = float(np.sqrt(np.mean((ref_win - target_win) ** 2)))

            if rmse < best_rmse_any:
                best_rmse_any = rmse
                best_ref_start_any = ref_start
                best_target_start_any = target_start
                best_delay_any = float(target_onsets[target_start] - reference_onsets[ref_start])

            if rmse > tolerance_s:
                continue

            delay = float(target_onsets[target_start] - reference_onsets[ref_start])
            hits = count_onset_matches(
                reference_onsets_s=reference_onsets,
                target_onsets_s=target_onsets,
                delay_s=delay,
                match_tol_s=max(2.0 * tolerance_s, 0.01),
            )

            if (hits > best_hits) or (hits == best_hits and rmse < best_rmse):
                best_hits = hits
                best_rmse = rmse
                best_ref_start = ref_start
                best_target_start = target_start
                best_delay = delay

    if best_ref_start is None or best_target_start is None or best_delay is None:
        if best_ref_start_any is None or best_target_start_any is None or best_delay_any is None:
            raise ValueError("Could not find any interval alignment candidate.")

        if best_rmse_any > 5.0 * tolerance_s:
            raise ValueError(
                "Could not find an interval alignment that also yields consistent onset alignment, "
                f"and best available RMSE is too high (rmse={best_rmse_any:.6f}, tol={tolerance_s:.6f})."
            )

        best_ref_start = best_ref_start_any
        best_target_start = best_target_start_any
        best_delay = best_delay_any

    ref_idx = int(best_ref_start)
    target_idx = int(best_target_start)
    diffs = (
        target_onsets[target_idx: target_idx + window_len]
        - reference_onsets[ref_idx: ref_idx + window_len]
    )

    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    if mad == 0.0:
        return median

    modified_z = 0.6745 * (diffs - median) / mad
    filtered = diffs[np.abs(modified_z) < 3.5]
    if filtered.size == 0:
        return median
    return float(filtered.mean())


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
