from pathlib import Path
from typing import Optional

import mne
import numpy as np

from dcap.bids.core.sync import onsets_and_intervals_from_samples
from dcap.bids.tasks.sorciere.models import (
    RawTriggerCandidate,
    SorciereAlignmentResult,
)


def align_sorciere_raw(
    *,
    raw: mne.io.BaseRaw,
    reference_audio_path: Path,
    annotation_origin_in_reference_s: float = 3.0,
    trigger_channel: int = 1,
    threshold: Optional[float] = None,
    min_trigger_gap_s: float = 0.1,
) -> SorciereAlignmentResult:
    sfreq = float(raw.info["sfreq"])
    reference_onsets_s, reference_intervals_s, reference_duration_s = load_reference_trigger_timing(
        reference_audio_path=reference_audio_path,
        trigger_channel=trigger_channel,
        threshold=threshold,
        min_trigger_gap_s=min_trigger_gap_s,
    )
    candidates = extract_raw_trigger_candidates(raw)
    if len(candidates) == 0:
        raise ValueError("No candidate trigger annotations found in Sorciere raw.")

    result = estimate_alignment_from_candidates(
        reference_onsets_s=reference_onsets_s,
        reference_intervals_s=reference_intervals_s,
        raw_candidates=candidates,
        sfreq=sfreq,
        annotation_origin_in_reference_s=annotation_origin_in_reference_s,
        reference_duration_s=reference_duration_s,
    )
    return result


def load_reference_trigger_timing(
    *,
    reference_audio_path: Path,
    trigger_channel: int = 1,
    threshold: Optional[float] = None,
    min_trigger_gap_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    signal, sfreq = _load_reference_audio(reference_audio_path)
    if signal.ndim != 2 or signal.shape[1] <= trigger_channel:
        raise ValueError(
            f"Expected at least {trigger_channel + 1} channels in reference audio: {reference_audio_path}"
        )

    trigger = np.asarray(signal[:, trigger_channel], dtype=float)
    onsets_s = _detect_trigger_onsets(
        trigger=trigger,
        sfreq=float(sfreq),
        threshold=threshold,
        min_trigger_gap_s=min_trigger_gap_s,
    )
    if onsets_s.size < 2:
        raise ValueError(f"Expected at least 2 trigger onsets in reference audio: {reference_audio_path}")
    intervals_s = np.diff(onsets_s)
    duration_s = float(signal.shape[0] / float(sfreq))
    return onsets_s[:-1], intervals_s, duration_s


def extract_raw_trigger_candidates(raw: mne.io.BaseRaw) -> list[RawTriggerCandidate]:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    reverse = {value: key for key, value in event_id.items()}

    candidates: list[RawTriggerCandidate] = []
    for event_code in sorted(np.unique(events[:, 2]).tolist()):
        onset_samples = np.asarray(events[events[:, 2] == event_code][:, 0], dtype=int)
        if onset_samples.size < 2:
            continue

        description = str(reverse.get(int(event_code), f"event_{event_code}"))
        if description.lower().startswith("bad"):
            continue
        if description.lower().startswith("new segment"):
            continue

        candidates.append(
            RawTriggerCandidate(
                description=description,
                event_code=int(event_code),
                onset_samples=onset_samples,
            )
        )

    return candidates


def estimate_alignment_from_candidates(
    *,
    reference_onsets_s: np.ndarray,
    reference_intervals_s: np.ndarray,
    raw_candidates: list[RawTriggerCandidate],
    sfreq: float,
    annotation_origin_in_reference_s: float,
    reference_duration_s: Optional[float] = None,
) -> SorciereAlignmentResult:
    tolerance = 1.0 / float(sfreq)

    best: tuple[RawTriggerCandidate, float, int, float] | None = None
    last_error: Optional[Exception] = None

    for candidate in raw_candidates:
        try:
            raw_onsets_s, raw_intervals_s = onsets_and_intervals_from_samples(candidate.onset_samples, sfreq)
            delay_s, rmse_s = _estimate_delay_with_rmse(
                reference_onsets_s=reference_onsets_s,
                reference_intervals_s=reference_intervals_s,
                raw_onsets_s=raw_onsets_s,
                raw_intervals_s=raw_intervals_s,
                tolerance_s=tolerance,
            )
            hits = _count_onset_matches(
                reference_onsets_s=reference_onsets_s,
                raw_onsets_s=raw_onsets_s,
                delay_s=delay_s,
                match_tol_s=max(2.0 * tolerance, 0.01),
            )
            score = (hits, -rmse_s, raw_onsets_s.size)
            if best is None or score > (best[2], -best[3], best[0].onset_samples.size - 1):
                best = (candidate, delay_s, hits, rmse_s)
        except Exception as exc:
            last_error = exc

    if best is None:
        if last_error is not None:
            raise ValueError(f"Could not align Sorciere trigger train: {last_error}") from last_error
        raise ValueError("Could not align Sorciere trigger train.")

    candidate, delay_s, hits, _rmse_s = best
    raw_onsets_s, _ = onsets_and_intervals_from_samples(candidate.onset_samples, sfreq)
    stimulus_start_s = float(delay_s + annotation_origin_in_reference_s)

    return SorciereAlignmentResult(
        selected_description=candidate.description,
        selected_event_code=int(candidate.event_code),
        delay_s=float(delay_s),
        stimulus_start_s=stimulus_start_s,
        matched_hits=int(hits),
        reference_onsets_s=np.asarray(reference_onsets_s, dtype=float),
        raw_onsets_s=np.asarray(raw_onsets_s, dtype=float),
        annotation_origin_in_reference_s=float(annotation_origin_in_reference_s),
        reference_duration_s=None if reference_duration_s is None else float(reference_duration_s),
        candidate_count=len(raw_candidates),
    )


def _load_reference_audio(reference_audio_path: Path) -> tuple[np.ndarray, float]:
    try:
        import soundfile as sf

        signal, sfreq = sf.read(reference_audio_path, always_2d=True)
        return np.asarray(signal), float(sfreq)
    except Exception:
        pass

    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(reference_audio_path)
        sample_width = int(segment.sample_width)
        if sample_width not in (1, 2, 4):
            raise ValueError(f"Unsupported sample width in {reference_audio_path}: {sample_width}")

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        arr = np.frombuffer(segment.raw_data, dtype=dtype_map[sample_width]).reshape((-1, segment.channels))
        scale = float(np.iinfo(dtype_map[sample_width]).max)
        return arr.astype(float) / scale, float(segment.frame_rate)
    except Exception as exc:
        raise ImportError(
            "Could not decode Sorciere reference audio. Install `soundfile` or a working `pydub` backend."
        ) from exc


def _detect_trigger_onsets(
    *,
    trigger: np.ndarray,
    sfreq: float,
    threshold: Optional[float],
    min_trigger_gap_s: float,
) -> np.ndarray:
    trigger = np.asarray(trigger, dtype=float)
    if trigger.size == 0:
        return np.asarray([], dtype=float)

    if threshold is None:
        lo = float(np.percentile(trigger, 5))
        hi = float(np.percentile(trigger, 95))
        threshold = lo + 0.5 * (hi - lo)

    binary = np.zeros(trigger.shape, dtype=int)
    binary[trigger > float(threshold)] = 1

    edges = np.diff(binary, prepend=0)
    onset_samples = np.where(edges == 1)[0]
    if onset_samples.size == 0:
        return np.asarray([], dtype=float)

    min_gap_samples = max(1, int(round(float(min_trigger_gap_s) * float(sfreq))))
    kept = [int(onset_samples[0])]
    for sample in onset_samples[1:]:
        if int(sample) - kept[-1] >= min_gap_samples:
            kept.append(int(sample))

    return np.asarray(kept, dtype=float) / float(sfreq)


def _estimate_delay_with_rmse(
    *,
    reference_onsets_s: np.ndarray,
    reference_intervals_s: np.ndarray,
    raw_onsets_s: np.ndarray,
    raw_intervals_s: np.ndarray,
    tolerance_s: float,
) -> tuple[float, float]:
    if reference_intervals_s.size == 0 or raw_intervals_s.size == 0:
        raise ValueError("Trigger interval sequences must be non-empty.")

    max_window = 12
    min_window = 5
    window_len = int(min(max_window, reference_intervals_s.size, raw_intervals_s.size))
    if window_len < min_window:
        raise ValueError(
            f"Not enough trigger intervals to match (reference={reference_intervals_s.size}, raw={raw_intervals_s.size})."
        )

    max_offset = 30
    max_ref_start = int(min(max_offset, reference_intervals_s.size - window_len))
    max_raw_start = int(min(max_offset, raw_intervals_s.size - window_len))

    best_hits = -1
    best_rmse = float("inf")
    best_delay: Optional[float] = None
    best_ref_start: Optional[int] = None
    best_raw_start: Optional[int] = None

    best_any_rmse = float("inf")
    best_any_delay: Optional[float] = None
    best_any_ref_start: Optional[int] = None
    best_any_raw_start: Optional[int] = None

    for ref_start in range(max_ref_start + 1):
        ref_win = reference_intervals_s[ref_start: ref_start + window_len]
        for raw_start in range(max_raw_start + 1):
            raw_win = raw_intervals_s[raw_start: raw_start + window_len]
            rmse = float(np.sqrt(np.mean((ref_win - raw_win) ** 2)))
            delay_s = float(raw_onsets_s[raw_start] - reference_onsets_s[ref_start])

            if rmse < best_any_rmse:
                best_any_rmse = rmse
                best_any_delay = delay_s
                best_any_ref_start = ref_start
                best_any_raw_start = raw_start

            if rmse > tolerance_s:
                continue

            hits = _count_onset_matches(
                reference_onsets_s=reference_onsets_s,
                raw_onsets_s=raw_onsets_s,
                delay_s=delay_s,
                match_tol_s=max(2.0 * tolerance_s, 0.01),
            )

            if (hits > best_hits) or (hits == best_hits and rmse < best_rmse):
                best_hits = hits
                best_rmse = rmse
                best_delay = delay_s
                best_ref_start = ref_start
                best_raw_start = raw_start

    if best_delay is None:
        if best_any_delay is None or best_any_ref_start is None or best_any_raw_start is None:
            raise ValueError("Could not find any interval alignment candidate.")
        if best_any_rmse > 5.0 * tolerance_s:
            raise ValueError(
                "Could not find an interval alignment that yields consistent onset alignment "
                f"(best rmse={best_any_rmse:.6f}, tol={tolerance_s:.6f})."
            )
        best_delay = best_any_delay
        best_rmse = best_any_rmse
        best_ref_start = best_any_ref_start
        best_raw_start = best_any_raw_start

    diffs = (
        raw_onsets_s[best_raw_start: best_raw_start + window_len]
        - reference_onsets_s[best_ref_start: best_ref_start + window_len]
    )
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    if mad == 0.0:
        return median, best_rmse

    modified_z = 0.6745 * (diffs - median) / mad
    filtered = diffs[np.abs(modified_z) < 3.5]
    if filtered.size == 0:
        return median, best_rmse
    return float(filtered.mean()), best_rmse


def _count_onset_matches(
    *,
    reference_onsets_s: np.ndarray,
    raw_onsets_s: np.ndarray,
    delay_s: float,
    match_tol_s: float,
) -> int:
    if reference_onsets_s.size == 0 or raw_onsets_s.size == 0:
        return 0

    shifted = reference_onsets_s + float(delay_s)
    ref_idx = 0
    raw_idx = 0
    hits = 0

    while ref_idx < shifted.size and raw_idx < raw_onsets_s.size:
        dt = shifted[ref_idx] - raw_onsets_s[raw_idx]
        if abs(dt) <= match_tol_s:
            hits += 1
            ref_idx += 1
            raw_idx += 1
        elif dt < -match_tol_s:
            ref_idx += 1
        else:
            raw_idx += 1

    return hits
