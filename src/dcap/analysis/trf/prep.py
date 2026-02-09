# =============================================================================
# TRF analysis: input normalization + segmentation
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .lags import LagSpec, compute_lags, max_abs_lag_seconds
from .types import RunInfo, SegmentSpec, TrfSegment


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Canonical dataset representation for TRF fitting."""

    segments: Tuple[TrfSegment, ...]
    runs: Tuple[RunInfo, ...]
    sfreq: float
    n_features: int
    n_outputs: int


def _validate_segment_spec(
    segment_spec: SegmentSpec,
    run_n_samples: int,
    sfreq: float,
    lag_spec: LagSpec,
) -> Tuple[int, int]:
    """Return (segment_len_samp, n_segments) for a run."""
    if sfreq <= 0:
        raise ValueError(f"sfreq must be > 0, got {sfreq}")

    seg_len_s = segment_spec.segment_len_s
    n_segs = segment_spec.n_segments_per_run

    if seg_len_s is None and n_segs is None:
        raise ValueError("SegmentSpec must set exactly one of segment_len_s or n_segments_per_run.")

    if seg_len_s is not None and seg_len_s <= 0:
        raise ValueError(f"segment_len_s must be > 0, got {seg_len_s}")

    if n_segs is not None and n_segs <= 0:
        raise ValueError(f"n_segments_per_run must be > 0, got {n_segs}")

    if seg_len_s is not None and n_segs is not None:
        # Allow both, but ensure compatibility.
        implied = (run_n_samples / sfreq) / float(n_segs)
        frac = abs(implied - float(seg_len_s)) / max(float(seg_len_s), 1e-12)
        if frac > segment_spec.incompat_tol_frac:
            raise ValueError(
                "Both segment_len_s and n_segments_per_run were provided but are incompatible: "
                f"segment_len_s={seg_len_s}, implied_len_s={implied:.6f} from n_segments_per_run={n_segs}."
            )

    if seg_len_s is None:
        seg_len_s = (run_n_samples / sfreq) / float(n_segs)

    segment_len_samp = int(np.floor(float(seg_len_s) * sfreq))
    segment_len_samp = max(segment_len_samp, 1)

    # Determine n_segments from length
    n_segments = int(run_n_samples // segment_len_samp)
    if not segment_spec.drop_last and run_n_samples % segment_len_samp != 0:
        n_segments += 1

    # Lag-based sanity checks
    max_lag_s = max_abs_lag_seconds(lag_spec)
    if max_lag_s > 0:
        if float(seg_len_s) <= segment_spec.hard_min_factor * max_lag_s:
            raise ValueError(
                "Segment length is too short relative to lag window: "
                f"segment_len_s={float(seg_len_s):.6f}, max_abs_lag_s={max_lag_s:.6f}. "
                "Increase segment length or reduce lag window."
            )
        if float(seg_len_s) < segment_spec.min_len_factor * max_lag_s:
            import warnings
            warnings.warn(
                "Segment length is short relative to lag window: "
                f"segment_len_s={float(seg_len_s):.6f}, max_abs_lag_s={max_lag_s:.6f}. "
                "CV estimates may be unstable.",
                RuntimeWarning,
            )

    return segment_len_samp, n_segments


def prepare_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sfreq: float,
    lag_spec: LagSpec,
    segment_spec: Optional[SegmentSpec] = None,
    run_ids: Optional[Sequence[str]] = None,
) -> PreparedDataset:
    """Normalize inputs and optionally segment into runs/segments.

    Supported input shapes
    ----------------------
    Continuous:
        X: (n_times, n_features)
        Y: (n_times, n_outputs)

    Epoched (runs):
        X: (n_times, n_runs, n_features)
        Y: (n_times, n_runs, n_outputs)

    Notes
    -----
    - If inputs are continuous and ``segment_spec`` is None, a single segment is created.
    - If inputs are epoched and ``segment_spec`` is None, each run becomes a single segment.

    Usage example
    -------------
        from dcap.analysis.trf.lags import LagSpec
        from dcap.analysis.trf.types import SegmentSpec

        ds = prepare_dataset(X, Y, sfreq=100.0, lag_spec=LagSpec(-0.1, 0.4),
                             segment_spec=SegmentSpec(n_segments_per_run=3))
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")
    if Y.ndim != X.ndim:
        raise ValueError(f"Y.ndim must match X.ndim, got X {X.shape}, Y {Y.shape}")

    if X.ndim == 2:
        n_times, n_features = X.shape
        if Y.shape[0] != n_times:
            raise ValueError("X and Y must have same n_times.")
        n_outputs = Y.shape[1]
        runs = (RunInfo(run_id=(run_ids[0] if run_ids else "run-0"), n_samples=n_times),)
        X_runs = X[:, None, :]
        Y_runs = Y[:, None, :]
    else:
        n_times, n_runs, n_features = X.shape
        if Y.shape[0] != n_times or Y.shape[1] != n_runs:
            raise ValueError("X and Y must match in (n_times, n_runs).")
        n_outputs = Y.shape[2]
        if run_ids is None:
            run_ids = [f"run-{i}" for i in range(n_runs)]
        if len(run_ids) != n_runs:
            raise ValueError("run_ids length must match n_runs.")
        runs = tuple(RunInfo(run_id=str(run_ids[i]), n_samples=n_times) for i in range(n_runs))
        X_runs = X
        Y_runs = Y

    segments: List[TrfSegment] = []
    for run_idx, run in enumerate(runs):
        x_run = X_runs[:, run_idx, :]
        y_run = Y_runs[:, run_idx, :]

        if segment_spec is None:
            segments.append(
                TrfSegment(
                    run_id=run.run_id,
                    segment_id=0,
                    start_sample=0,
                    stop_sample=run.n_samples,
                    x=x_run,
                    y=y_run,
                )
            )
            continue

        seg_len_samp, n_segments = _validate_segment_spec(
            segment_spec=segment_spec,
            run_n_samples=run.n_samples,
            sfreq=sfreq,
            lag_spec=lag_spec,
        )
        for seg_id in range(n_segments):
            start = seg_id * seg_len_samp
            stop = min((seg_id + 1) * seg_len_samp, run.n_samples)
            if stop <= start:
                continue
            if segment_spec.drop_last and (stop - start) < seg_len_samp:
                continue
            segments.append(
                TrfSegment(
                    run_id=run.run_id,
                    segment_id=seg_id,
                    start_sample=int(start),
                    stop_sample=int(stop),
                    x=x_run[start:stop, :],
                    y=y_run[start:stop, :],
                )
            )

    return PreparedDataset(
        segments=tuple(segments),
        runs=runs,
        sfreq=float(sfreq),
        n_features=int(n_features),
        n_outputs=int(n_outputs),
    )
