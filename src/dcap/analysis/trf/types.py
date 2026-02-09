# =============================================================================
# TRF analysis: canonical types
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .lags import LagSpec


# =============================================================================
#                               Config types
# =============================================================================

SegmentAssignment = Literal["blocked_per_run", "round_robin"]
CvScheme = Literal["blocked_kfold", "loo_run", "mc_blocked"]
AlphaMode = Literal["shared", "per_output"]
ScoringName = Literal["pearson", "spearman", "r2"]
ComplexHandling = Literal["error", "real", "imag", "magnitude"]


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """How to segment each run into contiguous segments.

    Exactly one of ``segment_len_s`` or ``n_segments_per_run`` must be set.

    Parameters
    ----------
    segment_len_s
        Desired segment length in seconds. If runs have different lengths,
        the last segment may be shorter depending on ``drop_last``.
    n_segments_per_run
        Split each run into exactly this many contiguous segments (approximately
        equal length per run).
    drop_last
        If True, drop the last segment if it would be shorter than the others.
    min_len_factor
        Warning threshold in multiples of max absolute lag window. If the
        segment length is < ``min_len_factor * max_abs_lag``, we emit a warning.
    hard_min_factor
        Error threshold. If segment length is <= ``hard_min_factor * max_abs_lag``,
        the segmentation is considered unusable.
    incompat_tol_frac
        If both ``segment_len_s`` and ``n_segments_per_run`` are provided (allowed),
        raise if implied segment length differs by more than this fraction.

    Usage example
    -------------
        seg = SegmentSpec(n_segments_per_run=3)
    """

    segment_len_s: Optional[float] = None
    n_segments_per_run: Optional[int] = None
    drop_last: bool = False
    min_len_factor: float = 8.0
    hard_min_factor: float = 2.0
    incompat_tol_frac: float = 0.01


@dataclass(frozen=True, slots=True)
class CvSpec:
    """Cross-validation specification over segments.

    Parameters
    ----------
    scheme
        CV scheme.
    n_splits
        Number of folds (required for kfold / mc_blocked).
    assignment
        Segment-to-fold assignment strategy for kfold.
    purge_s
        Gap/embargo around each test segment (in seconds) removed from training
        segments within the same run.
    weight_by_duration
        If True, aggregate segment scores weighted by segment duration.
    shuffle
        If True, shuffle segment order before assignment (generally not recommended
        for time-series unless you know what you're doing).
    random_state
        RNG seed for shuffling / Monte Carlo.

    Usage example
    -------------
        cv = CvSpec(scheme="blocked_kfold", n_splits=6, assignment="blocked_per_run", purge_s=0.5)
    """

    scheme: CvScheme = "blocked_kfold"
    n_splits: Optional[int] = None
    assignment: SegmentAssignment = "blocked_per_run"
    purge_s: float = 0.0
    weight_by_duration: bool = True
    shuffle: bool = False
    random_state: Optional[int] = None


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Model fitting hyperparameters.

    Exactly one of ``alpha`` or ``alphas`` may be provided.

    Parameters
    ----------
    alpha
        Ridge regularization strength.
    alphas
        Candidate alpha values for CV selection.
    alpha_mode
        "shared": one alpha for all outputs.
        "per_output": choose alpha per output channel (CV only).

    Usage example
    -------------
        fit = FitSpec(alphas=[0.1, 1.0, 10.0], alpha_mode="shared")
    """

    alpha: Optional[float] = None
    alphas: Optional[Sequence[float]] = None
    alpha_mode: AlphaMode = "shared"


@dataclass(frozen=True, slots=True)
class ScoringSpec:
    """Scoring configuration."""

    scoring: ScoringName = "pearson"
    complex_handling: ComplexHandling = "error"


# =============================================================================
#                               Data containers
# =============================================================================

@dataclass(frozen=True, slots=True)
class RunInfo:
    """Metadata about a run (a contiguous time series)."""

    run_id: str
    n_samples: int


@dataclass(frozen=True, slots=True)
class TrfSegment:
    """A contiguous time segment (local and dumb)."""

    run_id: str
    segment_id: int
    start_sample: int
    stop_sample: int
    x: np.ndarray  # (n_times, n_features)
    y: np.ndarray  # (n_times, n_outputs)

    @property
    def n_samples(self) -> int:
        return int(self.stop_sample - self.start_sample)


@dataclass(frozen=True, slots=True)
class Fold:
    """Train/test split over segments."""

    fold_id: int
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]


# =============================================================================
#                                  Results
# =============================================================================

@dataclass(frozen=True, slots=True)
class TrfModel:
    """Backend-agnostic TRF model (kernel + metadata)."""

    coef: np.ndarray  # (n_lags, n_features, n_outputs) float or complex
    intercept: np.ndarray  # (n_outputs,)
    lags_samp: np.ndarray  # (n_lags,)
    lags_s: np.ndarray  # (n_lags,)
    sfreq: float
    backend: str
    fit_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CvResult:
    """Cross-validation results for alpha selection."""

    alphas: np.ndarray  # (n_alphas,)
    fold_scores: np.ndarray  # (n_alphas, n_folds) aggregated across outputs
    mean_scores: np.ndarray  # (n_alphas,)
    scoring: str
    best_alpha: float
    best_alpha_by_output: Optional[np.ndarray] = None  # (n_outputs,)
    score_by_alpha_by_output: Optional[np.ndarray] = None  # (n_alphas, n_outputs)


@dataclass(frozen=True, slots=True)
class NestedCvResult:
    """Outer-loop evaluation results."""

    outer_scores: np.ndarray  # (n_outer_folds,)
    chosen_alpha: np.ndarray  # (n_outer_folds,) or (n_outer_folds, n_outputs)
    scoring: str


@dataclass(frozen=True, slots=True)
class TrfResult:
    """Persistable TRF result.

    This is intended to contain everything you might want to inspect later:
    kernel/coef, lags, alpha selection, scoring, and lightweight provenance.
    """

    model: TrfModel
    cv: Optional[CvResult] = None
    nested_cv: Optional[NestedCvResult] = None
    lag_spec: Optional[LagSpec] = None
    segment_spec: Optional[SegmentSpec] = None
    cv_spec: Optional[CvSpec] = None
    fit_spec: Optional[FitSpec] = None
    scoring_spec: Optional[ScoringSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
