
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np


FeatureKind = Literal["acoustic", "linguistic", "other"]
DerivativeMode = Literal["none", "diff", "absdiff"]


@dataclass(frozen=True)
class FeatureTimeBase:
    """Defines the sampling grid for a feature signal."""

    sfreq: float
    n_times: int
    t0_s: float = 0.0  # optional offset, useful if you later align to events


@dataclass(frozen=True)
class FeatureResult:
    """A computed feature as a time-aligned signal.

    Attributes
    ----------
    name
        Canonical feature name (registry key).
    kind
        High-level category (acoustic / linguistic).
    values
        Feature values shaped (n_signals, n_times) or (n_times,).
        Convention: **time is last axis**.
    time
        Timebase metadata for `values`.
    channel_names
        Optional names for the first dimension of `values` (if 2D).
        Example: ["env"] or ["flux"] or ["f0"].
    meta
        Free-form metadata (parameters, warnings, provenance).
    """

    name: str
    kind: FeatureKind
    values: np.ndarray
    time: FeatureTimeBase
    channel_names: Optional[Sequence[str]] = None
    meta: Optional[Mapping[str, Any]] = None
