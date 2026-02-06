from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from dcap.features.types import DerivativeMode, FeatureKind, FeatureResult, FeatureTimeBase


@dataclass(frozen=True)
class FeatureConfig:
    """Base config shared by all features.

    Notes
    -----
    Concrete feature configs should subclass this and add parameters.
    """

    derivative: DerivativeMode = "none"


class FeatureComputer(Protocol):
    """Computes one feature on a fixed time grid."""

    @property
    def name(self) -> str:
        ...

    @property
    def kind(self) -> FeatureKind:
        ...

    def compute(
        self,
        *,
        time: FeatureTimeBase,
        audio: Optional[np.ndarray] = None,
        audio_sfreq: Optional[float] = None,
        events_df: Optional[Any] = None,
        config: FeatureConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        """Compute the feature.

        Parameters
        ----------
        time
            Target grid for output.
        audio
            Audio samples. Shape can be (n_times,) or (n_channels, n_times).
            Convention: time-last if multi-channel.
        audio_sfreq
            Sampling rate of `audio`.
        events_df
            Optional annotations/events (linguistic features, etc.).
            Kept generic on purpose.
        config
            FeatureConfig (or subclass).
        context
            Optional extra info (subject/run ids, paths, etc.).

        Returns
        -------
        FeatureResult
            The computed feature aligned to `time`.

        Usage example
        -------------
            time = FeatureTimeBase(sfreq=100.0, n_times=40_000, t0_s=0.0)
            result = computer.compute(time=time, audio=wav, audio_sfreq=48_000, config=cfg)
        """
        ...
