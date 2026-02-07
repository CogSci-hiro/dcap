from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, TypeVar

import numpy as np

from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase


@dataclass(frozen=True)
class FeatureConfig:
    """Base config shared by all features."""
    derivative: str = "none"  # keep your DerivativeMode if you already have it


# Contravariant: a FeatureComputer that accepts HilbertEnvelopeConfig
# can be used where a FeatureComputer[FeatureConfig] is expected *if needed*.
C_cfg = TypeVar("C_cfg", bound=FeatureConfig, contravariant=True)


class FeatureComputer(Protocol[C_cfg]):
    """Computes one feature on a fixed time grid."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> FeatureKind: ...

    def compute(
        self,
        *,
        time: FeatureTimeBase,
        audio: Optional[np.ndarray] = None,
        audio_sfreq: Optional[float] = None,
        events_df: Optional[Any] = None,
        config: C_cfg,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult: ...
