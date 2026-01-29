# =============================================================================
# =============================================================================
#                     ########################################
#                     #          CONFIG: FILTERING           #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass(frozen=True)
class HighpassConfig:
    """
    Configuration for high-pass filtering.

    Attributes
    ----------
    l_freq
        High-pass cutoff in Hz.
    phase
        Filter phase. "zero" is zero-phase (non-causal) filtering.

    Usage example
    -------------
        cfg = HighpassConfig(l_freq=0.5, phase="zero")
    """

    l_freq: float = 0.5
    phase: Literal["zero", "minimum"] = "zero"

    def __post_init__(self) -> None:
        if self.l_freq <= 0:
            raise ValueError(f"l_freq must be > 0, got {self.l_freq}.")


@dataclass(frozen=True)
class GammaEnvelopeConfig:
    """
    Configuration for computing a gamma/HFA envelope time series.

    Attributes
    ----------
    band_hz
        Band-pass range in Hz (low, high).
    method
        Envelope extraction method.
        - "hilbert": magnitude of analytic signal
        - "rectified_smooth": abs(bandpassed) then smooth
    smoothing_sec
        Optional smoothing window in seconds (moving average). 0 disables smoothing.

    Usage example
    -------------
        cfg = GammaEnvelopeConfig(band_hz=(70.0, 150.0), method="hilbert", smoothing_sec=0.1)
    """

    band_hz: Tuple[float, float] = (70.0, 150.0)
    method: Literal["hilbert", "rectified_smooth"] = "hilbert"
    smoothing_sec: float = 0.1

    def __post_init__(self) -> None:
        low_hz, high_hz = float(self.band_hz[0]), float(self.band_hz[1])
        if not (0 < low_hz < high_hz):
            raise ValueError(f"band_hz must satisfy 0 < low < high, got {self.band_hz}.")
        if self.smoothing_sec < 0:
            raise ValueError(f"smoothing_sec must be >= 0, got {self.smoothing_sec}.")
