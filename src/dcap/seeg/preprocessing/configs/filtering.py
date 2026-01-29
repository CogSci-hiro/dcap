from dataclasses import dataclass
from typing import Literal, Tuple

@dataclass(frozen=True)
class HighpassConfig:
    l_freq: float = 0.5
    phase: Literal["zero", "minimum"] = "zero"

    def __post_init__(self) -> None:
        if self.l_freq <= 0:
            raise ValueError(f"l_freq must be > 0, got {self.l_freq}.")

@dataclass(frozen=True)
class GammaEnvelopeConfig:
    band_hz: Tuple[float, float] = (70.0, 150.0)
    method: Literal["hilbert", "rectified_smooth"] = "hilbert"
    smoothing_sec: float = 0.1

    def __post_init__(self) -> None:
        low_hz, high_hz = float(self.band_hz[0]), float(self.band_hz[1])
        if not (0 < low_hz < high_hz):
            raise ValueError(f"band_hz must satisfy 0 < low < high, got {self.band_hz}.")
        if self.smoothing_sec < 0:
            raise ValueError(f"smoothing_sec must be >= 0, got {self.smoothing_sec}.")
