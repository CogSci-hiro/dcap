from dataclasses import dataclass
from typing import Literal, Optional, Sequence

@dataclass(frozen=True)
class LineNoiseConfig:
    method: Literal["notch", "zapline"] = "notch"
    freq_base: float = 50.0
    max_harmonic_hz: float = 250.0
    picks: Optional[Sequence[str]] = None
    chunk_sec: float = 60.0
    nremove: int = 1

    def __post_init__(self) -> None:
        if self.freq_base <= 0:
            raise ValueError(f"freq_base must be > 0, got {self.freq_base}.")
        if self.max_harmonic_hz <= 0:
            raise ValueError(f"max_harmonic_hz must be > 0, got {self.max_harmonic_hz}.")
        if self.max_harmonic_hz < self.freq_base:
            raise ValueError(
                f"max_harmonic_hz ({self.max_harmonic_hz}) must be >= freq_base ({self.freq_base})."
            )
