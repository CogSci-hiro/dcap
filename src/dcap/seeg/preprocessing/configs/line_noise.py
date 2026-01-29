# =============================================================================
# =============================================================================
#                     ########################################
#                     #       CONFIG: LINE NOISE REMOVAL     #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Literal, Optional, Sequence


@dataclass(frozen=True)
class LineNoiseConfig:
    """
    Configuration for line-noise removal.

    Attributes
    ----------
    method
        "notch" uses MNE notch filtering; "zapline" uses meegkit zapline (if installed).
    freq_base
        Base line frequency in Hz (typically 50 or 60).
    max_harmonic_hz
        Highest harmonic to remove (inclusive).
    picks
        Optional channel names to process. None means all channels.

    Usage example
    -------------
        cfg = LineNoiseConfig(method="notch", freq_base=50.0, max_harmonic_hz=250.0)
    """

    method: Literal["notch", "zapline"] = "notch"
    freq_base: float = 50.0
    max_harmonic_hz: float = 250.0
    picks: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if self.freq_base <= 0:
            raise ValueError(f"freq_base must be > 0, got {self.freq_base}.")
        if self.max_harmonic_hz <= 0:
            raise ValueError(f"max_harmonic_hz must be > 0, got {self.max_harmonic_hz}.")
        if self.max_harmonic_hz < self.freq_base:
            raise ValueError(
                f"max_harmonic_hz ({self.max_harmonic_hz}) must be >= freq_base ({self.freq_base})."
            )
