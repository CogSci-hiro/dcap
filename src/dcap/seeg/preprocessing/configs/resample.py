from dataclasses import dataclass

@dataclass(frozen=True)
class ResampleConfig:
    sfreq_out: float = 512.0

    def __post_init__(self) -> None:
        if self.sfreq_out <= 0:
            raise ValueError(f"sfreq_out must be > 0, got {self.sfreq_out}.")
