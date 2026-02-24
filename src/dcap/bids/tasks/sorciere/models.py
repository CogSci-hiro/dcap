from dataclasses import dataclass


@dataclass(frozen=True)
class SorciereTiming:
    """
    Timing configuration for Sorciere (passive listening).

    Attributes
    ----------
    stimulus_start_delay_s
        Offset from the sync trigger-train alignment point to the actual stimulus
        onset in the reference WAV.
    """

    stimulus_start_delay_s: float = 0.0

