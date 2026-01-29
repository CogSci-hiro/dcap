# =============================================================================
# =============================================================================
#                     ########################################
#                     #       CONFIG: BAD CHANNELS           #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Literal


BadChannelReason = Literal[
    "flat",
    "noisy",
    "interictal",
    "ictal",
    "hardware",
    "outside_brain",
    "unknown",
]


@dataclass(frozen=True)
class BadChannelsConfig:
    """
    Configuration for handling bad channels.

    This config does NOT perform detection by itself.
    It defines how bad channels should be represented and propagated.

    Attributes
    ----------
    enabled
        Whether bad-channel handling is active.
    drop
        If True, bad channels are removed from derived views.
        If False, they are kept but marked.
    default_reason
        Reason assigned when channels are marked bad without a specific label.

    Usage example
    -------------
        cfg = BadChannelsConfig(enabled=True, drop=False)
    """

    enabled: bool = True
    drop: bool = False
    default_reason: BadChannelReason = "unknown"
