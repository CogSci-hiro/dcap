"""
Task-specific BIDS conversion utilities.

Important
---------
Converters should be deterministic and testable. They must never require
private identifiers to operate; mapping keys should be resolved via the
registry layer at runtime.

"""

# =============================================================================
#                                DCAP: BIDS
# =============================================================================

from dcap.bids.core.config import BidsCoreConfig, BidsAnatConfig

__all__ = [
    "BidsCoreConfig",
    "BidsAnatConfig",
]
