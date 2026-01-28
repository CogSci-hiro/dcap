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

from dcap.bids.core.config import BidsConvertConfig, DiapixTimingConfig

__all__ = [
    "BidsConvertConfig",
    "DiapixTimingConfig",
    "convert_subject_to_bids",
]
