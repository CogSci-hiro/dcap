# dcap/bids/tasks/__init__.py
# =============================================================================
#                         DCAP: BIDS tasks package
# =============================================================================
#
# IMPORTANT:
# - Do NOT instantiate tasks at import time.
# - Tasks require runtime context (private mapping, assets, etc.).
# - Use dcap.bids.tasks.registry.resolve_task(...) instead.
#
# =============================================================================

from dcap.bids.tasks.base import BidsTask, PreparedEvents, RecordingUnit
from dcap.bids.tasks.registry import TaskFactoryContext, resolve_task

__all__ = [
    "BidsTask",
    "PreparedEvents",
    "RecordingUnit",
    "TaskFactoryContext",
    "resolve_task",
]
