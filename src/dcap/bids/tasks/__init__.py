# =============================================================================
#                           BIDS: Task registry
# =============================================================================

from dcap.bids.tasks.base import BidsTask
from dcap.bids.tasks.diapix.task import DiapixTask


_TASKS: dict[str, BidsTask] = {
    "diapix": DiapixTask(),
}


def get_task(name: str) -> BidsTask:
    key = str(name).strip().lower()
    if key not in _TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(_TASKS)}")
    return _TASKS[key]
