# dcap/bids/tasks/diapix/sidecars.py
# =============================================================================
#                              DIAPIX SIDECARS
# =============================================================================

from pathlib import Path
from typing import Any, Dict, Optional


def build_task_sidecar_fields() -> Dict[str, Any]:
    """
    Return task-specific JSON sidecar fields to be merged next to iEEG recordings.

    Note
    ----
    Core does not define which fields should exist. This returns *task policy only*.

    Usage example
    -------------
        fields = build_task_sidecar_fields()
    """
    # Keep this minimal and truthful. Add as needed.
    return {
        "TaskName": "diapix",
    }
