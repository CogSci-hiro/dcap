# =============================================================================
# TRF analysis: backend registry
# =============================================================================

from __future__ import annotations

from typing import Dict

from .base import TrfBackend
from .ridge import RidgeLagBackend

_BACKENDS: Dict[str, TrfBackend] = {
    "ridge": RidgeLagBackend(),
}

def get_backend(name: str) -> TrfBackend:
    name = str(name)
    if name == "mne-rf":
        # Import lazily (optional dependency)
        from .mne_rf import MneRfBackend
        if "mne-rf" not in _BACKENDS:
            _BACKENDS["mne-rf"] = MneRfBackend()
    if name not in _BACKENDS:
        raise KeyError(f"Unknown TRF backend {name!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]
