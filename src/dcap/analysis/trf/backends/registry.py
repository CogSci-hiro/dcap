# =============================================================================
#                    TRF backends: registry / factory
# =============================================================================

from typing import Callable, Dict, Mapping

from dcap.analysis.trf.backends.base import TrfBackend
from dcap.analysis.trf.backends.mne_rf import MneReceptiveFieldBackend

BackendFactory = Callable[[], TrfBackend]

_BACKENDS: Dict[str, BackendFactory] = {
    "mne-rf": lambda: MneReceptiveFieldBackend(),
}


def get_backend(name: str) -> TrfBackend:
    """
    Construct a TRF backend by name.

    Usage example
    -------------
        backend = get_backend("mne-rf")
    """
    try:
        return _BACKENDS[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(f"Unknown TRF backend {name!r}. Available: {available}") from exc


def list_backends() -> Mapping[str, BackendFactory]:
    """Return a copy of the backend registry."""
    return dict(_BACKENDS)
