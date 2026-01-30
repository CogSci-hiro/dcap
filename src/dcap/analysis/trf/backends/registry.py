# =============================================================================
#                    TRF backends: registry / factory
# =============================================================================
#
# Central registry for Temporal Response Function (TRF) backends.
#
# This module provides a lightweight name → factory mapping that allows
# TRF backends to be selected dynamically (e.g., from configuration or CLI)
# without importing backend implementations throughout the codebase.
#
# Design principles
# -----------------
# - Backends are referenced by *stable string identifiers*.
# - Each backend is constructed via a zero-argument factory.
# - The registry is intentionally explicit (no plugin auto-discovery).
# - Importing this module must not trigger heavy optional dependencies
#   beyond what each backend already guards internally.
#
# Adding a new backend
# -------------------
# 1. Implement a class conforming to the `TrfBackend` protocol.
# 2. Ensure it has a stable `name` attribute.
# 3. Import it here and register a factory in `_BACKENDS`.
#
# Example:
#     from dcap.analysis.trf.backends.foo import FooBackend
#     _BACKENDS["foo"] = lambda: FooBackend()
#

from typing import Callable, Dict, Mapping

from dcap.analysis.trf.backends.base import TrfBackend
from dcap.analysis.trf.backends.mne_rf import MneReceptiveFieldBackend


# -----------------------------------------------------------------------------
# Backend factory type
# -----------------------------------------------------------------------------
#
# A backend factory is a zero-argument callable returning a new backend
# instance. Factories are used instead of classes directly to allow
# future extension (e.g., dependency injection, conditional construction).
#
BackendFactory = Callable[[], TrfBackend]


# -----------------------------------------------------------------------------
# Backend registry
# -----------------------------------------------------------------------------
#
# Mapping from backend name → factory.
#
# Keys are user-facing identifiers (e.g., config / CLI values).
# Values are callables that construct backend instances.
#
_BACKENDS: Dict[str, BackendFactory] = {
    "mne-rf": lambda: MneReceptiveFieldBackend(),
}


def get_backend(name: str) -> TrfBackend:
    """
    Construct a TRF backend by name.

    This function is the *only* supported entry point for backend
    instantiation. Callers should never access `_BACKENDS` directly.

    Parameters
    ----------
    name : str
        Backend identifier (e.g., ``"mne-rf"``).

    Returns
    -------
    TrfBackend
        A newly constructed backend instance.

    Raises
    ------
    ValueError
        If the requested backend name is not registered.

    Usage example
    -------------
        backend = get_backend("mne-rf")
    """
    try:
        return _BACKENDS[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Unknown TRF backend {name!r}. Available: {available}"
        ) from exc


def list_backends() -> Mapping[str, BackendFactory]:
    """
    List all registered TRF backends.

    Returns
    -------
    mapping
        A shallow copy of the backend registry mapping backend names
        to their corresponding factories.

    Notes
    -----
    The returned mapping is intentionally a copy to prevent accidental
    mutation of the global registry.
    """
    return dict(_BACKENDS)
