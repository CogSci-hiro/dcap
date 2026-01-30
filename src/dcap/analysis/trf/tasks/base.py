# =============================================================================
#                       TRF tasks: adapter interface
# =============================================================================
#
# Task adapters handle task-specific alignment/cropping/stacking rules and produce
# TRF-ready arrays for a backend.
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TrfEpochedData:
    """
    Epoched TRF-ready data.

    Attributes
    ----------
    X : ndarray, shape (n_times, n_epochs, n_features)
        Stimulus/features (e.g., speech envelope).
    Y : ndarray, shape (n_times, n_epochs, n_outputs)
        Neural responses (e.g., iEEG channels).
    sfreq : float
        Shared sampling frequency (Hz).
    epoch_ids : sequence of str
        Per-epoch identifiers (e.g., run labels).

    Usage example
    -------------
        data = TrfEpochedData(X=X, Y=Y, sfreq=100.0, epoch_ids=["run-1", "run-2"])
    """

    X: np.ndarray
    Y: np.ndarray
    sfreq: float
    epoch_ids: Sequence[str]


class TrfTaskAdapter(Protocol):
    """Protocol for TRF task adapters."""

    name: str

    def load_epoched(self, bids_root: Path, *, subject: str, session: str | None) -> TrfEpochedData:
        """Load and return epoched TRF-ready arrays."""
        ...
