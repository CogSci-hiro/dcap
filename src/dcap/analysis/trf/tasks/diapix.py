# =============================================================================
#                         TRF task adapter: Diapix
# =============================================================================
#
# Task rules (current):
# - BIDS events.tsv contains a 'conversation_start' event corresponding to audio onset.
# - There are 4 runs; these map to the TRF epochs dimension expected by MNE:
#     X: (n_times, n_epochs, n_features)
#     Y: (n_times, n_epochs, n_outputs)
#
# This module is intentionally a *skeleton* around I/O:
# we define the contract and alignment logic hooks, but leave dataset-specific
# file discovery/loading to your BIDS IO layer.
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from dcap.analysis.trf.alignment import align_by_event_sample, event_time_to_sample
from dcap.analysis.trf.prep import stack_time_epoch_feature
from dcap.analysis.trf.tasks.base import TrfEpochedData


@dataclass(frozen=True, slots=True)
class DiapixConfig:
    """
    Diapix TRF extraction configuration.

    Parameters
    ----------
    expected_runs : int
        Expected number of runs (epochs).
    event_name : str
        Name of the event marking audio onset.

    Usage example
    -------------
        cfg = DiapixConfig(expected_runs=4, event_name="conversation_start")
    """

    expected_runs: int = 4
    event_name: str = "conversation_start"


class DiapixTrfAdapter:
    """
    TRF task adapter for Diapix.

    This adapter should:
    1) Load per-run audio waveform and compute envelope (or accept precomputed envelope).
    2) Load per-run neural time series.
    3) Read per-run BIDS events.tsv and find `conversation_start`.
    4) Align audio-derived envelope start to neural start using that event.
    5) Crop to overlap and stack runs into epochs dimension.

    Notes
    -----
    This skeleton does not yet implement BIDS I/O. Use `build_from_runs(...)` to
    plug in arrays produced by your preprocessing/BIDS utilities.
    """

    name = "diapix"

    def __init__(self, config: DiapixConfig) -> None:
        self._config = config

    def build_from_runs(
        self,
        *,
        X_runs: Sequence[np.ndarray],
        Y_runs: Sequence[np.ndarray],
        sfreq_x: float,
        sfreq_y: float,
        conversation_start_onsets_s: Sequence[float],
        first_samp_y: int = 0,
        run_ids: Sequence[str] | None = None,
    ) -> TrfEpochedData:
        """
        Build epoched TRF arrays from per-run inputs.

        Parameters
        ----------
        X_runs : sequence of ndarray, each (n_times_x, n_features)
            Stimulus features per run (e.g., envelope[:, None]). X is referenced
            to audio onset (X_run[0] == audio onset).
        Y_runs : sequence of ndarray, each (n_times_y, n_outputs)
            Neural data per run.
        sfreq_x : float
            Sampling frequency of X (Hz).
        sfreq_y : float
            Sampling frequency of Y (Hz).
        conversation_start_onsets_s : sequence of float
            Per-run event onset times (seconds) in the Y timeline (BIDS events).
            This onset corresponds to X[0] (audio onset).
        first_samp_y : int
            Optional offset for Y timeline conversion.
        run_ids : sequence of str, optional
            Run identifiers.

        Returns
        -------
        data : TrfEpochedData
            Epoched and aligned data ready for TRF backends.

        Notes
        -----
        This function assumes sfreq_x == sfreq_y. If not, resample before calling.

        Usage example
        -------------
            adapter = DiapixTrfAdapter(DiapixConfig())
            data = adapter.build_from_runs(
                X_runs=[env1[:, None], env2[:, None], env3[:, None], env4[:, None]],
                Y_runs=[Y1, Y2, Y3, Y4],
                sfreq_x=100.0,
                sfreq_y=100.0,
                conversation_start_onsets_s=[12.3, 10.0, 9.8, 11.1],
            )
        """

        n_runs = len(X_runs)
        if n_runs != len(Y_runs) or n_runs != len(conversation_start_onsets_s):
            raise ValueError("X_runs, Y_runs, and onsets must have the same length.")
        if n_runs != self._config.expected_runs:
            raise ValueError(f"Expected {self._config.expected_runs} runs, got {n_runs}.")
        if sfreq_x != sfreq_y:
            raise ValueError("sfreq_x must match sfreq_y. Resample before calling.")

        if run_ids is None:
            run_ids = [f"run-{idx + 1}" for idx in range(n_runs)]

        aligned_X: List[np.ndarray] = []
        aligned_Y: List[np.ndarray] = []

        for X_run, Y_run, onset_s in zip(X_runs, Y_runs, conversation_start_onsets_s):
            if X_run.ndim != 2:
                raise ValueError("Each X_run must be 2D (n_times, n_features).")
            if Y_run.ndim != 2:
                raise ValueError("Each Y_run must be 2D (n_times, n_outputs).")

            y_onset_samp = event_time_to_sample(onset_s=onset_s, sfreq=sfreq_y, first_samp=first_samp_y)

            # Shared timeline convention:
            # - X_run[0] corresponds to audio onset.
            # - conversation_start marks audio onset in Y's timeline at y_onset_samp.
            # Therefore: X start sample in Y timeline is y_onset_samp.
            X_aligned, Y_aligned, _ = align_by_event_sample(
                x=X_run,
                x_start_sample=y_onset_samp,
                y=Y_run,
                y_start_sample=0,
            )

            aligned_X.append(X_aligned)
            aligned_Y.append(Y_aligned)

        min_len = min(arr.shape[0] for arr in aligned_X + aligned_Y)
        aligned_X = [arr[:min_len, :] for arr in aligned_X]
        aligned_Y = [arr[:min_len, :] for arr in aligned_Y]

        X_epoched = stack_time_epoch_feature(aligned_X)  # (time, epoch, feature)
        Y_epoched = stack_time_epoch_feature(aligned_Y)  # (time, epoch, output)

        return TrfEpochedData(X=X_epoched, Y=Y_epoched, sfreq=sfreq_x, epoch_ids=list(run_ids))

    def load_epoched(self, bids_root: Path, *, subject: str, session: str | None) -> TrfEpochedData:
        raise NotImplementedError(
            "DiapixTrfAdapter.load_epoched() requires project-specific BIDS I/O. "
            "Use build_from_runs(...) for now."
        )
