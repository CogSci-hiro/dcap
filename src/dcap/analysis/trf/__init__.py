"""dcap.analysis.trf

Minimal Temporal Response Function (TRF) analysis utilities.

This subpackage is designed to be:
- testable (pure functions, typed containers)
- headless (no plotting/report generation)
- reusable across tasks/datasets

The current state is a *skeleton*: function bodies raise NotImplementedError
until DSP/modeling logic is filled in.

Public API
----------
- :func:`dcap.analysis.trf.compute_speech_envelope`
- :func:`dcap.analysis.trf.build_lagged_design_matrix`
- :func:`dcap.analysis.trf.fit_trf_ridge`
- :func:`dcap.analysis.trf.predict_trf`
- :func:`dcap.analysis.trf.save_trf_result`
- :func:`dcap.analysis.trf.load_trf_result`

"""

from dcap.analysis.trf.design_matrix import build_lagged_design_matrix
from dcap.analysis.trf.envelope import compute_speech_envelope
from dcap.analysis.trf.fit import fit_trf_ridge, predict_trf
from dcap.analysis.trf.io import load_trf_result, save_trf_result
from dcap.analysis.trf.metrics import pearson_corr_by_output, r2_by_output
from dcap.analysis.trf.types import EnvelopeConfig, LagConfig, TrfFitConfig, TrfResult

__all__ = [
    "EnvelopeConfig",
    "LagConfig",
    "TrfFitConfig",
    "TrfResult",
    "build_lagged_design_matrix",
    "compute_speech_envelope",
    "fit_trf_ridge",
    "load_trf_result",
    "pearson_corr_by_output",
    "predict_trf",
    "r2_by_output",
    "save_trf_result",
]
