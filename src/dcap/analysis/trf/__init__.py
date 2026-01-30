# =============================================================================
#                           TRF analysis subpackage
# =============================================================================

from dcap.analysis.trf.alignment import align_by_event_sample, event_time_to_sample
from dcap.analysis.trf.design_matrix import LagConfig, build_lagged_design_matrix, make_lag_samples
from dcap.analysis.trf.envelope import EnvelopeConfig, compute_speech_envelope
from dcap.analysis.trf.prep import (
    crop_by_samples,
    resample_poly_1d,
    resample_poly_time_last,
    stack_time_epoch_feature,
)
from dcap.analysis.trf.fit import TrfFitConfig, TrfFitResult, fit_trf, fit_trf_ridge, predict_trf

__all__ = [
    "EnvelopeConfig",
    "compute_speech_envelope",
    "LagConfig",
    "make_lag_samples",
    "build_lagged_design_matrix",
    "resample_poly_1d",
    "resample_poly_time_last",
    "crop_by_samples",
    "stack_time_epoch_feature",
    "event_time_to_sample",
    "align_by_event_sample",
    "fit_trf",
    "fit_trf_ridge",
    "predict_trf",
    "TrfFitConfig",
    "TrfFitResult",
]
