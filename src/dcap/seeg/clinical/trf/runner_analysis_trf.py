# src/dcap/seeg/trf/runner_analysis_trf.py
# =============================================================================
#                     ########################################
#                     #        TRF RUNNER (analysis.trf)      #
#                     ########################################
# =============================================================================

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import mne
from scipy.stats import zscore

from dcap.analysis.trf.fit import fit_trf_ridge, predict_trf, LagConfig
from dcap.seeg.trf.contracts import TRFConfig, TRFInput, TRFResult


def run_trf_with_analysis_trf(trf_input: TRFInput, cfg: TRFConfig) -> TRFResult:
    """
    Clinical TRF runner bridging dcap.seeg.trf.contracts -> dcap.analysis.trf.

    Notes
    -----
    - Currently supports single-run TRF (epoch dimension = 1).
    - Assumes `trf_input.signal_raw` is the *neural* signal to model (e.g., gamma envelope).
    - Uses `events_df` only for provenance now; event-aware alignment can be added as needed.

    Returns
    -------
    TRFResult
        Backend-agnostic result packaged for reporting.
    """
    raw = trf_input.signal_raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("TRFInput.signal_raw must be an MNE Raw.")

    # -------------------------------------------------------------------------
    # Build Y: neural data (time, epoch, output)
    # -------------------------------------------------------------------------
    picks = mne.pick_types(
        raw.info,
        seeg=True,
        ecog=True,
        eeg=False,
        meg=False,
        stim=False,
        misc=False,
        eog=False,
        ecg=False,
    )
    if picks.size == 0:
        raise ValueError("No sEEG/ECoG channels available for TRF.")

    y_2d = raw.get_data(picks=picks)  # (n_ch, n_times)
    y = np.transpose(y_2d, (1, 0))    # (n_times, n_outputs)
    y = y[:, np.newaxis, :]           # (n_times, n_epochs=1, n_outputs)

    sfreq = float(raw.info["sfreq"])

    # -------------------------------------------------------------------------
    # Build X: stimulus feature(s) (time, epoch, feature)
    #
    # Minimal placeholder: a constant "stim present" regressor.
    # Replace with speech envelope aligned to conversation_start when ready.
    # -------------------------------------------------------------------------
    n_times = y.shape[0]
    x = np.ones((n_times, 1, 1), dtype=np.float32)  # (time, epoch, feature)

    # Optional: z-score along time per epoch (feature-wise / channel-wise)
    x = zscore(x, axis=0)
    y = zscore(y, axis=0)

    lag_cfg = LagConfig(
        tmin_ms=float(cfg.tmin_ms),
        tmax_ms=float(cfg.tmax_ms),
        step_ms=float(cfg.step_ms),
    )

    fit = fit_trf_ridge(
        x,
        y,
        sfreq=sfreq,
        lag_config=lag_cfg,
        alpha=float(cfg.alpha),
        backend=str(getattr(cfg, "backend", "mne-rf")),
    )

    y_hat = predict_trf(x, fit)

    # Simple score: correlation per output channel over time (epoch 0)
    scores = _corr_per_channel(y[:, 0, :], y_hat[:, 0, :])  # (n_outputs,)

    # Package result (adapt fields to your TRFResult contract)
    return TRFResult(
        backend=str(getattr(cfg, "backend", "mne-rf")),
        sfreq=sfreq,
        channel_names=[raw.ch_names[int(i)] for i in picks],
        coef=fit.coef_,
        intercept=fit.intercept_,
        scores=scores,
        extra=fit.extra,
        config=asdict(cfg),
    )


def _corr_per_channel(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Pearson corr per channel over time.

    Parameters
    ----------
    y, y_hat : (n_times, n_outputs)

    Returns
    -------
    corr : (n_outputs,)
    """
    y0 = y - np.mean(y, axis=0, keepdims=True)
    y1 = y_hat - np.mean(y_hat, axis=0, keepdims=True)
    denom = (np.std(y0, axis=0) * np.std(y1, axis=0)) + 1e-12
    corr = np.mean(y0 * y1, axis=0) / denom
    return corr.astype(np.float32)
