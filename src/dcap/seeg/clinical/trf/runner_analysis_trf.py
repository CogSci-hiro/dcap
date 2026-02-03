from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import List, Sequence, Union

import numpy as np
import mne
from scipy.stats import zscore as scipy_zscore

from dcap.analysis.trf.fit import LagConfig, fit_trf_ridge, fit_trf_ridge_cv, predict_trf
from dcap.analysis.trf.tasks import DiapixConfig, DiapixTrfAdapter
from dcap.seeg.trf.contracts import TRFConfig, TRFInput, TRFResult


# =============================================================================
#                           Clinical TRF runner (CV)
# =============================================================================

def run_trf_with_analysis_trf(
    trf_input: TRFInput,
    cfg: TRFConfig,
    bids_root: Path,
    subject_id: str,
) -> TRFResult:
    """
    Clinical TRF runner bridging `dcap.seeg.trf` contracts to `dcap.analysis.trf`,
    upgraded to support:
      - multi-run input as epochs
      - cross-validation across runs (epochs)
      - alpha selection during CV
    """
    warnings: List[str] = []

    # -------------------------------------------------------------------------
    # Validate and unpack neural data source
    # -------------------------------------------------------------------------
    signal = trf_input.signal_raw
    if not isinstance(signal, (mne.io.BaseRaw, mne.BaseEpochs, list, tuple)):
        raise TypeError("TRFInput.signal_raw must be an MNE Raw, Epochs, or list[Raw].")

    # -------------------------------------------------------------------------
    # Build X and Y
    # -------------------------------------------------------------------------
    use_diapix_adapter = getattr(trf_input, "task", None) == "diapix"

    if use_diapix_adapter:
        session_id = getattr(trf_input, "session_id", None)

        # NOTE: For now we keep adapter API as-is. This assumes the adapter can
        # return X epoched in a way that matches your intended 'run == epoch'.
        adapter = DiapixTrfAdapter(
            DiapixConfig(
                target_sfreq=float(_get_signal_sfreq(signal)),
            )
        )
        data = adapter.load_epoched(bids_root, subject=subject_id, session=session_id)

        x = data.X.astype(np.float32)  # (time, epoch, feat)
        sfreq = float(data.sfreq)

        # Y from the provided signal (Raw/Epochs/list[Raw])
        y = _signal_to_epoched_y(signal, picks=_pick_seeg_ecog_from_signal(signal))

        # Conservative alignment (crop to intersection)
        if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
            n_times = min(x.shape[0], y.shape[0])
            n_epochs = min(x.shape[1], y.shape[1])
            x = x[:n_times, :n_epochs, :]
            y = y[:n_times, :n_epochs, :]
            warnings.append(
                "X/Y shape mismatch; cropped to common intersection "
                f"(n_times={n_times}, n_epochs={n_epochs})."
            )

    else:
        # Fallback: constant regressor
        y = _signal_to_epoched_y(signal, picks=_pick_seeg_ecog_from_signal(signal))
        sfreq = float(_get_signal_sfreq(signal))

        n_times = y.shape[0]
        x = np.ones((n_times, y.shape[1], 1), dtype=np.float32)

        warnings.append(
            "TRF runner is using a placeholder stimulus regressor (constant X). "
            "Scores/kernels are not interpretable until X is replaced with an aligned speech envelope."
        )

    # -------------------------------------------------------------------------
    # Standardize (z-score) along time axis
    # -------------------------------------------------------------------------
    x = scipy_zscore(x, axis=0, ddof=0)
    y = scipy_zscore(y, axis=0, ddof=0)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Lag configuration
    # -------------------------------------------------------------------------
    lag_cfg = LagConfig(
        tmin_ms=float(cfg.tmin_ms),
        tmax_ms=float(cfg.tmax_ms),
        step_ms=float(cfg.step_ms),
    )

    # -------------------------------------------------------------------------
    # Cross-validate alpha across epochs (runs)
    # -------------------------------------------------------------------------
    n_epochs = int(x.shape[1])

    # Minimal alpha grid: relative to cfg.alpha, log-spaced
    alpha_grid = _default_alpha_grid(base_alpha=float(cfg.alpha))

    cv_extra: dict = {}
    alpha_used = float(cfg.alpha)

    if n_epochs >= 2:
        cv = fit_trf_ridge_cv(
            x,
            y,
            sfreq=float(sfreq),
            lag_config=lag_cfg,
            alphas=alpha_grid,
        )
        alpha_used = float(cv.best_alpha)
        cv_extra = {
            "alphas": cv.alphas,
            "mean_score_by_alpha": cv.mean_score_by_alpha,
            "fold_score_by_alpha": cv.fold_score_by_alpha,
            "best_alpha": cv.best_alpha,
        }
    else:
        warnings.append("Only one epoch available; skipping CV and using cfg.alpha.")

    # -------------------------------------------------------------------------
    # Fit on all epochs with chosen alpha + predict
    # -------------------------------------------------------------------------
    fit = fit_trf_ridge(
        x,
        y,
        sfreq=float(sfreq),
        lag_config=lag_cfg,
        alpha=float(alpha_used),
    )
    y_hat = predict_trf(x, fit)

    # -------------------------------------------------------------------------
    # Score: per-channel Pearson correlation, aggregated across epochs
    # -------------------------------------------------------------------------
    scores_by_epoch = []
    for e in range(n_epochs):
        scores_by_epoch.append(_corr_per_channel(y[:, e, :], y_hat[:, e, :]))
    scores_by_epoch_arr = np.stack(scores_by_epoch, axis=0)  # (epoch, chan)
    scores_mean_per_chan = np.mean(scores_by_epoch_arr, axis=0)  # (chan,)
    score_mean = float(np.mean(scores_mean_per_chan))

    # -------------------------------------------------------------------------
    # Channel naming
    # -------------------------------------------------------------------------
    channel_names = _channel_names_from_signal(signal, picks=_pick_seeg_ecog_from_signal(signal))

    return TRFResult(
        model_name=str(getattr(cfg, "backend", "mne-rf")),
        coefficients=np.asarray(fit.coef_),
        times_sec=np.asarray(fit.extra.get("times_sec", []), dtype=float),
        metrics={"score_mean": score_mean},
        extra={
            "sfreq": float(sfreq),
            "channel_names": channel_names,
            "intercept": np.asarray(fit.intercept_),

            # Backwards-compatible key expected by seeg/clinical/pipeline.py
            "scores": scores_mean_per_chan.astype(np.float32),

            # Keep richer diagnostics too
            "scores_mean_per_channel": scores_mean_per_chan.astype(np.float32),
            "scores_by_epoch": scores_by_epoch_arr.astype(np.float32),

            "alpha_used": float(alpha_used),
            "cv": cv_extra,
            "backend_extra": fit.extra,
            "config": asdict(cfg),
            "warnings": warnings,
            "x_feature_name": "speech_envelope" if use_diapix_adapter else "constant",
        }

    )


# =============================================================================
#                              Helper functions
# =============================================================================

def _default_alpha_grid(*, base_alpha: float) -> np.ndarray:
    """
    Small, robust alpha grid for ridge.

    For clinical pipelines, you generally want a compact grid to keep runtime sane.
    """
    base = float(base_alpha) if np.isfinite(base_alpha) and base_alpha > 0 else 1.0
    multipliers = np.array([1e-2, 1e-1, 1.0, 1e1, 1e2], dtype=float)
    return base * multipliers


def _get_signal_sfreq(signal: Union[mne.io.BaseRaw, mne.BaseEpochs, Sequence[mne.io.BaseRaw]]) -> float:
    if isinstance(signal, mne.io.BaseRaw):
        return float(signal.info["sfreq"])
    if isinstance(signal, mne.BaseEpochs):
        return float(signal.info["sfreq"])
    # list/tuple of Raw
    if len(signal) == 0:
        raise ValueError("signal_raw list is empty.")
    return float(signal[0].info["sfreq"])


def _pick_seeg_ecog_from_signal(
    signal: Union[mne.io.BaseRaw, mne.BaseEpochs, Sequence[mne.io.BaseRaw]]
) -> np.ndarray:
    if isinstance(signal, mne.io.BaseRaw):
        return _pick_seeg_ecog(signal)
    if isinstance(signal, mne.BaseEpochs):
        return _pick_seeg_ecog(signal)
    if len(signal) == 0:
        raise ValueError("signal_raw list is empty.")
    return _pick_seeg_ecog(signal[0])


def _channel_names_from_signal(
    signal: Union[mne.io.BaseRaw, mne.BaseEpochs, Sequence[mne.io.BaseRaw]],
    *,
    picks: np.ndarray,
) -> list[str]:
    if isinstance(signal, (mne.io.BaseRaw, mne.BaseEpochs)):
        return [signal.ch_names[int(i)] for i in picks]
    # list/tuple of Raw
    return [signal[0].ch_names[int(i)] for i in picks]


def _pick_seeg_ecog(inst: Union[mne.io.BaseRaw, mne.BaseEpochs]) -> np.ndarray:
    picks = mne.pick_types(
        inst.info,
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
        ch_type_counts = {t: inst.get_channel_types().count(t) for t in set(inst.get_channel_types())}
        raise ValueError(
            "No sEEG/ECoG channels available for TRF. "
            f"Channel types present: {ch_type_counts}."
        )
    return picks


def _signal_to_epoched_y(
    signal: Union[mne.io.BaseRaw, mne.BaseEpochs, Sequence[mne.io.BaseRaw]],
    *,
    picks: np.ndarray,
) -> np.ndarray:
    """
    Convert Raw/Epochs/list[Raw] into Y shaped (n_times, n_epochs, n_outputs).

    Conventions
    -----------
    - Raw becomes a single epoch.
    - Epochs: each MNE epoch is an epoch in Y.
    - list[Raw]: each Raw is treated as one epoch (cropped to common length).
    """
    if isinstance(signal, mne.io.BaseRaw):
        y_2d = signal.get_data(picks=picks)  # (chan, time)
        y = np.transpose(y_2d, (1, 0))       # (time, chan)
        return y[:, np.newaxis, :].astype(np.float32)

    if isinstance(signal, mne.BaseEpochs):
        # Epochs.get_data(): (epoch, chan, time)
        y_3d = signal.get_data(picks=picks)
        y_3d = np.transpose(y_3d, (2, 0, 1))  # (time, epoch, chan)
        return y_3d.astype(np.float32)

    # list/tuple of Raw
    raws = list(signal)
    if len(raws) == 0:
        raise ValueError("signal_raw list is empty.")

    y_list = []
    min_n_times = None

    for r in raws:
        y_2d = r.get_data(picks=picks)        # (chan, time)
        y_t = np.transpose(y_2d, (1, 0))      # (time, chan)
        y_list.append(y_t.astype(np.float32))
        min_n_times = y_t.shape[0] if min_n_times is None else min(min_n_times, y_t.shape[0])

    # Crop to common length and stack along epoch axis
    y_list = [yy[: int(min_n_times), :] for yy in y_list]
    y_stack = np.stack(y_list, axis=1)  # (time, epoch, chan)
    return y_stack.astype(np.float32)


def _corr_per_channel(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    y0 = y - np.mean(y, axis=0, keepdims=True)
    y1 = y_hat - np.mean(y_hat, axis=0, keepdims=True)
    denom = (np.std(y0, axis=0) * np.std(y1, axis=0)) + 1e-12
    corr = np.mean(y0 * y1, axis=0) / denom
    return corr.astype(np.float32)
