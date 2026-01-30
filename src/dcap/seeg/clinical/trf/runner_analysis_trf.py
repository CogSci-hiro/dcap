from pathlib import Path
from dataclasses import asdict
from typing import List

import numpy as np
import mne
from scipy.stats import zscore as scipy_zscore

from dcap.analysis.trf.fit import LagConfig, fit_trf_ridge, predict_trf
from dcap.analysis.trf.tasks import DiapixConfig, DiapixTrfAdapter
from dcap.seeg.trf.contracts import TRFConfig, TRFInput, TRFResult


def run_trf_with_analysis_trf(trf_input: TRFInput, cfg: TRFConfig, bids_root: Path, subject_id: str) -> TRFResult:
    warnings: List[str] = []

    raw = trf_input.signal_raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("TRFInput.signal_raw must be an MNE Raw.")

    # -------------------------------------------------------------------------
    # Decide whether we can build real X (envelope) via Diapix adapter
    # -------------------------------------------------------------------------
    use_diapix_adapter = getattr(trf_input, "task", None) == "diapix"
    if use_diapix_adapter:
        session_id = getattr(trf_input, "session_id", None)

        adapter = DiapixTrfAdapter(
            DiapixConfig(
                target_sfreq=float(raw.info["sfreq"]),  # or cfg.target_sfreq if you add it
            )
        )
        data = adapter.load_epoched(bids_root, subject=subject_id, session=session_id)

        x = data.X.astype(np.float32)  # (time, epoch, feature)
        sfreq = float(data.sfreq)

        # Build Y from the same raw you already have OR from adapter outputs
        # If you trust your current raw view (e.g. gamma envelope), use it:
        y = _raw_to_epoched_y(raw, picks=_pick_seeg_ecog(raw))

        # If you want perfect consistency with the adapter's neural load, use:
        # y = data.Y.astype(np.float32)
        # and then use channel names from that load path.

        # IMPORTANT: enforce equal time/epochs
        if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
            n_times = min(x.shape[0], y.shape[0])
            n_epochs = min(x.shape[1], y.shape[1])
            x = x[:n_times, :n_epochs, :]
            y = y[:n_times, :n_epochs, :]

    else:
        # ---------------------------------------------------------------------
        # Fallback: placeholder X (current behavior)
        # ---------------------------------------------------------------------
        picks = _pick_seeg_ecog(raw)
        y = _raw_to_epoched_y(raw, picks=picks)
        sfreq = float(raw.info["sfreq"])

        n_times = y.shape[0]
        x = np.ones((n_times, 1, 1), dtype=np.float32)
        warnings.append(
            "TRF runner is using a placeholder stimulus regressor (constant X). "
            "Scores/kernels are not interpretable until X is replaced with an aligned speech envelope."
        )

    # -------------------------------------------------------------------------
    # Z-score along time axis=0 (consistent with your existing approach)
    # -------------------------------------------------------------------------
    x = scipy_zscore(x, axis=0, ddof=0)
    y = scipy_zscore(y, axis=0, ddof=0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
    )
    y_hat = predict_trf(x, fit)

    scores = _corr_per_channel(y[:, 0, :], y_hat[:, 0, :])

    # channel names: if you used _raw_to_epoched_y, you have picks
    picks_for_names = _pick_seeg_ecog(raw)

    return TRFResult(
        model_name=str(getattr(cfg, "backend", "mne-rf")),
        coefficients=np.asarray(fit.coef_),
        times_sec=np.asarray(fit.extra.get("times_sec", []), dtype=float),
        metrics={"score_mean": float(np.mean(scores))},
        extra={
            "sfreq": float(sfreq),
            "channel_names": [raw.ch_names[int(i)] for i in picks_for_names],
            "intercept": np.asarray(fit.intercept_),
            "scores": scores,
            "backend_extra": fit.extra,
            "config": asdict(cfg),
            "warnings": warnings,
            "x_feature_name": "speech_envelope" if use_diapix_adapter else "constant",
        },
    )


def _pick_seeg_ecog(raw: mne.io.BaseRaw) -> np.ndarray:
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
        ch_type_counts = {t: raw.get_channel_types().count(t) for t in set(raw.get_channel_types())}
        raise ValueError(
            "No sEEG/ECoG channels available for TRF. "
            f"Channel types present: {ch_type_counts}."
        )
    return picks


def _raw_to_epoched_y(raw: mne.io.BaseRaw, picks: np.ndarray) -> np.ndarray:
    y_2d = raw.get_data(picks=picks)              # (n_channels, n_times)
    y = np.transpose(y_2d, (1, 0))                # (n_times, n_outputs)
    return y[:, np.newaxis, :].astype(np.float32) # (n_times, 1, n_outputs)


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