from pathlib import Path
from dataclasses import asdict
from typing import List

import numpy as np
import mne
from scipy.stats import zscore as scipy_zscore

from dcap.analysis.trf.fit import LagConfig, fit_trf_ridge, predict_trf
from dcap.analysis.trf.tasks import DiapixConfig, DiapixTrfAdapter
from dcap.seeg.trf.contracts import TRFConfig, TRFInput, TRFResult


def run_trf_with_analysis_trf(
    trf_input: TRFInput,
    cfg: TRFConfig,
    bids_root: Path,
    subject_id: str,
) -> TRFResult:
    """
    Clinical TRF runner bridging `dcap.seeg.trf` contracts to `dcap.analysis.trf`.

    This function is a composition/adapter layer: it takes a clinical `TRFInput`
    (which carries an MNE Raw, plus identifiers and optional task metadata) and
    executes the TRF fitting pipeline implemented in `dcap.analysis.trf`.

    It supports two modes:
    1) Diapix adapter mode (preferred):
       - builds a real stimulus regressor `X` via `DiapixTrfAdapter`
       - builds a neural response `Y` from the provided `signal_raw` (often gamma envelope)
    2) Fallback / placeholder mode:
       - uses a constant regressor `X = 1` to keep the plumbing operational
       - produces uninterpretable kernels/scores (but allows end-to-end testing)

    Parameters
    ----------
    trf_input
        Clinical TRF input payload. Must contain:
        - `signal_raw`: MNE Raw used to build neural response Y
        - optionally `task` (e.g., "diapix") and `session_id`
    cfg
        TRF configuration (lags, alpha, backend label, etc.).
    bids_root
        BIDS root directory used by task adapters to load stimulus features and epochs.
    subject_id
        BIDS subject identifier used by task adapters.

    Returns
    -------
    TRFResult
        Report-ready TRF result containing coefficients, lag times, metrics,
        channel names, and provenance/warnings in `extra`.

    Notes
    -----
    - This runner standardizes inputs via z-scoring and NaN-handling, then calls:
        `fit_trf_ridge` and `predict_trf`.
    - Prediction uses the fitted model returned by `fit_trf_ridge`.
    - Scores are computed as per-channel Pearson correlation on the first epoch.
    """
    # Collect soft warnings that should appear in the report bundle but do not
    # necessarily justify raising an exception.
    warnings: List[str] = []

    # -------------------------------------------------------------------------
    # Validate and unpack neural data source
    # -------------------------------------------------------------------------
    #
    # `signal_raw` is expected to be an MNE Raw carrying the neural time series
    # to be modeled (often a gamma envelope view).
    #
    raw = trf_input.signal_raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("TRFInput.signal_raw must be an MNE Raw.")

    # -------------------------------------------------------------------------
    # Decide whether we can build a real stimulus regressor X via a task adapter
    # -------------------------------------------------------------------------
    #
    # We treat `task` as an optional hint that enables task-specific loading.
    # If the task is unknown or absent, we fall back to a placeholder regressor.
    #
    use_diapix_adapter = getattr(trf_input, "task", None) == "diapix"

    if use_diapix_adapter:
        # ---------------------------------------------------------------------
        # Task-specific path: load aligned regressor(s) + epochs via Diapix adapter
        # ---------------------------------------------------------------------
        #
        # The adapter is responsible for:
        # - reading stimulus-derived features from BIDS/derivatives/private assets
        # - producing epoched X (time, epoch, feature)
        # - providing sfreq consistent with those arrays
        #
        # We additionally allow session-aware loading if present.
        #
        session_id = getattr(trf_input, "session_id", None)

        adapter = DiapixTrfAdapter(
            DiapixConfig(
                # We request the adapter to resample stimulus features to match
                # the neural sampling rate of `raw`. If you later support a
                # TRFConfig target sfreq, prefer cfg.target_sfreq instead.
                target_sfreq=float(raw.info["sfreq"]),
            )
        )

        data = adapter.load_epoched(bids_root, subject=subject_id, session=session_id)

        # X: (time, epoch, feature)
        # Cast to float32 to keep memory footprint reasonable and ensure
        # consistent dtype across downstream linear algebra.
        x = data.X.astype(np.float32)
        sfreq = float(data.sfreq)

        # ---------------------------------------------------------------------
        # Build Y (neural responses)
        # ---------------------------------------------------------------------
        #
        # Option A (current default): derive Y from `raw` (the clinical view).
        # This makes the runner compatible with whichever raw view was provided
        # upstream (e.g., gamma envelope vs. broadband).
        #
        y = _raw_to_epoched_y(raw, picks=_pick_seeg_ecog(raw))

        # Option B (alternative): use the neural data loaded by the adapter.
        # This is sometimes preferable if you want perfect "same-loader"
        # consistency of epochs and channel ordering.
        #
        # y = data.Y.astype(np.float32)

        # ---------------------------------------------------------------------
        # Enforce time/epoch alignment
        # ---------------------------------------------------------------------
        #
        # Adapter and raw-derived arrays can differ slightly due to cropping,
        # resampling edge effects, or mismatched epoch definitions.
        # For now, we conservatively crop to the common intersection.
        #
        if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
            n_times = min(x.shape[0], y.shape[0])
            n_epochs = min(x.shape[1], y.shape[1])
            x = x[:n_times, :n_epochs, :]
            y = y[:n_times, :n_epochs, :]

    else:
        # ---------------------------------------------------------------------
        # Fallback: placeholder X (for plumbing tests / incomplete integrations)
        # ---------------------------------------------------------------------
        #
        # This keeps the TRF code path executable when no task adapter exists.
        # The resulting model is not scientifically interpretable because X is
        # constant over time (it has no meaningful structure to explain Y).
        #
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
    # Standardize (z-score) along time axis
    # -------------------------------------------------------------------------
    #
    # Convention:
    # - axis=0 corresponds to time (n_times)
    # - we z-score each epoch/feature (for X) and each epoch/channel (for Y)
    #
    # We also replace any NaN/Inf introduced by zero-variance signals (or missing
    # values) with zeros to keep linear algebra stable.
    #
    x = scipy_zscore(x, axis=0, ddof=0)
    y = scipy_zscore(y, axis=0, ddof=0)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Lag configuration (ms -> sample grid handled downstream)
    # -------------------------------------------------------------------------
    #
    # `analysis.trf` builds an explicit lag grid from (tmin, tmax, step) in ms
    # given the sampling rate. The grid determines the receptive field support.
    #
    lag_cfg = LagConfig(
        tmin_ms=float(cfg.tmin_ms),
        tmax_ms=float(cfg.tmax_ms),
        step_ms=float(cfg.step_ms),
    )

    # -------------------------------------------------------------------------
    # Fit + predict
    # -------------------------------------------------------------------------
    #
    # `fit_trf_ridge` returns a standardized fit result with:
    # - coef_ : kernel weights
    # - intercept_
    # - extra : backend-specific details, including lag times in seconds
    #
    fit = fit_trf_ridge(
        x,
        y,
        sfreq=sfreq,
        lag_config=lag_cfg,
        alpha=float(cfg.alpha),
    )

    y_hat = predict_trf(x, fit)

    # -------------------------------------------------------------------------
    # Score: per-channel Pearson correlation on the first epoch
    # -------------------------------------------------------------------------
    #
    # Current clinical assumption: single epoch (or treat epoch 0 as representative).
    # If you later want multi-epoch aggregation, compute correlations per epoch and
    # summarize (mean/median) across epochs.
    #
    scores = _corr_per_channel(y[:, 0, :], y_hat[:, 0, :])

    # -------------------------------------------------------------------------
    # Channel naming / provenance
    # -------------------------------------------------------------------------
    #
    # If Y came from `_raw_to_epoched_y`, channel ordering is determined by picks.
    # We recompute picks here to derive channel names (safe as long as raw unchanged).
    #
    picks_for_names = _pick_seeg_ecog(raw)

    # -------------------------------------------------------------------------
    # Assemble report-ready TRFResult
    # -------------------------------------------------------------------------
    #
    # `TRFResult.extra` is a dumping ground for:
    # - reproducibility metadata (sfreq, config, backend extras)
    # - reporting assets (channel names, per-channel scores)
    # - warnings that should appear in a report but shouldn't abort analysis
    #
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
    """
    Pick sEEG/ECoG channels from a Raw.

    Parameters
    ----------
    raw
        Input Raw.

    Returns
    -------
    picks
        Integer indices of channels to use as neural outputs.

    Notes
    -----
    - We explicitly exclude EEG/MEG/stim/etc. to avoid mixing modalities.
    - If no channels are found, we raise with a helpful summary of channel types.
    """
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
    if picks.size == 0:  # noqa
        ch_type_counts = {t: raw.get_channel_types().count(t) for t in set(raw.get_channel_types())}
        raise ValueError(
            "No sEEG/ECoG channels available for TRF. "
            f"Channel types present: {ch_type_counts}."
        )
    return picks


def _raw_to_epoched_y(raw: mne.io.BaseRaw, picks: np.ndarray) -> np.ndarray:
    """
    Convert Raw data into an epoched-style Y array used by `analysis.trf`.

    Parameters
    ----------
    raw
        Input Raw (continuous time).
    picks
        Channel indices defining the outputs to model.

    Returns
    -------
    y
        Array shaped (n_times, n_epochs, n_outputs).

    Notes
    -----
    This runner currently operates in a "single epoch" mode to match the
    `analysis.trf` API, which supports both continuous and epoched input.
    Here we represent continuous data as a single epoch dimension of size 1.
    """
    # Raw.get_data returns (n_channels, n_times) for picked channels
    y_2d = raw.get_data(picks=picks)

    # Transpose into (n_times, n_outputs) expected by the TRF code
    y = np.transpose(y_2d, (1, 0))

    # Add singleton epoch dimension: (n_times, 1, n_outputs)
    return y[:, np.newaxis, :].astype(np.float32)


def _corr_per_channel(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Pearson correlation per channel over time.

    Parameters
    ----------
    y, y_hat
        Arrays of shape (n_times, n_outputs).

    Returns
    -------
    corr
        Correlation per output channel, shape (n_outputs,).

    Notes
    -----
    This is implemented explicitly (rather than calling SciPy) to:
    - avoid per-channel loops
    - keep tight control over broadcasting and dtype
    - make behavior stable across environments

    A small epsilon is added to the denominator to avoid divide-by-zero when
    either signal has zero variance.
    """
    # Center signals per channel
    y0 = y - np.mean(y, axis=0, keepdims=True)
    y1 = y_hat - np.mean(y_hat, axis=0, keepdims=True)

    # Compute correlation: E[y0*y1] / (std(y0)*std(y1))
    denom = (np.std(y0, axis=0) * np.std(y1, axis=0)) + 1e-12
    corr = np.mean(y0 * y1, axis=0) / denom
    return corr.astype(np.float32)
