# =============================================================================
#                     ########################################
#                     #       BLOCK 4: FILTERING PATHS       #
#                     ########################################
# =============================================================================
#
# - High-pass filtering (drift removal)
# - Gamma/HFA envelope branch (feature-ready)
#
# Logic only:
# - No file I/O
# - No CLI / printing
#
# =============================================================================
from dataclasses import asdict
from typing import Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import GammaEnvelopeConfig, HighpassConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


# =============================================================================
#                              INTERNAL CONSTANTS
# =============================================================================
_DEFAULT_FIR_DESIGN: str = "firwin"
_DEFAULT_FIR_PHASE: str = "zero"
_ENVELOPE_CH_TYPE: str = "misc"
_ENVELOPE_SUFFIX: str = "HFAenv"


# =============================================================================
#                               HELPER FUNCTIONS
# =============================================================================
def _smooth_moving_average(data: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Smooth a (n_channels, n_times) array with a moving average.

    Parameters
    ----------
    data
        Array of shape (n_channels, n_times).
    window_samples
        Window length in samples. If <= 1, returns data unchanged.

    Returns
    -------
    smoothed
        Smoothed array.

    Usage example
    -------------
        smoothed = _smooth_moving_average(data, window_samples=101)
    """
    if window_samples <= 1:
        return data

    # Use convolution per channel (simple + dependency-free).
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)

    smoothed = np.empty_like(data, dtype=float)
    for channel_index in range(data.shape[0]):
        smoothed[channel_index] = np.convolve(data[channel_index], kernel, mode="same")

    return smoothed


def _make_envelope_raw(
    template_raw: mne.io.BaseRaw,
    envelope_data: np.ndarray,
    suffix: str,
) -> mne.io.BaseRaw:
    """
    Build a derived RawArray for envelope-like time series.

    Parameters
    ----------
    template_raw
        Source Raw whose sampling rate and metadata are used as reference.
    envelope_data
        Array of shape (n_channels, n_times).
    suffix
        Suffix appended to channel names.

    Returns
    -------
    env_raw
        Derived RawArray with channel names suffixed.

    Usage example
    -------------
        env_raw = _make_envelope_raw(raw, env_data, suffix="HFAenv")
    """
    if envelope_data.ndim != 2:
        raise ValueError(f"Expected 2D data (n_channels, n_times), got {envelope_data.ndim}D.")

    if envelope_data.shape[1] != template_raw.n_times:
        raise ValueError("Envelope data length does not match template raw length.")

    channel_names = [f"{name}_{suffix}" for name in template_raw.ch_names]
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=float(template_raw.info["sfreq"]),
        ch_types=[_ENVELOPE_CH_TYPE] * len(channel_names),
    )

    env_raw = mne.io.RawArray(envelope_data, info, verbose=False)

    # Carry annotations forward so downstream epoching can still respect them.
    env_raw.set_annotations(template_raw.annotations.copy())

    return env_raw


# =============================================================================
#                          BLOCK 4A: HIGH-PASS FILTER
# =============================================================================
def highpass_filter(
    raw: mne.io.BaseRaw,
    cfg: HighpassConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    High-pass filter for drift removal.

    Parameters
    ----------
    raw
        MNE Raw object.
    cfg
        High-pass configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        High-pass filtered Raw (a copy).
    artifact
        Block artifact.

    Usage example
    -------------
        ctx = PreprocContext()
        raw_hp, artifact = highpass_filter(raw, HighpassConfig(l_freq=0.5), ctx)
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("highpass_filter expects an mne.io.BaseRaw.")

    if cfg.l_freq <= 0:
        raise ValueError(f"High-pass cutoff must be > 0 Hz, got {cfg.l_freq}.")

    ctx.add_record("highpass", asdict(cfg))

    raw_out = raw.copy()
    raw_out.filter(
        l_freq=float(cfg.l_freq),
        h_freq=None,
        picks=None,
        method="fir",
        phase=_DEFAULT_FIR_PHASE if cfg.phase == "zero" else "minimum",
        fir_design=_DEFAULT_FIR_DESIGN,
        verbose=False,
    )

    artifact = BlockArtifact(
        name="highpass",
        parameters=asdict(cfg),
        summary_metrics={
            "l_freq_hz": float(cfg.l_freq),
            "phase": cfg.phase,
        },
        warnings=[],
        figures=[],
    )
    return raw_out, artifact


# =============================================================================
#                      BLOCK 4B: GAMMA / HFA ENVELOPE PATH
# =============================================================================
def compute_gamma_envelope(
    raw: mne.io.BaseRaw,
    cfg: GammaEnvelopeConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Compute gamma/HFA envelope time series.

    This is a *feature path* (not a generic preprocessing step). The output is a
    derived Raw where channels represent envelope values, not voltage.

    Parameters
    ----------
    raw
        MNE Raw object.
    cfg
        Gamma envelope configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    env_raw
        Derived Raw (RawArray) containing envelope values.
    artifact
        Block artifact.

    Notes
    -----
    - The output channel names are suffixed with `_HFAenv` to prevent confusion
      with voltage channels.
    - This function band-pass filters to cfg.band_hz then extracts an envelope.

    Usage example
    -------------
        ctx = PreprocContext()
        env_raw, artifact = compute_gamma_envelope(
            raw,
            GammaEnvelopeConfig(band_hz=(70.0, 150.0), method="hilbert", smoothing_sec=0.1),
            ctx,
        )
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("compute_gamma_envelope expects an mne.io.BaseRaw.")

    low_hz, high_hz = float(cfg.band_hz[0]), float(cfg.band_hz[1])
    if not (0 < low_hz < high_hz):
        raise ValueError(f"Invalid band_hz: {cfg.band_hz}. Must satisfy 0 < low < high.")

    if cfg.smoothing_sec < 0:
        raise ValueError(f"smoothing_sec must be >= 0, got {cfg.smoothing_sec}.")

    ctx.add_record("gamma_envelope", asdict(cfg))

    sfreq = float(raw.info["sfreq"])
    smoothing_samples = int(round(cfg.smoothing_sec * sfreq))

    # Band-pass copy
    band_raw = raw.copy()
    band_raw.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        picks=None,
        method="fir",
        phase=_DEFAULT_FIR_PHASE,
        fir_design=_DEFAULT_FIR_DESIGN,
        verbose=False,
    )

    if cfg.method == "hilbert":
        # MNE handles Hilbert neatly for Raw; envelope=True returns magnitude.
        band_raw.apply_hilbert(envelope=True, verbose=False)
        envelope_data = band_raw.get_data()

    elif cfg.method == "rectified_smooth":
        # Rectify then smooth.
        envelope_data = np.abs(band_raw.get_data())

    else:
        raise ValueError(f"Unknown gamma envelope method: {cfg.method!r}")

    if smoothing_samples > 1:
        envelope_data = _smooth_moving_average(envelope_data, window_samples=smoothing_samples)

    env_raw = _make_envelope_raw(raw, envelope_data, suffix=_ENVELOPE_SUFFIX)

    warnings: list[str] = []
    if cfg.smoothing_sec == 0:
        warnings.append("No smoothing applied to envelope (smoothing_sec=0).")
    if cfg.method == "hilbert":
        warnings.append("Envelope computed via Hilbert magnitude on band-passed data.")
    else:
        warnings.append("Envelope computed via rectification on band-passed data.")

    artifact = BlockArtifact(
        name="gamma_envelope",
        parameters=asdict(cfg),
        summary_metrics={
            "band_low_hz": low_hz,
            "band_high_hz": high_hz,
            "smoothing_sec": float(cfg.smoothing_sec),
            "smoothing_samples": int(smoothing_samples),
            "output_suffix": _ENVELOPE_SUFFIX,
        },
        warnings=warnings,
        figures=[],
    )
    return env_raw, artifact
