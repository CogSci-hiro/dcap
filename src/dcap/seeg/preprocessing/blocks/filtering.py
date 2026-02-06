# =============================================================================
# =============================================================================
#                     ########################################
#                     #       BLOCK 4: FILTERING PATHS       #
#                     ########################################
# =============================================================================
# =============================================================================
"""
Filtering utilities.

Two APIs
--------
1) Analysis-friendly API (library style)
   - highpass(raw, ...) -> Raw
   - gamma_envelope(raw, ...) -> Raw

2) Clinical/report wrappers
   - highpass_view(raw, cfg, ctx) -> (Raw, BlockArtifact)
   - gamma_envelope_view(raw, cfg, ctx) -> (Raw, BlockArtifact)

Notes
-----
- Highpass uses MNE Raw.filter with FIR and configurable phase.
- Gamma envelope is computed by band-pass -> envelope extraction -> optional smoothing.
- Envelope output is a derived RawArray. We preserve annotations.
"""

from dataclasses import asdict
from typing import Literal, Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import GammaEnvelopeConfig, HighpassConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


_DEFAULT_FIR_DESIGN: str = "firwin"
_DEFAULT_ENVELOPE_SUFFIX: str = "HFAenv"
_DEFAULT_ENVELOPE_CH_TYPE: str = "misc"


# =============================================================================
#                     ########################################
#                     #              Internals               #
#                     ########################################
# =============================================================================
def _smooth_moving_average(data_ch_time: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Channel-wise moving average smoothing.

    Parameters
    ----------
    data_ch_time
        Array (n_channels, n_times).
    window_samples
        Window length in samples.

    Returns
    -------
    smoothed
        Smoothed array of same shape.
    """
    if window_samples <= 1:
        return data_ch_time

    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    smoothed = np.empty_like(data_ch_time, dtype=float)
    for channel_index in range(data_ch_time.shape[0]):
        smoothed[channel_index] = np.convolve(data_ch_time[channel_index], kernel, mode="same")
    return smoothed


def _copy_raw_with_data(template_raw: mne.io.BaseRaw, data: np.ndarray) -> mne.io.BaseRaw:
    """
    Create a RawArray that preserves Info + annotations.

    This is handy when we compute a new data matrix for the same channels.
    """
    info = template_raw.info.copy()
    out = mne.io.RawArray(data, info, verbose=False)
    out.set_annotations(template_raw.annotations.copy())
    return out


# =============================================================================
#                     ########################################
#                     #          Public API (analysis)        #
#                     ########################################
# =============================================================================
def highpass(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    *,
    phase: Literal["zero", "minimum"] = "zero",
    fir_design: str = _DEFAULT_FIR_DESIGN,
    copy: bool = True,
) -> mne.io.BaseRaw:
    """
    High-pass filter for drift removal.

    Parameters
    ----------
    raw
        Input Raw.
    l_freq
        High-pass cutoff (Hz).
    phase
        "zero" (zero-phase, non-causal) or "minimum" (causal-ish).
    fir_design
        FIR design passed to MNE (default "firwin").
    copy
        If True, returns a filtered copy. If False, filters in-place.

    Returns
    -------
    raw_out
        Filtered Raw.

    Usage example
    -------------
        raw_hp = highpass(raw, l_freq=1.0, phase="zero")
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("highpass expects an mne.io.BaseRaw.")

    raw_out = raw.copy() if copy else raw
    raw_out.filter(
        l_freq=float(l_freq),
        h_freq=None,
        method="fir",
        phase="zero" if phase == "zero" else "minimum",
        fir_design=str(fir_design),
        verbose=False,
    )
    return raw_out


def gamma_envelope(
    raw: mne.io.BaseRaw,
    band_hz: Tuple[float, float] = (70.0, 150.0),
    *,
    method: Literal["hilbert", "rectified_smooth"] = "hilbert",
    smoothing_sec: float = 0.05,
    suffix: str = _DEFAULT_ENVELOPE_SUFFIX,
    out_ch_type: str = _DEFAULT_ENVELOPE_CH_TYPE,
    copy: bool = True,
) -> mne.io.BaseRaw:
    """
    Compute gamma/HFA envelope as a derived RawArray.

    Pipeline
    --------
    1) Band-pass filter into [low_hz, high_hz]
    2) Envelope:
         - "hilbert": magnitude of analytic signal (MNE apply_hilbert(envelope=True))
         - "rectified_smooth": abs() (then optional smoothing)
    3) Optional moving-average smoothing (window = smoothing_sec * sfreq)
    4) Return new RawArray with (optionally) suffixed channel names

    Parameters
    ----------
    raw
        Input Raw.
    band_hz
        (low_hz, high_hz) band.
    method
        Envelope extraction method.
    smoothing_sec
        Moving average smoothing window in seconds. Use 0 for no smoothing.
    suffix
        Channel name suffix appended as "<name>_<suffix>" to indicate derived envelope.
        If empty string, keep original names.
    out_ch_type
        Channel type for envelope (often "misc").
    copy
        If True, band-pass operates on a copy of raw. (Recommended.)

    Returns
    -------
    env_raw
        Envelope RawArray with preserved annotations.

    Usage example
    -------------
        env = gamma_envelope(raw, band_hz=(70, 150), method="hilbert", smoothing_sec=0.05)
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("gamma_envelope expects an mne.io.BaseRaw.")

    sfreq = float(raw.info["sfreq"])
    low_hz, high_hz = float(band_hz[0]), float(band_hz[1])

    # Band-pass on a copy so we don't mutate the input Raw
    band_raw = raw.copy() if copy else raw
    band_raw.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method="fir",
        phase="zero",
        fir_design=_DEFAULT_FIR_DESIGN,
        verbose=False,
    )

    warnings: list[str] = []

    if method == "hilbert":
        band_raw.apply_hilbert(envelope=True, verbose=False)
        envelope_data = band_raw.get_data()
        warnings.append("Envelope computed via Hilbert magnitude on band-passed data.")
    elif method == "rectified_smooth":
        envelope_data = np.abs(band_raw.get_data())
        warnings.append("Envelope computed via rectification on band-passed data.")
    else:
        raise ValueError(f"Unknown gamma envelope method: {method!r}")

    # Optional smoothing
    if smoothing_sec <= 0:
        warnings.append("No smoothing applied to envelope (smoothing_sec<=0).")
    else:
        window_samples = int(round(float(smoothing_sec) * sfreq))
        envelope_data = _smooth_moving_average(envelope_data, window_samples=window_samples)

    # Build output channel names (fixes the current bug where suffix info is overwritten)
    if suffix:
        ch_names = [f"{name}_{suffix}" for name in raw.ch_names]
    else:
        ch_names = list(raw.ch_names)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=[str(out_ch_type)] * len(ch_names),
    )

    env_raw = mne.io.RawArray(envelope_data, info, verbose=False)
    env_raw.set_annotations(raw.annotations.copy())

    # (Optional) copy montage/dig if you want envelope channels to be spatially located.
    # In many pipelines, leaving this out is fine because envelope channels are derived.
    # If you want: env_raw.info["dig"] = raw.info.get("dig", None)

    # Attach warnings for the caller if they care (analysis API currently returns only Raw).
    # If you want warnings in analysis API too, we can return (raw, warnings).
    _ = warnings

    return env_raw


# =============================================================================
#                     ########################################
#                     #       Compat API (clinical block)     #
#                     ########################################
# =============================================================================
def highpass_view(
    raw: mne.io.BaseRaw,
    cfg: HighpassConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Clinical wrapper for highpass.

    Records provenance and returns a BlockArtifact for report/QC.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("highpass_view expects an mne.io.BaseRaw.")

    ctx.add_record("highpass", asdict(cfg))

    raw_out = highpass(
        raw,
        l_freq=float(cfg.l_freq),
        phase="zero" if cfg.phase == "zero" else "minimum",
        fir_design=_DEFAULT_FIR_DESIGN,
        copy=True,
    )

    artifact = BlockArtifact(
        name="highpass",
        parameters=asdict(cfg),
        summary_metrics={"l_freq_hz": float(cfg.l_freq)},
        warnings=[],
        figures=[],
    )
    return raw_out, artifact


def gamma_envelope_view(
    raw: mne.io.BaseRaw,
    cfg: GammaEnvelopeConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Clinical wrapper for gamma envelope.

    Records provenance and returns BlockArtifact with band/smoothing details.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("gamma_envelope_view expects an mne.io.BaseRaw.")

    ctx.add_record("gamma_envelope", asdict(cfg))

    low_hz, high_hz = float(cfg.band_hz[0]), float(cfg.band_hz[1])

    warnings: list[str] = []

    env_raw = gamma_envelope(
        raw,
        band_hz=(low_hz, high_hz),
        method=str(cfg.method),  # type: ignore[arg-type]
        smoothing_sec=float(cfg.smoothing_sec),
        suffix=_DEFAULT_ENVELOPE_SUFFIX,
        out_ch_type=_DEFAULT_ENVELOPE_CH_TYPE,
        copy=True,
    )

    # Mirror the warnings produced by the analysis implementation for report/QC.
    # (We recompute them based on cfg for simplicity.)
    if str(cfg.method) == "hilbert":
        warnings.append("Envelope computed via Hilbert magnitude on band-passed data.")
    else:
        warnings.append("Envelope computed via rectification on band-passed data.")
    if float(cfg.smoothing_sec) <= 0:
        warnings.append("No smoothing applied to envelope (smoothing_sec<=0).")

    artifact = BlockArtifact(
        name="gamma_envelope",
        parameters=asdict(cfg),
        summary_metrics={
            "band_low_hz": low_hz,
            "band_high_hz": high_hz,
            "smoothing_sec": float(cfg.smoothing_sec),
        },
        warnings=warnings,
        figures=[],
    )
    return env_raw, artifact


# Optional compat aliases (if you want minimal downstream churn):
highpass_filter = highpass_view
compute_gamma_envelope = gamma_envelope_view
