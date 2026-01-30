# =============================================================================
# =============================================================================
#                     ########################################
#                     #       BLOCK 4: FILTERING PATHS       #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import asdict
from typing import Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import GammaEnvelopeConfig, HighpassConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


_DEFAULT_FIR_DESIGN: str = "firwin"
_ENVELOPE_SUFFIX: str = "HFAenv"


def _smooth_moving_average(data: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return data
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    smoothed = np.empty_like(data, dtype=float)
    for channel_index in range(data.shape[0]):
        smoothed[channel_index] = np.convolve(data[channel_index], kernel, mode="same")
    return smoothed


def highpass_filter(
    raw: mne.io.BaseRaw,
    cfg: HighpassConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """High-pass filter for drift removal."""
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("highpass_filter expects an mne.io.BaseRaw.")

    ctx.add_record("highpass", asdict(cfg))
    raw_out = raw.copy()
    raw_out.filter(
        l_freq=float(cfg.l_freq),
        h_freq=None,
        method="fir",
        phase="zero" if cfg.phase == "zero" else "minimum",
        fir_design=_DEFAULT_FIR_DESIGN,
        verbose=False,
    )

    artifact = BlockArtifact(
        name="highpass",
        parameters=asdict(cfg),
        summary_metrics={"l_freq_hz": float(cfg.l_freq)},
        warnings=[],
        figures=[],
    )
    return raw_out, artifact


def compute_gamma_envelope(
    raw: mne.io.BaseRaw,
    cfg: GammaEnvelopeConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """Compute gamma/HFA envelope time series as a derived RawArray."""
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("compute_gamma_envelope expects an mne.io.BaseRaw.")

    ctx.add_record("gamma_envelope", asdict(cfg))

    sfreq = float(raw.info["sfreq"])
    smoothing_samples = int(round(cfg.smoothing_sec * sfreq))

    low_hz, high_hz = float(cfg.band_hz[0]), float(cfg.band_hz[1])

    band_raw = raw.copy()
    band_raw.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method="fir",
        phase="zero",
        fir_design=_DEFAULT_FIR_DESIGN,
        verbose=False,
    )

    if cfg.method == "hilbert":
        band_raw.apply_hilbert(envelope=True, verbose=False)
        envelope_data = band_raw.get_data()
        warnings = ["Envelope computed via Hilbert magnitude on band-passed data."]
    elif cfg.method == "rectified_smooth":
        envelope_data = np.abs(band_raw.get_data())
        warnings = ["Envelope computed via rectification on band-passed data."]
    else:
        raise ValueError(f"Unknown gamma envelope method: {cfg.method!r}")

    if smoothing_samples > 1:
        envelope_data = _smooth_moving_average(envelope_data, window_samples=smoothing_samples)
    elif cfg.smoothing_sec == 0:
        warnings.append("No smoothing applied to envelope (smoothing_sec=0).")

    channel_names = [f"{name}_{_ENVELOPE_SUFFIX}" for name in raw.ch_names]
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["misc"] * len(channel_names),
    )

    ch_types = raw.get_channel_types()
    info = mne.create_info(
        ch_names=raw.ch_names,
        sfreq=float(raw.info["sfreq"]),
        ch_types=ch_types,
    )
    env_raw = mne.io.RawArray(envelope_data, info, verbose=False)
    env_raw.set_annotations(raw.annotations.copy())

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
