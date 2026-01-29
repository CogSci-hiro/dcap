# =============================================================================
#                       Analysis: TRF (speech envelope)
# =============================================================================
#
# Minimal speech-envelope extraction utilities.
#
# This module intentionally does NOT:
# - decode audio files
# - assume any BIDS layout
#
# Instead, higher-level code should load audio into a 1D float array
# (mono waveform) and call these functions.
#
# =============================================================================

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dcap.analysis.trf.types import EnvelopeConfig


FloatArray = NDArray[np.floating]


def compute_speech_envelope(
    audio: FloatArray,
    sfreq: float,
    cfg: EnvelopeConfig,
) -> Tuple[FloatArray, float]:
    """Compute a speech envelope from a mono audio waveform.

    Parameters
    ----------
    audio
        1D mono waveform, shape (n_samples,).
    sfreq
        Sampling rate of `audio` in Hz.
    cfg
        Envelope extraction configuration.

    Returns
    -------
    envelope
        Envelope time series, resampled to `cfg.target_sfreq` if needed.
    envelope_sfreq
        Sampling rate of returned envelope (Hz).

    Notes
    -----
    This is a skeleton. The intended baseline pipeline is:
    1) rectify (optional)
    2) amplitude envelope (e.g., Hilbert magnitude)
    3) low-pass filter (optional)
    4) resample to `cfg.target_sfreq`

    Usage example
    -------------
        import numpy as np
        from dcap.analysis.trf import EnvelopeConfig, compute_speech_envelope

        audio = np.random.randn(48000).astype(float)
        env, env_sfreq = compute_speech_envelope(
            audio=audio,
            sfreq=48000.0,
            cfg=EnvelopeConfig(target_sfreq=200.0),
        )
    """
    _validate_audio_1d(audio=audio)
    _validate_positive_float(name="sfreq", value=sfreq)
    _validate_positive_float(name="cfg.target_sfreq", value=cfg.target_sfreq)

    raise NotImplementedError(
        "TODO: implement envelope extraction (Hilbert/rectify/lowpass/resample)."

        "This skeleton deliberately leaves the DSP details for the next step."
    )


def _validate_audio_1d(audio: FloatArray) -> None:
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1D, got shape={audio.shape!r}.")
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError(f"audio must be floating dtype, got dtype={audio.dtype!r}.")


def _validate_positive_float(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")
