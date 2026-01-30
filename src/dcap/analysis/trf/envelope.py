# =============================================================================
#                     TRF analysis: speech envelope
# =============================================================================
#
# Compute a broadband speech envelope using the Hilbert transform.
#
# This is intentionally minimal:
# - mono audio
# - no compression beyond optional power-law
# - resampling handled explicitly
#
# =============================================================================

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert, resample_poly

# review
# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class EnvelopeConfig:
    """
    Configuration for speech envelope extraction.

    Parameters
    ----------
    target_sfreq : float
        Sampling frequency of the output envelope (Hz).
    power : float, optional
        Power-law compression exponent applied to the envelope.
        Typical values: 1.0 (none), 0.5 (sqrt), 0.3.
    """

    target_sfreq: float
    power: float = 1.0


# =============================================================================
# Core API
# =============================================================================

def compute_speech_envelope(
    audio: np.ndarray,
    sfreq: float,
    config: EnvelopeConfig,
) -> np.ndarray:
    """
    Compute a broadband speech envelope using the Hilbert transform.

    Parameters
    ----------
    audio : ndarray, shape (n_samples,)
        Mono audio waveform.
    sfreq : float
        Sampling frequency of the input audio (Hz).
    config : EnvelopeConfig
        Envelope extraction configuration.

    Returns
    -------
    envelope : ndarray, shape (n_samples_out,)
        Speech envelope sampled at `config.target_sfreq`.

    Notes
    -----
    - Envelope is |Hilbert(audio)|.
    - Optional power-law compression is applied after envelope extraction.
    - Resampling uses polyphase filtering (scipy.signal.resample_poly).

    Usage example
    -------------
        cfg = EnvelopeConfig(target_sfreq=100.0, power=0.5)
        env = compute_speech_envelope(audio, sfreq=44100.0, config=cfg)
    """

    if audio.ndim != 1:
        raise ValueError("`audio` must be mono (1D array).")

    # -------------------------------------------------------------------------
    # Hilbert envelope
    # -------------------------------------------------------------------------
    analytic = hilbert(audio)
    envelope = np.abs(analytic)

    # -------------------------------------------------------------------------
    # Power-law compression
    # -------------------------------------------------------------------------
    if config.power != 1.0:
        if config.power <= 0:
            raise ValueError("`power` must be > 0.")
        envelope = envelope ** config.power

    # -------------------------------------------------------------------------
    # Resampling
    # -------------------------------------------------------------------------
    if sfreq != config.target_sfreq:
        up, down = _rational_approximation(config.target_sfreq / sfreq)
        envelope = resample_poly(envelope, up=up, down=down)

    return envelope


# =============================================================================
# Utilities
# =============================================================================

def _rational_approximation(ratio: float, max_denominator: int = 1000) -> tuple[int, int]:
    """
    Approximate a float ratio as a rational number for resample_poly.
    """
    from fractions import Fraction

    frac = Fraction(ratio).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator
