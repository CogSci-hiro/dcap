# =============================================================================
# =============================================================================
#                 ############################################
#                 #           BLOCK: COCHLEOGRAM             #
#                 ############################################
# =============================================================================
# =============================================================================
"""
Cochleogram feature (auditory filterbank envelope representation).

This module provides a dcap-native feature computer that produces a cochleogram:
a time-frequency representation meant to approximate early auditory processing.

The core idea
-------------
Given a mono waveform x(t), we create N frequency channels (bands), filter the waveform
into each band, extract an amplitude envelope per band, and optionally low-pass that
envelope (a crude "synapse" stage).

We keep two computation modes:
- mode="efficient": uses bandpass Butterworth filters (SOS) per band (fast, stable).
- mode="accurate": uses FIR gammatone filters per band (slower, closer to classic
  gammatone filterbank implementations).

Output contract
---------------
- FeatureResult.values has shape (n_channels, n_times)  # channels-first for TRF
- FeatureResult.time is exactly the provided FeatureTimeBase
- channel_names encode center frequencies

Config knobs
------------
- Frequency channels are spaced in "octaves" (log2 frequency space) with step `octave_spacing`.
- Center frequencies are between [f_min_hz, f_max_hz].
- Optional early downsampling of envelopes to `env_target_fs_hz` for speed,
  then we resample again to FeatureTimeBase.sfreq if needed.

Important: calibration
----------------------
Absolute units are arbitrary unless your waveform is calibrated in Pascals. For TRF,
z-scoring/normalizing features downstream is typically recommended.

Usage example
-------------
    from dcap.features.types import FeatureTimeBase
    from dcap.features.acoustic.cochleogram import CochleogramComputer, CochleogramConfig

    time = FeatureTimeBase(sfreq=100.0, n_times=30_000, t0_s=0.0)
    cfg = CochleogramConfig(mode="efficient", env_target_fs_hz=200.0, derivative="none")

    comp = CochleogramComputer()
    out = comp.compute(time=time, audio=wav, audio_sfreq=16_000.0, config=cfg)

    # out.values: (n_channels, n_times)
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
from scipy.signal import butter, hilbert, lfilter, resample_poly, sosfiltfilt

from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.postprocess import apply_derivative
from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase


# =============================================================================
#                             CONSTANTS / DEFAULTS
# =============================================================================

_DEFAULT_DOWNSMIX: str = "mean"

# Numerical stability floors
_TINY: float = 1e-12

# If we build FIR gammatone filters, we need a finite impulse response length.
# This is a speed/accuracy tradeoff; longer -> sharper/cleaner, slower.
# We scale this length with the center frequency's time constant.
_MAX_FIR_SECONDS: float = 0.08  # cap FIR to 80 ms (prevents huge filters at low freqs)


# =============================================================================
#                         HELPERS: DOWNMIX + RESAMPLING
# =============================================================================

def _as_mono_audio(*, audio: np.ndarray, downmix: str) -> np.ndarray:
    """
    Convert audio to mono 1D waveform.

    Parameters
    ----------
    audio
        Shape (n_times,) or (n_channels, n_times).
    downmix
        "mean" averages channels.

    Returns
    -------
    np.ndarray
        Shape (n_times,)
    """
    x = np.asarray(audio, dtype=float)

    if x.ndim == 1:
        return x

    if x.ndim != 2:
        raise ValueError(f"audio must be 1D or 2D (channels x time). Got shape={x.shape}.")

    n_channels, _ = x.shape
    if n_channels == 1:
        return x[0]

    if downmix == "mean":
        return np.mean(x, axis=0)

    raise ValueError(f"Unknown downmix='{downmix}'. Supported: 'mean'.")


def _rational_approximation(*, ratio: float, max_denominator: int) -> tuple[int, int]:
    """
    Find integers (up, down) such that up/down ≈ ratio.
    Used for polyphase resampling with resample_poly.
    """
    from fractions import Fraction

    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def _resample_1d(*, x: np.ndarray, input_sfreq: float, output_sfreq: float) -> np.ndarray:
    """
    Resample a 1D signal with polyphase filtering.

    Notes
    -----
    - Good anti-aliasing behavior.
    - Deterministic and stable.
    """
    if input_sfreq <= 0 or output_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    if np.isclose(input_sfreq, output_sfreq):
        return x

    ratio = float(output_sfreq) / float(input_sfreq)
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x, up=up, down=down)


def _resample_2d_time_major(*, x_tf: np.ndarray, input_sfreq: float, output_sfreq: float) -> np.ndarray:
    """
    Resample a (T, C) array along time only.
    """
    if input_sfreq <= 0 or output_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    if np.isclose(input_sfreq, output_sfreq):
        return x_tf

    ratio = float(output_sfreq) / float(input_sfreq)
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x_tf, up=up, down=down, axis=0)


def _force_length_time_major(*, x_tf: np.ndarray, n_times: int) -> np.ndarray:
    """
    Crop/pad a (T, C) array to exactly n_times along the time axis.
    """
    if n_times <= 0:
        raise ValueError("n_times must be positive.")

    if x_tf.shape[0] == n_times:
        return x_tf

    if x_tf.shape[0] > n_times:
        return x_tf[:n_times, :]

    pad = np.zeros((n_times - x_tf.shape[0], x_tf.shape[1]), dtype=x_tf.dtype)
    return np.concatenate([x_tf, pad], axis=0)


# =============================================================================
#                         ERB SCALE + BAND DEFINITIONS
# =============================================================================

def _erb_rate_hz_to_erb(f_hz: np.ndarray) -> np.ndarray:
    """
    Glasberg & Moore ERB-rate scale.

    ERBrate(f) = 21.4 log10(4.37e-3 f + 1)
    """
    return 21.4 * np.log10(4.37e-3 * f_hz + 1.0)


def _erb_rate_erb_to_hz(e: np.ndarray) -> np.ndarray:
    """
    Inverse ERB-rate scale.
    """
    return (10.0 ** (e / 21.4) - 1.0) / 4.37e-3


def _erb_bandwidth_hz(f_hz: np.ndarray) -> np.ndarray:
    """
    ERB bandwidth in Hz (Glasberg & Moore).

    ERB(f) = 24.7(4.37e-3 f + 1)
    """
    return 24.7 * (4.37e-3 * f_hz + 1.0)


def _octave_spaced_centers(*, f_min_hz: float, f_max_hz: float, octave_spacing: float) -> np.ndarray:
    """
    Build center frequencies spaced uniformly in log2 frequency space.

    If octave_spacing = 1.0: centers are one octave apart.
    If octave_spacing = 0.1: 10 bands per octave.

    Returns
    -------
    centers_hz : np.ndarray
        Increasing, within [f_min_hz, f_max_hz]
    """
    if f_min_hz <= 0 or f_max_hz <= 0 or f_max_hz <= f_min_hz:
        raise ValueError("Invalid f_min_hz/f_max_hz.")
    if octave_spacing <= 0:
        raise ValueError("octave_spacing must be positive.")

    lo = np.log2(f_min_hz)
    hi = np.log2(f_max_hz)

    # Include hi by extending slightly; then clip.
    n = int(np.floor((hi - lo) / octave_spacing)) + 1
    exponents = lo + octave_spacing * np.arange(n, dtype=float)
    centers = 2.0 ** exponents

    # Ensure last center <= f_max_hz
    centers = centers[centers <= f_max_hz + 1e-9]
    centers = np.maximum(centers, f_min_hz)
    return centers.astype(float)


def _erb_band_edges(*, centers_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Define 1-ERB-wide band edges around each center frequency.

    We approximate gammatone channel bandwidth using ERB.
    """
    bw = _erb_bandwidth_hz(centers_hz)
    lows = np.maximum(1.0, centers_hz - 0.5 * bw)
    highs = centers_hz + 0.5 * bw
    return lows, highs


# =============================================================================
#                        FILTER BACKENDS (EFFICIENT vs ACCURATE)
# =============================================================================

def _butter_bandpass_sos(*, low_hz: float, high_hz: float, fs_hz: float, order: int) -> np.ndarray:
    """
    Design a Butterworth bandpass filter in SOS form.

    Notes
    -----
    - Using SOS avoids numerical issues for higher orders.
    - We use zero-phase application via sosfiltfilt in the efficient mode.
    """
    nyq = 0.5 * fs_hz
    if not (0.0 < low_hz < high_hz < nyq):
        raise ValueError(f"Invalid band edges low={low_hz}, high={high_hz}, nyq={nyq}.")
    wn = [low_hz / nyq, high_hz / nyq]
    return butter(order, wn, btype="bandpass", output="sos")


def _gammatone_fir(*, center_hz: float, fs_hz: float, order: int, tau_factor: float) -> np.ndarray:
    """
    Build a real FIR approximation of a gammatone filter impulse response.

    This is a pragmatic implementation:
    - classic gammatone has impulse response:
        g(t) = t^(n-1) * exp(-2π b t) * cos(2π f_c t)
      where b is bandwidth parameter (~ ERB).
    - We choose b = tau_factor / (2π) * ERB(f_c) ??? (repos differ).
      Here we interpret tau_factor as a multiplier on the ERB bandwidth.

    We normalize energy to keep channel magnitudes comparable.

    This is not guaranteed to match any specific toolbox exactly, but it provides a
    "more auditory-like" bandpass than Butterworth and is deterministic.

    Returns
    -------
    h : np.ndarray
        FIR impulse response, shape (n_taps,)
    """
    if center_hz <= 0:
        raise ValueError("center_hz must be positive.")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if order <= 0:
        raise ValueError("order must be positive.")
    if tau_factor <= 0:
        raise ValueError("tau_factor must be positive.")

    erb = float(_erb_bandwidth_hz(np.array([center_hz]))[0])
    # Bandwidth parameter b (Hz). We keep it proportional to ERB.
    b_hz = tau_factor * erb

    # Choose FIR duration based on decay time constant. We ensure enough decay for stable response.
    # Rough decay time ~ 1/(2π b). We take several time constants but cap to _MAX_FIR_SECONDS.
    tau_decay = 1.0 / (2.0 * np.pi * max(b_hz, 1e-6))
    dur_s = min(_MAX_FIR_SECONDS, 6.0 * tau_decay)  # 6 time constants
    n_taps = int(max(64, np.round(dur_s * fs_hz)))  # enforce minimum taps

    t = np.arange(n_taps, dtype=float) / fs_hz
    # Gammatone envelope term t^(n-1) exp(-2π b t)
    env = (t ** (order - 1)) * np.exp(-2.0 * np.pi * b_hz * t)
    carrier = np.cos(2.0 * np.pi * center_hz * t)
    h = env * carrier

    # Normalize: unit L2 energy (prevents huge scaling differences across bands)
    norm = float(np.sqrt(np.sum(h * h))) + _TINY
    h = h / norm
    return h.astype(float)


# =============================================================================
#                        ENVELOPE EXTRACTION + "SYNAPSE" LPF
# =============================================================================

def _band_envelope(*, band_signal: np.ndarray) -> np.ndarray:
    """
    Envelope extraction for a band-limited signal.

    We use |Hilbert(band_signal)| (analytic magnitude), which is standard for subband envelopes.
    """
    return np.abs(hilbert(band_signal))


def _lowpass_sos(*, cutoff_hz: float, fs_hz: float, order: int) -> np.ndarray:
    """
    SOS coefficients for a low-pass Butterworth.
    """
    nyq = 0.5 * fs_hz
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive.")
    if cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz must be < Nyquist={nyq} for fs={fs_hz}.")
    return butter(order, cutoff_hz / nyq, btype="low", output="sos")


# =============================================================================
#                              ENGINE: CORE COMPUTATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class CochleogramEngineConfig:
    """
    Low-level cochleogram engine config.

    Parameters
    ----------
    mode
        "efficient" (Butterworth) or "accurate" (FIR gammatone).
    f_min_hz, f_max_hz
        Frequency range of channels.
    octave_spacing
        Channel spacing in octaves (log2 spacing).
    gammatone_order, gammatone_tau_factor
        Only used in mode="accurate".
    bandpass_order
        Only used in mode="efficient".
    synapse_lowpass_cutoff_hz, synapse_lowpass_order
        Low-pass applied to each band envelope. Set cutoff_hz=None to disable.
    env_target_fs_hz
        If not None, downsample the envelope (per channel) early to this rate to speed up.
        This is done *after* envelope extraction and synapse low-pass.
    """
    mode: Literal["accurate", "efficient"] = "efficient"

    f_min_hz: float = 100.0
    f_max_hz: float = 10_000.0
    octave_spacing: float = 0.05

    gammatone_order: int = 4
    gammatone_tau_factor: float = 4.0

    bandpass_order: int = 4  # used in efficient mode

    synapse_lowpass_cutoff_hz: Optional[float] = 30.0
    synapse_lowpass_order: int = 2

    env_target_fs_hz: Optional[float] = None


class _CochleogramEngine:
    """
    Compute a cochleogram in a backend-agnostic way.

    Returns (time-major):
    - coch_tf: shape (T_env, C)
    - center_frequencies_hz: shape (C,)
    - fs_env_hz: sampling frequency of coch_tf along time axis
    - x_octaves: log2(center / f_min), shape (C,)
    """

    def __init__(self, config: CochleogramEngineConfig) -> None:
        self._cfg = config

    def apply(
        self,
        *,
        audio_mono: np.ndarray,
        fs_hz: float,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Compute cochleogram for a mono waveform.

        Parameters
        ----------
        audio_mono
            Waveform samples, shape (n_times,).
        fs_hz
            Sampling frequency of the waveform.

        Returns
        -------
        coch_tf
            (T_env, C) float array of envelopes per channel.
        center_frequencies_hz
            (C,) float array.
        fs_env_hz
            Sampling frequency corresponding to coch_tf time axis.
        x_octaves
            (C,) float array of channel positions in octaves from f_min.
        """
        x = np.asarray(audio_mono, dtype=float).reshape(-1)
        if x.size == 0:
            raise ValueError("audio_mono is empty.")
        if fs_hz <= 0:
            raise ValueError("fs_hz must be positive.")

        # Build center frequencies
        centers_hz = _octave_spaced_centers(
            f_min_hz=float(self._cfg.f_min_hz),
            f_max_hz=float(self._cfg.f_max_hz),
            octave_spacing=float(self._cfg.octave_spacing),
        )
        if centers_hz.size == 0:
            raise ValueError("No center frequencies produced. Check f_min_hz/f_max_hz/octave_spacing.")

        x_octaves = np.log2(centers_hz / float(self._cfg.f_min_hz))

        # Define band edges (1 ERB wide)
        lows_hz, highs_hz = _erb_band_edges(centers_hz=centers_hz)

        # Clip edges below Nyquist, and ensure low<high
        nyq = 0.5 * fs_hz
        highs_hz = np.minimum(highs_hz, 0.999 * nyq)
        lows_hz = np.minimum(lows_hz, 0.998 * nyq)

        n_channels = int(centers_hz.size)
        # We'll compute envelopes at audio rate first (then optional early downsample)
        env_tc = np.empty((x.shape[0], n_channels), dtype=float)

        # Precompute synapse LPF if enabled
        syn_sos: Optional[np.ndarray]
        if self._cfg.synapse_lowpass_cutoff_hz is None:
            syn_sos = None
        else:
            syn_sos = _lowpass_sos(
                cutoff_hz=float(self._cfg.synapse_lowpass_cutoff_hz),
                fs_hz=float(fs_hz),
                order=int(self._cfg.synapse_lowpass_order),
            )

        # Channel-by-channel computation (explicit loop; easier to reason about, less "magic")
        for ch in range(n_channels):
            lo = float(lows_hz[ch])
            hi = float(highs_hz[ch])
            cf = float(centers_hz[ch])

            # If the band is invalid (can happen near Nyquist), produce zeros to keep dimensions stable.
            if not (0.0 < lo < hi < nyq):
                env_tc[:, ch] = 0.0
                continue

            if self._cfg.mode == "efficient":
                # Efficient backend: Butterworth bandpass, applied zero-phase (no delay).
                sos = _butter_bandpass_sos(
                    low_hz=lo,
                    high_hz=hi,
                    fs_hz=float(fs_hz),
                    order=int(self._cfg.bandpass_order),
                )
                band_sig = sosfiltfilt(sos, x)

            elif self._cfg.mode == "accurate":
                # Accurate backend: FIR gammatone.
                # We convolve with lfilter (causal) then we accept the inherent delay.
                # For envelopes that are heavily low-passed and later resampled, this is usually fine.
                # If you want exact zero-phase, we'd need filtfilt-style application, but FIR filtfilt is slower.
                h = _gammatone_fir(
                    center_hz=cf,
                    fs_hz=float(fs_hz),
                    order=int(self._cfg.gammatone_order),
                    tau_factor=float(self._cfg.gammatone_tau_factor),
                )
                band_sig = lfilter(h, [1.0], x)

            else:
                raise ValueError("mode must be 'efficient' or 'accurate'.")

            # Envelope extraction per band
            env = _band_envelope(band_signal=band_sig)

            # "Synapse" stage: low-pass the envelope (zero-phase) if requested
            if syn_sos is not None:
                env = sosfiltfilt(syn_sos, env)

            # Final safety: clamp tiny negative numerical wiggles
            env[env < 0] = 0.0
            env_tc[:, ch] = env

        # At this point env_tc is (T_audio, C) at fs_hz

        fs_env_hz = float(fs_hz)
        coch_tf = env_tc  # (T, C)

        # Optional early downsample for speed (common for cochleogram features)
        if self._cfg.env_target_fs_hz is not None:
            target = float(self._cfg.env_target_fs_hz)
            if target <= 0:
                raise ValueError("env_target_fs_hz must be positive.")
            if target >= fs_env_hz:
                # If target is higher/equal, do nothing (explicitly).
                pass
            else:
                coch_tf = _resample_2d_time_major(x_tf=coch_tf, input_sfreq=fs_env_hz, output_sfreq=target)
                fs_env_hz = target

        return coch_tf.astype(float, copy=False), centers_hz.astype(float), fs_env_hz, x_octaves.astype(float)


# =============================================================================
#                        dcap CONFIG + FEATURE COMPUTER
# =============================================================================

@dataclass(frozen=True, slots=True)
class CochleogramConfig(FeatureConfig):
    """
    dcap-level cochleogram configuration.

    Parameters
    ----------
    mode
        "accurate" or "efficient"
    f_min_hz, f_max_hz, octave_spacing
        Channel definition in frequency space.
    gammatone_order, gammatone_tau_factor
        Used only in "accurate"
    bandpass_order
        Used only in "efficient"
    synapse_lowpass_cutoff_hz, synapse_lowpass_order
        Envelope LPF stage (set cutoff_hz=None to disable).
    env_target_fs_hz
        Optional internal envelope sampling rate to speed up computation.
    derivative
        Postprocess derivative on final FeatureTimeBase grid: "none" | "diff" | "absdiff"
    downmix
        Multi-channel downmix strategy, currently only "mean".
    """
    mode: Literal["accurate", "efficient"] = "efficient"

    f_min_hz: float = 100.0
    f_max_hz: float = 10_000.0
    octave_spacing: float = 0.05

    gammatone_order: int = 4
    gammatone_tau_factor: float = 4.0

    bandpass_order: int = 4

    synapse_lowpass_cutoff_hz: Optional[float] = 30.0
    synapse_lowpass_order: int = 2

    env_target_fs_hz: Optional[float] = None

    derivative: str = "none"
    downmix: str = _DEFAULT_DOWNSMIX


class CochleogramComputer(FeatureComputer[CochleogramConfig]):
    """
    dcap FeatureComputer wrapper that:
    - runs the cochleogram engine
    - resamples to FeatureTimeBase grid
    - enforces exact n_times
    - applies optional derivative
    - returns FeatureResult with channels-first values
    """

    @property
    def name(self) -> str:
        return "acoustic.cochleogram"

    @property
    def kind(self) -> FeatureKind:
        return "acoustic"

    def compute(
        self,
        *,
        time: FeatureTimeBase,
        audio: Optional[np.ndarray] = None,
        audio_sfreq: Optional[float] = None,
        events_df: Optional[Any] = None,
        config: CochleogramConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("CochleogramComputer requires audio and audio_sfreq.")

        # Convert to mono; cochleogram is defined on a single waveform
        mono = _as_mono_audio(audio=np.asarray(audio, dtype=float), downmix=config.downmix)

        # Build engine config (explicit 1:1 mapping)
        engine_cfg = CochleogramEngineConfig(
            mode=config.mode,
            f_min_hz=float(config.f_min_hz),
            f_max_hz=float(config.f_max_hz),
            octave_spacing=float(config.octave_spacing),
            gammatone_order=int(config.gammatone_order),
            gammatone_tau_factor=float(config.gammatone_tau_factor),
            bandpass_order=int(config.bandpass_order),
            synapse_lowpass_cutoff_hz=None if config.synapse_lowpass_cutoff_hz is None else float(config.synapse_lowpass_cutoff_hz),
            synapse_lowpass_order=int(config.synapse_lowpass_order),
            env_target_fs_hz=None if config.env_target_fs_hz is None else float(config.env_target_fs_hz),
        )

        engine = _CochleogramEngine(engine_cfg)

        coch_tf, centers_hz, fs_env_hz, x_octaves = engine.apply(audio_mono=mono, fs_hz=float(audio_sfreq))
        # coch_tf: (T_env, C) at fs_env_hz

        # Align to requested TRF grid (FeatureTimeBase)
        coch_tf = _resample_2d_time_major(x_tf=coch_tf, input_sfreq=float(fs_env_hz), output_sfreq=float(time.sfreq))
        coch_tf = _force_length_time_major(x_tf=coch_tf, n_times=int(time.n_times))

        # Optional derivative postprocess per channel (computed on final grid)
        if config.derivative != "none":
            out_tf = np.empty_like(coch_tf, dtype=float)
            for ch in range(coch_tf.shape[1]):
                out_tf[:, ch] = apply_derivative(
                    x=np.asarray(coch_tf[:, ch], dtype=float),
                    sfreq=float(time.sfreq),
                    mode=config.derivative,
                )
            coch_tf = out_tf

        # Convert to channels-first for TRF: (C, T)
        values_ct = np.asarray(coch_tf, dtype=float).T

        channel_names = [f"cf_{cf:.3f}hz" for cf in centers_hz.tolist()]

        meta: dict[str, Any] = {
            "mode": config.mode,
            "audio_sfreq": float(audio_sfreq),
            "fs_env_hz": float(fs_env_hz),
            "f_min_hz": float(config.f_min_hz),
            "f_max_hz": float(config.f_max_hz),
            "octave_spacing": float(config.octave_spacing),
            "gammatone_order": int(config.gammatone_order),
            "gammatone_tau_factor": float(config.gammatone_tau_factor),
            "bandpass_order": int(config.bandpass_order),
            "synapse_lowpass_cutoff_hz": config.synapse_lowpass_cutoff_hz,
            "synapse_lowpass_order": int(config.synapse_lowpass_order),
            "env_target_fs_hz": config.env_target_fs_hz,
            "center_frequencies_hz": centers_hz.tolist(),
            "x_octaves": x_octaves.tolist(),
            "derivative": config.derivative,
            "downmix": config.downmix,
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=values_ct,
            time=time,
            channel_names=channel_names,
            meta=meta,
        )
