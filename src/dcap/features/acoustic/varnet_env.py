# =============================================================================
# =============================================================================
#                 ############################################
#                 #        BLOCK: VARNET-STYLE ENVELOPE      #
#                 ############################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.postprocess import apply_derivative
from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase


_DEFAULT_DOWNSMIX: str = "mean"


def _as_mono_audio(*, audio: np.ndarray, downmix: str) -> np.ndarray:
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
    from fractions import Fraction

    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def _resample_1d(*, x: np.ndarray, input_sfreq: float, output_sfreq: float) -> np.ndarray:
    if input_sfreq <= 0 or output_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    if np.isclose(input_sfreq, output_sfreq):
        return x
    ratio = float(output_sfreq) / float(input_sfreq)
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x, up=up, down=down)


def _force_length(*, x: np.ndarray, n_times: int) -> np.ndarray:
    if n_times <= 0:
        raise ValueError("n_times must be positive.")
    if x.shape[0] == n_times:
        return x
    if x.shape[0] > n_times:
        return x[:n_times]
    return np.pad(x, (0, n_times - x.shape[0]), mode="constant", constant_values=0.0)


def _erb_rate_hz_to_erb(f_hz: np.ndarray) -> np.ndarray:
    # Glasberg & Moore ERB-rate scale
    return 21.4 * np.log10(4.37e-3 * f_hz + 1.0)


def _erb_rate_erb_to_hz(e: np.ndarray) -> np.ndarray:
    return (10 ** (e / 21.4) - 1.0) / 4.37e-3


def _erb_bandwidth_hz(f_hz: np.ndarray) -> np.ndarray:
    # ERB bandwidth in Hz (Glasberg & Moore)
    return 24.7 * (4.37e-3 * f_hz + 1.0)


def _erb_spaced_bands(
    *,
    fmin_hz: float,
    fmax_hz: float,
    n_bands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 1-ERB-wide band edges around ERB-spaced center frequencies.

    Returns
    -------
    centers_hz, low_edges_hz, high_edges_hz
    """
    if n_bands <= 0:
        raise ValueError("n_bands must be positive.")
    if fmin_hz <= 0 or fmax_hz <= 0 or fmax_hz <= fmin_hz:
        raise ValueError("Invalid fmin_hz/fmax_hz.")

    e_min = float(_erb_rate_hz_to_erb(np.array([fmin_hz]))[0])
    e_max = float(_erb_rate_hz_to_erb(np.array([fmax_hz]))[0])
    e = np.linspace(e_min, e_max, n_bands, dtype=float)

    centers = _erb_rate_erb_to_hz(e)
    bw = _erb_bandwidth_hz(centers)

    lows = np.maximum(1.0, centers - 0.5 * bw)
    highs = centers + 0.5 * bw

    return centers, lows, highs


def _lowpass_envelope(
    *,
    x: np.ndarray,
    sfreq: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive.")
    if cutoff_hz >= 0.5 * sfreq:
        raise ValueError(f"cutoff_hz must be < Nyquist={0.5 * sfreq} (sfreq={sfreq}).")
    sos = butter(order, cutoff_hz / (0.5 * sfreq), btype="low", output="sos")
    return sosfiltfilt(sos, x)


@dataclass(frozen=True)
class VarnetEnvelopeConfig(FeatureConfig):
    """Varnet-style filterbank envelope.

    This implements a widely used “auditory-like” envelope:
      - ERB-spaced bandpass filterbank (Butterworth approximation; swappable later)
      - Hilbert magnitude per band
      - Optional lowpass on each band envelope
      - Combine across bands (mean/sum/rms)
      - Resample to FeatureTimeBase grid
      - Optional derivative postprocess

    Parameters
    ----------
    derivative
        "none" | "diff" | "absdiff"
    n_bands
        Number of ERB-spaced bands. Common default: 32.
    fmin_hz, fmax_hz
        Frequency range of the filterbank in Hz.
    envelope_lowpass_hz
        If provided, lowpass cutoff (Hz) applied to each band envelope.
        A common “slow envelope” choice is 8 Hz.
    envelope_sfreq
        Internal sampling rate for the envelope before final alignment to FeatureTimeBase.
        If None, defaults to FeatureTimeBase.sfreq.
    combine
        How to combine band envelopes into one regressor:
        - "mean": average across bands
        - "sum": sum across bands
        - "rms": sqrt(mean(env^2))
    bandpass_order
        Butterworth order for bandpass filters (approximate gammatone). Default 4.
    lowpass_order
        Butterworth order for envelope lowpass. Default 4.
    downmix
        How to downmix multi-channel audio to mono.

    Usage example
    -------------
        time = FeatureTimeBase(sfreq=100.0, n_times=24_000, t0_s=0.0)
        cfg = VarnetEnvelopeConfig(
            n_bands=32,
            fmin_hz=80.0,
            fmax_hz=8020.0,
            envelope_lowpass_hz=8.0,
            combine="mean",
        )

        comp = VarnetEnvelopeComputer()
        out = comp.compute(time=time, audio=wav, audio_sfreq=48_000.0, config=cfg)
    """

    n_bands: int = 32
    fmin_hz: float = 80.0
    fmax_hz: float = 8020.0
    envelope_lowpass_hz: Optional[float] = 8.0
    envelope_sfreq: Optional[float] = None
    combine: Literal["mean", "sum", "rms"] = "mean"
    bandpass_order: int = 4
    lowpass_order: int = 4
    downmix: str = _DEFAULT_DOWNSMIX


class VarnetEnvelopeComputer(FeatureComputer[VarnetEnvelopeConfig]):
    @property
    def name(self) -> str:
        return "acoustic.varnet_envelope"

    @property
    def kind(self) -> FeatureKind:
        return "acoustic"

    def compute(  # noqa
        self,
        *,
        time: FeatureTimeBase,
        audio: Optional[np.ndarray] = None,
        audio_sfreq: Optional[float] = None,
        events_df: Optional[Any] = None,
        config: VarnetEnvelopeConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("VarnetEnvelopeComputer requires audio and audio_sfreq.")

        x = _as_mono_audio(audio=audio, downmix=config.downmix)
        fs = float(audio_sfreq)

        env_sfreq = float(time.sfreq) if config.envelope_sfreq is None else float(config.envelope_sfreq)

        centers_hz, lows_hz, highs_hz = _erb_spaced_bands(
            fmin_hz=float(config.fmin_hz),
            fmax_hz=float(config.fmax_hz),
            n_bands=int(config.n_bands),
        )

        # Clip upper edges to below Nyquist to avoid invalid filter design.
        nyq = 0.5 * fs
        highs_hz = np.minimum(highs_hz, 0.999 * nyq)
        lows_hz = np.minimum(lows_hz, 0.998 * nyq)

        # Compute per-band envelopes at audio sampling rate.
        band_envs = np.empty((int(config.n_bands), x.shape[0]), dtype=float)

        for band_idx in range(int(config.n_bands)):
            lo = float(lows_hz[band_idx])
            hi = float(highs_hz[band_idx])

            # If band collapses (e.g. too close to Nyquist), fall back to zeros.
            if not (0.0 < lo < hi < nyq):
                band_envs[band_idx] = 0.0
                continue

            # Bandpass filter (Butterworth SOS), zero-phase to avoid delays.
            sos = butter(
                int(config.bandpass_order),
                [lo / nyq, hi / nyq],
                btype="bandpass",
                output="sos",
            )
            band_sig = sosfiltfilt(sos, x)

            # Hilbert magnitude envelope for that band.
            env = np.abs(hilbert(band_sig))

            # Optional lowpass of the envelope (still at audio rate).
            if config.envelope_lowpass_hz is not None:
                env = _lowpass_envelope(
                    x=env,
                    sfreq=fs,
                    cutoff_hz=float(config.envelope_lowpass_hz),
                    order=int(config.lowpass_order),
                )

            band_envs[band_idx] = env

        # Combine across bands to a single envelope regressor.
        if config.combine == "mean":
            env_audio = np.mean(band_envs, axis=0)
        elif config.combine == "sum":
            env_audio = np.sum(band_envs, axis=0)
        elif config.combine == "rms":
            env_audio = np.sqrt(np.mean(band_envs**2, axis=0))
        else:
            raise ValueError("combine must be 'mean', 'sum', or 'rms'.")

        # Resample envelope to internal envelope sampling rate (often equal to TRF grid).
        env = _resample_1d(x=env_audio, input_sfreq=fs, output_sfreq=env_sfreq)

        # Align to FeatureTimeBase grid if needed (env_sfreq may differ from time.sfreq).
        if not np.isclose(env_sfreq, float(time.sfreq)):
            env = _resample_1d(x=env, input_sfreq=env_sfreq, output_sfreq=float(time.sfreq))

        env = _force_length(x=env, n_times=int(time.n_times))

        # Derivative postprocess on final grid.
        env = apply_derivative(x=env, sfreq=float(time.sfreq), mode=config.derivative)

        meta: dict[str, Any] = {
            "audio_sfreq": float(audio_sfreq),
            "target_sfreq": float(time.sfreq),
            "envelope_sfreq": float(env_sfreq),
            "n_bands": int(config.n_bands),
            "fmin_hz": float(config.fmin_hz),
            "fmax_hz": float(config.fmax_hz),
            "envelope_lowpass_hz": None if config.envelope_lowpass_hz is None else float(config.envelope_lowpass_hz),
            "combine": str(config.combine),
            "bandpass_order": int(config.bandpass_order),
            "lowpass_order": int(config.lowpass_order),
            "downmix": str(config.downmix),
            "centers_hz": centers_hz.tolist(),
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=env.astype(float, copy=False),
            time=time,
            channel_names=["env"],
            meta=meta,
        )
