
# =============================================================================
# =============================================================================
#                 ############################################
#                 #     BLOCK: HILBERT SPEECH ENVELOPE       #
#                 ############################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.signal import hilbert, resample_poly

import mne

from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.postprocess import apply_derivative
from dcap.features.types import FeatureResult, FeatureTimeBase, FeatureKind

_DEFAULT_FIR_DESIGN: str = "firwin"
_DEFAULT_PAD: str = "reflect_limited"
_DEFAULT_DOWNSMIX: str = "mean"


def _as_mono_audio(*, audio: np.ndarray, downmix: str) -> np.ndarray:
    """Convert audio to mono (n_times,)."""
    x = np.asarray(audio, dtype=float)

    if x.ndim == 1:
        return x

    if x.ndim != 2:
        raise ValueError(f"audio must be 1D or 2D (channels x time). Got shape={x.shape}.")

    # convention in dcap: time-last for multi-channel arrays
    n_channels, n_times = x.shape
    if n_channels == 1:
        return x[0]

    if downmix == "mean":
        return np.mean(x, axis=0)

    raise ValueError(f"Unknown downmix='{downmix}'. Supported: 'mean'.")


def _resample_to_timebase(
    *,
    x: np.ndarray,
    input_sfreq: float,
    output_sfreq: float,
) -> np.ndarray:
    """Resample 1D signal to a new sampling rate using polyphase resampling."""
    if input_sfreq <= 0 or output_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")

    if np.isclose(input_sfreq, output_sfreq):
        return x

    # Compute integer up/down factors with good numerical stability
    # up/down ≈ output/input
    # We keep it simple: use rational approximation from floats.
    ratio: float = float(output_sfreq) / float(input_sfreq)
    # Bound denominator to avoid giant filters for weird rates
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x, up=up, down=down)


def _rational_approximation(*, ratio: float, max_denominator: int) -> tuple[int, int]:
    """Approximate ratio with integers up/down."""
    # Use Python's fractions if available (standard lib), but keep deps minimal.
    from fractions import Fraction

    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def _force_length(*, x: np.ndarray, n_times: int) -> np.ndarray:
    """Crop or pad a 1D signal to exactly n_times."""
    if n_times <= 0:
        raise ValueError("n_times must be positive.")

    if x.shape[0] == n_times:
        return x

    if x.shape[0] > n_times:
        return x[:n_times]

    pad_width = n_times - x.shape[0]
    return np.pad(x, pad_width=(0, pad_width), mode="constant", constant_values=0.0)


def _lowpass_if_requested(
    *,
    x: np.ndarray,
    sfreq: float,
    lowpass_hz: Optional[float],
    fir_design: str,
    pad: str,
) -> np.ndarray:
    """Optional low-pass filtering using MNE's filter_data."""
    if lowpass_hz is None:
        return x

    if lowpass_hz <= 0:
        raise ValueError("lowpass_hz must be positive when provided.")
    if lowpass_hz >= 0.5 * sfreq:
        raise ValueError(
            f"lowpass_hz={lowpass_hz} must be < Nyquist={0.5 * sfreq} for sfreq={sfreq}."
        )

    # MNE expects shape (n_channels, n_times) or (n_times,)
    # We'll keep it 1D and let MNE handle it.
    return mne.filter.filter_data(
        data=x,
        sfreq=float(sfreq),
        l_freq=None,
        h_freq=float(lowpass_hz),
        fir_design=fir_design,
        pad=pad,
        verbose=False,
    )


@dataclass(frozen=True)
class HilbertEnvelopeConfig(FeatureConfig):
    """Configuration for Hilbert-based speech envelope.

    Parameters
    ----------
    derivative
        Post-processing derivative mode: "none" | "diff" | "absdiff".
    lowpass_hz
        Optional low-pass cutoff (Hz) applied **to the envelope** after Hilbert magnitude.
        If None, no low-pass filtering is applied.
    downmix
        If audio has multiple channels (channels x time), how to convert to mono.
        Currently supported: "mean".
    fir_design
        FIR design passed to MNE filtering when lowpass_hz is not None.
    pad
        Padding mode passed to MNE filtering when lowpass_hz is not None.

    Usage example
    -------------
        time = FeatureTimeBase(sfreq=100.0, n_times=24_000, t0_s=0.0)

        cfg = HilbertEnvelopeConfig(
            lowpass_hz=8.0,
            derivative="absdiff",
        )

        comp = HilbertEnvelopeComputer()
        feat = comp.compute(
            time=time,
            audio=wav,
            audio_sfreq=48_000.0,
            config=cfg,
        )
    """

    lowpass_hz: Optional[float] = None
    downmix: str = _DEFAULT_DOWNSMIX
    fir_design: str = _DEFAULT_FIR_DESIGN
    pad: str = _DEFAULT_PAD


class HilbertEnvelopeComputer(FeatureComputer):
    """Hilbert envelope feature computer.

    Notes
    -----
    This feature:
    1) downmixes audio to mono (if needed),
    2) resamples to the requested FeatureTimeBase.sfreq,
    3) computes envelope = |hilbert(audio)|,
    4) optionally low-pass filters the envelope,
    5) crops/pads to exactly FeatureTimeBase.n_times,
    6) applies derivative postprocess if requested.

    Usage example
    -------------
        time = FeatureTimeBase(sfreq=100.0, n_times=24_000, t0_s=0.0)
        cfg = HilbertEnvelopeConfig(lowpass_hz=8.0, derivative="none")

        comp = HilbertEnvelopeComputer()
        out = comp.compute(time=time, audio=wav, audio_sfreq=48_000.0, config=cfg)
    """

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
            config: HilbertEnvelopeConfig,
            context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("HilbertEnvelopeComputer requires audio and audio_sfreq.")

        mono = _as_mono_audio(audio=audio, downmix=config.downmix)

        # 1) Hilbert envelope at ORIGINAL audio sampling rate
        env_audio = np.abs(hilbert(mono))

        # 2) Optional low-pass filtering of the envelope at audio_sfreq
        env_audio = _lowpass_if_requested(
            x=env_audio,
            sfreq=float(audio_sfreq),
            lowpass_hz=config.lowpass_hz,
            fir_design=config.fir_design,
            pad=config.pad,
        )

        # 3) Resample envelope to target timebase sampling rate
        env = _resample_to_timebase(
            x=env_audio,
            input_sfreq=float(audio_sfreq),
            output_sfreq=float(time.sfreq),
        )

        # 4) Enforce exact length on target grid
        env = _force_length(x=env, n_times=int(time.n_times))

        # 5) Optional derivative computed on the final grid
        env = apply_derivative(x=env, sfreq=float(time.sfreq), mode=config.derivative)

        meta: dict[str, Any] = {
            "audio_sfreq": float(audio_sfreq),
            "target_sfreq": float(time.sfreq),
            "lowpass_hz": None if config.lowpass_hz is None else float(config.lowpass_hz),
            "downmix": config.downmix,
            "fir_design": config.fir_design,
            "pad": config.pad,
            "derivative": config.derivative,
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=env.astype(float, copy=False),
            time=time,
            channel_names=["env"],
            meta=meta,
        )

