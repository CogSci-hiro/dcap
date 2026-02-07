
# =============================================================================
# =============================================================================
#                 ############################################
#                 #            BLOCK: SPECTROGRAM            #
#                 ############################################
# =============================================================================
# =============================================================================
"""
Spectrogram feature (STFT magnitude / power / log-power) aligned to FeatureTimeBase.

We provide two modes:
- mode="accurate": librosa.stft-style centered framing with reflection padding,
  Hann window, and frame-to-time mapping via center-of-window timestamps.
- mode="efficient": scipy.signal.stft, typically faster and simpler.

Both return a time-frequency representation aligned to a requested FeatureTimeBase.

Output contract
---------------
- FeatureResult.values shape: (n_bins, n_times)  # bins-first for TRF
- FeatureResult.time: exactly the provided FeatureTimeBase
- channel_names: "f_{hz:.1f}hz" for each FFT bin frequency

Notes
-----
- Absolute scaling depends on windowing conventions; for TRF we usually z-score downstream.
- We compute the spectrogram at an internal frame rate determined by hop_length,
  then resample to time.sfreq and crop/pad to time.n_times.

Usage example
-------------
    time = FeatureTimeBase(sfreq=100.0, n_times=30_000, t0_s=0.0)
    cfg = SpectrogramConfig(mode="efficient", n_fft=512, hop_length=160, output="log_power")
    out = SpectrogramComputer().compute(time=time, audio=wav, audio_sfreq=16_000.0, config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window, resample_poly, stft

from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase


# =============================================================================
#                             HELPERS
# =============================================================================

_TINY: float = 1e-12


def _as_mono_audio(*, audio: np.ndarray, downmix: str) -> np.ndarray:
    x = np.asarray(audio, dtype=float)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"audio must be 1D or 2D (channels x time). Got shape={x.shape}.")
    if x.shape[0] == 1:
        return x[0]
    if downmix == "mean":
        return np.mean(x, axis=0)
    raise ValueError(f"Unknown downmix='{downmix}'. Supported: 'mean'.")


def _rational_approximation(*, ratio: float, max_denominator: int) -> tuple[int, int]:
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def _resample_2d_time_major(*, x_tf: NDArray[np.floating], input_sfreq: float, output_sfreq: float) -> NDArray[np.floating]:
    """
    Resample a (T, F) array along time axis only.
    """
    if input_sfreq <= 0 or output_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    if np.isclose(input_sfreq, output_sfreq):
        return x_tf
    ratio = float(output_sfreq) / float(input_sfreq)
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x_tf, up=up, down=down, axis=0)


def _force_length_time_major(*, x_tf: NDArray[np.floating], n_times: int) -> NDArray[np.floating]:
    if x_tf.shape[0] == n_times:
        return x_tf
    if x_tf.shape[0] > n_times:
        return x_tf[:n_times, :]
    pad = np.zeros((n_times - x_tf.shape[0], x_tf.shape[1]), dtype=x_tf.dtype)
    return np.concatenate([x_tf, pad], axis=0)


def _frame_sfreq(*, audio_sfreq: float, hop_length: int) -> float:
    if hop_length <= 0:
        raise ValueError("hop_length must be positive.")
    return float(audio_sfreq) / float(hop_length)


def _stft_time_major_accurate(
    *,
    x: NDArray[np.floating],
    fs_hz: float,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int],
    window: str,
    center: bool,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating], float]:
    """
    "Accurate" STFT:
    - uses explicit framing and rFFT per frame
    - optional centering via reflection padding
    Returns:
      Z_tf: (T_frames, F_bins) complex
      freqs_hz: (F_bins,)
      frame_sfreq: frames per second
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if n_fft <= 0:
        raise ValueError("n_fft must be positive.")
    if win_length is None:
        win_length = n_fft
    if not (0 < win_length <= n_fft):
        raise ValueError("win_length must be in (0, n_fft].")

    hop = int(hop_length)

    if center:
        pad = n_fft // 2
        x = np.pad(x, (pad, pad), mode="reflect")

    w = get_window(window, win_length, fftbins=True).astype(float)  # noqa numpy
    if win_length < n_fft:
        # zero-pad window to n_fft
        w = np.pad(w, (0, n_fft - win_length), mode="constant")

    n = x.shape[0]
    if n < n_fft:
        # Pad so we have at least one frame
        x = np.pad(x, (0, n_fft - n), mode="constant")
        n = x.shape[0]

    # Number of frames (inclusive of last hop if it fits)
    n_frames = 1 + (n - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )

    frames_win = frames * w[None, :]
    Z = np.fft.rfft(frames_win, n=n_fft, axis=1)  # (T, F)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz).astype(float)
    fs_frames = _frame_sfreq(audio_sfreq=fs_hz, hop_length=hop)

    return Z.astype(np.complex128, copy=False), freqs, fs_frames


def _stft_time_major_efficient(
    *,
    x: NDArray[np.floating],
    fs_hz: float,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int],
    window: str,
    center: bool,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating], float]:
    """
    "Efficient" STFT using scipy.signal.stft.

    Returns Z_tf: (T_frames, F_bins) complex.
    """
    if win_length is None:
        win_length = n_fft
    if not (0 < win_length <= n_fft):
        raise ValueError("win_length must be in (0, n_fft].")

    noverlap = int(win_length - hop_length)
    if noverlap < 0:
        raise ValueError("hop_length must be <= win_length.")

    boundary = "zeros"
    padded = False
    # center=True corresponds roughly to boundary padding; not identical to reflect-padding,
    # but good enough for an "efficient" mode.
    if not center:
        boundary = None

    f_hz, t_s, Z_f_t = stft(
        x=np.asarray(x, dtype=float),
        fs=float(fs_hz),
        window=window,
        nperseg=int(win_length),
        noverlap=int(noverlap),
        nfft=int(n_fft),
        detrend=False,
        return_onesided=True,
        boundary=boundary,
        padded=padded,
    )
    # scipy returns (F, T). Convert to (T, F)
    Z_tf = np.asarray(Z_f_t, dtype=np.complex128).T
    fs_frames = float(len(t_s) - 1) / float(t_s[-1] - t_s[0]) if len(t_s) > 1 else _frame_sfreq(audio_sfreq=fs_hz, hop_length=int(hop_length))

    return Z_tf, np.asarray(f_hz, dtype=float), fs_frames


# =============================================================================
#                               CONFIG + COMPUTER
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpectrogramConfig(FeatureConfig):
    mode: Literal["accurate", "efficient"] = "efficient"

    n_fft: int = 512
    hop_length: int = 160
    win_length: Optional[int] = None
    window: str = "hann"
    center: bool = True

    output: Literal["magnitude", "power", "log_power"] = "log_power"
    log_floor: float = 1e-12

    fmin_hz: Optional[float] = None
    fmax_hz: Optional[float] = None

    downmix: str = "mean"


class SpectrogramComputer(FeatureComputer[SpectrogramConfig]):
    @property
    def name(self) -> str:
        return "acoustic.spectrogram"

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
        config: SpectrogramConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("SpectrogramComputer requires audio and audio_sfreq.")

        x = _as_mono_audio(audio=np.asarray(audio, dtype=float), downmix=config.downmix)

        # STFT
        if config.mode == "accurate":
            Z_tf, freqs_hz, fs_frames = _stft_time_major_accurate(
                x=x,
                fs_hz=float(audio_sfreq),
                n_fft=int(config.n_fft),
                hop_length=int(config.hop_length),
                win_length=None if config.win_length is None else int(config.win_length),
                window=str(config.window),
                center=bool(config.center),
            )
        elif config.mode == "efficient":
            Z_tf, freqs_hz, fs_frames = _stft_time_major_efficient(
                x=x,
                fs_hz=float(audio_sfreq),
                n_fft=int(config.n_fft),
                hop_length=int(config.hop_length),
                win_length=None if config.win_length is None else int(config.win_length),
                window=str(config.window),
                center=bool(config.center),
            )
        else:
            raise ValueError("mode must be 'accurate' or 'efficient'.")

        # Convert to desired scale
        if config.output == "magnitude":
            S_tf = np.abs(Z_tf)
        elif config.output == "power":
            S_tf = (np.abs(Z_tf) ** 2)
        elif config.output == "log_power":
            power = (np.abs(Z_tf) ** 2)
            S_tf = np.log10(np.maximum(power, float(config.log_floor)))
        else:
            raise ValueError("output must be 'magnitude', 'power', or 'log_power'.")

        # Optional frequency cropping
        keep = np.ones(freqs_hz.shape[0], dtype=bool)
        if config.fmin_hz is not None:
            keep &= freqs_hz >= float(config.fmin_hz)
        if config.fmax_hz is not None:
            keep &= freqs_hz <= float(config.fmax_hz)

        freqs_hz = freqs_hz[keep]
        S_tf = S_tf[:, keep]

        # Resample STFT frames to FeatureTimeBase grid
        S_tf = _resample_2d_time_major(x_tf=np.asarray(S_tf, dtype=float), input_sfreq=float(fs_frames), output_sfreq=float(time.sfreq))
        S_tf = _force_length_time_major(x_tf=S_tf, n_times=int(time.n_times))

        # Return bins-first: (F, T)
        values_ft = S_tf.T
        channel_names = [f"f_{f:.3f}hz" for f in freqs_hz.tolist()]

        meta: dict[str, Any] = {
            "mode": config.mode,
            "n_fft": int(config.n_fft),
            "hop_length": int(config.hop_length),
            "win_length": int(config.win_length) if config.win_length is not None else None,
            "window": str(config.window),
            "center": bool(config.center),
            "output": str(config.output),
            "frame_sfreq": float(fs_frames),
            "freqs_hz": freqs_hz.tolist(),
            "fmin_hz": config.fmin_hz,
            "fmax_hz": config.fmax_hz,
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=np.asarray(values_ft, dtype=float),
            time=time,
            channel_names=channel_names,
            meta=meta,
        )
