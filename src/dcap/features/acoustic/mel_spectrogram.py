# =============================================================================
# =============================================================================
#                 ############################################
#                 #          BLOCK: MEL SPECTROGRAM          #
#                 ############################################
# =============================================================================
# =============================================================================
"""
Mel spectrogram feature aligned to FeatureTimeBase.

Pipeline
--------
1) Downmix to mono
2) STFT power spectrogram (accurate or efficient backend)
3) Apply mel filterbank to power spectrum -> mel power
4) Optional log10 transform
5) Resample to FeatureTimeBase.sfreq and crop/pad to FeatureTimeBase.n_times

Modes
-----
- mode="accurate": explicit framing + rFFT (reflect padding if center=True)
- mode="efficient": scipy.signal.stft backend

Output contract
---------------
- FeatureResult.values: shape (n_mels, n_times)
- channel_names: "mel_{i:03d}" (stable)
- meta contains mel parameters and (optionally) mel center frequencies

Usage example
-------------
    time = FeatureTimeBase(sfreq=100.0, n_times=30_000, t0_s=0.0)
    cfg = MelSpectrogramConfig(mode="efficient", n_fft=512, hop_length=160, n_mels=80, output="log_power")
    out = MelSpectrogramComputer().compute(time=time, audio=wav, audio_sfreq=16_000.0, config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, get_window, resample_poly, sosfiltfilt, stft


# dcap
from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase


_TINY: float = 1e-12


# =============================================================================
#                              BASIC HELPERS
# =============================================================================

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


# =============================================================================
#                         MEL SCALE + FILTERBANK
# =============================================================================

def _hz_to_mel(*, f_hz: NDArray[np.floating], htk: bool) -> NDArray[np.floating]:
    """
    Convert Hz to mel.

    htk=True:  mel = 2595 * log10(1 + f/700)
    htk=False: same formula (we keep it simple; Slaney differs slightly in some libs).
    """
    f = np.asarray(f_hz, dtype=float)
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(*, m: NDArray[np.floating], htk: bool) -> NDArray[np.floating]:
    m = np.asarray(m, dtype=float)
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sr_hz: float,
    n_fft: int,
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float,
    htk: bool,
    norm: Literal["none", "slaney"],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Build a triangular mel filterbank matrix.

    Returns
    -------
    fb : (n_mels, n_freq_bins)
        Each row is a mel filter over linear FFT bins.
    mel_centers_hz : (n_mels,)
        Center frequency of each mel band in Hz (peak of triangle).
    fft_freqs_hz : (n_freq_bins,)
        Frequencies for FFT bins (rfft).
    """
    if n_fft <= 0:
        raise ValueError("n_fft must be positive.")
    if n_mels <= 0:
        raise ValueError("n_mels must be positive.")
    if sr_hz <= 0:
        raise ValueError("sr_hz must be positive.")
    if fmin_hz < 0:
        raise ValueError("fmin_hz must be >= 0.")
    nyq = 0.5 * sr_hz
    if not (0.0 <= fmin_hz < fmax_hz <= nyq):
        raise ValueError(f"Require 0 <= fmin < fmax <= Nyquist ({nyq}).")

    n_freq_bins = n_fft // 2 + 1
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr_hz).astype(float)

    # Mel grid: n_mels + 2 points (for triangular edges)
    mel_min = float(_hz_to_mel(f_hz=np.array([fmin_hz], dtype=float), htk=htk)[0])
    mel_max = float(_hz_to_mel(f_hz=np.array([fmax_hz], dtype=float), htk=htk)[0])

    mels = np.linspace(mel_min, mel_max, num=n_mels + 2, dtype=float)
    hz = _mel_to_hz(m=mels, htk=htk)

    # Map Hz points to nearest FFT bin indices
    bin_idx = np.floor((n_fft + 1) * hz / sr_hz).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_freq_bins - 1)

    fb = np.zeros((n_mels, n_freq_bins), dtype=float)
    mel_centers_hz = hz[1:-1].astype(float)

    for i in range(n_mels):
        left = bin_idx[i]
        center = bin_idx[i + 1]
        right = bin_idx[i + 2]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        if right >= n_freq_bins:
            right = n_freq_bins - 1

        # Rising edge
        if center > left:
            fb[i, left:center] = (np.arange(left, center, dtype=float) - float(left)) / (float(center - left) + _TINY)

        # Falling edge
        if right > center:
            fb[i, center:right] = (float(right) - np.arange(center, right, dtype=float)) / (float(right - center) + _TINY)

    # Optional Slaney-style area normalization (roughly equal energy per band)
    if norm == "slaney":
        # Normalize by bandwidth in Hz (triangle area scaling)
        # Using mel edge Hz points: hz[i], hz[i+1], hz[i+2]
        enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
        fb *= enorm[:, None]

    return fb.astype(float), mel_centers_hz, fft_freqs


# =============================================================================
#                                STFT BACKENDS
# =============================================================================

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
    x = np.asarray(x, dtype=float).reshape(-1)

    if win_length is None:
        win_length = n_fft
    if not (0 < win_length <= n_fft):
        raise ValueError("win_length must be in (0, n_fft].")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive.")

    if center:
        pad = n_fft // 2
        x = np.pad(x, (pad, pad), mode="reflect")

    w = get_window(window, win_length, fftbins=True).astype(float)  # noqa numpy
    if win_length < n_fft:
        w = np.pad(w, (0, n_fft - win_length), mode="constant")

    n = x.shape[0]
    if n < n_fft:
        x = np.pad(x, (0, n_fft - n), mode="constant")
        n = x.shape[0]

    hop = int(hop_length)
    n_frames = 1 + (n - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    frames_win = frames * w[None, :]
    Z = np.fft.rfft(frames_win, n=n_fft, axis=1)

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
    if win_length is None:
        win_length = n_fft
    if not (0 < win_length <= n_fft):
        raise ValueError("win_length must be in (0, n_fft].")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive.")

    noverlap = int(win_length - hop_length)
    if noverlap < 0:
        raise ValueError("hop_length must be <= win_length.")

    boundary = "zeros" if center else None

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
        padded=False,
    )
    Z_tf = np.asarray(Z_f_t, dtype=np.complex128).T  # (T, F)

    fs_frames = float(len(t_s) - 1) / float(t_s[-1] - t_s[0]) if len(t_s) > 1 else _frame_sfreq(audio_sfreq=fs_hz, hop_length=int(hop_length))
    return Z_tf, np.asarray(f_hz, dtype=float), fs_frames


# =============================================================================
#                               CONFIG + COMPUTER
# =============================================================================

@dataclass(frozen=True, slots=True)
class MelSpectrogramConfig(FeatureConfig):
    mode: Literal["accurate", "efficient"] = "efficient"

    n_fft: int = 512
    hop_length: int = 160
    win_length: Optional[int] = None
    window: str = "hann"
    center: bool = True

    n_mels: int = 80
    fmin_hz: float = 0.0
    fmax_hz: Optional[float] = None  # default: Nyquist
    htk: bool = True
    norm: Literal["none", "slaney"] = "slaney"

    output: Literal["power", "log_power"] = "log_power"
    log_floor: float = 1e-12

    downmix: str = "mean"


class MelSpectrogramComputer(FeatureComputer[MelSpectrogramConfig]):
    @property
    def name(self) -> str:
        return "acoustic.mel_spectrogram"

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
        config: MelSpectrogramConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("MelSpectrogramComputer requires audio and audio_sfreq.")

        x = _as_mono_audio(audio=np.asarray(audio, dtype=float), downmix=config.downmix)

        sr_hz = float(audio_sfreq)
        nyq = 0.5 * sr_hz
        fmax = nyq if config.fmax_hz is None else float(config.fmax_hz)

        # STFT
        if config.mode == "accurate":
            Z_tf, freqs_hz, fs_frames = _stft_time_major_accurate(
                x=x,
                fs_hz=sr_hz,
                n_fft=int(config.n_fft),
                hop_length=int(config.hop_length),
                win_length=None if config.win_length is None else int(config.win_length),
                window=str(config.window),
                center=bool(config.center),
            )
        elif config.mode == "efficient":
            Z_tf, freqs_hz, fs_frames = _stft_time_major_efficient(
                x=x,
                fs_hz=sr_hz,
                n_fft=int(config.n_fft),
                hop_length=int(config.hop_length),
                win_length=None if config.win_length is None else int(config.win_length),
                window=str(config.window),
                center=bool(config.center),
            )
        else:
            raise ValueError("mode must be 'accurate' or 'efficient'.")

        # Power spectrogram (T, F)
        P_tf = (np.abs(Z_tf) ** 2).astype(float)

        # Mel filterbank (n_mels, F)
        fb, mel_centers_hz, fft_freqs_hz = _mel_filterbank(
            sr_hz=sr_hz,
            n_fft=int(config.n_fft),
            n_mels=int(config.n_mels),
            fmin_hz=float(config.fmin_hz),
            fmax_hz=float(fmax),
            htk=bool(config.htk),
            norm=str(config.norm),
        )

        # Sanity: fft_freqs_hz should match freqs_hz from STFT.
        # In practice they should be identical for rFFT; but we do not hard-fail if not.
        if fft_freqs_hz.shape != freqs_hz.shape:
            raise RuntimeError("Internal inconsistency: FFT frequency grid shape mismatch.")

        # Apply mel filterbank: (T, n_mels) = (T, F) @ (F, n_mels)
        M_t_mel = P_tf @ fb.T

        if config.output == "power":
            M_t_mel_out = M_t_mel
        elif config.output == "log_power":
            M_t_mel_out = np.log10(np.maximum(M_t_mel, float(config.log_floor)))
        else:
            raise ValueError("output must be 'power' or 'log_power'.")

        # Resample to TRF grid
        M_t_mel_out = _resample_2d_time_major(
            x_tf=np.asarray(M_t_mel_out, dtype=float),
            input_sfreq=float(fs_frames),
            output_sfreq=float(time.sfreq),
        )
        M_t_mel_out = _force_length_time_major(x_tf=M_t_mel_out, n_times=int(time.n_times))

        # Return (n_mels, n_times)
        values = M_t_mel_out.T
        channel_names = [f"mel_{i:03d}" for i in range(values.shape[0])]

        meta: dict[str, Any] = {
            "mode": config.mode,
            "n_fft": int(config.n_fft),
            "hop_length": int(config.hop_length),
            "win_length": int(config.win_length) if config.win_length is not None else None,
            "window": str(config.window),
            "center": bool(config.center),
            "n_mels": int(config.n_mels),
            "fmin_hz": float(config.fmin_hz),
            "fmax_hz": float(fmax),
            "htk": bool(config.htk),
            "norm": str(config.norm),
            "output": str(config.output),
            "frame_sfreq": float(fs_frames),
            "mel_centers_hz": mel_centers_hz.tolist(),
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=np.asarray(values, dtype=float),
            time=time,
            channel_names=channel_names,
            meta=meta,
        )
