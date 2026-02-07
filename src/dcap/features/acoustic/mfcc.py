# =============================================================================
# =============================================================================
#                 ############################################
#                 #               BLOCK: MFCC                #
#                 ############################################
# =============================================================================
# =============================================================================
"""
MFCC feature aligned to FeatureTimeBase.

Pipeline
--------
1) Downmix to mono
2) STFT power spectrogram (accurate or efficient backend)
3) Mel filterbank -> mel power
4) log(mel power)
5) DCT-II along mel axis -> MFCCs
6) Optional liftering
7) Optional deltas: delta / delta-delta (computed on final TRF grid)

Output
------
FeatureResult.values: (n_mfcc * (1 + include_delta + include_delta2), n_times)
Channel naming:
  mfcc_00 ... mfcc_{n_mfcc-1}
  dmfcc_00 ... (if include_delta)
  ddmfcc_00 ... (if include_delta2)

Usage example
-------------
    time = FeatureTimeBase(sfreq=100.0, n_times=30_000, t0_s=0.0)
    cfg = MfccConfig(mode="efficient", n_mels=80, n_mfcc=13, include_delta=True)
    out = MfccComputer().compute(time=time, audio=wav, audio_sfreq=16_000.0, config=cfg)
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.signal import get_window, resample_poly, stft

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
    f = np.asarray(f_hz, dtype=float)
    # We keep HTK formula; it matches most toolboxes closely.
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
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Triangular mel filterbank matrix.

    Returns
    -------
    fb : (n_mels, n_freq_bins)
    mel_centers_hz : (n_mels,)
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

    mel_min = float(_hz_to_mel(f_hz=np.array([fmin_hz], dtype=float), htk=htk)[0])
    mel_max = float(_hz_to_mel(f_hz=np.array([fmax_hz], dtype=float), htk=htk)[0])

    mels = np.linspace(mel_min, mel_max, num=n_mels + 2, dtype=float)
    hz = _mel_to_hz(m=mels, htk=htk)

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
        right = min(right, n_freq_bins - 1)

        if center > left:
            fb[i, left:center] = (np.arange(left, center, dtype=float) - float(left)) / (float(center - left) + _TINY)
        if right > center:
            fb[i, center:right] = (float(right) - np.arange(center, right, dtype=float)) / (float(right - center) + _TINY)

    if norm == "slaney":
        enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
        fb *= enorm[:, None]

    return fb.astype(float), mel_centers_hz


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
) -> tuple[NDArray[np.complexfloating], float]:
    """
    Return Z_tf (T, F) complex and frame sampling frequency (frames/sec).
    """
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

    fs_frames = _frame_sfreq(audio_sfreq=fs_hz, hop_length=hop)
    return Z.astype(np.complex128, copy=False), fs_frames


def _stft_time_major_efficient(
    *,
    x: NDArray[np.floating],
    fs_hz: float,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int],
    window: str,
    center: bool,
) -> tuple[NDArray[np.complexfloating], float]:
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

    _, t_s, Z_f_t = stft(
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
    return Z_tf, fs_frames


# =============================================================================
#                           DELTA COMPUTATION (HTK-ish)
# =============================================================================

def _delta(*, x: NDArray[np.floating], width: int) -> NDArray[np.floating]:
    """
    Compute deltas along time axis for a (T, D) matrix.

    HTK-style regression delta with window +/- width:
        delta[t] = sum_{n=1..W} n (x[t+n] - x[t-n]) / (2 * sum_{n=1..W} n^2)

    We pad edges by replicating border values.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2D (T, D).")

    T, D = x.shape
    denom = 2.0 * float(np.sum(np.arange(1, width + 1, dtype=float) ** 2))
    padded = np.pad(x, ((width, width), (0, 0)), mode="edge")

    out = np.zeros((T, D), dtype=float)
    for n in range(1, width + 1):
        out += float(n) * (padded[width + n : width + n + T, :] - padded[width - n : width - n + T, :])

    return (out / denom).astype(float)


def _lifter(*, mfcc: NDArray[np.floating], lifter: int) -> NDArray[np.floating]:
    """
    Apply sinusoidal liftering to MFCCs (along coefficient axis).

    lifter = 0 disables.
    """
    if lifter <= 0:
        return mfcc
    n_ceps = mfcc.shape[1]
    n = np.arange(n_ceps, dtype=float)
    lift = 1.0 + 0.5 * float(lifter) * np.sin(np.pi * n / float(lifter))
    return (mfcc * lift[None, :]).astype(float)


# =============================================================================
#                               CONFIG + COMPUTER
# =============================================================================

@dataclass(frozen=True, slots=True)
class MfccConfig(FeatureConfig):
    mode: Literal["accurate", "efficient"] = "efficient"

    n_fft: int = 512
    hop_length: int = 160
    win_length: Optional[int] = None
    window: str = "hann"
    center: bool = True

    n_mels: int = 80
    n_mfcc: int = 13
    fmin_hz: float = 0.0
    fmax_hz: Optional[float] = None  # default Nyquist

    htk: bool = True
    mel_norm: Literal["none", "slaney"] = "slaney"

    # MFCC computation choices
    dct_type: int = 2
    dct_norm: Optional[Literal["ortho"]] = "ortho"

    lifter: int = 0

    include_delta: bool = False
    include_delta2: bool = False
    delta_width: int = 2

    # Always log-mel for MFCC; you can pick log base if desired later.
    log_floor: float = 1e-12

    downmix: str = "mean"


class MfccComputer(FeatureComputer[MfccConfig]):
    @property
    def name(self) -> str:
        return "acoustic.mfcc"

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
        config: MfccConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("MfccComputer requires audio and audio_sfreq.")

        if config.n_mfcc <= 0 or config.n_mels <= 0:
            raise ValueError("n_mfcc and n_mels must be positive.")
        if config.n_mfcc > config.n_mels:
            raise ValueError("n_mfcc must be <= n_mels (MFCC uses DCT over mel bands).")

        x = _as_mono_audio(audio=np.asarray(audio, dtype=float), downmix=config.downmix)

        sr_hz = float(audio_sfreq)
        nyq = 0.5 * sr_hz
        fmax = nyq if config.fmax_hz is None else float(config.fmax_hz)

        # STFT
        if config.mode == "accurate":
            Z_tf, fs_frames = _stft_time_major_accurate(
                x=x,
                fs_hz=sr_hz,
                n_fft=int(config.n_fft),
                hop_length=int(config.hop_length),
                win_length=None if config.win_length is None else int(config.win_length),
                window=str(config.window),
                center=bool(config.center),
            )
        elif config.mode == "efficient":
            Z_tf, fs_frames = _stft_time_major_efficient(
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

        # Mel filterbank
        fb, mel_centers_hz = _mel_filterbank(
            sr_hz=sr_hz,
            n_fft=int(config.n_fft),
            n_mels=int(config.n_mels),
            fmin_hz=float(config.fmin_hz),
            fmax_hz=float(fmax),
            htk=bool(config.htk),
            norm=str(config.mel_norm),
        )

        # Mel power (T, M)
        mel_power = P_tf @ fb.T

        # Log-mel (T, M)
        log_mel = np.log(np.maximum(mel_power, float(config.log_floor)))

        # DCT-II over mel axis -> (T, n_mfcc)
        mfcc_full = dct(log_mel, type=int(config.dct_type), axis=1, norm=config.dct_norm)
        mfcc = mfcc_full[:, : int(config.n_mfcc)]

        # Optional liftering
        mfcc = _lifter(mfcc=mfcc, lifter=int(config.lifter))

        # Align MFCC frames to FeatureTimeBase grid
        mfcc = _resample_2d_time_major(x_tf=mfcc, input_sfreq=float(fs_frames), output_sfreq=float(time.sfreq))
        mfcc = _force_length_time_major(x_tf=mfcc, n_times=int(time.n_times))  # (T, n_mfcc)

        # Optional deltas computed on final grid (so they correspond to TRF sampling)
        blocks = [mfcc]
        names = [f"mfcc_{i:02d}" for i in range(mfcc.shape[1])]

        if config.include_delta:
            d1 = _delta(x=mfcc, width=int(config.delta_width))
            blocks.append(d1)
            names.extend([f"dmfcc_{i:02d}" for i in range(d1.shape[1])])

        if config.include_delta2:
            # delta2 is delta(delta)
            d1 = _delta(x=mfcc, width=int(config.delta_width))
            d2 = _delta(x=d1, width=int(config.delta_width))
            blocks.append(d2)
            names.extend([f"ddmfcc_{i:02d}" for i in range(d2.shape[1])])

        feat_td = np.concatenate(blocks, axis=1)  # (T, D_total)
        values = feat_td.T  # (D_total, T)

        meta: dict[str, Any] = {
            "mode": config.mode,
            "n_fft": int(config.n_fft),
            "hop_length": int(config.hop_length),
            "win_length": int(config.win_length) if config.win_length is not None else None,
            "window": str(config.window),
            "center": bool(config.center),
            "n_mels": int(config.n_mels),
            "n_mfcc": int(config.n_mfcc),
            "fmin_hz": float(config.fmin_hz),
            "fmax_hz": float(fmax),
            "htk": bool(config.htk),
            "mel_norm": str(config.mel_norm),
            "dct_type": int(config.dct_type),
            "dct_norm": config.dct_norm,
            "lifter": int(config.lifter),
            "include_delta": bool(config.include_delta),
            "include_delta2": bool(config.include_delta2),
            "delta_width": int(config.delta_width),
            "frame_sfreq": float(fs_frames),
            "mel_centers_hz": mel_centers_hz.tolist(),
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=np.asarray(values, dtype=float),
            time=time,
            channel_names=names,
            meta=meta,
        )
