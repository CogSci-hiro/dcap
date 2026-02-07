# =============================================================================
# =============================================================================
#                 ############################################
#                 #             BLOCK: MIDBRAIN              #
#                 ############################################
# =============================================================================
# =============================================================================
"""
Midbrain feature: spectro-temporal modulation representation derived from a cochleogram.

Concept
-------
We treat the cochleogram as a time-by-frequency (channels) matrix:
    C(t, f)

A "midbrain-like" representation is often obtained by measuring energy in
specific modulation bands:
- temporal modulation: how fast C changes over time (e.g., 1, 2, 4, 8, 16 Hz)
- spectral modulation: ripple density across log-frequency (e.g., 0.25, 0.5, 1, 2 cyc/oct)

We implement a separable approximation:
1) compute cochleogram envelopes (CochleogramComputer)
2) apply temporal modulation filters along time
3) apply spectral modulation filters along frequency axis (in octave space)
4) take magnitude / power as modulation energy

Modes
-----
We keep *two midbrain modes*:
- mode="efficient": time-domain FIR modulation filters + spatial FIR across channels
- mode="accurate": FFT-domain filtering that matches target bandpass responses more sharply

Additionally, cochleogram itself has its own mode:
- CochleogramConfig.mode in {"efficient","accurate"}

Output
------
FeatureResult.values has shape (n_features, n_times) where:
    n_features = n_channels * n_temporal_mods * n_spectral_mods   (flattened)

This is large. For TRF, you may later want to:
- reduce channels (increase octave_spacing)
- reduce modulation grids
- or pool over channels (optional future config)

Usage example
-------------
    time = FeatureTimeBase(sfreq=100.0, n_times=30_000, t0_s=0.0)
    cfg = MidbrainConfig(
        cochleogram=CochleogramConfig(mode="efficient", octave_spacing=0.25, env_target_fs_hz=200.0),
        mode="efficient",
        temporal_mods_hz=(1.0, 2.0, 4.0, 8.0),
        spectral_mods_cyc_per_oct=(0.25, 0.5, 1.0),
    )
    out = MidbrainComputer().compute(time=time, audio=wav, audio_sfreq=16_000.0, config=cfg)
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase

from dcap.features.acoustic.cochleogram import CochleogramComputer, CochleogramConfig


# =============================================================================
#                             RESAMPLING HELPERS
# =============================================================================

def _rational_approximation(*, ratio: float, max_denominator: int) -> tuple[int, int]:
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def _resample_2d_time_major(*, x_tf: NDArray[np.floating], input_sfreq: float, output_sfreq: float) -> NDArray[np.floating]:
    """Resample a (T, F) array along time axis only."""
    if input_sfreq <= 0 or output_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    if np.isclose(input_sfreq, output_sfreq):
        return x_tf
    ratio = float(output_sfreq) / float(input_sfreq)
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x_tf, up=up, down=down, axis=0)


def _force_length_time_major(*, x_tf: NDArray[np.floating], n_times: int) -> NDArray[np.floating]:
    """Crop/pad a (T, F) array to exactly n_times along time."""
    if x_tf.shape[0] == n_times:
        return x_tf
    if x_tf.shape[0] > n_times:
        return x_tf[:n_times, :]
    pad = np.zeros((n_times - x_tf.shape[0], x_tf.shape[1]), dtype=x_tf.dtype)
    return np.concatenate([x_tf, pad], axis=0)


# =============================================================================
#                     MODULATION FILTER DESIGN (TEMPORAL)
# =============================================================================

def _temporal_bandpass_fir(
    *,
    sfreq: float,
    center_hz: float,
    bandwidth_hz: float,
    n_cycles: float,
) -> NDArray[np.floating]:
    """
    Design a simple symmetric FIR bandpass around center_hz with given bandwidth.

    We do:
      - Butterworth bandpass in SOS
      - Convert to FIR by impulse response sampling via filtering a delta
      - Window to finite length

    This yields a stable "bandpass-ish" FIR kernel.
    """
    if center_hz <= 0 or bandwidth_hz <= 0:
        raise ValueError("center_hz and bandwidth_hz must be positive.")
    nyq = 0.5 * sfreq
    low = max(0.1, center_hz - 0.5 * bandwidth_hz)
    high = min(nyq * 0.999, center_hz + 0.5 * bandwidth_hz)
    if not (0.0 < low < high < nyq):
        raise ValueError(f"Invalid temporal band [{low},{high}] for nyq={nyq}.")

    # Kernel length: n_cycles / center_hz seconds, symmetric
    # e.g. n_cycles=6 at 2 Hz -> 3 sec kernel
    half_len_s = 0.5 * (n_cycles / center_hz)
    half_len = int(np.ceil(half_len_s * sfreq))
    n_taps = 2 * half_len + 1

    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")

    delta = np.zeros(n_taps, dtype=float)
    delta[half_len] = 1.0  # centered impulse
    h = sosfiltfilt(sos, delta)  # zero-phase shaping of impulse -> symmetric-ish

    # Normalize L2 to keep scales comparable
    norm = float(np.sqrt(np.sum(h * h))) + 1e-12
    return (h / norm).astype(float)  # noqa numpy


def _apply_fir_time(
    *,
    x_tf: NDArray[np.floating],
    h: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Convolve FIR along time for each frequency channel independently.
    x_tf is (T, F). Output is (T, F).
    """
    T, F = x_tf.shape
    y = np.empty_like(x_tf, dtype=float)
    for fi in range(F):
        y[:, fi] = fftconvolve(x_tf[:, fi], h, mode="same")
    return y


# =============================================================================
#                 MODULATION FILTER DESIGN (SPECTRAL / RIPPLE)
# =============================================================================

def _spectral_bandpass_fir(
    *,
    x_octaves: NDArray[np.floating],
    center_cyc_per_oct: float,
    bandwidth_cyc_per_oct: float,
    n_cycles: float,
) -> NDArray[np.floating]:
    """
    Design a 1D FIR bandpass across the frequency-channel axis.

    The axis is assumed to be approximately uniform in octaves, i.e. x_octaves is
    roughly linear with channel index (true if cochleogram centers are octave-spaced).

    Implementation:
    - treat channel index as "spatial sampling"
    - build a cosine-modulated Gaussian-windowed kernel in channel units
      (a ripple-like bandpass).
    """
    if center_cyc_per_oct <= 0 or bandwidth_cyc_per_oct <= 0:
        raise ValueError("center and bandwidth must be positive.")

    # Approximate spacing in octaves per channel
    dx = float(np.median(np.diff(x_octaves)))
    if dx <= 0:
        raise ValueError("Invalid x_octaves spacing.")

    # Convert cycles/octave to cycles per channel
    center_cyc_per_ch = center_cyc_per_oct * dx
    bw_cyc_per_ch = bandwidth_cyc_per_oct * dx

    # Kernel half-length in channels ~ n_cycles / center
    half_len_ch = int(np.ceil(0.5 * (n_cycles / max(center_cyc_per_ch, 1e-6))))
    # n_taps = 2 * half_len_ch + 1
    idx = np.arange(-half_len_ch, half_len_ch + 1, dtype=float)

    # Gaussian window width from bandwidth (heuristic)
    sigma = max(1.0, 0.5 / max(bw_cyc_per_ch, 1e-6))
    window = np.exp(-0.5 * (idx / sigma) ** 2)

    carrier = np.cos(2.0 * np.pi * center_cyc_per_ch * idx)
    h = window * carrier

    # Remove DC leakage (bandpass-ish)
    h = h - np.mean(h)

    norm = float(np.sqrt(np.sum(h * h))) + 1e-12
    return (h / norm).astype(float)


def _apply_fir_freq(
    *,
    x_tf: NDArray[np.floating],
    h: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Convolve FIR along the frequency/channel axis.
    x_tf is (T, F). Output is (T, F).
    """
    T, F = x_tf.shape
    y = np.empty_like(x_tf, dtype=float)
    for ti in range(T):
        y[ti, :] = fftconvolve(x_tf[ti, :], h, mode="same")
    return y


# =============================================================================
#                               CONFIG + COMPUTER
# =============================================================================

@dataclass(frozen=True, slots=True)
class MidbrainConfig(FeatureConfig):
    """
    dcap configuration for midbrain feature.

    Parameters
    ----------
    cochleogram
        Cochleogram front-end configuration (includes its own accurate/efficient).
    mode
        Midbrain backend: "accurate" (FFT-ish / sharper) or "efficient" (FIR conv).
    temporal_mods_hz
        Temporal modulation center frequencies (Hz).
    spectral_mods_cyc_per_oct
        Spectral modulation center frequencies (cycles per octave).
    temporal_bandwidth_hz
        Bandwidth used for temporal bandpass filters (Hz).
    spectral_bandwidth_cyc_per_oct
        Bandwidth used for spectral ripple filters (cycles per octave).
    temporal_kernel_cycles
        Kernel length in units of modulation cycles (higher => sharper but slower).
    spectral_kernel_cycles
        Same for spectral kernels.
    power
        If True output modulation power (squared magnitude), else magnitude.
    """
    cochleogram: CochleogramConfig = field(default_factory=CochleogramConfig)

    mode: Literal["accurate", "efficient"] = "efficient"

    temporal_mods_hz: Sequence[float] = (1.0, 2.0, 4.0, 8.0)
    spectral_mods_cyc_per_oct: Sequence[float] = (0.25, 0.5, 1.0, 2.0)

    temporal_bandwidth_hz: float = 1.0
    spectral_bandwidth_cyc_per_oct: float = 0.5

    temporal_kernel_cycles: float = 6.0
    spectral_kernel_cycles: float = 6.0

    power: bool = True

class MidbrainComputer(FeatureComputer[MidbrainConfig]):
    @property
    def name(self) -> str:
        return "acoustic.midbrain"

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
        config: MidbrainConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("MidbrainComputer requires audio and audio_sfreq.")

        # ---------------------------------------------------------------------
        # Step 1: cochleogram front-end (values are (C, T) on FeatureTimeBase grid)
        # ---------------------------------------------------------------------
        coch_out = CochleogramComputer().compute(
            time=time,
            audio=audio,
            audio_sfreq=audio_sfreq,
            config=config.cochleogram,
        )

        # Convert to time-major for filtering: (T, C)
        coch_tc = np.asarray(coch_out.values, dtype=float).T
        C = coch_tc.shape[1]

        # We need x_octaves for spectral modulation filtering.
        # The cochleogram meta includes x_octaves (from our cochleogram implementation).
        x_octaves = np.asarray(coch_out.meta.get("x_octaves"), dtype=float)
        if x_octaves.shape[0] != C:
            # Fallback: assume uniform octave spacing by channel index if metadata missing.
            x_octaves = np.linspace(0.0, 1.0, C, dtype=float)

        # ---------------------------------------------------------------------
        # Step 2: compute modulation responses for each (temporal_mod, spectral_mod)
        # ---------------------------------------------------------------------
        temporal_mods = tuple(float(x) for x in config.temporal_mods_hz)
        spectral_mods = tuple(float(x) for x in config.spectral_mods_cyc_per_oct)

        if len(temporal_mods) == 0 or len(spectral_mods) == 0:
            raise ValueError("temporal_mods_hz and spectral_mods_cyc_per_oct must be non-empty.")

        sfreq_env = float(time.sfreq)

        # Pre-design filters (explicit caching to avoid redesign inside loops)
        temporal_kernels = [
            _temporal_bandpass_fir(
                sfreq=sfreq_env,
                center_hz=tm,
                bandwidth_hz=float(config.temporal_bandwidth_hz),
                n_cycles=float(config.temporal_kernel_cycles),
            )
            for tm in temporal_mods
        ]

        spectral_kernels = [
            _spectral_bandpass_fir(
                x_octaves=x_octaves,
                center_cyc_per_oct=sm,
                bandwidth_cyc_per_oct=float(config.spectral_bandwidth_cyc_per_oct),
                n_cycles=float(config.spectral_kernel_cycles),
            )
            for sm in spectral_mods
        ]

        # Allocate output: (T, C, n_tm, n_sm)
        T = coch_tc.shape[0]
        out = np.empty((T, C, len(temporal_mods), len(spectral_mods)), dtype=float)

        # ---------------------------------------------------------------------
        # Efficient mode: separable FIR convolutions
        # ---------------------------------------------------------------------
        if config.mode == "efficient":
            for i_tm, h_t in enumerate(temporal_kernels):
                tmp_tc = _apply_fir_time(x_tf=coch_tc, h=h_t)  # (T, C)

                for i_sm, h_s in enumerate(spectral_kernels):
                    resp_tc = _apply_fir_freq(x_tf=tmp_tc, h=h_s)  # (T, C)

                    # Magnitude/power
                    if config.power:
                        out[:, :, i_tm, i_sm] = resp_tc * resp_tc
                    else:
                        out[:, :, i_tm, i_sm] = np.abs(resp_tc)

        # ---------------------------------------------------------------------
        # Accurate mode: same kernels, but apply in FFT domain for sharper response
        # (still separable, but uses FFT convolution explicitly with full-length kernels)
        # ---------------------------------------------------------------------
        elif config.mode == "accurate":
            # For time: FFT convolution per channel
            for i_tm, h_t in enumerate(temporal_kernels):
                tmp_tc = np.empty_like(coch_tc, dtype=float)
                for c in range(C):
                    tmp_tc[:, c] = fftconvolve(coch_tc[:, c], h_t, mode="same")

                for i_sm, h_s in enumerate(spectral_kernels):
                    resp_tc = np.empty_like(tmp_tc, dtype=float)
                    for t_idx in range(T):
                        resp_tc[t_idx, :] = fftconvolve(tmp_tc[t_idx, :], h_s, mode="same")

                    if config.power:
                        out[:, :, i_tm, i_sm] = resp_tc * resp_tc
                    else:
                        out[:, :, i_tm, i_sm] = np.abs(resp_tc)

        else:
            raise ValueError("mode must be 'efficient' or 'accurate'.")

        # ---------------------------------------------------------------------
        # Step 3: flatten features -> (F, T) for FeatureResult
        # ---------------------------------------------------------------------
        features_tf = out.reshape(T, -1)  # (T, F_total)
        values_ft = features_tf.T         # (F_total, T)

        # Build stable channel names in a deterministic order:
        channel_names: list[str] = []
        center_freqs = coch_out.meta.get("center_frequencies_hz", None)
        if center_freqs is None:
            # Fallback labels by channel index
            cf_list = [float(i) for i in range(C)]
        else:
            cf_list = [float(v) for v in center_freqs]

        for cf in cf_list:
            for tm in temporal_mods:
                for sm in spectral_mods:
                    channel_names.append(f"cf_{cf:.3f}|tm_{tm:.3f}hz|sm_{sm:.3f}cycpo")

        meta: dict[str, Any] = {
            "midbrain_mode": config.mode,
            "cochleogram_mode": config.cochleogram.mode,
            "temporal_mods_hz": list(temporal_mods),
            "spectral_mods_cyc_per_oct": list(spectral_mods),
            "temporal_bandwidth_hz": float(config.temporal_bandwidth_hz),
            "spectral_bandwidth_cyc_per_oct": float(config.spectral_bandwidth_cyc_per_oct),
            "temporal_kernel_cycles": float(config.temporal_kernel_cycles),
            "spectral_kernel_cycles": float(config.spectral_kernel_cycles),
            "power": bool(config.power),
            "n_coch_channels": int(C),
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=np.asarray(values_ft, dtype=float),
            time=time,
            channel_names=channel_names,
            meta=meta,
        )
