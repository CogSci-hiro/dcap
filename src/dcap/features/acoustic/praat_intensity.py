# =============================================================================
# =============================================================================
#                #############################################
#                #     BLOCK: PRAAT INTENSITY ENVELOPE       #
#                #############################################
# =============================================================================
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.signal import resample_poly

from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.postprocess import apply_derivative
from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase


_DEFAULT_DOWNSMIX: str = "mean"


def _require_parselmouth() -> Any:
    try:
        import parselmouth  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Praat intensity requires the optional dependency 'praat-parselmouth'. "
            "Install it with: pip install praat-parselmouth"
        ) from exc
    return parselmouth


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
    if np.isclose(input_sfreq, output_sfreq):
        return x
    ratio = float(output_sfreq) / float(input_sfreq)
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x, up=up, down=down)


def _force_length(*, x: np.ndarray, n_times: int) -> np.ndarray:
    if x.shape[0] == n_times:
        return x
    if x.shape[0] > n_times:
        return x[:n_times]
    return np.pad(x, (0, n_times - x.shape[0]), mode="constant", constant_values=0.0)


def _interp_to_grid(*, t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    # Praat intensity has finite support; outside range, use edge values (or 0)
    if y_src.size == 0:
        return np.zeros_like(t_dst, dtype=float)
    return np.interp(t_dst, t_src, y_src, left=float(y_src[0]), right=float(y_src[-1]))


@dataclass(frozen=True)
class PraatIntensityConfig(FeatureConfig):
    """Configuration for Praat intensity contour.

    Parameters
    ----------
    derivative
        Post-processing derivative mode: "none" | "diff" | "absdiff".
    minimum_pitch_hz
        Praat parameter: "Minimum pitch" (Hz). Controls the analysis window length
        (Praat uses ~3.2 / pitchFloor seconds effective window). :contentReference[oaicite:2]{index=2}
    time_step_s
        Praat parameter: "Time step" (s). If None, Praat chooses a default.
        For TRF grids, a sensible explicit value is often 0.01 (100 Hz) or 0.005 (200 Hz).
    subtract_mean
        Praat parameter: subtract mean pressure before analysis.
    output_scale
        "db" returns the Praat contour (dB).
        "linear_power" returns 10 ** (dB / 10).

    Usage example
    -------------
        time = FeatureTimeBase(sfreq=100.0, n_times=24_000, t0_s=0.0)
        cfg = PraatIntensityConfig(time_step_s=0.01, minimum_pitch_hz=75.0, output_scale="db")

        comp = PraatIntensityComputer()
        feat = comp.compute(time=time, audio=wav, audio_sfreq=48_000.0, config=cfg)
    """

    minimum_pitch_hz: float = 100.0
    time_step_s: Optional[float] = None
    subtract_mean: bool = True
    output_scale: str = "db"
    downmix: str = _DEFAULT_DOWNSMIX


class PraatIntensityComputer(FeatureComputer[PraatIntensityConfig]):
    @property
    def name(self) -> str:
        return "acoustic.praat_intensity"

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
        config: PraatIntensityConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        if audio is None or audio_sfreq is None:
            raise ValueError("PraatIntensityComputer requires audio and audio_sfreq.")

        parselmouth = _require_parselmouth()

        mono = _as_mono_audio(audio=audio, downmix=config.downmix)

        # Parselmouth Sound expects samples in float and a sampling frequency
        snd = parselmouth.Sound(values=mono, sampling_frequency=float(audio_sfreq))

        intensity_obj = snd.to_intensity(
            minimum_pitch=float(config.minimum_pitch_hz),
            time_step=None if config.time_step_s is None else float(config.time_step_s),
            subtract_mean=bool(config.subtract_mean),
        )  # :contentReference[oaicite:3]{index=3}

        # Intensity values are stored as a Matrix-like grid (frames)
        # intensity_obj.values has shape (1, n_frames)
        db_values = np.asarray(intensity_obj.values, dtype=float).squeeze(axis=0)

        # Praat/Parselmouth can produce NaNs or extreme values in silence.
        # Normalize handling so db and linear_power are consistent.
        if np.any(~np.isfinite(db_values)):
            finite_mask = np.isfinite(db_values)
            if np.any(finite_mask):
                floor = float(np.min(db_values[finite_mask]))
            else:
                floor = -300.0
            db_values = np.where(np.isfinite(db_values), db_values, floor)

        # Time axis: intensity_obj.xs() returns frame times in seconds (Parselmouth API)
        # We avoid depending on too many helpers: compute from x1 + dx if needed.
        dx = float(intensity_obj.dx)
        # Prefer x1 if present (frame center of first column in Praat objects)
        x1 = float(getattr(intensity_obj, "x1", intensity_obj.xmin + 0.5 * dx))
        n_frames = int(db_values.shape[0])
        t_src = x1 + dx * np.arange(n_frames, dtype=float)

        # Convert scale if requested
        if config.output_scale == "db":
            y_src = db_values
        elif config.output_scale == "linear_power":
            y_src = np.power(10.0, db_values / 10.0)
        else:
            raise ValueError("output_scale must be 'db' or 'linear_power'.")

        # Align to requested FeatureTimeBase grid
        t_dst = float(time.t0_s) + (np.arange(int(time.n_times), dtype=float) / float(time.sfreq))
        y = _interp_to_grid(t_src=t_src, y_src=y_src, t_dst=t_dst)

        y = apply_derivative(x=y, sfreq=float(time.sfreq), mode=config.derivative)

        meta: dict[str, Any] = {
            "audio_sfreq": float(audio_sfreq),
            "target_sfreq": float(time.sfreq),
            "minimum_pitch_hz": float(config.minimum_pitch_hz),
            "time_step_s": None if config.time_step_s is None else float(config.time_step_s),
            "subtract_mean": bool(config.subtract_mean),
            "output_scale": config.output_scale,
            "downmix": config.downmix,
            "derivative": config.derivative,
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=y.astype(float, copy=False),
            time=time,
            channel_names=["intensity"],
            meta=meta,
        )
