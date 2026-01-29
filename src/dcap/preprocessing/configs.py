# =============================================================================
#                        ##################################
#                        #   PREPROCESSING CONFIG SCHEMA  #
#                        ##################################
# =============================================================================
#
# Dataclass configs for preprocessing blocks and pipelines.
#
# NOTE
# - These configs are logic-only and may be created by a CLI layer later.
# - Keep "v1" minimal: expose only the knobs you truly want users to touch.
#
# =============================================================================

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CoordinatesConfig:
    """
    Configuration for coordinate attachment.

    Parameters
    ----------
    coord_frame
        Human-readable label describing the coordinate frame.
    unit
        Unit of the provided coordinates.
    compute_neighbors
        Whether to compute and store a neighbor graph (for Laplacian).
    neighbors_k
        Number of nearest neighbors if compute_neighbors is True.
    neighbors_radius_mm
        Optional radius constraint in millimeters.

    Usage example
    -------------
        cfg = CoordinatesConfig(coord_frame="tkr", unit="mm", compute_neighbors=True)
    """

    coord_frame: str = "unknown"
    unit: Literal["m", "mm"] = "mm"
    compute_neighbors: bool = False
    neighbors_k: int = 6
    neighbors_radius_mm: Optional[float] = None


@dataclass(frozen=True)
class LineNoiseConfig:
    """
    Configuration for line-noise removal.

    Parameters
    ----------
    method
        "notch" or "zapline" (meegkit).
    freq_base
        Base line frequency (50 or 60).
    max_harmonic_hz
        Maximum harmonic frequency to process.
    picks
        Optional channel selection.

    Usage example
    -------------
        cfg = LineNoiseConfig(method="notch", freq_base=50, max_harmonic_hz=250.0)
    """

    method: Literal["notch", "zapline"] = "notch"
    freq_base: Literal[50, 60] = 50
    max_harmonic_hz: float = 250.0
    picks: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class HighpassConfig:
    """
    Configuration for high-pass filtering.

    Parameters
    ----------
    l_freq
        High-pass cutoff in Hz.
    phase
        Filter phase behavior (kept minimal in v1).

    Usage example
    -------------
        cfg = HighpassConfig(l_freq=0.5)
    """

    l_freq: float = 0.5
    phase: Literal["zero", "minimum"] = "zero"


@dataclass(frozen=True)
class GammaEnvelopeConfig:
    """
    Configuration for gamma / HFA envelope generation.

    Parameters
    ----------
    band_hz
        (low, high) frequency band in Hz.
    method
        Envelope method.
    smoothing_sec
        Smoothing window in seconds.

    Usage example
    -------------
        cfg = GammaEnvelopeConfig(band_hz=(70.0, 150.0), smoothing_sec=0.1)
    """

    band_hz: Tuple[float, float] = (70.0, 150.0)
    method: Literal["hilbert", "rectified_smooth"] = "hilbert"
    smoothing_sec: float = 0.1


@dataclass(frozen=True)
class ResampleConfig:
    """
    Configuration for resampling.

    Parameters
    ----------
    sfreq_out
        Target sampling frequency.
    method
        Resampling method label (implementation-dependent).

    Usage example
    -------------
        cfg = ResampleConfig(sfreq_out=512.0)
    """

    sfreq_out: float = 512.0
    method: Literal["mne_default"] = "mne_default"


@dataclass(frozen=True)
class BadChannelsConfig:
    """
    Configuration for bad channel suggestion (semi-automatic).

    Parameters
    ----------
    detect_flat
        Suggest channels with near-zero variance.
    detect_high_variance
        Suggest channels with extreme variance.
    detect_line_noise_dominance
        Suggest channels dominated by line noise.
    z_thresh
        Robust Z threshold for outlier metrics.

    Usage example
    -------------
        cfg = BadChannelsConfig(detect_flat=True, z_thresh=6.0)
    """

    detect_flat: bool = True
    detect_high_variance: bool = True
    detect_line_noise_dominance: bool = True
    z_thresh: float = 6.0


@dataclass(frozen=True)
class RereferenceConfig:
    """
    Configuration for rereferencing.

    Parameters
    ----------
    methods
        List of rereferencing methods to generate.
    car_scope
        Scope for CAR.
    bipolar_use_shafts
        If True, build bipolar derivations using shaft ordering.
    laplacian_mode
        Laplacian definition strategy.

    Usage example
    -------------
        cfg = RereferenceConfig(methods=("car", "bipolar", "laplacian"))
    """

    methods: Sequence[Literal["car", "bipolar", "wm_ref", "laplacian"]] = ("car", "bipolar")
    car_scope: Literal["global", "by_shaft"] = "global"
    bipolar_use_shafts: bool = True
    laplacian_mode: Literal["knn_3d", "shaft_1d"] = "shaft_1d"


@dataclass(frozen=True)
class ClinicalPreprocConfig:
    """
    Minimal orchestration config for a clinical-style preprocessing pipeline.

    Notes
    -----
    This config is intentionally conservative. The CLI layer can later expose
    additional knobs, but the library should remain stable.

    Parameters
    ----------
    line_noise
        Line-noise removal configuration.
    highpass
        High-pass filter configuration.
    gamma_envelope
        Gamma envelope configuration (optional).
    resample
        Resampling configuration.
    bad_channels
        Bad channel suggestion configuration.
    rereference
        Rereferencing configuration.

    Usage example
    -------------
        cfg = ClinicalPreprocConfig(
            line_noise=LineNoiseConfig(method="notch", freq_base=50),
            highpass=HighpassConfig(l_freq=0.5),
        )
    """

    line_noise: LineNoiseConfig = LineNoiseConfig()
    highpass: HighpassConfig = HighpassConfig()
    gamma_envelope: Optional[GammaEnvelopeConfig] = None
    resample: Optional[ResampleConfig] = None
    bad_channels: BadChannelsConfig = BadChannelsConfig()
    rereference: RereferenceConfig = RereferenceConfig()
