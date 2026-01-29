# =============================================================================
# =============================================================================
#                     ########################################
#                     #      CONFIG: CLINICAL PREPROCESS     #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional

from dcap.seeg.preprocessing.configs.coordinates import CoordinatesConfig
from dcap.seeg.preprocessing.configs.filtering import GammaEnvelopeConfig, HighpassConfig
from dcap.seeg.preprocessing.configs.line_noise import LineNoiseConfig
from dcap.seeg.preprocessing.configs.resample import ResampleConfig
from dcap.seeg.preprocessing.configs.rereference import RereferenceConfig


@dataclass(frozen=True)
class ClinicalPreprocConfig:
    """
    Configuration bundle for the common clinical preprocessing pipeline.

    Attributes
    ----------
    do_coordinates
        If True, run coordinates attachment when electrodes_table + coords are provided.
    do_line_noise
        If True, remove line noise.
    do_highpass
        If True, apply high-pass filter.
    do_resample
        If True, resample to target sfreq.
    do_rereference
        If True, generate rereferenced views.
    coords
        Coordinate attachment configuration.
    line_noise
        Line-noise configuration.
    highpass
        High-pass configuration.
    resample
        Resampling configuration.
    rereference
        Rereferencing configuration.
    gamma_envelope
        Optional gamma envelope configuration. This is a feature path and is NOT applied by
        the preprocessing pipeline unless your clinical orchestrator chooses to.

    Usage example
    -------------
        cfg = ClinicalPreprocConfig(
            do_line_noise=True,
            do_highpass=True,
            do_resample=True,
            do_rereference=True,
            resample=ResampleConfig(sfreq_out=512.0),
            rereference=RereferenceConfig(methods=("car", "bipolar")),
        )
    """

    do_coordinates: bool = True
    do_line_noise: bool = True
    do_highpass: bool = True
    do_resample: bool = True
    do_rereference: bool = True

    coords: CoordinatesConfig = field(default_factory=CoordinatesConfig)
    line_noise: LineNoiseConfig = field(default_factory=LineNoiseConfig)
    highpass: HighpassConfig = field(default_factory=HighpassConfig)
    resample: ResampleConfig = field(default_factory=ResampleConfig)
    rereference: RereferenceConfig = field(default_factory=RereferenceConfig)

    gamma_envelope: Optional[GammaEnvelopeConfig] = None
