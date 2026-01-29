from dataclasses import dataclass, field
from typing import Optional

from dcap.seeg.preprocessing.configs.coordinates import CoordinatesConfig
from dcap.seeg.preprocessing.configs.filtering import GammaEnvelopeConfig, HighpassConfig
from dcap.seeg.preprocessing.configs.line_noise import LineNoiseConfig
from dcap.seeg.preprocessing.configs.resample import ResampleConfig
from dcap.seeg.preprocessing.configs.rereference import RereferenceConfig

@dataclass(frozen=True)
class ClinicalPreprocConfig:
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
