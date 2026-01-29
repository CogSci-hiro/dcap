# =============================================================================
# =============================================================================
#                      ########################################
#                      #        PREPROCESSING CONFIGS         #
#                      ########################################
# =============================================================================
# =============================================================================

from dcap.seeg.preprocessing.configs.coordinates import CoordinatesConfig
from dcap.seeg.preprocessing.configs.filtering import GammaEnvelopeConfig, HighpassConfig
from dcap.seeg.preprocessing.configs.line_noise import LineNoiseConfig
from dcap.seeg.preprocessing.configs.resample import ResampleConfig
from dcap.seeg.preprocessing.configs.rereference import RereferenceConfig
from dcap.seeg.preprocessing.configs.clinical import ClinicalPreprocConfig

__all__ = [
    "CoordinatesConfig",
    "GammaEnvelopeConfig",
    "HighpassConfig",
    "LineNoiseConfig",
    "ResampleConfig",
    "RereferenceConfig",
    "ClinicalPreprocConfig",
]
