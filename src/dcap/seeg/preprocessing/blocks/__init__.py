# =============================================================================
#                     ########################################
#                     #        BLOCKS PUBLIC EXPORTS         #
#                     ########################################
# =============================================================================

from dcap.seeg.preprocessing.blocks.bad_channels import suggest_bad_channels
from dcap.seeg.preprocessing.blocks.coordinates import attach_coordinates
from dcap.seeg.preprocessing.blocks.filtering import compute_gamma_envelope, highpass_filter
from dcap.seeg.preprocessing.blocks.line_noise import remove_line_noise
from dcap.seeg.preprocessing.blocks.rereference import rereference
from dcap.seeg.preprocessing.blocks.resample import resample_raw

__all__ = [
    "attach_coordinates",
    "remove_line_noise",
    "highpass_filter",
    "compute_gamma_envelope",
    "resample_raw",
    "suggest_bad_channels",
    "rereference",
]
