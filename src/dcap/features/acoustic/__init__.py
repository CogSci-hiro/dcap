from dcap.features.registry import register_feature
from dcap.features.acoustic.hilbert_env import HilbertEnvelopeComputer
from dcap.features.acoustic.praat_intensity import PraatIntensityComputer

register_feature("acoustic.hilbert_envelope", HilbertEnvelopeComputer)
register_feature("acoustic.praat_intensity", PraatIntensityComputer)
