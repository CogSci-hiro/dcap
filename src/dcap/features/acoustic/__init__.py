from dcap.features.registry import register_feature
from dcap.features.acoustic.praat_intensity import PraatIntensityComputer

register_feature("acoustic.praat_intensity", PraatIntensityComputer)