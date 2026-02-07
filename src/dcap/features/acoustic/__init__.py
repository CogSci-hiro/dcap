from dcap.features.registry import register_feature
from dcap.features.acoustic.hilbert_env import HilbertEnvelopeComputer
from dcap.features.acoustic.praat_intensity import PraatIntensityComputer
from dcap.features.acoustic.varnet_env import VarnetEnvelopeComputer
from dcap.features.acoustic.oganian_env import OganianEnvelopeComputer
from dcap.features.acoustic.spectrogram import SpectrogramComputer
from dcap.features.acoustic.mel_spectrogram import MelSpectrogramComputer
from dcap.features.acoustic.mfcc import MfccComputer
from dcap.features.acoustic.cochleogram import CochleogramComputer
from dcap.features.acoustic.midbrain import MidbrainComputer

register_feature("acoustic.hilbert_envelope", HilbertEnvelopeComputer)
register_feature("acoustic.praat_intensity", PraatIntensityComputer)
register_feature("acoustic.varnet_envelope", VarnetEnvelopeComputer)
register_feature("acoustic.oganian_envelope", OganianEnvelopeComputer)
register_feature("acoustic.spectrogram", SpectrogramComputer)
register_feature("acoustic.mel_spectrogram", MelSpectrogramComputer)
register_feature("acoustic.cochleogram", CochleogramComputer)
register_feature("acoustic.midbrain", MidbrainComputer)
register_feature("acoustic.mfcc", MfccComputer)
