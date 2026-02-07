# =============================================================================
# =============================================================================
#                 ############################################
#                 #      BLOCK: OGANIAN-STYLE ENVELOPE       #
#                 ############################################
# =============================================================================
# =============================================================================
"""
Oganian-style speech envelope feature.

This module implements two envelope variants commonly used in the "Oganian-style"
peak-rate / envelope-tracking literature:

1) envtype="broadband"
   A simple broadband envelope:
       rectified waveform -> low-pass filter (10 Hz by default) -> resample
   This is close to:  env(t) = LPF(|x(t)|)

2) envtype="loudness"
   A more complex "specific loudness"-like proxy adapted from the provided MATLAB
   (function name in your MATLAB looked like Syl_1984_Schotola or similar):
       - FFT of waveform
       - For each critical-band center (Bark bands 1..22), apply a frequency weighting
       - Transform back to time domain (IFFT), square-law rectify, smooth with a
         1.3 ms RC low-pass (forward + reverse for zero phase)
       - Log-compress
       - Combine bands with signed weights (some bands contribute positively, some negatively)
       - Smooth the resulting loudness proxy with repeated zero-phase moving-average filtering

The "loudness" branch intentionally preserves several nonstandard implementation
details from the MATLAB:
   - The frequency weighting is applied only to the positive-frequency bins.
     Then real(ifft(...)) is taken. This is not a textbook symmetric real FIR filter,
     but we keep it for fidelity to your reference implementation.
   - The final smoothing is a 3-sample moving average applied 13 times with filtfilt,
     matching the MATLAB loop.

Output contract
---------------
The feature returns a single 1D signal aligned to FeatureTimeBase:
    values.shape == (time.n_times,)

The feature supports post-processing derivatives via `FeatureConfig.derivative`:
    "none"     -> raw envelope
    "diff"     -> time-derivative (finite difference scaled by sfreq)
    "absdiff"  -> absolute time-derivative

Important units / scaling
-------------------------
- This code does not assume physical calibration of the input waveform.
- Absolute amplitude units are arbitrary; downstream TRF should usually z-score.

Implementation note on sampling
-------------------------------
For both envtypes we compute the envelope at some "envelope sampling rate" and then
align to the target FeatureTimeBase grid (time.sfreq, time.n_times).

Usage example
-------------
    time = FeatureTimeBase(sfreq=100.0, n_times=24_000, t0_s=0.0)
    cfg = OganianEnvelopeConfig(envtype="loudness", derivative="none")

    comp = OganianEnvelopeComputer()
    out = comp.compute(time=time, audio=wav, audio_sfreq=48_000.0, config=cfg)
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, resample_poly

from dcap.features.base import FeatureComputer, FeatureConfig
from dcap.features.postprocess import apply_derivative
from dcap.features.types import FeatureKind, FeatureResult, FeatureTimeBase

# =============================================================================
#                             CONSTANTS / DEFAULTS
# =============================================================================

# How to downmix multi-channel audio to mono.
# Convention in this codebase: multi-channel audio is shape (n_channels, n_times).
_DEFAULT_DOWNSMIX: str = "mean"

# The "broadband" envelope low-pass cutoff is 10 Hz in the MATLAB reference.
_DEFAULT_BROADBAND_LP_HZ: float = 10.0

# Numerical floor to avoid log(0) and log of negative tiny values after smoothing.
# This is a "power-like" floor (because we take log of squared/smoothed signals).
_TINY_POWER: float = 1e-20


# =============================================================================
#                             HELPER: DOWNMIX TO MONO
# =============================================================================
def _as_mono_audio(*, audio: np.ndarray, downmix: str) -> np.ndarray:
    """
    Convert audio to a 1D mono waveform.

    Parameters
    ----------
    audio
        Either shape (n_times,) for mono, or (n_channels, n_times) for multi-channel.
    downmix
        Strategy for multi-channel conversion. Currently only:
        - "mean": average across channels

    Returns
    -------
    np.ndarray
        Mono waveform, shape (n_times,).

    Notes
    -----
    We enforce a single standard here because envelope computation is typically
    defined on a single waveform. If later you want per-channel envelopes,
    we'd return (n_channels, n_times) instead and adjust FeatureResult accordingly.
    """
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


# =============================================================================
#                         HELPERS: RESAMPLING (POLYPHASE)
# =============================================================================
def _rational_approximation(*, ratio: float, max_denominator: int) -> tuple[int, int]:
    """
    Approximate a float ratio with an integer rational up/down.

    We use Fraction.limit_denominator() to find integers (up, down) such that:
        up / down ≈ ratio

    This is used for scipy.signal.resample_poly, which is stable and anti-aliasing.
    """
    from fractions import Fraction

    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def _resample_1d(*, x: np.ndarray, input_sfreq: float, output_sfreq: float) -> np.ndarray:
    """
    Resample a 1D signal from input_sfreq to output_sfreq using polyphase filtering.

    Parameters
    ----------
    x
        1D array, shape (n_times,).
    input_sfreq
        Sampling frequency of x (Hz).
    output_sfreq
        Desired sampling frequency (Hz).

    Returns
    -------
    np.ndarray
        Resampled signal at output_sfreq.

    Notes
    -----
    - We use resample_poly rather than FFT resampling for better anti-aliasing.
    - For envelope signals (low-frequency), this is generally robust and efficient.
    """
    if input_sfreq <= 0 or output_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    if np.isclose(input_sfreq, output_sfreq):
        return x

    ratio = float(output_sfreq) / float(input_sfreq)
    up, down = _rational_approximation(ratio=ratio, max_denominator=10_000)
    return resample_poly(x, up=up, down=down)


def _force_length(*, x: np.ndarray, n_times: int) -> np.ndarray:
    """
    Crop or pad a 1D signal to exactly n_times samples.

    This guarantees strict alignment to FeatureTimeBase.n_times, which prevents
    downstream off-by-one errors (common in TRF pipelines).
    """
    if n_times <= 0:
        raise ValueError("n_times must be positive.")
    if x.shape[0] == n_times:
        return x
    if x.shape[0] > n_times:
        return x[:n_times]
    return np.pad(x, (0, n_times - x.shape[0]), mode="constant", constant_values=0.0)


# =============================================================================
#                     BROADBAND ENVELOPE (SIMPLE VARIANT)
# =============================================================================
def _oganian_broadband_envelope(
    *,
    x: np.ndarray,
    sfreq: float,
    target_sfreq: float,
    lowpass_hz: float,
) -> np.ndarray:
    """
    Broadband envelope: rectify -> 2nd order Butterworth LPF -> zero-phase -> resample.

    This is a faithful port of the MATLAB "broadband" branch:

        env = abs(x)
        [b,a] = butter(2, cutoff/(fs/2))
        env = filtfilt(b,a, env)
        env = resample(env, targetFs, fs)

    Parameters
    ----------
    x
        Mono waveform, shape (n_times,).
    sfreq
        Waveform sampling rate (Hz).
    target_sfreq
        Desired envelope sampling rate (Hz) before final alignment to FeatureTimeBase.
    lowpass_hz
        Low-pass cutoff frequency (Hz), typically 10 Hz.

    Returns
    -------
    np.ndarray
        Envelope at target_sfreq, shape (n_times_env,).

    Notes
    -----
    - Rectification (abs) converts the audio waveform to an amplitude-like signal.
    - LPF at ~10 Hz extracts slow amplitude modulations (syllabic rate).
    - filtfilt produces zero phase (no group delay).
    - We enforce non-negativity after filtering (small negative ripple can occur).
    """
    rect = np.abs(x)

    if lowpass_hz <= 0:
        raise ValueError("lowpass_hz must be positive.")
    if lowpass_hz >= 0.5 * sfreq:
        raise ValueError(f"lowpass_hz must be < Nyquist={0.5 * sfreq} (sfreq={sfreq}).")

    # 2nd order Butterworth low-pass. Normalized cutoff is Wn = fc / (fs/2)
    b, a = butter(N=2, Wn=float(lowpass_hz) / (0.5 * float(sfreq)), btype="low")  # noqa

    # Zero-phase filtering (forward+backward).
    env = filtfilt(b, a, rect)

    # Resample to an envelope sampling rate (often equal to target FeatureTimeBase.sfreq).
    env = _resample_1d(x=env, input_sfreq=sfreq, output_sfreq=target_sfreq)

    # Numerical cleanup: keep envelope non-negative.
    env[env < 0] = 0.0

    return env


# =============================================================================
#             LOUDNESS BRANCH: RC SMOOTHING (ZERO-PHASE 1ST ORDER)
# =============================================================================
def _rc_smooth_zero_phase(*, x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """
    Apply a 1st-order RC low-pass filter in forward direction and then reverse direction.

    This reproduces MATLAB:
        r = exp(-t/tau);
        b = 1-r; a = [1 -r];
        y = filter(b,a,x);
        y = flipud(filter(b,a,flipud(y)));

    Interpretation
    --------------
    This is an exponential smoother with time constant tau:
        y[n] = (1-r)*x[n] + r*y[n-1]
    where r = exp(-dt/tau), dt = 1/fs.

    Forward+reverse produces zero phase (no net delay), similar to filtfilt,
    but for this specific 1st-order IIR.
    """
    if tau_s <= 0:
        raise ValueError("tau_s must be positive.")

    dt = 1.0 / float(fs)
    r = float(np.exp(-dt / float(tau_s)))

    # Filter coefficients:
    #   b = [1-r]
    #   a = [1, -r]
    b = np.array([1.0 - r], dtype=float)
    a = np.array([1.0, -r], dtype=float)

    # Forward IIR filtering
    y = lfilter(b, a, x)

    # Reverse and filter again (zero-phase)
    y = np.flipud(lfilter(b, a, np.flipud(y)))

    return y


# =============================================================================
#              LOUDNESS BRANCH: SPECIFIC-LOUDNESS-LIKE ENVELOPE
# =============================================================================
def _oganian_specific_loudness_envelope(
    *,
    p: np.ndarray,
    fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Loudness proxy ported from MATLAB "Syl_1984_Schotola".

    This computes a single 1D envelope by:
    1) Projecting the waveform into 22 Bark-centered bands via frequency-domain weightings
    2) For each band:
       - transform back to time domain
       - square-law rectify
       - smooth with 1.3 ms RC low-pass (zero-phase)
       - log compress
       - exponentiate with exp(0.5*log(.)) = sqrt(.), then interpolate to target_fs
    3) Combine bands with a fixed weight vector gv (some bands contribute negatively)
    4) Smooth with 3-point moving average, applied 13× with filtfilt (zero-phase)

    Parameters
    ----------
    p
        Mono waveform (pressure-like signal), shape (n_times,).
    fs
        Audio sampling rate (Hz).
    target_fs
        Output sampling rate of the envelope (Hz).

    Returns
    -------
    np.ndarray
        Loudness proxy envelope at target_fs.

    Fidelity notes
    --------------
    - We preserve the MATLAB "one-sided spectrum shaping" behavior.
      Concretely: we fill weights only for bins 1..(N/2-1) and leave everything
      else (including negative frequencies) at zero, then take real(ifft()).
      This is unusual but kept for fidelity.
    - The Bark scale mapping is exactly the MATLAB formula (Zwicker & Terhardt 1980).
    """
    # Ensure 1D float
    p = np.asarray(p, dtype=float).reshape(-1)
    n = int(p.shape[0])

    if n == 0:
        return np.zeros(0, dtype=float)

    # Time axis for original waveform samples
    tN = np.arange(n, dtype=float) / float(fs)

    # Determine how many samples we want at target_fs over the same duration
    # MATLAB: N1 = fix(N*targetFs/fs)
    n1 = int(np.fix(n * float(target_fs) / float(fs)))
    t1 = np.arange(n1, dtype=float) / float(target_fs)

    # FFT of the waveform
    # MATLAB: P = fft(p, N)
    P = np.fft.fft(p, n=n)

    # Positive frequencies including DC and Nyquist
    # MATLAB: frqs = (0:N/2)*(fs/N)
    frqs = np.arange(0, n // 2 + 1, dtype=float) * (float(fs) / float(n))
    nfrqs = int(frqs.shape[0])

    # Bark scale transform (Zwicker & Terhardt 1980):
    # z = 13*atan(.76*f/1000) + 3.5*(atan(f/7500))^2
    z = 13.0 * np.arctan(0.76 * frqs / 1000.0) + 3.5 * (np.arctan(frqs / 7500.0) ** 2)

    # MATLAB uses: z = z(2:nfrqs-1)
    # Meaning: exclude DC (index 0) and Nyquist (last index), effectively setting them to 0.
    z_mid = z[1 : nfrqs - 1]

    # RC smoothing time constant in seconds (1.3 ms)
    tau = 0.0013

    # Bark center indices czs = 1..22 (inclusive)
    czs = np.arange(1, 23, dtype=float)
    nczs = int(czs.shape[0])

    # Nv will store the per-band envelopes sampled at target_fs
    # Shape: (n1 time points, 22 bands)
    Nv = np.zeros((n1, nczs), dtype=float)

    # Frequency-domain weighting vector (length N) reused per band.
    # We fill only positive frequencies (excluding DC and Nyquist), leave others at 0.
    F = np.zeros((n,), dtype=float)

    for idx, cz in enumerate(czs):
        # MATLAB:
        # delta = (z - cz) - 0.215
        # F(2:nfrqs-1) = 10^( 0.7 - 0.75*delta - 1.75*(0.196 + delta^2) )
        #
        # This defines a broad critical-band-like weighting as a function of Bark distance.
        delta = (z_mid - cz) - 0.215
        shape = np.power(
            10.0,
            (0.7 - 0.75 * delta - 1.75 * (0.196 + (delta**2))),
        )

        # Reset and fill the mid positive-frequency bins.
        F.fill(0.0)
        F[1 : nfrqs - 1] = shape

        # Apply the weighting in the frequency domain and invert back to time.
        # Because F is real, P*F is complex; real(ifft()) keeps the real part.
        lev = np.real(np.fft.ifft(P * F, n=n))

        # Square-law rectification: power-like quantity
        lev = lev**2

        # Smooth with 1.3 ms exponential smoother (zero-phase)
        lev = _rc_smooth_zero_phase(x=lev, fs=fs, tau_s=tau)

        # Log compression: ln(power)
        # The floor prevents -inf and undefined values.
        lev = np.log(np.maximum(lev, _TINY_POWER))

        # MATLAB: Nv(:, band) = interp1q(tN, exp(0.5*Lev), t1)
        #
        # exp(0.5*log(power)) = sqrt(power), i.e., back to amplitude-like scale.
        # Note: "interp1q" is fast linear interpolation; numpy.interp is linear too.
        nv = np.interp(t1, tN, np.exp(0.5 * lev))

        Nv[:, idx] = nv

    # Enforce non-negativity (mostly redundant but consistent with MATLAB)
    Nv = np.maximum(0.0, Nv)

    # gv weights: vector length 22
    # MATLAB:
    #   gv = ones(nczs,1);
    #   gv(czs<3) = 0;        (bands 1-2 are set to 0 contribution)
    #   gv(czs>19) = -1;      (bands 20-22 are negative contribution)
    gv = np.ones((nczs,), dtype=float)
    gv[:2] = 0.0          # cz=1,2
    gv[19:] = -1.0        # cz=20,21,22

    # Combine band envelopes into a single loudness proxy:
    # Nm(t) = sum_band Nv(t,band) * gv(band)
    Nm = Nv @ gv

    # Final temporal smoothing:
    # MATLAB:
    #   b = ones(3,1)/3;
    #   for i=1:13, Nm = filtfilt(b,1,Nm); end
    #
    # This is repeated 3-point moving-average smoothing, applied 13 times, zero-phase.
    b = np.ones((3,), dtype=float) / 3.0
    for _ in range(13):
        Nm = filtfilt(b, [1.0], Nm)

    return Nm


# =============================================================================
#                           PUBLIC CONFIG DATACLASS
# =============================================================================
@dataclass(frozen=True)
class OganianEnvelopeConfig(FeatureConfig):
    """
    Configuration for Oganian-style envelope.

    Parameters
    ----------
    derivative
        Derivative post-processing mode: "none" | "diff" | "absdiff".
        Applied after the envelope is aligned to FeatureTimeBase (i.e., on the TRF grid).
    envtype
        "loudness": use the specific-loudness-like proxy.
        "broadband": use abs + lowpass + resample.
    envelope_sfreq
        Internal sampling rate used for envelope computation output before final alignment.
        If None, defaults to FeatureTimeBase.sfreq.
        - For "loudness", the MATLAB directly outputs at targetFs; we use envelope_sfreq as that target.
        - For "broadband", we resample the filtered rectified envelope to envelope_sfreq.
    broadband_lowpass_hz
        Low-pass cutoff for "broadband" envtype (default 10 Hz, matching MATLAB).
    downmix
        How to convert multi-channel audio to mono. Supported: "mean".
    """

    envtype: str = "loudness"  # allowed: "loudness" | "broadband"
    envelope_sfreq: Optional[float] = None
    broadband_lowpass_hz: float = _DEFAULT_BROADBAND_LP_HZ
    downmix: str = _DEFAULT_DOWNSMIX


# =============================================================================
#                           PUBLIC FEATURE COMPUTER
# =============================================================================
class OganianEnvelopeComputer(FeatureComputer[OganianEnvelopeConfig]):
    """
    Compute Oganian-style envelope and return FeatureResult aligned to FeatureTimeBase.

    Implementation details
    ----------------------
    - Audio is downmixed to mono.
    - Envelope is computed at env_sfreq (defaults to FeatureTimeBase.sfreq).
    - If env_sfreq != FeatureTimeBase.sfreq, envelope is resampled again to time.sfreq.
    - Output is cropped/padded to exactly time.n_times.
    - Optional derivative post-processing is applied on the final grid.
    """

    @property
    def name(self) -> str:
        return "acoustic.oganian_envelope"

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
        config: OganianEnvelopeConfig,
        context: Optional[Mapping[str, Any]] = None,
    ) -> FeatureResult:
        # ---------------------------------------------------------------------
        # Validate required inputs
        # ---------------------------------------------------------------------
        if audio is None or audio_sfreq is None:
            raise ValueError("OganianEnvelopeComputer requires audio and audio_sfreq.")

        # ---------------------------------------------------------------------
        # Step 1: mono waveform
        # ---------------------------------------------------------------------
        x = _as_mono_audio(audio=audio, downmix=config.downmix)

        # ---------------------------------------------------------------------
        # Step 2: choose envelope sampling rate
        # ---------------------------------------------------------------------
        # If not specified, compute envelope on the same grid as the FeatureTimeBase.
        env_sfreq = float(time.sfreq) if config.envelope_sfreq is None else float(config.envelope_sfreq)

        # ---------------------------------------------------------------------
        # Step 3: compute the envelope according to envtype
        # ---------------------------------------------------------------------
        if config.envtype == "broadband":
            # Broadband pipeline:
            #   abs(audio) -> LPF(10 Hz) -> resample to env_sfreq
            env = _oganian_broadband_envelope(
                x=x,
                sfreq=float(audio_sfreq),
                target_sfreq=env_sfreq,
                lowpass_hz=float(config.broadband_lowpass_hz),
            )

        elif config.envtype == "loudness":
            # Loudness proxy pipeline:
            #   Bark-weighted analysis -> combine -> smooth -> output at env_sfreq
            env = _oganian_specific_loudness_envelope(
                p=x,
                fs=float(audio_sfreq),
                target_fs=env_sfreq,
            )
        else:
            raise ValueError("envtype must be 'loudness' or 'broadband'.")

        # ---------------------------------------------------------------------
        # Step 4: resample envelope to FeatureTimeBase.sfreq if needed
        # ---------------------------------------------------------------------
        # This ensures *all* features end on a shared TRF grid, which avoids subtle
        # alignment bugs later.
        if not np.isclose(env_sfreq, float(time.sfreq)):
            env = _resample_1d(x=env, input_sfreq=env_sfreq, output_sfreq=float(time.sfreq))

        # ---------------------------------------------------------------------
        # Step 5: enforce exact length of the time grid
        # ---------------------------------------------------------------------
        env = _force_length(x=env, n_times=int(time.n_times))

        # ---------------------------------------------------------------------
        # Step 6: derivative post-processing (optional)
        # ---------------------------------------------------------------------
        # This is computed on the final grid, so derivative units are:
        #   (envelope-units) per second.
        env = apply_derivative(x=env, sfreq=float(time.sfreq), mode=config.derivative)

        # ---------------------------------------------------------------------
        # Step 7: metadata (provenance)
        # ---------------------------------------------------------------------
        meta: dict[str, Any] = {
            "audio_sfreq": float(audio_sfreq),
            "target_sfreq": float(time.sfreq),
            "envtype": config.envtype,
            "envelope_sfreq": env_sfreq,
            "broadband_lowpass_hz": float(config.broadband_lowpass_hz),
            "downmix": config.downmix,
            "derivative": config.derivative,
        }

        return FeatureResult(
            name=self.name,
            kind="acoustic",
            values=env.astype(float, copy=False),
            time=time,
            channel_names=["env"],
            meta=meta,
        )
