# =============================================================================
# =============================================================================
#                 ############################################
#                 #       POSTPROCESSING: DERIVATIVES        #
#                 ############################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


# =============================================================================
#                                  CONSTANTS
# =============================================================================

EPS: float = 1e-12


# =============================================================================
#                                   TYPES
# =============================================================================

DerivativeMode = Literal["none", "diff", "absdiff", "hr", "hr_absdiff"]
RectifyMode = Literal["none", "abs"]


# =============================================================================
#                                   CONFIG
# =============================================================================

@dataclass(frozen=True, slots=True)
class DerivativeSpec:
    """
    Specification for a derivative post-processing step.

    Parameters
    ----------
    mode
        Derivative mode:
        - "none": no derivative
        - "diff": first discrete difference (scaled by sfreq)
        - "absdiff": abs(diff)
        - "hr": high-rate derivative (upsample -> diff -> downsample), signed
        - "hr_absdiff": abs(hr)
    hr_factor
        Upsampling factor used by "hr" modes. Must be >= 1.
        hr_factor=1 makes "hr" identical to "diff" (up to numerical noise).

    Usage example
    -------------
        spec = DerivativeSpec(mode="hr_absdiff", hr_factor=4)
        y = apply_derivative(x, sfreq=100.0, spec=spec)
    """
    mode: DerivativeMode = "none"
    hr_factor: int = 4


# =============================================================================
#                                   HELPERS
# =============================================================================

def _ensure_time_last(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Ensure input is either (T,) or (F, T). Time must be the last axis.

    Returns a float64 array view/copy.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1D (T,) or 2D (F,T). Got shape={x.shape}.")


def _diff_scaled(x: NDArray[np.floating], sfreq: float) -> NDArray[np.floating]:
    """
    First difference with units/sec scaling:
        d[t] = (x[t] - x[t-1]) * sfreq
    with d[0] = 0 for a stable, length-preserving output.
    """
    if sfreq <= 0:
        raise ValueError("sfreq must be positive.")

    y = np.zeros_like(x, dtype=float)
    if x.shape[-1] <= 1:
        return y

    y[..., 1:] = np.diff(x, axis=-1) * float(sfreq)
    return y


def _resample_time_last(
    x: NDArray[np.floating],
    *,
    up: int,
    down: int,
    target_len: int,
) -> NDArray[np.floating]:
    """
    Resample along the last axis using polyphase filtering and force length.

    Notes
    -----
    resample_poly can return length off by 1 depending on filter length;
    we crop/pad to `target_len`.
    """
    if up <= 0 or down <= 0:
        raise ValueError("up/down must be positive integers.")
    if target_len <= 0:
        raise ValueError("target_len must be positive.")

    y = resample_poly(x, up=up, down=down, axis=-1).astype(float, copy=False)  # noqa

    n = y.shape[-1]
    if n == target_len:
        return y
    if n > target_len:
        return y[..., :target_len]
    pad_width = [(0, 0)] * y.ndim
    pad_width[-1] = (0, target_len - n)
    return np.pad(y, pad_width=pad_width, mode="constant", constant_values=0.0)


# =============================================================================
#                                   PUBLIC API
# =============================================================================

def apply_derivative(
    x: NDArray[np.floating],
    *,
    sfreq: float,
    spec: DerivativeSpec,
) -> NDArray[np.floating]:
    """
    Apply derivative post-processing.

    Parameters
    ----------
    x
        Input array, shape (T,) or (F, T).
    sfreq
        Sampling frequency of x (Hz).
    spec
        Derivative specification.

    Returns
    -------
    y
        Array of same shape as x, with derivative applied.
    """
    x = _ensure_time_last(x)
    if sfreq <= 0:
        raise ValueError("sfreq must be positive.")

    mode = spec.mode
    hr_factor = int(spec.hr_factor)

    if hr_factor < 1:
        raise ValueError("spec.hr_factor must be >= 1.")

    if mode == "none":
        return np.asarray(x, dtype=float)

    if mode == "diff":
        return _diff_scaled(x, sfreq=sfreq)

    if mode == "absdiff":
        return np.abs(_diff_scaled(x, sfreq=sfreq))

    if mode in ("hr", "hr_absdiff"):
        # High-rate derivative:
        # 1) Upsample by hr_factor
        # 2) Differentiate at sfreq_hr = sfreq * hr_factor
        # 3) Downsample back to original length
        #
        # IMPORTANT: hr_factor=1 should reduce to diff exactly (up to tiny numerics).
        if hr_factor == 1:
            y = _diff_scaled(x, sfreq=sfreq)
        else:
            # Upsample: length ~ T * hr_factor
            up_len = int(x.shape[-1] * hr_factor)
            x_up = _resample_time_last(x, up=hr_factor, down=1, target_len=up_len)

            sfreq_hr = float(sfreq) * float(hr_factor)
            d_up = _diff_scaled(x_up, sfreq=sfreq_hr)

            # Downsample back to T
            y = _resample_time_last(d_up, up=1, down=hr_factor, target_len=int(x.shape[-1]))

        if mode == "hr_absdiff":
            return np.abs(y)
        return y

    raise ValueError(f"Unknown derivative mode: {mode!r}")


def parse_derivative(
    derivative: str,
    *,
    hr_factor: int = 4,
) -> DerivativeSpec:
    """
    Backwards-compatible parser for the string conventions used in earlier feature configs.

    Supports:
    - "none"
    - "diff"
    - "absdiff"
    - "hr"
    - "hr_absdiff"

    Usage example
    -------------
        spec = parse_derivative("hr_absdiff", hr_factor=4)
        y = apply_derivative(x, sfreq=100.0, spec=spec)
    """
    d = derivative.strip().lower()
    valid = {"none", "diff", "absdiff", "hr", "hr_absdiff"}
    if d not in valid:
        raise ValueError(f"Unsupported derivative={derivative!r}. Supported: {sorted(valid)}")
    return DerivativeSpec(mode=d, hr_factor=int(hr_factor))
