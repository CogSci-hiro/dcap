# =============================================================================
#                         TRF analysis: alignment
# =============================================================================
#
# Minimal alignment utilities shared by task adapters.
#
# =============================================================================

import numpy as np

# review
def event_time_to_sample(onset_s: float, sfreq: float, first_samp: int = 0) -> int:
    """
    Convert an onset time in seconds to a sample index.

    Parameters
    ----------
    onset_s : float
        Event onset time in seconds.
    sfreq : float
        Sampling frequency (Hz).
    first_samp : int
        Optional offset (e.g., MNE Raw.first_samp).

    Returns
    -------
    sample : int
        Sample index corresponding to onset.

    Usage example
    -------------
        start_samp = event_time_to_sample(onset_s=12.34, sfreq=1000.0, first_samp=0)
    """
    if sfreq <= 0:
        raise ValueError("`sfreq` must be > 0.")
    return int(np.rint(onset_s * sfreq)) + int(first_samp)


def align_by_event_sample(
    x: np.ndarray,
    x_start_sample: int,
    y: np.ndarray,
    y_start_sample: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Align two time-series arrays by their start sample indices.

    Parameters
    ----------
    x : ndarray
        First array with time axis=0.
    x_start_sample : int
        Sample index of x[0] in a shared timeline.
    y : ndarray
        Second array with time axis=0.
    y_start_sample : int
        Sample index of y[0] in a shared timeline.

    Returns
    -------
    x_aligned : ndarray
        Cropped x such that its timeline matches y_aligned.
    y_aligned : ndarray
        Cropped y such that its timeline matches x_aligned.
    start_sample : int
        Shared start sample in the original timeline.

    Notes
    -----
    The returned arrays are cropped to the maximal overlap.

    Usage example
    -------------
        x2, y2, start = align_by_event_sample(x, 1000, y, 1200)
    """
    x0 = int(x_start_sample)
    y0 = int(y_start_sample)

    start = max(x0, y0)
    x_offset = start - x0
    y_offset = start - y0

    x_len = x.shape[0] - x_offset
    y_len = y.shape[0] - y_offset
    overlap = min(x_len, y_len)

    if overlap <= 0:
        raise ValueError("No overlap after alignment.")

    return x[x_offset:x_offset + overlap, ...], y[y_offset:y_offset + overlap, ...], start
