# =============================================================================
#                         Analysis: TRF (metrics)
# =============================================================================
#
# Minimal metric utilities for TRF evaluation.
#
# =============================================================================

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def pearson_corr_by_output(y_true: FloatArray, y_pred: FloatArray) -> FloatArray:
    """Compute Pearson correlation per output channel.

    Parameters
    ----------
    y_true
        Ground truth, shape (n_samples, n_outputs) or (n_samples,).
    y_pred
        Predictions, shape (n_samples, n_outputs) or (n_samples,).

    Returns
    -------
    corr
        Correlation per output, shape (n_outputs,).

    Usage example
    -------------
        corr = pearson_corr_by_output(y_true, y_pred)
    """
    y_true_2d = _as_2d(y_true)
    y_pred_2d = _as_2d(y_pred)

    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(f"Shape mismatch: {y_true_2d.shape!r} vs {y_pred_2d.shape!r}.")

    raise NotImplementedError("TODO: implement correlation.")  # noqa: TRY003


def r2_by_output(y_true: FloatArray, y_pred: FloatArray) -> FloatArray:
    """Compute R^2 per output channel.

    Usage example
    -------------
        r2 = r2_by_output(y_true, y_pred)
    """
    y_true_2d = _as_2d(y_true)
    y_pred_2d = _as_2d(y_pred)

    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(f"Shape mismatch: {y_true_2d.shape!r} vs {y_pred_2d.shape!r}.")

    raise NotImplementedError("TODO: implement R^2.")  # noqa: TRY003


def _as_2d(x: FloatArray) -> FloatArray:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1D or 2D array, got shape={x.shape!r}.")
