# =============================================================================
# TRF analysis: plotting helpers
# =============================================================================

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .types import TrfModel


KernelView = Literal["real", "imag", "magnitude"]


def _select_view(coef: np.ndarray, view: KernelView) -> np.ndarray:
    if view == "real":
        return np.real(coef)
    if view == "imag":
        return np.imag(coef)
    if view == "magnitude":
        return np.abs(coef)
    raise ValueError(f"Unknown view={view!r}")


def plot_kernel_1d(
    model: TrfModel,
    *,
    feature_index: int = 0,
    output_index: int = 0,
    view: KernelView = "real",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Lag (s)",
    ylabel: str = "Kernel weight",
) -> plt.Axes:
    """Plot one TRF kernel (one feature → one output) as a function of lag.

    Usage example
    -------------
        ax = plot_kernel_1d(model, feature_index=0, output_index=3)
    """
    coef = np.asarray(model.coef)
    if coef.ndim != 3:
        raise ValueError("model.coef must be (n_lags, n_features, n_outputs).")

    _, n_features, n_outputs = coef.shape
    if not (0 <= feature_index < n_features):
        raise IndexError(f"feature_index out of range: {feature_index} (n_features={n_features})")
    if not (0 <= output_index < n_outputs):
        raise IndexError(f"output_index out of range: {output_index} (n_outputs={n_outputs})")

    y = _select_view(coef[:, feature_index, output_index], view=view)
    x = np.asarray(model.lags_s, dtype=float)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, y)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        suffix = "" if (not np.iscomplexobj(coef) or view == "real") else f" ({view})"
        title = f"TRF kernel: feature {feature_index} → output {output_index}{suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_kernels_multi_output(
    model: TrfModel,
    *,
    feature_index: int = 0,
    output_indices: Optional[Sequence[int]] = None,
    view: KernelView = "real",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Lag (s)",
    ylabel: str = "Kernel weight",
    legend: bool = True,
) -> plt.Axes:
    """Plot kernels for one feature across multiple outputs."""
    coef = np.asarray(model.coef)
    _, n_features, n_outputs = coef.shape
    if not (0 <= feature_index < n_features):
        raise IndexError(f"feature_index out of range: {feature_index} (n_features={n_features})")

    if output_indices is None:
        output_indices = list(range(n_outputs))

    x = np.asarray(model.lags_s, dtype=float)

    if ax is None:
        _, ax = plt.subplots()

    for out_idx in output_indices:
        out_idx = int(out_idx)
        if not (0 <= out_idx < n_outputs):
            raise IndexError(f"output_index out of range: {out_idx} (n_outputs={n_outputs})")
        y = _select_view(coef[:, feature_index, out_idx], view=view)
        ax.plot(x, y, label=str(out_idx))

    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        suffix = "" if (not np.iscomplexobj(coef) or view == "real") else f" ({view})"
        title = f"TRF kernels: feature {feature_index} → outputs{suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(title="Output", ncols=2, fontsize="small")

    return ax
