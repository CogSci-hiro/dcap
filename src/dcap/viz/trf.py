# =============================================================================
# =============================================================================
#                    #########################################
#                    #   TRF KERNEL VISUALIZATION PRIMITIVES  #
#                    #########################################
# =============================================================================
# =============================================================================
#
# This module provides plotting utilities for Temporal Response Function (TRF)
# kernel arrays. These are intended to be reusable building blocks across:
# - minimal analyses
# - QC pages
# - clinical reports
#
# Conventions
# -----------
# - Kernels are expected as a 2D array shaped (n_channels, n_timesteps).
# - Time axis can be given explicitly via `times_s` or implicitly via `dt_s`.
# - Heatmap uses a diverging colormap centered at 0.
#

from typing import Optional, Sequence, Tuple, Union, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure


# =============================================================================
# =============================================================================
#                                CONSTANTS
# =============================================================================
# =============================================================================

_DEFAULT_CMAP: str = "RdBu_r"
_DEFAULT_COLORBAR_LABEL: str = "TRF kernel amplitude"
_DEFAULT_LINEWIDTH: float = 1.25
_DEFAULT_FIGURE_DPI: int = 120

# Avoid magic numbers for figure sizing; tweak as needed later.
_DEFAULT_HEATMAP_FIGSIZE_IN: Tuple[float, float] = (10.0, 6.0)
_DEFAULT_LINES_FIGSIZE_IN: Tuple[float, float] = (10.0, 1.25)  # per channel, scaled below

_VLIM_MODE = Literal["symmetric_max", "percentile"]
_LINE_LAYOUT = Literal["stacked", "overlay"]


# =============================================================================
# =============================================================================
#                                 HELPERS
# =============================================================================
# =============================================================================

def _validate_kernels_2d(kernels: np.ndarray) -> np.ndarray:
    """Validate kernels and coerce to a 2D float array."""
    if not isinstance(kernels, np.ndarray):
        raise TypeError(f"`kernels` must be a numpy array, got {type(kernels)}")

    if kernels.ndim != 2:
        raise ValueError(f"`kernels` must be 2D (n_channels, n_timesteps), got shape={kernels.shape}")

    kernels_f = np.asarray(kernels, dtype=float)
    if not np.isfinite(kernels_f).all():
        raise ValueError("`kernels` contains NaN or inf values; please clean before plotting.")

    return kernels_f


def _resolve_times_s(
    *,
    n_timesteps: int,
    times_s: Optional[np.ndarray],
    dt_s: Optional[float],
) -> np.ndarray:
    """Resolve a time axis in seconds."""
    if times_s is not None:
        times_arr = np.asarray(times_s, dtype=float)
        if times_arr.ndim != 1:
            raise ValueError(f"`times_s` must be 1D, got shape={times_arr.shape}")
        if len(times_arr) != n_timesteps:
            raise ValueError(
                f"`times_s` length must match n_timesteps={n_timesteps}, got len(times_s)={len(times_arr)}"
            )
        return times_arr

    if dt_s is None:
        # Default to sample index in seconds (1 Hz) if caller provides neither.
        dt_s = 1.0

    if dt_s <= 0:
        raise ValueError(f"`dt_s` must be > 0, got {dt_s}")

    return np.arange(n_timesteps, dtype=float) * float(dt_s)


def _resolve_channel_names(
    *,
    n_channels: int,
    channel_names: Optional[Sequence[str]],
) -> Sequence[str]:
    """Resolve channel labels."""
    if channel_names is None:
        return [f"ch{idx:03d}" for idx in range(n_channels)]

    if len(channel_names) != n_channels:
        raise ValueError(
            f"`channel_names` length must match n_channels={n_channels}, got len(channel_names)={len(channel_names)}"
        )
    return list(channel_names)


def _compute_symmetric_vlim(
    kernels: np.ndarray,
    *,
    vlim_mode: _VLIM_MODE,
    percentile: float,
) -> float:
    """Compute a symmetric vlim around 0."""
    abs_vals = np.abs(kernels)

    if vlim_mode == "symmetric_max":
        v = float(abs_vals.max())
        return v if v > 0 else 1.0

    if vlim_mode == "percentile":
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"`percentile` must be in (0, 100], got {percentile}")
        v = float(np.percentile(abs_vals, percentile))
        return v if v > 0 else 1.0

    raise ValueError(f"Unknown vlim_mode={vlim_mode!r}")


def _convert_time_units(times_s: np.ndarray, *, time_unit: Literal["s", "ms"]) -> np.ndarray:
    """Convert time axis for display."""
    if time_unit == "s":
        return times_s
    if time_unit == "ms":
        return times_s * 1_000.0
    raise ValueError(f"Unsupported time_unit={time_unit!r}")


def _format_time_label(time_unit: Literal["s", "ms"]) -> str:
    return "Time (s)" if time_unit == "s" else "Time (ms)"


# =============================================================================
# =============================================================================
#                              PUBLIC PLOTTERS
# =============================================================================
# =============================================================================

def plot_trf_kernels_heatmap(
    kernels: np.ndarray,
    *,
    times_s: Optional[np.ndarray] = None,
    dt_s: Optional[float] = None,
    time_unit: Literal["s", "ms"] = "ms",
    channel_names: Optional[Sequence[str]] = None,
    cmap: str = _DEFAULT_CMAP,
    vlim: Optional[float] = None,
    vlim_mode: _VLIM_MODE = "percentile",
    vlim_percentile: float = 99.0,
    colorbar_label: str = _DEFAULT_COLORBAR_LABEL,
    title: Optional[str] = "TRF kernels (heatmap)",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize_in: Tuple[float, float] = _DEFAULT_HEATMAP_FIGSIZE_IN,
    dpi: int = _DEFAULT_FIGURE_DPI,
) -> Tuple[Figure, Axes]:
    """
    Plot TRF kernels as a channel-by-time heatmap.

    Parameters
    ----------
    kernels:
        TRF kernel array shaped (n_channels, n_timesteps).
        Rows correspond to channels, columns to time points.
    times_s:
        Optional explicit time axis in seconds, length n_timesteps.
    dt_s:
        Optional timestep in seconds, used if `times_s` is not provided.
    time_unit:
        Display unit for the x-axis: "s" or "ms".
    channel_names:
        Optional channel label strings, length n_channels.
    cmap:
        Matplotlib colormap name for a diverging colormap (default: "RdBu_r").
    vlim:
        Optional symmetric limit for the colormap. If None, computed from data.
    vlim_mode:
        How to compute `vlim` when None:
        - "symmetric_max": uses max(abs(kernels))
        - "percentile": uses percentile of abs(kernels)
    vlim_percentile:
        Percentile used when vlim_mode="percentile".
    colorbar_label:
        Label for the colorbar.
    title:
        Optional figure title.
    fig, ax:
        Optional Matplotlib figure/axes to draw into.
    figsize_in:
        Default size used if `fig`/`ax` not provided.
    dpi:
        Figure DPI (only used when creating a new figure).

    Returns
    -------
    fig, ax:
        Matplotlib Figure and Axes objects.

    Usage example
    -------------
        import numpy as np
        from dcap.viz.trf import plot_trf_kernels_heatmap

        kernels = np.random.randn(64, 301) * 0.05
        times_s = np.linspace(-0.1, 0.2, kernels.shape[1])

        fig, ax = plot_trf_kernels_heatmap(
            kernels,
            times_s=times_s,
            channel_names=[f"E{i:02d}" for i in range(kernels.shape[0])],
            time_unit="ms",
        )
    """
    kernels_f = _validate_kernels_2d(kernels)
    n_channels, n_timesteps = kernels_f.shape

    times_axis_s = _resolve_times_s(n_timesteps=n_timesteps, times_s=times_s, dt_s=dt_s)
    times_axis = _convert_time_units(times_axis_s, time_unit=time_unit)
    channel_labels = _resolve_channel_names(n_channels=n_channels, channel_names=channel_names)

    if vlim is None:
        vlim = _compute_symmetric_vlim(kernels_f, vlim_mode=vlim_mode, percentile=vlim_percentile)

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize_in, dpi=dpi)

    # Compute imshow extent so x-axis is in time units and y-axis spans channels.
    # We place channels as integer rows; extent maps pixel centers to coordinate system.
    x0 = float(times_axis[0])
    x1 = float(times_axis[-1])
    # For regular spacing, we extend by half-step; for irregular, use a simple bounding extent.
    if n_timesteps >= 2:
        dx = float(np.median(np.diff(times_axis)))
        x_min = x0 - dx / 2.0
        x_max = x1 + dx / 2.0
    else:
        x_min, x_max = x0 - 0.5, x0 + 0.5

    y_min = -0.5
    y_max = n_channels - 0.5

    im = ax.imshow(
        kernels_f,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(x_min, x_max, y_max, y_min),  # invert y so channel 0 is top
    )

    ax.set_xlabel(_format_time_label(time_unit))
    ax.set_ylabel("Channel")

    # Show channel labels only when it won't destroy readability.
    if n_channels <= 64:
        ax.set_yticks(np.arange(n_channels, dtype=float))
        ax.set_yticklabels(channel_labels)
    else:
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    # Colorbar on the right with a meaningful label.
    cbar = fig.colorbar(im, ax=ax, location="right", pad=0.02, fraction=0.046)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    return fig, ax


def plot_trf_kernels_lines(
    kernels: np.ndarray,
    *,
    times_s: Optional[np.ndarray] = None,
    dt_s: Optional[float] = None,
    time_unit: Literal["s", "ms"] = "ms",
    channel_names: Optional[Sequence[str]] = None,
    layout: _LINE_LAYOUT = "stacked",
    sharey: bool = True,
    linewidth: float = _DEFAULT_LINEWIDTH,
    alpha: float = 0.9,
    zero_line: bool = True,
    title: Optional[str] = "TRF kernels (lines)",
    fig: Optional[Figure] = None,
    axes: Optional[Union[Axes, Sequence[Axes]]] = None,
    base_figsize_in: Tuple[float, float] = _DEFAULT_LINES_FIGSIZE_IN,
    dpi: int = _DEFAULT_FIGURE_DPI,
) -> Tuple[Figure, Sequence[Axes]]:
    """
    Plot TRF kernels as per-channel line plots versus time.

    Two layouts are supported:
    - "stacked": one subplot per channel (small multiples), shared x-axis.
    - "overlay": all channels overlaid on one axis (useful for small n_channels).

    Parameters
    ----------
    kernels:
        TRF kernel array shaped (n_channels, n_timesteps).
    times_s:
        Optional explicit time axis in seconds, length n_timesteps.
    dt_s:
        Optional timestep in seconds, used if `times_s` is not provided.
    time_unit:
        Display unit for the x-axis: "s" or "ms".
    channel_names:
        Optional channel labels, length n_channels.
    layout:
        "stacked" or "overlay".
    sharey:
        If layout="stacked", share y-axis limits across subplots.
    linewidth:
        Line width for kernel traces.
    alpha:
        Line transparency.
    zero_line:
        Whether to draw a horizontal y=0 reference line.
    title:
        Optional figure title.
    fig, axes:
        Optional existing Matplotlib figure and axes. If layout="stacked", `axes`
        should be a sequence with length n_channels.
    base_figsize_in:
        Base (width, height_per_channel) used for stacked layout when creating a new fig.
    dpi:
        Figure DPI (only used when creating a new figure).

    Returns
    -------
    fig, axes:
        Matplotlib Figure and a sequence of Axes (length 1 for overlay; n_channels for stacked).

    Usage example
    -------------
        import numpy as np
        from dcap.viz.trf import plot_trf_kernels_lines

        kernels = np.random.randn(8, 401) * 0.05
        times_s = np.linspace(-0.2, 0.2, kernels.shape[1])

        fig, axes = plot_trf_kernels_lines(
            kernels,
            times_s=times_s,
            channel_names=[f"E{i:02d}" for i in range(kernels.shape[0])],
            layout="stacked",
        )
    """
    kernels_f = _validate_kernels_2d(kernels)
    n_channels, n_timesteps = kernels_f.shape

    times_axis_s = _resolve_times_s(n_timesteps=n_timesteps, times_s=times_s, dt_s=dt_s)
    times_axis = _convert_time_units(times_axis_s, time_unit=time_unit)
    channel_labels = _resolve_channel_names(n_channels=n_channels, channel_names=channel_names)

    if layout == "overlay":
        if axes is None or fig is None:
            fig, ax = plt.subplots(figsize=(base_figsize_in[0], 4.0), dpi=dpi)
        else:
            if isinstance(axes, Sequence):
                raise ValueError("For layout='overlay', `axes` must be a single Axes.")
            ax = axes

        for ch_idx in range(n_channels):
            ax.plot(times_axis, kernels_f[ch_idx, :], linewidth=linewidth, alpha=alpha, label=channel_labels[ch_idx])

        if zero_line:
            ax.axhline(0.0, linewidth=1.0, alpha=0.6)

        ax.set_xlabel(_format_time_label(time_unit))
        ax.set_ylabel("TRF kernel amplitude")

        if title is not None:
            ax.set_title(title)

        # Legend only when it's reasonable.
        if n_channels <= 12:
            ax.legend(loc="best", fontsize="small", frameon=False)

        fig.tight_layout()
        return fig, [ax]

    if layout != "stacked":
        raise ValueError(f"Unsupported layout={layout!r}. Use 'stacked' or 'overlay'.")

    # Stacked layout: one axis per channel.
    if axes is None or fig is None:
        width_in = float(base_figsize_in[0])
        height_per_channel_in = float(base_figsize_in[1])
        height_in = max(2.5, height_per_channel_in * n_channels)
        fig, axes_arr = plt.subplots(
            nrows=n_channels,
            ncols=1,
            sharex=True,
            sharey=sharey,
            figsize=(width_in, height_in),
            dpi=dpi,
        )
        # When n_channels == 1, matplotlib returns a single Axes.
        if isinstance(axes_arr, Axes):
            axes_list: Sequence[Axes] = [axes_arr]
        else:
            axes_list = list(axes_arr)
    else:
        if isinstance(axes, Axes):
            raise ValueError("For layout='stacked', `axes` must be a sequence of Axes (length n_channels).")
        axes_list = list(axes)
        if len(axes_list) != n_channels:
            raise ValueError(f"`axes` length must match n_channels={n_channels}, got len(axes)={len(axes_list)}")

    for ch_idx, ax in enumerate(axes_list):
        ax.plot(times_axis, kernels_f[ch_idx, :], linewidth=linewidth, alpha=alpha)
        if zero_line:
            ax.axhline(0.0, linewidth=1.0, alpha=0.6)

        ax.set_ylabel(channel_labels[ch_idx], rotation=0, labelpad=28, va="center")

    axes_list[-1].set_xlabel(_format_time_label(time_unit))

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes_list
