# =============================================================================
#                     ########################################
#                     #        ELECTRODE 3D LOCALIZATION      #
#                     ########################################
# =============================================================================
"""Static 3D electrode localization plot for clinical HTML reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from dcap.viz.style import DEFAULT_STYLE, StyleConfig

from .geometry import compute_equal_aspect_limits
from .validate import validate_and_clean_electrodes_df
from .views import VIEWS_2X2


# =============================================================================
#                     ########################################
#                     #               CONSTANTS               #
#                     ########################################
# =============================================================================
AXIS_LABELS = ("X", "Y", "Z")

# Clinical-safe default colors (muted)
DEFAULT_POINT_COLOR = "0.35"  # grayscale string
HIGHLIGHT_POINT_COLOR = "0.10"

DEFAULT_ALPHA = 0.95
DEFAULT_EDGE_COLOR = "0.1"
DEFAULT_EDGE_WIDTH = 0.6

GRID_ALPHA = 0.25
TITLE_FONT_WEIGHT = "semibold"


def plot_electrodes_3d(
    *,
    electrodes_df: pd.DataFrame,
    out_path: Path,
    coords_space: Optional[str] = None,
    title: Optional[str] = None,
    highlight: Optional[Sequence[str]] = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 150,
    values_col: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    marker: str = "o",
    base_size: float = 18.0,
    highlight_size: float = 40.0,
    annotate: bool = False,
    style: StyleConfig = DEFAULT_STYLE,
) -> None:
    """
    Plot 3D electrode locations (4 views) and save as a single PNG.

    Parameters
    ----------
    electrodes_df
        Canonical electrode table with at least columns:
        - "name": str
        - "x", "y", "z": numeric (coordinates)

        Example format:

        +------+-------+-------+------+-------+--------+
        | name | x     | y     | z    | space | score  |
        +------+-------+-------+------+-------+--------+
        | LA1  | -34.2 | -12.0 | 18.5 | MNI   | 0.12   |
        | LA2  | -33.7 | -10.9 | 16.9 | MNI   | 0.08   |
        | RA1  |  29.1 |  -8.4 | 21.2 | MNI   | 0.31   |
        +------+-------+-------+------+-------+--------+

    out_path
        Destination PNG path. Parent directories are created if needed.
    coords_space
        Optional coordinate space label (e.g., "MNI", "T1w", "patient").
        Used for axis/title labeling only.
    title
        Optional plot title. If None, a default is generated.
    highlight
        Optional list of electrode names to emphasize.
    figsize
        Matplotlib figure size in inches.
    dpi
        Output resolution (overrides style.dpi if provided).
    values_col
        Optional numeric column name to color-code electrodes (with colorbar).
    cmap
        Matplotlib colormap name for values_col.
    vmin, vmax
        Optional color scaling bounds for values_col. If None, inferred from data.
    marker
        Matplotlib marker for electrodes.
    base_size, highlight_size
        Scatter sizes for normal and highlighted electrodes.
    annotate
        If True, label points with electrode names (small font).
    style
        Global styling config (fonts). Defaults to DEFAULT_STYLE.

    Returns
    -------
    None
        Writes PNG to `out_path`.
    """
    cleaned_df = validate_and_clean_electrodes_df(electrodes_df, values_col=values_col)

    names = cleaned_df["name"].astype(str).to_numpy()
    xyz = cleaned_df[["x", "y", "z"]].to_numpy(dtype=float)

    highlight_set = set(highlight or [])
    is_highlight = np.array([name in highlight_set for name in names], dtype=bool)

    xlim, ylim, zlim = compute_equal_aspect_limits(xyz)

    effective_dpi = int(dpi) if dpi is not None else int(style.dpi)
    fig, axes = _make_figure(figsize=figsize, dpi=effective_dpi, font_size=style.font_size)

    coords_label = coords_space or (cleaned_df["space"].iloc[0] if "space" in cleaned_df.columns else None)
    plot_title = title or _default_title(coords_label=coords_label)

    fig.suptitle(plot_title, fontsize=style.font_size + 2, fontweight=TITLE_FONT_WEIGHT)

    # Optional value-based coloring
    scatter_mappable: Optional[mpl.cm.ScalarMappable] = None
    colors = None
    if values_col is not None:
        values = pd.to_numeric(cleaned_df[values_col], errors="coerce").to_numpy(dtype=float)
        finite_values_mask = np.isfinite(values)
        # If some values are missing, we'll still plot those electrodes in neutral gray.
        # (We keep geometry stable, no row dropping here.)
        value_vmin = float(np.nanmin(values[finite_values_mask])) if vmin is None else float(vmin)
        value_vmax = float(np.nanmax(values[finite_values_mask])) if vmax is None else float(vmax)

        norm = mpl.colors.Normalize(vmin=value_vmin, vmax=value_vmax)
        colormap = mpl.cm.get_cmap(cmap)
        colors = np.array([DEFAULT_POINT_COLOR] * len(values), dtype=object)
        colors[finite_values_mask] = [colormap(norm(v)) for v in values[finite_values_mask]]

        scatter_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        scatter_mappable.set_array(values[finite_values_mask])

    for view_index, view in enumerate(VIEWS_2X2):
        row = view_index // 2
        col = view_index % 2
        ax = axes[row, col]

        ax.view_init(elev=view.elev_deg, azim=view.azim_deg)
        ax.set_title(view.name, fontsize=style.font_size)

        _plot_one_axis(
            ax=ax,
            xyz=xyz,
            names=names,
            is_highlight=is_highlight,
            colors=colors,
            marker=marker,
            base_size=base_size,
            highlight_size=highlight_size,
            annotate=annotate,
            coords_label=coords_label,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            font_size=style.font_size,
        )

    if scatter_mappable is not None:
        _add_colorbar(fig, scatter_mappable, label=values_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=effective_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
#                     ########################################
#                     #             FIGURE HELPERS            #
#                     ########################################
# =============================================================================
def _make_figure(
    *,
    figsize: tuple[float, float],
    dpi: int,
    font_size: int,
) -> tuple[plt.Figure, np.ndarray]:
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": font_size,
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    return fig, axes


def _default_title(*, coords_label: Optional[str]) -> str:
    if coords_label:
        return f"Electrode localization ({coords_label})"
    return "Electrode localization"


def _plot_one_axis(
    *,
    ax: mpl.axes.Axes,
    xyz: np.ndarray,
    names: np.ndarray,
    is_highlight: np.ndarray,
    colors: Optional[np.ndarray],
    marker: str,
    base_size: float,
    highlight_size: float,
    annotate: bool,
    coords_label: Optional[str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    font_size: int,
) -> None:
    # Base scatter
    if colors is None:
        base_colors = DEFAULT_POINT_COLOR
    else:
        base_colors = colors

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=base_size,
        c=base_colors,
        marker=marker,
        alpha=DEFAULT_ALPHA,
        edgecolors=DEFAULT_EDGE_COLOR,
        linewidths=DEFAULT_EDGE_WIDTH,
    )

    # Highlight overlay (if any)
    if np.any(is_highlight):
        xyz_h = xyz[is_highlight]
        ax.scatter(
            xyz_h[:, 0],
            xyz_h[:, 1],
            xyz_h[:, 2],
            s=highlight_size,
            c=HIGHLIGHT_POINT_COLOR if colors is None else None,
            marker=marker,
            alpha=1.0,
            edgecolors=DEFAULT_EDGE_COLOR,
            linewidths=DEFAULT_EDGE_WIDTH + 0.2,
        )

    # Optional name annotations
    if annotate:
        for (x, y, z), name in zip(xyz, names):
            ax.text(x, y, z, str(name), fontsize=max(6, font_size - 2))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    xlab, ylab, zlab = AXIS_LABELS
    if coords_label:
        xlab = f"{xlab} ({coords_label})"
        ylab = f"{ylab} ({coords_label})"
        zlab = f"{zlab} ({coords_label})"

    ax.set_xlabel(xlab, fontsize=font_size)
    ax.set_ylabel(ylab, fontsize=font_size)
    ax.set_zlabel(zlab, fontsize=font_size)

    ax.grid(True, alpha=GRID_ALPHA)


def _add_colorbar(fig: plt.Figure, mappable: mpl.cm.ScalarMappable, *, label: str) -> None:
    # Horizontal colorbar at the bottom; keeps 2x2 grid tidy.
    cbar = fig.colorbar(mappable, ax=fig.axes, orientation="horizontal", fraction=0.05, pad=0.04)
    cbar.set_label(label)
