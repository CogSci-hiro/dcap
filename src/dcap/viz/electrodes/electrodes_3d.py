# =============================================================================
#                     ########################################
#                     #        ELECTRODE 3D LOCALIZATION      #
#                     ########################################
# =============================================================================
"""Static 3D electrode localization plot with fsaverage underlay (PyVistaQt screenshots)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from dcap.viz.style import DEFAULT_STYLE, StyleConfig

from .validate import validate_and_clean_electrodes_df
from .views import VIEWS_2X2, MNEViewSpec


# =============================================================================
#                     ########################################
#                     #               CONSTANTS               #
#                     ########################################
# =============================================================================
DEFAULT_SURFACE = "white"
FOCAL_POINT = "auto"

MM_TO_M = 1.0 / 1000.0

COLORBAR_RECT = (0.15, 0.05, 0.7, 0.02)  # (left, bottom, width, height)
COLORBAR_ORIENTATION = "horizontal"

DEFAULT_ALPHA = 1.0
DEFAULT_CMAP = "inferno"

VTK_WINDOW_SIZE_PX = (800, 800)


# =============================================================================
#                     ########################################
#                     #              PUBLIC API               #
#                     ########################################
# =============================================================================
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
    cmap: str = DEFAULT_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    marker: str = "o",  # kept for API compat; in 3D we render spheres
    base_size: float = 200.0,  # interpreted as "visual weight", mapped to point_size
    highlight_size: float = 280.0,
    annotate: bool = False,  # not implemented in 3D screenshot version (hook kept)
    style: StyleConfig = DEFAULT_STYLE,
) -> None:
    """
    Plot 3D electrode locations on an fsaverage brain outline and save a 2×2 PNG.

    Parameters
    ----------
    electrodes_df
        Must include: name, x, y, z (mm). Optionally include values_col.
    out_path
        PNG output path.
    coords_space
        Label for title/metadata only.
    title
        Optional title; if None uses a default.
    highlight
        Optional list of electrode names to emphasize.
    values_col
        Optional numeric column to color electrodes by.
    """
    import mne  # noqa: WPS433
    import pyvista as pv  # noqa: WPS433

    # Must be set before any plotter is created
    os.environ["PYVISTA_OFF_SCREEN"] = "true"

    # Your environment uses pyvistaqt; that's okay as long as we never enter the interactor loop.
    mne.viz.set_3d_backend("pyvistaqt")
    pv.OFF_SCREEN = True

    cleaned_df = validate_and_clean_electrodes_df(electrodes_df, values_col=values_col)
    if cleaned_df.empty:
        raise ValueError("electrodes_df is empty after cleaning.")

    names = cleaned_df["name"].astype(str).to_numpy()
    xyz_m = cleaned_df[["x", "y", "z"]].to_numpy(dtype=float) * MM_TO_M

    highlight_set = set(highlight or [])

    coords_label = coords_space or (cleaned_df["space"].iloc[0] if "space" in cleaned_df.columns else None)
    plot_title = title or _default_title(coords_label=coords_label)

    effective_dpi = int(dpi) if dpi is not None else int(style.dpi)

    # Values for coloring
    values, norm, scalar_mappable = _prepare_values(
        cleaned_df=cleaned_df,
        values_col=values_col,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    montage = _make_mne_montage(ch_names=names, xyz_m=xyz_m)

    info = mne.create_info(
        ch_names=list(names),
        sfreq=1000.0,
        ch_types=["seeg"] * len(names),
    )
    info.set_montage(montage)

    subjects_dir = _get_fsaverage_subjects_dir(mne)

    brain_fig = mne.viz.plot_alignment(
        info,
        trans="fsaverage",
        subject="fsaverage",
        coord_frame="mri",
        subjects_dir=subjects_dir,
        surfaces=DEFAULT_SURFACE,
    )

    plotter = brain_fig.plotter
    _harden_plotter(plotter)

    # Add electrode actors ONCE (3D), then screenshot from each view
    _add_electrodes_to_scene(
        pv=pv,
        plotter=plotter,
        names=names,
        xyz_m=xyz_m,
        values=values,
        cmap=cmap,
        norm=norm,
        highlight_set=highlight_set,
        base_size=base_size,
        highlight_size=highlight_size,
    )

    mpl.rcParams.update(
        {
            "figure.dpi": effective_dpi,
            "savefig.dpi": effective_dpi,
            "font.size": style.font_size,
        }
    )
    fig2, axes = plt.subplots(2, 2, figsize=figsize)
    fig2.patch.set_facecolor("white")
    fig2.suptitle(plot_title, fontsize=max(12, style.font_size + 2))

    try:
        for i, view in enumerate(VIEWS_2X2):
            ax = axes[i // 2, i % 2]

            mne.viz.set_3d_view(
                figure=brain_fig,
                azimuth=view.azimuth,
                elevation=view.elevation,
                roll=view.roll,
                distance="auto",
                focalpoint=FOCAL_POINT,
            )

            plotter.render()
            img = plotter.screenshot(return_img=True, window_size=VTK_WINDOW_SIZE_PX)

            ax.imshow(img)
            ax.set_axis_off()

        if scalar_mappable is not None and values_col is not None:
            cax = fig2.add_axes(COLORBAR_RECT)  # noqa: WPS437
            cbar = fig2.colorbar(scalar_mappable, cax=cax, orientation=COLORBAR_ORIENTATION)
            cbar.ax.tick_params(labelsize=max(8, style.font_size))
            cax.set_title(values_col, fontsize=max(9, style.font_size))

        fig2.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out_path, dpi=effective_dpi, facecolor="white")
    finally:
        plt.close(fig2)
        mne.viz.close_3d_figure(brain_fig)


# =============================================================================
#                     ########################################
#                     #              HELPERS                  #
#                     ########################################
# =============================================================================
def _default_title(*, coords_label: Optional[str]) -> str:
    if coords_label:
        return f"Electrode localization ({coords_label})"
    return "Electrode localization"


def _get_fsaverage_subjects_dir(mne_module) -> Path:
    fetch = getattr(mne_module.datasets, "fetch_fsaverage", None)
    if callable(fetch):
        fs_dir = Path(fetch(verbose=False))
        return fs_dir.parent
    sample_path = Path(mne_module.datasets.sample.data_path())
    return sample_path / "subjects"


def _make_mne_montage(*, ch_names: np.ndarray, xyz_m: np.ndarray):
    import mne  # noqa: WPS433
    ch_pos = {str(name): xyz_m[i, :] for i, name in enumerate(ch_names)}
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="mri")


def _harden_plotter(plotter) -> None:
    # Your proven anti-hang levers:
    try:
        plotter.iren = None
    except Exception:
        pass

    try:
        plotter.window_size = VTK_WINDOW_SIZE_PX
    except Exception:
        pass


def _add_electrodes_to_scene(
    *,
    pv,
    plotter,
    names: np.ndarray,
    xyz_m: np.ndarray,
    values: Optional[np.ndarray],
    cmap: str,
    norm,
    highlight_set: set[str],
    base_size: float,
    highlight_size: float,
) -> None:
    # Convert your "sizes" into PyVista point_size (roughly)
    point_size = float(np.clip(base_size / 25.0, 4.0, 18.0))
    highlight_point_size = float(np.clip(highlight_size / 20.0, point_size + 2.0, 26.0))

    points = pv.PolyData(xyz_m)

    if values is None:
        plotter.add_mesh(
            points,
            color=(0.35, 0.35, 0.35),
            render_points_as_spheres=True,
            point_size=point_size,
        )
    else:
        v = values.astype(float)
        finite = np.isfinite(v)

        vmin = float(norm.vmin)  # type: ignore[union-attr]
        vmax = float(norm.vmax)  # type: ignore[union-attr]
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0

        v01 = (v - vmin) / denom
        v01[~finite] = 0.0

        points["values01"] = v01
        plotter.add_mesh(
            points,
            scalars="values01",
            cmap=cmap,
            clim=(0.0, 1.0),
            render_points_as_spheres=True,
            point_size=point_size,
        )

    if highlight_set:
        mask = np.array([str(n) in highlight_set for n in names], dtype=bool)
        if np.any(mask):
            hi_points = pv.PolyData(xyz_m[mask])
            plotter.add_mesh(
                hi_points,
                color=(0.0, 0.0, 0.0),
                render_points_as_spheres=True,
                point_size=highlight_point_size,
                opacity=1.0,
            )


def _prepare_values(
    *,
    cleaned_df: pd.DataFrame,
    values_col: Optional[str],
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[mpl.colors.Normalize], Optional[mpl.cm.ScalarMappable]]:
    if values_col is None:
        return None, None, None

    values = pd.to_numeric(cleaned_df[values_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, None, None

    vvmin = float(np.nanmin(values[finite])) if vmin is None else float(vmin)
    vvmax = float(np.nanmax(values[finite])) if vmax is None else float(vmax)

    norm = mpl.colors.Normalize(vmin=vvmin, vmax=vvmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap(cmap))
    sm.set_array(values[finite])
    return values, norm, sm
