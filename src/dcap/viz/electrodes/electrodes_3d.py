# =============================================================================
#                     ########################################
#                     #        ELECTRODE 3D LOCALIZATION      #
#                     ########################################
# =============================================================================
"""Static 3D electrode localization plot with fsaverage underlay (MNE snapshot + 2D marker overlay)."""

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

COLORBAR_RECT = (0.15, 0.05, 0.7, 0.02)  # (left, bottom, width, height)
COLORBAR_ORIENTATION = "horizontal"

DEFAULT_MARKER = "o"
DEFAULT_EDGE_COLOR = "k"
DEFAULT_ALPHA = 1.0
DEFAULT_CMAP = "inferno"

VTK_WINDOW_SIZE_PX = (800, 800)


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
    marker: str = DEFAULT_MARKER,
    base_size: float = 20.0,
    highlight_size: float = 280.0,
    annotate: bool = False,
    style: StyleConfig = DEFAULT_STYLE,
) -> None:
    """
    Plot 3D electrode locations on an fsaverage brain outline and save a 2×2 PNG.

    This uses MNE's snapshot_brain_montage() (like your reference) to obtain:
    - the brain underlay image
    - pixel-space xy locations for sensors
    Then overlays 2D Matplotlib markers to match your original look.
    """
    import mne  # noqa: WPS433

    # Must be set before creating any PyVista plotter
    os.environ["PYVISTA_OFF_SCREEN"] = "true"

    # Your environment uses pyvistaqt; we keep it but prevent the GUI from taking over
    mne.viz.set_3d_backend("pyvistaqt")

    cleaned_df = validate_and_clean_electrodes_df(electrodes_df, values_col=values_col)
    if cleaned_df.empty:
        raise ValueError("electrodes_df is empty after cleaning.")

    # -------------------------------------------------------------------------
    # Units: auto-detect mm vs m to avoid double scaling
    # -------------------------------------------------------------------------
    xyz = cleaned_df[["x", "y", "z"]].to_numpy(dtype=float)
    abs_max = float(np.nanmax(np.abs(xyz)))
    xyz_m = xyz if abs_max < 1.0 else xyz * 1e-3  # meters if already < 1 m

    names = cleaned_df["name"].astype(str).to_numpy()
    highlight_set = set(highlight or [])

    coords_label = coords_space or (cleaned_df["space"].iloc[0] if "space" in cleaned_df.columns else None)
    plot_title = title or _default_title(coords_label=coords_label)

    effective_dpi = int(dpi) if dpi is not None else int(style.dpi)

    # -------------------------------------------------------------------------
    # Montage: MNI Talairach compat (matches your reference get_montage)
    # -------------------------------------------------------------------------
    ch_pos = {str(name): xyz_m[i, :] for i, name in enumerate(names)}
    montage = _make_dig_montage_mni_tal_compat(ch_pos_m=ch_pos)

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

    # Compute brain center from scene bounds (xmin,xmax,ymin,ymax,zmin,zmax)
    xmin, xmax, ymin, ymax, zmin, zmax = plotter.bounds
    brain_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0], dtype=float)

    # HARDEN plotter like your working minimal script
    plotter = brain_fig.plotter
    _harden_plotter(plotter)

    # Matplotlib figure (2x2)
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

    # Values colormap prep
    values, norm, scalar_mappable = _prepare_values(
        cleaned_df=cleaned_df,
        values_col=values_col,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    try:
        for i, view in enumerate(VIEWS_2X2):
            ax = axes[i // 2, i % 2]

            # Optional: per-view focalpoint centering using subset (old heuristic)

            # Precompute once, outside the loop:
            global_center_m, global_extent_m = _compute_scene_center_and_extent(xyz_m=xyz_m)

            # A stable camera distance. 6–10× extent works well.
            camera_distance = float(6.0 * global_extent_m)

            mask = _view_subset_mask(names=names, view_name=view.name)
            xyz_focus = xyz_m[mask] if np.any(mask) else xyz_m

            focus_center_m, _ = _compute_scene_center_and_extent(xyz_m=xyz_focus)

            mne.viz.set_3d_view(
                figure=brain_fig,
                azimuth=view.azimuth,
                elevation=view.elevation,
                roll=view.roll,
                distance=camera_distance,  # <-- key change: no "auto"
                focalpoint=brain_center,    # <-- key change
            )

            # Little too much space at the bottom
            if view.name in {"Front", "Left", "Right"}:
                _pan_camera_down(plotter, frac=-0.05)  # tweak 0.03–0.08

            keep_names = _names_for_view(all_names=names, view_name=view.name)
            sub_montage = _subset_montage_by_names(montage=montage, keep_names=keep_names)

            plotter.render()

            if len(keep_names) == 0:
                ax.set_axis_off()
                continue

            sub_montage = _subset_montage_by_names(montage=montage, keep_names=keep_names)

            sub_pos = sub_montage.get_positions().get("ch_pos", {}) or {}
            if len(sub_pos) == 0:
                # Nothing to plot in this view (e.g., left hemi has no "'" channels)
                ax.set_axis_off()
                continue

            xy, image = mne.viz.snapshot_brain_montage(
                brain_fig,
                sub_montage,
                hide_sensors=True,
            )

            ax.imshow(image)

            # Use the same keep_names order for plotting
            xy_points = np.vstack([xy[n] for n in keep_names if n in xy])
            plotted_names = [n for n in keep_names if n in xy]

            if values is None:
                ax.scatter(
                    xy_points[:, 0],
                    xy_points[:, 1],
                    c="0.35",
                    s=base_size,
                    alpha=DEFAULT_ALPHA,
                    marker=marker,
                    edgecolor=DEFAULT_EDGE_COLOR,
                    linewidths=0.8,
                )
            else:
                name_to_value = {str(n): float(v) for n, v in zip(names, values)}
                sel_values = np.array([name_to_value[n] for n in plotted_names], dtype=float)

                ax.scatter(
                    xy_points[:, 0],
                    xy_points[:, 1],
                    c=sel_values,
                    s=base_size,
                    alpha=DEFAULT_ALPHA,
                    marker=marker,
                    cmap=cmap,
                    norm=norm,
                    edgecolor=DEFAULT_EDGE_COLOR,
                    linewidths=0.8,
                )

            if highlight_set:
                hi_mask = np.array([n in highlight_set for n in plotted_names], dtype=bool)
                if np.any(hi_mask):
                    ax.scatter(
                        xy_points[hi_mask, 0],
                        xy_points[hi_mask, 1],
                        s=highlight_size,
                        facecolors="none",
                        edgecolors=DEFAULT_EDGE_COLOR,
                        linewidths=1.2,
                        alpha=1.0,
                    )

            # Annotation hook (optional, off by default)
            if annotate:
                for (xpix, ypix), n in zip(xy_points, plotted_names):
                    ax.text(xpix, ypix, n, fontsize=max(6, style.font_size - 2))

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
    return f"Electrode localization ({coords_label})" if coords_label else "Electrode localization"


def _get_fsaverage_subjects_dir(mne_module) -> Path:
    fetch = getattr(mne_module.datasets, "fetch_fsaverage", None)
    if callable(fetch):
        fs_dir = Path(fetch(verbose=False))
        return fs_dir.parent
    sample_path = Path(mne_module.datasets.sample.data_path())
    return sample_path / "subjects"


def _harden_plotter(plotter) -> None:
    # Your proven anti-hang levers
    try:
        plotter.iren = None
    except Exception:
        pass
    try:
        plotter.window_size = VTK_WINDOW_SIZE_PX
    except Exception:
        pass


def _select_xyz_for_view(*, names: np.ndarray, xyz_m: np.ndarray, view_name: str) -> np.ndarray:
    name_list = [str(n) for n in names]
    if view_name == "Right":
        mask = np.array(["'" not in n for n in name_list], dtype=bool)
        return xyz_m[mask] if np.any(mask) else xyz_m
    if view_name == "Left":
        mask = np.array(["'" in n for n in name_list], dtype=bool)
        return xyz_m[mask] if np.any(mask) else xyz_m
    return xyz_m


def _make_dig_montage_mni_tal_compat(*, ch_pos_m: dict[str, np.ndarray]):
    import mne

    # Try the nice path
    try:
        return mne.channels.make_dig_montage(ch_pos=ch_pos_m, coord_frame="mni_tal")
    except Exception:
        pass

    # Compat path for MNE builds that reject the string
    from mne._fiff.constants import FIFF
    from mne._fiff._digitization import DigPoint
    from mne.channels import DigMontage

    dig = []
    for name, rr in ch_pos_m.items():
        rr = np.asarray(rr, dtype=float).reshape(3)
        dig.append(
            DigPoint(
                kind=FIFF.FIFFV_POINT_EEG,
                ident=0,
                r=rr,
                coord_frame=FIFF.FIFFV_COORD_MNI_TAL,
            )
        )
    return DigMontage(dig=dig, ch_names=list(ch_pos_m.keys()))


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


def _compute_scene_center_and_extent(*, xyz_m: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute a stable scene center and extent from electrode coordinates.

    Returns
    -------
    center_m : (3,) ndarray
        Mean position in meters.
    extent_m : float
        A robust size scale (max distance from center), in meters.
    """
    center = np.mean(xyz_m, axis=0)
    radii = np.linalg.norm(xyz_m - center[None, :], axis=1)
    extent = float(np.max(radii)) if xyz_m.shape[0] > 1 else 0.05  # 5 cm fallback
    extent = max(extent, 0.02)  # floor at 2 cm so we don't zoom too hard
    return center, extent


def _view_subset_mask(*, names: np.ndarray, view_name: str) -> np.ndarray:
    name_list = [str(n) for n in names]
    if view_name == "Right":
        return np.array(["'" not in n for n in name_list], dtype=bool)
    if view_name == "Left":
        return np.array(["'" in n for n in name_list], dtype=bool)
    return np.ones(len(name_list), dtype=bool)


def _pan_camera_down(plotter, *, frac: float) -> None:
    """
    Pan the camera downward in screen space.

    Parameters
    ----------
    plotter
        PyVista Plotter instance (brain_fig.plotter).
    frac
        Fraction of the scene size to pan. Typical range: 0.02–0.08.
        Positive values move content DOWN (reduce empty bottom margin).
    """
    cam = plotter.camera

    pos = np.array(cam.position, dtype=float)
    foc = np.array(cam.focal_point, dtype=float)
    up = np.array(cam.up, dtype=float)

    # Normalize the up vector
    up_norm = up / (np.linalg.norm(up) + 1e-12)

    # Use scene scale from bounds to convert frac -> world units
    xmin, xmax, ymin, ymax, zmin, zmax = plotter.bounds
    scene_scale = float(max(xmax - xmin, ymax - ymin, zmax - zmin))

    # Move camera and focal point "down" (opposite of up vector)
    delta = -frac * scene_scale * up_norm

    cam.position = tuple(pos + delta)
    cam.focal_point = tuple(foc + delta)


def _subset_montage_by_names(*, montage, keep_names: list[str]):
    """
    Subset montage in a way that snapshot_brain_montage() always understands.

    We rebuild from ch_pos (not dig points) because snapshot_brain_montage
    depends on ch_pos being present.
    """
    import mne

    pos = montage.get_positions()
    ch_pos = pos.get("ch_pos", {}) or {}

    sub_ch_pos = {name: ch_pos[name] for name in keep_names if name in ch_pos}

    return mne.channels.make_dig_montage(
        ch_pos=sub_ch_pos,
        coord_frame="mni_tal",  # accepted by your MNE (per error message)
    )



def _names_for_view(*, all_names: np.ndarray, view_name: str) -> list[str]:
    names = [str(n) for n in all_names]

    if view_name == "Right":
        # exclude apostrophe => right hemi
        return [n for n in names if "'" not in n]

    if view_name == "Left":
        # include apostrophe => left hemi
        return [n for n in names if "'" in n]

    return names  # Top / Front

