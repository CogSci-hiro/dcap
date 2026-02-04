# =============================================================================
#                     ########################################
#                     #        ELECTRODE 3D LOCALIZATION      #
#                     ########################################
# =============================================================================
"""Static 3D electrode localization plot with fsaverage underlay (MNE snapshot + 2D marker overlay)."""

import os
from pathlib import Path
import re
from typing import Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl  # noqa
from matplotlib import pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa

from dcap.viz.style import DEFAULT_STYLE, StyleConfig  # noqa

from .validate import validate_and_clean_electrodes_df  # noqa
from .views import VIEWS_2X2  # noqa


# =============================================================================
#                     ########################################
#                     #               CONSTANTS               #
#                     ########################################
# =============================================================================
DEFAULT_SURFACE = "pial"

COLORBAR_RECT = (0.15, 0.05, 0.7, 0.02)  # (left, bottom, width, height)
COLORBAR_ORIENTATION = "horizontal"

DEFAULT_MARKER = "o"
DEFAULT_FACE_COLOR = "#FFA500"
DEFAULT_EDGE_COLOR = "k"
DEFAULT_ALPHA = 1.0
DEFAULT_CMAP = "plasma"

VTK_WINDOW_SIZE_PX = (800, 800)

# Threshold behavior
ThresholdMode = Literal["ge", "gt", "le", "lt"]


_APOSTROPHE_PATTERN = re.compile(r".*'.*")


# =============================================================================
#                     ########################################
#                     #             ELECTRODES 3D             #
#                     ########################################
# =============================================================================
def plot_electrodes_3d(
    *,
    electrodes_df: "pd.DataFrame",
    out_path: "Path",
    coords_space: "Optional[str]" = None,
    title: "Optional[str]" = None,
    highlight: "Optional[Sequence[str]]" = None,
    figsize: "tuple[float, float]" = (6.0, 6.0),
    dpi: int = 150,
    color_values: "Optional[Sequence[float]]" = None,
    size_values: "Optional[Sequence[float]]" = None,
    cmap: str = DEFAULT_CMAP,
    vmin: "Optional[float]" = None,
    vmax: "Optional[float]" = None,
    size_min: "Optional[float]" = None,
    size_max: "Optional[float]" = None,
    threshold: "Optional[float]" = None,
    threshold_mode: "ThresholdMode" = "ge",
    threshold_on: "Literal['auto', 'color', 'size']" = "auto",
    marker: str = DEFAULT_MARKER,
    base_size: float = 20.0,
    highlight_size: float = 280.0,
    annotate: bool = False,
    style: "StyleConfig" = DEFAULT_STYLE,
    render_markers: bool = True,
    show_below_threshold: bool = True,
    below_threshold_size: float = 5.0,
    below_threshold_alpha: float = 0.35,
    below_threshold_color: str = "k",
) -> None:
    """
    Render a static 2x2 fsaverage underlay with 2D electrode overlay (Top/Front/Right/Left).

    This function intentionally matches the robust behavior of `visualization.py`:
    - The 3D brain underlay is created without electrodes (montage-free figure).
    - For each view, we set a camera with `mne.viz.set_3d_view(...)`.
    - We obtain a 2D snapshot plus pixel coordinates via `mne.viz.snapshot_brain_montage(...)`.
      IMPORTANT: `hide_sensors=False` is used to avoid hemi clipping edge-cases.

    Thresholding behavior
    ---------------------
    - "Kept" electrodes (meeting threshold) are plotted using your usual sizing/color mapping.
    - If `show_below_threshold=True`, below-threshold electrodes are also plotted as fixed-size
      neutral dots for context.

    Parameters
    ----------
    electrodes_df
        Electrode table with at least columns: name, x, y, z (and optionally space).
        Coordinates may be in meters or millimeters; units are inferred by magnitude.
    out_path
        Output PNG path.
    coords_space
        Label for the coordinate space (e.g., "MNI152") used for the title.
    title
        Optional plot title.
    highlight
        Names to highlight (drawn larger).
    figsize
        Matplotlib figure size.
    dpi
        Output DPI.
    color_values
        Optional per-electrode values for color mapping (aligned to electrodes_df rows).
    size_values
        Optional per-electrode values for size mapping (aligned to electrodes_df rows).
    cmap
        Colormap name for `color_values`.
    vmin, vmax
        Optional color scale bounds.
    size_min, size_max
        Optional size scale bounds (for `size_values` mapping).
    threshold
        Threshold used to select "kept" electrodes. If None, all are kept.
    threshold_mode
        Comparison mode for thresholding ("ge", "gt", "le", "lt", ... depending on your enum).
    threshold_on
        Which values are thresholded: "auto" / "color" / "size".
    marker
        Matplotlib marker.
    base_size
        Default marker size for kept electrodes when no `size_values` provided.
    highlight_size
        Size used for highlighted electrodes.
    annotate
        Whether to annotate electrodes with names.
    style
        Styling configuration.
    render_markers
        If False, render only the underlay snapshots (still computes montages internally).
    show_below_threshold
        If True and `threshold` is not None, draw below-threshold electrodes as fixed-size dots.
    below_threshold_size
        Marker size for below-threshold dots.
    below_threshold_alpha
        Alpha for below-threshold dots.
    below_threshold_color
        Color for below-threshold dots.

    Usage example
    -------------
        from pathlib import Path
        import pandas as pd

        df = pd.DataFrame(
            {
                "name": ["A1'", "A2", "A3'", "A4"],
                "x": [10, 12, -8, -10],
                "y": [20, 18, 22, 19],
                "z": [30, 28, 31, 27],
            }
        )

        plot_electrodes_3d(
            electrodes_df=df,
            out_path=Path("electrodes.png"),
            threshold=0.5,
            color_values=[0.1, 0.9, 0.2, 0.7],
            show_below_threshold=True,
        )
    """
    import os
    from pathlib import Path

    import matplotlib
    import numpy as np

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt  # noqa: WPS433

    import mne  # noqa: WPS433
    import pandas as pd  # noqa: WPS433

    # =============================================================================
    #                     ########################################
    #                     #            CONSTANTS / SETUP          #
    #                     ########################################
    # =============================================================================
    if not isinstance(out_path, Path):
        out_path = Path(out_path)

    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    mne.viz.set_3d_backend("pyvistaqt")

    # =============================================================================
    #                     ########################################
    #                     #          INPUT VALIDATION / CLEAN     #
    #                     ########################################
    # =============================================================================
    cleaned_df = validate_and_clean_electrodes_df(electrodes_df, values_col=None)
    if cleaned_df.empty:
        raise ValueError("electrodes_df is empty after cleaning.")

    xyz = cleaned_df[["x", "y", "z"]].to_numpy(dtype=float)
    abs_max = float(np.nanmax(np.abs(xyz)))
    xyz_m = xyz if abs_max < 1.0 else xyz * 1e-3

    names = cleaned_df["name"].astype(str).to_numpy()
    highlight_set = set(highlight or [])

    color_arr = _as_aligned_float_array(values=color_values, expected_len=len(cleaned_df), name="color_values")
    size_arr = _as_aligned_float_array(values=size_values, expected_len=len(cleaned_df), name="size_values")

    keep_mask = _compute_threshold_mask(
        n_electrodes=len(cleaned_df),
        color_values=color_arr,
        size_values=size_arr,
        threshold=threshold,
        threshold_mode=threshold_mode,
        threshold_on=threshold_on,
    )

    # If no threshold is set, treat everything as kept.
    if threshold is None:
        keep_mask = np.ones(len(cleaned_df), dtype=bool)

    below_mask = ~keep_mask

    if not np.any(keep_mask):
        raise ValueError("Thresholding removed all electrodes; nothing to plot.")

    # Split arrays (kept vs below-threshold)
    names_kept = names[keep_mask]
    xyz_m_kept = xyz_m[keep_mask]
    color_kept = color_arr[keep_mask] if color_arr is not None else None
    size_kept = size_arr[keep_mask] if size_arr is not None else None

    names_below = names[below_mask]
    xyz_m_below = xyz_m[below_mask]  # noqa: F841  (kept for readability/future debugging)

    kept_set = set(names_kept.tolist())
    below_set = set(names_below.tolist())

    # =============================================================================
    #                     ########################################
    #                     #      MONTAGE FOR PROJECTION (ALL)     #
    #                     ########################################
    # =============================================================================
    # IMPORTANT: montage must include ALL electrodes that may be projected in any view,
    # otherwise snapshot_brain_montage won't return xy for below-threshold points.
    ch_pos_all: dict[str, np.ndarray] = {str(names[i]): xyz_m[i, :] for i in range(len(names))}
    montage_all = _make_dig_montage_mni_tal_compat(ch_pos_m=ch_pos_all)

    # =============================================================================
    #                     ########################################
    #                     #          BRAIN UNDERLAY FIGURE        #
    #                     ########################################
    # =============================================================================
    subjects_dir = _get_fsaverage_subjects_dir(mne)
    brain_fig = _make_fsaverage_underlay_figure(
        mne=mne,
        subjects_dir=subjects_dir,
        surface=DEFAULT_SURFACE,
    )
    plotter = brain_fig.plotter
    _harden_plotter(plotter)

    # =============================================================================
    #                     ########################################
    #                     #          MATPLOTLIB FIGURE SETUP      #
    #                     ########################################
    # =============================================================================
    _configure_matplotlib(style=style, dpi=int(dpi))
    fig2, axes = plt.subplots(2, 2, figsize=figsize)
    fig2.patch.set_facecolor("white")
    fig2.suptitle(
        title or _default_title(coords_label=coords_space),
        fontsize=max(12, style.font_size + 2),
    )

    color_norm, color_scalar_mappable = _prepare_color_mapping(
        color_values=color_kept,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
    )
    sizes_px = _prepare_size_mapping(
        size_values=size_kept,
        base_size=base_size,
        size_min=size_min,
        size_max=size_max,
    )

    # =============================================================================
    #                     ########################################
    #                     #               RENDER LOOP             #
    #                     ########################################
    # =============================================================================
    try:
        for i, view in enumerate(VIEWS_2X2):
            ax = axes[i // 2, i % 2]

            # Reset -> set view -> render (order matters)
            _reset_camera_orientation(plotter)
            mne.viz.set_3d_view(
                figure=brain_fig,
                azimuth=view.azimuth,
                elevation=view.elevation,
                roll=view.roll,
                distance="auto",
                focalpoint="auto",
            )
            plotter.render()

            # Names to PROJECT for this view must be based on ALL electrodes
            view_names_all = _names_for_view(all_names=names, view_name=view.name)
            if len(view_names_all) == 0:
                ax.set_axis_off()
                continue

            # Snapshot montage for this view (projection + underlay)
            sub_montage = _subset_montage_by_names(montage=montage_all, keep_names=view_names_all)
            sub_pos = sub_montage.get_positions().get("ch_pos", {}) or {}
            if len(sub_pos) == 0:
                ax.set_axis_off()
                continue

            xy, image = mne.viz.snapshot_brain_montage(
                brain_fig,
                sub_montage,
                hide_sensors=False,  # critical: avoids hemi clipping edge-case
            )
            ax.imshow(image)

            if not render_markers:
                ax.set_axis_off()
                continue

            name_to_xy = xy  # dict[str, tuple[float, float]]

            # ---------------------------------------------------------------------
            # Below-threshold context dots (fixed size)
            # ---------------------------------------------------------------------
            if show_below_threshold and (threshold is not None) and np.any(below_mask):
                plotted_below = [n for n in view_names_all if (n in below_set) and (n in name_to_xy)]
                if plotted_below:
                    xy_below = np.vstack([name_to_xy[n] for n in plotted_below])
                    ax.scatter(
                        xy_below[:, 0],
                        xy_below[:, 1],
                        s=float(below_threshold_size),
                        c=below_threshold_color,
                        alpha=float(below_threshold_alpha),
                        marker=marker,
                        linewidths=0,
                        zorder=2,
                    )

            # ---------------------------------------------------------------------
            # Kept electrodes (value-based or static markers)
            # ---------------------------------------------------------------------
            plotted_names = [n for n in view_names_all if (n in kept_set) and (n in name_to_xy)]
            if len(plotted_names) == 0:
                ax.set_axis_off()
                continue

            xy_points = np.vstack([name_to_xy[n] for n in plotted_names])

            # Map name -> index within kept arrays (for color_kept / sizes_px)
            name_to_index = {n: idx for idx, n in enumerate(names_kept.tolist())}

            plotted_color = None
            if color_kept is not None:
                plotted_color = np.array([color_kept[name_to_index[n]] for n in plotted_names], dtype=float)

            plotted_sizes = None
            if sizes_px is not None:
                plotted_sizes = np.array([sizes_px[name_to_index[n]] for n in plotted_names], dtype=float)

            if plotted_color is None:
                _scatter_static_markers(
                    ax=ax,
                    xy_points=xy_points,
                    marker=marker,
                    sizes=plotted_sizes,
                    default_size=base_size,
                )
            else:
                _scatter_value_markers(
                    ax=ax,
                    xy_points=xy_points,
                    values=plotted_color,
                    marker=marker,
                    sizes=plotted_sizes,
                    default_size=base_size,
                    cmap=cmap,
                    norm=color_norm,
                )

            _scatter_highlights(
                ax=ax,
                xy_points=xy_points,
                plotted_names=plotted_names,
                highlight_set=highlight_set,
                highlight_size=highlight_size,
            )

            if annotate:
                _annotate_points(ax=ax, xy_points=xy_points, names=plotted_names, font_size=style.font_size)

            ax.set_axis_off()

        # =============================================================================
        #                     ########################################
        #                     #               FINALIZE                #
        #                     ########################################
        # =============================================================================
        if render_markers and (color_scalar_mappable is not None):
            _add_colorbar(
                fig=fig2,
                scalar_mappable=color_scalar_mappable,
                title="values",
                font_size=style.font_size,
            )

        fig2.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out_path, dpi=int(dpi), facecolor="white")

    finally:
        plt.close(fig2)
        mne.viz.close_3d_figure(brain_fig)



def _make_fsaverage_underlay_figure(*, mne, subjects_dir: Path, surface: str):
    """
    Create a montage-free fsaverage underlay figure.

    This is the critical separation-of-concerns:
    - brain figure & camera must NOT depend on electrodes
    - electrodes are projected via snapshot_brain_montage() only
    """
    info = mne.create_info(ch_names=["dummy"], sfreq=1000.0, ch_types=["misc"])
    fig = mne.viz.plot_alignment(
        info,
        trans="fsaverage",
        subject="fsaverage",
        coord_frame="mri",
        subjects_dir=subjects_dir,
        surfaces=surface,
    )

    return fig


# =============================================================================
#                     ########################################
#                     #              HELPERS                  #
#                     ########################################
# =============================================================================
def _configure_matplotlib(*, style: StyleConfig, dpi: int) -> None:
    """Apply global Matplotlib defaults for a deterministic, clinical-safe figure."""
    mpl.rcParams.update(
        {
            "figure.dpi": int(dpi),
            "savefig.dpi": int(dpi),
            "font.size": style.font_size,
        }
    )


def _default_title(*, coords_label: Optional[str]) -> str:
    """Default title builder."""
    return f"Electrode localization ({coords_label})" if coords_label else "Electrode localization"


def _get_fsaverage_subjects_dir(mne_module) -> Path:
    """Resolve an fsaverage subjects_dir in a version-compatible way."""
    fetch = getattr(mne_module.datasets, "fetch_fsaverage", None)
    if callable(fetch):
        fs_dir = Path(fetch(verbose=False))
        return fs_dir.parent
    sample_path = Path(mne_module.datasets.sample.data_path())
    return sample_path / "subjects"


def _harden_plotter(plotter) -> None:
    """
    Anti-hang hardening for PyVistaQt in IDE contexts.

    These are the levers you discovered empirically:
    - remove the interactor (prevents event loop / GUI takeover)
    - constrain window size (reduces memory/VRAM usage and makes screenshots stable)
    """
    try:
        plotter.iren = None
    except Exception:  # noqa
        pass
    try:
        plotter.window_size = VTK_WINDOW_SIZE_PX
    except Exception:  # noqa
        pass


def _as_aligned_float_array(*, values: Optional[Sequence[float]], expected_len: int, name: str) -> Optional[np.ndarray]:
    """
    Convert an optional sequence to a float array and validate length.

    Parameters
    ----------
    values
        Optional sequence of numeric values.
    expected_len
        Expected number of values (must match len(cleaned_df)).
    name
        Name for error messages.

    Returns
    -------
    Optional[np.ndarray]
        None if values is None, otherwise float array.
    """
    if values is None:
        return None

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence, got shape {arr.shape}.")
    if arr.shape[0] != expected_len:
        raise ValueError(f"{name} length mismatch: expected {expected_len}, got {arr.shape[0]}.")
    return arr


def _compute_threshold_mask(
    *,
    n_electrodes: int,
    color_values: Optional[np.ndarray],
    size_values: Optional[np.ndarray],
    threshold: Optional[float],
    threshold_mode: ThresholdMode,
    threshold_on: Literal["auto", "color", "size"],
) -> np.ndarray:
    """
    Build a boolean mask selecting electrodes to keep based on optional thresholding.

    If threshold is None, returns an all-True mask of length n_electrodes.
    """
    if threshold is None:
        # No thresholding => keep everything (plain electrode plotting mode).
        return np.ones(int(n_electrodes), dtype=bool)

    if threshold_on == "color":
        if color_values is None:
            raise ValueError("threshold_on='color' requires color_values.")
        ref = color_values
    elif threshold_on == "size":
        if size_values is None:
            raise ValueError("threshold_on='size' requires size_values.")
        ref = size_values
    else:
        # auto
        if color_values is not None:
            ref = color_values
        elif size_values is not None:
            ref = size_values
        else:
            raise ValueError("threshold provided, but neither color_values nor size_values were provided.")

    finite = np.isfinite(ref)
    mask = np.zeros(ref.shape[0], dtype=bool)

    thr = float(threshold)

    if threshold_mode == "ge":
        mask[finite] = ref[finite] >= thr
    elif threshold_mode == "gt":
        mask[finite] = ref[finite] > thr
    elif threshold_mode == "le":
        mask[finite] = ref[finite] <= thr
    elif threshold_mode == "lt":
        mask[finite] = ref[finite] < thr
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    return mask


def _make_dig_montage_mni_tal_compat(*, ch_pos_m: dict[str, np.ndarray]):
    """
    Create an MNI-Tal montage in a way that works across MNE builds.

    We try the public API first; if rejected, we fall back to constructing DigPoints.
    """
    import mne

    try:
        return mne.channels.make_dig_montage(ch_pos=ch_pos_m, coord_frame="mni_tal")
    except Exception:  # noqa
        pass

    from mne._fiff.constants import FIFF  # noqa
    from mne._fiff._digitization import DigPoint  # noqa
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


def _prepare_color_mapping(
    *,
    color_values: Optional[np.ndarray],
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    threshold: Optional[float],
) -> tuple[Optional[mpl.colors.Normalize], Optional[mpl.cm.ScalarMappable]]:
    """
    Build (norm, scalar_mappable) for color mapping if color_values is provided.

    Behavior
    --------
    - If vmin/vmax are explicitly provided, they win.
    - Else if threshold is provided:
        * colorbar min is set to threshold (NOT the data min)
        * colorbar max is set to max(color_values)
      This matches the clinical view where sub-threshold electrodes are not shown.
    - Else: fall back to data min/max.
    """
    if color_values is None:
        return None, None

    finite = np.isfinite(color_values)
    if not np.any(finite):
        return None, None

    data_min = float(np.nanmin(color_values[finite]))
    data_max = float(np.nanmax(color_values[finite]))

    if vmin is not None:
        vvmin = float(vmin)
    elif threshold is not None:
        vvmin = float(threshold)
    else:
        vvmin = data_min

    vvmax = float(vmax) if vmax is not None else data_max

    # Guard against degenerate normalization (vmin >= vmax)
    if not np.isfinite(vvmin) or not np.isfinite(vvmax) or vvmin >= vvmax:
        vvmin = data_min
        vvmax = data_max

    norm = mpl.colors.Normalize(vmin=vvmin, vmax=vvmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap(cmap))
    sm.set_array(color_values[finite])
    return norm, sm


def _prepare_size_mapping(
    *,
    size_values: Optional[np.ndarray],
    base_size: float,
    size_min: Optional[float],
    size_max: Optional[float],
) -> Optional[np.ndarray]:
    """
    Convert size_values to a per-point marker size array (in Matplotlib "points^2").

    Behavior:
    - If size_values is None -> return None (caller uses base_size everywhere).
    - Else: normalize size_values to [0, 1] using finite subset, then map to a
      size range:
        [size_min, size_max] if provided, else [base_size, 6*base_size].

    Non-finite size values default to base_size (safe and deterministic).
    """
    if size_values is None:
        return None

    finite = np.isfinite(size_values)
    if not np.any(finite):
        return None

    vmin = float(np.nanmin(size_values[finite]))
    vmax = float(np.nanmax(size_values[finite]))
    denom = (vmax - vmin) if vmax > vmin else 1.0

    normalized = np.zeros_like(size_values, dtype=float)
    normalized[finite] = (size_values[finite] - vmin) / denom
    normalized[~finite] = 0.0

    out_min = float(size_min) if size_min is not None else float(base_size)
    out_max = float(size_max) if size_max is not None else float(6.0 * base_size)

    sizes = out_min + normalized * (out_max - out_min)
    return sizes


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

    up_norm = up / (np.linalg.norm(up) + 1e-12)

    xmin, xmax, ymin, ymax, zmin, zmax = plotter.bounds
    scene_scale = float(max(xmax - xmin, ymax - ymin, zmax - zmin))

    delta = -frac * scene_scale * up_norm

    cam.position = tuple(pos + delta)
    cam.focal_point = tuple(foc + delta)


def _subset_montage_by_names(*, montage, keep_names: list[str]):
    """
    Subset montage in a way that snapshot_brain_montage() always understands.

    We rebuild from ch_pos (not dig points) because snapshot_brain_montage depends on
    ch_pos being present.
    """
    import mne

    pos = montage.get_positions()
    ch_pos = pos.get("ch_pos", {}) or {}
    sub_ch_pos = {name: ch_pos[name] for name in keep_names if name in ch_pos}

    return mne.channels.make_dig_montage(
        ch_pos=sub_ch_pos,
        coord_frame="mni_tal",
    )


def _names_for_view(*, all_names: Sequence[str], view_name: str) -> list[str]:
    """
    Match visualization.py montage selection:
    - top/front: both hemispheres
    - right: names without apostrophe
    - left: names with apostrophe
    """
    if view_name.lower() in {"top", "front"}:
        return list(all_names)

    if view_name.lower() in {"right", "r"}:
        return [n for n in all_names if not _APOSTROPHE_PATTERN.match(n)]

    if view_name.lower() in {"left", "l"}:
        return [n for n in all_names if _APOSTROPHE_PATTERN.match(n)]

    # fallback: don't surprise the user
    return list(all_names)


def _scatter_static_markers(
    *,
    ax,
    xy_points: np.ndarray,
    marker: str,
    sizes: Optional[np.ndarray],
    default_size: float,
) -> None:
    """
    Scatter markers with a constant face color.

    If sizes is provided, it must align with xy_points (same length). Otherwise
    we use default_size for all points.
    """
    s = sizes if sizes is not None else float(default_size)
    ax.scatter(
        xy_points[:, 0],
        xy_points[:, 1],
        s=s,
        alpha=DEFAULT_ALPHA,
        marker=marker,
        facecolors=DEFAULT_FACE_COLOR,
        edgecolors=DEFAULT_EDGE_COLOR,
        linewidths=0.5,
    )


def _scatter_value_markers(
    *,
    ax,
    xy_points: np.ndarray,
    values: np.ndarray,
    marker: str,
    sizes: Optional[np.ndarray],
    default_size: float,
    cmap: str,
    norm: Optional[mpl.colors.Normalize],
) -> None:
    """
    Scatter markers whose face colors come from a value array + colormap.

    If sizes is provided, it must align with xy_points (same length). Otherwise
    we use default_size for all points.
    """
    s = sizes if sizes is not None else float(default_size)
    ax.scatter(
        xy_points[:, 0],
        xy_points[:, 1],
        c=values,
        s=s,
        alpha=DEFAULT_ALPHA,
        marker=marker,
        cmap=cmap,
        norm=norm,
        edgecolors=DEFAULT_EDGE_COLOR,
        linewidths=0.5,
    )


def _scatter_highlights(
    *,
    ax,
    xy_points: np.ndarray,
    plotted_names: list[str],
    highlight_set: set[str],
    highlight_size: float,
) -> None:
    """Draw a ring overlay around highlighted electrodes (existing behavior)."""
    if not highlight_set:
        return

    hi_mask = np.array([n in highlight_set for n in plotted_names], dtype=bool)
    if not np.any(hi_mask):
        return

    ax.scatter(
        xy_points[hi_mask, 0],
        xy_points[hi_mask, 1],
        s=highlight_size,
        facecolors="none",
        edgecolors=DEFAULT_EDGE_COLOR,
        linewidths=1.2,
        alpha=1.0,
    )


def _annotate_points(*, ax, xy_points: np.ndarray, names: list[str], font_size: int) -> None:
    """Add small text labels next to points (optional)."""
    for (xpix, ypix), n in zip(xy_points, names):
        ax.text(xpix, ypix, n, fontsize=max(6, font_size - 2))


def _add_colorbar(*, fig, scalar_mappable, title: str, font_size: int) -> None:
    """Attach a horizontal colorbar (matching your original layout)."""
    cax = fig.add_axes(COLORBAR_RECT)
    cbar = fig.colorbar(scalar_mappable, cax=cax, orientation=COLORBAR_ORIENTATION)
    cbar.ax.tick_params(labelsize=max(8, font_size))
    cax.set_title(title, fontsize=max(9, font_size))


def _reset_camera_orientation(plotter) -> None:
    """
    Reset camera orientation to avoid unintended rotation (VTK view-up drift).

    We do this before mne.viz.set_3d_view so that azimuth/elevation/roll land
    on a consistent baseline each time.
    """
    cam = plotter.camera

    # Force a canonical "up" direction in world coordinates.
    # For fsaverage/mri frame this is typically +Z.
    try:
        cam.up = (0.0, 0.0, 1.0)
    except Exception:
        pass

    # Remove any accumulated roll.
    try:
        # VTK-style: roll is relative; setting to 0 isn't always supported directly,
        # but many wrappers expose cam.roll as a property.
        cam.roll = 0.0
    except Exception:
        pass

