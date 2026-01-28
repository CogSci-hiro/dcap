# =============================================================================
#                           Electrode plotting utilities
# =============================================================================
# > Library code for plotting sEEG electrode montages on fsaverage (MNE).
# > This file contains reusable functions and helpers only (no CLI logic).

from __future__ import annotations  # remove if your project style forbids it

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mat73
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.electrodes import get_mni_mono_coordinates, get_montage

# =============================================================================
#                                   Constants
# =============================================================================

FOCAL_POINT: str | np.ndarray = "auto"
FSAVERAGE_SUBJECT: str = "fsaverage"
DEFAULT_SURFACE: str = "pial"

UNKNOWN_ROI_NAME: str = "none"
ATLAS_UNKNOWN_LABEL: str = "Unknown"

HEMI_LEFT: str = "lh"
HEMI_RIGHT: str = "rh"
HEMI_BOTH: str = "both"


@dataclass(frozen=True)
class ViewSpec:
    """Camera view for MNE 3D figures."""
    azimuth: float
    elevation: float
    roll: float


TOP_VIEW = ViewSpec(azimuth=0, elevation=0, roll=0)
FRONT_VIEW = ViewSpec(azimuth=90, elevation=90, roll=2)  # roll=0 can look tilted
RIGHT_VIEW = ViewSpec(azimuth=0, elevation=90, roll=-90)
LEFT_VIEW = ViewSpec(azimuth=0, elevation=-90, roll=90)

DEFAULT_VIEWS: tuple[ViewSpec, ...] = (TOP_VIEW, FRONT_VIEW, RIGHT_VIEW, LEFT_VIEW)


# =============================================================================
#                               Public API helpers
# =============================================================================

def load_atlas_mat(atlas_file: str | Path) -> dict[str, Any]:
    """
    Load an electrode-to-atlas `.mat` file (mat73).

    Parameters
    ----------
    atlas_file
        Path to an atlas `.mat` file, typically `elec2atlas.mat`.

    Returns
    -------
    atlas
        Parsed MATLAB structure as nested Python dicts/lists.
    """
    return mat73.loadmat(str(atlas_file))


def load_raw(raw_file: str | Path, preload: bool = False) -> mne.io.BaseRaw:
    """
    Load an MNE Raw file.

    Parameters
    ----------
    raw_file
        Path to a raw file readable by MNE (e.g., EDF).
    preload
        Whether to preload into memory.

    Returns
    -------
    raw
        MNE Raw object.
    """
    return mne.io.read_raw(str(raw_file), preload=preload, verbose="ERROR")


def build_montage(
    raw: mne.io.BaseRaw,
    atlas: Mapping[str, Any],
    montage_type: str,
    *,
    exclude_unknown: bool = True,
) -> mne.channels.DigMontage:
    """
    Build a DigMontage from a raw file and atlas.

    Parameters
    ----------
    raw
        MNE raw object.
    atlas
        Atlas dict loaded from `mat73.loadmat`.
    montage_type
        Montage type string expected by your `get_montage` (e.g., "monopolar", "bipolar").
    exclude_unknown
        Whether to drop unknown electrodes when computing MNI coordinates.

    Returns
    -------
    montage
        MNE DigMontage with channel names and dig points.
    """
    mni_coordinates = get_mni_mono_coordinates(atlas, exclude_unknown=exclude_unknown)
    montage, _, _ = get_montage(raw, mni_coordinates, montage_type)
    return montage


def prepare_fsaverage_alignment_figure(
    raw_info: mne.Info,
    *,
    surface: str = DEFAULT_SURFACE,
    subjects_dir: Path | None = None,
) -> Any:
    """
    Create an MNE 3D alignment figure using fsaverage.

    Parameters
    ----------
    raw_info
        Raw info object (for plotting sensors).
    surface
        Freesurfer surface to plot (e.g., "pial", "inflated").
    subjects_dir
        Freesurfer subjects directory. If None, uses MNE sample dataset path.

    Returns
    -------
    fig_3d
        MNE 3D figure handle.
    """
    if subjects_dir is None:
        subjects_dir = mne.datasets.sample.data_path() / "subjects"

    fig_3d = mne.viz.plot_alignment(
        raw_info,
        trans=FSAVERAGE_SUBJECT,
        subject=FSAVERAGE_SUBJECT,
        coord_frame="mri",
        subjects_dir=subjects_dir,
        surfaces=surface,
    )
    return fig_3d


# =============================================================================
#                               Public plotting API
# =============================================================================

def plot_electrodes_2d(
    raw_file: str | Path,
    atlas_file: str | Path,
    out_file: str | Path | None,
    *,
    montage_type: str,
    show_names: bool = False,
) -> None:
    """
    Plot electrode locations using MNE's 2D montage plot.

    Parameters
    ----------
    raw_file
        Path to raw data (e.g., EDF).
    atlas_file
        Path to `elec2atlas.mat`.
    out_file
        If provided, save the figure to this path; if None, just show.
    montage_type
        Montage type (e.g., "monopolar", "bipolar").
    show_names
        Whether to show channel names on the plot.

    Returns
    -------
    None
    """
    atlas = load_atlas_mat(atlas_file)
    raw = load_raw(raw_file, preload=False)
    montage = build_montage(raw, atlas, montage_type, exclude_unknown=True)

    fig = montage.plot(show_names=show_names)
    if out_file is not None:
        fig.savefig(str(out_file), bbox_inches="tight", dpi=200)
    else:
        plt.show()


def plot_electrodes_3d_snapshots(
    raw_file: str | Path,
    atlas_file: str | Path,
    out_file: str | Path,
    *,
    montage_type: str,
    exclude_unknown: bool = True,
    surface: str = DEFAULT_SURFACE,
    color: str = "C0",
    size: float = 200.0,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (20.0, 20.0),
) -> None:
    """
    Plot 3D electrode locations on fsaverage and export 4 snapshot views (top/front/right/left).

    Notes
    -----
    Right/left hemisphere split uses your original convention:
    - channels containing an apostrophe are treated as left hemisphere.

    Parameters
    ----------
    raw_file
        Path to raw file.
    atlas_file
        Path to atlas `.mat`.
    out_file
        Output image file path (e.g., PNG).
    montage_type
        Montage type (e.g., "monopolar", "bipolar").
    exclude_unknown
        Whether to remove unknown channels at montage build time.
    surface
        Freesurfer surface.
    color
        Scatter color (matplotlib color spec).
    size
        Marker size.
    alpha
        Marker alpha.
    figsize
        Figure size.

    Returns
    -------
    None
    """
    atlas = load_atlas_mat(atlas_file)
    raw = load_raw(raw_file, preload=False)
    montage = build_montage(raw, atlas, montage_type, exclude_unknown=exclude_unknown)

    fig_3d = prepare_fsaverage_alignment_figure(raw.info, surface=surface)

    fig_2d, axes = plt.subplots(2, 2, figsize=figsize)
    fig_2d.suptitle("Electrode locations", fontsize=30)

    for view_index, view in enumerate(DEFAULT_VIEWS):
        _set_mne_3d_view(fig_3d, view)

        if view_index == 2:
            selection = _select_by_hemi_apostrophe(montage.copy(), hemi=HEMI_RIGHT)
        elif view_index == 3:
            selection = _select_by_hemi_apostrophe(montage.copy(), hemi=HEMI_LEFT)
        else:
            selection = montage.copy()

        xy, image = mne.viz.snapshot_brain_montage(fig_3d, selection, hide_sensors=False)
        row, col = divmod(view_index, 2)

        axes[row, col].imshow(image)
        xy_points = np.vstack([xy[channel] for channel in selection.ch_names])

        axes[row, col].scatter(
            *xy_points.T,
            c=color,
            s=size,
            alpha=alpha,
            marker="o",
            edgecolor="k",
        )
        axes[row, col].set_axis_off()

    fig_2d.tight_layout()
    fig_2d.savefig(str(out_file), bbox_inches="tight", dpi=200)
    plt.close(fig_2d)


def plot_subject_scores_3d(
    raw_file: str | Path,
    atlas_file: str | Path,
    score_file: str | Path,
    out_file: str | Path,
    *,
    montage_type: str,
    title: str,
    score_column: str = "scores",
    channel_column: str = "channel",
    threshold: float = 0.001,
    exclude_unknown: bool = False,
    surface: str = DEFAULT_SURFACE,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (20.0, 20.0),
    cmap: str = "inferno",
) -> None:
    """
    Plot per-channel scores on fsaverage with 4 snapshot views.

    Expected score CSV format
    -------------------------
    The file should have at least these columns:

    | channel | scores |
    |--------:|-------:|
    | LA1     | 0.12   |
    | LA2'    | 0.05   |

    Parameters
    ----------
    raw_file
        Path to raw file.
    atlas_file
        Path to atlas `.mat`.
    score_file
        Path to CSV with per-channel scores.
    out_file
        Output image path.
    montage_type
        Montage type (e.g., "bipolar").
    title
        Figure title.
    score_column
        Name of the score column in `score_file`.
    channel_column
        Name of the channel column in `score_file`.
    threshold
        Values <= threshold are dropped.
    exclude_unknown
        Exclude unknown channels when building montage.
    surface
        Freesurfer surface.
    alpha
        Marker alpha.
    figsize
        Figure size.
    cmap
        Matplotlib colormap name.

    Returns
    -------
    None
    """
    fig_3d, montage, _atlas = _prepare_figure(
        atlas_file=atlas_file,
        raw_file=raw_file,
        exclude_unknown=exclude_unknown,
        montage_type=montage_type,
        surface=surface,
    )

    scores_df = pd.read_csv(score_file)
    if channel_column not in scores_df.columns or score_column not in scores_df.columns:
        raise ValueError(
            f"score_file must contain columns {channel_column!r} and {score_column!r}. "
            f"Found: {list(scores_df.columns)}"
        )

    scores_df = scores_df.loc[scores_df[score_column] > threshold, [channel_column, score_column]].copy()

    fig_2d, axes = plt.subplots(2, 2, figsize=figsize)
    fig_2d.suptitle(title, fontsize=30)

    _plot_values_on_views(
        fig_3d=fig_3d,
        axes=axes,
        montage=montage,
        scores_df=scores_df,
        channel_column=channel_column,
        score_column=score_column,
        alpha=alpha,
        cmap=cmap,
    )

    _add_horizontal_colorbar(
        fig=fig_2d,
        vmin=float(scores_df[score_column].min()),
        vmax=float(scores_df[score_column].max()),
        cmap=cmap,
        rect=(0.15, 0.05, 0.7, 0.02),
        label="Scores",
    )

    fig_2d.savefig(str(out_file), bbox_inches="tight", dpi=200)
    plt.close(fig_2d)


def plot_subject_electrodes_by_roi_3d(
    raw_file: str | Path,
    atlas_file: str | Path,
    out_file: str | Path,
    *,
    roi_regex_to_label: Mapping[str, str],
    montage_type: str,
    exclude_unknown: bool = True,
    surface: str = DEFAULT_SURFACE,
    size: float = 200.0,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (20.0, 20.0),
) -> None:
    """
    Plot electrode locations colored by ROI group (regex on atlas ROI name).

    Parameters
    ----------
    raw_file
        Path to raw file.
    atlas_file
        Path to atlas `.mat`.
    out_file
        Output image path.
    roi_regex_to_label
        Mapping from regex (matched against ROI name) to legend label.
        Example:
            {
                r".*Superior temporal.*": "STG",
                r".*Inferior frontal.*": "IFG",
            }
    montage_type
        Montage type.
    exclude_unknown
        Exclude unknown at montage build.
    surface
        Freesurfer surface.
    size
        Marker size.
    alpha
        Marker alpha.
    figsize
        Figure size.

    Returns
    -------
    None
    """
    fig_3d, montage, atlas = _prepare_figure(
        atlas_file=atlas_file,
        raw_file=raw_file,
        exclude_unknown=exclude_unknown,
        montage_type=montage_type,
        surface=surface,
    )

    fig_2d, axes = plt.subplots(2, 2, figsize=figsize)
    fig_2d.suptitle("Electrode locations", fontsize=30)

    colors = _make_distinct_colors(len(roi_regex_to_label))

    for (regex, roi_label), color in zip(roi_regex_to_label.items(), colors, strict=True):
        _plot_roi(
            fig_3d=fig_3d,
            axes=axes,
            atlas=atlas,
            montage=montage,
            regex=regex,
            roi_label=roi_label,
            color=color,
            size=size,
            alpha=alpha,
        )

    _deduplicate_legend()
    fig_2d.tight_layout()
    fig_2d.savefig(str(out_file), bbox_inches="tight", dpi=200)
    plt.close(fig_2d)


# =============================================================================
#                           Internal helpers (private)
# =============================================================================

def _prepare_figure(
    *,
    atlas_file: str | Path,
    raw_file: str | Path,
    exclude_unknown: bool,
    montage_type: str,
    surface: str,
) -> tuple[Any, mne.channels.DigMontage, dict[str, Any]]:
    atlas = load_atlas_mat(atlas_file)
    raw = load_raw(raw_file, preload=False)
    montage = build_montage(raw, atlas, montage_type, exclude_unknown=exclude_unknown)
    fig_3d = prepare_fsaverage_alignment_figure(raw.info, surface=surface)
    return fig_3d, montage, atlas


def _set_mne_3d_view(fig_3d: Any, view: ViewSpec) -> None:
    mne.viz.set_3d_view(
        figure=fig_3d,
        azimuth=view.azimuth,
        elevation=view.elevation,
        roll=view.roll,
        distance="auto",
        focalpoint=FOCAL_POINT,
    )


def _select_by_hemi_apostrophe(montage: mne.channels.DigMontage, *, hemi: str) -> mne.channels.DigMontage:
    ch_names: list[str] = []
    dig: list[Any] = []

    for channel_name, dig_point in zip(montage.ch_names, montage.dig, strict=False):
        is_left = bool(re.match(r".*\'.*", channel_name))
        if hemi == HEMI_LEFT and not is_left:
            continue
        if hemi == HEMI_RIGHT and is_left:
            continue
        ch_names.append(channel_name)
        dig.append(dig_point)

    montage.ch_names = ch_names
    montage.dig = dig
    return montage


def _channel_to_roi(
    atlas: Mapping[str, Any],
    *,
    atlas_name: str = "Destrieux",
    min_prob: float = 10.0,
) -> dict[str, str]:
    channel_to_roi: dict[str, str] = {}

    channels = [ch for ch in atlas["coi"]["label"]]
    probs_list = atlas[atlas_name]["prob"]
    labels_list = atlas[atlas_name]["label"]

    for index, (prob_entry, label_entry) in enumerate(zip(probs_list, labels_list, strict=False)):
        roi_name = label_entry[0]
        roi_prob = float(prob_entry[0])

        channel_name = channels[index][0]

        if roi_prob < min_prob or roi_name == ATLAS_UNKNOWN_LABEL:
            channel_to_roi[channel_name] = UNKNOWN_ROI_NAME
        else:
            channel_to_roi[channel_name] = roi_name

    return channel_to_roi


def _select_channels_by_roi_regex(
    *,
    channel_to_roi: Mapping[str, str],
    montage: mne.channels.DigMontage,
    roi_regex: str,
    roi_label: str,
    hemi: str = HEMI_BOTH,
) -> mne.channels.DigMontage:
    selected_names: list[str] = []
    selected_dig: list[Any] = []
    counter = 0

    for channel_name, dig_point in zip(montage.ch_names, montage.dig, strict=False):
        roi_name = channel_to_roi.get(channel_name, UNKNOWN_ROI_NAME)

        if not re.match(roi_regex, roi_name):
            continue

        is_left = bool(re.match(r".*\'.*", channel_name))
        if hemi == HEMI_LEFT and not is_left:
            continue
        if hemi == HEMI_RIGHT and is_left:
            continue

        selected_names.append(f"{roi_label}-{counter}")
        selected_dig.append(dig_point)
        counter += 1

    montage.ch_names = selected_names
    montage.dig = selected_dig
    return montage


def _select_hemi_channels_from_df(
    montage: mne.channels.DigMontage,
    *,
    hemi: str,
    selected_channels: set[str],
) -> mne.channels.DigMontage:
    selected_names: list[str] = []
    selected_dig: list[Any] = []

    for channel_name, dig_point in zip(montage.ch_names, montage.dig, strict=False):
        is_left = bool(re.match(r".*\'.*", channel_name))
        if hemi == HEMI_LEFT and not is_left:
            continue
        if hemi == HEMI_RIGHT and is_left:
            continue
        if channel_name not in selected_channels:
            continue

        selected_names.append(channel_name)
        selected_dig.append(dig_point)

    montage.ch_names = selected_names
    montage.dig = selected_dig
    return montage


def _plot_roi(
    *,
    fig_3d: Any,
    axes: np.ndarray,
    atlas: Mapping[str, Any],
    montage: mne.channels.DigMontage,
    regex: str,
    roi_label: str,
    color: tuple[float, float, float],
    size: float,
    alpha: float,
) -> None:
    channel_to_roi = _channel_to_roi(atlas)

    for view_index, view in enumerate(DEFAULT_VIEWS):
        _set_mne_3d_view(fig_3d, view)

        if view_index == 2:
            hemi = HEMI_RIGHT
        elif view_index == 3:
            hemi = HEMI_LEFT
        else:
            hemi = HEMI_BOTH

        selection = _select_channels_by_roi_regex(
            channel_to_roi=channel_to_roi,
            montage=montage.copy(),
            roi_regex=regex,
            roi_label=roi_label,
            hemi=hemi,
        )

        try:
            xy, image = mne.viz.snapshot_brain_montage(fig_3d, selection, hide_sensors=False)
        except ValueError:
            continue

        row, col = divmod(view_index, 2)
        axes[row, col].imshow(image)

        xy_points = np.vstack([xy[channel] for channel in selection.ch_names])
        axes[row, col].scatter(
            *xy_points.T,
            color=color,
            s=size,
            alpha=alpha,
            marker="o",
            edgecolor="k",
            label=roi_label,
        )
        axes[row, col].set_axis_off()


def _plot_values_on_views(
    *,
    fig_3d: Any,
    axes: np.ndarray,
    montage: mne.channels.DigMontage,
    scores_df: pd.DataFrame,
    channel_column: str,
    score_column: str,
    alpha: float,
    cmap: str,
    max_size: float = 200.0,
) -> None:
    scores = scores_df[score_column].to_numpy(dtype=float)
    if scores.size == 0:
        raise ValueError("No scores left after thresholding; nothing to plot.")

    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max == score_min:
        score_max = score_min + 1e-12  # avoid divide by zero

    selected_channels = set(scores_df[channel_column].astype(str).tolist())

    for view_index, view in enumerate(DEFAULT_VIEWS):
        _set_mne_3d_view(fig_3d, view)

        if view_index == 2:
            hemi = HEMI_RIGHT
        elif view_index == 3:
            hemi = HEMI_LEFT
        else:
            hemi = HEMI_BOTH

        selection = _select_hemi_channels_from_df(
            montage=montage.copy(),
            hemi=hemi,
            selected_channels=selected_channels,
        )

        if len(selection.ch_names) == 0:
            continue

        sub_df = scores_df[scores_df[channel_column].isin(selection.ch_names)].copy()
        sub_scores = sub_df[score_column].to_numpy(dtype=float)

        norm_scores = (sub_scores - score_min) / (score_max - score_min)
        sizes = norm_scores * max_size

        xy, image = mne.viz.snapshot_brain_montage(fig_3d, selection, hide_sensors=False)
        row, col = divmod(view_index, 2)

        axes[row, col].imshow(image)
        xy_points = np.vstack([xy[channel] for channel in selection.ch_names])

        axes[row, col].scatter(
            *xy_points.T,
            c=sub_scores,
            s=sizes,
            alpha=alpha,
            marker="o",
            cmap=cmap,
            edgecolor="k",
            vmin=score_min,
            vmax=score_max,
        )
        axes[row, col].set_axis_off()


def _add_horizontal_colorbar(
    *,
    fig: plt.Figure,
    vmin: float,
    vmax: float,
    cmap: str,
    rect: tuple[float, float, float, float],
    label: str,
) -> None:
    norm = plt.Normalize(vmin, vmax)
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)
    cax.set_title(label, fontsize=14)


def _make_distinct_colors(n_colors: int) -> list[tuple[float, float, float]]:
    # Simple matplotlib-based distinct colors (no seaborn dependency).
    if n_colors <= 0:
        return []
    cmap = plt.get_cmap("tab20")
    return [tuple(cmap(i % cmap.N)[:3]) for i in range(n_colors)]


def _deduplicate_legend() -> None:
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label: dict[str, Any] = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", bbox_to_anchor=(1.3, 1.0))
