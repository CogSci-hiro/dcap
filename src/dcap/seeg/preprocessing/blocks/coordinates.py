# =============================================================================
#                     ########################################
#                     #   BLOCK 2: COORDINATES / GEOMETRY    #
#                     ########################################
# =============================================================================
#
# Attach electrode coordinates to an MNE Raw and enrich preprocessing context
# with geometry (coords + optional neighbor graph).
#
# Logic only:
# - No file I/O
# - No CLI / printing
#
# =============================================================================

from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import CoordinatesConfig
from dcap.seeg.preprocessing.types import BlockArtifact, Geometry, PreprocContext


# =============================================================================
#                              INTERNAL CONSTANTS
# =============================================================================
_SUPPORTED_UNITS: Tuple[str, ...] = ("m", "mm")


# =============================================================================
#                               HELPER FUNCTIONS
# =============================================================================
def _to_meters(coords: Sequence[float], unit: str) -> Tuple[float, float, float]:
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape != (3,):
        raise ValueError(f"Expected 3D coordinates, got shape {coords_arr.shape}.")

    if unit == "m":
        coords_m = coords_arr
    elif unit == "mm":
        coords_m = coords_arr / 1000.0
    else:
        raise ValueError(f"Unsupported unit {unit!r}. Supported: {_SUPPORTED_UNITS}.")

    return float(coords_m[0]), float(coords_m[1]), float(coords_m[2])


def _compute_neighbors_knn(
    coords_m: Mapping[str, Tuple[float, float, float]],
    k: int,
    radius_m: Optional[float],
) -> Dict[str, List[str]]:
    names = list(coords_m.keys())
    xyz = np.asarray([coords_m[name] for name in names], dtype=float)  # (n, 3)

    if xyz.shape[0] == 0:
        return {}

    # Pairwise distances
    diffs = xyz[:, None, :] - xyz[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))  # (n, n)
    np.fill_diagonal(dists, np.inf)

    neighbors: Dict[str, List[str]] = {}
    for i, name in enumerate(names):
        order = np.argsort(dists[i])

        chosen: List[str] = []
        for j in order:
            if len(chosen) >= k:
                break
            if radius_m is not None and dists[i, j] > radius_m:
                break
            chosen.append(names[int(j)])

        neighbors[name] = chosen

    return neighbors


# =============================================================================
#                               PUBLIC BLOCK API
# =============================================================================
def attach_coordinates(
    raw: mne.io.BaseRaw,
    electrodes_table: Mapping[str, Sequence[float]],
    cfg: CoordinatesConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Attach electrode coordinates to channels and enrich context geometry.

    Parameters
    ----------
    raw
        MNE Raw object.
    electrodes_table
        Mapping from channel name to (x, y, z) coordinates in unit `cfg.unit`.
    cfg
        Coordinates configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        Raw object with montage attached (copy).
    artifact
        Block artifact with summary + missing list.

    Notes
    -----
    - This attaches a DigMontage using `mne.channels.make_dig_montage`.
    - Coordinates are stored in `ctx.geometry.coords_m` in meters.
    - If `cfg.compute_neighbors` is True, a kNN neighbor graph is stored in
      `ctx.geometry.neighbors`.

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = attach_coordinates(
            raw=raw,
            electrodes_table={"A1": (12.3, -4.5, 6.7)},
            cfg=CoordinatesConfig(unit="mm", compute_neighbors=True, neighbors_k=6),
            ctx=ctx,
        )
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("attach_coordinates expects an mne.io.BaseRaw.")

    if cfg.unit not in _SUPPORTED_UNITS:
        raise ValueError(f"Unsupported unit {cfg.unit!r}. Supported: {_SUPPORTED_UNITS}.")

    if cfg.compute_neighbors and cfg.neighbors_k <= 0:
        raise ValueError(f"neighbors_k must be > 0 when compute_neighbors=True, got {cfg.neighbors_k}.")

    ctx.add_record("coordinates", asdict(cfg))

    # Convert all provided coords to meters
    coords_m_all: Dict[str, Tuple[float, float, float]] = {
        name: _to_meters(xyz, cfg.unit) for name, xyz in electrodes_table.items()
    }

    # Keep only channels that exist in raw
    raw_ch_set = set(raw.ch_names)
    coords_m_in_raw = {name: xyz for name, xyz in coords_m_all.items() if name in raw_ch_set}

    missing_in_raw = sorted([name for name in coords_m_all.keys() if name not in raw_ch_set])
    missing_in_table = sorted([name for name in raw.ch_names if name not in coords_m_all])

    # Attach montage to a copy (avoid surprises)
    raw_out = raw.copy()

    montage = mne.channels.make_dig_montage(
        ch_pos=coords_m_in_raw,
        coord_frame="unknown",  # we store the human label in cfg.coord_frame instead
    )
    raw_out.set_montage(montage, on_missing="ignore", verbose=False)

    # Enrich context geometry
    neighbors: Dict[str, List[str]] = {}
    if cfg.compute_neighbors:
        radius_m: Optional[float] = None
        if cfg.neighbors_radius_mm is not None:
            radius_m = float(cfg.neighbors_radius_mm) / 1000.0
        neighbors = _compute_neighbors_knn(
            coords_m=coords_m_in_raw,
            k=int(cfg.neighbors_k),
            radius_m=radius_m,
        )

    ctx.geometry = Geometry(
        coords_m=coords_m_in_raw,
        neighbors=neighbors,
        shafts={},  # can be filled later when you integrate shaft/contact metadata
    )

    warnings: List[str] = []
    if missing_in_raw:
        warnings.append(f"{len(missing_in_raw)} coord entries not found in raw channels (ignored).")
    if missing_in_table:
        warnings.append(f"{len(missing_in_table)} raw channels had no coordinates in table.")

    artifact = BlockArtifact(
        name="coordinates",
        parameters=asdict(cfg),
        summary_metrics={
            "n_coords_total": len(coords_m_all),
            "n_coords_attached": len(coords_m_in_raw),
            "n_missing_in_raw": len(missing_in_raw),
            "n_missing_in_table": len(missing_in_table),
            "neighbors_computed": bool(cfg.compute_neighbors),
        },
        warnings=warnings,
        figures=[],
    )

    # Optional: store coord frame label for downstream reporting (logic-only)
    ctx.decisions["coord_frame_label"] = cfg.coord_frame

    return raw_out, artifact
