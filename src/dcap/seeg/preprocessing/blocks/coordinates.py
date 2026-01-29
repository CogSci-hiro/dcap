# =============================================================================
# =============================================================================
#                     ########################################
#                     #   BLOCK 2: COORDINATES / GEOMETRY    #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import CoordinatesConfig
from dcap.seeg.preprocessing.types import BlockArtifact, Geometry, PreprocContext


def _to_meters(coords: Sequence[float], unit: str) -> Tuple[float, float, float]:
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape != (3,):
        raise ValueError(f"Expected 3D coordinates, got shape {coords_arr.shape}.")
    if unit == "m":
        coords_m = coords_arr
    elif unit == "mm":
        coords_m = coords_arr / 1000.0
    else:
        raise ValueError(f"Unsupported unit {unit!r}.")
    return float(coords_m[0]), float(coords_m[1]), float(coords_m[2])


def _compute_neighbors_knn(
    coords_m: Mapping[str, Tuple[float, float, float]],
    k: int,
    radius_m: Optional[float],
) -> Dict[str, List[str]]:
    names = list(coords_m.keys())
    xyz = np.asarray([coords_m[name] for name in names], dtype=float)

    if xyz.size == 0:
        return {}

    diffs = xyz[:, None, :] - xyz[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
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


def attach_coordinates(
    raw: mne.io.BaseRaw,
    electrodes_table: Mapping[str, Sequence[float]],
    cfg: CoordinatesConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """Attach electrode coordinates as a montage and populate ctx.geometry."""
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("attach_coordinates expects an mne.io.BaseRaw.")

    ctx.add_record("coordinates", asdict(cfg))

    coords_m_all: Dict[str, Tuple[float, float, float]] = {
        name: _to_meters(xyz, cfg.unit) for name, xyz in electrodes_table.items()
    }
    raw_ch_set = set(raw.ch_names)
    coords_m_in_raw = {name: xyz for name, xyz in coords_m_all.items() if name in raw_ch_set}

    missing_in_raw = sorted([name for name in coords_m_all.keys() if name not in raw_ch_set])
    missing_in_table = sorted([name for name in raw.ch_names if name not in coords_m_all])

    raw_out = raw.copy()
    montage = mne.channels.make_dig_montage(ch_pos=coords_m_in_raw, coord_frame="unknown")
    raw_out.set_montage(montage, on_missing="ignore", verbose=False)

    neighbors: Dict[str, List[str]] = {}
    if cfg.compute_neighbors:
        radius_m = float(cfg.neighbors_radius_mm) / 1000.0 if cfg.neighbors_radius_mm is not None else None
        neighbors = _compute_neighbors_knn(coords_m_in_raw, k=int(cfg.neighbors_k), radius_m=radius_m)

    ctx.geometry = Geometry(coords_m=coords_m_in_raw, neighbors=neighbors, shafts={})
    ctx.decisions["coord_frame_label"] = cfg.coord_frame

    warnings: List[str] = []
    if missing_in_raw:
        warnings.append(f"{len(missing_in_raw)} coord entries not found in raw channels (ignored).")
    if missing_in_table:
        warnings.append(f"{len(missing_in_table)} raw channels had no coordinates in table.")

    artifact = BlockArtifact(
        name="coordinates",
        parameters=asdict(cfg),
        summary_metrics={
            "n_coords_total": float(len(coords_m_all)),
            "n_coords_attached": float(len(coords_m_in_raw)),
            "neighbors_computed": float(1 if cfg.compute_neighbors else 0),
        },
        warnings=warnings,
        figures=[],
    )
    return raw_out, artifact
