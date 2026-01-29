# =============================================================================
#                     ########################################
#                     #   BLOCK 2: COORDINATES / GEOMETRY    #
#                     ########################################
# =============================================================================
#
# Attach electrode coordinates and enrich preprocessing context with geometry.
#
# Logic only:
# - No file I/O (electrode tables are passed in as Python objects).
# - No CLI / printing.
#
# =============================================================================

from dataclasses import asdict
from typing import Mapping, Sequence, Tuple

import numpy as np

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
        raise ValueError(f"Unsupported unit: {unit!r}")

    return float(coords_m[0]), float(coords_m[1]), float(coords_m[2])


def attach_coordinates(
    raw: "mne.io.BaseRaw",
    electrodes_table: Mapping[str, Sequence[float]],
    cfg: CoordinatesConfig,
    ctx: PreprocContext,
) -> Tuple["mne.io.BaseRaw", BlockArtifact]:
    """
    Attach electrode coordinates to channels and enrich context geometry.

    Parameters
    ----------
    raw
        MNE Raw object.
    electrodes_table
        Mapping from channel name to (x, y, z) coordinates. Units are declared by `cfg.unit`.
    cfg
        Coordinates configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        Raw object (passthrough in v0).
    artifact
        Block artifact describing what was attached.

    Notes
    -----
    This v0 block stores coordinates in `ctx.geometry` but does not yet set an MNE montage.
    That will be implemented once your standardization layer defines channel matching rules.

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = attach_coordinates(
            raw=raw,
            electrodes_table={"A1": (12.3, -4.5, 6.7)},
            cfg=CoordinatesConfig(unit="mm"),
            ctx=ctx,
        )
    """
    import mne  # local import to keep import-time light

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("attach_coordinates expects an mne.io.BaseRaw.")

    ctx.add_record("coordinates", asdict(cfg))

    coords_m = {name: _to_meters(xyz, cfg.unit) for name, xyz in electrodes_table.items()}
    ctx.geometry = Geometry(coords_m=coords_m)

    artifact = BlockArtifact(
        name="coordinates",
        parameters=asdict(cfg),
        summary_metrics={"n_coords": len(coords_m)},
        warnings=["MNE montage attachment not implemented yet; geometry stored in ctx.geometry only."],
        figures=[],
    )
    return raw, artifact
