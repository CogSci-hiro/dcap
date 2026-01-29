# =============================================================================
#                     ########################################
#                     #       BLOCK 7: REREFERENCING         #
#                     ########################################
# =============================================================================
#
# Generate rereferenced views:
# - CAR (common average): global or by-shaft
# - bipolar (shaft-local consecutive differences)
# - WM reference (mean WM or per-contact mapping)
# - Laplacian: shaft_1d (second spatial derivative) or knn_3d (local mean subtraction)
#
# Logic only:
# - No file I/O
# - No CLI / printing
#
# =============================================================================

import re
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import RereferenceConfig
from dcap.seeg.preprocessing.types import BlockArtifact, Geometry, PreprocContext


# =============================================================================
#                              INTERNAL CONSTANTS
# =============================================================================
_DEFAULT_SEEG_CH_TYPE: str = "seeg"


# =============================================================================
#                               HELPER FUNCTIONS
# =============================================================================
def _copy_raw_with_data(template_raw: mne.io.BaseRaw, data: np.ndarray) -> mne.io.BaseRaw:
    """
    Create a RawArray using template metadata and provided data.

    Parameters
    ----------
    template_raw
        Raw whose Info/annotations are used as template.
    data
        Array of shape (n_channels, n_times).

    Returns
    -------
    raw_out
        New RawArray with copied metadata.

    Usage example
    -------------
        raw_out = _copy_raw_with_data(raw, data)
    """
    info = template_raw.info.copy()
    raw_out = mne.io.RawArray(data, info, verbose=False)
    raw_out.set_annotations(template_raw.annotations.copy())
    return raw_out


def _index_map(channel_names: Sequence[str]) -> Dict[str, int]:
    """
    Build a channel name -> index map.

    Usage example
    -------------
        name_to_index = _index_map(raw.ch_names)
    """
    return {name: idx for idx, name in enumerate(channel_names)}


def _infer_shafts_from_names(channel_names: Sequence[str]) -> Dict[str, List[str]]:
    """
    Best-effort inference of shafts from names like 'A1', 'A2', ..., 'B1', ...

    Notes
    -----
    This is NOT validation; it is a fallback heuristic.

    Usage example
    -------------
        shafts = _infer_shafts_from_names(["A1", "A2", "B1"])
    """
    pattern = re.compile(r"^(?P<shaft>[A-Za-z]+)(?P<contact>\d+)$")
    by_shaft: Dict[str, List[Tuple[int, str]]] = {}

    for name in channel_names:
        match = pattern.match(name)
        if match is None:
            continue
        shaft = match.group("shaft")
        contact_index = int(match.group("contact"))
        by_shaft.setdefault(shaft, []).append((contact_index, name))

    shafts: Dict[str, List[str]] = {}
    for shaft, items in by_shaft.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        shafts[shaft] = [name for _, name in items_sorted]

    return shafts


def _get_shafts(ctx_geometry: Optional[Geometry], raw: mne.io.BaseRaw) -> Tuple[Dict[str, List[str]], bool]:
    """
    Retrieve shaft ordering from context geometry if available, else infer.

    Returns
    -------
    shafts
        Mapping shaft -> ordered channel names.
    inferred
        True if inferred (fallback), False if supplied by context.
    """
    if ctx_geometry is not None and ctx_geometry.shafts:
        return dict(ctx_geometry.shafts), False

    return _infer_shafts_from_names(raw.ch_names), True


def _compute_car_global(data: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """
    Apply global common average reference to selected channels.

    Usage example
    -------------
        data_car = _compute_car_global(data, picks)
    """
    referenced = data.copy()
    if picks.size == 0:
        return referenced
    reference_signal = np.mean(data[picks, :], axis=0, keepdims=True)
    referenced[picks, :] = referenced[picks, :] - reference_signal
    return referenced


def _compute_car_by_groups(data: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    """
    Apply group-wise common average reference.

    Usage example
    -------------
        data_car = _compute_car_by_groups(data, [np.array([0,1]), np.array([2,3])])
    """
    referenced = data.copy()
    for group_indices in groups:
        if group_indices.size == 0:
            continue
        reference_signal = np.mean(data[group_indices, :], axis=0, keepdims=True)
        referenced[group_indices, :] = referenced[group_indices, :] - reference_signal
    return referenced


def _build_bipolar_view(raw: mne.io.BaseRaw, shafts: Mapping[str, Sequence[str]]) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
    """
    Build bipolar derivations along shafts.

    Returns
    -------
    bipolar_raw
        Raw with bipolar channels, or None if no pairs could be formed.
    warnings
        List of warnings.

    Usage example
    -------------
        bipolar_raw, warnings = _build_bipolar_view(raw, shafts)
    """
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])

    bipolar_channel_names: List[str] = []
    bipolar_data: List[np.ndarray] = []

    for shaft_name, contacts in shafts.items():
        if len(contacts) < 2:
            continue
        for first_name, second_name in zip(contacts[:-1], contacts[1:]):
            if first_name not in name_to_index or second_name not in name_to_index:
                continue
            first_index = name_to_index[first_name]
            second_index = name_to_index[second_name]

            bipolar_channel_names.append(f"{first_name}-{second_name}")
            bipolar_data.append(data[first_index, :] - data[second_index, :])

    if len(bipolar_channel_names) == 0:
        warnings.append("No bipolar pairs could be formed (missing shafts or insufficient contacts).")
        return None, warnings

    bipolar_array = np.vstack([x[np.newaxis, :] for x in bipolar_data])
    info = mne.create_info(
        ch_names=bipolar_channel_names,
        sfreq=sfreq,
        ch_types=[_DEFAULT_SEEG_CH_TYPE] * len(bipolar_channel_names),
    )
    bipolar_raw = mne.io.RawArray(bipolar_array, info, verbose=False)
    bipolar_raw.set_annotations(raw.annotations.copy())

    return bipolar_raw, warnings


def _build_laplacian_shaft_1d(raw: mne.io.BaseRaw, shafts: Mapping[str, Sequence[str]]) -> Tuple[mne.io.BaseRaw, List[str]]:
    """
    Shaft-local 1D Laplacian:
        x_i - 0.5 * (x_{i-1} + x_{i+1})
    Edge contacts are left unchanged with a warning.

    Usage example
    -------------
        lap_raw, warnings = _build_laplacian_shaft_1d(raw, shafts)
    """
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    referenced = data.copy()

    edge_contacts: List[str] = []

    for shaft_name, contacts in shafts.items():
        if len(contacts) < 3:
            # Not enough to form an interior Laplacian; skip silently.
            continue

        for pos, contact_name in enumerate(contacts):
            if contact_name not in name_to_index:
                continue

            is_edge = pos == 0 or pos == (len(contacts) - 1)
            if is_edge:
                edge_contacts.append(contact_name)
                continue

            prev_name = contacts[pos - 1]
            next_name = contacts[pos + 1]
            if prev_name not in name_to_index or next_name not in name_to_index:
                edge_contacts.append(contact_name)
                continue

            i = name_to_index[contact_name]
            prev_i = name_to_index[prev_name]
            next_i = name_to_index[next_name]

            referenced[i, :] = data[i, :] - 0.5 * (data[prev_i, :] + data[next_i, :])

    if edge_contacts:
        warnings.append(
            f"Shaft_1d Laplacian left {len(edge_contacts)} edge/unsupported contacts unchanged."
        )

    return _copy_raw_with_data(raw, referenced), warnings


def _build_laplacian_knn_3d(
    raw: mne.io.BaseRaw,
    neighbors: Mapping[str, Sequence[str]],
) -> Tuple[mne.io.BaseRaw, List[str]]:
    """
    3D kNN Laplacian-like local reference:
        x_i - mean(x_neighbors)

    Contacts with no neighbors are left unchanged.

    Usage example
    -------------
        lap_raw, warnings = _build_laplacian_knn_3d(raw, neighbors)
    """
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    referenced = data.copy()

    no_neighbor_contacts: List[str] = []

    for contact_name in raw.ch_names:
        neighbor_names = list(neighbors.get(contact_name, []))
        if len(neighbor_names) == 0:
            no_neighbor_contacts.append(contact_name)
            continue

        valid_neighbor_indices = [name_to_index[n] for n in neighbor_names if n in name_to_index]
        if len(valid_neighbor_indices) == 0:
            no_neighbor_contacts.append(contact_name)
            continue

        i = name_to_index[contact_name]
        neighbor_mean = np.mean(data[valid_neighbor_indices, :], axis=0, keepdims=False)
        referenced[i, :] = data[i, :] - neighbor_mean

    if no_neighbor_contacts:
        warnings.append(
            f"kNN_3d Laplacian left {len(no_neighbor_contacts)} contacts unchanged (no neighbors)."
        )

    return _copy_raw_with_data(raw, referenced), warnings


def _build_wm_reference(
    raw: mne.io.BaseRaw,
    ctx: PreprocContext,
) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
    """
    White-matter referencing.

    Supported inputs (stored in ctx.decisions):
    - "wm_contacts": Sequence[str]
        Use mean of these contacts as reference for all channels.
    - "wm_reference_map": Mapping[str, str]
        Map from channel -> WM channel to subtract (per-contact referencing).

    Returns
    -------
    wm_raw
        WM-referenced Raw, or None if no WM information available.
    warnings
        List of warnings.

    Usage example
    -------------
        ctx.decisions["wm_contacts"] = ["WM1", "WM2"]
        wm_raw, warnings = _build_wm_reference(raw, ctx)
    """
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    referenced = data.copy()

    wm_reference_map = ctx.decisions.get("wm_reference_map", None)
    wm_contacts = ctx.decisions.get("wm_contacts", None)

    if isinstance(wm_reference_map, dict) and wm_reference_map:
        missing_pairs = 0
        for channel_name, wm_name in wm_reference_map.items():
            if channel_name not in name_to_index or wm_name not in name_to_index:
                missing_pairs += 1
                continue
            ch_i = name_to_index[channel_name]
            wm_i = name_to_index[wm_name]
            referenced[ch_i, :] = data[ch_i, :] - data[wm_i, :]

        if missing_pairs > 0:
            warnings.append(f"WM reference map had {missing_pairs} missing channel/Wm pairs.")

        return _copy_raw_with_data(raw, referenced), warnings

    if isinstance(wm_contacts, (list, tuple)) and len(wm_contacts) > 0:
        valid_indices = [name_to_index[name] for name in wm_contacts if name in name_to_index]
        if len(valid_indices) == 0:
            warnings.append("wm_contacts provided but none were found in raw.ch_names.")
            return None, warnings

        wm_mean = np.mean(data[valid_indices, :], axis=0, keepdims=True)
        referenced = data - wm_mean
        return _copy_raw_with_data(raw, referenced), warnings

    warnings.append("No WM reference information found in ctx.decisions (wm_contacts / wm_reference_map).")
    return None, warnings


# =============================================================================
#                                 PUBLIC API
# =============================================================================
def rereference(
    raw: mne.io.BaseRaw,
    cfg: RereferenceConfig,
    ctx: PreprocContext,
) -> Tuple[Dict[str, mne.io.BaseRaw], BlockArtifact]:
    """
    Generate one or more rereferenced views (CAR, bipolar, WM ref, Laplacian).

    Parameters
    ----------
    raw
        MNE Raw object.
    cfg
        Rereferencing configuration.
    ctx
        Preprocessing context. Geometry (shafts/neighbors) may be read from ctx.geometry.

    Returns
    -------
    views
        Mapping from view name to Raw object. Always includes "original".
    artifact
        Block artifact summarizing produced views.

    Usage example
    -------------
        ctx = PreprocContext()
        cfg = RereferenceConfig(methods=("car", "bipolar", "laplacian"), car_scope="by_shaft")
        views, artifact = rereference(raw, cfg, ctx)
        raw_car = views["car"]
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("rereference expects an mne.io.BaseRaw.")

    ctx.add_record("rereference", asdict(cfg))

    views: Dict[str, mne.io.BaseRaw] = {"original": raw}
    warnings: List[str] = []

    data = raw.get_data()
    name_to_index = _index_map(raw.ch_names)

    # Picks: for now, apply to all channels. You can later refine this (e.g., exclude stim/trigger).
    picks = np.arange(len(raw.ch_names), dtype=int)

    # Shaft ordering and neighbor graph (best-effort).
    shafts, shafts_inferred = _get_shafts(ctx.geometry, raw)
    if shafts_inferred:
        if shafts:
            warnings.append("Shaft ordering inferred from channel names (fallback heuristic).")
        else:
            warnings.append("No shaft ordering available; bipolar/by-shaft CAR may be skipped or degraded.")

    # -------------------------------------------------------------------------
    # CAR
    # -------------------------------------------------------------------------
    if "car" in cfg.methods:
        if cfg.car_scope == "global":
            car_data = _compute_car_global(data, picks)
            views["car"] = _copy_raw_with_data(raw, car_data)

        elif cfg.car_scope == "by_shaft":
            if not shafts:
                warnings.append("Requested CAR by_shaft but no shafts available; falling back to global CAR.")
                car_data = _compute_car_global(data, picks)
                views["car"] = _copy_raw_with_data(raw, car_data)
            else:
                group_indices: List[np.ndarray] = []
                for contacts in shafts.values():
                    indices = [name_to_index[name] for name in contacts if name in name_to_index]
                    if len(indices) > 0:
                        group_indices.append(np.asarray(indices, dtype=int))

                if len(group_indices) == 0:
                    warnings.append("Requested CAR by_shaft but no valid shaft groups found; falling back to global CAR.")
                    car_data = _compute_car_global(data, picks)
                    views["car"] = _copy_raw_with_data(raw, car_data)
                else:
                    car_data = _compute_car_by_groups(data, group_indices)
                    views["car"] = _copy_raw_with_data(raw, car_data)
        else:
            raise ValueError(f"Unknown car_scope: {cfg.car_scope!r}")

    # -------------------------------------------------------------------------
    # Bipolar
    # -------------------------------------------------------------------------
    if "bipolar" in cfg.methods:
        if not shafts:
            warnings.append("Requested bipolar but no shafts available; bipolar view not produced.")
        else:
            bipolar_raw, bipolar_warnings = _build_bipolar_view(raw, shafts)
            warnings.extend(bipolar_warnings)
            if bipolar_raw is not None:
                views["bipolar"] = bipolar_raw

    # -------------------------------------------------------------------------
    # WM reference
    # -------------------------------------------------------------------------
    if "wm_ref" in cfg.methods:
        wm_raw, wm_warnings = _build_wm_reference(raw, ctx)
        warnings.extend(wm_warnings)
        if wm_raw is not None:
            views["wm_ref"] = wm_raw

    # -------------------------------------------------------------------------
    # Laplacian
    # -------------------------------------------------------------------------
    if "laplacian" in cfg.methods:
        if cfg.laplacian_mode == "shaft_1d":
            if not shafts:
                warnings.append("Requested Laplacian shaft_1d but no shafts available; laplacian view not produced.")
            else:
                lap_raw, lap_warnings = _build_laplacian_shaft_1d(raw, shafts)
                warnings.extend(lap_warnings)
                views["laplacian"] = lap_raw

        elif cfg.laplacian_mode == "knn_3d":
            if ctx.geometry is None or not ctx.geometry.neighbors:
                warnings.append("Requested Laplacian knn_3d but ctx.geometry.neighbors is empty; laplacian view not produced.")
            else:
                lap_raw, lap_warnings = _build_laplacian_knn_3d(raw, ctx.geometry.neighbors)
                warnings.extend(lap_warnings)
                views["laplacian"] = lap_raw

        else:
            raise ValueError(f"Unknown laplacian_mode: {cfg.laplacian_mode!r}")

    artifact = BlockArtifact(
        name="rereference",
        parameters=asdict(cfg),
        summary_metrics={
            "views": sorted(list(views.keys())),
            "n_views": len(views),
            "shafts_available": bool(shafts),
            "neighbors_available": bool(ctx.geometry is not None and bool(ctx.geometry.neighbors)),
        },
        warnings=warnings,
        figures=[],
    )

    return views, artifact
