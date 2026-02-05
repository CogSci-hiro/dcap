# =============================================================================
# =============================================================================
#                     ########################################
#                     #       BLOCK 7: REREFERENCING         #
#                     ########################################
# =============================================================================
# =============================================================================
"""
sEEG rereferencing utilities.

Why two APIs?
-------------
This module supports two distinct use-cases:

1) Analysis-friendly single-view API (library style)
   - Use this when you just want "a Raw rereferenced in some way".
   - This intentionally mirrors the feel of MNE methods.

2) Clinical-report multi-view API ("views")
   - Use this when you want several rereferenced variants at once, plus
     provenance/warnings as a BlockArtifact, so downstream QC/reporting can
     choose an analysis view.

Key functions
-------------
- rereference(raw, method=...) -> BaseRaw
    Single rereferenced raw (most convenient for normal analysis).

- rereference_views(raw, methods=...) -> dict[str, BaseRaw]
    Build many views ("original", "car", "bipolar", ...).

- rereference_view(raw, cfg, ctx) -> (views, artifact)
    Compat wrapper for the clinical preprocessing pipeline. This is the
    old behavior (previously named `rereference`).

Virtual electrode geometry
--------------------------
For bipolar rereferencing, new "virtual" channels are created. If the input Raw
has channel positions (via montage / info['chs'][i]['loc']), this module will
attach positions for bipolar channels as midpoints between the paired contacts.

Notes
-----
- Bipolar creation uses MNE's built-in `mne.set_bipolar_reference` where possible.
- CAR and Laplacian variants here operate directly on the data matrix and then
  rebuild a Raw with identical Info + annotations, preserving dig where possible.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import mne
import numpy as np

from dcap.seeg.preprocessing.configs import RereferenceConfig
from dcap.seeg.preprocessing.types import BlockArtifact, Geometry, PreprocContext

_DEFAULT_SEEG_CH_TYPE: str = "seeg"


# =============================================================================
#                     ########################################
#                     #          Public API (analysis)        #
#                     ########################################
# =============================================================================
def rereference(
    raw: mne.io.BaseRaw,
    method: str = "car",
    *,
    car_scope: str = "global",
    laplacian_mode: str = "shaft_1d",
    include_ecog: bool = True,
    exclude_bads_from_reference: bool = True,
    keep_original_names_for_laplacian: bool = True,
    # Optional geometry hints:
    geometry: Optional[Geometry] = None,
    # WM reference hints (optional):
    wm_contacts: Optional[Sequence[str]] = None,
    wm_reference_map: Optional[Mapping[str, str]] = None,
) -> mne.io.BaseRaw:
    """
    Apply a single rereferencing method and return one Raw.

    Parameters
    ----------
    raw
        Input MNE Raw. Must be preloaded (or loadable) because data is accessed.
    method
        One of: {"original", "car", "bipolar", "laplacian", "wm_ref"}.
    car_scope
        If method="car": {"global", "by_shaft"}.
    laplacian_mode
        If method="laplacian": {"shaft_1d", "knn_3d"}.
    include_ecog
        Whether to include ECoG channels when computing reference pools.
    exclude_bads_from_reference
        If True, channels in raw.info["bads"] are excluded from any reference pool
        (CAR, etc).
    keep_original_names_for_laplacian
        Laplacian here is implemented as a data transform on the same channels,
        so channel names remain unchanged.
    geometry
        Optional Geometry containing shaft ordering / neighbors. If not provided,
        a fallback heuristic will infer shafts from channel names like "A1".
    wm_contacts
        If method="wm_ref": list of WM channels to average and subtract from all.
    wm_reference_map
        If method="wm_ref": mapping channel -> WM channel to subtract per-channel.

    Returns
    -------
    raw_out
        Rereferenced Raw.

    Usage example
    -------------
        from dcap import rereference

        raw_car = rereference(raw, method="car")
        raw_bip = rereference(raw, method="bipolar")
        raw_lap = rereference(raw, method="laplacian", laplacian_mode="shaft_1d")
    """
    views, warnings = rereference_views(
        raw,
        methods=(method,),
        car_scope=car_scope,
        laplacian_mode=laplacian_mode,
        include_ecog=include_ecog,
        exclude_bads_from_reference=exclude_bads_from_reference,
        keep_original_names_for_laplacian=keep_original_names_for_laplacian,
        geometry=geometry,
        wm_contacts=wm_contacts,
        wm_reference_map=wm_reference_map,
    )
    # method may be "original" or any other
    key = method
    if key not in views:
        # Defensive: fall back to original if "original" was requested, else error.
        if method == "original":
            return raw
        raise ValueError(f"Requested method {method!r} but it was not produced. Warnings: {warnings}")
    return views[key]


def rereference_views(
    raw: mne.io.BaseRaw,
    methods: Sequence[str] = ("car", "bipolar", "laplacian", "wm_ref"),
    *,
    car_scope: str = "global",
    laplacian_mode: str = "shaft_1d",
    include_ecog: bool = True,
    exclude_bads_from_reference: bool = True,
    keep_original_names_for_laplacian: bool = True,
    geometry: Optional[Geometry] = None,
    wm_contacts: Optional[Sequence[str]] = None,
    wm_reference_map: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, mne.io.BaseRaw], List[str]]:
    """
    Build a dictionary of rereferenced "views".

    Parameters
    ----------
    raw
        Input Raw.
    methods
        Methods to produce. Supports: "original", "car", "bipolar", "laplacian", "wm_ref".
    car_scope, laplacian_mode, include_ecog, exclude_bads_from_reference, geometry, wm_contacts, wm_reference_map
        See `rereference()`.

    Returns
    -------
    views
        Mapping view-name -> Raw. Always includes "original".
    warnings
        Human-readable warnings about fallbacks, missing geometry, etc.

    Usage example
    -------------
        views, warnings = rereference_views(raw, methods=("car", "bipolar"))
        raw_analysis = views["bipolar"]
    """
    _validate_raw(raw)

    requested = set(methods)
    views: Dict[str, mne.io.BaseRaw] = {"original": raw}
    warnings: List[str] = []

    picks = _pick_ieeg_channels(raw, include_ecog=include_ecog)
    if exclude_bads_from_reference:
        picks = _exclude_bads(raw, picks)

    shafts, shafts_inferred = _get_shafts(geometry, raw)
    if shafts_inferred:
        if shafts:
            warnings.append("Shaft ordering inferred from channel names (fallback heuristic).")
        else:
            warnings.append("No shaft ordering available; bipolar/by-shaft CAR may be skipped or degraded.")

    if "car" in requested:
        views["car"], car_w = _make_car_view(raw, picks=picks, car_scope=car_scope, shafts=shafts)
        warnings.extend(car_w)

    if "bipolar" in requested:
        bip, bip_w = _make_bipolar_view(raw, shafts=shafts)
        warnings.extend(bip_w)
        if bip is not None:
            views["bipolar"] = bip

    if "laplacian" in requested:
        lap, lap_w = _make_laplacian_view(
            raw,
            laplacian_mode=laplacian_mode,
            shafts=shafts,
            geometry=geometry,
            keep_original_names=keep_original_names_for_laplacian,
        )
        warnings.extend(lap_w)
        if lap is not None:
            views["laplacian"] = lap

    if "wm_ref" in requested:
        wm, wm_w = _make_wm_ref_view(raw, wm_contacts=wm_contacts, wm_reference_map=wm_reference_map)
        warnings.extend(wm_w)
        if wm is not None:
            views["wm_ref"] = wm

    # Ensure "original" exists even if user didn't request it explicitly
    views.setdefault("original", raw)

    return views, warnings


# =============================================================================
#                     ########################################
#                     #       Compat API (clinical block)     #
#                     ########################################
# =============================================================================
def rereference_view(
    raw: mne.io.BaseRaw,
    cfg: RereferenceConfig,
    ctx: PreprocContext,
) -> Tuple[Dict[str, mne.io.BaseRaw], BlockArtifact]:
    """
    Compat wrapper for the clinical preprocessing pipeline.

    This function is the previous block behavior (formerly named `rereference`):
    it returns a dict of views and a BlockArtifact with provenance + warnings.

    Usage example
    -------------
        views, artifact = rereference_view(raw, cfg, ctx)
        raw_for_qc = views["car"]
    """
    _validate_raw(raw)

    ctx.add_record("rereference", asdict(cfg))

    # Translate config -> library-style args
    # (Keep the names consistent with existing config fields.)
    views, warnings = rereference_views(
        raw,
        methods=tuple(cfg.methods),
        car_scope=str(cfg.car_scope),
        laplacian_mode=str(cfg.laplacian_mode),
        include_ecog=True,
        exclude_bads_from_reference=True,
        geometry=ctx.geometry,
        wm_contacts=ctx.decisions.get("wm_contacts", None) if isinstance(ctx.decisions, dict) else None,
        wm_reference_map=ctx.decisions.get("wm_reference_map", None) if isinstance(ctx.decisions, dict) else None,
    )

    artifact = BlockArtifact(
        name="rereference",
        parameters=asdict(cfg),
        summary_metrics={"n_views": float(len(views))},
        warnings=warnings,
        figures=[],
    )
    return views, artifact


# =============================================================================
#                     ########################################
#                     #              Internals               #
#                     ########################################
# =============================================================================
def _validate_raw(raw: mne.io.BaseRaw) -> None:
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("Expected an mne.io.BaseRaw.")
    # Ensure data is available. We don't force preload=True at load time, but we do
    # need the samples.
    _ = raw.get_data()


def _copy_raw_with_data(template_raw: mne.io.BaseRaw, data: np.ndarray) -> mne.io.BaseRaw:
    """
    Create a RawArray that preserves Info + annotations.

    Notes
    -----
    - Using template_raw.info.copy() preserves dig/montage information if present.
    - Channel order and count must match the template.
    """
    info = template_raw.info.copy()
    raw_out = mne.io.RawArray(data, info, verbose=False)
    raw_out.set_annotations(template_raw.annotations.copy())
    return raw_out


def _index_map(channel_names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(channel_names)}


def _pick_ieeg_channels(raw: mne.io.BaseRaw, *, include_ecog: bool) -> np.ndarray:
    return mne.pick_types(
        raw.info,
        seeg=True,
        ecog=bool(include_ecog),
        eeg=False,
        meg=False,
        stim=False,
        misc=False,
        eog=False,
        ecg=False,
    )


def _exclude_bads(raw: mne.io.BaseRaw, picks: np.ndarray) -> np.ndarray:
    bad_set = set(raw.info.get("bads", []))
    if not bad_set or picks.size == 0:
        return picks
    return np.array([p for p in picks if raw.ch_names[int(p)] not in bad_set], dtype=int)


def _infer_shafts_from_names(channel_names: Sequence[str]) -> Dict[str, List[str]]:
    """
    Infer shaft groupings from names like 'A1', 'A2', ... (letters + digits).
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
        shafts[shaft] = [nm for _, nm in items_sorted]
    return shafts


def _get_shafts(geometry: Optional[Geometry], raw: mne.io.BaseRaw) -> Tuple[Dict[str, List[str]], bool]:
    """
    Return shafts mapping and a flag indicating whether it was inferred.
    """
    if geometry is not None and geometry.shafts:
        return {k: list(v) for k, v in geometry.shafts.items()}, False
    return _infer_shafts_from_names(raw.ch_names), True


# ----------------------------- CAR -------------------------------------------
def _compute_car_global(data: np.ndarray, picks: np.ndarray) -> np.ndarray:
    referenced = data.copy()
    if picks.size == 0:
        return referenced
    reference_signal = np.mean(data[picks, :], axis=0, keepdims=True)
    referenced[picks, :] = referenced[picks, :] - reference_signal
    return referenced


def _compute_car_by_groups(data: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    referenced = data.copy()
    for group_indices in groups:
        if group_indices.size == 0:
            continue
        reference_signal = np.mean(data[group_indices, :], axis=0, keepdims=True)
        referenced[group_indices, :] = referenced[group_indices, :] - reference_signal
    return referenced


def _make_car_view(
    raw: mne.io.BaseRaw,
    *,
    picks: np.ndarray,
    car_scope: str,
    shafts: Mapping[str, Sequence[str]],
) -> Tuple[mne.io.BaseRaw, List[str]]:
    warnings: List[str] = []
    data = raw.get_data()

    if car_scope == "global":
        return _copy_raw_with_data(raw, _compute_car_global(data, picks)), warnings

    if car_scope == "by_shaft":
        if not shafts:
            warnings.append("Requested CAR by_shaft but no shafts available; falling back to global CAR.")
            return _copy_raw_with_data(raw, _compute_car_global(data, picks)), warnings

        name_to_index = _index_map(raw.ch_names)
        groups: List[np.ndarray] = []
        for contacts in shafts.values():
            indices = [name_to_index[name] for name in contacts if name in name_to_index]
            if indices:
                groups.append(np.asarray(indices, dtype=int))

        if not groups:
            warnings.append("Requested CAR by_shaft but no valid shaft groups; falling back to global CAR.")
            return _copy_raw_with_data(raw, _compute_car_global(data, picks)), warnings

        return _copy_raw_with_data(raw, _compute_car_by_groups(data, groups)), warnings

    raise ValueError(f"Unknown car_scope: {car_scope!r}")


# ---------------------------- Bipolar ----------------------------------------
def _iter_bipolar_pairs(shafts: Mapping[str, Sequence[str]]) -> List[Tuple[str, str, str]]:
    """
    Return list of (anode, cathode, new_name) pairs.
    """
    pairs: List[Tuple[str, str, str]] = []
    for contacts in shafts.values():
        if len(contacts) < 2:
            continue
        for first_name, second_name in zip(contacts[:-1], contacts[1:]):
            pairs.append((first_name, second_name, f"{first_name}-{second_name}"))
    return pairs


def _extract_channel_positions(raw: mne.io.BaseRaw) -> Dict[str, np.ndarray]:
    """
    Extract per-channel xyz positions from info['chs'][i]['loc'] when present.
    """
    pos: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(raw.ch_names):
        loc = raw.info["chs"][idx]["loc"]
        xyz = np.array(loc[:3], dtype=float)
        if np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 0:
            pos[name] = xyz
    return pos


def _attach_bipolar_positions(
    bipolar_raw: mne.io.BaseRaw,
    *,
    source_positions: Mapping[str, np.ndarray],
    pairs: Sequence[Tuple[str, str, str]],
) -> None:
    """
    Attach virtual electrode positions for bipolar channels as midpoints.

    If either endpoint position is missing, the bipolar channel will keep whatever
    MNE assigned (or zero).
    """
    name_to_idx = _index_map(bipolar_raw.ch_names)
    for anode, cathode, new_name in pairs:
        if new_name not in name_to_idx:
            continue
        if anode not in source_positions or cathode not in source_positions:
            continue
        midpoint = 0.5 * (source_positions[anode] + source_positions[cathode])
        ch_i = int(name_to_idx[new_name])
        bipolar_raw.info["chs"][ch_i]["loc"][:3] = midpoint


def _make_bipolar_view(raw: mne.io.BaseRaw, *, shafts: Mapping[str, Sequence[str]]) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
    warnings: List[str] = []
    if not shafts:
        warnings.append("Requested bipolar but no shafts available; bipolar view not produced.")
        return None, warnings

    pairs = _iter_bipolar_pairs(shafts)
    if not pairs:
        warnings.append("No bipolar pairs could be formed (missing shafts or insufficient contacts).")
        return None, warnings

    # Use MNE's built-in implementation for correctness.
    anodes = [a for a, _, _ in pairs]
    cathodes = [c for _, c, _ in pairs]
    new_names = [n for _, _, n in pairs]

    # copy=True returns a new Raw instance.
    bipolar_raw = mne.set_bipolar_reference(
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=new_names,
        drop_refs=False,
        copy=True,
        verbose=False,
    )

    # Try to set virtual electrode positions (midpoints) if we can.
    src_pos = _extract_channel_positions(raw)
    if src_pos:
        _attach_bipolar_positions(bipolar_raw, source_positions=src_pos, pairs=pairs)

    return bipolar_raw, warnings


# ---------------------------- Laplacian --------------------------------------
def _build_laplacian_shaft_1d(raw: mne.io.BaseRaw, shafts: Mapping[str, Sequence[str]]) -> Tuple[mne.io.BaseRaw, List[str]]:
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    referenced = data.copy()
    edge_contacts: List[str] = []

    for contacts in shafts.values():
        if len(contacts) < 3:
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
        warnings.append(f"Shaft_1d Laplacian left {len(edge_contacts)} edge/unsupported contacts unchanged.")

    return _copy_raw_with_data(raw, referenced), warnings


def _build_laplacian_knn_3d(raw: mne.io.BaseRaw, neighbors: Mapping[str, Sequence[str]]) -> Tuple[mne.io.BaseRaw, List[str]]:
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
        neighbor_mean = np.mean(data[valid_neighbor_indices, :], axis=0)
        referenced[i, :] = data[i, :] - neighbor_mean

    if no_neighbor_contacts:
        warnings.append(f"kNN_3d Laplacian left {len(no_neighbor_contacts)} contacts unchanged (no neighbors).")

    return _copy_raw_with_data(raw, referenced), warnings


def _make_laplacian_view(
    raw: mne.io.BaseRaw,
    *,
    laplacian_mode: str,
    shafts: Mapping[str, Sequence[str]],
    geometry: Optional[Geometry],
    keep_original_names: bool,
) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
    warnings: List[str] = []

    if not keep_original_names:
        # Today we keep names unchanged because this is a same-channel transform.
        # If later you want a "virtual-laplacian" set, this can be extended.
        warnings.append("keep_original_names_for_laplacian=False is not supported; using original names.")

    if laplacian_mode == "shaft_1d":
        if not shafts:
            warnings.append("Requested Laplacian shaft_1d but no shafts available; not produced.")
            return None, warnings
        lap, w = _build_laplacian_shaft_1d(raw, shafts)
        warnings.extend(w)
        return lap, warnings

    if laplacian_mode == "knn_3d":
        if geometry is None or not geometry.neighbors:
            warnings.append("Requested Laplacian knn_3d but geometry.neighbors is empty; not produced.")
            return None, warnings
        lap, w = _build_laplacian_knn_3d(raw, geometry.neighbors)
        warnings.extend(w)
        return lap, warnings

    raise ValueError(f"Unknown laplacian_mode: {laplacian_mode!r}")


# ---------------------------- WM reference -----------------------------------
def _make_wm_ref_view(
    raw: mne.io.BaseRaw,
    *,
    wm_contacts: Optional[Sequence[str]],
    wm_reference_map: Optional[Mapping[str, str]],
) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    referenced = data.copy()

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
            warnings.append(f"WM reference map had {missing_pairs} missing channel/WM pairs.")
        return _copy_raw_with_data(raw, referenced), warnings

    if isinstance(wm_contacts, (list, tuple)) and len(wm_contacts) > 0:
        valid_indices = [name_to_index[name] for name in wm_contacts if name in name_to_index]
        if len(valid_indices) == 0:
            warnings.append("wm_contacts provided but none were found in raw.ch_names.")
            return None, warnings
        wm_mean = np.mean(data[valid_indices, :], axis=0, keepdims=True)
        referenced_all = data - wm_mean
        return _copy_raw_with_data(raw, referenced_all), warnings

    warnings.append("No WM reference info provided (wm_contacts / wm_reference_map).")
    return None, warnings
