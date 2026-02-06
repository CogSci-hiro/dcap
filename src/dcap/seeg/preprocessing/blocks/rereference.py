# =============================================================================
# =============================================================================
#                     ########################################
#                     #       BLOCK 7: REREFERENCING         #
#                     ########################################
# =============================================================================
# =============================================================================
"""
sEEG rereferencing utilities.

High-level intent
-----------------
This module deliberately supports TWO workflows:

A) Analysis-friendly API
   - `rereference(raw, method=...) -> Raw`
   - ergonomic for scripts/notebooks; mirrors MNE style ("just give me a Raw")

B) Clinical report / QC API
   - `rereference_view(raw, cfg, ctx) -> (views, artifact)`
   - returns multiple candidate rereferences ("views") + a BlockArtifact carrying:
     - parameters/provenance
     - warnings (e.g., missing geometry)
     - summary metrics for report/QC dashboards

The 'views' concept
-------------------
A "view" is just a named Raw variant:
- "original": unchanged input
- "car": common average reference
- "bipolar": adjacent contact differences (virtual channels)
- "laplacian": local spatial derivative along shaft or via neighbor graph
- "wm_ref": white-matter reference subtraction (global or per-channel map)

Virtual electrode geometry
--------------------------
Bipolar rereferencing creates NEW channels like "A1-A2". For these to be usable
in spatial plots, we attach a 3D location:
    p(A1-A2) = (p(A1) + p(A2)) / 2
when endpoint positions exist (montage or info['chs'][i]['loc']).

Implementation notes
--------------------
- Bipolar uses MNE `mne.set_bipolar_reference` (correctness, metadata handling).
- CAR / Laplacian / WM-ref are implemented as direct data transforms:
  we build a RawArray with copied Info + annotations.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import mne
import numpy as np

from dcap.seeg.preprocessing.configs import RereferenceConfig
from dcap.seeg.preprocessing.types import BlockArtifact, Geometry, PreprocContext

_DEFAULT_SEEG_CH_TYPE: str = "seeg"  # Currently unused; could be used for defaults/picks later.


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

    Design choice:
    -------------
    This is just a thin wrapper around `rereference_views(...)` that requests
    exactly ONE method. This keeps the method logic centralized.

    If the requested method can't be produced, we raise with warnings attached
    (except for "original", where we just return the input raw).
    """
    # Compute views for a single method.
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
        # Defensive fallback: "original" always exists conceptually even if not built.
        if method == "original":
            return raw
        # Attach warnings to error message to make failures actionable.
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

    Core algorithm:
    ---------------
    1) Validate input Raw (ensures data accessible)
    2) Compute iEEG picks (sEEG + optional ECoG)
    3) Compute shaft grouping (from Geometry if available; else infer from names)
    4) For each requested method:
         - compute view Raw (or None if impossible)
         - record warnings for report/QC and debugging

    Why warnings instead of always raising?
    --------------------------------------
    In a "view" context, you often want "best effort":
    e.g. generate CAR even if bipolar isn't possible (missing geometry).
    The caller (report/QC) can show a warning and choose another view.
    """
    _validate_raw(raw)

    requested = set(methods)

    # Always provide an "original" view so callers can always fall back.
    views: Dict[str, mne.io.BaseRaw] = {"original": raw}
    warnings: List[str] = []

    # Picks define which channels participate in references (CAR etc).
    # NOTE: This does not change which channels exist; it's about reference pools.
    picks = _pick_ieeg_channels(raw, include_ecog=include_ecog)
    if exclude_bads_from_reference:
        # Excluding bads prevents a known-bad channel from contaminating a reference signal.
        picks = _exclude_bads(raw, picks)

    # Shaft grouping is needed for:
    # - by-shaft CAR
    # - bipolar (adjacent contacts)
    # - shaft_1d Laplacian
    shafts, shafts_inferred = _get_shafts(geometry, raw)
    if shafts_inferred:
        # If we inferred shafts from names, it might be correct for typical "A1" naming,
        # but it is still a heuristic.
        if shafts:
            warnings.append("Shaft ordering inferred from channel names (fallback heuristic).")
        else:
            warnings.append("No shaft ordering available; bipolar/by-shaft CAR may be skipped or degraded.")

    # CAR is always possible as long as picks exist (otherwise it becomes a no-op).
    if "car" in requested:
        views["car"], car_w = _make_car_view(raw, picks=picks, car_scope=car_scope, shafts=shafts)
        warnings.extend(car_w)

    # Bipolar requires shaft ordering.
    if "bipolar" in requested:
        bip, bip_w = _make_bipolar_view(raw, shafts=shafts)
        warnings.extend(bip_w)
        if bip is not None:
            views["bipolar"] = bip

    # Laplacian requires either shafts (shaft_1d) or geometry.neighbors (knn_3d).
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

    # WM reference requires explicit info about WM channels or mapping.
    if "wm_ref" in requested:
        wm, wm_w = _make_wm_ref_view(raw, wm_contacts=wm_contacts, wm_reference_map=wm_reference_map)
        warnings.extend(wm_w)
        if wm is not None:
            views["wm_ref"] = wm

    # Defensive: ensure "original" exists even if user didn't request it explicitly.
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

    What this wrapper does:
    -----------------------
    - Adds provenance into ctx (ctx.add_record)
    - Translates config fields into the analysis-style function signature
    - Returns:
        (views, artifact)
      where artifact is used by the report/QC pipeline.

    Important:
    ----------
    - This function is *deliberately config-driven* because clinical pipelines
      are configured centrally.
    - For ad-hoc analysis, prefer `rereference(raw, method=...)`.
    """
    _validate_raw(raw)

    # Store structured provenance into the preprocessing context.
    # This makes the clinical report reproducible (what parameters were used?).
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
        # Geometry is usually tracked in ctx for clinical runs.
        geometry=ctx.geometry,
        # WM decisions are often made earlier in the pipeline and stored in ctx.decisions.
        wm_contacts=ctx.decisions.get("wm_contacts", None) if isinstance(ctx.decisions, dict) else None,
        wm_reference_map=ctx.decisions.get("wm_reference_map", None) if isinstance(ctx.decisions, dict) else None,
    )

    # Build block artifact for report/QC.
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
    # Basic runtime type guard (useful because MNE has multiple Raw subclasses).
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("Expected an mne.io.BaseRaw.")

    # Ensure data is available. We don't force preload=True at load time, but we do
    # need the samples. raw.get_data() will load/compute as needed.
    _ = raw.get_data()


def _copy_raw_with_data(template_raw: mne.io.BaseRaw, data: np.ndarray) -> mne.io.BaseRaw:
    """
    Create a RawArray that preserves Info + annotations.

    Why rebuild a RawArray?
    -----------------------
    For CAR/Laplacian/WM-ref we compute a new data matrix, then wrap it back into
    an MNE Raw. Using template_raw.info.copy() preserves:
    - channel names/types
    - sampling rate
    - montage/dig (if present in Info)
    """
    info = template_raw.info.copy()
    raw_out = mne.io.RawArray(data, info, verbose=False)
    raw_out.set_annotations(template_raw.annotations.copy())
    return raw_out


def _index_map(channel_names: Sequence[str]) -> Dict[str, int]:
    # Convenience to avoid repeated list.index calls (which are O(n)).
    return {name: idx for idx, name in enumerate(channel_names)}


def _pick_ieeg_channels(raw: mne.io.BaseRaw, *, include_ecog: bool) -> np.ndarray:
    # Picks channels used in reference computations.
    # NOTE: We explicitly exclude stim/eeg/meg/etc.
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
    # Excluding bad channels from reference pool prevents them from contaminating
    # CAR/WM means etc.
    bad_set = set(raw.info.get("bads", []))
    if not bad_set or picks.size == 0:
        return picks
    return np.array([p for p in picks if raw.ch_names[int(p)] not in bad_set], dtype=int)


def _infer_shafts_from_names(channel_names: Sequence[str]) -> Dict[str, List[str]]:
    """
    Infer shaft groupings from names like 'A1', 'A2', ... (letters + digits).

    This is a fallback:
    - works when contacts follow a consistent naming convention
    - can fail silently if naming differs ("LA01", "A-1", etc.)
    """
    pattern = re.compile(r"^(?P<shaft>[A-Za-z]+)(?P<contact>\d+)$")

    # Temporary structure: shaft -> list[(contact_number, channel_name)]
    by_shaft: Dict[str, List[Tuple[int, str]]] = {}
    for name in channel_names:
        match = pattern.match(name)
        if match is None:
            # Non-matching channel names are ignored by this heuristic.
            continue
        shaft = match.group("shaft")
        contact_index = int(match.group("contact"))
        by_shaft.setdefault(shaft, []).append((contact_index, name))

    # Sort each shaft by contact number so adjacency is correct.
    shafts: Dict[str, List[str]] = {}
    for shaft, items in by_shaft.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        shafts[shaft] = [nm for _, nm in items_sorted]
    return shafts


def _get_shafts(geometry: Optional[Geometry], raw: mne.io.BaseRaw) -> Tuple[Dict[str, List[str]], bool]:
    """
    Return shafts mapping and a flag indicating whether it was inferred.

    Returns
    -------
    shafts
        Mapping shaft_name -> ordered contact names.
    shafts_inferred
        True if inferred from channel names (heuristic), False if from Geometry.
    """
    if geometry is not None and geometry.shafts:
        return {k: list(v) for k, v in geometry.shafts.items()}, False
    return _infer_shafts_from_names(raw.ch_names), True


# ----------------------------- CAR -------------------------------------------
def _compute_car_global(data: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """
    Common-average reference (global pool).

    Formula:
        X'_i(t) = X_i(t) - mean_{j in picks} X_j(t)
    for all i in picks.
    """
    referenced = data.copy()
    if picks.size == 0:
        # No reference pool: treat as no-op.
        return referenced
    reference_signal = np.mean(data[picks, :], axis=0, keepdims=True)
    referenced[picks, :] = referenced[picks, :] - reference_signal
    return referenced


def _compute_car_by_groups(data: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    """
    Common-average reference within each group (e.g., per shaft).

    For each group G:
        X'_i(t) = X_i(t) - mean_{j in G} X_j(t)
    """
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
    """
    Create a CAR view.

    Behavior:
    - global: CAR across all picks
    - by_shaft: CAR computed separately per shaft group (falls back to global if missing)
    """
    warnings: List[str] = []
    data = raw.get_data()

    if car_scope == "global":
        return _copy_raw_with_data(raw, _compute_car_global(data, picks)), warnings

    if car_scope == "by_shaft":
        # by_shaft requires shaft grouping. If unavailable, fall back to global CAR.
        if not shafts:
            warnings.append("Requested CAR by_shaft but no shafts available; falling back to global CAR.")
            return _copy_raw_with_data(raw, _compute_car_global(data, picks)), warnings

        name_to_index = _index_map(raw.ch_names)
        groups: List[np.ndarray] = []

        # Convert shaft contact names into integer indices in the data matrix.
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

    For each shaft with contacts [A1, A2, A3], we generate:
      (A1, A2, "A1-A2")
      (A2, A3, "A2-A3")
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

    Notes:
    - MNE stores channel "loc" as a 12-length vector; first 3 values are xyz.
    - We treat a location as valid if it's finite and non-zero.
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

    For new channel C = A1-A2:
        p(C) = (p(A1) + p(A2)) / 2

    If an endpoint position is missing, we keep the location currently stored in
    bipolar_raw (MNE may set it to 0 or something else).
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


def _make_bipolar_view(
    raw: mne.io.BaseRaw,
    *,
    shafts: Mapping[str, Sequence[str]],
) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
    """
    Create a bipolar view using MNE's implementation.

    Returns None if we can't form pairs (no shafts / too few contacts).
    """
    warnings: List[str] = []
    if not shafts:
        warnings.append("Requested bipolar but no shafts available; bipolar view not produced.")
        return None, warnings

    pairs = _iter_bipolar_pairs(shafts)
    if not pairs:
        warnings.append("No bipolar pairs could be formed (missing shafts or insufficient contacts).")
        return None, warnings

    # Use MNE's built-in implementation for correctness & consistent metadata.
    anodes = [a for a, _, _ in pairs]
    cathodes = [c for _, c, _ in pairs]
    new_names = [n for _, _, n in pairs]

    # drop_refs=False keeps original channels too; you may or may not want that.
    # (Clinical/QC views often keep originals for comparison.)
    bipolar_raw = mne.set_bipolar_reference(
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=new_names,
        drop_refs=False,
        copy=True,
        verbose=False,
    )

    # Attach virtual electrode positions (midpoints) when possible.
    src_pos = _extract_channel_positions(raw)
    if src_pos:
        _attach_bipolar_positions(bipolar_raw, source_positions=src_pos, pairs=pairs)

    return bipolar_raw, warnings


# ---------------------------- Laplacian --------------------------------------
def _build_laplacian_shaft_1d(
    raw: mne.io.BaseRaw,
    shafts: Mapping[str, Sequence[str]],
) -> Tuple[mne.io.BaseRaw, List[str]]:
    """
    1D Laplacian along each shaft.

    For interior contacts k:
        X'_k = X_k - 0.5*(X_{k-1} + X_{k+1})

    Edge contacts are left unchanged and reported via warnings.
    """
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    referenced = data.copy()
    edge_contacts: List[str] = []

    for contacts in shafts.values():
        # Need at least 3 contacts to have an interior point with two neighbors.
        if len(contacts) < 3:
            continue
        for pos, contact_name in enumerate(contacts):
            if contact_name not in name_to_index:
                continue

            # Edge contacts don't have two neighbors -> keep unchanged.
            is_edge = pos == 0 or pos == (len(contacts) - 1)
            if is_edge:
                edge_contacts.append(contact_name)
                continue

            prev_name = contacts[pos - 1]
            next_name = contacts[pos + 1]
            # If neighbor names are missing in raw, treat as edge/unsupported.
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


def _build_laplacian_knn_3d(
    raw: mne.io.BaseRaw,
    neighbors: Mapping[str, Sequence[str]],
) -> Tuple[mne.io.BaseRaw, List[str]]:
    """
    Graph/neighbor Laplacian using a provided neighbor list per contact.

    For each contact i with neighbors N(i):
        X'_i = X_i - mean_{j in N(i)} X_j

    Contacts with no neighbors are left unchanged and reported via warnings.
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

        # Keep only neighbors that exist in the raw (robustness).
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
    """
    Create a laplacian view.

    Modes:
    - shaft_1d: requires shaft ordering
    - knn_3d: requires geometry.neighbors (explicit graph)

    Returns None with warning if required geometry is missing.
    """
    warnings: List[str] = []

    if not keep_original_names:
        # Current implementation does not create "virtual laplacian channels" with new names.
        # It modifies the same channels in-place in the returned RawArray.
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
    """
    White-matter (WM) referencing.

    Two supported strategies:
    1) Per-channel mapping:
       wm_reference_map = {"A1": "WM1", "A2": "WM1", ...}
       X'_A1 = X_A1 - X_WM1

    2) Global WM mean:
       wm_contacts = ["WM1", "WM2", ...]
       X'_i = X_i - mean(WM channels)
    """
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    referenced = data.copy()

    # Strategy 1: explicit mapping
    if isinstance(wm_reference_map, dict) and wm_reference_map:
        missing_pairs = 0
        for channel_name, wm_name in wm_reference_map.items():
            # Robustness: skip missing channel names.
            if channel_name not in name_to_index or wm_name not in name_to_index:
                missing_pairs += 1
                continue
            ch_i = name_to_index[channel_name]
            wm_i = name_to_index[wm_name]
            referenced[ch_i, :] = data[ch_i, :] - data[wm_i, :]

        if missing_pairs > 0:
            warnings.append(f"WM reference map had {missing_pairs} missing channel/WM pairs.")

        return _copy_raw_with_data(raw, referenced), warnings

    # Strategy 2: subtract mean of listed WM contacts from all channels
    if isinstance(wm_contacts, (list, tuple)) and len(wm_contacts) > 0:
        valid_indices = [name_to_index[name] for name in wm_contacts if name in name_to_index]
        if len(valid_indices) == 0:
            warnings.append("wm_contacts provided but none were found in raw.ch_names.")
            return None, warnings

        wm_mean = np.mean(data[valid_indices, :], axis=0, keepdims=True)
        referenced_all = data - wm_mean
        return _copy_raw_with_data(raw, referenced_all), warnings

    # No WM info provided -> cannot compute view.
    warnings.append("No WM reference info provided (wm_contacts / wm_reference_map).")
    return None, warnings
