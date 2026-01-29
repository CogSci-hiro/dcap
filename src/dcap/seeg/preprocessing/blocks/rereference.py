# =============================================================================
# =============================================================================
#                     ########################################
#                     #       BLOCK 7: REREFERENCING         #
#                     ########################################
# =============================================================================
# =============================================================================

import re
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import RereferenceConfig
from dcap.seeg.preprocessing.types import BlockArtifact, Geometry, PreprocContext


_DEFAULT_SEEG_CH_TYPE: str = "seeg"


def _copy_raw_with_data(template_raw: mne.io.BaseRaw, data: np.ndarray) -> mne.io.BaseRaw:
    info = template_raw.info.copy()
    raw_out = mne.io.RawArray(data, info, verbose=False)
    raw_out.set_annotations(template_raw.annotations.copy())
    return raw_out


def _index_map(channel_names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(channel_names)}


def _infer_shafts_from_names(channel_names: Sequence[str]) -> Dict[str, List[str]]:
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


def _get_shafts(ctx_geometry: Optional[Geometry], raw: mne.io.BaseRaw) -> Tuple[Dict[str, List[str]], bool]:
    if ctx_geometry is not None and ctx_geometry.shafts:
        return {k: list(v) for k, v in ctx_geometry.shafts.items()}, False
    return _infer_shafts_from_names(raw.ch_names), True


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


def _build_bipolar_view(raw: mne.io.BaseRaw, shafts: Mapping[str, Sequence[str]]) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
    warnings: List[str] = []
    name_to_index = _index_map(raw.ch_names)
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])

    bipolar_channel_names: List[str] = []
    bipolar_data: List[np.ndarray] = []

    for contacts in shafts.values():
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


def _build_wm_reference(raw: mne.io.BaseRaw, ctx: PreprocContext) -> Tuple[Optional[mne.io.BaseRaw], List[str]]:
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


def rereference(
    raw: mne.io.BaseRaw,
    cfg: RereferenceConfig,
    ctx: PreprocContext,
) -> Tuple[Dict[str, mne.io.BaseRaw], BlockArtifact]:
    """Generate one or more rereferenced views."""
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("rereference expects an mne.io.BaseRaw.")

    ctx.add_record("rereference", asdict(cfg))

    views: Dict[str, mne.io.BaseRaw] = {"original": raw}
    warnings: List[str] = []

    data = raw.get_data()
    picks = np.arange(len(raw.ch_names), dtype=int)

    shafts, shafts_inferred = _get_shafts(ctx.geometry, raw)
    if shafts_inferred:
        if shafts:
            warnings.append("Shaft ordering inferred from channel names (fallback heuristic).")
        else:
            warnings.append("No shaft ordering available; bipolar/by-shaft CAR may be skipped or degraded.")

    if "car" in cfg.methods:
        if cfg.car_scope == "global":
            views["car"] = _copy_raw_with_data(raw, _compute_car_global(data, picks))
        elif cfg.car_scope == "by_shaft":
            if not shafts:
                warnings.append("Requested CAR by_shaft but no shafts available; falling back to global CAR.")
                views["car"] = _copy_raw_with_data(raw, _compute_car_global(data, picks))
            else:
                name_to_index = _index_map(raw.ch_names)
                groups = []
                for contacts in shafts.values():
                    indices = [name_to_index[name] for name in contacts if name in name_to_index]
                    if indices:
                        groups.append(np.asarray(indices, dtype=int))
                if not groups:
                    warnings.append("Requested CAR by_shaft but no valid shaft groups; falling back to global CAR.")
                    views["car"] = _copy_raw_with_data(raw, _compute_car_global(data, picks))
                else:
                    views["car"] = _copy_raw_with_data(raw, _compute_car_by_groups(data, groups))
        else:
            raise ValueError(f"Unknown car_scope: {cfg.car_scope!r}")

    if "bipolar" in cfg.methods:
        if shafts:
            bipolar_raw, bipolar_warnings = _build_bipolar_view(raw, shafts)
            warnings.extend(bipolar_warnings)
            if bipolar_raw is not None:
                views["bipolar"] = bipolar_raw
        else:
            warnings.append("Requested bipolar but no shafts available; bipolar view not produced.")

    if "wm_ref" in cfg.methods:
        wm_raw, wm_warnings = _build_wm_reference(raw, ctx)
        warnings.extend(wm_warnings)
        if wm_raw is not None:
            views["wm_ref"] = wm_raw

    if "laplacian" in cfg.methods:
        if cfg.laplacian_mode == "shaft_1d":
            if shafts:
                lap_raw, lap_warnings = _build_laplacian_shaft_1d(raw, shafts)
                warnings.extend(lap_warnings)
                views["laplacian"] = lap_raw
            else:
                warnings.append("Requested Laplacian shaft_1d but no shafts available; not produced.")
        elif cfg.laplacian_mode == "knn_3d":
            if ctx.geometry is not None and ctx.geometry.neighbors:
                lap_raw, lap_warnings = _build_laplacian_knn_3d(raw, ctx.geometry.neighbors)
                warnings.extend(lap_warnings)
                views["laplacian"] = lap_raw
            else:
                warnings.append("Requested Laplacian knn_3d but ctx.geometry.neighbors is empty; not produced.")
        else:
            raise ValueError(f"Unknown laplacian_mode: {cfg.laplacian_mode!r}")

    artifact = BlockArtifact(
        name="rereference",
        parameters=asdict(cfg),
        summary_metrics={"n_views": float(len(views))},
        warnings=warnings,
        figures=[],
    )
    return views, artifact
