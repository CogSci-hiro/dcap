from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

_ATLAS_KEYS_TO_SKIP = {"cfg", "coi"}


def _as_label_vector(value: Any, n_contacts: int) -> np.ndarray:
    """
    Convert a MATLAB-imported label field into a 1D array of length n_contacts.

    Supports:
    - cellstr / object array of strings
    - MATLAB char matrix: shape (n_contacts, max_len) or (max_len, n_contacts)

    Returns
    -------
    labels : np.ndarray
        Shape (n_contacts,), dtype=str
    """
    arr = np.asarray(value)

    # Case 1: already list-like / object-like strings
    if arr.ndim == 1:
        return arr.astype(str).reshape(-1)

    # Case 2: MATLAB char matrix (2D) -> join characters per row/col
    if arr.ndim == 2:
        # If it's a typical char matrix, elements are single-character strings/bytes.
        # Try interpreting rows as labels first.
        if arr.shape[0] == n_contacts:
            labels = ["".join(map(str, row)).strip() for row in arr]
            return np.asarray(labels, dtype=str)

        # Or columns as labels
        if arr.shape[1] == n_contacts:
            labels = ["".join(map(str, col)).strip() for col in arr.T]
            return np.asarray(labels, dtype=str)

    # Fallback: stringify and hope, but it likely won't match n_contacts
    return arr.astype(str).reshape(-1)


def _as_1d_str_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    return arr.astype(str).reshape(-1)


def _as_2d_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected array of shape (n, 3), got {arr.shape}.")
    return arr


def _extract_prob_per_contact(prob: Any, n_contacts: int) -> Optional[np.ndarray]:
    if prob is None:
        return None

    prob_arr = np.asarray(prob, dtype=float)

    if prob_arr.ndim == 1 and prob_arr.shape[0] == n_contacts:
        return prob_arr

    if prob_arr.ndim == 2 and prob_arr.shape[0] == n_contacts:
        # Conservative: keep the max probability per contact
        return prob_arr.max(axis=1)

    return None


def parse_elec2atlas_payload(
    payload: Mapping[str, Any],
    *,
    keep_atlas_table: bool = True,
    wide_atlas_columns: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Parse elec2atlas MAT payload into electrode coordinates + atlas assignments.

    Parameters
    ----------
    payload
        Dict produced by scipy.io.loadmat(..., simplify_cells=True).
    keep_atlas_table
        If True, return a long-form atlas table with one row per (contact, atlas).
    wide_atlas_columns
        If True, also add wide columns `atlas_<Atlas>` and optional `prob_<Atlas>`
        into electrodes_df. (This is convenient but can get wide.)

    Returns
    -------
    electrodes_df
        One row per contact with coordinates in MNI and original space.
    atlas_df
        Long-form table with columns: contact, atlas, label, prob (optional).
        None if keep_atlas_table is False.

    Usage example
    -------------
        electrodes_df, atlas_df = parse_elec2atlas_payload(mat, keep_atlas_table=True)
    """
    if "coi" not in payload:
        raise KeyError("Expected top-level key 'coi' in elec2atlas payload.")

    coi = payload["coi"]
    if not isinstance(coi, dict):
        raise TypeError(f"Expected payload['coi'] to be dict, got {type(coi)}.")

    contacts = _as_1d_str_array(coi["label"])
    n_contacts = int(contacts.shape[0])

    elecpos_mni = _as_2d_float_array(coi["elecpos_mni"])
    elecpos_ori = _as_2d_float_array(coi["elecpos_ori"])

    if elecpos_mni.shape[0] != n_contacts:
        raise ValueError("coi['elecpos_mni'] length does not match number of contacts.")
    if elecpos_ori.shape[0] != n_contacts:
        raise ValueError("coi['elecpos_ori'] length does not match number of contacts.")

    electrodes_df = pd.DataFrame(
        {
            "contact": contacts,
            "x_mni": elecpos_mni[:, 0],
            "y_mni": elecpos_mni[:, 1],
            "z_mni": elecpos_mni[:, 2],
            "x_ori": elecpos_ori[:, 0],
            "y_ori": elecpos_ori[:, 1],
            "z_ori": elecpos_ori[:, 2],
        }
    )

    atlas_rows = []

    for atlas_name, atlas in payload.items():
        if atlas_name in _ATLAS_KEYS_TO_SKIP:
            continue
        if not isinstance(atlas, dict):
            continue
        if "label" not in atlas:
            continue

        labels = _as_label_vector(atlas["label"], n_contacts)
        if labels.shape[0] != n_contacts:
            continue

        prob_per_contact = _extract_prob_per_contact(atlas.get("prob", None), n_contacts)

        if keep_atlas_table:
            if prob_per_contact is None:
                atlas_rows.extend(
                    {"contact": contacts[i], "atlas": atlas_name, "label": labels[i]}
                    for i in range(n_contacts)
                )
            else:
                atlas_rows.extend(
                    {
                        "contact": contacts[i],
                        "atlas": atlas_name,
                        "label": labels[i],
                        "prob": float(prob_per_contact[i]),
                    }
                    for i in range(n_contacts)
                )

        if wide_atlas_columns:
            electrodes_df[f"atlas_{atlas_name}"] = labels
            if prob_per_contact is not None:
                electrodes_df[f"prob_{atlas_name}"] = prob_per_contact

    atlas_df = None
    if keep_atlas_table:
        atlas_df = pd.DataFrame(atlas_rows)
        # Keep a stable column order
        if "prob" in atlas_df.columns:
            atlas_df = atlas_df[["contact", "atlas", "label", "prob"]]
        else:
            atlas_df = atlas_df[["contact", "atlas", "label"]]

    return electrodes_df, atlas_df