from typing import Dict, List, Tuple

import mne


def choose_analysis_view(
    *,
    views: Dict[str, mne.io.BaseRaw],
    requested: str,
    fallback_order: Tuple[str, ...] = ("bipolar", "laplacian", "wm_ref", "car", "original"),
) -> Tuple[str, mne.io.BaseRaw, List[str]]:
    """
    Select the analysis view deterministically and transparently.

    Rules
    -----
    1) If `requested` exists in `views`, return it.
    2) Else, try fallbacks in `fallback_order`.
    3) If none exist, raise.

    Returns
    -------
    chosen_name, chosen_raw, warnings
    """
    warnings: List[str] = []

    req = (requested or "").strip().lower()
    if not req:
        req = "original"

    # Normalize common aliases
    aliases = {
        "car_global": "car",
        "car-by-shaft": "car",
        "car_by_shaft": "car",
        "common_average": "car",
        "avg": "car",
        "lap": "laplacian",
        "lapp": "laplacian",
        "bip": "bipolar",
        "wm": "wm_ref",
    }
    req = aliases.get(req, req)

    if req in views:
        return req, views[req], warnings

    warnings.append(
        f"Requested analysis view {requested!r} was not produced. "
        f"Available views: {sorted(views.keys())}. Falling back."
    )

    for fb in fallback_order:
        if fb in views:
            warnings.append(f"Falling back to analysis view {fb!r}.")
            return fb, views[fb], warnings

    raise RuntimeError(f"No usable analysis view found. Available views: {sorted(views.keys())}.")
