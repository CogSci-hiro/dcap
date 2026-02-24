from typing import Final

import mne
import numpy as np

from dcap.bids.core.events import PreparedEvents, make_events, no_events


# Provisional trigger semantics for picture naming, inferred from:
# - timing regularity in raw annotations (10003 -> 10005 is highly stable)
# - task presentation log entries (Picture / isi / pause / Response 99)
# These labels can be refined later without changing the underlying codes.
NAMING_EVENT_ID: Final[dict[str, int]] = {
    "run_start": 88,
    "response": 99,
    "trial_prep": 98,
    "picture_onset": 10003,
    "pause_onset": 10004,
    "isi_onset": 10005,
    "special_marker": 10006,
}

# Annotation-only/comment markers emitted by BrainVision import; excluded from BIDS events.
_IGNORED_CODES: Final[set[int]] = {10001, 10002}


def prepare_naming_events(raw: mne.io.BaseRaw) -> PreparedEvents:
    """
    Extract picture naming triggers from annotations and map them to stable labels.

    Notes
    -----
    - Keeps all task-relevant trigger codes (including block/pause/response markers).
    - Ignores BrainVision comment annotations at sample 0/1.
    """
    events, _ = mne.events_from_annotations(raw, verbose=False)
    if events.size == 0:
        return no_events()

    values = events[:, 2].astype(int, copy=False)
    keep_mask = np.isin(values, np.array(sorted(NAMING_EVENT_ID.values()), dtype=int))

    filtered = np.asarray(events[keep_mask], dtype=int)
    if filtered.size == 0:
        return no_events()

    unknown_codes = set(np.unique(values)) - set(NAMING_EVENT_ID.values()) - _IGNORED_CODES
    if unknown_codes:
        # Fail fast so we do not silently discard newly observed task markers.
        unknown = ", ".join(str(int(c)) for c in sorted(unknown_codes))
        raise ValueError(
            "Unknown naming trigger code(s) found in annotations: "
            f"{unknown}. Update dcap.bids.tasks.naming.events.NAMING_EVENT_ID."
        )

    return make_events(filtered, NAMING_EVENT_ID)

