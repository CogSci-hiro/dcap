# =============================================================================
#                         BIDS: Event construction
# =============================================================================
#
# Convert annotations/triggers into BIDS events arrays.
#
# REVIEW
# =============================================================================

from dataclasses import dataclass
from typing import Dict, Tuple

import mne
import numpy as np


@dataclass(frozen=True)
class ConversationEventSpec:
    """
    Spec for generating conversation start/end events.

    Usage example
    -------------
        spec = ConversationEventSpec(start_delay_s=4.0, duration_s=240.0)
    """
    start_delay_s: float
    duration_s: float


def build_conversation_events(
    raw: mne.io.BaseRaw,
    delay_s: float,
    spec: ConversationEventSpec,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build two events: conversation_start and conversation_end.

    Notes
    -----
    Event codes start at 1 (avoid 0 because it's often treated as "no event").

    Parameters
    ----------
    raw
        Raw data.
    delay_s
        Offset between WAV and raw trigger timelines in seconds.
    spec
        Event spec.

    Returns
    -------
    events
        MNE events array (n_events, 3).
    event_id
        Mapping of event names to integer codes.

    Usage example
    -------------
        events, event_id = build_conversation_events(raw, delay_s=0.25, spec=ConversationEventSpec(4.0, 240.0))
    """
    sfreq = float(raw.info["sfreq"])

    conversation_start_s = spec.start_delay_s + float(delay_s)
    conversation_start_sample = int(round(conversation_start_s * sfreq))
    if conversation_start_sample < 0:
        # Skeleton policy: do not pad raw here; clamp to 0.
        # If you need padding to preserve negative start, implement it separately.
        conversation_start_sample = 0

    conversation_end_sample = conversation_start_sample + int(round(spec.duration_s * sfreq))

    event_id = {"conversation_start": 1, "conversation_end": 2}

    events = np.zeros((2, 3), dtype=int)
    events[0, 0] = conversation_start_sample
    events[0, 2] = event_id["conversation_start"]
    events[1, 0] = conversation_end_sample
    events[1, 2] = event_id["conversation_end"]

    return events, event_id


def triggers_from_annotations(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convert raw annotations to an MNE events array.

    Usage example
    -------------
        events, event_id = triggers_from_annotations(raw)
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return events, event_id
