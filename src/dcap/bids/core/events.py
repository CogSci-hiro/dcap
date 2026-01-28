# src/dcap/bids/core/events.py
# =============================================================================
#                        BIDS Core: Events containers
# =============================================================================
#
# Task-agnostic event types and small utilities.
#
# Tasks are responsible for deciding *what* events mean and *how* they are
# constructed. Core provides a minimal, typed container so the converter can
# accept or omit events consistently.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


# =============================================================================
# Public containers
# =============================================================================

@dataclass(frozen=True)
class PreparedEvents:
    """
    Container for events and event_id to be passed to MNE-BIDS.

    Parameters
    ----------
    events
        MNE-style events array of shape (n_events, 3) with dtype int.
        Column meanings:
          - sample index
          - previous value (unused; usually 0)
          - event code
    event_id
        Mapping from event name (string) to integer event code.

    Notes
    -----
    - Provide both `events` and `event_id`, or neither (both None).
    - Core code does not interpret event semantics.

    Usage example
    -------------
        events = np.array([[100, 0, 1], [200, 0, 2]], dtype=int)
        event_id = {"start": 1, "end": 2}
        prepared = PreparedEvents(events=events, event_id=event_id)
    """

    events: Optional[np.ndarray]
    event_id: Optional[Mapping[str, int]]


# =============================================================================
# Convenience constructors
# =============================================================================

def no_events() -> PreparedEvents:
    """
    Convenience constructor for tasks that do not define events.

    Returns
    -------
    PreparedEvents
        A container with both fields set to None.

    Usage example
    -------------
        return no_events()
    """
    return PreparedEvents(events=None, event_id=None)


def make_events(events: np.ndarray, event_id: Mapping[str, int]) -> PreparedEvents:
    """
    Convenience constructor with basic validation.

    Parameters
    ----------
    events
        Events array, must be shape (n_events, 3).
    event_id
        Mapping from names to integer codes.

    Returns
    -------
    PreparedEvents
        Validated container.

    Usage example
    -------------
        prepared = make_events(events, {"conversation_start": 1})
    """
    events_arr = np.asarray(events, dtype=int)
    if events_arr.ndim != 2 or events_arr.shape[1] != 3:
        raise ValueError(f"events must have shape (n_events, 3), got {events_arr.shape}")

    event_id_dict = {str(k): int(v) for k, v in event_id.items()}
    if len(event_id_dict) == 0:
        raise ValueError("event_id must not be empty when events are provided.")

    return PreparedEvents(events=events_arr, event_id=event_id_dict)
