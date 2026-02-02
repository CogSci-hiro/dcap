# =============================================================================
#                     ########################################
#                     #            VIEW DEFINITIONS           #
#                     ########################################
# =============================================================================
"""Canonical 3D camera views for electrode localization plots."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ViewSpec:
    """Matplotlib 3D camera view specification."""

    name: str
    elev_deg: float
    azim_deg: float


# =============================================================================
#                     ########################################
#                     #              DEFAULT VIEWS            #
#                     ########################################
# =============================================================================
# Note: these are deterministic, fixed angles. Adjust once if you want a
# different "clinical look", but keep them stable thereafter.
TOP_VIEW = ViewSpec(name="Top", elev_deg=90.0, azim_deg=0.0)
FRONT_VIEW = ViewSpec(name="Front", elev_deg=10.0, azim_deg=90.0)
RIGHT_VIEW = ViewSpec(name="Right", elev_deg=10.0, azim_deg=0.0)
LEFT_VIEW = ViewSpec(name="Left", elev_deg=10.0, azim_deg=180.0)


VIEWS_2X2: Sequence[ViewSpec] = (TOP_VIEW, FRONT_VIEW, RIGHT_VIEW, LEFT_VIEW)
