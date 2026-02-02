# =============================================================================
#                     ########################################
#                     #            VIEW DEFINITIONS           #
#                     ########################################
# =============================================================================
"""Canonical MNE 3D camera views for electrode montage snapshots."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MNEViewSpec:
    """MNE-compatible 3D view parameters."""

    name: str
    azimuth: float
    elevation: float
    roll: float


TOP_VIEW = MNEViewSpec(name="Top", azimuth=0.0, elevation=0.0, roll=0.0)
FRONT_VIEW = MNEViewSpec(name="Front", azimuth=90.0, elevation=90.0, roll=75.0)
RIGHT_VIEW = MNEViewSpec(name="Right", azimuth=0.0, elevation=90.0, roll=-90.0)
LEFT_VIEW = MNEViewSpec(name="Left", azimuth=0.0, elevation=-90.0, roll=90.0)

VIEWS_2X2: Sequence[MNEViewSpec] = (TOP_VIEW, FRONT_VIEW, RIGHT_VIEW, LEFT_VIEW)
