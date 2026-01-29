# =============================================================================
# =============================================================================
#                     ########################################
#                     #     CONFIG: COORDINATES / GEOMETRY   #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class CoordinatesConfig:
    """
    Configuration for attaching electrode coordinates and computing geometry helpers.

    Attributes
    ----------
    unit
        Coordinate unit. Use "mm" for typical clinical exports or "m" for MNE-native.
    coord_frame
        Human-readable coordinate frame label (e.g., "mni", "tkr", "native").
        This is NOT strict validation; it is recorded for provenance and reporting.
    compute_neighbors
        If True, compute a neighbor graph (kNN with optional radius constraint).
    neighbors_k
        Number of nearest neighbors per contact (k in kNN).
    neighbors_radius_mm
        Optional radius constraint in millimeters. If provided, neighbors farther
        than this radius are excluded (even if fewer than k remain).

    Usage example
    -------------
        cfg = CoordinatesConfig(
            unit="mm",
            coord_frame="mni",
            compute_neighbors=True,
            neighbors_k=6,
            neighbors_radius_mm=12.0,
        )
    """

    unit: Literal["mm", "m"] = "mm"
    coord_frame: str = "unknown"

    compute_neighbors: bool = False
    neighbors_k: int = 6
    neighbors_radius_mm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.neighbors_k <= 0:
            raise ValueError(f"neighbors_k must be > 0, got {self.neighbors_k}.")

        if self.neighbors_radius_mm is not None and self.neighbors_radius_mm <= 0:
            raise ValueError(
                f"neighbors_radius_mm must be > 0 when provided, got {self.neighbors_radius_mm}."
            )
