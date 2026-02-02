# =============================================================================
#                               Viz Styling
# =============================================================================
"""Centralized styling defaults for all DCAP visualizations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleConfig:
    """Global figure styling configuration."""

    dpi: int = 150
    figure_width_in: float = 10.0
    figure_height_in: float = 6.0
    font_size: int = 10


DEFAULT_STYLE = StyleConfig()
