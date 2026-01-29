# =============================================================================
#                           Viz Models and Schemas
# =============================================================================
"""Data structures passed into `dcap.viz`.

The visualization layer should operate on **viz-ready tables** rather than reading raw files.
This module defines minimal dataclasses for report inputs and outputs.

Usage example
    from dcap.viz.models import ReportArtifacts
    artifacts = ReportArtifacts(figures={}, tables={}, summary={})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for report generation."""

    title: str
    include_manifest: bool = True
    include_summary_json: bool = True
    output_format: str = "html"  # {"html", "pdf"} (pdf is optional)


@dataclass
class ReportArtifacts:
    """Container for generated report artifacts."""

    figures: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def add_table(self, name: str, table: pd.DataFrame) -> None:
        self.tables[name] = table
