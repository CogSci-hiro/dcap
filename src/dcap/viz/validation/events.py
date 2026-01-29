# =============================================================================
#                               Events Validation
# =============================================================================
"""Events Validation (skeleton).

This module will contain report-ready visualizations.
For now, it returns placeholder Matplotlib figures to validate end-to-end wiring.

Usage example
    from dcap.viz.overview import events as mod
    figures = mod.make_figures()
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def make_figures(*, extra_title: str | None = None) -> dict[str, Any]:
    figs: dict[str, Any] = {}
    fig = plt.figure()
    label = "Events Validation".strip()
    if extra_title:
        label = f"{label} | {extra_title}"
    fig.suptitle(label)
    figs["events_placeholder"] = fig
    return figs
