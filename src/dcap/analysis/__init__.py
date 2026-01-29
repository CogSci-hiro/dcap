"""dcap.analysis

Core analysis algorithms that operate on preprocessed, analysis-ready data.

Design goals
------------
- Pure computation: no figures, no HTML, no file-layout assumptions.
- Stable, typed APIs callable from CLI and visualization layers.

Notes
-----
Visualization/reporting code belongs in :mod:`dcap.viz`.
"""

from dcap.analysis import trf

__all__ = ["trf"]
