"""Temporal Receptive Field (TRF) subpackage.

Public API
----------
- TemporalReceptiveField : MNE-like front door (fit/predict/score/save)
- read_trf : Load a saved TRF object (MNE-style reader)

Notes
-----
This folder is a refactor bundle generated for integration into the DCAP codebase.
"""

from .api import TemporalReceptiveField
from .io import read_trf

__all__ = ["TemporalReceptiveField", "read_trf"]
