"""
dcap: Data & Clinical Analysis Platform (sEEG)

This package provides infrastructure-level tooling for:
- BIDS conversion (task-specific)
- Validation and QC
- Metadata registry (public + private layers)
- Preprocessing primitives
- Visualization helpers
- Analysis primitives (machinery only)

Note
----
Scientific decisions belong in downstream analysis projects.

"""
from importlib.metadata import version as _pkg_version

__all__ = ["__version__"]

__version__ = _pkg_version("dcap")
