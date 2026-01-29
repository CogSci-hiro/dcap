"""Visualization and reporting library for DCAP.

Notes
-----
- This subpackage contains *no* CLI parsing.
- All orchestration belongs in `dcap.cli`.
"""

from dcap.viz.api import (
    make_dataset_report,
    make_overview_bundle,
    make_patient_report,
    make_qc_report,
    make_task_report,
    make_validation_bundle,
)

__all__ = [
    "make_dataset_report",
    "make_task_report",
    "make_patient_report",
    "make_qc_report",
    "make_overview_bundle",
    "make_validation_bundle",
]
