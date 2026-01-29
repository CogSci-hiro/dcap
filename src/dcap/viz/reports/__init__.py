# =============================================================================
#                                Viz Reports
# =============================================================================
"""Report builders for DCAP visualizations.

Notes
-----
- Report builders accept **viz-ready tables** and produce exported artifacts.
- CLI orchestration and table construction belong elsewhere (dcap.cli / pipeline).

Usage example
    from dcap.viz.reports.patient import build_patient_report
"""

from dcap.viz.reports.patient import build_patient_report
from dcap.viz.reports.patient_spec import PatientClinicalReportSpec

__all__ = ["build_patient_report", "PatientClinicalReportSpec"]
