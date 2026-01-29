# =============================================================================
#                       Patient Clinical Report Specification
# =============================================================================
"""Specification objects for patient reports.

This module defines the structure of the *clinical* patient report as a declarative spec.
The spec is used to validate inputs and to provide deterministic IDs for figures and tables.

Usage example
    from dcap.viz.reports.patient_spec import PatientClinicalReportSpec

    spec = PatientClinicalReportSpec.default()
    missing = spec.validate_input_tables(tables)
    if missing:
        print("Missing tables:", missing)
"""

from dataclasses import dataclass, field
from typing import Literal


ReportMode = Literal["clinical", "research"]


@dataclass(frozen=True)
class FigureSpec:
    """A single figure artifact defined by a stable ID."""

    figure_id: str
    title: str
    section_id: str


@dataclass(frozen=True)
class TableSpec:
    """A single table artifact defined by a stable ID."""

    table_id: str
    title: str
    section_id: str
    required: bool = False


@dataclass(frozen=True)
class SectionSpec:
    """A report section containing figures and tables."""

    section_id: str
    title: str
    figures: tuple[FigureSpec, ...] = ()
    tables: tuple[TableSpec, ...] = ()


@dataclass(frozen=True)
class PatientClinicalReportSpec:
    """Declarative specification for the clinical patient report.

    Notes
    -----
    - Clinical mode is always patient-specific.
    - The spec focuses on *structure* and stable artifact naming.
    - Task inclusion can be dynamic at runtime (depending on available tables).

    Usage example
        spec = PatientClinicalReportSpec.default()
        for section in spec.sections:
            print(section.section_id, section.title)
    """

    mode: ReportMode = "clinical"
    clinical_tasks: tuple[str, ...] = ("naming", "sorciere", "diapix")

    # These are "expected" inputs. In early development, callers may choose not to provide them.
    # Use validate_input_tables() to generate warnings or enforce strict mode.
    required_table_keys: tuple[str, ...] = (
        "clinical_subject_df",
        "preprocessing_common_df",
        "file_integrity_df",
        "sampling_df",
        "channels_df",
        "events_df",
        "annotations_df",
    )

    sections: tuple[SectionSpec, ...] = field(default_factory=tuple)

    @staticmethod
    def default() -> "PatientClinicalReportSpec":
        """Create the default clinical patient report specification."""
        sections: list[SectionSpec] = []

        sections.append(SectionSpec(section_id="header", title="Header"))

        sections.append(
            SectionSpec(
                section_id="patient_snapshot",
                title="Patient Snapshot (Anonymized)",
                figures=(
                    FigureSpec(
                        figure_id="patient_minutes_per_task",
                        title="Minutes per task",
                        section_id="patient_snapshot",
                    ),
                ),
                tables=(
                    TableSpec(
                        table_id="patient_snapshot",
                        title="Patient snapshot table",
                        section_id="patient_snapshot",
                    ),
                    TableSpec(
                        table_id="patient_task_participation",
                        title="Task participation (runs × minutes)",
                        section_id="patient_snapshot",
                    ),
                    TableSpec(
                        table_id="sensitive_field_leakage",
                        title="Sensitive field leakage checks",
                        section_id="patient_snapshot",
                    ),
                ),
            )
        )

        sections.append(
            SectionSpec(
                section_id="common_preprocessing",
                title="Common Preprocessing Summary",
                tables=(
                    TableSpec(
                        table_id="preprocessing_common",
                        title="Common preprocessing (key/value)",
                        section_id="common_preprocessing",
                    ),
                ),
            )
        )

        sections.append(
            SectionSpec(
                section_id="task_naming",
                title="Task: Naming (High Gamma)",
                figures=(
                    FigureSpec(
                        figure_id="naming_hg_full",
                        title="High gamma activity per electrode (full)",
                        section_id="task_naming",
                    ),
                    FigureSpec(
                        figure_id="naming_hg_selected",
                        title="Selected electrodes (details)",
                        section_id="task_naming",
                    ),
                    FigureSpec(
                        figure_id="naming_hg_topography",
                        title="High gamma topography (3D)",
                        section_id="task_naming",
                    ),
                ),
                tables=(
                    TableSpec(
                        table_id="naming_inputs",
                        title="Naming inputs and run inclusion",
                        section_id="task_naming",
                    ),
                    TableSpec(
                        table_id="naming_preprocessing",
                        title="Naming task preprocessing (key/value)",
                        section_id="task_naming",
                    ),
                    TableSpec(
                        table_id="naming_hg_summary",
                        title="High gamma electrode summary",
                        section_id="task_naming",
                    ),
                    TableSpec(
                        table_id="naming_top_electrodes",
                        title="Top electrodes by high gamma metric",
                        section_id="task_naming",
                    ),
                ),
            )
        )

        for task in ("sorciere", "diapix"):
            section_id = f"task_{task}"
            sections.append(
                SectionSpec(
                    section_id=section_id,
                    title=f"Task: {task.capitalize()} (TRF Sanity)",
                    figures=(
                        FigureSpec(
                            figure_id=f"{task}_trf_scores",
                            title="TRF score plots",
                            section_id=section_id,
                        ),
                        FigureSpec(
                            figure_id=f"{task}_trf_topography",
                            title="TRF score topography (3D)",
                            section_id=section_id,
                        ),
                        FigureSpec(
                            figure_id=f"{task}_trf_kernels",
                            title="TRF kernels for selected channels",
                            section_id=section_id,
                        ),
                    ),
                    tables=(
                        TableSpec(
                            table_id=f"{task}_trf_scores",
                            title="TRF scores (table)",
                            section_id=section_id,
                        ),
                        TableSpec(
                            table_id=f"{task}_trf_lag_curves",
                            title="TRF lag curves",
                            section_id=section_id,
                        ),
                        TableSpec(
                            table_id=f"{task}_trf_top_channels",
                            title="Top channels by TRF score",
                            section_id=section_id,
                        ),
                        TableSpec(
                            table_id=f"{task}_predictor_qc",
                            title="Predictor QC summary",
                            section_id=section_id,
                        ),
                    ),
                )
            )

        return PatientClinicalReportSpec(sections=tuple(sections))

    def validate_input_tables(self, tables: dict) -> list[str]:
        """Return a list of missing expected input table keys."""
        missing: list[str] = []
        for key in self.required_table_keys:
            if key not in tables:
                missing.append(key)
        return missing

    def all_figure_ids(self) -> tuple[str, ...]:
        """Return all figure IDs declared in the spec."""
        ids: list[str] = []
        for section in self.sections:
            ids.extend([f.figure_id for f in section.figures])
        return tuple(ids)

    def all_table_ids(self) -> tuple[str, ...]:
        """Return all table IDs declared in the spec."""
        ids: list[str] = []
        for section in self.sections:
            ids.extend([t.table_id for t in section.tables])
        return tuple(ids)
