# =============================================================================
#                     Patient Report Builder (Clinical Mode)
# =============================================================================
"""Patient-specific report builder.

This module wires the *clinical patient report* using a declarative specification.
It is responsible for section assembly, placeholder figure creation, and export wiring.
It does not perform heavy computations (HG/TRF computation is upstream).

Usage example
    from pathlib import Path
    import pandas as pd
    from dcap.viz.reports.patient import build_patient_report

    build_patient_report(
        subject="sub-001",
        mode="clinical",
        tables={},
        out_dir=Path("./out/sub-001"),
    )
"""

from pathlib import Path
from typing import Any

import pandas as pd

from dcap.viz import export
from dcap.viz.models import ReportArtifacts, ReportConfig
from dcap.viz.reports.patient_spec import PatientClinicalReportSpec, ReportMode


def build_patient_report(
    *,
    subject: str,
    mode: ReportMode,
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
    config: ReportConfig | None = None,
    strict: bool = False,
) -> Path:
    """Build a patient-specific report.

    Parameters
    ----------
    subject
        Subject identifier (anonymized).
    mode
        Report mode. "clinical" is implemented structurally; "research" is not implemented.
    tables
        Viz-ready tables required for report construction.
        This skeleton tolerates missing tables unless `strict=True`.
    out_dir
        Output directory for the report.
    config
        Optional report configuration.
    strict
        If True, missing expected inputs raise an error.

    Returns
    -------
    Path
        Output directory path.

    Raises
    ------
    NotImplementedError
        If mode="research".
    ValueError
        If strict validation fails or unknown mode is provided.
    """
    if mode == "research":
        raise NotImplementedError("mode='research' is not implemented yet.")

    if mode != "clinical":
        raise ValueError(f"Unknown mode: {mode!r}")

    spec = PatientClinicalReportSpec.default()
    missing = spec.validate_input_tables(tables)
    if missing and strict:
        raise ValueError(f"Missing expected input tables: {missing}")

    cfg = config or ReportConfig(title=f"DCAP Clinical Patient Report — {subject}")

    artifacts = ReportArtifacts()
    artifacts.summary = {
        "subject": subject,
        "mode": mode,
        "status": "skeleton",
        "missing_inputs": missing,
    }
    artifacts.manifest = {
        "subject": subject,
        "mode": mode,
        "generator": "dcap.viz.reports.patient",
        "spec": {
            "sections": [s.section_id for s in spec.sections],
            "figure_ids": list(spec.all_figure_ids()),
            "table_ids": list(spec.all_table_ids()),
        },
    }

    # -------------------------------------------------------------------------
    # Section 0 — Header / Cover
    # -------------------------------------------------------------------------
    artifacts.figures["header_placeholder"] = _placeholder_figure(f"Header — {subject}")

    # -------------------------------------------------------------------------
    # Section 1 — Patient Snapshot
    # -------------------------------------------------------------------------
    artifacts.figures["patient_minutes_per_task"] = _placeholder_figure(f"Minutes per task — {subject}")
    clinical_subject = tables.get("clinical_subject_df")
    if clinical_subject is not None and "subject" in clinical_subject.columns:
        artifacts.tables["patient_snapshot"] = clinical_subject[clinical_subject["subject"] == subject].copy()

    # -------------------------------------------------------------------------
    # Section 2 — Common preprocessing
    # -------------------------------------------------------------------------
    preprocessing = tables.get("preprocessing_common_df")
    if preprocessing is not None:
        artifacts.tables["preprocessing_common"] = preprocessing.copy()

    # -------------------------------------------------------------------------
    # Section 3 — Naming task (HG)
    # -------------------------------------------------------------------------
    artifacts.figures["naming_hg_full"] = _placeholder_figure(f"Naming — High Gamma (Full) — {subject}")
    artifacts.figures["naming_hg_selected"] = _placeholder_figure(f"Naming — Selected Electrodes — {subject}")
    artifacts.figures["naming_hg_topography"] = _placeholder_figure(f"Naming — High Gamma Topography (3D) — {subject}")

    scores = tables.get("naming_inputs_df")
    if scores is not None:
        artifacts.tables["naming_inputs"] = scores.copy()

    # -------------------------------------------------------------------------
    # Section 4 — Sorciere (TRF)
    # -------------------------------------------------------------------------
    _add_trf_task_placeholders(artifacts, subject=subject, task="sorciere", tables=tables)

    # -------------------------------------------------------------------------
    # Section 5 — Diapix (TRF)
    # -------------------------------------------------------------------------
    _add_trf_task_placeholders(artifacts, subject=subject, task="diapix", tables=tables)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    figures_dir = export.ensure_dir(out_dir / "figures")
    tables_dir = export.ensure_dir(out_dir / "tables")

    figure_paths = _save_figures(artifacts.figures, figures_dir)
    export.save_tables(artifacts.tables, tables_dir)
    export.save_summary(artifacts.summary, out_dir / "summary.json")
    export.save_manifest(artifacts.manifest, out_dir / "manifest.json")
    export.export_report_html(title=cfg.title, figure_paths=figure_paths, out_file=out_dir / "report.html")

    return out_dir


# =============================================================================
#                                 Utilities
# =============================================================================

def _add_trf_task_placeholders(
    artifacts: ReportArtifacts,
    *,
    subject: str,
    task: str,
    tables: dict[str, pd.DataFrame],
) -> None:
    artifacts.figures[f"{task}_trf_scores"] = _placeholder_figure(f"{task.capitalize()} — TRF Scores — {subject}")
    artifacts.figures[f"{task}_trf_topography"] = _placeholder_figure(f"{task.capitalize()} — TRF Topography (3D) — {subject}")
    artifacts.figures[f"{task}_trf_kernels"] = _placeholder_figure(f"{task.capitalize()} — TRF Kernels — {subject}")

    task_scores = tables.get(f"{task}_trf_scores_df")
    if task_scores is not None:
        artifacts.tables[f"{task}_trf_scores"] = task_scores.copy()


def _placeholder_figure(title: str) -> Any:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.suptitle(title)
    return fig


def _save_figures(figures: dict[str, Any], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for name, fig in figures.items():
        out_path = out_dir / f"{name}.png"
        try:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        except Exception:
            out_path.write_bytes(b"")
        paths.append(out_path)
    return paths
