# =============================================================================
#                        Patient Report (Skeleton)
# =============================================================================
"""
Patient-specific report builder.

This module defines the structure of patient reports.
It does NOT perform data loading or heavy computation.

Supported modes
----------------
- mode="clinical": clinical-facing report (IMPLEMENTED: structure only)
- mode="research": research-facing report (NOT IMPLEMENTED)

Design principles
-----------------
- One report == one patient
- Viz-ready tables are passed in
- All selection rules and parameters are explicit
- No CLI logic here

Usage example
-------------
from pathlib import Path
from dcap.viz.reports.patient import build_patient_report

build_patient_report(
    subject="sub-001",
    mode="clinical",
    tables=tables,
    out_dir=Path("out/sub-001"),
)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from dcap.viz.models import ReportArtifacts, ReportConfig
from dcap.viz import export


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

PatientReportMode = Literal["clinical", "research"]


def build_patient_report(
    *,
    subject: str,
    mode: PatientReportMode,
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
    config: ReportConfig | None = None,
) -> Path:
    """
    Build a patient-specific report.

    Parameters
    ----------
    subject
        Subject identifier (anonymized).
    mode
        Report mode. Must be "clinical" or "research".
    tables
        Dictionary of viz-ready tables required for report construction.
        Keys are agreed canonical names (e.g., inventory_df, sampling_df, etc.).
    out_dir
        Output directory for the report.
    config
        Optional report configuration. If None, defaults are used.

    Returns
    -------
    Path
        Path to the report output directory.

    Raises
    ------
    NotImplementedError
        If mode="research".
    """
    if mode == "clinical":
        return _build_clinical_patient_report(
            subject=subject,
            tables=tables,
            out_dir=out_dir,
            config=config,
        )

    if mode == "research":
        raise NotImplementedError(
            "Patient report mode='research' is not implemented yet."
        )

    raise ValueError(f"Unknown patient report mode: {mode!r}")


# -------------------------------------------------------------------------
# Clinical mode implementation (STRUCTURE ONLY)
# -------------------------------------------------------------------------


def _build_clinical_patient_report(
    *,
    subject: str,
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
    config: ReportConfig | None,
) -> Path:
    """
    Build a clinical-mode patient report.

    This function wires together all report sections but does not
    implement actual plotting or analysis logic yet.
    """
    cfg = config or ReportConfig(
        title=f"DCAP Clinical Patient Report — {subject}"
    )

    artifacts = ReportArtifacts()
    artifacts.manifest = _build_manifest(subject=subject, mode="clinical")

    # ------------------------------------------------------------------
    # Section 0 — Header / Cover
    # ------------------------------------------------------------------
    _add_header_section(artifacts, subject=subject)

    # ------------------------------------------------------------------
    # Section 1 — Patient Snapshot (Anonymized)
    # ------------------------------------------------------------------
    _add_patient_snapshot_section(artifacts, subject, tables)

    # ------------------------------------------------------------------
    # Section 2 — Common Preprocessing Summary
    # ------------------------------------------------------------------
    _add_common_preprocessing_section(artifacts, tables)

    # ------------------------------------------------------------------
    # Section 3+ — Clinical Task Sections
    # ------------------------------------------------------------------
    if _task_present(tables, "naming"):
        _add_naming_section(artifacts, subject, tables)

    if _task_present(tables, "sorciere"):
        _add_trf_task_section(
            artifacts,
            subject,
            tables,
            task="sorciere",
        )

    if _task_present(tables, "diapix"):
        _add_trf_task_section(
            artifacts,
            subject,
            tables,
            task="diapix",
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    figures_dir = export.ensure_dir(out_dir / "figures")
    tables_dir = export.ensure_dir(out_dir / "tables")

    figure_paths = _save_figures_placeholder(
        artifacts.figures, figures_dir
    )

    export.save_tables(artifacts.tables, tables_dir)
    export.save_summary(artifacts.summary, out_dir / "summary.json")
    export.save_manifest(artifacts.manifest, out_dir / "manifest.json")

    export.export_report_html(
        title=cfg.title,
        figure_paths=figure_paths,
        out_file=out_dir / "report.html",
    )

    return out_dir


# -------------------------------------------------------------------------
# Section builders (skeletons)
# -------------------------------------------------------------------------


def _add_header_section(
    artifacts: ReportArtifacts,
    *,
    subject: str,
) -> None:
    """Section 0 — Header / cover ribbon."""
    artifacts.summary.setdefault("subject", subject)
    artifacts.summary.setdefault("mode", "clinical")
    artifacts.summary.setdefault("status", "skeleton")


def _add_patient_snapshot_section(
    artifacts: ReportArtifacts,
    subject: str,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Section 1 — Patient snapshot (anonymized)."""
    # Expected input: clinical_subject_df (optional)
    df = tables.get("clinical_subject_df")
    if df is not None:
        artifacts.tables["patient_snapshot"] = df[df["subject"] == subject]


def _add_common_preprocessing_section(
    artifacts: ReportArtifacts,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Section 2 — Common preprocessing summary."""
    # Expected input: preprocessing_common_df (key/value)
    df = tables.get("preprocessing_common_df")
    if df is not None:
        artifacts.tables["preprocessing_common"] = df


def _add_naming_section(
    artifacts: ReportArtifacts,
    subject: str,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Clinical task: Naming (high gamma)."""
    # Placeholders only
    artifacts.figures["naming_hg_full"] = _placeholder_figure(
        f"Naming — High Gamma (Full) — {subject}"
    )
    artifacts.figures["naming_hg_selected"] = _placeholder_figure(
        f"Naming — Selected Electrodes — {subject}"
    )
    artifacts.figures["naming_hg_topography"] = _placeholder_figure(
        f"Naming — High Gamma Topography — {subject}"
    )


def _add_trf_task_section(
    artifacts: ReportArtifacts,
    subject: str,
    tables: dict[str, pd.DataFrame],
    *,
    task: str,
) -> None:
    """Clinical task: TRF-based (Sorcière, Diapix)."""
    prefix = f"{task}_trf"

    artifacts.figures[f"{prefix}_scores"] = _placeholder_figure(
        f"{task.capitalize()} — TRF Scores — {subject}"
    )
    artifacts.figures[f"{prefix}_topography"] = _placeholder_figure(
        f"{task.capitalize()} — TRF Topography — {subject}"
    )
    artifacts.figures[f"{prefix}_kernels"] = _placeholder_figure(
        f"{task.capitalize()} — TRF Kernels — {subject}"
    )


# -------------------------------------------------------------------------
# Utilities (skeleton)
# -------------------------------------------------------------------------


def _task_present(tables: dict[str, pd.DataFrame], task: str) -> bool:
    """Return True if task-specific tables appear to be present."""
    return any(key.startswith(task) for key in tables)


def _build_manifest(*, subject: str, mode: str) -> dict:
    """Build minimal manifest (expand later)."""
    return {
        "subject": subject,
        "mode": mode,
        "generator": "dcap.viz.reports.patient",
    }


def _placeholder_figure(title: str):
    """Create a tiny placeholder Matplotlib figure."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.suptitle(title)
    return fig


def _save_figures_placeholder(figures: dict, out_dir: Path) -> list[Path]:
    """Save figures, assuming Matplotlib or placeholders."""
    paths: list[Path] = []
    for name, fig in figures.items():
        out = out_dir / f"{name}.png"
        try:
            fig.savefig(out, dpi=150, bbox_inches="tight")
        except Exception:
            out.write_bytes(b"")
        paths.append(out)
    return paths
