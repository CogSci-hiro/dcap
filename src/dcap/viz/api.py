# =============================================================================
#                           Viz Public API (Library)
# =============================================================================
"""Public API for generating figure bundles and reports.

Notes
-----
- This module is called by `dcap.cli` commands.
- In this skeleton, we generate placeholder artifacts to validate wiring.

Usage example
    from pathlib import Path
    from dcap.viz.api import make_dataset_report
    make_dataset_report(out_dir=Path("./out"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dcap.viz import export
from dcap.viz.models import ReportArtifacts, ReportConfig
from dcap.viz.overview import inventory as inventory_mod
from dcap.viz.overview import missingness as missingness_mod
from dcap.viz.overview import timeline as timeline_mod
from dcap.viz.validation import annotations as annotations_mod
from dcap.viz.validation import channels as channels_mod
from dcap.viz.validation import events as events_mod
from dcap.viz.validation import file_integrity as integrity_mod
from dcap.viz.validation import sampling as sampling_mod
from dcap.viz.minimal_analyses import trf_baseline as trf_mod
from dcap.viz.minimal_analyses import clinical as clinical_mod


def make_overview_bundle(*, out_dir: Path) -> Path:
    """Generate overview figure bundle (inventory + timeline + missingness)."""
    artifacts = ReportArtifacts()
    artifacts.figures |= inventory_mod.make_figures()
    artifacts.figures |= timeline_mod.make_figures()
    artifacts.figures |= missingness_mod.make_figures()

    return _export_bundle(artifacts, title="DCAP Overview Bundle", out_dir=out_dir)


def make_validation_bundle(*, out_dir: Path) -> Path:
    """Generate validation figure bundle (integrity + sampling + channels + events + annotations)."""
    artifacts = ReportArtifacts()
    artifacts.figures |= integrity_mod.make_figures()
    artifacts.figures |= sampling_mod.make_figures()
    artifacts.figures |= channels_mod.make_figures()
    artifacts.figures |= events_mod.make_figures()
    artifacts.figures |= annotations_mod.make_figures()

    return _export_bundle(artifacts, title="DCAP Validation Bundle", out_dir=out_dir)


def make_dataset_report(*, out_dir: Path) -> Path:
    artifacts = ReportArtifacts()
    artifacts.figures |= inventory_mod.make_figures()
    artifacts.figures |= timeline_mod.make_figures()
    artifacts.figures |= missingness_mod.make_figures()
    artifacts.figures |= integrity_mod.make_figures()
    artifacts.figures |= sampling_mod.make_figures()

    artifacts.summary = {
        "scope": "dataset",
        "status": "skeleton",
    }
    return _export_report(artifacts, title="DCAP Dataset Report", out_dir=out_dir)


def make_task_report(*, task: str, out_dir: Path) -> Path:
    artifacts = ReportArtifacts()
    artifacts.figures |= inventory_mod.make_figures(extra_title=f"Task={task}")
    artifacts.figures |= timeline_mod.make_figures(extra_title=f"Task={task}")
    artifacts.figures |= events_mod.make_figures(extra_title=f"Task={task}")

    artifacts.summary = {"scope": "task", "task": task, "status": "skeleton"}
    return _export_report(artifacts, title=f"DCAP Task Report: {task}", out_dir=out_dir)


def make_patient_report(*, subject: str, out_dir: Path, mode: str = "clinical") -> Path:
    """Generate a patient report.

    Notes
    -----
    - Thin wrapper delegating to `dcap.viz.reports.patient`.
    - In this skeleton, viz-ready input tables are not constructed yet.

    Parameters
    ----------
    subject
        Subject ID (e.g., "sub-001").
    out_dir
        Output directory.
    mode
        "clinical" or "research" (research is not implemented yet).

    Returns
    -------
    Path
        Output directory.
    """
    from dcap.viz.reports.patient import build_patient_report

    tables: dict[str, pd.DataFrame] = {}
    return build_patient_report(subject=subject, mode=mode, tables=tables, out_dir=out_dir)


    artifacts.summary = {"scope": "patient", "subject": subject, "status": "skeleton"}
    return _export_report(artifacts, title=f"DCAP Patient Report: {subject}", out_dir=out_dir)


def make_qc_report(*, out_dir: Path) -> Path:
    artifacts = ReportArtifacts()
    artifacts.figures |= integrity_mod.make_figures()
    artifacts.figures |= sampling_mod.make_figures()
    artifacts.figures |= channels_mod.make_figures()
    artifacts.figures |= events_mod.make_figures()
    artifacts.figures |= annotations_mod.make_figures()
    artifacts.figures |= trf_mod.make_figures()

    artifacts.summary = {"scope": "qc", "status": "skeleton"}
    return _export_report(artifacts, title="DCAP QC Report", out_dir=out_dir)


def _export_bundle(artifacts: ReportArtifacts, *, title: str, out_dir: Path) -> Path:
    figures_dir = export.ensure_dir(out_dir / "figures")
    figure_paths = _save_figures(artifacts.figures, figures_dir)
    export.export_report_html(title=title, figure_paths=figure_paths, out_file=out_dir / "bundle.html")
    return out_dir


def _export_report(artifacts: ReportArtifacts, *, title: str, out_dir: Path) -> Path:
    figures_dir = export.ensure_dir(out_dir / "figures")
    tables_dir = export.ensure_dir(out_dir / "tables")

    figure_paths = _save_figures(artifacts.figures, figures_dir)
    export.save_tables(artifacts.tables, tables_dir)
    export.save_summary(artifacts.summary, out_dir / "summary.json")
    export.save_manifest(artifacts.manifest, out_dir / "manifest.json")
    export.export_report_html(title=title, figure_paths=figure_paths, out_file=out_dir / "report.html")
    return out_dir


def _save_figures(figures: dict[str, Any], out_dir: Path) -> list[Path]:
    """Save matplotlib figures if present; otherwise write placeholders."""
    paths: list[Path] = []
    for name, fig in figures.items():
        out_path = out_dir / f"{name}.png"
        try:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        except Exception:
            # If fig isn't a Matplotlib figure in this skeleton, write a tiny placeholder file.
            out_path.write_bytes(b"")
        paths.append(out_path)
    return paths
