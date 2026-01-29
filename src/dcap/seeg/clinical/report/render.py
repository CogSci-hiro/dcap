from __future__ import annotations  # remove if you don't want; not required here

from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from dcap.seeg.clinical.bundle import ClinicalAnalysisBundle


def _bundle_provenance_table(bundle: ClinicalAnalysisBundle) -> pd.DataFrame:
    ctx = bundle.preprocessing_context

    # Adjust this line if your context uses a different field name
    proc_history = getattr(ctx, "proc_history", [])

    rows = []
    for item in proc_history:
        # Be flexible: item might be dict or a small record object
        if isinstance(item, dict):
            step = str(item.get("step", item.get("name", "unknown")))
            params = item.get("parameters", item.get("params", {}))
        else:
            step = str(getattr(item, "step", getattr(item, "name", "unknown")))
            params = getattr(item, "parameters", getattr(item, "params", {}))

        rows.append({"step": step, "parameters": params})

    return pd.DataFrame(rows, columns=["step", "parameters"])


def _bundle_warnings_table(bundle: ClinicalAnalysisBundle) -> pd.DataFrame:
    rows = []
    for artifact in bundle.preprocessing_artifacts:
        step = getattr(artifact, "name", "unknown")
        warnings = getattr(artifact, "warnings", []) or []
        for w in warnings:
            rows.append({"step": step, "warning": str(w)})

    return pd.DataFrame(rows, columns=["step", "warning"])


def render_report_v0(bundle: ClinicalAnalysisBundle, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Identify analysis view policy (from ctx.decisions) --------
    decisions = getattr(bundle.preprocessing_context, "decisions", {}) or {}
    view_requested = decisions.get("analysis_view_requested", "original")
    view_used = decisions.get("analysis_view_used", "original")

    # -------- Describe produced outputs --------
    view_names = sorted(list(bundle.raw_views.keys()))
    envelope_names = sorted(list(bundle.envelopes.keys())) if bundle.envelopes else []

    # -------- Tables --------
    prov_df = _bundle_provenance_table(bundle)
    warn_df = _bundle_warnings_table(bundle)

    # -------- Minimal TRF status --------
    trf_status = "computed" if bundle.trf_result is not None else "not computed"

    # -------- Write Markdown --------
    report_path = out_dir / f"{bundle.subject_id}_clinical_report.md"

    def df_to_md(df: pd.DataFrame, max_rows: int = 50) -> str:
        if df.empty:
            return "_(none)_"
        return df.head(max_rows).to_markdown(index=False)

    md = []
    md.append(f"# Clinical report — {bundle.subject_id}")
    md.append("")
    md.append("## Identifiers")
    md.append(f"- **Subject**: {bundle.subject_id}")
    md.append(f"- **Session**: {bundle.session_id or '(none)'}")
    md.append(f"- **Run**: {bundle.run_id or '(none)'}")
    md.append("")
    md.append("## Outputs produced")
    md.append(f"- **Raw views**: {', '.join(view_names)}")
    md.append(f"- **Envelopes**: {', '.join(envelope_names) if envelope_names else '(none)'}")
    md.append(f"- **TRF**: {trf_status}")
    md.append("")
    md.append("## Analysis view policy")
    md.append(f"- **Requested**: {view_requested}")
    md.append(f"- **Used**: {view_used}")
    md.append("")
    md.append("## Preprocessing provenance")
    md.append(df_to_md(prov_df))
    md.append("")
    md.append("## Warnings")
    md.append(df_to_md(warn_df))
    md.append("")

    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path
