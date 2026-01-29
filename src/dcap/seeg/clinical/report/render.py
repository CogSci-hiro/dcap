from pathlib import Path
from typing import List

import pandas as pd

from dcap.seeg.clinical.bundle import ClinicalAnalysisBundle


def _provenance_table(bundle: ClinicalAnalysisBundle) -> pd.DataFrame:
    rows = []
    for item in bundle.preprocessing_context.proc_history:
        rows.append({"step": str(item.get("step", "unknown")), "parameters": item.get("parameters", {})})
    return pd.DataFrame(rows, columns=["step", "parameters"])


def _warnings_table(bundle: ClinicalAnalysisBundle) -> pd.DataFrame:
    rows = []
    for artifact in bundle.preprocessing_artifacts:
        for w in artifact.warnings:
            rows.append({"step": artifact.name, "warning": str(w)})
    return pd.DataFrame(rows, columns=["step", "warning"])


def render_report_v0(bundle: ClinicalAnalysisBundle, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = bundle.preprocessing_context.decisions
    view_requested = decisions.get("analysis_view_requested", "original")
    view_used = decisions.get("analysis_view_used", "original")

    view_names = sorted(list(bundle.raw_views.keys()))
    envelope_names = sorted(list(bundle.envelopes.keys())) if bundle.envelopes else []

    prov_df = _provenance_table(bundle)
    warn_df = _warnings_table(bundle)

    report_path = out_dir / f"{bundle.subject_id}_clinical_report.md"

    def df_to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_(none)_"
        return df.to_markdown(index=False)

    md: List[str] = []
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
    md.append(f"- **TRF**: {'computed' if bundle.trf_result is not None else 'not computed'}")
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
