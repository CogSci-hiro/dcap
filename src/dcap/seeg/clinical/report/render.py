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


def _df_to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(none)_"
    return df.to_markdown(index=False)


def _recording_kv_to_md(rec: dict) -> str:
    lines = []
    for k, v in rec.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) if lines else "_(none)_"


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
    md.append("## QC summary")

    if bundle.qc is None:
        md.append("_(QC not computed)_")
    else:
        md.append("### Recording")
        md.append(_recording_kv_to_md(dict(bundle.qc.recording)))
        md.append("")
        md.append("### Views")
        md.append(_df_to_md(bundle.qc.views))
        md.append("")
        md.append("### Channel QC (original)")
        if bundle.qc.channel_qc is None:
            md.append("_(not computed)_")
        else:
            flagged = bundle.qc.channel_qc.loc[
                (bundle.qc.channel_qc["is_flat"]) | (bundle.qc.channel_qc["is_outlier"]),
                ["channel", "variance", "log10_variance", "is_flat", "is_outlier"],
            ].copy()
            md.append(_df_to_md(flagged))
        md.append("")
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
