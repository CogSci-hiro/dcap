# =============================================================================
#                     ########################################
#                     #      CLINICAL REPORT RENDERER MD     #
#                     ########################################
# =============================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from dcap.seeg.clinical.bundle import ClinicalAnalysisBundle
from dcap.seeg.clinical.report.assets import ReportAssetDirs, relpath_for_embed, write_placeholder_png
from dcap.seeg.clinical.report.base import ReportPaths, df_to_md


class MdClinicalReportRenderer:
    """
    Markdown clinical report renderer.

    Notes
    -----
    - Intended mostly for quick diffs/debugging.
    - HTML renderer is the default for clinical output.

    Usage example
    -------------
        renderer = MdClinicalReportRenderer()
        paths = renderer.render(bundle, Path("out"))
        print(paths.report_path)
    """

    def render(self, bundle: ClinicalAnalysisBundle, out_dir: Path) -> ReportPaths:
        asset_dirs = ReportAssetDirs.from_out_dir(out_dir)
        asset_dirs.ensure()

        report_path = out_dir / f"{bundle.subject_id}_clinical_report.md"

        decisions: Dict[str, Any] = bundle.preprocessing_context.decisions
        view_requested = decisions.get("analysis_view_requested", "original")
        view_used = decisions.get("analysis_view_used", "original")

        view_names = sorted(list(bundle.raw_views.keys()))
        envelope_names = sorted(list(bundle.envelopes.keys())) if bundle.envelopes else []

        prov_df = _provenance_table(bundle)
        warn_df = _warnings_table(bundle)

        md: List[str] = []
        md.append(f"# Clinical report — {bundle.subject_id}")
        md.append("")
        md.append("## Identifiers")
        md.append(f"- **Subject**: {bundle.subject_id}")
        md.append(f"- **Session**: {bundle.session_id or '(none)'}")
        md.append(f"- **Run**: {bundle.run_id or '(none)'}")
        md.append("")

        md.append("## Outputs produced")
        md.append(f"- **Raw views**: {', '.join(view_names) if view_names else '(none)'}")
        md.append(f"- **Envelopes**: {', '.join(envelope_names) if envelope_names else '(none)'}")
        md.append(f"- **TRF**: {'computed' if bundle.trf_result is not None else 'not computed'}")
        md.append("")

        md.append("## QC summary")
        md.extend(_render_qc_md(bundle))
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

        md.extend(_render_electrode_sections_md(bundle, asset_dirs=asset_dirs, report_dir=out_dir))
        md.append("")
        md.extend(_render_trf_section_md(bundle, asset_dirs=asset_dirs, report_dir=out_dir))
        md.append("")

        report_path.write_text("\n".join(md), encoding="utf-8")
        return ReportPaths(report_path=report_path, figures_dir=asset_dirs.figures_dir, tables_dir=asset_dirs.tables_dir)


def _provenance_table(bundle: ClinicalAnalysisBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in bundle.preprocessing_context.proc_history:
        rows.append({"step": str(item.get("step", "unknown")), "parameters": item.get("parameters", {})})
    return pd.DataFrame(rows, columns=["step", "parameters"])


def _warnings_table(bundle: ClinicalAnalysisBundle) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for artifact in bundle.preprocessing_artifacts:
        for w in artifact.warnings:
            rows.append({"step": artifact.name, "warning": str(w)})
    return pd.DataFrame(rows, columns=["step", "warning"])


def _recording_kv_to_md(rec: dict) -> str:
    lines: list[str] = []
    for k, v in rec.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) if lines else "_(none)_"


def _render_qc_md(bundle: ClinicalAnalysisBundle) -> List[str]:
    lines: List[str] = []
    if bundle.qc is None:
        lines.append("_(QC not computed)_")
        return lines

    lines.append("### Recording")
    lines.append(_recording_kv_to_md(dict(bundle.qc.recording)))
    lines.append("")
    lines.append("### Views")
    lines.append(df_to_md(bundle.qc.views))
    lines.append("")
    lines.append("### Channel QC (original)")
    if bundle.qc.channel_qc is None or bundle.qc.channel_qc.empty:
        lines.append("_(not computed)_")
        return lines

    flagged = bundle.qc.channel_qc.loc[
        (bundle.qc.channel_qc["is_flat"]) | (bundle.qc.channel_qc["is_outlier"]),
        ["channel", "variance", "log10_variance", "is_flat", "is_outlier"],
    ].copy()
    lines.append(df_to_md(flagged))
    return lines


def _render_electrode_sections_md(
    bundle: ClinicalAnalysisBundle,
    *,
    asset_dirs: ReportAssetDirs,
    report_dir: Path,
) -> List[str]:
    """
    Electrode list + localization plot placeholders.

    Expected future inputs
    ----------------------
    - bundle.electrodes_df: DataFrame with at least:
        - "name"
        - "x", "y", "z" (MNI or patient-space, whichever you standardize on)

    Example electrodes_df
    ---------------------
    | name | x     | y     | z     |
    |------|-------|-------|-------|
    | LA1  | -34.2 | -12.0 |  18.5 |
    | LA2  | -33.7 | -10.9 |  16.9 |
    """
    lines: List[str] = []
    lines.append("## Electrodes")
    lines.append("")

    electrodes_df: Optional[pd.DataFrame] = getattr(bundle, "electrodes_df", None)
    if electrodes_df is None or electrodes_df.empty:
        lines.append("_(electrode table not available)_")
    else:
        if "name" in electrodes_df.columns:
            names = electrodes_df["name"].astype(str).tolist()
            lines.append("### Electrode list")
            lines.append(_compact_name_list_md(names))
            lines.append("")
        else:
            lines.append("_(electrode table missing 'name' column)_")

    # Placeholder 3D localization plot
    fig_path = asset_dirs.figures_dir / "electrodes_3d.png"
    write_placeholder_png(fig_path)

    lines.append("### Electrode localization (3D)")
    lines.append(_embed_md_png(fig_path, report_dir=report_dir, alt="Electrodes 3D"))
    return lines


def _render_trf_section_md(
    bundle: ClinicalAnalysisBundle,
    *,
    asset_dirs: ReportAssetDirs,
    report_dir: Path,
) -> List[str]:
    trf = bundle.trf_result
    if trf is None:
        return ["## TRF analysis", "", "_(TRF not computed)_"]

    lines: List[str] = []
    lines.append("## TRF analysis")
    lines.append("")

    analysis_view = bundle.preprocessing_context.decisions.get("analysis_view_used", "unknown")
    lines.append(f"- Analysis view: {analysis_view}")
    lines.append(f"- Model: {trf.model_name}")

    lag_cfg = (trf.extra.get("lag_config", {}) or {}) if hasattr(trf, "extra") else {}
    tmin_ms = lag_cfg.get("tmin_ms")
    tmax_ms = lag_cfg.get("tmax_ms")
    if tmin_ms is not None and tmax_ms is not None:
        lines.append(f"- Lags: {tmin_ms} … {tmax_ms} ms")

    alpha = trf.extra.get("alpha") if hasattr(trf, "extra") else None
    if alpha is not None:
        lines.append(f"- Regularization (alpha): {alpha}")

    lines.append("")

    if getattr(trf, "metrics", None):
        lines.append("### Summary metrics")
        for name, value in trf.metrics.items():
            lines.append(f"- {name}: {value:.4g}")
        lines.append("")

    # ---------------------------------------------------------------------
    # Score table (DataFrame + saved TSV path placeholder)
    # ---------------------------------------------------------------------
    score_df: Optional[pd.DataFrame] = trf.extra.get("score_df") if hasattr(trf, "extra") else None
    score_table_path = trf.extra.get("score_table_path") if hasattr(trf, "extra") else None

    if score_df is not None and not score_df.empty:
        out_tsv = asset_dirs.tables_dir / "trf_scores.tsv"
        score_df.to_csv(out_tsv, sep="\t", index=False)
        score_table_path = str(out_tsv)

        lines.append("### TRF scores table")
        lines.append(f"(Saved table: {score_table_path})")
        lines.append(df_to_md(score_df))
        lines.append("")
    elif score_table_path:
        lines.append("### TRF scores table")
        lines.append(f"(Saved table: {score_table_path})")
        lines.append("")

    # ---------------------------------------------------------------------
    # Placeholder figures (viz package will populate later)
    # ---------------------------------------------------------------------
    fig_scores_3d = asset_dirs.figures_dir / "trf_scores_3d.png"
    fig_kernel = asset_dirs.figures_dir / "trf_kernel.png"
    fig_scores_bar = asset_dirs.figures_dir / "trf_scores_bar.png"

    write_placeholder_png(fig_scores_3d)
    write_placeholder_png(fig_kernel)
    write_placeholder_png(fig_scores_bar)

    lines.append("### TRF score localization (3D)")
    lines.append(_embed_md_png(fig_scores_3d, report_dir=report_dir, alt="TRF scores 3D"))
    lines.append("")

    lines.append("### TRF kernel")
    lines.append(_embed_md_png(fig_kernel, report_dir=report_dir, alt="TRF kernel"))
    lines.append("")

    lines.append("### TRF scores across channels")
    lines.append(_embed_md_png(fig_scores_bar, report_dir=report_dir, alt="TRF scores bar plot"))
    lines.append("")

    warnings = trf.extra.get("warnings", []) if hasattr(trf, "extra") else []
    if warnings:
        lines.append("### TRF notes / warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return lines


def _compact_name_list_md(names: List[str], *, max_chars: int = 1400) -> str:
    """
    Render a compact, wrapped electrode name list.

    Usage example
    -------------
        text = _compact_name_list_md(["LA1", "LA2", "LB1"])
    """
    joined = ", ".join(names)
    if len(joined) <= max_chars:
        return joined
    # Hard fallback: show head/tail
    head = ", ".join(names[:30])
    tail = ", ".join(names[-10:])
    return f"{head}, …, {tail}  _(total: {len(names)})_"


def _embed_md_png(path: Path, *, report_dir: Path, alt: str = "") -> str:
    rel = relpath_for_embed(path, base_dir=report_dir)
    return f"![{alt}]({rel})"
