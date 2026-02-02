# =============================================================================
#                     ########################################
#                     #     CLINICAL REPORT RENDERER HTML    #
#                     ########################################
# =============================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from dcap.seeg.clinical.bundle import ClinicalAnalysisBundle
from dcap.seeg.clinical.report.assets import ReportAssetDirs, relpath_for_embed, write_placeholder_png
from dcap.seeg.clinical.report.base import ReportPaths, df_to_html_table
from dcap.viz.electrodes import plot_electrodes_3d


class HtmlClinicalReportRenderer:
    """
    HTML clinical report renderer (default).

    Usage example
    -------------
        renderer = HtmlClinicalReportRenderer()
        paths = renderer.render(bundle, Path("out"))
        print(paths.report_path)
    """

    def render(self, bundle: ClinicalAnalysisBundle, out_dir: Path) -> ReportPaths:
        asset_dirs = ReportAssetDirs.from_out_dir(out_dir)
        asset_dirs.ensure()

        report_path = out_dir / f"{bundle.subject_id}_clinical_report.html"

        decisions: Dict[str, Any] = bundle.preprocessing_context.decisions
        view_requested = decisions.get("analysis_view_requested", "original")
        view_used = decisions.get("analysis_view_used", "original")

        view_names = sorted(list(bundle.raw_views.keys()))
        envelope_names = sorted(list(bundle.envelopes.keys())) if bundle.envelopes else []

        prov_df = _provenance_table(bundle)
        warn_df = _warnings_table(bundle)

        qc_figs = _discover_qc_figures(out_dir)

        # ---------------------------------------------------------------------
        # Electrode localization figure (3D)
        # ---------------------------------------------------------------------
        fig_electrodes_3d = asset_dirs.figures_dir / "electrodes_3d.png"

        electrodes_df = getattr(bundle, "electrodes_df", None)
        coords_space = getattr(bundle, "coords_space", None)

        # Optional: allow caller/pipeline to specify highlight contacts via ctx decisions
        highlight = None
        try:
            decisions = bundle.preprocessing_context.decisions
            if isinstance(decisions, dict):
                highlight_value = decisions.get("electrodes_highlight")
                if isinstance(highlight_value, list):
                    highlight = [str(x) for x in highlight_value]
        except Exception:
            highlight = None

        plot_electrodes_3d(
            electrodes_df=electrodes_df,
            out_path=fig_electrodes_3d,
            coords_space=coords_space,
            title=f"Electrode localization — {bundle.subject_id}",
            highlight=highlight,
        )
        # Never crash the report because a figure failed.
        # We fall back to a placeholder image and continue rendering.
        # write_placeholder_png(fig_electrodes_3d)

        # ---------------------------------------------------------------------
        # Placeholder figures
        # ---------------------------------------------------------------------

        fig_trf_scores_3d = asset_dirs.figures_dir / "trf_scores_3d.png"
        fig_trf_kernel = asset_dirs.figures_dir / "trf_kernel.png"
        fig_trf_scores_bar = asset_dirs.figures_dir / "trf_scores_bar.png"

        write_placeholder_png(fig_electrodes_3d)
        write_placeholder_png(fig_trf_scores_3d)
        write_placeholder_png(fig_trf_kernel)
        write_placeholder_png(fig_trf_scores_bar)

        # ---------------------------------------------------------------------
        # Electrode info (optional)
        # ---------------------------------------------------------------------
        electrodes_df: Optional[pd.DataFrame] = getattr(bundle, "electrodes_df", None)
        electrode_names: List[str] = []
        if electrodes_df is not None and not electrodes_df.empty and "name" in electrodes_df.columns:
            electrode_names = electrodes_df["name"].astype(str).tolist()

        # ---------------------------------------------------------------------
        # TRF score table (optional)
        # ---------------------------------------------------------------------
        trf = bundle.trf_result
        score_df: Optional[pd.DataFrame] = None
        score_table_rel: Optional[str] = None
        if trf is not None and hasattr(trf, "extra"):
            score_df = trf.extra.get("score_df")
            score_table_path = trf.extra.get("score_table_path")
            if score_df is not None and not score_df.empty:
                out_tsv = asset_dirs.tables_dir / "trf_scores.tsv"
                score_df.to_csv(out_tsv, sep="\t", index=False)
                score_table_rel = relpath_for_embed(out_tsv, base_dir=out_dir)
            elif score_table_path:
                # If user provided a path, just display it as text (no guarantee it’s inside out_dir)
                score_table_rel = str(score_table_path)

        html = _render_html(
            bundle=bundle,
            out_dir=out_dir,
            view_requested=view_requested,
            view_used=view_used,
            view_names=view_names,
            envelope_names=envelope_names,
            prov_df=prov_df,
            warn_df=warn_df,
            electrode_names=electrode_names,
            electrodes_3d_rel=relpath_for_embed(fig_electrodes_3d, base_dir=out_dir),
            trf_scores_3d_rel=relpath_for_embed(fig_trf_scores_3d, base_dir=out_dir),
            trf_kernel_rel=relpath_for_embed(fig_trf_kernel, base_dir=out_dir),
            trf_scores_bar_rel=relpath_for_embed(fig_trf_scores_bar, base_dir=out_dir),
            score_df=score_df,
            score_table_rel=score_table_rel,
            qc_figs=qc_figs
        )

        report_path.write_text(html, encoding="utf-8")
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


def _render_html(
    *,
    bundle: ClinicalAnalysisBundle,
    out_dir: Path,
    view_requested: str,
    view_used: str,
    view_names: List[str],
    envelope_names: List[str],
    prov_df: pd.DataFrame,
    warn_df: pd.DataFrame,
    electrode_names: List[str],
    electrodes_3d_rel: str,
    trf_scores_3d_rel: str,
    trf_kernel_rel: str,
    trf_scores_bar_rel: str,
    score_df: Optional[pd.DataFrame],
    score_table_rel: Optional[str],
    qc_figs: Dict[str, List[Path]],
) -> str:
    subject = bundle.subject_id
    session = bundle.session_id or "(none)"
    run = bundle.run_id or "(none)"

    qc_html = _render_qc_html(bundle, out_dir=out_dir, qc_figs=qc_figs)
    electrodes_html = _render_electrodes_html(electrode_names=electrode_names, electrodes_3d_rel=electrodes_3d_rel)
    trf_html = _render_trf_html(
        bundle=bundle,
        trf_scores_3d_rel=trf_scores_3d_rel,
        trf_kernel_rel=trf_kernel_rel,
        trf_scores_bar_rel=trf_scores_bar_rel,
        score_df=score_df,
        score_table_rel=score_table_rel,
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Clinical report — {subject}</title>
  <style>
    :root {{
      --fg: #111;
      --muted: #666;
      --bg: #fff;
      --card: #fafafa;
      --border: #e6e6e6;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    body {{
      font-family: var(--sans);
      color: var(--fg);
      background: var(--bg);
      margin: 24px;
      line-height: 1.35;
    }}
    h1 {{ margin: 0 0 12px 0; }}
    h2 {{ margin-top: 28px; border-top: 1px solid var(--border); padding-top: 18px; }}
    h3 {{ margin-top: 18px; }}
    .meta {{ color: var(--muted); margin-bottom: 18px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin: 12px 0;
    }}
    .kv ul {{ margin: 8px 0 0 18px; }}
    .dcap-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      margin-top: 8px;
    }}
    .dcap-table th, .dcap-table td {{
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .mono {{ font-family: var(--mono); }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 2px 10px;
      margin: 2px 6px 2px 0;
      background: #fff;
      font-size: 0.88rem;
    }}
    img.figure {{
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
    }}
    .small {{ font-size: 0.9rem; color: var(--muted); }}
    
        .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .gallery-item {{
      margin: 0;
    }}
    figure.gallery-item img.figure {{
      width: 100%;
      height: auto;
    }}

  </style>
</head>
<body>
  <h1>Clinical report — {subject}</h1>
  <div class="meta">Generated output folder: <span class="mono">{out_dir.as_posix()}</span></div>

  <div class="card">
    <h2>Identifiers</h2>
    <div class="kv">
      <ul>
        <li><b>Subject</b>: {subject}</li>
        <li><b>Session</b>: {session}</li>
        <li><b>Run</b>: {run}</li>
      </ul>
    </div>
  </div>

  <div class="card">
    <h2>Outputs produced</h2>
    <div class="kv">
      <ul>
        <li><b>Raw views</b>: {", ".join(view_names) if view_names else "(none)"}</li>
        <li><b>Envelopes</b>: {", ".join(envelope_names) if envelope_names else "(none)"}</li>
        <li><b>TRF</b>: {"computed" if bundle.trf_result is not None else "not computed"}</li>
      </ul>
    </div>
  </div>

  <div class="card">
    <h2>QC summary</h2>
    {qc_html}
  </div>

  <div class="card">
    <h2>Analysis view policy</h2>
    <div class="kv">
      <ul>
        <li><b>Requested</b>: {view_requested}</li>
        <li><b>Used</b>: {view_used}</li>
      </ul>
    </div>
  </div>

  <div class="card">
    <h2>Preprocessing provenance</h2>
    {df_to_html_table(prov_df)}
  </div>

  <div class="card">
    <h2>Warnings</h2>
    {df_to_html_table(warn_df)}
  </div>

  <div class="card">
    {electrodes_html}
  </div>

  <div class="card">
    {trf_html}
  </div>
</body>
</html>
"""


def _render_qc_html(bundle: ClinicalAnalysisBundle, *, out_dir: Path, qc_figs: Dict[str, List[Path]]) -> str:
    if bundle.qc is None:
        return "<em>(QC not computed)</em>"

    parts: list[str] = []

    parts.append("<h3>Recording</h3>")
    rec = dict(bundle.qc.recording)
    if rec:
        lis = "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in rec.items()])
        parts.append(f"<ul>{lis}</ul>")
    else:
        parts.append("<em>(none)</em>")

    parts.append("<h3>Views</h3>")
    parts.append(df_to_html_table(bundle.qc.views))

    parts.append("<h3>Channel QC (original)</h3>")
    if bundle.qc.channel_qc is None or bundle.qc.channel_qc.empty:
        parts.append("<em>(not computed)</em>")
        return "\n".join(parts)

    flagged = bundle.qc.channel_qc.loc[
        (bundle.qc.channel_qc["is_flat"]) | (bundle.qc.channel_qc["is_outlier"]),
        ["channel", "variance", "log10_variance", "is_flat", "is_outlier"],
    ].copy()
    parts.append(df_to_html_table(flagged))

    parts.append(_render_existing_png_gallery(title="PSD", paths=qc_figs.get("psd", []), out_dir=out_dir))
    parts.append(
        _render_existing_png_gallery(title="Time series", paths=qc_figs.get("timeseries", []), out_dir=out_dir))
    return "\n".join(parts)


def _render_electrodes_html(*, electrode_names: List[str], electrodes_3d_rel: str) -> str:
    pills = ""
    if electrode_names:
        # Show a lot, but keep it compact
        pills = "".join([f'<span class="pill mono">{name}</span>' for name in electrode_names])
    else:
        pills = "<em>(electrode list not available)</em>"

    return f"""
    <h2>Electrodes</h2>

    <h3>Electrode list</h3>
    <div>{pills}</div>

    <h3>Electrode localization (3D)</h3>
    <div class="small">Placeholder image — will be generated by dcap.viz later.</div>
    <img class="figure" src="{electrodes_3d_rel}" alt="Electrode localization (3D)" />
    """


def _render_trf_html(
    *,
    bundle: ClinicalAnalysisBundle,
    trf_scores_3d_rel: str,
    trf_kernel_rel: str,
    trf_scores_bar_rel: str,
    score_df: Optional[pd.DataFrame],
    score_table_rel: Optional[str],
) -> str:
    trf = bundle.trf_result
    if trf is None:
        return "<h2>TRF analysis</h2><em>(TRF not computed)</em>"

    analysis_view = bundle.preprocessing_context.decisions.get("analysis_view_used", "unknown")

    lag_cfg = (trf.extra.get("lag_config", {}) or {}) if hasattr(trf, "extra") else {}
    tmin_ms = lag_cfg.get("tmin_ms")
    tmax_ms = lag_cfg.get("tmax_ms")

    metrics_html = ""
    if getattr(trf, "metrics", None):
        items = "".join([f"<li><b>{k}</b>: {v:.4g}</li>" for k, v in trf.metrics.items()])
        metrics_html = f"<h3>Summary metrics</h3><ul>{items}</ul>"

    lag_html = ""
    if tmin_ms is not None and tmax_ms is not None:
        lag_html = f"<li><b>Lags</b>: {tmin_ms} … {tmax_ms} ms</li>"

    alpha = trf.extra.get("alpha") if hasattr(trf, "extra") else None
    alpha_html = f"<li><b>Regularization (alpha)</b>: {alpha}</li>" if alpha is not None else ""

    scores_table_html = "<em>(score table not available)</em>"
    if score_df is not None and not score_df.empty:
        scores_table_html = df_to_html_table(score_df, max_rows=60)
    if score_table_rel:
        scores_table_html = f'<div class="small">Saved table: <span class="mono">{score_table_rel}</span></div>' + scores_table_html

    warnings = trf.extra.get("warnings", []) if hasattr(trf, "extra") else []
    warn_html = ""
    if warnings:
        warn_items = "".join([f"<li>{w}</li>" for w in warnings])
        warn_html = f"<h3>TRF notes / warnings</h3><ul>{warn_items}</ul>"

    return f"""
    <h2>TRF analysis</h2>

    <ul>
      <li><b>Analysis view</b>: {analysis_view}</li>
      <li><b>Model</b>: {trf.model_name}</li>
      {lag_html}
      {alpha_html}
    </ul>

    {metrics_html}

    <h3>TRF score localization (3D)</h3>
    <div class="small">Placeholder image — will be generated by dcap.viz later.</div>
    <img class="figure" src="{trf_scores_3d_rel}" alt="TRF score localization (3D)" />

    <h3>TRF kernel</h3>
    <div class="small">Placeholder image — will be generated by dcap.viz later.</div>
    <img class="figure" src="{trf_kernel_rel}" alt="TRF kernel" />

    <h3>TRF scores across channels</h3>
    <div class="small">Placeholder image — will be generated by dcap.viz later.</div>
    <img class="figure" src="{trf_scores_bar_rel}" alt="TRF scores across channels" />

    <h3>TRF scores table</h3>
    {scores_table_html}

    {warn_html}
    """


def _find_pngs_under(
    root: Path,
    *,
    max_depth: int = 3,
) -> List[Path]:
    """
    Find PNG images under a root directory, limited by depth.

    Usage example
    -------------
        pngs = _find_pngs_under(Path("out"))
    """
    if not root.exists():
        return []

    root = root.resolve()
    results: list[Path] = []

    # Depth-limited walk (simple + deterministic)
    for path in root.rglob("*.png"):
        try:
            rel_parts = path.resolve().relative_to(root).parts
        except Exception:
            continue

        if len(rel_parts) <= max_depth + 1:
            results.append(path)

    # Sort to keep deterministic ordering
    return sorted(results, key=lambda p: p.as_posix())


def _score_path_by_keywords(path: Path, keywords: Sequence[str]) -> int:
    """
    Score a path by keyword hits in filename and parent names.

    Higher score = better match.

    Usage example
    -------------
        score = _score_path_by_keywords(Path("qc_psd.png"), ["psd"])
    """
    text = (path.as_posix()).lower()
    score = 0
    for kw in keywords:
        if kw in text:
            score += 10
    # Prefer shorter paths (usually more canonical)
    score -= len(text) // 50
    return score


def _pick_best_matches(
    png_paths: Sequence[Path],
    *,
    keywords: Sequence[str],
    k: int = 4,
) -> List[Path]:
    """
    Pick up to k best-matching PNGs by keyword score.

    Usage example
    -------------
        best = _pick_best_matches(pngs, keywords=["psd"])
    """
    scored: list[Tuple[int, Path]] = []
    for p in png_paths:
        scored.append((_score_path_by_keywords(p, keywords), p))

    scored.sort(key=lambda t: t[0], reverse=True)
    best = [p for s, p in scored if s > 0][:k]
    return best


def _discover_qc_figures(out_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover existing QC figures created elsewhere (e.g., make_qc_figures).

    Returns
    -------
    figures
        Dict with keys:
        - "psd": list of png paths
        - "timeseries": list of png paths

    Usage example
    -------------
        figs = _discover_qc_figures(Path("out/sub-001"))
    """
    candidate_roots = [
        out_dir,
        out_dir / "qc",
        out_dir / "qc_figures",
        out_dir / "figures",
        out_dir / "plots",
    ]

    pngs: list[Path] = []
    for root in candidate_roots:
        pngs.extend(_find_pngs_under(root, max_depth=4))

    # De-dup (by resolved path)
    unique: dict[str, Path] = {}
    for p in pngs:
        unique[p.resolve().as_posix()] = p
    pngs = sorted(unique.values(), key=lambda p: p.as_posix())

    psd_keywords = ["psd", "power", "spectrum", "welch"]
    ts_keywords = ["timeseries", "time_series", "timeseries", "raw", "trace", "overview"]

    return {
        "psd": _pick_best_matches(pngs, keywords=psd_keywords, k=4),
        "timeseries": _pick_best_matches(pngs, keywords=ts_keywords, k=4),
    }


def _render_existing_png_gallery(
    *,
    title: str,
    paths: Sequence[Path],
    out_dir: Path,
) -> str:
    """
    Render an HTML gallery for existing PNG files.

    Usage example
    -------------
        html = _render_existing_png_gallery(
            title="PSD",
            paths=[Path("out/qc_psd.png")],
            out_dir=Path("out"),
        )
    """
    if not paths:
        return f"<h3>{title}</h3><em>(not found)</em>"

    items: list[str] = [f"<h3>{title}</h3>"]
    items.append('<div class="gallery">')

    for p in paths:
        rel = relpath_for_embed(p, base_dir=out_dir)
        items.append(
            f"""
            <figure class="gallery-item">
              <img class="figure" src="{rel}" alt="{title}" />
              <figcaption class="small mono">{rel}</figcaption>
            </figure>
            """
        )

    items.append("</div>")
    return "\n".join(items)

