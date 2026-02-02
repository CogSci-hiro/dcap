# =============================================================================
#                     ########################################
#                     #     CLINICAL REPORT RENDERER BASE    #
#                     ########################################
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd

from dcap.seeg.clinical.bundle import ClinicalAnalysisBundle


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """
    Output paths produced by a report renderer.

    Attributes
    ----------
    report_path
        Main report file (HTML or Markdown).
    figures_dir
        Directory where figures were written.
    tables_dir
        Directory where tables were written.

    Usage example
    -------------
        paths = ReportPaths(
            report_path=Path("out/sub-001_report.html"),
            figures_dir=Path("out/figures"),
            tables_dir=Path("out/tables"),
        )
    """

    report_path: Path
    figures_dir: Path
    tables_dir: Path


class ClinicalReportRenderer(Protocol):
    """
    Interface for clinical report renderers.

    Renderers must be deterministic: given a bundle + out_dir, they should write
    the report and any referenced assets into out_dir and return ReportPaths.

    Usage example
    -------------
        renderer: ClinicalReportRenderer = HtmlClinicalReportRenderer()
        paths = renderer.render(bundle=my_bundle, out_dir=Path("out"))
        print(paths.report_path)
    """

    def render(self, bundle: ClinicalAnalysisBundle, out_dir: Path) -> ReportPaths:
        raise NotImplementedError


def df_to_md(df: Optional[pd.DataFrame]) -> str:
    """
    Convert a DataFrame to a Markdown table.

    Parameters
    ----------
    df
        DataFrame or None.

    Returns
    -------
    markdown
        Markdown table string or '_(none)_'.

    Usage example
    -------------
        text = df_to_md(pd.DataFrame([{"a": 1, "b": 2}]))
    """
    if df is None or df.empty:
        return "_(none)_"
    return df.to_markdown(index=False)


def df_to_html_table(df: Optional[pd.DataFrame], *, max_rows: int = 30) -> str:
    """
    Convert a DataFrame to a compact HTML table.

    Parameters
    ----------
    df
        DataFrame or None.
    max_rows
        Max rows to display (head).

    Returns
    -------
    html
        HTML table string or '<em>(none)</em>'.

    Notes
    -----
    - This is intentionally dependency-free (no Jinja2).
    - Styling is handled by the HTML renderer via CSS.

    Usage example
    -------------
        html = df_to_html_table(pd.DataFrame([{"a": 1, "b": 2}]))
    """
    if df is None or df.empty:
        return "<em>(none)</em>"

    view = df.head(max_rows).copy()
    return view.to_html(index=False, escape=True, border=0, classes="dcap-table")
