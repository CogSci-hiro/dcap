# =============================================================================
#                     ########################################
#                     #          CLINICAL REPORT API         #
#                     ########################################
# =============================================================================

from pathlib import Path
from typing import Optional

from dcap.seeg.clinical.bundle import ClinicalAnalysisBundle
from dcap.seeg.clinical.report.base import ClinicalReportRenderer, ReportPaths
from dcap.seeg.clinical.report.renderer_html import HtmlClinicalReportRenderer
from dcap.seeg.clinical.report.renderer_md import MdClinicalReportRenderer


def render_report(
    bundle: ClinicalAnalysisBundle,
    out_dir: Path,
    *,
    format: str = "html",
    renderer: Optional[ClinicalReportRenderer] = None,
) -> ReportPaths:
    """
    Render a clinical report to HTML (default) or Markdown.

    Parameters
    ----------
    bundle
        Clinical analysis bundle.
    out_dir
        Output directory for report + assets.
    format
        'html' or 'md' (ignored if `renderer` is provided).
    renderer
        Optional custom renderer implementing ClinicalReportRenderer.

    Returns
    -------
    paths
        Paths to the generated report and asset folders.

    Usage example
    -------------
        paths = render_report(bundle, Path("out"))  # HTML by default
        paths_md = render_report(bundle, Path("out_md"), format="md")
    """
    if renderer is not None:
        return renderer.render(bundle=bundle, out_dir=out_dir)

    fmt = format.strip().lower()
    if fmt in {"html", "htm"}:
        return HtmlClinicalReportRenderer().render(bundle=bundle, out_dir=out_dir)
    if fmt in {"md", "markdown"}:
        return MdClinicalReportRenderer().render(bundle=bundle, out_dir=out_dir)

    raise ValueError(f"Unknown report format: {format!r}. Expected 'html' or 'md'.")
