# =============================================================================
# =============================================================================
#                       #####################################
#                       #   CLINICAL RUNNER (BIDS -> REPORT) #
#                       #####################################
# =============================================================================
# =============================================================================

from pathlib import Path
import re
from typing import Optional

from dcap.seeg.clinical.configs import ClinicalAnalysisConfig
from dcap.seeg.clinical.pipeline import run_clinical_analysis
from dcap.seeg.clinical.report.render import render_report_v0
from dcap.seeg.io.bids import load_bids_run
from dcap.seeg.preprocessing.configs import ClinicalPreprocConfig, GammaEnvelopeConfig
from dcap.seeg.trf.contracts import TRFConfig


def run_clinical_report_from_bids(
    *,
    bids_root: Path,
    out_dir: Path,
    subject_id: str,
    session_id: Optional[str],
    task: str,
    run_id: Optional[str],
    preproc_cfg: ClinicalPreprocConfig,
    analysis_cfg: Optional[ClinicalAnalysisConfig] = None,
    envelope_cfg: Optional[GammaEnvelopeConfig] = None,
    trf_cfg: Optional[TRFConfig] = None,
    trf_runner: Optional[object] = None,
) -> Path:
    """
    Run an end-to-end clinical analysis from BIDS and write a report.

    Usage example
    -------------
        report_path = run_clinical_report_from_bids(
            bids_root=Path("/data/bids"),
            out_dir=Path("./out/sub-001"),
            subject_id="sub-001",
            session_id=None,
            task="conversation",
            run_id="1",
            preproc_cfg=ClinicalPreprocConfig(),
            analysis_cfg=ClinicalAnalysisConfig(analysis_view="original"),
            envelope_cfg=GammaEnvelopeConfig(),
        )
    """
    raw, events_df, electrodes_table = load_bids_run(
        bids_root=bids_root,
        subject_id=subject_id,
        session_id=session_id,
        task=task,
        run_id=run_id,
    )

    bundle = run_clinical_analysis(
        raw=raw,
        bids_root=Path(bids_root),
        subject_id=subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}",
        session_id=session_id if (session_id is None or session_id.startswith("ses-")) else f"ses-{session_id}",
        run_id=(None if run_id is None else (run_id if run_id.startswith("run-") else f"run-{run_id}")),
        preproc_cfg=preproc_cfg,
        analysis_cfg=analysis_cfg,
        electrodes_table=electrodes_table,
        coords_cfg=None,
        envelope_cfg=envelope_cfg,
        trf_cfg=trf_cfg,
        events_df=events_df,
        ctx=None,
        trf_runner=trf_runner,
        out_dir=out_dir
    )

    report_path = render_report_v0(bundle=bundle, out_dir=out_dir)
    return report_path
