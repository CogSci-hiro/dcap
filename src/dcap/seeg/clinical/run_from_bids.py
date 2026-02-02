# =============================================================================
# =============================================================================
#                       ######################################
#                       #   CLINICAL RUNNER (BIDS -> REPORT) #
#                       ######################################
# =============================================================================
# =============================================================================

from pathlib import Path
from typing import Optional

import pandas as pd

from dcap.seeg.clinical.configs import ClinicalAnalysisConfig
from dcap.seeg.clinical.pipeline import run_clinical_analysis
from dcap.seeg.clinical.report import render_report
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
    report_format: str = "html",
) -> Path:
    """
    Run an end-to-end clinical analysis from BIDS and write a report.

    Parameters
    ----------
    bids_root
        Root directory of the BIDS dataset.
    out_dir
        Output directory where analysis artifacts and the report will be written.
    subject_id
        Subject identifier (with or without 'sub-' prefix).
    session_id
        Session identifier (with or without 'ses-' prefix), or None.
    task
        BIDS task name.
    run_id
        Run identifier (with or without 'run-' prefix), or None.
    preproc_cfg
        Preprocessing configuration.
    analysis_cfg
        Optional analysis configuration.
    envelope_cfg
        Optional gamma envelope configuration.
    trf_cfg
        Optional TRF configuration.
    trf_runner
        Optional injected TRF backend/runner.
    report_format
        Report format: 'html' (default) or 'md'.

    Returns
    -------
    report_path
        Path to the generated report file.

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
            report_format="html",
        )
    """
    subject_id_norm = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    session_id_norm = (
        session_id
        if (session_id is None or session_id.startswith("ses-"))
        else f"ses-{session_id}"
    )
    run_id_norm = (
        None
        if run_id is None
        else (run_id if run_id.startswith("run-") else f"run-{run_id}")
    )

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
        subject_id=subject_id_norm,
        session_id=session_id_norm,
        run_id=run_id_norm,
        preproc_cfg=preproc_cfg,
        analysis_cfg=analysis_cfg,
        electrodes_table=electrodes_table,
        coords_cfg=None,
        envelope_cfg=envelope_cfg,
        trf_cfg=trf_cfg,
        events_df=events_df,
        ctx=None,
        trf_runner=trf_runner,
        out_dir=out_dir,
    )

    # -------------------------------------------------------------------------
    # Bridge: make electrode info available to the report renderer
    # -------------------------------------------------------------------------
    _attach_electrodes_df(bundle=bundle, electrodes_table=electrodes_table)

    report_paths = render_report(bundle=bundle, out_dir=out_dir, format=report_format)
    return report_paths.report_path


def _attach_electrodes_df(*, bundle: object, electrodes_table: object) -> None:
    """
    Best-effort adapter: expose electrode info to the report renderer.

    The new renderers look for `bundle.electrodes_df` (a DataFrame). Until the
    bundle contract is updated, we attach it dynamically.

    Expected electrode table format (example)
    ----------------------------------------
    | name | x     | y     | z     | space |
    |------|-------|-------|-------|-------|
    | LA1  | -34.2 | -12.0 | 18.5  | MNI   |
    | LA2  | -33.7 | -10.9 | 16.9  | MNI   |

    Usage example
    -------------
        _attach_electrodes_df(bundle=bundle, electrodes_table=electrodes_table)
    """
    if electrodes_table is None:
        return

    electrodes_df: Optional[pd.DataFrame]

    if isinstance(electrodes_table, pd.DataFrame):
        electrodes_df = electrodes_table
    else:
        # If load_bids_run returns a non-DF table-like object, skip for now.
        electrodes_df = None

    if electrodes_df is None or electrodes_df.empty:
        return

    # Standardize minimal columns where possible
    if "name" not in electrodes_df.columns:
        for candidate in ("electrode", "electrode_name", "label"):
            if candidate in electrodes_df.columns:
                electrodes_df = electrodes_df.rename(columns={candidate: "name"})
                break

    # Attach for renderers
    try:
        setattr(bundle, "electrodes_df", electrodes_df)
    except Exception:
        # If bundle is frozen/slots-only, we silently skip for now.
        return
