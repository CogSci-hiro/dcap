# =============================================================================
# =============================================================================
#                       ######################################
#                       #   CLINICAL RUNNER (BIDS -> REPORT) #
#                       ######################################
# =============================================================================
# =============================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mne
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
    run_ids: Optional[Sequence[str]] = None,  # NEW (preferred)
) -> Path:
    """
    Run an end-to-end clinical analysis from BIDS and write a report.

    Notes
    -----
    - Multi-run is supported via `run_ids`. If `run_ids` is None, falls back to
      `run_id` for single-run behavior.
    """
    subject_id_norm = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    session_id_norm = (
        session_id
        if (session_id is None or session_id.startswith("ses-"))
        else f"ses-{session_id}"
    )

    # Normalize requested runs
    run_ids_norm = _normalize_run_ids(run_ids=run_ids, run_id=run_id)

    # -------------------------------------------------------------------------
    # Load all runs (Raw + events) and keep electrodes table (assume consistent)
    # -------------------------------------------------------------------------
    raws: List[mne.io.BaseRaw] = []
    events_by_run: Dict[str, Any] = {}
    electrodes_table: Any = None

    for rid in run_ids_norm:
        raw_i, events_df_i, electrodes_table_i = load_bids_run(
            bids_root=bids_root,
            subject_id=subject_id,
            session_id=session_id,
            task=task,
            run_id=rid,
        )
        raws.append(raw_i)
        events_by_run[rid] = events_df_i

        # Prefer first non-null electrode table; assume consistent across runs
        if electrodes_table is None and electrodes_table_i is not None:
            electrodes_table = electrodes_table_i

    # -------------------------------------------------------------------------
    # Decide what to pass as events_df
    # -------------------------------------------------------------------------
    #
    # For TRF across epochs (runs), your TRF adapter/runner should be run-aware.
    # Minimal approach: pass a dict by run_id and update TRFInput typing later.
    #
    events_payload: Any
    if trf_cfg is not None:
        events_payload = events_by_run
    else:
        # Keep a single-run-like default for non-TRF paths
        events_payload = events_by_run[run_ids_norm[0]]

    bundle = run_clinical_analysis(
        raws=raws,
        bids_root=bids_root,
        subject_id=subject_id_norm,
        session_id=session_id_norm,
        run_ids=list(run_ids_norm),
        preproc_cfg=preproc_cfg,
        analysis_cfg=analysis_cfg,
        electrodes_table=electrodes_table,
        coords_cfg=None,
        envelope_cfg=envelope_cfg,
        trf_cfg=trf_cfg,
        events_df=events_payload,
        ctx=None,
        trf_runner=trf_runner,
        out_dir=out_dir,
    )

    # -------------------------------------------------------------------------
    # Bridge (temporary): electrode info for renderer(s)
    # -------------------------------------------------------------------------
    _attach_electrodes_df(bundle=bundle, electrodes_table=electrodes_table)

    report_paths = render_report(bundle=bundle, out_dir=out_dir, format=report_format)
    return report_paths.report_path


def _normalize_run_ids(*, run_ids: Optional[Sequence[str]], run_id: Optional[str]) -> List[str]:
    """
    Normalize run identifiers into a list of BIDS-style `run-XX` strings.

    Rules
    -----
    - If `run_ids` is provided, use it.
    - Else if `run_id` is provided, treat as single run.
    - Else raise (for now) — later we can implement "discover all runs".
    """
    if run_ids is not None and len(run_ids) > 0:
        src = list(run_ids)
    elif run_id is not None:
        src = [run_id]
    else:
        raise ValueError("Either run_ids or run_id must be provided (multi-run default).")

    out: List[str] = []
    for r in src:
        if r is None:
            continue
        rr = str(r)
        if not rr.startswith("run-"):
            rr = f"run-{rr}"
        out.append(rr)
    if len(out) == 0:
        raise ValueError("No valid run ids after normalization.")
    return out



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
