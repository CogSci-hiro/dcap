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
from dcap.errors.policy import ErrorPolicy, ErrorMode, run_with_policy
from dcap.errors.record import ErrorLog


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
    run_ids: Optional[Sequence[str]] = None,
) -> Path:
    """
    Multi-run clinical analysis from BIDS and write a report.

    Error handling
    --------------
    - Core analysis failures raise (load + preprocessing + TRF fitting).
    - Report assembly and optional bridges use ErrorPolicy and are never silent.
    """

    # -------------------------------------------------------------------------
    # Error policy / log (reports should degrade gracefully, not silently)
    # -------------------------------------------------------------------------
    policy = ErrorPolicy(mode=ErrorMode.COLLECT)   # dev/tests can override to RAISE
    error_log = ErrorLog()

    subject_id_norm = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    session_id_norm = (
        session_id
        if (session_id is None or session_id.startswith("ses-"))
        else f"ses-{session_id}"
    )

    run_ids_norm = _normalize_run_ids(
        bids_root=bids_root,
        subject_id_norm=subject_id_norm,
        session_id_norm=session_id_norm,
        task=task,
        run_ids=run_ids,
        run_id=run_id,
    )

    # -------------------------------------------------------------------------
    # Load all runs (core step: let it raise if it fails)
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

        if electrodes_table is None and electrodes_table_i is not None:
            electrodes_table = electrodes_table_i

    # For TRF we pass a run-aware container for now
    events_payload: Any = events_by_run if trf_cfg is not None else events_by_run[run_ids_norm[0]]

    bundle = run_clinical_analysis(
        raws=raws,
        bids_root=bids_root,  # IMPORTANT: keep as Path, not str
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
    # Attach policy + log to bundle (best-effort, but never silent)
    # -------------------------------------------------------------------------
    run_with_policy(
        lambda: _attach_error_handles_to_bundle(bundle=bundle, policy=policy, error_log=error_log),
        policy=policy,
        stage="reports.bridge",
        artifact="bundle_error_handles",
        context={"subject_id": subject_id_norm, "session_id": session_id_norm},
        error_log=error_log,
        on_error_return=None,
        optional=True,
    )

    # -------------------------------------------------------------------------
    # Bridge: electrode info for renderer (optional but should never be silent)
    # -------------------------------------------------------------------------
    run_with_policy(
        lambda: _attach_electrodes_df(bundle=bundle, electrodes_table=electrodes_table),
        policy=policy,
        stage="reports.bridge",
        artifact="electrodes_df",
        context={"subject_id": subject_id_norm, "session_id": session_id_norm},
        error_log=error_log,
        on_error_return=None,
        optional=True,
    )

    # -------------------------------------------------------------------------
    # Render report (should degrade gracefully: if it fails, write a fallback)
    # -------------------------------------------------------------------------
    report_paths = run_with_policy(
        lambda: render_report(bundle=bundle, out_dir=out_dir, format=report_format),
        policy=policy,
        stage="reports.render",
        artifact=f"clinical_report_{report_format}",
        context={"out_dir": str(out_dir), "format": report_format},
        error_log=error_log,
        on_error_return=None,
        optional=False,
    )

    if report_paths is not None:
        return report_paths.report_path

    # If rendering failed and policy allowed continuation, write a fallback report
    fallback_path = _write_fallback_error_report(
        out_dir=out_dir,
        subject_id=subject_id_norm,
        session_id=session_id_norm,
        run_ids=list(run_ids_norm),
        error_log=error_log,
        report_format=report_format,
    )
    return fallback_path

def _normalize_run_ids(
    *,
    bids_root: Path,
    subject_id_norm: str,
    session_id_norm: Optional[str],
    task: str,
    run_ids: Optional[Sequence[str]],
    run_id: Optional[str],
) -> List[str]:
    """
    Normalize run identifiers into a list of BIDS-style `run-XX` strings.

    Behavior
    --------
    - If `run_ids` is provided, use it.
    - Else if `run_id` is provided, treat as single run.
    - Else auto-discover runs from BIDS; if none found, raise a helpful error.
    """
    if run_ids is not None and len(run_ids) > 0:
        src = list(run_ids)
    elif run_id is not None:
        src = [run_id]
    else:
        discovered = _discover_run_ids_from_bids(
            bids_root=bids_root,
            subject_id=subject_id_norm,
            session_id=session_id_norm,
            task=task,
        )
        if len(discovered) == 0:
            raise ValueError(
                "No runs specified and none could be auto-discovered. "
                "Provide --run <ID> (e.g. --run 01) or --runs <IDs> (e.g. --runs 01 02 03 04)."
            )
        src = discovered

    out: List[str] = []
    for r in src:
        rr = str(r)
        if not rr.startswith("run-"):
            rr = f"run-{rr}"
        out.append(rr)

    if len(out) == 0:
        raise ValueError("No valid run ids after normalization.")
    return out


def _attach_error_handles_to_bundle(*, bundle: object, policy: ErrorPolicy, error_log: ErrorLog) -> None:
    """
    Attach error policy/log to bundle for downstream renderers.

    This is best-effort because some bundles may be frozen or slots-only.
    Any failure should be handled by the caller via run_with_policy.
    """
    setattr(bundle, "error_policy", policy)
    setattr(bundle, "error_log", error_log)


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
    setattr(bundle, "electrodes_df", electrodes_df)

def _write_fallback_error_report(
    *,
    out_dir: Path,
    subject_id: str,
    session_id: Optional[str],
    run_ids: Sequence[str],
    error_log: ErrorLog,
    report_format: str,
) -> Path:
    """
    Always produce an artifact explaining what failed.

    Writes a simple Markdown file. If HTML was requested, we still write MD
    because it's robust and readable in any environment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "clinical_report_ERROR.md"

    lines: List[str] = [f"# Clinical report failed to render\n",
                        f"- subject_id: `{subject_id}`\n",
                        f"- session_id: `{session_id}`\n",
                        f"- runs: `{list(run_ids)}`\n",
                        f"- requested_format: `{report_format}`\n\n",
                        "## Errors\n\n"]

    # We don't know the exact ErrorLog API, but typical is `.records` being a list.
    # If your ErrorLog uses a different attribute, tweak here once.
    records = getattr(error_log, "records", None)
    if not records:
        lines.append("_No error records found (unexpected)._ \n")
    else:
        for i, rec in enumerate(records, start=1):
            # Make this robust to dict-like or dataclass-like records
            stage = getattr(rec, "stage", None) or rec.get("stage") if isinstance(rec, dict) else None
            artifact = getattr(rec, "artifact", None) or rec.get("artifact") if isinstance(rec, dict) else None
            message = getattr(rec, "message", None) or rec.get("message") if isinstance(rec, dict) else None
            exc_type = getattr(rec, "exc_type", None) or rec.get("exc_type") if isinstance(rec, dict) else None

            lines.append(f"{i}. **stage**: `{stage}` | **artifact**: `{artifact}`\n")
            if exc_type:
                lines.append(f"   - exc_type: `{exc_type}`\n")
            if message:
                lines.append(f"   - message: {message}\n")
            lines.append("\n")

    path.write_text("".join(lines), encoding="utf-8")
    return path


def _discover_run_ids_from_bids(
    *,
    bids_root: Path,
    subject_id: str,
    session_id: Optional[str],
    task: str,
) -> list[str]:
    """
    Best-effort run discovery from the BIDS directory structure.

    Returns a sorted list of `run-XX` strings.
    """
    subject_dir = bids_root / subject_id
    if session_id is not None:
        subject_dir = subject_dir / session_id

    # Usually EEG/iEEG live under ieeg/ or eeg/. We’ll scan both.
    candidate_dirs = [subject_dir / "ieeg", subject_dir / "eeg"]

    run_ids: set[str] = set()
    patterns = [
        f"{subject_id}*task-{task}*run-*.tsv",
        f"{subject_id}*task-{task}*run-*.edf",
        f"{subject_id}*task-{task}*run-*.vhdr",
        f"{subject_id}*task-{task}*run-*.set",
    ]

    for d in candidate_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.glob(pat):
                name = p.name
                # Extract "run-XX" substring
                idx = name.find("run-")
                if idx == -1:
                    continue
                run_part = name[idx : idx + 6]  # "run-XX" (assumes 2 digits)
                # Allow longer run ids: run-001 etc
                j = idx + 4
                while j < len(name) and name[j].isdigit():
                    j += 1
                run_part = name[idx:j]
                run_ids.add(run_part)

    return sorted(run_ids)

