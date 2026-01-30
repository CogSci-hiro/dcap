# =============================================================================
# =============================================================================
#                       #####################################
#                       #   CLINICAL ANALYSIS ORCHESTRATOR  #
#                       #####################################
# =============================================================================
# =============================================================================
#
# Composition layer (clinical):
# ----------------------------
# This module is the *orchestrator* for a single clinical analysis run.
# It composes multiple library subsystems into one coherent pipeline and
# returns a "report-ready" bundle object.
#
# High-level data flow
# --------------------
# BIDS IO (outside this module) -> preprocessing -> analysis view selection
# -> optional gamma envelope -> optional TRF -> optional QC figures -> bundle
#
# Boundaries / responsibilities
# -----------------------------
# This layer is "logic only":
# - No file I/O (except optional figure writing via make_qc_figures; see below)
# - No CLI / printing
# - No interactive plotting
#
# It *does* coordinate:
# - which preprocessing view is used for analysis
# - whether envelope/TRF should be computed
# - provenance decisions to store in the preprocessing context
#
# Outputs
# -------
# The returned `ClinicalAnalysisBundle` is intended to be consumed by:
# - report renderers (markdown/PDF)
# - QC summaries
# - downstream export steps (handled elsewhere)
#
# Notes on determinism
# --------------------
# The orchestrator should be deterministic given:
# - the `raw` input
# - configs
# - electrode metadata
# Any randomness or model selection should be explicit and tracked in `ctx`.
#
# =============================================================================

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import mne

from dcap.seeg.clinical.bundle import ClinicalAnalysisBundle, ClinicalAnalysisNotes
from dcap.seeg.clinical.configs import ClinicalAnalysisConfig
from dcap.seeg.clinical.policy import choose_analysis_view
from dcap.seeg.preprocessing.blocks.filtering import compute_gamma_envelope
from dcap.seeg.preprocessing.configs import (
    ClinicalPreprocConfig,
    CoordinatesConfig,
    GammaEnvelopeConfig,
)
from dcap.seeg.preprocessing.pipelines.clinical import run_clinical_preproc
from dcap.seeg.preprocessing.types import PreprocContext
from dcap.seeg.trf.contracts import TRFConfig, TRFInput, TRFResult
from dcap.seeg.clinical.viz.qc_figures import make_qc_figures
from dcap.seeg.clinical.qc import compute_clinical_qc
from dcap.seeg.clinical.trf.runner_analysis_trf import run_trf_with_analysis_trf


def run_clinical_analysis(
    *,
    raw: mne.io.BaseRaw,
    bids_root: Optional[str],
    subject_id: str,
    session_id: Optional[str],
    run_id: Optional[str],
    preproc_cfg: ClinicalPreprocConfig,
    analysis_cfg: Optional[ClinicalAnalysisConfig] = None,
    electrodes_table: Optional[Mapping[str, Sequence[float]]] = None,
    coords_cfg: Optional[CoordinatesConfig] = None,
    envelope_cfg: Optional[GammaEnvelopeConfig] = None,
    trf_cfg: Optional[TRFConfig] = None,
    events_df: Optional[Any] = None,
    notes: Optional[Mapping[str, str]] = None,
    ctx: Optional[PreprocContext] = None,
    trf_runner: Optional[Any] = None,
    out_dir: Path | None = None,
) -> ClinicalAnalysisBundle:
    """
    Run a clinical analysis composition pipeline and return a report-ready bundle.

    This function is the single "happy path" orchestrator for clinical analysis:
    it assumes `raw` is already loaded (typically via a BIDS IO layer) and
    coordinates preprocessing, view selection, optional envelope/TRF computation,
    and QC summary creation.

    Parameters
    ----------
    raw
        Loaded Raw object (from BIDS IO layer). Must be a valid MNE Raw instance.
    bids_root
        Optional BIDS root path, carried through for provenance or downstream logic.
        This orchestrator does not read from BIDS; it only passes identifiers through.
    subject_id, session_id, run_id
        Identifiers used for reporting and filenames downstream.
    preproc_cfg
        Clinical preprocessing configuration controlling filtering, referencing,
        channel selection, etc.
    analysis_cfg
        Optional clinical analysis configuration controlling policy decisions,
        e.g. which view is preferred for analysis ("clean", "reref", etc.).
        If None, defaults to `ClinicalAnalysisConfig()`.
    electrodes_table, coords_cfg
        Optional coordinate attachment inputs:
        - `electrodes_table` maps channel name -> coordinates (or similar numeric vector)
        - `coords_cfg` controls coordinate interpretation/attachment policy
    envelope_cfg
        Optional gamma envelope configuration. If provided, an envelope Raw
        (currently "gamma") is computed from the chosen analysis view.
    trf_cfg
        Optional TRF configuration. If provided, TRF computation is attempted.
        This requires:
        - `events_df` (alignment/provenance/events table)
        - `envelope_cfg` (since TRF here is defined on the gamma envelope)
    events_df
        Optional events table needed for TRF. Required when `trf_cfg` is provided.
        Type is kept as `Any` because this pipeline treats it as an opaque table-like
        object (usually a pandas DataFrame upstream).
    notes
        Optional key-value notes for reporting (freeform). These are carried into
        the returned bundle.
    ctx
        Optional existing `PreprocContext` to continue a preprocessing history.
        If None, preprocessing creates a new context.
    trf_runner
        Optional callable implementing the TRF runner contract:
            (TRFInput, TRFConfig) -> TRFResult
        If None and `trf_cfg` is provided, we fall back to `run_trf_with_analysis_trf`.
    out_dir
        Optional output directory for QC figures. This is the *one* controlled
        side-effect permitted here: saving figures is a practical requirement in
        clinical workflows. If you want strict "no I/O", keep this as None.

    Returns
    -------
    bundle
        ClinicalAnalysisBundle consumed by clinical reporting and export.

    Usage example
    -------------
        bundle = run_clinical_analysis(
            raw=raw,
            bids_root=str(bids_root),
            subject_id="sub-001",
            session_id="ses-01",
            run_id="run-1",
            preproc_cfg=ClinicalPreprocConfig(),
            envelope_cfg=GammaEnvelopeConfig(),
        )
    """
    # -------------------------------------------------------------------------
    # Input validation (keep strict and early)
    # -------------------------------------------------------------------------
    #
    # `raw` must be an MNE Raw instance: downstream preprocessing assumes MNE
    # semantics (info, annotations, preload state, etc.).
    #
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("run_clinical_analysis expects an mne.io.BaseRaw.")

    # -------------------------------------------------------------------------
    # Step 1: preprocessing (produces multiple views + artifacts + provenance ctx)
    # -------------------------------------------------------------------------
    #
    # `run_clinical_preproc` is expected to return:
    # - `views`: named Raw objects representing different processing "views"
    #   (e.g., original, cleaned, rereferenced, etc.)
    # - `artifacts`: structured outputs (filter logs, warnings, snapshots)
    # - `ctx`: a mutable provenance context tracking steps, decisions, warnings
    #
    preproc_result = run_clinical_preproc(
        raw=raw,
        electrodes_table=electrodes_table,
        cfg=preproc_cfg,
        coords_cfg=coords_cfg,
        ctx=ctx,
    )

    # -------------------------------------------------------------------------
    # Step 2: choose analysis configuration and select an analysis view
    # -------------------------------------------------------------------------
    #
    # Clinical preprocessing may produce multiple candidate views; policy decides
    # which one is used for envelope/TRF/QC computations. We also keep track of:
    # - what was requested
    # - what was actually used
    #
    analysis_cfg_final = analysis_cfg if analysis_cfg is not None else ClinicalAnalysisConfig()

    view_used, raw_analysis, warnings = choose_analysis_view(
        views=preproc_result.views,
        requested=analysis_cfg_final.analysis_view,
    )

    # Collect preprocessing artifacts; we may append more artifacts below.
    artifacts = list(preproc_result.artifacts)

    # Envelope container (keyed by name to allow extension beyond "gamma")
    envelopes = None

    # -------------------------------------------------------------------------
    # Step 3 (optional): compute gamma envelope on the chosen analysis view
    # -------------------------------------------------------------------------
    #
    # Envelope computation is optional because not all clinical analyses require
    # band-limited envelopes; when enabled, we:
    # - compute the envelope raw
    # - append an artifact describing the envelope computation (and warnings)
    # - store the envelope in a dict for downstream steps (TRF, reporting)
    #
    if envelope_cfg is not None:
        env_raw, env_artifact = compute_gamma_envelope(
            raw=raw_analysis,
            cfg=envelope_cfg,
            ctx=preproc_result.ctx,
        )

        artifacts.append(env_artifact)
        envelopes = {"gamma": env_raw}

        # Record the analysis view decision for provenance and report text.
        # Note: these decisions belong in ctx regardless of whether the envelope
        # is computed; kept here since view selection is currently only "used"
        # downstream when envelope/TRF are active.
        preproc_result.ctx.decisions["analysis_view_requested"] = analysis_cfg_final.analysis_view
        preproc_result.ctx.decisions["analysis_view_used"] = view_used

    # -------------------------------------------------------------------------
    # Step 4 (optional): compute TRF on the gamma envelope
    # -------------------------------------------------------------------------
    #
    # Clinical TRF is currently defined specifically on the gamma envelope.
    # Therefore TRF requires:
    # - `events_df` for alignment/provenance and (depending on runner) event logic
    # - a computed "gamma" envelope
    #
    # The runner is injectable to allow:
    # - different TRF implementations (analysis.trf, custom ridge, etc.)
    # - easier testing (swap in a mock)
    #
    trf_result: Optional[TRFResult] = None
    if trf_cfg is not None:
        if events_df is None:
            raise ValueError("events_df must be provided when trf_cfg is provided.")

        if envelopes is None or "gamma" not in envelopes:
            raise ValueError("Gamma envelope must be computed when trf_cfg is provided.")

        runner = trf_runner if trf_runner is not None else run_trf_with_analysis_trf

        # Construct a canonical TRFInput payload. This is the contract boundary
        # between clinical orchestration and TRF analysis implementation.
        #
        # NOTE: task="diapix" is currently hard-coded as DEBUG; in production this
        # should come from metadata / BIDS entities or analysis_cfg.
        #
        trf_input = TRFInput(
            signal_raw=envelopes["gamma"],
            events_df=events_df,
            subject_id=subject_id,
            task="diapix",  # DEBUG
            bids_root=bids_root,
        )

        # Run TRF. The runner is expected to return a TRFResult or raise.
        #
        # IMPORTANT: keep runner signature consistent across implementations.
        # If you need extra context, add fields to TRFInput/TRFConfig rather
        # than passing ad-hoc positional arguments.
        #
        trf_result = runner(trf_input, trf_cfg, bids_root, subject_id)

    # -------------------------------------------------------------------------
    # Step 5 (optional): save QC figures (controlled side-effect)
    # -------------------------------------------------------------------------
    #
    # QC figures are written only if out_dir is provided. This is intended to
    # support clinical workflows where a folder of PNG/PDF artifacts is needed.
    #
    # If you want *strict* logic-only orchestration, keep out_dir=None and let
    # a higher-level IO layer call make_qc_figures instead.
    #
    if out_dir is not None:
        make_qc_figures(
            out_dir=out_dir,
            raw_original=preproc_result.views["original"],
            raw_analysis=raw_analysis,
            analysis_view_name=view_used,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
        )

    # -------------------------------------------------------------------------
    # Step 6: compute QC summary for reporting (no plotting, just metrics/tables)
    # -------------------------------------------------------------------------
    #
    # QC is computed over *all* raw views to support reporting comparisons
    # (e.g., original vs cleaned). The QC object is embedded in the final bundle.
    #
    qc = compute_clinical_qc(raw_views=preproc_result.views, include_channel_table=True)

    # -------------------------------------------------------------------------
    # Step 7: wrap notes and construct the final bundle
    # -------------------------------------------------------------------------
    #
    # Bundle is the single return value for downstream reporting. It contains:
    # - identifiers
    # - raw views
    # - artifacts + provenance context
    # - computed envelopes
    # - optional TRF result
    # - QC summary
    # - freeform notes
    #
    bundle_notes = ClinicalAnalysisNotes(items=dict(notes) if notes is not None else {})

    return ClinicalAnalysisBundle(
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        raw_views=preproc_result.views,
        preprocessing_artifacts=artifacts,
        preprocessing_context=preproc_result.ctx,
        envelopes=envelopes,
        trf_result=trf_result,
        qc=qc,
        notes=bundle_notes,
    )
