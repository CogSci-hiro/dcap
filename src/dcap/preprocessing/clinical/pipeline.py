# =============================================================================
# =============================================================================
#                       #####################################
#                       #   CLINICAL ANALYSIS ORCHESTRATOR  #
#                       #####################################
# =============================================================================
# =============================================================================
#
# Composition layer:
# BIDS IO (outside) -> preprocessing -> optional envelope -> optional TRF -> bundle
#
# Logic only:
# - No file I/O
# - No CLI / printing
#
# =============================================================================

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import mne

from dcap.preprocessing.clinical.bundle import ClinicalAnalysisBundle, ClinicalAnalysisNotes
from dcap.preprocessing.blocks.filtering import compute_gamma_envelope
from dcap.preprocessing.configs import (
    ClinicalPreprocConfig,
    CoordinatesConfig,
    GammaEnvelopeConfig,
)
from dcap.preprocessing.pipelines.clinical import run_clinical_preproc
from dcap.preprocessing.types import PreprocContext
from dcap.analysis.trf.contracts import TRFConfig, TRFInput, TRFResult


AnalysisView = Literal["original", "car", "bipolar", "laplacian", "wm_ref"]

@dataclass(frozen=True)
class ClinicalAnalysisConfig:
    analysis_view: AnalysisView = "original"



def run_clinical_analysis(
    *,
    raw: mne.io.BaseRaw,
    subject_id: str,
    session_id: Optional[str],
    run_id: Optional[str],
    preproc_cfg: ClinicalPreprocConfig,
    electrodes_table: Optional[Mapping[str, Sequence[float]]] = None,
    coords_cfg: Optional[CoordinatesConfig] = None,
    envelope_cfg: Optional[GammaEnvelopeConfig] = None,
    trf_cfg: Optional[TRFConfig] = None,
    events_df: Optional[Any] = None,
    notes: Optional[Mapping[str, str]] = None,
    ctx: Optional[PreprocContext] = None,
    trf_runner: Optional[Any] = None,
) -> ClinicalAnalysisBundle:
    """
    Run a clinical analysis composition pipeline and return a report-ready bundle.

    Parameters
    ----------
    raw
        Loaded Raw object (from BIDS IO layer).
    subject_id, session_id, run_id
        Identifiers used in reporting.
    preproc_cfg
        Clinical preprocessing configuration.
    electrodes_table, coords_cfg
        Optional coordinate attachment inputs.
    envelope_cfg
        Optional gamma envelope configuration. If provided, a "gamma" envelope Raw is produced.
    trf_cfg
        Optional TRF configuration. If provided, TRF computation is attempted using `trf_runner`.
    events_df
        Optional events table needed for TRF. Required if trf_cfg is provided.
    notes
        Optional key-value notes for reporting.
    ctx
        Optional existing PreprocContext.
    trf_runner
        Optional callable: (TRFInput, TRFConfig) -> TRFResult. If None and trf_cfg is provided,
        a NotImplementedError is raised.

    Returns
    -------
    bundle
        ClinicalAnalysisBundle consumed by reporting.

    Usage example
    -------------
        bundle = run_clinical_analysis(
            raw=raw,
            subject_id="sub-001",
            session_id="ses-01",
            run_id="run-1",
            preproc_cfg=ClinicalPreprocConfig(),
            envelope_cfg=GammaEnvelopeConfig(),
        )
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("run_clinical_analysis expects an mne.io.BaseRaw.")

    preproc_result = run_clinical_preproc(
        raw=raw,
        electrodes_table=electrodes_table,
        cfg=preproc_cfg,
        coords_cfg=coords_cfg,
        ctx=ctx,
    )

    artifacts = list(preproc_result.artifacts)
    envelopes = None

    if envelope_cfg is not None:
        # Default: prefer CAR if present, else original.
        if "car" in preproc_result.views:
            envelope_source_raw = preproc_result.views["car"]
        else:
            envelope_source_raw = preproc_result.views["original"]

        env_raw, env_artifact = compute_gamma_envelope(
            raw=envelope_source_raw,
            cfg=envelope_cfg,
            ctx=preproc_result.ctx,
        )

        artifacts.append(env_artifact)
        envelopes = {"gamma": env_raw}

    trf_result: Optional[TRFResult] = None

    if trf_cfg is not None:
        if events_df is None:
            raise ValueError("events_df must be provided when trf_cfg is provided.")
        if envelopes is None or "gamma" not in envelopes:
            raise ValueError("Gamma envelope must be computed when trf_cfg is provided.")
        if trf_runner is None:
            raise NotImplementedError("TRF requested but no trf_runner was provided.")

        trf_input = TRFInput(signal_raw=envelopes["gamma"], events_df=events_df)
        trf_result = trf_runner(trf_input, trf_cfg)

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
        notes=bundle_notes,
    )
