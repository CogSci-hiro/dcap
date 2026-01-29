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


def run_clinical_analysis(
    *,
    raw: mne.io.BaseRaw,
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
) -> ClinicalAnalysisBundle:
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("run_clinical_analysis expects an mne.io.BaseRaw.")

    analysis_cfg_final = analysis_cfg if analysis_cfg is not None else ClinicalAnalysisConfig()

    preproc_result = run_clinical_preproc(
        raw=raw,
        cfg=preproc_cfg,
        electrodes_table=electrodes_table,
        coords_cfg=coords_cfg,
        ctx=ctx,
    )

    artifacts = list(preproc_result.artifacts)
    envelopes = None

    if envelope_cfg is not None:
        view_used, envelope_source_raw = choose_analysis_view(
            raw_views=preproc_result.views,
            requested=analysis_cfg_final.analysis_view,
        )

        env_raw, env_artifact = compute_gamma_envelope(
            raw=envelope_source_raw,
            cfg=envelope_cfg,
            ctx=preproc_result.ctx,
        )
        artifacts.append(env_artifact)
        envelopes = {"gamma": env_raw}

        preproc_result.ctx.decisions["analysis_view_requested"] = analysis_cfg_final.analysis_view
        preproc_result.ctx.decisions["analysis_view_used"] = view_used

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
