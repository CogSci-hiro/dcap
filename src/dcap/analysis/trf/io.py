# =============================================================================
# TRF analysis: persistence (MNE-style)
# =============================================================================

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .types import CvResult, LagSpec, SegmentSpec, CvSpec, FitSpec, ScoringSpec, TrfModel, TrfResult


def _save_trf_result(path: Path, result: TrfResult) -> None:
    """Save a TRF result to NPZ + JSON metadata."""
    path = Path(path)
    meta: Dict[str, Any] = {
        "lag_spec": asdict(result.lag_spec) if result.lag_spec is not None else None,
        "segment_spec": asdict(result.segment_spec) if result.segment_spec is not None else None,
        "cv_spec": asdict(result.cv_spec) if result.cv_spec is not None else None,
        "fit_spec": asdict(result.fit_spec) if result.fit_spec is not None else None,
        "scoring_spec": asdict(result.scoring_spec) if result.scoring_spec is not None else None,
        "metadata": result.metadata,
    }

    cv = result.cv
    if cv is not None:
        meta["cv"] = {
            "alphas": cv.alphas.tolist(),
            "fold_scores_shape": list(cv.fold_scores.shape),
            "mean_scores": cv.mean_scores.tolist(),
            "scoring": cv.scoring,
            "best_alpha": cv.best_alpha,
            "best_alpha_by_output": None if cv.best_alpha_by_output is None else cv.best_alpha_by_output.tolist(),
            "score_by_alpha_by_output_shape": None if cv.score_by_alpha_by_output is None else list(cv.score_by_alpha_by_output.shape),
        }
    else:
        meta["cv"] = None

    np.savez_compressed(
        path,
        coef=result.model.coef,
        intercept=result.model.intercept,
        lags_samp=result.model.lags_samp,
        lags_s=result.model.lags_s,
        sfreq=np.array([result.model.sfreq], dtype=float),
        fold_scores=(np.array([]) if cv is None else cv.fold_scores),
        score_by_alpha_by_output=(np.array([]) if (cv is None or cv.score_by_alpha_by_output is None) else cv.score_by_alpha_by_output),
        meta=json.dumps(meta),
        backend=np.array([result.model.backend], dtype=object),
        fit_params=json.dumps(result.model.fit_params),
    )


def read_trf(fname: str | Path) -> Any:
    """Read a saved TRF from disk (MNE-style reader)."""
    from .api import TemporalReceptiveField
    path = Path(fname)
    with np.load(path, allow_pickle=True) as npz:
        coef = npz["coef"]
        intercept = npz["intercept"]
        lags_samp = npz["lags_samp"]
        lags_s = npz["lags_s"]
        sfreq = float(npz["sfreq"][0])
        backend = str(npz["backend"][0])
        fit_params = json.loads(str(npz["fit_params"]))
        meta = json.loads(str(npz["meta"]))

        lag_spec = LagSpec(**meta["lag_spec"]) if meta.get("lag_spec") else None
        segment_spec = SegmentSpec(**meta["segment_spec"]) if meta.get("segment_spec") else None
        cv_spec = CvSpec(**meta["cv_spec"]) if meta.get("cv_spec") else None
        fit_spec = FitSpec(**meta["fit_spec"]) if meta.get("fit_spec") else None
        scoring_spec = ScoringSpec(**meta["scoring_spec"]) if meta.get("scoring_spec") else None

        cv_meta = meta.get("cv")
        cv = None
        if cv_meta is not None:
            fold_scores = npz["fold_scores"]
            score_by_alpha_by_output = npz["score_by_alpha_by_output"]
            if fold_scores.size == 0:
                fold_scores = None
            if score_by_alpha_by_output.size == 0:
                score_by_alpha_by_output = None
            cv = CvResult(
                alphas=np.asarray(cv_meta["alphas"], dtype=float),
                fold_scores=np.asarray(fold_scores, dtype=float) if fold_scores is not None else np.zeros((0, 0)),
                mean_scores=np.asarray(cv_meta["mean_scores"], dtype=float),
                scoring=str(cv_meta["scoring"]),
                best_alpha=float(cv_meta["best_alpha"]),
                best_alpha_by_output=None if cv_meta["best_alpha_by_output"] is None else np.asarray(cv_meta["best_alpha_by_output"], dtype=float),
                score_by_alpha_by_output=None if score_by_alpha_by_output is None else np.asarray(score_by_alpha_by_output, dtype=float),
            )

        model = TrfModel(
            coef=coef,
            intercept=intercept,
            lags_samp=lags_samp,
            lags_s=lags_s,
            sfreq=sfreq,
            backend=backend,
            fit_params=fit_params,
        )
        result = TrfResult(
            model=model,
            cv=cv,
            nested_cv=None,
            lag_spec=lag_spec,
            segment_spec=segment_spec,
            cv_spec=cv_spec,
            fit_spec=fit_spec,
            scoring_spec=scoring_spec,
            metadata=meta.get("metadata", {}),
        )

    return TemporalReceptiveField.from_result(result)
