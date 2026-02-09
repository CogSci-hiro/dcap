# =============================================================================
# TRF analysis: fitting orchestration (segments + CV)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backends.registry import get_backend
from .cv import iter_folds
from .lags import LagSpec, compute_lags
from .metrics import aggregate_outputs, metric_dispatch
from .types import CvResult, CvSpec, FitSpec, ScoringSpec, TrfModel, TrfResult
from .prep import PreparedDataset


def _concat_segments(segments, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [segments[i].x for i in indices]
    ys = [segments[i].y for i in indices]
    lens = np.array([segments[i].n_samples for i in indices], dtype=int)
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, segments[0].x.shape[1]))
    Y = np.concatenate(ys, axis=0) if ys else np.zeros((0, segments[0].y.shape[1]))
    return X, Y, lens


def fit_once(
    dataset: PreparedDataset,
    *,
    lag_spec: LagSpec,
    fit_spec: FitSpec,
    backend: str = "ridge",
    backend_params: Optional[Dict[str, Any]] = None,
) -> TrfResult:
    """Fit a TRF once on all provided segments."""
    if fit_spec.alpha is None:
        raise ValueError("fit_once requires fit_spec.alpha to be set.")
    if fit_spec.alphas is not None:
        raise ValueError("fit_once does not accept fit_spec.alphas; use fit_trf_auto for CV selection.")

    lags_samp, lags_s = compute_lags(lag_spec, dataset.sfreq)

    be = get_backend(backend)
    params = dict(backend_params or {})
    # mode is derived from LagSpec
    params.setdefault("mode", lag_spec.mode)

    X_all, Y_all, _ = _concat_segments(dataset.segments, range(len(dataset.segments)))
    fit = be.fit(X_all, Y_all, lags_samp=lags_samp, alpha=float(fit_spec.alpha), sfreq=dataset.sfreq, **params)

    model = TrfModel(
        coef=fit.coef,
        intercept=fit.intercept,
        lags_samp=lags_samp,
        lags_s=lags_s,
        sfreq=dataset.sfreq,
        backend=backend,
        fit_params={"alpha": float(fit_spec.alpha), **params},
    )
    return TrfResult(
        model=model,
        lag_spec=lag_spec,
        segment_spec=None,
        cv_spec=None,
        fit_spec=fit_spec,
        scoring_spec=None,
        metadata={},
    )


def select_alpha_cv(
    dataset: PreparedDataset,
    *,
    lag_spec: LagSpec,
    fit_spec: FitSpec,
    cv_spec: CvSpec,
    scoring_spec: ScoringSpec,
    backend: str = "ridge",
    backend_params: Optional[Dict[str, Any]] = None,
    output_agg: str = "mean",
) -> CvResult:
    """Select ridge alpha via CV over segments."""
    if fit_spec.alphas is None or len(fit_spec.alphas) == 0:
        raise ValueError("select_alpha_cv requires fit_spec.alphas to be provided.")
    if fit_spec.alpha is not None:
        raise ValueError("Provide either alpha or alphas, not both.")

    alphas = np.asarray(list(fit_spec.alphas), dtype=float)
    if np.any(alphas <= 0):
        raise ValueError("All alpha values must be > 0.")

    lags_samp, _ = compute_lags(lag_spec, dataset.sfreq)
    be = get_backend(backend)
    params = dict(backend_params or {})
    params.setdefault("mode", lag_spec.mode)

    folds = list(iter_folds(dataset.segments, cv_spec, sfreq=dataset.sfreq))
    if len(folds) == 0:
        raise ValueError("No folds generated; check CV spec and segmentation.")

    score_fn = metric_dispatch(scoring_spec.scoring)

    n_alphas = alphas.size
    n_folds = len(folds)
    fold_scores = np.zeros((n_alphas, n_folds), dtype=float)

    # Optional per-output selection
    per_output = (fit_spec.alpha_mode == "per_output")
    score_by_alpha_by_output = None
    best_alpha_by_output = None

    if per_output:
        n_outputs = dataset.n_outputs
        score_by_alpha_by_output = np.zeros((n_alphas, n_outputs), dtype=float)

    for a_i, alpha in enumerate(alphas):
        for f_i, fold in enumerate(folds):
            X_tr, Y_tr, _ = _concat_segments(dataset.segments, fold.train_indices)
            X_te, Y_te, seg_lens = _concat_segments(dataset.segments, fold.test_indices)

            fit = be.fit(X_tr, Y_tr, lags_samp=lags_samp, alpha=float(alpha), sfreq=dataset.sfreq, **params)
            Y_hat = be.predict(fit, X_te, lags_samp=lags_samp, sfreq=dataset.sfreq, **params)

            # Score per segment then aggregate (time-series friendly)
            seg_scores = []
            seg_start = 0
            for L in seg_lens:
                seg_stop = seg_start + int(L)
                y_seg = Y_te[seg_start:seg_stop, :]
                yhat_seg = Y_hat[seg_start:seg_stop, :]
                per_out = score_fn(y_seg, yhat_seg, complex_handling=scoring_spec.complex_handling)
                seg_scores.append(per_out)
                seg_start = seg_stop

            if len(seg_scores) == 0:
                raise ValueError("Empty test fold after concatenation.")

            seg_scores = np.stack(seg_scores, axis=0)  # (n_segments_in_fold, n_outputs)

            if cv_spec.weight_by_duration:
                w = seg_lens.astype(float)
                w = w / max(w.sum(), 1e-12)
                per_out_fold = (seg_scores * w[:, None]).sum(axis=0)
            else:
                per_out_fold = seg_scores.mean(axis=0)

            if per_output:
                score_by_alpha_by_output[a_i, :] += per_out_fold / float(n_folds)

            fold_scores[a_i, f_i] = aggregate_outputs(per_out_fold, agg=output_agg)

    mean_scores = fold_scores.mean(axis=1)
    best_idx = int(np.argmax(mean_scores))
    best_alpha = float(alphas[best_idx])

    if per_output and score_by_alpha_by_output is not None:
        best_alpha_by_output = alphas[np.argmax(score_by_alpha_by_output, axis=0)].astype(float)

    return CvResult(
        alphas=alphas,
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        scoring=scoring_spec.scoring,
        best_alpha=best_alpha,
        best_alpha_by_output=best_alpha_by_output,
        score_by_alpha_by_output=score_by_alpha_by_output,
    )


def fit_trf_auto(
    dataset: PreparedDataset,
    *,
    lag_spec: LagSpec,
    segment_spec,
    fit_spec: FitSpec,
    scoring_spec: ScoringSpec,
    cv_spec: Optional[CvSpec] = None,
    backend: str = "ridge",
    backend_params: Optional[Dict[str, Any]] = None,
) -> TrfResult:
    """Fit TRF with optional alpha selection via CV, then refit on all data."""
    if fit_spec.alpha is not None and fit_spec.alphas is not None:
        raise ValueError("Provide either fit_spec.alpha or fit_spec.alphas, not both.")

    if fit_spec.alpha is None and (fit_spec.alphas is None or cv_spec is None):
        raise ValueError("If alpha is not provided, you must provide alphas and cv_spec for CV selection.")

    cv_res = None
    chosen_alpha = fit_spec.alpha

    if chosen_alpha is None:
        cv_res = select_alpha_cv(
            dataset,
            lag_spec=lag_spec,
            fit_spec=fit_spec,
            cv_spec=cv_spec,
            scoring_spec=scoring_spec,
            backend=backend,
            backend_params=backend_params,
        )
        chosen_alpha = cv_res.best_alpha

    # Fit on all data with chosen alpha
    final_fit_spec = FitSpec(alpha=float(chosen_alpha), alpha_mode="shared")
    result = fit_once(
        dataset,
        lag_spec=lag_spec,
        fit_spec=final_fit_spec,
        backend=backend,
        backend_params=backend_params,
    )

    return TrfResult(
        model=result.model,
        cv=cv_res,
        nested_cv=None,
        lag_spec=lag_spec,
        segment_spec=segment_spec,
        cv_spec=cv_spec,
        fit_spec=fit_spec,
        scoring_spec=scoring_spec,
        metadata={},
    )
