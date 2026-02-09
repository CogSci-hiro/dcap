# =============================================================================
# TRF analysis: user-facing MNE-like API
# =============================================================================

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .fit import fit_trf_auto
from .io import _save_trf_result
from .lags import LagSpec, compute_lags
from .metrics import aggregate_outputs, metric_dispatch
from .prep import PreparedDataset, prepare_dataset
from .types import CvSpec, FitSpec, ScoringSpec, SegmentSpec, TrfResult


class TemporalReceptiveField:
    """MNE-like front door for TRF fitting.

    This class centralizes:
    - segmentation (runs -> segments),
    - CV (blocked k-fold / LOO-run),
    - scoring (pearson / spearman / r2),
    - backend selection.

    Usage example
    -------------
        trf = TemporalReceptiveField(
            lag_spec=LagSpec(-0.1, 0.4, mode="valid"),
            segment_spec=SegmentSpec(n_segments_per_run=3),
            fit_spec=FitSpec(alphas=[0.1, 1.0, 10.0]),
            cv_spec=CvSpec(scheme="blocked_kfold", n_splits=6, purge_s=0.5),
            scoring_spec=ScoringSpec(scoring="pearson"),
            backend="ridge",
        )
        trf.fit(X, Y, sfreq=100.0)
        y_hat = trf.predict(X, sfreq=100.0)
        trf.save("sub-01_trf.npz")
    """

    def __init__(
        self,
        *,
        lag_spec: LagSpec,
        segment_spec: Optional[SegmentSpec] = None,
        fit_spec: Optional[FitSpec] = None,
        cv_spec: Optional[CvSpec] = None,
        scoring_spec: Optional[ScoringSpec] = None,
        backend: str = "ridge",
        backend_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.lag_spec = lag_spec
        self.segment_spec = segment_spec
        self.fit_spec = fit_spec or FitSpec(alpha=1.0)
        self.cv_spec = cv_spec
        self.scoring_spec = scoring_spec or ScoringSpec(scoring="pearson")
        self.backend = backend
        self.backend_params = backend_params or {}

        self.result_: Optional[TrfResult] = None
        self.sfreq_: Optional[float] = None

    def fit(self, X: np.ndarray, Y: np.ndarray, *, sfreq: float, run_ids: Optional[Sequence[str]] = None) -> "TemporalReceptiveField":
        dataset = prepare_dataset(
            X, Y,
            sfreq=sfreq,
            lag_spec=self.lag_spec,
            segment_spec=self.segment_spec,
            run_ids=run_ids,
        )
        self.sfreq_ = float(sfreq)
        self.result_ = fit_trf_auto(
            dataset,
            lag_spec=self.lag_spec,
            segment_spec=self.segment_spec,
            fit_spec=self.fit_spec,
            scoring_spec=self.scoring_spec,
            cv_spec=self.cv_spec,
            backend=self.backend,
            backend_params=self.backend_params,
        )
        return self

    def predict(self, X: np.ndarray, *, sfreq: Optional[float] = None) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Call fit() or load a saved TRF before predict().")
        if sfreq is None:
            sfreq = self.sfreq_
        if sfreq is None:
            raise ValueError("sfreq must be provided if the object was not fit in this session.")

        X = np.asarray(X)
        if X.ndim == 3:
            # (time, run, feat) -> predict per run
            preds = []
            for r in range(X.shape[1]):
                preds.append(self._predict_2d(X[:, r, :], sfreq=float(sfreq)))
            return np.stack(preds, axis=1)
        return self._predict_2d(X, sfreq=float(sfreq))

    def _predict_2d(self, X2: np.ndarray, *, sfreq: float) -> np.ndarray:
    from .predict_kernel import predict_from_kernel

    model = self.result_.model
    mode = "valid"
    if self.result_.lag_spec is not None:
        mode = self.result_.lag_spec.mode

    return predict_from_kernel(
        X2,
        coef=model.coef,
        intercept=model.intercept,
        lags_samp=model.lags_samp,
        mode=mode,
    )

    def score(self, X: np.ndarray, Y: np.ndarray, *, sfreq: Optional[float] = None, output_agg: str = "mean") -> float:
        if sfreq is None:
            sfreq = self.sfreq_
        if sfreq is None:
            raise ValueError("sfreq must be provided if the object was not fit in this session.")
        y_hat = self.predict(X, sfreq=float(sfreq))
        # flatten runs if needed
        if y_hat.ndim == 3:
            y_hat = y_hat.reshape(y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2])
            Y = np.asarray(Y).reshape(Y.shape[0] * Y.shape[1], Y.shape[2])
        score_fn = metric_dispatch(self.scoring_spec.scoring)
        per_out = score_fn(np.asarray(Y), np.asarray(y_hat), complex_handling=self.scoring_spec.complex_handling)
        return aggregate_outputs(per_out, agg=output_agg)

def plot_kernel(
    self,
    *,
    feature_index: int = 0,
    output_index: int = 0,
    view: str = "real",
    ax=None,
    title: str | None = None,
):
    """Plot a single TRF kernel.

    Parameters
    ----------
    feature_index
        Which feature to plot.
    output_index
        Which output to plot.
    view
        For complex kernels: 'real', 'imag', or 'magnitude'.
    ax
        Optional matplotlib Axes to draw into.
    title
        Optional plot title.

    Returns
    -------
    ax
        Matplotlib axes.

    Usage example
    -------------
        trf.plot_kernel(feature_index=0, output_index=3)
    """
    if self.result_ is None:
        raise RuntimeError("Nothing to plot; fit() or read_trf() first.")
    from .plot import plot_kernel_1d
    return plot_kernel_1d(
        self.result_.model,
        feature_index=int(feature_index),
        output_index=int(output_index),
        view=view,  # type: ignore[arg-type]
        ax=ax,
        title=title,
    )

    def save(self, fname: str | Path) -> None:
        if self.result_ is None:
            raise RuntimeError("Nothing to save; fit() first.")
        _save_trf_result(Path(fname), self.result_)

    @classmethod
    def from_result(cls, result: TrfResult) -> "TemporalReceptiveField":
        trf = cls(
            lag_spec=result.lag_spec or LagSpec(result.model.lags_s.min(), result.model.lags_s.max(), mode="valid"),
            segment_spec=result.segment_spec,
            fit_spec=result.fit_spec or FitSpec(alpha=float(result.model.fit_params.get("alpha", 1.0))),
            cv_spec=result.cv_spec,
            scoring_spec=result.scoring_spec or ScoringSpec(scoring="pearson"),
            backend=result.model.backend,
            backend_params={k: v for k, v in result.model.fit_params.items() if k != "alpha"},
        )
        trf.result_ = result
        trf.sfreq_ = float(result.model.sfreq)
        return trf
