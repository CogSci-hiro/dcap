"""Simulated TRF recovery demo (artifact-based).

Run:
    python examples/trf_simulated_recovery.py

Outputs:
    ./_artifacts/trf_sim_recovery.npz
    ./_artifacts/kernel_hat_f0_o0.png
    ./_artifacts/kernel_true_f0_o0.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dcap.analysis.trf import TemporalReceptiveField
from dcap.analysis.trf.lags import LagSpec, compute_lags
from dcap.analysis.trf.types import CvSpec, FitSpec, ScoringSpec, SegmentSpec, TrfModel
from dcap.analysis.trf.predict_kernel import predict_from_kernel
from dcap.analysis.trf.plot import plot_kernel_1d


def main() -> None:
    out_dir = Path("_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    sfreq = 100.0
    lag_spec = LagSpec(-0.05, 0.2, mode="valid")
    lags_samp, _ = compute_lags(lag_spec, sfreq)

    rng = np.random.default_rng(1)
    n_times, n_features, n_outputs = 30_000, 2, 4

    X = rng.standard_normal((n_times, n_features))
    coef_true = rng.standard_normal((len(lags_samp), n_features, n_outputs))
    for _ in range(2):
        coef_true = (coef_true + np.roll(coef_true, 1, axis=0) + np.roll(coef_true, -1, axis=0)) / 3.0
    intercept_true = rng.standard_normal((n_outputs,)) * 0.05

    Y_clean = predict_from_kernel(X, coef=coef_true, intercept=intercept_true, lags_samp=lags_samp, mode="valid")
    Y = Y_clean + rng.standard_normal(Y_clean.shape) * 0.2

    trf = TemporalReceptiveField(
        lag_spec=lag_spec,
        segment_spec=SegmentSpec(n_segments_per_run=6),
        fit_spec=FitSpec(alphas=[0.1, 1.0, 10.0, 100.0]),
        cv_spec=CvSpec(scheme="blocked_kfold", n_splits=6, assignment="blocked_per_run", purge_s=0.25),
        scoring_spec=ScoringSpec(scoring="pearson"),
        backend="ridge",
    ).fit(X, Y, sfreq=sfreq)

    score = trf.score(X, Y, sfreq=sfreq)
    print(f"Score ({trf.scoring_spec.scoring}): {score:.3f}")

    np.savez_compressed(
        out_dir / "trf_sim_recovery.npz",
        sfreq=np.array([sfreq], dtype=float),
        lags_samp=lags_samp,
        lags_s=trf.result_.model.lags_s,
        coef_true=coef_true,
        coef_hat=trf.result_.model.coef,
        intercept_true=intercept_true,
        intercept_hat=trf.result_.model.intercept,
    )

    ax1 = plot_kernel_1d(trf.result_.model, feature_index=0, output_index=0, title="Estimated kernel (f0→o0)")
    ax1.figure.savefig(out_dir / "kernel_hat_f0_o0.png", dpi=150)
    plt.close(ax1.figure)

    model_true = TrfModel(
        coef=coef_true,
        intercept=intercept_true,
        lags_samp=lags_samp,
        lags_s=lags_samp.astype(float) / sfreq,
        sfreq=sfreq,
        backend="sim",
        fit_params={},
    )
    ax2 = plot_kernel_1d(model_true, feature_index=0, output_index=0, title="True kernel (f0→o0)")
    ax2.figure.savefig(out_dir / "kernel_true_f0_o0.png", dpi=150)
    plt.close(ax2.figure)

    print(f"Saved artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
