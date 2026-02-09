import os
import numpy as np
import pytest

from dcap.analysis.trf import TemporalReceptiveField
from dcap.analysis.trf.lags import LagSpec, compute_lags
from dcap.analysis.trf.types import CvSpec, FitSpec, ScoringSpec, SegmentSpec

from ._utils import make_simulated_trf_data, corr_flat


@pytest.mark.visual
def test_ridge_recovers_kernel_and_saves_optional_artifacts(artifacts_dir):
    sfreq = 100.0
    lag_spec = LagSpec(-0.05, 0.2, mode="valid")
    lags_samp, _ = compute_lags(lag_spec, sfreq)

    sim = make_simulated_trf_data(
        seed=1,
        sfreq=sfreq,
        n_times=20_000,
        n_features=2,
        n_outputs=4,
        lags_samp=lags_samp,
        noise_std=0.2,
    )

    trf = TemporalReceptiveField(
        lag_spec=lag_spec,
        segment_spec=SegmentSpec(n_segments_per_run=5),
        fit_spec=FitSpec(alphas=[0.1, 1.0, 10.0, 100.0], alpha_mode="shared"),
        cv_spec=CvSpec(scheme="blocked_kfold", n_splits=5, assignment="blocked_per_run", purge_s=0.25),
        scoring_spec=ScoringSpec(scoring="pearson"),
        backend="ridge",
    ).fit(sim.X, sim.Y, sfreq=sfreq)

    coef_hat = trf.result_.model.coef
    r = corr_flat(coef_hat, sim.coef_true)
    assert r > 0.2

    score = trf.score(sim.X, sim.Y, sfreq=sfreq)
    assert np.isfinite(score)

    if os.environ.get("DCAP_TRF_SAVE_ARTIFACTS", "0") == "1":
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifacts_dir / "sim_recovery.npz",
            sfreq=np.array([sfreq], dtype=float),
            lags_samp=lags_samp,
            lags_s=trf.result_.model.lags_s,
            coef_true=sim.coef_true,
            coef_hat=coef_hat,
            intercept_true=sim.intercept_true,
            intercept_hat=trf.result_.model.intercept,
        )

    if os.environ.get("DCAP_TRF_SAVE_FIGURES", "0") == "1":
        import matplotlib.pyplot as plt
        from dcap.analysis.trf.plot import plot_kernel_1d

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ax = plot_kernel_1d(trf.result_.model, feature_index=0, output_index=0)
        ax.figure.savefig(artifacts_dir / "kernel_hat_f0_o0.png", dpi=150)
        plt.close(ax.figure)
