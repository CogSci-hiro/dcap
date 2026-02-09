"""Compare CV assignment strategies on epoched data.

Run:
    python examples/trf_cv_strategies.py
"""

from __future__ import annotations

import numpy as np

from dcap.analysis.trf.lags import LagSpec
from dcap.analysis.trf.prep import prepare_dataset
from dcap.analysis.trf.fit import select_alpha_cv
from dcap.analysis.trf.types import CvSpec, FitSpec, ScoringSpec, SegmentSpec


def main() -> None:
    rng = np.random.default_rng(0)

    sfreq = 100.0
    lag_spec = LagSpec(-0.05, 0.2, mode="valid")

    # 4 runs: (time, run, features)
    X = rng.standard_normal((20_000, 4, 2))
    Y = rng.standard_normal((20_000, 4, 3))

    ds = prepare_dataset(
        X, Y,
        sfreq=sfreq,
        lag_spec=lag_spec,
        segment_spec=SegmentSpec(n_segments_per_run=6),
    )

    fit_spec = FitSpec(alphas=[0.1, 1.0, 10.0, 100.0])
    scoring_spec = ScoringSpec(scoring="pearson")

    for assignment in ["blocked_per_run", "round_robin"]:
        cv_spec = CvSpec(scheme="blocked_kfold", n_splits=6, assignment=assignment, purge_s=0.25)
        cv = select_alpha_cv(
            ds,
            lag_spec=lag_spec,
            fit_spec=fit_spec,
            cv_spec=cv_spec,
            scoring_spec=scoring_spec,
            backend="ridge",
        )
        print(f"{assignment}: best_alpha={cv.best_alpha:.3g}, mean_scores={cv.mean_scores}")


if __name__ == "__main__":
    main()
