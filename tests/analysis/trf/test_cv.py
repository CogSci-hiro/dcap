import numpy as np
import pytest

from dcap.analysis.trf.cv import iter_folds
from dcap.analysis.trf.lags import LagSpec
from dcap.analysis.trf.prep import prepare_dataset
from dcap.analysis.trf.types import CvSpec, SegmentSpec


def test_blocked_kfold_generates_disjoint_folds():
    X = np.random.randn(2000, 4, 2)
    Y = np.random.randn(2000, 4, 3)
    ds = prepare_dataset(
        X, Y, sfreq=100.0, lag_spec=LagSpec(-0.1, 0.2),
        segment_spec=SegmentSpec(n_segments_per_run=4),
    )
    cv = CvSpec(scheme="blocked_kfold", n_splits=4, assignment="blocked_per_run", purge_s=0.0)
    folds = list(iter_folds(ds.segments, cv, sfreq=ds.sfreq))
    assert len(folds) >= 2
    for f in folds:
        assert set(f.train_indices).isdisjoint(set(f.test_indices))


def test_purge_can_wipe_training_and_raise():
    X = np.random.randn(500, 1, 2)
    Y = np.random.randn(500, 1, 1)
    ds = prepare_dataset(
        X, Y, sfreq=100.0, lag_spec=LagSpec(-0.1, 0.2),
        segment_spec=SegmentSpec(n_segments_per_run=2),
    )
    cv = CvSpec(scheme="blocked_kfold", n_splits=2, assignment="blocked_per_run", purge_s=10.0)
    with pytest.raises(ValueError):
        list(iter_folds(ds.segments, cv, sfreq=ds.sfreq))
