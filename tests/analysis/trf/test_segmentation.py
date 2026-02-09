import numpy as np
import pytest

from dcap.analysis.trf.lags import LagSpec
from dcap.analysis.trf.prep import prepare_dataset
from dcap.analysis.trf.types import SegmentSpec


def test_prepare_dataset_defaults():
    X = np.random.randn(1000, 3)
    Y = np.random.randn(1000, 2)
    ds = prepare_dataset(X, Y, sfreq=100.0, lag_spec=LagSpec(-0.1, 0.2))
    assert len(ds.segments) == 1
    assert ds.segments[0].x.shape == (1000, 3)

    X = np.random.randn(1000, 4, 3)
    Y = np.random.randn(1000, 4, 2)
    ds = prepare_dataset(X, Y, sfreq=100.0, lag_spec=LagSpec(-0.1, 0.2))
    assert len(ds.segments) == 4


def test_segment_spec_requires_one_of_len_or_nsegs():
    X = np.random.randn(1000, 3)
    Y = np.random.randn(1000, 2)
    with pytest.raises(ValueError):
        prepare_dataset(
            X, Y, sfreq=100.0, lag_spec=LagSpec(-0.1, 0.2),
            segment_spec=SegmentSpec(segment_len_s=None, n_segments_per_run=None),
        )


def test_segment_spec_incompatible_len_and_nsegs_raises():
    X = np.random.randn(1000, 3)
    Y = np.random.randn(1000, 2)
    with pytest.raises(ValueError):
        prepare_dataset(
            X, Y, sfreq=100.0, lag_spec=LagSpec(-0.1, 0.2),
            segment_spec=SegmentSpec(segment_len_s=1.0, n_segments_per_run=2, incompat_tol_frac=0.01),
        )


def test_segment_too_short_relative_to_lags_raises():
    X = np.random.randn(1000, 3)
    Y = np.random.randn(1000, 2)
    seg = SegmentSpec(segment_len_s=3.0, hard_min_factor=1.5)
    with pytest.raises(ValueError):
        prepare_dataset(X, Y, sfreq=100.0, lag_spec=LagSpec(-2.0, 2.0), segment_spec=seg)
