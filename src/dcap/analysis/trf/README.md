# TRF refactor bundle (DCAP)

This folder contains an implementation of the refactor we discussed:

- MNE-like front door: `TemporalReceptiveField`
- MNE-style I/O: `TemporalReceptiveField.save()` and `dcap.read_trf()`
- Segment-aware CV with purge gap; supports epoched and continuous arrays
- Wrapper-level scoring: Pearson, R², Spearman
- Backend registry: stable `ridge` backend + optional `mne-rf`

## Quick start

```python
import numpy as np
from dcap.analysis.trf import TemporalReceptiveField
from dcap.analysis.trf.lags import LagSpec
from dcap.analysis.trf.types import SegmentSpec, FitSpec, CvSpec, ScoringSpec

sfreq = 100.0
X = np.random.randn(10000, 2)     # (time, features)
Y = np.random.randn(10000, 16)    # (time, outputs)

trf = TemporalReceptiveField(
    lag_spec=LagSpec(tmin_s=-0.1, tmax_s=0.4, mode="valid"),
    segment_spec=SegmentSpec(n_segments_per_run=5),
    fit_spec=FitSpec(alphas=[0.1, 1.0, 10.0], alpha_mode="shared"),
    cv_spec=CvSpec(scheme="blocked_kfold", n_splits=5, assignment="blocked_per_run", purge_s=0.5),
    scoring_spec=ScoringSpec(scoring="spearman"),
    backend="ridge",
).fit(X, Y, sfreq=sfreq)

trf.save("demo_trf.npz")
trf2 = __import__("dcap").analysis.trf.read_trf("demo_trf.npz")
score = trf2.score(X, Y, sfreq=sfreq)
print(score)
```
