# =============================================================================
# TRF analysis: cross-validation splitters over segments
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .types import CvSpec, Fold, TrfSegment


def _group_by_run(segments: Sequence[TrfSegment]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, seg in enumerate(segments):
        groups.setdefault(seg.run_id, []).append(idx)
    return groups


def _apply_purge(
    segments: Sequence[TrfSegment],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    purge_samp: int,
) -> Tuple[int, ...]:
    if purge_samp <= 0:
        return tuple(train_indices)

    test_by_run: Dict[str, List[TrfSegment]] = {}
    for ti in test_indices:
        test_by_run.setdefault(segments[ti].run_id, []).append(segments[ti])

    kept: List[int] = []
    for tr in train_indices:
        seg = segments[tr]
        tests = test_by_run.get(seg.run_id, [])
        if not tests:
            kept.append(tr)
            continue
        # Purge if segment is within purge_samp of any test segment in same run
        too_close = False
        for tseg in tests:
            if seg.stop_sample + purge_samp <= tseg.start_sample:
                continue
            if tseg.stop_sample + purge_samp <= seg.start_sample:
                continue
            too_close = True
            break
        if not too_close:
            kept.append(tr)
    return tuple(kept)


def iter_folds(segments: Sequence[TrfSegment], cv_spec: CvSpec, *, sfreq: float) -> Iterable[Fold]:
    """Yield CV folds over segments."""
    if cv_spec.scheme == "loo_run":
        run_groups = _group_by_run(segments)
        run_ids = list(run_groups.keys())
        purge_samp = int(np.ceil(float(cv_spec.purge_s) * float(sfreq)))
        for fold_id, run_id in enumerate(run_ids):
            test_idx = tuple(run_groups[run_id])
            train_idx = tuple(i for r, idxs in run_groups.items() if r != run_id for i in idxs)
            train_idx = _apply_purge(segments, train_idx, test_idx, purge_samp=purge_samp)
            if len(train_idx) == 0:
                raise ValueError("Purge removed all training segments in a fold. Reduce purge_s or segment size.")
            yield Fold(fold_id=fold_id, train_indices=train_idx, test_indices=test_idx)
        return

    if cv_spec.scheme == "blocked_kfold":
        if cv_spec.n_splits is None or cv_spec.n_splits <= 1:
            raise ValueError("blocked_kfold requires n_splits > 1.")
        n_splits = int(cv_spec.n_splits)
        n_segments = len(segments)
        if n_splits > n_segments:
            raise ValueError(f"n_splits={n_splits} > total segments={n_segments}. Increase segment length or reduce n_splits.")

        rng = np.random.default_rng(cv_spec.random_state) if cv_spec.shuffle else None

        # Assign fold IDs per run to preserve run-local ordering
        run_groups = _group_by_run(segments)
        seg_to_fold = np.full(n_segments, -1, dtype=int)

        for run_id, idxs in run_groups.items():
            idxs_sorted = sorted(idxs, key=lambda i: segments[i].start_sample)
            if cv_spec.shuffle and rng is not None:
                rng.shuffle(idxs_sorted)

            if cv_spec.assignment == "round_robin":
                for j, seg_idx in enumerate(idxs_sorted):
                    seg_to_fold[seg_idx] = j % n_splits
            elif cv_spec.assignment == "blocked_per_run":
                # contiguous blocks assigned to folds by position
                # map segment order -> fold index via proportional bins
                m = len(idxs_sorted)
                for j, seg_idx in enumerate(idxs_sorted):
                    fold = int(np.floor((j / m) * n_splits))
                    fold = min(fold, n_splits - 1)
                    seg_to_fold[seg_idx] = fold
            else:
                raise ValueError(f"Unknown assignment={cv_spec.assignment!r}")

        purge_samp = int(np.ceil(float(cv_spec.purge_s) * float(sfreq)))

        for fold_id in range(n_splits):
            test_idx = tuple(np.where(seg_to_fold == fold_id)[0].tolist())
            train_idx = tuple(np.where(seg_to_fold != fold_id)[0].tolist())
            if len(test_idx) == 0:
                continue
            train_idx = _apply_purge(segments, train_idx, test_idx, purge_samp=purge_samp)
            if len(train_idx) == 0:
                raise ValueError("Purge removed all training segments in a fold. Reduce purge_s or segment size.")
            yield Fold(fold_id=fold_id, train_indices=train_idx, test_indices=test_idx)
        return

    raise ValueError(f"Unsupported cv scheme {cv_spec.scheme!r}")
