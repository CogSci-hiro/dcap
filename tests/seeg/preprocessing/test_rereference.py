# =============================================================================
# =============================================================================
#                     ########################################
#                     #         TEST: REREFERENCING          #
#                     ########################################
# =============================================================================
# =============================================================================
"""
Unit tests for sEEG rereferencing.

These tests are intentionally *invariant-driven*: they verify mathematical
properties that must hold if rereferencing is implemented correctly.

They also cover metadata invariants (sfreq, n_times, annotations) and the
virtual-electrode geometry rule for bipolar channels (midpoint position).

Notes
-----
- Some tests (virtual electrode locations) may fail until the rereference
  implementation explicitly attaches coordinates to bipolar channels.
- If you later split APIs into analysis (`rereference(raw, method=...)`) and
  clinical (`rereference_view(raw, cfg, ctx)`), adapt imports accordingly.

Usage
-----
    pytest -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.testing as npt
import pytest

import mne


# =============================================================================
#                     ########################################
#                     #             Test fixtures            #
#                     ########################################
# =============================================================================
_SFREQ_HZ: float = 100.0
_N_TIMES: int = 10
_TOL: float = 1e-12


@dataclass(frozen=True)
class _DummyGeometry:
    """Minimal geometry stub for tests."""
    shafts: Dict[str, List[str]]
    neighbors: Dict[str, List[str]]


class _DummyCtx:
    """Minimal context stub (matches attributes used by rereference block)."""

    def __init__(self, *, geometry: Optional[_DummyGeometry], decisions: Optional[Dict[str, object]] = None) -> None:
        self.geometry = geometry
        self.decisions: Dict[str, object] = decisions or {}
        self._records: List[Tuple[str, Dict[str, object]]] = []

    def add_record(self, name: str, payload: Dict[str, object]) -> None:
        self._records.append((name, payload))


@dataclass(frozen=True)
class _DummyCfg:
    """Minimal config stub compatible with asdict(cfg) and the block logic."""
    methods: Sequence[str]
    car_scope: str = "global"          # "global" | "by_shaft"
    laplacian_mode: str = "shaft_1d"   # "shaft_1d" | "knn_3d"


@pytest.fixture()
def toy_raw() -> mne.io.BaseRaw:
    """
    Build a tiny deterministic Raw with positions and annotations.

    Channels
    --------
    A1, A2, A3, B1 are sEEG channels.
    STI 014 is a stim channel (should not be modified by iEEG rereferencing).
    """
    ch_names = ["A1", "A2", "A3", "B1", "STI 014"]
    ch_types = ["seeg", "seeg", "seeg", "seeg", "stim"]

    info = mne.create_info(ch_names=ch_names, sfreq=_SFREQ_HZ, ch_types=ch_types)

    # Deterministic data: simple patterns that make averages and differences easy to verify.
    t = np.arange(_N_TIMES, dtype=float)
    data = np.vstack(
        [
            1.0 + 0.1 * t,          # A1
            2.0 + 0.1 * t,          # A2
            4.0 + 0.2 * t,          # A3
            -1.0 + 0.0 * t,         # B1 (constant)
            np.zeros_like(t),       # stim
        ]
    )

    raw = mne.io.RawArray(data, info, verbose=False)

    # Add channel positions so bipolar midpoint geometry can be tested.
    # Units don't matter here; just deterministic 3D coordinates.
    ch_pos = {
        "A1": np.array([0.0, 0.0, 0.0]),
        "A2": np.array([0.0, 0.0, 2.0]),
        "A3": np.array([0.0, 0.0, 4.0]),
        "B1": np.array([10.0, 0.0, 0.0]),
        # Stim channel position omitted.
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage, on_missing="ignore")

    # Add annotations to ensure we preserve them.
    annot = mne.Annotations(onset=[0.01], duration=[0.02], description=["test_event"])
    raw.set_annotations(annot)

    return raw


@pytest.fixture()
def toy_geometry() -> _DummyGeometry:
    return _DummyGeometry(
        shafts={
            "A": ["A1", "A2", "A3"],
            "B": ["B1"],
        },
        neighbors={
            # Simple neighbor map for knn tests (not used by shaft_1d).
            "A1": ["A2"],
            "A2": ["A1", "A3"],
            "A3": ["A2"],
            "B1": [],
        },
    )


def _import_block_rereference():
    """
    Import the rereference block function.

    This helper makes the test file resilient to future refactors:
    - If the block API is renamed to `rereference_view`, update here.
    """
    try:
        from dcap.seeg.preprocessing.blocks.rereference import rereference as rereference_block  # type: ignore
        return rereference_block
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import dcap.seeg.preprocessing.blocks.rereference.rereference. "
            "If you refactored names, update _import_block_rereference()."
        ) from exc


# =============================================================================
#                     ########################################
#                     #                 CAR                  #
#                     ########################################
# =============================================================================
def test_car_global_mean_zero_over_reference_pool(toy_raw: mne.io.BaseRaw, toy_geometry: _DummyGeometry) -> None:
    rereference_block = _import_block_rereference()
    cfg = _DummyCfg(methods=("car",), car_scope="global")
    ctx = _DummyCtx(geometry=toy_geometry)

    views, artifact = rereference_block(toy_raw, cfg, ctx)

    raw_car = views["car"]
    data_car = raw_car.get_data()

    # Reference pool: seeg+ecog picks, excluding stim by pick_types.
    picks = mne.pick_types(raw_car.info, seeg=True, ecog=True, stim=False, eeg=False, meg=False)
    mean_over_pool = np.mean(data_car[picks, :], axis=0)

    npt.assert_allclose(mean_over_pool, 0.0, atol=_TOL)


def test_car_does_not_modify_stim_channel(toy_raw: mne.io.BaseRaw, toy_geometry: _DummyGeometry) -> None:
    rereference_block = _import_block_rereference()
    cfg = _DummyCfg(methods=("car",), car_scope="global")
    ctx = _DummyCtx(geometry=toy_geometry)

    views, _ = rereference_block(toy_raw, cfg, ctx)

    stim_idx = toy_raw.ch_names.index("STI 014")
    original = toy_raw.get_data()[stim_idx]
    after = views["car"].get_data()[stim_idx]

    npt.assert_allclose(after, original, atol=_TOL)


def test_car_by_shaft_mean_zero_per_shaft(toy_raw: mne.io.BaseRaw, toy_geometry: _DummyGeometry) -> None:
    rereference_block = _import_block_rereference()
    cfg = _DummyCfg(methods=("car",), car_scope="by_shaft")
    ctx = _DummyCtx(geometry=toy_geometry)

    views, _ = rereference_block(toy_raw, cfg, ctx)
    data_car = views["car"].get_data()

    name_to_idx = {name: i for i, name in enumerate(views["car"].ch_names)}
    for shaft_name, contacts in toy_geometry.shafts.items():
        if len(contacts) < 2:
            continue
        idxs = [name_to_idx[c] for c in contacts if c in name_to_idx]
        if not idxs:
            continue
        mean_over_shaft = np.mean(data_car[idxs, :], axis=0)
        npt.assert_allclose(mean_over_shaft, 0.0, atol=_TOL)


# =============================================================================
#                     ########################################
#                     #               Bipolar                #
#                     ########################################
# =============================================================================
def test_bipolar_data_is_difference(toy_raw: mne.io.BaseRaw, toy_geometry: _DummyGeometry) -> None:
    rereference_block = _import_block_rereference()
    cfg = _DummyCfg(methods=("bipolar",))
    ctx = _DummyCtx(geometry=toy_geometry)

    views, _ = rereference_block(toy_raw, cfg, ctx)

    raw_bip = views["bipolar"]
    assert "A1-A2" in raw_bip.ch_names

    a1 = toy_raw.get_data()[toy_raw.ch_names.index("A1")]
    a2 = toy_raw.get_data()[toy_raw.ch_names.index("A2")]
    expected = a1 - a2

    got = raw_bip.get_data()[raw_bip.ch_names.index("A1-A2")]
    npt.assert_allclose(got, expected, atol=_TOL)

    # Metadata invariants
    assert raw_bip.info["sfreq"] == toy_raw.info["sfreq"]
    assert raw_bip.n_times == toy_raw.n_times
    assert len(raw_bip.annotations) == len(toy_raw.annotations)


def test_bipolar_virtual_electrode_location_is_midpoint(toy_raw: mne.io.BaseRaw, toy_geometry: _DummyGeometry) -> None:
    """
    Desired behavior:
    positions for bipolar channels should be midpoints of endpoints when available.

    This test will fail until implementation explicitly assigns loc[:3] for bipolar channels.
    """
    rereference_block = _import_block_rereference()
    cfg = _DummyCfg(methods=("bipolar",))
    ctx = _DummyCtx(geometry=toy_geometry)

    views, _ = rereference_block(toy_raw, cfg, ctx)
    raw_bip = views["bipolar"]

    # Endpoint xyz from source raw
    def _xyz(raw: mne.io.BaseRaw, ch_name: str) -> np.ndarray:
        idx = raw.ch_names.index(ch_name)
        return np.array(raw.info["chs"][idx]["loc"][:3], dtype=float)

    expected_mid = 0.5 * (_xyz(toy_raw, "A1") + _xyz(toy_raw, "A2"))

    bip_idx = raw_bip.ch_names.index("A1-A2")
    got_mid = np.array(raw_bip.info["chs"][bip_idx]["loc"][:3], dtype=float)

    npt.assert_allclose(got_mid, expected_mid, atol=1e-6)


# =============================================================================
#                     ########################################
#                     #              Laplacian               #
#                     ########################################
# =============================================================================
def test_laplacian_shaft_1d_matches_formula(toy_raw: mne.io.BaseRaw, toy_geometry: _DummyGeometry) -> None:
    rereference_block = _import_block_rereference()
    cfg = _DummyCfg(methods=("laplacian",), laplacian_mode="shaft_1d")
    ctx = _DummyCtx(geometry=toy_geometry)

    views, _ = rereference_block(toy_raw, cfg, ctx)
    raw_lap = views["laplacian"]

    data = toy_raw.get_data()
    a1 = data[toy_raw.ch_names.index("A1")]
    a2 = data[toy_raw.ch_names.index("A2")]
    a3 = data[toy_raw.ch_names.index("A3")]

    expected_a2 = a2 - 0.5 * (a1 + a3)
    got_a2 = raw_lap.get_data()[raw_lap.ch_names.index("A2")]
    npt.assert_allclose(got_a2, expected_a2, atol=_TOL)

    # Edge policy: A1 and A3 unchanged for shaft_1d
    got_a1 = raw_lap.get_data()[raw_lap.ch_names.index("A1")]
    got_a3 = raw_lap.get_data()[raw_lap.ch_names.index("A3")]
    npt.assert_allclose(got_a1, a1, atol=_TOL)
    npt.assert_allclose(got_a3, a3, atol=_TOL)


# =============================================================================
#                     ########################################
#                     #               WM ref                 #
#                     ########################################
# =============================================================================
def test_wm_reference_map_subtracts_specified_wm(toy_raw: mne.io.BaseRaw, toy_geometry: _DummyGeometry) -> None:
    rereference_block = _import_block_rereference()
    cfg = _DummyCfg(methods=("wm_ref",))
    ctx = _DummyCtx(
        geometry=toy_geometry,
        decisions={"wm_reference_map": {"A1": "B1"}},
    )

    views, _ = rereference_block(toy_raw, cfg, ctx)
    raw_wm = views["wm_ref"]

    data = toy_raw.get_data()
    a1 = data[toy_raw.ch_names.index("A1")]
    b1 = data[toy_raw.ch_names.index("B1")]

    expected = a1 - b1
    got = raw_wm.get_data()[raw_wm.ch_names.index("A1")]

    npt.assert_allclose(got, expected, atol=_TOL)


# =============================================================================
#                     ########################################
#                     #            General invariants         #
#                     ########################################
# =============================================================================
@pytest.mark.parametrize("method", ["car", "laplacian", "wm_ref"])
def test_methods_preserve_timebase_and_annotations(
    toy_raw: mne.io.BaseRaw,
    toy_geometry: _DummyGeometry,
    method: str,
) -> None:
    rereference_block = _import_block_rereference()

    if method == "car":
        cfg = _DummyCfg(methods=(method,), car_scope="global")
        ctx = _DummyCtx(geometry=toy_geometry)
    elif method == "laplacian":
        cfg = _DummyCfg(methods=(method,), laplacian_mode="shaft_1d")
        ctx = _DummyCtx(geometry=toy_geometry)
    else:
        cfg = _DummyCfg(methods=(method,))
        ctx = _DummyCtx(geometry=toy_geometry, decisions={"wm_contacts": ["B1"]})

    views, _ = rereference_block(toy_raw, cfg, ctx)

    raw_out = views[method]
    assert raw_out.info["sfreq"] == toy_raw.info["sfreq"]
    assert raw_out.n_times == toy_raw.n_times
    assert len(raw_out.annotations) == len(toy_raw.annotations)
    assert np.isfinite(raw_out.get_data()).all()
