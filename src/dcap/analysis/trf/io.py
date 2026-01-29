# =============================================================================
#                          Analysis: TRF (I/O)
# =============================================================================
#
# Persistence utilities for TRF results.
#
# - Use .npz for arrays (weights, lags, metrics)
# - Use .json for lightweight metadata/provenance
#
# IMPORTANT: Avoid writing sensitive identifiers into shareable artifacts.
#
# =============================================================================

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from dcap.analysis.trf.types import TrfResult


def save_trf_result(
    result: TrfResult,
    npz_path: str | Path,
    metadata_json_path: str | Path | None = None,
) -> None:
    """Save a :class:`~dcap.analysis.trf.types.TrfResult` to disk.

    Parameters
    ----------
    result
        TRF result to save.
    npz_path
        Output .npz path for numeric arrays.
    metadata_json_path
        Optional output .json for `result.metadata`. If None, no JSON is written.

    Usage example
    -------------
        from pathlib import Path
        from dcap.analysis.trf import save_trf_result

        save_trf_result(result, npz_path=Path("out/trf.npz"), metadata_json_path=Path("out/trf.json"))
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, Any] = {
        "weights": result.weights,
        "lags_s": result.lags_s,
        "alpha": np.array([result.alpha], dtype=float),
    }

    if result.metrics is not None:
        for key, value in result.metrics.items():
            arrays[f"metric__{key}"] = value

    np.savez_compressed(npz_path, **arrays)

    if metadata_json_path is not None:
        metadata_json_path = Path(metadata_json_path)
        metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path=metadata_json_path, payload=dict(result.metadata or {}))


def load_trf_result(
    npz_path: str | Path,
    metadata_json_path: str | Path | None = None,
) -> TrfResult:
    """Load a :class:`~dcap.analysis.trf.types.TrfResult` from disk.

    Parameters
    ----------
    npz_path
        Input .npz path produced by :func:`save_trf_result`.
    metadata_json_path
        Optional JSON file containing metadata.

    Returns
    -------
    result
        Loaded TRF result.

    Usage example
    -------------
        from dcap.analysis.trf import load_trf_result
        result = load_trf_result("out/trf.npz", metadata_json_path="out/trf.json")
    """
    npz_path = Path(npz_path)

    with np.load(npz_path, allow_pickle=False) as npz:
        weights = npz["weights"]
        lags_s = npz["lags_s"]
        alpha_arr = npz["alpha"]
        alpha = float(alpha_arr.ravel()[0])

        metrics: Dict[str, Any] = {}
        for key in npz.files:
            if key.startswith("metric__"):
                metrics[key.replace("metric__", "", 1)] = npz[key]

    metadata: Optional[Mapping[str, Any]] = None
    if metadata_json_path is not None:
        metadata = _read_json(path=Path(metadata_json_path))

    return TrfResult(
        weights=weights,
        lags_s=lags_s,
        alpha=alpha,
        metrics=metrics or None,
        metadata=metadata,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
