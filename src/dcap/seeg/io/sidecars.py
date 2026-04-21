from pathlib import Path
from typing import Iterable, Optional


_ENTITY_SUFFIXES = ("_eeg", "_ieeg", "_meg", "_raw", "-raw")


def iter_neighbor_sidecar_candidates(data_path: Path, *, sidecar_suffix: str) -> Iterable[Path]:
    """
    Yield strict neighboring sidecar candidates for a BIDS-like recording path.

    The first candidate preserves the immediate stem. Later candidates strip all
    suffixes and common terminal datatype/output tokens so files such as
    ``*_raw.fif`` can still resolve ``*_channels.tsv`` or ``*_events.tsv``.
    """
    seen: set[Path] = set()

    bases = [data_path.stem, _strip_all_suffixes(data_path.name)]
    normalized_base = _strip_terminal_entity_suffix(_strip_all_suffixes(data_path.name))
    if normalized_base not in bases:
        bases.append(normalized_base)

    for base in bases:
        candidate = data_path.parent / f"{base}{sidecar_suffix}"
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def find_neighbor_sidecar(data_path: Path, *, sidecar_suffix: str) -> Optional[Path]:
    for candidate in iter_neighbor_sidecar_candidates(data_path, sidecar_suffix=sidecar_suffix):
        if candidate.exists():
            return candidate
    return None


def _strip_all_suffixes(name: str) -> str:
    base = name
    while True:
        suffix = Path(base).suffix
        if suffix == "":
            return base
        base = Path(base).stem


def _strip_terminal_entity_suffix(base: str) -> str:
    for suffix in _ENTITY_SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base
