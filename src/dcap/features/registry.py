from __future__ import annotations

from typing import Callable, Dict, Mapping

from dcap.features.base import FeatureComputer

FeatureFactory = Callable[[], FeatureComputer]

_FEATURES: Dict[str, FeatureFactory] = {}


def register_feature(name: str, factory: FeatureFactory) -> None:
    if name in _FEATURES:
        raise ValueError(f"Feature already registered: {name}")
    _FEATURES[name] = factory


def get_feature(name: str) -> FeatureComputer:
    if name not in _FEATURES:
        available = ", ".join(sorted(_FEATURES))
        raise KeyError(f"Unknown feature '{name}'. Available: {available}")
    return _FEATURES[name]()


def list_features() -> Mapping[str, FeatureFactory]:
    return dict(_FEATURES)
