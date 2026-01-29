from dataclasses import dataclass
from typing import Literal, Tuple

RereferenceMethod = Literal["car", "bipolar", "laplacian", "wm_ref"]
CarScope = Literal["global", "by_shaft"]
LaplacianMode = Literal["shaft_1d", "knn_3d"]

@dataclass(frozen=True)
class RereferenceConfig:
    methods: Tuple[RereferenceMethod, ...] = ("car",)
    car_scope: CarScope = "global"
    laplacian_mode: LaplacianMode = "shaft_1d"
    keep_original: bool = True

    def __post_init__(self) -> None:
        if len(self.methods) == 0:
            raise ValueError("methods must contain at least one rereferencing method.")
