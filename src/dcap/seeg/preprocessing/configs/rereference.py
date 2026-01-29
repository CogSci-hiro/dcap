# =============================================================================
# =============================================================================
#                     ########################################
#                     #         CONFIG: REREFERENCING        #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Literal, Tuple


RereferenceMethod = Literal["car", "bipolar", "laplacian", "wm_ref"]
CarScope = Literal["global", "by_shaft"]
LaplacianMode = Literal["shaft_1d", "knn_3d"]


@dataclass(frozen=True)
class RereferenceConfig:
    """
    Configuration for rereferencing methods.

    Attributes
    ----------
    methods
        Tuple of methods to generate. "original" is always available outside this block.
    car_scope
        Common average reference scope when "car" is selected.
    laplacian_mode
        Laplacian mode when "laplacian" is selected.
    keep_original
        If True, downstream pipelines should keep the original monopolar view alongside reref views.

    Usage example
    -------------
        cfg = RereferenceConfig(
            methods=("car", "bipolar", "laplacian"),
            car_scope="by_shaft",
            laplacian_mode="shaft_1d",
        )
    """

    methods: Tuple[RereferenceMethod, ...] = ("car",)
    car_scope: CarScope = "global"
    laplacian_mode: LaplacianMode = "shaft_1d"
    keep_original: bool = True

    def __post_init__(self) -> None:
        if len(self.methods) == 0:
            raise ValueError("methods must contain at least one rereferencing method.")
