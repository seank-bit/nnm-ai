from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EmbeddingPayload:
    dense: NDArray[np.float32]
    sparse: list[dict[int, float]]
    colbert: list[NDArray[np.float32]]

    def __post_init__(self) -> None:
        n = self.dense.shape[0]
        if len(self.sparse) != n or len(self.colbert) != n:
            raise ValueError(
                f"dimension mismatch: dense={n}, sparse={len(self.sparse)}, "
                f"colbert={len(self.colbert)}"
            )
