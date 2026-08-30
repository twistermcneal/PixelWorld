from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Landscape:
    prompt: str
    seed: int
    biome: str
    terrain: np.ndarray
    regions: np.ndarray
    vegetation: np.ndarray
    rgb: np.ndarray
    object_map: np.ndarray
    walkable: np.ndarray
    interaction: np.ndarray
    terrain_params: tuple
    objects: dict[int, dict[str, Any]]
