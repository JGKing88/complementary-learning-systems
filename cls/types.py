from __future__ import annotations

from typing import Callable, List, Tuple


Position = Tuple[int, int]

# fov_fn takes (distance_to_wall, grid_size) -> visible_width
FovFunction = Callable[[int, int], int]

# Vector action and heading representations (dx, dy)
Vector2 = Tuple[int, int]


