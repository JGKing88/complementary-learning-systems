"""Training layouts for the corner-layout sweep (plan sec 6.5).

    python -m analysis.corner_layouts --out_dir $CLS_RUNS/layouts

64 arenas of 20x20 inside rect:0,0,400,400 -- A1x's count, so the number of
training cells is fixed -- arranged so the coordinate values they show differ:

  block_p20   8x8 lattice, pitch 20: one contiguous block, 160 values per axis
  grid_p30    8x8, pitch 30: 160 values per axis in 8 bands, 10-cell gaps
  grid_p50    8x8, pitch 50: 160 values per axis in 8 bands, 30-cell gaps
  perm_s3     arena i at (3i, 3 pi(i)): 64 staggered bands, 209 contiguous values
  perm_s6     arena i at (6i, 6 pi(i)): 64 staggered bands, 398 contiguous values
              (every X and Y value of the corner, with A1x's 64 arenas)

Each layout is written as a JSON list of [x, y] offsets for --place_offsets;
the summary prints distinct values and contiguous bands per axis and the
distinct cells covered.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SIZE, N, CORNER = 20, 64, 400


def layouts(seed: int = 0) -> dict[str, list[tuple[int, int]]]:
    perm = np.random.RandomState(seed).permutation(N)
    grid = lambda p: [(p * i, p * j) for i in range(8) for j in range(8)]
    return {
        "block_p20": grid(20),
        "grid_p30": grid(30),
        "grid_p50": grid(50),
        "perm_s3": [(3 * i, 3 * int(perm[i])) for i in range(N)],
        "perm_s6": [(6 * i, 6 * int(perm[i])) for i in range(N)],
    }


def axis_stats(vals) -> tuple[int, int]:
    seen = np.zeros(CORNER, bool)
    for v in vals:
        seen[v:v + SIZE] = True
    runs = int(np.sum(seen[1:] & ~seen[:-1]) + seen[0])
    return int(seen.sum()), runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name, offs in layouts(a.seed).items():
        assert len(offs) == N and all(0 <= x and x + SIZE <= CORNER and 0 <= y and y + SIZE <= CORNER
                                      for x, y in offs), name
        cells = np.zeros((CORNER, CORNER), bool)
        for x, y in offs:
            cells[x:x + SIZE, y:y + SIZE] = True
        xv, xr = axis_stats([o[0] for o in offs])
        yv, yr = axis_stats([o[1] for o in offs])
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as f:
            json.dump([[int(x), int(y)] for x, y in offs], f)
        print(f"{name:10s} X {xv:3d} values in {xr:2d} bands | Y {yv:3d} in {yr:2d} | "
              f"distinct cells {int(cells.sum()):,}")


if __name__ == "__main__":
    main()
