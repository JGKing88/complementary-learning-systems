"""Where on the scaffold the measurements are taken: tracks, loops, arenas, rings.

All coordinates are unwrapped integers; ``Population.code`` reduces them mod
``Npos``. Nothing here encodes anything.
"""
from __future__ import annotations

import numpy as np


def loop(npos: int, axis: int, row: int) -> tuple[np.ndarray, np.ndarray]:
    """One full scaffold row (``axis=0``, along x) or column (``axis=1``).

    Every code built on the grid code is periodic with period ``npos`` along
    each axis, so this is a closed loop and circular lags are exact.
    """
    t = np.arange(npos)
    r = np.full(npos, row)
    return (t, r) if axis == 0 else (r, t)


def serpentine(x0: int, y0: int, length: int, *, run: int, gap: int,
               axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A continuous zigzag track ``length`` cells long.

    It runs ``run`` cells along ``axis``, steps ``gap`` cells sideways, runs
    back, and so on, so consecutive straights are ``gap`` cells apart. Rich
    2014's 48 m track was a zigzag of about 1 m segments; here the straights
    are as long as the scaffold period, which keeps turns rare. A track
    shorter than ``run`` is one straight.
    """
    steps = []
    sgn = 1
    while len(steps) < length - 1:
        steps += [(sgn, 0)] * (run - 1)
        steps += [(0, 1)] * gap
        sgn = -sgn
    steps = np.asarray(steps[:length - 1], dtype=np.int64).reshape(-1, 2)
    path = np.concatenate([np.zeros((1, 2), np.int64), np.cumsum(steps, axis=0)])
    a, b = path[:, 0], path[:, 1]
    xs, ys = (a, b) if axis == 0 else (b, a)
    return xs + x0, ys + y0


def serpentine_capacity(npos: int, run: int, gap: int) -> int:
    """Longest serpentine whose sideways extent stays inside one scaffold period."""
    n_turns = (npos - 1) // gap
    return (n_turns + 1) * run + n_turns * gap


def arena(x0: int, y0: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    """A ``side x side`` square, in ``(row, col) = (y, x)`` raster order."""
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return xx.reshape(-1) + x0, yy.reshape(-1) + y0


def torus_distance(a: np.ndarray, b: np.ndarray, npos: int) -> np.ndarray:
    d = np.abs(a - b) % npos
    return np.minimum(d, npos - d)


def spread_offsets(n: int, side: int, npos: int, min_sep: int, seed: int,
                   tries: int = 20000, strict: bool = True) -> list[tuple[int, int]]:
    """``n`` arena corners whose arenas sit at least ``min_sep`` cells apart on the torus.

    With ``strict=False``, a layout that cannot fit (large arenas on the
    1716-cell torus) falls back to corners spaced evenly along the diagonal
    from a random start, which may overlap; ``spread_ok`` reports which.
    """
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int]] = []
    for _ in range(tries):
        c = rng.integers(0, npos, 2)
        ok = True
        for o in out:
            dx = torus_distance(np.array(c[0]), np.array(o[0]), npos)
            dy = torus_distance(np.array(c[1]), np.array(o[1]), npos)
            if max(int(dx), int(dy)) < side + min_sep:
                ok = False
                break
        if ok:
            out.append((int(c[0]), int(c[1])))
            if len(out) == n:
                return out
    if strict:
        raise RuntimeError(f"could not place {n} arenas of side {side} {min_sep} apart")
    x0, y0 = (int(v) for v in rng.integers(0, npos, 2))
    return [((x0 + i * npos // n) % npos, (y0 + i * npos // n) % npos) for i in range(n)]


def spread_ok(offsets, side: int, npos: int) -> bool:
    """True if no two arenas share a cell."""
    for i, a in enumerate(offsets):
        for b in offsets[i + 1:]:
            dx = int(torus_distance(np.array(a[0]), np.array(b[0]), npos))
            dy = int(torus_distance(np.array(a[1]), np.array(b[1]), npos))
            if max(dx, dy) < side:
                return False
    return True


def rings(radii: np.ndarray, n_angles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer offsets on circles of the given radii, and their exact lengths."""
    th = np.arange(n_angles) * 2 * np.pi / n_angles
    dx = np.rint(np.outer(radii, np.cos(th))).astype(np.int64).reshape(-1)
    dy = np.rint(np.outer(radii, np.sin(th))).astype(np.int64).reshape(-1)
    return dx, dy, np.hypot(dx, dy)


__all__ = ["arena", "loop", "rings", "serpentine", "serpentine_capacity",
           "spread_offsets", "torus_distance"]
