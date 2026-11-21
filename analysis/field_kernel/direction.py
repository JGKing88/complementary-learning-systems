"""A fast stand-in for the probe's direction test.

    python -m analysis.field_kernel.direction --spec "ideal:r=23.2,relu_mix=1,active=0.1" --step 1 4

One trial: a goal g and a cue cell x in the same 20-cell box (2 to 14 cells apart), plus K - 1
other goals at random scaffold cells. Hebbian recall in one step is ``q = sum_k xi_k (xi_k . z(x))``.
The direction estimate is q projected on the local basis, as in production: differences of the code
one cell North and East of x (``step = 1``), Gram-Schmidt orthonormalised; ``step = s > 1`` uses the
centred difference over +- s cells. Reported: the mean absolute angle between the estimate and
g - x, and the share within 45 degrees.

``walk`` adds a greedy walk: from x, step one cell along the estimate's larger component (the cue
moves, the memory stays) and count the walks that reach the goal within 3 times the start distance.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from analysis.field_kernel.measure import build

NPOS = 1716


def basis(enc, x, y, step):
    if step > 1:
        dn = enc.closed_form(x, y + step) - enc.closed_form(x, y - step)
        de = enc.closed_form(x + step, y) - enc.closed_form(x - step, y)
    else:
        c = enc.closed_form(x, y)
        dn = enc.closed_form(x, y + 1) - c
        de = enc.closed_form(x + 1, y) - c
    # Gram-Schmidt as production: North kept, East reduced
    n = dn / np.maximum(np.linalg.norm(dn, axis=1, keepdims=True), 1e-12)
    e = de - (de * n).sum(1, keepdims=True) * n
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return e, n


def estimate(enc, mem, x, y, step):
    """``mem`` is (trials, K, D). Returns the (East, North) estimate per trial."""
    z = enc.closed_form(x, y)
    q = np.einsum("tkd,tk->td", mem, np.einsum("tkd,td->tk", mem, z))
    e, n = basis(enc, x, y, step)
    return np.stack([(q * e).sum(1), (q * n).sum(1)], axis=1)


def run(spec: str, *, step: int = 1, K: int = 5, trials: int = 1500, seed: int = 0,
        dmin: float = 2.0, dmax: float = 14.0, walk: bool = False) -> dict:
    enc = build(spec)
    rng = np.random.default_rng(seed)
    gx = rng.integers(100, NPOS - 100, trials)
    gy = rng.integers(100, NPOS - 100, trials)
    d = rng.uniform(dmin, dmax, trials)
    th = rng.uniform(0, 2 * math.pi, trials)
    x = gx - np.rint(d * np.cos(th)).astype(np.int64)
    y = gy - np.rint(d * np.sin(th)).astype(np.int64)
    ox = rng.integers(0, NPOS, (trials, K - 1))
    oy = rng.integers(0, NPOS, (trials, K - 1))
    mem = np.stack([enc.closed_form(gx, gy)]
                   + [enc.closed_form(ox[:, k], oy[:, k]) for k in range(K - 1)], axis=1)
    est = estimate(enc, mem, x, y, step)
    true = np.arctan2(gy - y, gx - x)
    err = np.abs(np.angle(np.exp(1j * (np.arctan2(est[:, 1], est[:, 0]) - true))))
    out = {"spec": spec, "step": step, "K": K, "D": enc.out_dim,
           "err_deg": float(np.degrees(err.mean())), "acc45": float((err < math.pi / 4).mean())}
    if walk:
        px, py = x.copy(), y.copy()
        d0 = np.hypot(gx - x, gy - y)
        done = np.zeros(trials, bool)
        for _ in range(int(3 * dmax)):
            est = estimate(enc, mem, px, py, step)
            horiz = np.abs(est[:, 0]) >= np.abs(est[:, 1])
            sx = np.where(horiz, np.sign(est[:, 0]), 0).astype(np.int64)
            sy = np.where(~horiz, np.sign(est[:, 1]), 0).astype(np.int64)
            px = np.where(done, px, px + sx)
            py = np.where(done, py, py + sy)
            done |= (np.abs(px - gx) + np.abs(py - gy)) <= 1
        out["reach"] = float(done.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--step", type=int, nargs="+", default=[1])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--walk", action="store_true")
    a = ap.parse_args()
    for spec in a.spec:
        for s in a.step:
            o = run(spec, step=s, K=a.K, trials=a.trials, walk=a.walk)
            print(f"{spec}  step {s}:  err {o['err_deg']:5.1f} deg  acc45 {o['acc45']:.3f}"
                  + (f"  reach {o['reach']:.3f}" if a.walk else ""), flush=True)


if __name__ == "__main__":
    main()
