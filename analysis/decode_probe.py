"""Where does a memoryless displacement decode work? Synthetic code pairs, no scaffold.

    python -m analysis.decode_probe RUN_DIR [RUN_DIR ...] [--far 700,700,500,500] [--range]

For each run (a `train_goal_pairs` PairRegressor or a `train_goal_lifetimes`
`dist`-arm agent) the direction error, in degrees, on (p, g) pairs with
Chebyshev |g - p| in [1, 19] whose endpoints both lie in

  inside   the run's declared place rect (the corner), any position in it
  far      `--far` rect, far from the corner on both axes
  bands    inside the corner, split by whether the pair's X and Y values
           were ever covered by a training env's footprint (fixed placement
           only; under translation the whole corner is covered)

Codes are synthesised with `gbook_at` on the run's own lattice: theta is the
anchor orientation when the run trained at a single one (`lattice_mix_theta`
with `mix_standard_frac` 1), else 0. `--range` adds error by |Delta| beyond
the trained 19 (plan sec 1.5). This is the probe behind the A1x band result
(2026-09-15) and the dense-tiling control A1xd (2026-09-16).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from gridcode.lattice import gbook_at
from hopfield_nav.policy.agent_rnn import RNNAgent
from hopfield_nav.policy.pair_regressor import PairRegressor, normalize_direction
from hopfield_nav.training.goal_pairs_setup import ARMS, agent_cfg_for_mode, parse_rect

RANGE_DS = [1, 5, 10, 15, 19, 20, 22, 24, 26, 30, 35, 40, 50, 60]


class Decoder:
    """One callable `(p, delta) -> unit direction` from either checkpoint kind."""

    def __init__(self, run_dir: str, theta_override: float | None = None, ckpt: str = ""):
        """`run_dir` may carry a checkpoint file after a colon (`RUN:pairs_u6000.pt`)."""
        if ":" in run_dir:
            run_dir, ckpt = run_dir.split(":", 1)
        self.run = os.path.basename(run_dir.rstrip("/")) + (f":{ckpt}" if ckpt else "")
        pairs = os.path.join(run_dir, ckpt or "pairs_final.pt")
        life = os.path.join(run_dir, ckpt or "life_final.pt")
        path = pairs if os.path.exists(pairs) else life
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["argv"]
        self.lambdas = list(a["lambdas"])
        self.fwhm = float(a["fwhm_ratio"])
        self.size = int(a["size"])
        if "model_state_dict" in ck:
            m = PairRegressor(ck["input_dim"], a["hidden_size"], a["num_layers"], a["movement_mode"],
                              nonlinearity=a["nonlinearity"], dropout=a.get("dropout", 0.0))
            m.load_state_dict(ck["model_state_dict"])
            self._fwd = m.predict_direction
        else:
            if a.get("arm") != "dist":
                raise SystemExit(f"{self.run}: only the memoryless `dist` arm is a decode; got {a.get('arm')}")
            acfg = agent_cfg_for_mode("grid", "continuous", hidden_size=a["hidden_size"],
                                      num_rnn_layers=a["num_layers"], rnn_nonlinearity=a["nonlinearity"],
                                      init_log_std=a["init_log_std"], **ARMS["dist"])
            m = RNNAgent(acfg, ck["input_dim"])
            m.load_state_dict(ck["agent_state_dict"])

            def fwd(x):
                dist, _ = m(x[:, None, :], None)
                return normalize_direction(dist.mean[:, 0])
            self._fwd = fwd
        m.eval()
        # Orientation: the anchor when the run trained at exactly one, else standard.
        theta = 0.0
        if a.get("lattice_theta_random") and float(a.get("lattice_mix_standard_frac", 0.0)) >= 1.0:
            theta = np.radians(float(a.get("lattice_mix_theta_deg", 0.0)))
        self.theta = theta if theta_override is None else np.radians(theta_override)
        self.translated = bool(a.get("lattice_theta_random")) and bool(a.get("lattice_translate", False))
        # Corner and training footprints from the world record.
        self.rect = None
        self.offsets = None
        wpath = os.path.join(run_dir, "world.json")
        if os.path.exists(wpath):
            with open(wpath) as f:
                w = json.load(f)["split"]
            pl = w["domains"]["place"]
            if pl.get("kind") == "Rect":
                self.rect = (int(pl["x0"]), int(pl["y0"]), int(pl["w"]), int(pl["h"]))
            self.offsets = np.array([t["offset"] for t in w["train"]], dtype=int)

    def __call__(self, p: np.ndarray, delta: np.ndarray) -> np.ndarray:
        gp = gbook_at(p, self.lambdas, self.fwhm, self.theta)
        gg = gbook_at(p + delta, self.lambdas, self.fwhm, self.theta)
        x = torch.from_numpy(np.concatenate([gp, gg], 1).astype(np.float32))
        with torch.no_grad():
            return self._fwd(x).numpy()

    def seen_axes(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Boolean masks over the corner's X and Y values covered by a training footprint."""
        if self.rect is None or self.offsets is None or self.translated:
            return None
        x0, y0, w, h = self.rect
        sx = np.zeros(w, bool)
        sy = np.zeros(h, bool)
        for ox, oy in self.offsets:
            sx[ox - x0:ox - x0 + self.size] = True
            sy[oy - y0:oy - y0 + self.size] = True
        return sx, sy


def angular_error(pred: np.ndarray, delta: np.ndarray) -> np.ndarray:
    u = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((pred * u).sum(1), -1.0, 1.0)))


def pairs_in(rng, xs: np.ndarray, ys: np.ndarray, n: int, max_abs: int, period: int = 1716):
    """`n` candidate pairs, `p` on the given X and Y values, `Delta` uniform in
    the Chebyshev ball of radius `max_abs` (minus 0); keeps pairs whose `g` also
    lands on those values."""
    xm = np.zeros(period, bool)
    ym = np.zeros(period, bool)
    xm[np.asarray(xs) % period] = True
    ym[np.asarray(ys) % period] = True
    p = np.stack([rng.choice(xs, n), rng.choice(ys, n)], 1).astype(float)
    d = rng.randint(-max_abs, max_abs + 1, size=(n, 2)).astype(float)
    g = (p + d).astype(int) % period
    ok = (np.abs(d).max(1) >= 1) & xm[g[:, 0]] & ym[g[:, 1]]
    return p[ok], d[ok]


def pairs_at(rng, d: int, lo: np.ndarray, hi: np.ndarray, n: int):
    """Chebyshev distance exactly `d`, both endpoints inside [lo, hi) on each axis."""
    ax = rng.randint(0, 2, size=n)
    sgn = rng.choice([-1, 1], size=n)
    other = rng.randint(-d, d + 1, size=n)
    delta = np.zeros((n, 2))
    delta[np.arange(n), ax] = sgn * d
    delta[np.arange(n), 1 - ax] = other
    p = np.stack([rng.randint(lo[i] + d, hi[i] - d, size=n) for i in range(2)], 1).astype(float)
    return p, delta


def report(dec: Decoder, far, n: int, max_abs: int, do_range: bool, rng):
    print(f"\n== {dec.run}  theta {np.degrees(dec.theta):.0f} deg"
          + (f"  corner {dec.rect}" if dec.rect else "  (no place rect)")
          + ("  translation-randomised" if dec.translated else "  fixed placement"))
    regions = []
    if dec.rect:
        x0, y0, w, h = dec.rect
        regions.append(("inside corner", np.arange(x0, x0 + w), np.arange(y0, y0 + h)))
    regions.append((f"far {far}", np.arange(far[0], far[0] + far[2]), np.arange(far[1], far[1] + far[3])))
    for name, xs, ys in regions:
        p, d = pairs_in(rng, xs, ys, n, max_abs)
        e = angular_error(dec(p, d), d)
        print(f"  {name:28s} mean {e.mean():6.1f}  median {np.median(e):6.1f}  (n={len(e)})")
    seen = dec.seen_axes()
    if seen is not None:
        sx, sy = seen
        x0, y0, w, h = dec.rect
        print(f"  training footprints cover X {sx.sum()}/{w}, Y {sy.sum()}/{h} of the corner's values")
        for name, mx, my in [("X seen, Y seen", sx, sy), ("X seen, Y unseen", sx, ~sy),
                             ("X unseen, Y seen", ~sx, sy), ("X unseen, Y unseen", ~sx, ~sy)]:
            if mx.sum() == 0 or my.sum() == 0:
                print(f"  {name:28s} (no such values)")
                continue
            p, d = pairs_in(rng, np.where(mx)[0] + x0, np.where(my)[0] + y0, n, max_abs)
            if len(d) == 0:
                print(f"  {name:28s} (no pairs within {max_abs})")
                continue
            e = angular_error(dec(p, d), d)
            print(f"  {name:28s} mean {e.mean():6.1f}  median {np.median(e):6.1f}  (n={len(e)})")
    if do_range:
        lo = np.array(far[:2])
        hi = lo + np.array(far[2:])
        print("  by |Delta| (Chebyshev), far rect:", end="")
        for d in RANGE_DS:
            p, delta = pairs_at(rng, d, lo, hi, n)
            print(f"  {d}:{angular_error(dec(p, delta), delta).mean():.0f}", end="")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories under agent_ckpts")
    ap.add_argument("--far", type=str, default="700,700,500,500", help="X0,Y0,W,H of the far rect")
    ap.add_argument("--theta", type=float, default=None, help="degrees; override the run's own orientation")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--max_abs", type=int, default=19)
    ap.add_argument("--range", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    far = tuple(int(v) for v in parse_rect(args.far))
    for run in args.runs:
        dec = Decoder(run, args.theta)
        report(dec, far, args.n, args.max_abs, args.range, np.random.RandomState(args.seed))


if __name__ == "__main__":
    main()
