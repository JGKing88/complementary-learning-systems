"""Screen every trained arm in the campaign on the metric that predicts reach.

`nav_screen_check.py` scores a hand-picked shortlist. This walks the sweep
directories instead, because the point of §10.4 is that the campaign's arms were
selected on `r_min` and most of them have never been scored on anything else.
`attract_lambda` is the clearest case: the alias rate is monotone *upward* from
Level 6's 2.0, so w52/w53/w54 spent three waves walking the wrong way, and the
untested-for-navigation direction is down -- where w52 already has checkpoints.

Two numbers per arm per gain:

  far>.25  fraction of pairs at least 200 cells apart above cosine 0.25, the
           line that separates dead goals from live ones (§10.3). Lower better.
  res90    median cells before cosine to a reference falls below 0.9. The guard
           on the other side: gain buys the alias rate by spending exactly this.

Both are properties of the encoder alone -- no Hopfield, no beta -- so they are
unchanged by the `beta = gain` operating point.

One seed per arm here; the campaign's own method note is that seed spread is
3-5 radius units and two seeds has reversed twice, so treat the ordering as a
shortlist to probe, not a result.
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

RUNS = "/orcd/pool/003/jackking/cls_runs"
S = RUNS + "/sweeps"
NPOS = 1716
GAINS = [100, 300, 1000]
SEED = "seed=42"

# Waves worth screening, newest first. w21 is out of brief (its spread term sees
# the whole arena) and is here only as the bracket §10.6 item 4 asks for.
WAVES = ["w54_attract_far", "w53_attract_knee", "w52_attract_fwhm",
         "w49_g100_knee", "w50_g5_control", "w39_batch_pairs",
         "w45_g100_knobs", "w46_g100_spread", "w47_g100_capacity",
         "w48_g100_nospread", "w51_steps", "w21_arena_spread"]

OOB = {"w21_arena_spread"}


def arms() -> list[tuple[str, str]]:
    out = [("v35 (out of brief)",
            RUNS + "/encoders/run_20260422_185816/encoder_best.pt")]
    for w in WAVES:
        for d in sorted(glob.glob(f"{S}/{w}/*{SEED}")):
            ck = os.path.join(d, "encoder_final.pt")
            if not os.path.exists(ck):
                continue
            base = os.path.basename(d)
            arm = base.split("_", 1)[1].rsplit("_seed", 1)[0]
            tag = w.split("_")[0]
            lab = f"{tag} {arm}" + (" *" if w in OOB else "")
            out.append((lab, ck))
    return out


def main() -> None:
    rng = np.random.RandomState(7)
    n = 6000
    px, py = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
    qx, qy = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
    far = np.hypot(px - qx, py - qy) > 200
    fx, fy, gx, gy = px[far], py[far], qx[far], qy[far]

    nref = 250
    rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
    offs = np.arange(1, 31)

    rows = []
    for lab, ck in arms():
        try:
            enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
        except Exception as exc:                       # unloadable checkpoint
            print(f"  skip {lab}: {type(exc).__name__}", file=sys.stderr)
            continue
        field = Field(enc, list(cfg.lambdas), fwhm, own, NPOS)
        raw = torch.load(ck, map_location="cpu", weights_only=False)
        ur = raw.get("unique_radius")
        rmin = ur.get("r_min") if isinstance(ur, dict) else None

        cells = {}
        for g in GAINS:
            enc.gain = float(g)
            A = field.encode(fx, fy).astype(np.float64)
            B = field.encode(gx, gy).astype(np.float64)
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            B /= np.linalg.norm(B, axis=1, keepdims=True)
            alias = float(((A * B).sum(1) > 0.25).mean())

            R = field.encode(rx, ry).astype(np.float64)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            prof = np.empty((nref, len(offs)))
            for i, o in enumerate(offs):
                Q = field.encode(np.clip(rx + o, 0, NPOS - 1), ry)
                Q = Q.astype(np.float64)
                Q /= np.linalg.norm(Q, axis=1, keepdims=True)
                prof[:, i] = (R * Q).sum(1)
            res = [offs[b[0]] if (b := np.flatnonzero(r < 0.9)).size
                   else offs[-1] + 1 for r in prof]
            cells[g] = (alias, float(np.median(res)))
        rows.append((lab, own, rmin, cells))

    # Rank by the best gain that still clears the res90 >= 8 guard of §10.6.
    def score(row):
        best = min((c[0] for g, c in row[3].items() if c[1] >= 8),
                   default=None)
        return (best is None, best if best is not None else 1.0)

    rows.sort(key=score)

    print(f"far pairs {int(far.sum())}, refs {nref}, one seed per arm "
          f"({SEED})\n")
    hdr = f"{'arm':26s}{'own g':>7s}{'r_min':>7s}  "
    sub = f"{'':26s}{'':7s}{'':7s}  "
    for g in GAINS:
        hdr += f"{'gain ' + str(g):>19s}"
        sub += f"{'far>.25':>11s}{'res90':>8s}"
    print(hdr)
    print(sub)
    print("-" * len(sub))
    for lab, own, rmin, cells in rows:
        line = (f"{lab[:26]:26s}{own:7.4g}"
                + (f"{rmin:7.1f}" if rmin is not None else f"{'-':>7s}") + "  ")
        for g in GAINS:
            a, r = cells[g]
            line += f"{a:11.4f}{r:8.1f}"
        print(line)

    print("\nSorted by the lowest alias rate reachable at res90 >= 8, which is")
    print("the guard §10.6 item 5 says is the weakest link in the analysis.")
    print("* = out of brief (spread term sees positions outside the patches).")


if __name__ == "__main__":
    main()
