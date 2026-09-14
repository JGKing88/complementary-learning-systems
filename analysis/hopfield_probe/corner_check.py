"""Does an encoder trained on one 500x500 corner work outside it?

Sec 3.5 of THEORY_ENCODER_HOPFIELD.md found that production, trained on 10% of
the arena as 118 scattered patches, generalises to the 90% it never saw. That
90% is never far from a patch. ``w62_corner`` trains the same recipe on a single
tiled 500x500 corner (``--patch_arena 500``), so the unseen region is one
contiguous hole up to 1200 cells deep, and this measures the code as a function
of distance from the corner.

The input at any position is three module phases (periods 11, 12, 13), and the
corner covers every single-module and every pair-of-modules phase many times
over; only the joint triple, period 1716, is 8.5% covered. So the question is
whether what the MLP learned is a function of the phases, which transfers, or
of the corner, which does not.

Per reference position, from the full 1716^2 cosine map:

  C(1)           mean cos to the four axis neighbours -- the localisation
                 margin ``1 - C(1)`` of Sec 5.2 (J2)
  r_0.9          the radius at which cos falls through 0.9 (the kernel width)
  r_mono         median over rays of the monotone-decrease radius: how far the
                 similarity keeps falling, i.e. how far a descent is ballistic
  r_u16          the unique coding radius at trim 16 (EXPERIMENTS_UNIQUE_RADIUS)
  alias          the alias ceiling: max cos beyond 50 cells
  alias@corner   whether that worst alias sits INSIDE the training corner. An
                 unseen position whose nearest code is a corner code has
                 collapsed onto the training set; one whose worst alias is
                 another unseen position is spread but not well
  <0.1           fraction of the arena with cos < 0.1 -- Sec 3.5's 94.7%

References are binned by Chebyshev distance beyond the corner,
``max(0, gx - 499, gy - 499)``: inside, then 1-100, 100-300, 300-700 and 700+
(the opposite corner). Every reference keeps a 100-cell margin from the arena
edge, as ``ur_border=100`` does, so no disc is clipped before r = 100.

Encoders: the two ``w62_corner`` arms (corner500, scatter100) at both seeds,
``w53_attract_knee att16`` (the same recipe on 118 scattered patches -- the
full-budget reference; the ladder's "10%" encoder is w52 att0.5, a different
attract level) and the untrained MLP floor, whose cos is ~1 everywhere.

One cosine map is 1716 rows of a 1716-position encode; ~100 s on four CPU
threads and well under a second on a GPU, so run this on a GPU node.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder
from encoder_training.unique_radius import unique_radius_report

from cls_paths import encoders_dir, results_dir, sweeps_dir

NPOS = 1716
MARGIN = 100                       # ur_border: keeps max_r >= 100 everywhere
EXCLUSION = 50                     # unique_radius_report's alias exclusion
BANDS = (("inside", 0, 0), ("out 1-100", 1, 100), ("out 100-300", 100, 300),
         ("out 300-700", 300, 700), ("out 700+", 700, 10 ** 9))
DEFAULT_OUT = os.path.join(str(results_dir()),
                           "hopfield_probe/20260914/corner_check")


def default_encoders() -> list[tuple[str, str]]:
    S, E = sweeps_dir(), encoders_dir()
    out = []
    for i, arm in ((0, "corner500"), (2, "scatter100")):
        for j, seed in enumerate((42, 43)):
            out.append((f"{arm} s{seed}",
                        str(S / f"w62_corner/{i + j:03d}_{arm}_seed={seed}"
                              "/encoder_final.pt")))
    for i, seed in ((4, 42), (5, 43)):
        out.append((f"scatter118 att16 s{seed}",
                    str(S / f"w53_attract_knee/{i:03d}_att16_seed={seed}"
                          "/encoder_final.pt")))
    out.append(("untrained", str(E / "untrained_mlp.pt")))
    return out


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def corner_dist(gx: int, gy: int, side: int) -> int:
    """Chebyshev distance beyond the ``[0, side)^2`` corner; 0 inside it."""
    return max(0, gx - (side - 1), gy - (side - 1))


def sample_refs(rng: np.random.RandomState, lo: int, hi: int, n: int,
                side: int) -> list[tuple[int, int]]:
    """``n`` positions with corner distance in ``[lo, hi]`` (inside: lo=hi=0),
    uniform over that band, ``MARGIN`` from every arena edge."""
    refs: list[tuple[int, int]] = []
    while len(refs) < n:
        gx, gy = rng.randint(MARGIN, NPOS - MARGIN, size=2)
        d = corner_dist(int(gx), int(gy), side)
        if lo <= d <= hi:
            refs.append((int(gx), int(gy)))
    return refs


def cos_map(field: Field, ref: tuple[int, int]) -> np.ndarray:
    """``cos(z(ref), z(p))`` for every ``p``, indexed ``[gx, gy]``."""
    gx0, gy0 = ref
    z0 = _unit(field.encode(np.array([gx0]), np.array([gy0])))[0]
    xs = np.arange(NPOS)
    cos = np.empty((NPOS, NPOS), dtype=np.float32)
    for y in range(NPOS):
        cos[:, y] = _unit(field.encode(xs, np.full(NPOS, y))) @ z0
    return cos


def one_ref(field: Field, ref: tuple[int, int], side: int) -> dict:
    gx, gy = ref
    cos = cos_map(field, ref)
    rep = unique_radius_report(cos, gx, gy, trims=(16,), headline_trim=16,
                               margin_radii=(), profile_levels=(0.9, 0.5, 0.1),
                               exclusion_radius=EXCLUSION)
    c1 = float(np.mean([cos[gx + 1, gy], cos[gx - 1, gy],
                        cos[gx, gy + 1], cos[gx, gy - 1]]))

    # Where the alias ceiling lives: the best cell beyond the exclusion disc.
    ix, iy = np.ogrid[:NPOS, :NPOS]
    far = np.hypot(ix - gx, iy - gy) > EXCLUSION
    flat = int(np.argmax(np.where(far, cos, -2.0)))
    ax, ay = divmod(flat, NPOS)
    return {
        "ref": [gx, gy], "corner_dist": corner_dist(gx, gy, side),
        "c1": c1,
        "r_0.9": rep["r_at_cos0.9"], "r_0.5": rep["r_at_cos0.5"],
        "r_0.1": rep["r_at_cos0.1"],
        "r_mono": rep["r_monotone_median"],
        "r_u16": rep["r_trim16"], "r_u16_sat": rep["saturated_trim16"],
        "alias": float(cos[ax, ay]), "alias_at": [int(ax), int(ay)],
        "alias_in_corner": corner_dist(int(ax), int(ay), side) == 0,
        "alias_dist": float(np.hypot(ax - gx, ay - gy)),
        "frac_below_0.1": float(np.mean(cos < 0.1)),
        "cos_median": float(np.median(cos)),
    }


def one_encoder(label: str, path: str, side: int, n_per_band: int, seed: int,
                device: str) -> dict:
    enc, ecfg, gain, fwhm, _h = load_probe_encoder(
        path, device=device, fwhm_fallback=0.25)
    enc.gain = gain
    field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS, device=device)
    rng = np.random.RandomState(seed)
    rows = []
    t0 = time.time()
    for name, lo, hi in BANDS:
        for ref in sample_refs(rng, lo, hi, n_per_band, side):
            r = one_ref(field, ref, side)
            r["band"] = name
            rows.append(r)
    print(f"  {label}: {len(rows)} refs in {time.time() - t0:.0f}s",
          flush=True)
    return {"label": label, "path": path, "gain": gain, "refs": rows}


def table(res: dict) -> None:
    rows = res["refs"]
    print(f"\n=== {res['label']} ===")
    hdr = (f"  {'band':<13s}{'n':>3s}{'C(1)':>7s}{'r_0.9':>7s}{'r_mono':>8s}"
           f"{'r_u16':>7s}{'alias med':>10s}{'max':>6s}{'@corner':>9s}"
           f"{'<0.1':>7s}")
    print(hdr)
    for name, _lo, _hi in BANDS:
        b = [r for r in rows if r["band"] == name]
        if not b:
            continue

        def g(k):
            return np.array([r[k] for r in b], dtype=float)

        print(f"  {name:<13s}{len(b):>3d}{g('c1').mean():>7.3f}"
              f"{np.median(g('r_0.9')):>7.1f}{np.median(g('r_mono')):>8.1f}"
              f"{np.median(g('r_u16')):>7.1f}{np.median(g('alias')):>10.3f}"
              f"{g('alias').max():>6.3f}{g('alias_in_corner').mean():>9.2f}"
              f"{g('frac_below_0.1').mean():>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", type=int, default=500)
    ap.add_argument("--n_per_band", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--only", default=None,
                    help="substring of an encoder label to restrict to")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    encoders = [(l, p) for l, p in default_encoders()
                if not args.only or args.only in l]
    results = []
    for label, path in encoders:
        if not os.path.exists(path):
            print(f"  {label}: missing {path}", flush=True)
            continue
        res = one_encoder(label, path, args.side, args.n_per_band, args.seed,
                          args.device)
        results.append(res)
        table(res)
        fn = label.replace(" ", "_").replace("/", "_") + ".json"
        with open(os.path.join(args.out, fn), "w") as f:
            json.dump({**res, "side": args.side, "seed": args.seed}, f)
    print(f"\nwrote {len(results)} encoders to {args.out}")


if __name__ == "__main__":
    main()
