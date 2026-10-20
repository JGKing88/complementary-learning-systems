"""Full-scaffold similarity scan + far-field statistics, for any probe encoder.

Companion to ``run.py`` for the ideal-encoder comparison (``ideal_encoder.py``).
``run.py`` gives exact / basin / reach / acc45; this gives the rest of the
standard columns, computed exactly as the scripts that defined them:

far field (``why_attract_check.py``, same 4000 random pairs, rng 11, pairs
>200 cells apart):
  d_eff          participation ratio of the code covariance
  far sd         sd of cos over the far pairs; the law is sd = 1/sqrt(d_eff)
  far >0.25      alias rate: fraction of far pairs above 0.25

per reference position, from the full 1716^2 cosine map (the scan of
``corner_check.py`` on worktree-encoder-hopfield-eval-spec, 5ffe1aa; THEORY
Sec 3.7's "Scan" table):
  C(1)           mean cos to the four axis neighbours
  r_0.9 / r_0.5  radius where the shell-mean cos falls through 0.9 / 0.5
  r_mono         median over rays of the monotone-decrease radius
  r_u16          unique coding radius at trim 16
  alias          the alias ceiling: max cos beyond 50 cells, whole scaffold
  <0.1           fraction of the arena with cos < 0.1

References are binned exactly as corner_check bins them -- by Chebyshev
distance beyond the ``[0, 500)^2`` corner (inside, 1-100, 100-300, 300-700,
700+), 100-cell margin from the edge -- so a position-independent code shows
equal columns across bands and a corner-trained one does not.

    python -m analysis.hopfield_probe.ideal_scan_check \
        --ckpt ideal:r=4 --label "ideal r=4" --out DIR

One cosine map encodes 2.94M positions; run on Slurm, never the login node.
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

NPOS = 1716
MARGIN = 100
EXCLUSION = 50
SIDE = 500
BANDS = (("inside", 0, 0), ("out 1-100", 1, 100), ("out 100-300", 100, 300),
         ("out 300-700", 300, 700), ("out 700+", 700, 10 ** 9))


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


def corner_dist(gx: int, gy: int, side: int = SIDE) -> int:
    return max(0, gx - (side - 1), gy - (side - 1))


def sample_refs(rng, lo, hi, n, side=SIDE):
    refs = []
    while len(refs) < n:
        gx, gy = rng.randint(MARGIN, NPOS - MARGIN, size=2)
        if lo <= corner_dist(int(gx), int(gy), side) <= hi:
            refs.append((int(gx), int(gy)))
    return refs


def far_field(field: Field) -> dict:
    """``why_attract_check.py``'s block, verbatim in its sampling."""
    rng = np.random.RandomState(11)
    n = 4000
    sx, sy = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
    qx, qy = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
    far = np.hypot(sx - qx, sy - qy) > 200
    Z = unit(field.encode(sx, sy).astype(np.float64))
    C = np.cov(Z - Z.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    W = unit(field.encode(qx, qy).astype(np.float64))
    cos = (Z[far] * W[far]).sum(1)
    return {"d_eff": pr, "pr_over_D": pr / Z.shape[1], "D": int(Z.shape[1]),
            "far_mean": float(cos.mean()), "far_sd": float(cos.std()),
            "inv_sqrt_d_eff": float(1 / np.sqrt(pr)),
            "far_p99": float(np.percentile(cos, 99)),
            "far_max": float(cos.max()),
            "far_gt_0.25": float((cos > 0.25).mean()),
            "n_far": int(far.sum())}


def cos_map(field: Field, ref) -> np.ndarray:
    gx0, gy0 = ref
    z0 = unit(field.encode(np.array([gx0]), np.array([gy0])))[0]
    xs = np.arange(NPOS)
    out = np.empty((NPOS, NPOS), dtype=np.float32)
    for y in range(NPOS):
        out[:, y] = unit(field.encode(xs, np.full(NPOS, y))) @ z0
    return out


def one_ref(field: Field, ref) -> dict:
    gx, gy = ref
    cos = cos_map(field, ref)
    rep = unique_radius_report(cos, gx, gy, trims=(16,), headline_trim=16,
                               margin_radii=(), profile_levels=(0.9, 0.5, 0.1),
                               exclusion_radius=EXCLUSION)
    c1 = float(np.mean([cos[gx + 1, gy], cos[gx - 1, gy],
                        cos[gx, gy + 1], cos[gx, gy - 1]]))
    ix, iy = np.ogrid[:NPOS, :NPOS]
    far = np.hypot(ix - gx, iy - gy) > EXCLUSION
    flat = int(np.argmax(np.where(far, cos, -2.0)))
    ax, ay = divmod(flat, NPOS)
    far_vals = cos[far]
    return {
        "ref": [gx, gy], "corner_dist": corner_dist(gx, gy), "c1": c1,
        "r_0.9": rep["r_at_cos0.9"], "r_0.5": rep["r_at_cos0.5"],
        "r_0.1": rep["r_at_cos0.1"], "r_mono": rep["r_monotone_median"],
        "r_u16": rep["r_trim16"], "r_u16_sat": rep["saturated_trim16"],
        "alias": float(cos[ax, ay]), "alias_at": [int(ax), int(ay)],
        "alias_dist": float(np.hypot(ax - gx, ay - gy)),
        # whole-map far-field spread (all ~2.9M cells beyond 50), for the
        # extreme-value check alias ~ sd * sqrt(2 ln N)
        "far_map_sd": float(far_vals.std()),
        "frac_below_0.1": float(np.mean(cos < 0.1)),
        "cos_median": float(np.median(cos)),
    }


def table(res: dict) -> None:
    ff = res["far_field"]
    print(f"\n=== {res['label']} ===")
    print(f"  d_eff {ff['d_eff']:.1f} (PR/D {ff['pr_over_D']:.3f})  far sd "
          f"{ff['far_sd']:.4f}  1/sqrt(d_eff) {ff['inv_sqrt_d_eff']:.4f}  "
          f"far max {ff['far_max']:.3f}  >0.25 {ff['far_gt_0.25']:.4f}")
    print(f"  {'band':<13s}{'n':>3s}{'C(1)':>7s}{'r_0.9':>7s}{'r_0.5':>7s}"
          f"{'r_mono':>8s}{'r_u16':>7s}{'alias med':>10s}{'max':>7s}"
          f"{'map sd':>8s}{'<0.1':>7s}")
    for name, _lo, _hi in BANDS:
        b = [r for r in res["refs"] if r["band"] == name]
        if not b:
            continue

        def g(k):
            return np.array([r[k] for r in b], dtype=float)

        print(f"  {name:<13s}{len(b):>3d}{g('c1').mean():>7.3f}"
              f"{np.median(g('r_0.9')):>7.1f}{np.median(g('r_0.5')):>7.1f}"
              f"{np.median(g('r_mono')):>8.1f}{np.median(g('r_u16')):>7.1f}"
              f"{np.median(g('alias')):>10.3f}{g('alias').max():>7.3f}"
              f"{np.median(g('far_map_sd')):>8.4f}"
              f"{g('frac_below_0.1').mean():>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint path or ideal:r=R spec; repeatable")
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--n_per_band", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    labels = args.label or []
    for i, path in enumerate(args.ckpt):
        label = labels[i] if i < len(labels) else path
        enc, ecfg, gain, fwhm, header = load_probe_encoder(
            path, device=args.device, fwhm_fallback=0.25)
        field = Field(enc, list(ecfg.lambdas), fwhm, gain, NPOS,
                      device=args.device)
        t0 = time.time()
        res = {"label": label, "path": path, "gain": gain, "fwhm": fwhm,
               "header": header, "far_field": far_field(field), "refs": []}
        rng = np.random.RandomState(args.seed)
        for name, lo, hi in BANDS:
            for ref in sample_refs(rng, lo, hi, args.n_per_band):
                r = one_ref(field, ref)
                r["band"] = name
                res["refs"].append(r)
            print(f"  {label}: band {name} done at {time.time() - t0:.0f}s",
                  flush=True)
        table(res)
        fn = "".join(c if c.isalnum() or c in "-_." else "_"
                     for c in label) + ".json"
        with open(os.path.join(args.out, fn), "w") as f:
            json.dump({**res, "side": SIDE, "seed": args.seed}, f)
        print(f"  wrote {os.path.join(args.out, fn)}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
