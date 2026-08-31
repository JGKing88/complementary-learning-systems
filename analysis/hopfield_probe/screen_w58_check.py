"""Screen w58 (2.5% coverage) at matched res90, with the 10% incumbent beside it.

Same protocol as `screen_w56_check.py`: probing every arm at one fixed gain puts
each at a different chart length, so gain is swept per arm and the setting that
lands res90 nearest the target is reported. That makes the arms comparable at
matched res90, so the remaining difference is the alias rate.

Two things this has to report honestly that the w56 version did not need to:

  * **Whether res90 7 is reachable at all.** At low coverage the code may not
    stretch that far at any gain. If the nearest achievable res90 is well short
    of the target, that is the binding constraint and picking a nearest-fit gain
    silently would hide it -- so the achieved res90 is printed alongside, and a
    flag marks arms that never get within one cell of target.
  * **The gap to 10%.** `w52_attract_fwhm/*_att0.5` is the incumbent at 10%
    coverage (continuous reach 0.987). It is screened here on the same draw
    rather than quoted, so the coverage cost is measured rather than inferred.
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

S = "/orcd/pool/003/jackking/cls_runs/sweeps"
NPOS = 1716
GAINS = [3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
TARGET = 7.0

GROUPS = [
    ("att0.5 @10% (ref)", f"{S}/w52_attract_fwhm/*_att0.5_seed=*"),
    ("w58 q_a0.25", f"{S}/w58_cov2.5/*_q_a0.25_seed=*"),
    ("w58 q_a0.5", f"{S}/w58_cov2.5/*_q_a0.5_seed=*"),
    ("w58 q_a1", f"{S}/w58_cov2.5/*_q_a1_seed=*"),
    ("w58 q_rate1", f"{S}/w58_cov2.5/*_q_rate1_seed=*"),
    ("w58 sm25_a0.5", f"{S}/w58_cov2.5/*_sm25_a0.5_seed=*"),
    ("w58 sm70q_a0.5", f"{S}/w58_cov2.5/*_sm70q_a0.5_seed=*"),
    # w59 continues the attract axis past w58's boundary value of 1.0.
    ("w59 q_a2", f"{S}/w59_cov2.5_att_hi/*_q_a2_seed=*"),
    ("w59 q_a4", f"{S}/w59_cov2.5_att_hi/*_q_a4_seed=*"),
]


def build_probes(seed: int = 7, n: int = 6000, nref: int = 250):
    rng = np.random.RandomState(seed)
    px, py = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
    qx, qy = rng.randint(0, NPOS, n), rng.randint(0, NPOS, n)
    far = np.hypot(px - qx, py - qy) > 200
    rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
    return (px[far], py[far], qx[far], qy[far], rx, ry)


FX, FY, GX, GY, RX, RY = build_probes()
OFFS = np.arange(1, 31)


def at_gain(field, enc, g):
    """(alias rate above 0.25 in the far field, median res90) at inference gain g."""
    enc.gain = float(g)
    A = field.encode(FX, FY).astype(np.float64)
    B = field.encode(GX, GY).astype(np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    alias = float(((A * B).sum(1) > 0.25).mean())

    R = field.encode(RX, RY).astype(np.float64)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    prof = np.empty((len(RX), len(OFFS)))
    for i, o in enumerate(OFFS):
        Q = field.encode(np.clip(RX + o, 0, NPOS - 1), RY).astype(np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        prof[:, i] = (R * Q).sum(1)
    res = [OFFS[b[0]] if (b := np.flatnonzero(r < 0.9)).size else OFFS[-1] + 1
           for r in prof]
    return alias, float(np.median(res))


def main() -> None:
    rows = []
    for lab, pat in GROUPS:
        per_gain = {g: ([], []) for g in GAINS}
        for d in sorted(glob.glob(pat)):
            ck = os.path.join(d, "encoder_final.pt")
            if not os.path.exists(ck):
                continue
            enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
            field = Field(enc, list(cfg.lambdas), fwhm, own, NPOS)
            for g in GAINS:
                a, r = at_gain(field, enc, g)
                per_gain[g][0].append(a)
                per_gain[g][1].append(r)
        if not per_gain[GAINS[0]][0]:
            print(f"  {lab}: none found", file=sys.stderr)
            continue
        best = min(GAINS,
                   key=lambda g: abs(np.median(per_gain[g][1]) - TARGET))
        res = float(np.median(per_gain[best][1]))
        # res90 at the encoder's most-stretched setting, i.e. the lowest gain.
        res_max = float(np.median(per_gain[GAINS[0]][1]))
        rows.append((float(np.median(per_gain[best][0])), lab, best, res,
                     res_max, len(per_gain[best][0])))

    print(f"Gain swept per arm; the setting landing res90 nearest {TARGET:g} is")
    print("reported. Four encoder seeds each.\n")
    print(f"{'arm':20s}{'gain':>6s}{'res90':>7s}{'alias':>9s}"
          f"{'res90 max':>11s}{'n':>4s}  flag")
    print("-" * 66)
    for alias, lab, g, res, res_max, k in sorted(rows):
        flag = "" if abs(res - TARGET) <= 1.0 else "CANNOT REACH TARGET"
        print(f"{lab:20s}{g:6d}{res:7.1f}{alias:9.4f}{res_max:11.1f}{k:4d}"
              f"  {flag}")
    print("\n'res90 max' is res90 at the lowest gain swept -- the longest chart")
    print("the encoder can produce. If it is below the target, the code cannot")
    print("stretch that far and res90 is the binding constraint, not the alias")
    print("rate.")


if __name__ == "__main__":
    main()
