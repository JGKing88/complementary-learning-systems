"""Did the screen's out-of-sample prediction hold, and where does L6 still fail?

Three things, all of which could go against the Sec 10 story:

1. The screen ranked L6 @300 (far 0.0057) *below* v35 @100 (far 0.0036). If L6
   beat v35 by more than the seed spread, the rank prediction failed at the top.
2. The implied per-competitor p should track far>0.25 at the 1.6-3.0 ratio Sec
   10.2 measured. Two new arms, eight new encoders, is the first real test.
3. L6 @300 posts discrete reach 1.000 with continuous 0.986, so the residual
   failure has moved somewhere the discrete field cannot see -- most likely the
   sub-cell approach to the goal point.
"""
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import spearmanr

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import load_probe_encoder

ARCH = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
RUNS = "/orcd/pool/003/jackking/cls_runs"
S = RUNS + "/sweeps"
NEW = RUNS + "/results/hopfield_probe/20260827"
V35 = RUNS + "/encoders/run_20260422_185816/encoder_best.pt"
A16 = f"{S}/w53_attract_knee/004_att16_seed=42/encoder_final.pt"
A16b = f"{S}/w53_attract_knee/005_att16_seed=43/encoder_final.pt"
NPOS = 1716


def l6(seed):
    d = glob.glob(f"{S}/w49_g100_knee/*eps1_rate0.5_seed={seed}")[0]
    return d + "/encoder_final.pt"


ROWS = [
    ("v35 prod", f"{ARCH}/production", "v35", V35, None),
    ("att16-s42 prod", f"{ARCH}/production", "L7-s42", A16, None),
    ("att16-s43 prod", f"{ARCH}/production", "L7-s43", A16b, None),
    ("att32-s42 prod", f"{ARCH}/att16_vs_att32", "att32-s42",
     f"{S}/w54_attract_far/000_att32_seed=42/encoder_final.pt", None),
    ("att32-s43 prod", f"{ARCH}/att16_vs_att32", "att32-s43",
     f"{S}/w54_attract_far/001_att32_seed=43/encoder_final.pt", None),
    ("v35 g300+sat", f"{ARCH}/gain300_beta1e6", "v35", V35, 300.0),
    ("att16-s42 g300+sat", f"{ARCH}/gain300_beta1e6", "L7-s42", A16, 300.0),
    ("att16-s43 g300+sat", f"{ARCH}/gain300_beta1e6", "L7-s43", A16b, 300.0),
    ("v35 g100+sat", f"{ARCH}/v35_gain100_beta1e6", "v35-g100-sat", V35, 100.0),
]
for s in (42, 43, 44, 45):
    ROWS.append((f"L6-s{s} prod", f"{NEW}/l6_production", f"L6-s{s}", l6(s), None))
    ROWS.append((f"L6-s{s} g300+sat", f"{NEW}/l6_g300_sat", f"L6-s{s}", l6(s),
                 300.0))

rng = np.random.RandomState(7)
N = 6000
px, py = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
qx, qy = rng.randint(0, NPOS, N), rng.randint(0, NPOS, N)
far = np.hypot(px - qx, py - qy) > 200
fx, fy, gx2, gy2 = px[far], py[far], qx[far], qy[far]


def find(d, lab):
    for f in glob.glob(d + "/*.json"):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        got = r.get("header", {}).get("label") or os.path.basename(f)[:-5]
        if got == lab:
            return r
    return None


print(f"{'arm':22s}{'far>.25':>9s}{'cont':>8s}{'disc':>8s}"
      f"{'p@K10':>8s}{'ratio':>7s}")
print("-" * 62)
xs, ys, new_x, new_y = [], [], [], []
for lab, d, name, ck, gover in ROWS:
    r = find(d, name)
    if r is None:
        print(f"{lab:22s}  missing")
        continue
    enc, cfg_e, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    g = gover if gover is not None else own
    enc.gain = float(g)
    field = Field(enc, list(cfg_e.lambdas), fwhm, g, NPOS)
    A = field.encode(fx, fy).astype(np.float64)
    B = field.encode(gx2, gy2).astype(np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    x = float((((A * B).sum(1)) > 0.25).mean())

    td = r["test_d"]["k"]["5"]["1"]
    cont = td["continuous"]["scalars"]["reach_rate"]["mean"]
    disc = td["discrete"]["scalars"]["reach_rate"]["mean"]

    k10 = r["test_d"]["k"]["10"]["1"]["continuous"]["scalars"]["reach_rate"]
    m = min(10, r["config"].get("n_score_envs", 5))
    sel = [v for i, v in enumerate(k10["values"]) if (i % m) in (0, 1, 2)]
    f10 = float(np.mean([v < 0.5 for v in sel]))
    p = 1 - (1 - f10) ** (1 / 9) if 0 < f10 < 1 else 0.0
    ratio = p / x if x > 0 and p > 0 else float("nan")

    xs.append(x)
    ys.append(cont)
    if lab.startswith("L6"):
        new_x.append(x)
        new_y.append(cont)
    print(f"{lab:22s}{x:9.4f}{cont:8.3f}{disc:8.3f}"
          f"{p:8.4f}{ratio:7.2f}")

rho, pv = spearmanr(xs, ys)
print(f"\nSpearman(far>0.25, cont reach), all {len(xs)} arms: "
      f"{rho:+.3f}, p = {pv:.5f}")
rho2, pv2 = spearmanr(xs[:9], ys[:9])
print(f"  original 9 arms only:                     {rho2:+.3f}, "
      f"p = {pv2:.5f}")
rho3, pv3 = spearmanr(new_x, new_y)
print(f"  the 8 new L6 points only:                 {rho3:+.3f}, "
      f"p = {pv3:.5f}")

print("\n\n--- Where the remaining failures are: continuous vs discrete ---")
print(f"{'arm':22s}{'disc':>8s}{'cont':>8s}{'gap':>8s}"
      f"{'cont reach @ d<=1':>19s}")
print("-" * 66)
for lab, d, name, _ck, _g in ROWS:
    if not (lab.startswith("L6") or "g100+sat" in lab):
        continue
    r = find(d, name)
    if r is None:
        continue
    td = r["test_d"]["k"]["5"]["1"]
    disc = td["discrete"]["scalars"]["reach_rate"]["mean"]
    cont = td["continuous"]["scalars"]["reach_rate"]["mean"]
    c = td["continuous"]["reach_by_dist"]
    near = [v for v, n in zip(c["mean"], c["n"]) if n and n > 20][:2]
    print(f"{lab:22s}{disc:8.3f}{cont:8.3f}{cont - disc:+8.3f}"
          f"{np.mean(near):19.3f}")
