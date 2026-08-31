"""At what inference gain does att0.5 reach cos-to-binarisation ~0.96?

Sec 0's rule: saturating beta only makes a memory a fixed point if the pattern
is ALSO near a hypercube corner, since the saturated update is
`x <- sign(Wx)/sqrt(D)`. Condition (a) is the encoder's, and it is universal in
`cos_bin` (~0.96) while encoder-specific in the gain that reaches it -- v35 at
~100, level 7 at ~300. att0.5's has never been measured.

res90 is reported alongside because raising gain to reach a corner spends the
local chart, which is what set the res90 ~7 optimum in the first place.
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
GAINS = [100, 300, 1000, 3000, 10000]

rng = np.random.RandomState(3)
M = 400
gx, gy = rng.randint(0, NPOS, M), rng.randint(0, NPOS, M)
nref = 150
rx, ry = rng.randint(60, NPOS - 60, nref), rng.randint(60, NPOS - 60, nref)
offs = np.arange(1, 31)

print(f"{'arm':16s}{'gain':>7s}{'cos_bin':>9s}{'res90':>7s}")
print("-" * 39)
for lab, pat in (("att0.5-s43", f"{S}/w52_attract_fwhm/*_att0.5_seed=43"),
                 ("v35 (ref)", None)):
    if pat is None:
        ck = ("/orcd/pool/003/jackking/cls_runs/encoders/"
              "run_20260422_185816/encoder_best.pt")
    else:
        ck = os.path.join(sorted(glob.glob(pat))[0], "encoder_final.pt")
    enc, cfg, own, fwhm, _ = load_probe_encoder(ck, fwhm_fallback=0.25)
    field = Field(enc, list(cfg.lambdas), fwhm, own, NPOS)
    for g in GAINS:
        enc.gain = float(g)
        Z = field.encode(gx, gy).astype(np.float64)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        B = np.sign(Z) / np.sqrt(Z.shape[1])
        cb = float(np.mean((Z * B).sum(1)
                           / (np.linalg.norm(Z, axis=1)
                              * np.linalg.norm(B, axis=1))))
        R = field.encode(rx, ry).astype(np.float64)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        prof = np.empty((nref, len(offs)))
        for i, o in enumerate(offs):
            Q = field.encode(np.clip(rx + o, 0, NPOS - 1), ry).astype(
                np.float64)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            prof[:, i] = (R * Q).sum(1)
        res = float(np.median(
            [offs[b[0]] if (b := np.flatnonzero(r < 0.9)).size else offs[-1] + 1
             for r in prof]))
        print(f"{lab:16s}{g:7d}{cb:9.4f}{res:7.1f}")
    print()
