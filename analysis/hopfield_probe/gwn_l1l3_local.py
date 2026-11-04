"""Sec 21.1: L1 and L3 both from local rules, in one 14-cell box, for the probe.

Until now the probe saw learned L1 with trace-rule gains (each unit's mean
change per step over stored walk steps, then the Gaussian transfer function).
Here L3 is the incremental depression rule of Sec 20.1, run step by step on a
walk in the same box after L1 has finished learning:

    every step:  w <- w max(0, 1 - kappa |dz|^2),  w_0 = 1,  kappa = 0.005,
    read out after T = r^2 / (2 kappa) = 25,600 steps (r = 16),

so w = prod (1 - kappa d_t) ~ exp(-r^2 v / 2). A unit that never changes (the
exact n = 0 wave) is dropped, as in Sec 20.1; the encoder keeps the fewest
units holding 99.5% of sum w^2 (``prune``, a compute economy).

Three L1s, all learned in the 14-box at bump 0.10 without noise:

* ``exact``: exact L1 (isolates the L3 rule);
* ``1c``: the online shift rule (rate 0.5, 100k steps), then the eigenvectors
  of T_x + 0.618 T_y;
* ``2c``: the fully online lock-in units (2c, eta 0.2, turn rate 0.001 of the
  weight rate, 6 l^2 units per module, 300k steps), every unit wired in.

Each also gets trace-rule gains from the same depression walk, for comparison.
Kernel metrics here; the probe runs from ``run_gwn.sh probe`` (tasks 32-37).

    python -m analysis.hopfield_probe.gwn_l1l3_local --out DIR
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from . import gwn_l1_online as G
from .gwn_check import DEV, NU, R, SMOKE, World, _dump, _line, build, l1_out
from .gwn_encoder import exact_l1, random_triples, save_gwn, unit_waves
from .gwn_l1_learn import _assemble, codes_at
from .gwn_local_gain import KAPPA, encoder_from_gains, walk_in_env

FWHM = 0.10
SIDE = 14
T1 = 4_000 if SMOKE else 100_000
T2 = 8_000 if SMOKE else 300_000
T3 = 2_000 if SMOKE else int(round(R * R / (2 * KAPPA)))
G.ETAS = (0.2,)
PHI_RATIO = 0.001


def depression_gains(l1, tr, pos, chunk_u=16384, chunk_t=4096):
    """w = prod_t (1 - kappa d_t) over the walk ``pos``, step by step."""
    c, s = (t.float() for t in l1_out(l1, pos, FWHM))
    T = len(pos) - 1
    logw = torch.zeros(len(tr), dtype=torch.float64, device=DEV)
    for a in range(0, len(tr), chunk_u):
        trc = tr[a:a + chunk_u].to(DEV)
        acc = torch.zeros(len(trc), dtype=torch.float64, device=DEV)
        for t0 in range(0, T, chunk_t):
            t1 = min(T, t0 + chunk_t)
            C, S = unit_waves(c[t0:t1 + 1], s[t0:t1 + 1], trc)
            d = (C[1:] - C[:-1]) ** 2 + (S[1:] - S[:-1]) ** 2
            # w <- w max(0, 1 - kappa d): a step that changes a unit by more than
            # 1 / kappa silences it
            acc += torch.log((1 - KAPPA * d.double()).clamp_min(1e-300)).sum(0)
        logw[a:a + chunk_u] = acc
    return torch.where(logw == 0, torch.zeros_like(logw), torch.exp(logw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    encdir = os.path.join(a.out, "encoders")
    os.makedirs(encdir, exist_ok=True)
    W = World(fwhm=FWHM)
    IN, OUT, ax, (p0, p1) = G.make_steps(SIDE, FWHM, 0.0, T2, W, 7000)
    cells = np.unique(np.concatenate([p0, p1[-1:]]), axis=0)
    codes = codes_at(cells, FWHM)
    l1s = {"exact": exact_l1(FWHM).to(DEV)}
    res = {"T_l1c": T1, "T_l1_2c": T2, "T_l3": T3, "kappa": KAPPA, "r_pred": math.sqrt(2 * KAPPA * T3)}
    # L1 by the online shift rule
    lms = G.lms_shift(IN, OUT, ax, {T1})
    per = [p for p, _ in G.route1(IN, OUT, ax, T1, lms[T1])["1c"]]
    res["l1_1c"] = G.pairs_stats(per, FWHM)
    l1s["1c"], _ = _assemble(per, codes, FWHM)
    # L1 by the slow-turn lock-in units
    snaps = G.lock_in(IN, OUT, ax, "2c", seed=7100, checks={T2}, phi_ratio=PHI_RATIO)
    res["l1_2c"] = G.group_stats(snaps[T2][0], 0, FWHM)
    l1s["2c"], _ = _assemble(G.route2_pairs(snaps[T2][0], 0, dedup=False), codes, FWHM)
    del snaps, IN, OUT
    for k in ("1c", "2c"):
        st = res[f"l1_{k}"]
        print(f"L1 {k}: pure {st['frac_pure_0.99']:.3f}", flush=True)
    # L3: a fresh walk in the same box after L1 has learned
    pos = walk_in_env(*G.BOX, SIDE, T3, 7200)
    idx = np.linspace(0, T3 - 1, 512 if SMOKE else 8192).astype(int)
    for k, l1 in l1s.items():
        tr = random_triples(l1, NU, seed=0)
        w = depression_gains(l1, tr, pos)
        enc, info = encoder_from_gains(l1, tr, w, FWHM, {"l1": k, "l3": "depression", "side": SIDE,
                                                           "T3": T3, "kappa": KAPPA})
        m = W.metrics(enc)
        _line(f"{k} L1 + depression L3", info, m)
        save_gwn(enc.cpu(), os.path.join(encdir, f"local_{k}_dep.pt"))
        v, pw = G.local_slowness(l1, tr, (pos[:-1][idx], pos[1:][idx]), FWHM)
        enc_t, info_t = build(l1, tr, v, pw, FWHM)
        m_t = W.metrics(enc_t)
        _line(f"{k} L1 + trace L3", info_t, m_t)
        res[k] = {"depression": {**info, **m}, "trace": {**info_t, **m_t}}
        _dump(os.path.join(a.out, "l1l3_local.json"), res)
    print("DONE l1l3", flush=True)


if __name__ == "__main__":
    main()
