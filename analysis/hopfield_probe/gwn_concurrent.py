"""Sec 22.1 and 22.3: L1 and L3 learning at the same time, on one walk.

One 14-cell box, bump 0.10. The lock-in L1 (2c, turn 0.001 of the weight
rate) and the depression-rule L3 run together, step by step, from step 0
(``gwn_online.LockIn`` + ``L2Stream``). L3 variants on the same L2 stream:

    always    from step 0, kappa = r^2 / (2 T_total)
    late      kappa = 0.005 over the last 25,600 steps
    self0.9   self-timed critical period per L2 unit: silent until its three
    self0.99  L1 inputs' settledness exceeds 0.9 (or 0.99), then w = 1 and
              depression for 25,600 steps, then frozen
    seq       L1 frozen after learning, a fresh walk of 25,600 steps (Sec 21.1)

Conditions: ``clean`` (no noise, eta 0.2, 300k steps) and ``noise`` (cell
noise 0.01, eta 0.05 and 0.02, 1M steps; every variant recovers at rest).
Alongside, the shift rule 1c (rate 0.5 clean, 0.005 noisy) learns on the same
walk; its eigenvector pairs then get depression L3 on a fresh walk (``1c_seq``,
"local except the eigenvector step").

Kernel metrics for every encoder; encoders saved for the probe.

    python -m analysis.hopfield_probe.gwn_concurrent --out DIR
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from . import gwn_l1_online as G
from . import gwn_online as O
from .gwn_check import DEV, LAMBDAS, NU, R, SMOKE, World, _dump, _line
from .gwn_encoder import random_triples, save_gwn
from .gwn_l1_learn import _assemble, codes_at, eig_pairs
from .gwn_local_gain import KAPPA, encoder_from_gains
from .gwn_noise_scaling import impurity

FWHM = 0.10
SIDE = 14
T3 = 1_000 if SMOKE else O.T3
PHI_RATIO = 0.001
CONDS = ([("clean", 0.0, (0.2,), 6_000, 0.5)] if SMOKE else
         [("clean", 0.0, (0.2,), 300_000, 0.5), ("noise", 0.01, (0.05, 0.02), 1_000_000, 0.005)])
BLOCK = 2_000 if SMOKE else O.BLOCK


def variants(T_total):
    return {"always": ("always", R * R / (2 * T_total)),
            "late": ("late", KAPPA, T_total - T3),
            "self0.9": ("self", KAPPA, 0.9),
            "self0.99": ("self", KAPPA, 0.99),
            "seq": ("seq", KAPPA)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    encdir = os.path.join(a.out, "encoders")
    os.makedirs(encdir, exist_ok=True)
    W = World(fwhm=FWHM)
    O.T3 = T3
    res = {}
    for ci, (cond, noise, etas, T_total, rate) in enumerate(CONDS):
        G.ETAS = etas
        t0 = time.time()
        envs = O.env_set(1, SIDE, 0)
        lock = O.LockIn(etas, rule="2c", phi_ratio=PHI_RATIO, seed=10_000 + ci, track=True)
        lms = O.ShiftLMS((rate,), sig2=noise ** 2, corrected=(False,))
        spec = variants(T_total)
        streams = [O.L2Stream(lock, gi, NU, seed=10_100 + gi, T_total=T_total, variants=spec, noise=noise)
                   for gi in range(len(etas))]
        gen = torch.Generator(device="cpu").manual_seed(10_200 + ci)
        p0s, p1s, t = [], [], 0
        print(f"=== {cond}: noise {noise:g}, eta {etas}, {T_total} steps", flush=True)
        for pos in O.sessions(envs, T_total, 10_300 + ci, block=BLOCK):
            IN, OUT, ax, IN2 = O.session_steps(pos, FWHM, noise, gen, rest=noise > 0)
            inv = 1.0 / (IN ** 2).sum(-1)
            p0s.append(pos[:-1]); p1s.append(pos[1:])
            for k in range(len(IN)):
                z2 = G._proj(IN2[k], lock.W) if IN2 is not None else None
                z0, z1 = lock.step(IN[k], OUT[k], int(ax[k]), inv[k])
                lms.step(IN[k], OUT[k], int(ax[k]), inv[k])
                for st in streams:
                    st.step(t, z0, z1, z2)
                t += 1
            if t % 100_000 == 0 or t == T_total:
                print(f"{cond} step {t} ({time.time() - t0:.0f}s): settled share "
                      f"{float((lock.settle > 0.99).float().mean()):.3f}", flush=True)
        # seq: L1 frozen, a fresh walk of T3 steps in the same box
        Wf = lock.W.clone()
        tt = 0
        for pos in O.sessions(envs, T3, 10_400 + ci, block=BLOCK):
            IN, OUT, ax, IN2 = O.session_steps(pos, FWHM, noise, gen, rest=noise > 0)
            for k in range(len(IN)):
                z0, z1 = G._proj(IN[k], Wf), G._proj(OUT[k], Wf)
                z2 = G._proj(IN2[k], Wf) if IN2 is not None else None
                for st in streams:
                    st.step(tt, z0, z1, z2, only={"seq"})
                tt += 1
        p0, p1 = np.concatenate(p0s), np.concatenate(p1s)
        codes = codes_at(np.unique(np.concatenate([p0, p1]), axis=0), FWHM)
        r = {"noise": noise, "etas": etas, "T_total": T_total, "T3": T3,
             "settled_share_0.99": float((lock.settle > 0.99).float().mean()),
             "settled_share_0.9": float((lock.settle > 0.9).float().mean())}
        Wsnap = lock.snapshot()[0]
        for gi, (eta, st) in enumerate(zip(etas, streams)):
            gs = G.group_stats(Wsnap, gi, FWHM)
            gs["impurity"] = impurity(Wsnap, "lock", gi)
            r[f"l1_eta{eta:g}"] = gs
            print(f"{cond} eta {eta:g}: L1 pure {gs['frac_pure_0.99']:.3f} cov "
                  f"{gs[f'cov_conj_q{G.QMAX}_min']:.3f} mean impurity {gs['impurity']['pooled']['mean']:.2e}",
                  flush=True)
            l1 = st.l1()
            for k in spec:
                w = st.gains(k)
                tag = f"conc_{cond}_eta{eta:g}_{k}"
                if float(w.max()) < 1e-6:          # every unit silent or worn down: no code
                    r[tag] = {"no_surviving_units": True, **st.summary(k)}
                    print(f"{cond} eta {eta:g} L3 {k}: no surviving units {st.summary(k)}", flush=True)
                    continue
                enc, info = encoder_from_gains(l1, st.tr, w, FWHM, {"l1": "2c", "l3": k, "cond": cond, "eta": eta})
                m = W.metrics(enc)
                _line(f"{cond} eta {eta:g} L3 {k}", info, m)
                r[tag] = {**info, **st.summary(k), **m}
                save_gwn(enc.cpu(), os.path.join(encdir, f"{tag}.pt"))
        # the shift rule's eigenvector pairs + depression on a fresh walk
        X = lms.snapshot()[(rate, False)]
        per = []
        for m, l in enumerate(LAMBDAS):
            L = l * l
            per.append(eig_pairs(*(X[e, m, :L, :L].astype(np.float64) for e in (0, 1))))
        l1c, _ = _assemble(per, codes, FWHM)
        tr = random_triples(l1c, NU, seed=0)
        pos = next(O.sessions(envs, T3, 10_500 + ci, block=T3))
        w = O.depression_l1(l1c, tr, pos, FWHM, noise, seed=10_600 + ci)
        tag = f"conc_{cond}_1c_seq"
        enc, info = encoder_from_gains(l1c, tr, w, FWHM, {"l1": "1c", "l3": "seq", "cond": cond, "rate": rate})
        m = W.metrics(enc)
        _line(f"{cond} 1c rate {rate:g} L3 seq", info, m)
        r[tag] = {**info, "l1": G.pairs_stats(per, FWHM), **m}
        save_gwn(enc.cpu(), os.path.join(encdir, f"{tag}.pt"))
        r["seconds"] = time.time() - t0
        res[cond] = r
        _dump(os.path.join(a.out, "concurrent.json"), res)
    print("DONE concurrent", flush=True)


if __name__ == "__main__":
    main()
