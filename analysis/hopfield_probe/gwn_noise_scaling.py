"""Sec 22.2: how noise robustness of the online L1 scales with steps, environments and box size.

Cell noise 0.01, bump 0.10. One task per environment set:

    envs1/3/6/12   1, 3, 6 or 12 boxes of side 20 (sessions of 10k steps cycling)
    size14/40/80   one box of side 14, 40 or 80 (envs1 is side 20)

Each task streams 3M steps (``gwn_online.sessions``) through two learners at
once: the lock-in rule 2c (turn 0.001 of the weight rate; eta 0.05, 0.02, 0.01;
6 l^2 units per module each) and the shift rule 1c (rates 0.02, 0.005, 0.002,
0.001; eigenvectors at each checkpoint). At 100k, 300k, 1M and 3M steps:

* L1 quality: share of units/pairs above purity 0.99, coverage, mean impurity
  1 - purity (more sensitive than the share), and |T - P| for 1c;
* the network for the best eta (share >= 0.95, then lowest mean impurity) and
  the best rate (lowest |T - P|): learned and its exact twin
  (``gwn_l1_twin.network``: random triples, gains from this walk's steps with
  the at-rest correction, which equals depression with rest recovery to the
  precision that matters here).

    python -m analysis.hopfield_probe.gwn_noise_scaling run --task I --out DIR
    python -m analysis.hopfield_probe.gwn_noise_scaling summary --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from . import gwn_l1_online as G
from . import gwn_online as O
from .gwn_check import DEV, LAMBDAS, SMOKE, World, _dump
from .gwn_l1_learn import _assemble, codes_at, eig_pairs
from .gwn_l1_twin import network, twin_l1

FWHM = 0.10
NOISE = 0.01
ETAS = (0.05,) if SMOKE else (0.05, 0.02, 0.01)
RATES = (0.02, 0.005) if SMOKE else (0.02, 0.005, 0.002, 0.001)
PHI_RATIO = 0.001
T_CHECK = (2_000, 6_000) if SMOKE else (100_000, 300_000, 1_000_000, 3_000_000)
CONDS = ([("envs3", 3, 20)] if SMOKE else
         [("envs1", 1, 20), ("envs3", 3, 20), ("envs6", 6, 20), ("envs12", 12, 20),
          ("size14", 1, 14), ("size40", 1, 40), ("size80", 1, 80)])


def impurity(per_or_w, kind, gi=None):
    """Mean and 99th-percentile 1 - purity per module, and pooled."""
    out, allp = {}, []
    for m, l in enumerate(LAMBDAS):
        if kind == "lock":
            w = per_or_w[m][:l * l][:, gi * G.KPER + np.arange(G.QMAX * l * l)]
        else:
            w = np.stack([c + 1j * s for c, s in per_or_w[m]], 1)
        pur, _ = G.unit_purity(w, l)
        imp = 1 - pur
        out[str(l)] = {"mean": float(imp.mean()), "p99": float(np.quantile(imp, 0.99))}
        allp.append(imp)
    a = np.concatenate(allp)
    out["pooled"] = {"mean": float(a.mean()), "p99": float(np.quantile(a, 0.99))}
    return out


def shift_pairs(X):
    per, err = [], 0.0
    for m, l in enumerate(LAMBDAS):
        L = l * l
        Tx, Ty = (X[e, m, :L, :L].astype(np.float64) for e in (0, 1))
        err = max(err, max(float(np.abs(T - G.exact_shift(l, e)).max()) for e, T in ((0, Tx), (1, Ty))))
        per.append(eig_pairs(Tx, Ty))
    return per, err


def run(task, out):
    name, n_env, side = CONDS[task]
    G.ETAS = ETAS                        # group_stats / route2_pairs read these
    W = World(fwhm=FWHM)
    envs = O.env_set(n_env, side, 9000 + task)
    Tmax = max(T_CHECK)
    lock = O.LockIn(ETAS, rule="2c", phi_ratio=PHI_RATIO, seed=9100 + task)
    lms = O.ShiftLMS(RATES, sig2=NOISE ** 2, corrected=(False,))
    gen = torch.Generator(device="cpu").manual_seed(9200 + task)
    p0s, p1s, t, t0 = [], [], 0, time.time()
    res = {"name": name, "envs": envs, "noise": NOISE, "checkpoints": {}}
    print(f"=== {name}: {len(envs)} boxes of side {side}", flush=True)
    for pos in O.sessions(envs, Tmax, 9300 + task, block=2_000 if SMOKE else O.BLOCK):
        IN, OUT, ax, _ = O.session_steps(pos, FWHM, NOISE, gen)
        inv = 1.0 / (IN ** 2).sum(-1)
        p0s.append(pos[:-1]); p1s.append(pos[1:])
        for k in range(len(IN)):
            lock.step(IN[k], OUT[k], int(ax[k]), inv[k])
            lms.step(IN[k], OUT[k], int(ax[k]), inv[k])
        t += len(IN)
        if t not in T_CHECK:
            continue
        r = {"seconds": time.time() - t0}
        Wsnap = lock.snapshot()[0]
        p0, p1 = np.concatenate(p0s), np.concatenate(p1s)
        cells = np.unique(np.concatenate([p0, p1]), axis=0)
        idx = np.linspace(0, t - 1, 512 if SMOKE else 8192).astype(int)
        gsteps, codes = (p0[idx], p1[idx]), codes_at(cells, FWHM)
        r["cells_visited"] = int(len(cells))
        lk = {}
        for gi, eta in enumerate(ETAS):
            st = G.group_stats(Wsnap, gi, FWHM)
            st["impurity"] = impurity(Wsnap, "lock", gi)
            lk[eta] = st
            print(f"{name} T={t} 2c eta {eta:g}: pure {st['frac_pure_0.99']:.3f} "
                  f"cov {st[f'cov_conj_q{G.QMAX}_min']:.3f} mean impurity {st['impurity']['pooled']['mean']:.2e}",
                  flush=True)
        r["2c"] = {f"eta{e:g}": v for e, v in lk.items()}
        best = min(lk, key=lambda e: (lk[e]["frac_pure_0.99"] < 0.95, lk[e]["impurity"]["pooled"]["mean"]))
        per = G.route2_pairs(Wsnap, ETAS.index(best), dedup=False)
        l1, _ = _assemble(per, codes, FWHM)
        r["2c_best_eta"] = best
        r["2c_learned"] = network(l1, gsteps, FWHM, NOISE, W, f"{name} T={t} 2c eta {best:g} learned")
        r["2c_twin"] = network(twin_l1(per, FWHM), gsteps, FWHM, NOISE, W, f"{name} T={t} 2c eta {best:g} twin")
        sc, bestr = {}, None
        for (rate, _), X in lms.snapshot().items():
            per1, err = shift_pairs(X)
            st = G.pairs_stats(per1, FWHM)
            st["max_abs_T_minus_P"] = err
            st["impurity"] = impurity(per1, "pairs")
            sc[rate] = (st, per1)
            print(f"{name} T={t} 1c rate {rate:g}: pure {st['frac_pure_0.99']:.3f} |T-P| {err:.2e} "
                  f"mean impurity {st['impurity']['pooled']['mean']:.2e}", flush=True)
        r["1c"] = {f"rate{k:g}": v[0] for k, v in sc.items()}
        bestr = min(sc, key=lambda k: sc[k][0]["max_abs_T_minus_P"])
        per1 = sc[bestr][1]
        l1c, _ = _assemble(per1, codes, FWHM)
        r["1c_best_rate"] = bestr
        r["1c_learned"] = network(l1c, gsteps, FWHM, NOISE, W, f"{name} T={t} 1c rate {bestr:g} learned")
        r["1c_twin"] = network(twin_l1(per1, FWHM), gsteps, FWHM, NOISE, W, f"{name} T={t} 1c rate {bestr:g} twin")
        res["checkpoints"][str(t)] = r
        _dump(os.path.join(out, f"scaling_{name}.json"), res)
    print(f"DONE {name}", flush=True)


def _a(m):
    return m.get("far", m["unvisited"])["alias_max"]


def summary(out):
    rows = {json.load(open(p))["name"]: json.load(open(p)) for p in glob.glob(os.path.join(out, "scaling_*.json"))}
    order = [c[0] for c in CONDS if c[0] in rows]
    L = ["# Sec 22.2: noise scaling (cell noise 0.01, bump 0.10)", "",
         "Network alias (far cells), learned / exact twin, at 100k / 300k / 1M / 3M steps. "
         "In brackets: the best rate (1c) or eta (2c) and the pooled mean impurity 1 - purity.", ""]
    for learner in ("1c", "2c"):
        L += [f"## {learner}", "", "| condition | cells | " + " | ".join(f"{int(t) // 1000}k" for t in T_CHECK) + " |",
              "|---|---|" + "---|" * len(T_CHECK)]
        for nm in order:
            ck = rows[nm]["checkpoints"]
            cells = []
            for t in T_CHECK:
                r = ck.get(str(t))
                if r is None:
                    cells.append(""); continue
                key = "1c_best_rate" if learner == "1c" else "2c_best_eta"
                sub = r[learner][f"{'rate' if learner == '1c' else 'eta'}{r[key]:g}"]
                cells.append(f"{_a(r[learner + '_learned']):.3f} / {_a(r[learner + '_twin']):.3f} "
                             f"({r[key]:g}, {sub['impurity']['pooled']['mean']:.1e})")
            last = ck.get(str(max(int(k) for k in ck)), {})
            L.append(f"| {nm} | {last.get('cells_visited', '')} | " + " | ".join(cells) + " |")
        L.append("")
    open(os.path.join(out, "scaling_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.out, exist_ok=True)
    run(a.task, a.out) if a.mode == "run" else summary(a.out)


if __name__ == "__main__":
    main()
