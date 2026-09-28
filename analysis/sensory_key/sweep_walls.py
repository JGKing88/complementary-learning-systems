"""Sweep wall-code smoothness for the sensory-keyed goal memory.

For each (mode, ell) the walls are Gaussian-smoothed along their length (see
``retrieval_test.smooth_walls``) and the two retrieval questions are re-asked:
argmax own-goal accuracy and stored-vs-unstored top-score AUC. Alongside,
what the smoothing costs the *policy*: how alike neighbouring cells look, and
how many cells have a near-duplicate view elsewhere in their own env.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .retrieval_test import load_envs, run_subset, summarise, unit


def view_stats(obs):
    u = unit(obs)
    lag = lambda k: np.concatenate([(u[:, :, k:] * u[:, :, :-k]).sum(-1).ravel(),
                                    (u[:, k:] * u[:, :-k]).sum(-1).ravel()]).mean()
    E, S = obs.shape[:2]
    alias = []
    for e in range(E):
        v = u[e].reshape(S * S, -1)
        g = v @ v.T
        np.fill_diagonal(g, -2)
        alias.append((g.max(1) > 0.95).mean())
    return dict(nb1=float(lag(1)), nb3=float(lag(3)), nb6=float(lag(6)),
                alias95=float(np.mean(alias)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ells", default="0,0.5,1,2,4,8,16")
    p.add_argument("--modes", default="cont,sign")
    p.add_argument("--Ns", default="6,30")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for mode in a.modes.split(","):
        for ell in [float(x) for x in a.ells.split(",")]:
            if ell == 0 and mode != a.modes.split(",")[0]:
                continue
            if mode.startswith("wallconst") and ell != 0:
                continue
            obs, goals, n_rec = load_envs(94, 0, ell, mode)
            vs = view_stats(obs)
            rng = np.random.RandomState(1)
            row = dict(mode=mode if (ell > 0 or mode.startswith("wallconst")) else "iid", ell=ell, **vs)
            for N in [int(x) for x in a.Ns.split(",")]:
                subsets = ([list(range(n_rec))] if N == n_rec else
                           [sorted(rng.choice(len(goals), N, replace=False).tolist())
                            for _ in range(a.draws)])
                allrows = sum((run_subset(obs, goals, idx) for idx in subsets), [])
                s, _ = summarise(allrows)
                far = [b for b in s["by_dist"] if b["bin"] == ">10"][0]
                row[f"N{N}"] = dict(acc=s["acc"], auc=s["auc_top"],
                                    acc_far=far["acc"], auc_far=far["auc_top"],
                                    own_far=far["own_mean"],
                                    unstored_far=far["unstored_top_mean"])
            rows.append(row)
            n6, n30 = row.get("N6", {}), row.get("N30", {})
            print(f"{row['mode']:>4} ell={ell:>4}  nb1 {vs['nb1']:.2f} nb3 {vs['nb3']:.2f} "
                  f"nb6 {vs['nb6']:.2f} alias95 {vs['alias95']:.2f} | "
                  f"N6 acc {n6.get('acc', 0):.2f} auc {n6.get('auc', 0):.2f} "
                  f"(far acc {n6.get('acc_far', 0):.2f}) | "
                  f"N30 acc {n30.get('acc', 0):.2f} auc {n30.get('auc', 0):.2f} "
                  f"(far acc {n30.get('acc_far', 0):.2f} auc {n30.get('auc_far', 0):.2f}; "
                  f"own {n30.get('own_far', 0):.2f} vs unst {n30.get('unstored_far', 0):.2f})",
                  flush=True)
    with open(os.path.join(a.out, "sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {a.out}/sweep.json")


if __name__ == "__main__":
    main()
