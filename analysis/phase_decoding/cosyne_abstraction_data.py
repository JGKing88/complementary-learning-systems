"""Data for the Cosyne 2027 abstraction figure (definition A, ``matched`` rows).

    python -m analysis.phase_decoding.cosyne_abstraction_data --run d0_A \
        --out ../cosyne2027-figs/cosyne2027/data/abstraction.json

Geometry: per arena, the mean z-scored RNN state ``h`` and policy input ``x``
in each phase, on the rows matched for time since the trial began and distance
to goal (``run_analysis.select``, ``matched``). The 2 x n_arenas condition
means are projected onto their own top two principal components, signs set so
the mean explore-to-exploit vector points toward +PC1. Parallelism is
recomputed from the same means as a check against ``analysis_v3``.

Decoding curves and their nulls are copied from ``analysis_v3/<run>.json`` and
``<run>_random.json`` (the ``matched`` selection), so the figure shows exactly
the numbers in docs/PHASE_ABSTRACTION.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry as g
from .run_analysis import load_run, select

RESULTS = Path("/orcd/pool/003/jackking/cls_runs/results/phase_abstraction/analysis_v3")


def condition_means(z: np.ndarray, phase: np.ndarray, arena: np.ndarray):
    arenas = np.unique(arena)
    m0 = np.stack([z[(arena == a) & (phase == 0)].mean(0, dtype=np.float64) for a in arenas])
    m1 = np.stack([z[(arena == a) & (phase == 1)].mean(0, dtype=np.float64) for a in arenas])
    return arenas, m0, m1


def project(m0: np.ndarray, m1: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Top-2 PCA of the stacked condition means; PC1 oriented explore -> exploit."""
    allm = np.concatenate([m0, m1])
    mu = allm.mean(0)
    _, s, vt = np.linalg.svd(allm - mu, full_matrices=False)
    w = vt[:2].T
    p0, p1 = (m0 - mu) @ w, (m1 - mu) @ w
    if (p1 - p0).mean(0)[0] < 0:
        p0[:, 0], p1[:, 0] = -p0[:, 0], -p1[:, 0]
    var = (s ** 2 / np.sum(s ** 2))[:2].tolist()
    return p0, p1, var


def mean_pairwise_cos(d: np.ndarray) -> float:
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    c = u @ u.T
    return float(c[np.triu_indices(len(d), 1)].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="analysis/phase_decoding/configs/abstraction_v3.json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    data, meta = load_run(Path(cfg["runs"][a.run]["path"]))
    rng = np.random.default_rng(a.seed)
    idx = select(data, ~data["at_goal_in"].astype(bool), "matched", cfg, rng)
    out = {"run": a.run, "ckpt": meta["ckpt"], "selection": "matched", "seed": a.seed,
           "n_rows": int(len(idx)), "geometry": {}, "decoding": {}}
    for feat in ("h", "x"):
        z, _ = g.zscore_features(data[feat][idx], min_std=cfg["min_std"])
        arenas, m0, m1 = condition_means(z, data["phase"][idx], data["arena"][idx])
        p0, p1, var = project(m0, m1)
        out["geometry"][feat] = {"explore": p0.tolist(), "exploit": p1.tolist(),
                                 "pc_var_frac": var,
                                 "parallelism_recomputed": mean_pairwise_cos(m1 - m0)}
    for label, name, feat in (("trained", a.run, "h"), ("input", a.run, "x"),
                              ("untrained", a.run + "_random", "h")):
        m = json.loads((RESULTS / f"{name}.json").read_text())["by_condition"]["all"]["matched"][feat]
        ks = sorted(m["ccgp"], key=int)
        out["decoding"][label] = {
            "k": [int(k) for k in ks],
            "mean": [m["ccgp"][k]["mean"] for k in ks],
            "sd": [m["ccgp"][k]["sd"] for k in ks],
            "null_q95": [m["ccgp"][k]["null_q95"] for k in ks],
            "parallelism": m["parallelism"]["ps"],
            "parallelism_null_q95": m["parallelism"]["null_q95"],
        }
    Path(a.out).write_text(json.dumps(out, indent=1))
    for feat in ("h", "x"):
        print(feat, "PS recomputed", round(out["geometry"][feat]["parallelism_recomputed"], 3),
              "| analysis_v3", round(out["decoding"]["trained" if feat == "h" else "input"]["parallelism"], 3),
              "| PC var", [round(v, 3) for v in out["geometry"][feat]["pc_var_frac"]])


if __name__ == "__main__":
    main()
