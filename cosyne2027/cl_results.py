"""Main-results numbers: the success matrix and everything derived from it.

Run: ~/.conda/envs/cls/bin/python cosyne2027/cl_results.py
Writes cosyne2027/data/cl_results.json, which fig3_main_results.py reads.

Every model is scored from one matrix. a[k, j] is the success rate on
environment j over the last TAIL_FRAC of the updates in window k (the window
in which environment k is trained), where success means a held-out
evaluation rollout from a random start reaches the goal within T_EVAL steps.
From it, per seed:

    plasticity   P   = mean_j a[j, j]        each env, at the end of its window
    final        ACC = mean_j a[N-1, j]      every env, at the end
    forgetting   F   = P - ACC

Path optimality is path length / straight-line optimum on the successful
held-out rollouts of the last window (all envs), both in distance units
(evaluation.rnn.optimal_path_to_goal).

Sources. Controls: the continuous, pretrained, previous-action-channel cell
of the continual control suite (analysis/continual/run_wave_prev.sh, 8
seeds), which evaluates one held-out rollout per seen env after every update
with a 200-step cap and a deterministic policy. Success within T_EVAL = 100 is
read from those rollouts as steps_to_goal < 100: both evaluators count the
goal at the pre-step check of loop step t, and with a deterministic policy
the first 100 steps of a 200-step rollout are the 100-step rollout. Each
method's configuration is re-selected at T_EVAL with the suite's own rule
(analysis/continual/results_data.staircase_stems): among configurations whose
current-env score a[N-1, N-1] is at least 0.5, the highest mean retention on
envs 0..N-2. Agent-HaSH: cosyne2027/run_agenthash_heldout.sh, the same
schedule with sampled actions and no gradient training.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path("/orcd/pool/003/jackking/cls_runs")
# Arena count -> (control histories, Agent-HaSH histories, output file).
# N=20: run_controls_n20.sh (each method's N=5 configuration only) and
# run_agenthash_heldout.sh with N_ENVS=20.
SOURCES = {
    5: (RUNS / "histories/wave1_p",
        RUNS / "results/cosyne2027/agenthash_heldout", "cl_results.json"),
    20: (RUNS / "histories/cosyne2027_n20",
         RUNS / "results/cosyne2027/agenthash_heldout_n20",
         "cl_results_n20.json"),
}
CONTROLS, AGENTHASH, OUT_NAME = SOURCES[5]
SEEDS = range(1, 9)
T_EVAL = 100
TAIL_FRAC = 0.2
USABLE_CURRENT = 0.5

# Method -> configuration-name prefixes in CONTROLS (results_data.ARMS)
FAMILIES = {
    "Naive SGD": ("R_none", "A_"),
    "Experience Replay": ("B_",),
    "CLEAR": ("D_",),
    "Online EWC": ("C_",),
    "XdG + SI": ("N2_",),
    "Multi-head oracle": ("M_multihead",),
}


def success(v):
    return int(bool(v["reached"]) and v["steps_to_goal"] is not None
               and v["steps_to_goal"] < T_EVAL)


def seed_record(hist):
    """Per-update success on each seen env, the matrix, and path ratios."""
    blocks = hist["blocks"]
    n = len(blocks)
    step_of = {}
    for b0, b1, k in blocks:
        for s in range(b0, b1 + 1):
            step_of[s] = k
    first = blocks[0][0]
    n_steps = blocks[-1][1] - first + 1
    curve = np.full((n_steps, n), np.nan)       # success, env j, per update
    for s, _train, inner in hist["trace"]:
        for j, v in inner.items():
            curve[s - first, int(j)] = success(v)

    a = np.full((n, n), np.nan)
    for b0, b1, k in blocks:
        lo = b0 + int((1 - TAIL_FRAC) * (b1 - b0 + 1))
        rows = curve[lo - first:b1 - first + 1]
        a[k, :k + 1] = np.nanmean(rows[:, :k + 1], axis=0)

    b0, b1, _ = blocks[-1]
    ratios = [v["path_to_goal"] / v["optimal_to_goal"]
              for s, _t, inner in hist["trace"] if b0 <= s <= b1
              for v in inner.values()
              if success(v) and v.get("optimal_to_goal")]
    return curve, a, ratios


def summarise(paths):
    curves, mats, ratios = [], [], []
    for p in paths:
        c, a, r = seed_record(json.load(open(p)))
        curves.append(c)
        mats.append(a)
        ratios.extend(r)
    mats = np.array(mats)
    n = mats.shape[1]
    P = np.array([np.mean(np.diag(a)) for a in mats])
    ACC = np.array([np.mean(a[n - 1]) for a in mats])
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
    # Success on all envs seen so far, per update, mean over seeds
    seen = np.array([np.nanmean(c, axis=1) for c in curves])
    return {
        "n_seeds": len(mats),
        "P": float(P.mean()), "P_sem": sem(P),
        "ACC": float(ACC.mean()), "ACC_sem": sem(ACC),
        "F": float((P - ACC).mean()), "F_sem": sem(P - ACC),
        "retained": float(mats[:, n - 1, :n - 1].mean()),
        "current": float(mats[:, n - 1, n - 1].mean()),
        "path_ratio_median": (float(np.median(ratios)) if ratios
                              else None),
        "path_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "matrix": np.nanmean(mats, axis=0).tolist(),
        "seen_curve": seen.mean(axis=0).tolist(),
        "seen_curve_sem": (seen.std(axis=0, ddof=1)
                           / np.sqrt(len(seen))).tolist(),
        # Success on each env per update, mean over seeds; None before the
        # env is introduced. [update][env]
        "env_curves": [[None if np.isnan(x) else float(x) for x in row]
                       for row in env_mean(curves)],
    }


def env_mean(curves):
    """Mean over seeds of (update, env) success; NaN where never evaluated."""
    c = np.array(curves)
    n = np.sum(~np.isnan(c), axis=0)
    total = np.nansum(c, axis=0)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def control_paths(config):
    return [CONTROLS / f"{config}_s{s}.json" for s in SEEDS]


def select(prefixes):
    """The suite's rule at T_EVAL, over every configuration in the family."""
    configs = sorted({p.name.rsplit("_s", 1)[0] for p in CONTROLS.glob("*.json")
                      if p.name.startswith(prefixes)})
    rows = {}
    for c in configs:
        paths = control_paths(c)
        if all(p.exists() for p in paths):
            rows[c] = summarise(paths)
    usable = {c: r for c, r in rows.items() if r["current"] >= USABLE_CURRENT}
    if not usable:              # nothing complete (or nothing learns) yet
        return None, rows
    best = max(usable, key=lambda c: usable[c]["retained"])
    return best, rows


def main():
    global CONTROLS, AGENTHASH, OUT_NAME
    p = argparse.ArgumentParser()
    p.add_argument("--n_envs", type=int, default=5, choices=sorted(SOURCES))
    args = p.parse_args()
    CONTROLS, AGENTHASH, OUT_NAME = SOURCES[args.n_envs]
    out = {"T_eval": T_EVAL, "tail_frac": TAIL_FRAC, "n_envs": args.n_envs,
           "models": {}, "selection": {}}
    for method, prefixes in FAMILIES.items():
        if not any(p.name.startswith(prefixes) for p in CONTROLS.glob("*.json")):
            print(f"{method}: no histories in {CONTROLS} yet, skipped")
            continue
        best, rows = select(prefixes)
        if best is None:
            print(f"{method}: no complete, usable configuration yet, skipped")
            continue
        out["models"][method] = {"config": best, **rows[best]}
        out["selection"][method] = {
            c: {"retained": r["retained"], "current": r["current"]}
            for c, r in rows.items()}
        print(f"{method:20s} {best:28s} P {rows[best]['P']:.3f}  "
              f"ACC {rows[best]['ACC']:.3f}  F {rows[best]['F']:.3f}  "
              f"path {rows[best]['path_ratio_median']}")
    hash_paths = [AGENTHASH / f"agenthash_s{s}.json" for s in SEEDS]
    if all(p.exists() for p in hash_paths):
        r = summarise(hash_paths)
        out["models"]["Agent-HaSH"] = {"config": "d0_base u725", **r}
        print(f"{'Agent-HaSH':20s} {'d0_base u725':28s} P {r['P']:.3f}  "
              f"ACC {r['ACC']:.3f}  F {r['F']:.3f}  "
              f"path {r['path_ratio_median']}")
    else:
        print("Agent-HaSH histories not all present yet:", AGENTHASH)
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / OUT_NAME, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
