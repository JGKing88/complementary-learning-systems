"""Audit check 1: does the phase-decoding collector reproduce the trained
behaviour? Same checkpoint, same arenas: training's evaluate_navigation vs the
collector's explore and exploit trials. Sampled policy throughout."""
import argparse, dataclasses, json, sys
import numpy as np, torch
sys.path.insert(0, "/orcd/home/002/jackking/cls")
from analysis.phase_decoding.rollout import RolloutEngine
from analysis.phase_decoding.collect_trials import ExploreExploitCollector
from hopfield_nav.evaluation.metrics import evaluate_navigation

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n_arenas", type=int, default=10)
ap.add_argument("--n_trials", type=int, default=20)
ap.add_argument("--max_steps", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()

eng = RolloutEngine(a.ckpt, num_arenas=a.n_arenas)
b = eng.build_bundle()
n = min(a.n_arenas, len(b.envs))
out = {"ckpt": a.ckpt, "n_arenas": n, "n_trials": a.n_trials, "max_steps": a.max_steps}
nav = evaluate_navigation(eng.agent, b.envs[:n], b.vh, b.offsets[:n], eng.cfg,
                          eng.device, num_trials=a.n_trials, max_steps=a.max_steps,
                          n_distractors_list=[0, 5], deterministic=False, seed=0)
out["evaluate_navigation"] = {str(k): v for k, v in nav.items()}
sub = dataclasses.replace(b, envs=b.envs[:n], offsets=b.offsets[:n],
                          goals_local=b.goals_local[:n], quadrants=b.quadrants[:n])
for nd in (0, 5):
    ds = ExploreExploitCollector(eng).collect(
        sub, n_starts=a.n_trials, max_steps=a.max_steps, n_dist_min=nd,
        n_dist_max=nd, deterministic=False, seed=0, progress_every=10**9)
    for c in ("explore", "exploit"):
        s = [t for td in ds.per_arena.values()
             for t in (td.summaries_explore if c == "explore" else td.summaries_exploit)]
        r = np.array([t["reached"] for t in s]); st = np.array([t["steps_taken"] for t in s])
        out[f"collector_{c}_d{nd}"] = {"success_rate": float(r.mean()),
            "mean_steps_success": float(st[r].mean()) if r.any() else None}
print(json.dumps(out, indent=1))
json.dump(out, open(a.out, "w"), indent=1)
