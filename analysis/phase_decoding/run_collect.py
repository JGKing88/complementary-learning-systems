"""Collect per-step data for the abstraction analysis, after checking that the
collector reproduces the training eval on the same arenas.

    python -m analysis.phase_decoding.run_collect --ckpt CKPT --definition A \
        --split place=held_out,wall=held_out,goal=held_out --num_arenas 100 \
        --val_seed 0 --n_trials 40 --max_steps 60 --post_steps 0 \
        --n_dist_min 0 --n_dist_max 10 --stochastic --seed 0 --out OUT

The check runs first on ``--validate_arenas`` arenas at 5 distractors.
Definition A: exploit success and mean steps against ``evaluate_navigation``.
Definition B: first-touch success and steps against ``evaluate_task``
(one visit). A gap beyond 4 binomial standard errors aborts the run. The
random-weight control skips the check (it has no behaviour to reproduce).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hopfield_nav.evaluation.metrics import evaluate_navigation, evaluate_task

from .collect import CollectConfig, run_trial, signal_slice, collect
from .rollout import RolloutEngine
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors

VALIDATE_N_DIST = 5


def _collector_stats(engine, bundle, cc: CollectConfig, n_arenas: int) -> dict:
    sig = signal_slice(engine)
    rng = np.random.RandomState(cc.seed + 1)
    succ, steps = [], []
    for a in range(n_arenas):
        env, off = bundle.envs[a], bundle.offsets[a]
        for _ in range(cc.n_trials):
            patterns = list(sample_distractors(engine.vh, off, env.size, VALIDATE_N_DIST, rng))
            if cc.definition == "A":
                patterns.append(goal_encoding(engine.vh, off, env.goal_location))
            rng.shuffle(patterns)
            hop = engine.make_hopfield()
            for p in patterns:
                hop.input_memory(torch.from_numpy(p).float())
            r = run_trial(engine, env, off, hopfield=hop,
                          mode="end" if cc.definition == "A" else "store",
                          max_steps=cc.max_steps, post_steps=cc.post_steps,
                          deterministic=cc.deterministic, rng=rng, sig=sig,
                          phase_init=1 if cc.definition == "A" else 0)
            if cc.definition == "A":
                ok, n = r["reached"], r["steps_taken"]
            else:
                ok = r["switch_t"] is not None
                n = r["switch_t"] + 1 if ok else None
            succ.append(ok)
            if ok:
                steps.append(n)
    return {"success": float(np.mean(succ)), "steps": float(np.mean(steps)) if steps else None,
            "n": len(succ)}


def validate(engine, cc: CollectConfig, n_arenas: int) -> dict:
    bundle = engine.build_bundle()
    assert n_arenas <= len(bundle.envs)
    mine = _collector_stats(engine, bundle, cc, n_arenas)
    envs, offs = bundle.envs[:n_arenas], bundle.offsets[:n_arenas]
    if cc.definition == "A":
        ref = evaluate_navigation(engine.agent, envs, engine.vh, offs, engine.cfg,
                                  engine.device, num_trials=cc.n_trials,
                                  max_steps=cc.max_steps,
                                  n_distractors_list=[VALIDATE_N_DIST],
                                  deterministic=cc.deterministic, seed=cc.seed)[VALIDATE_N_DIST]
        ref = {"success": ref["success_rate"], "steps": ref["mean_steps"]}
    else:
        assert not cc.deterministic, "evaluate_task samples; validate B sampled"
        ref = evaluate_task(engine.agent, envs, engine.vh, offs, engine.cfg,
                            engine.device, num_trials=cc.n_trials,
                            max_steps=cc.max_steps,
                            n_distractors_list=[VALIDATE_N_DIST], seed=cc.seed,
                            visits=1)[VALIDATE_N_DIST]
        ref = {"success": ref["found_rate"], "steps": ref["steps_first"]}
    p = 0.5 * (mine["success"] + ref["success"])
    se = np.sqrt(max(p * (1 - p), 1e-4) * 2 / mine["n"])
    gap = abs(mine["success"] - ref["success"])
    out = {"collector": mine, "reference": ref, "n_dist": VALIDATE_N_DIST,
           "success_gap": gap, "success_gap_se": float(gap / se)}
    print("[validate]", json.dumps(out), flush=True)
    if gap > 4 * se:
        raise RuntimeError(f"collector does not reproduce the training eval: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--definition", required=True, choices=["A", "B"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--num_arenas", type=int, required=True)
    ap.add_argument("--val_seed", type=int, required=True)
    ap.add_argument("--n_trials", type=int, required=True)
    ap.add_argument("--max_steps", type=int, required=True)
    ap.add_argument("--post_steps", type=int, required=True)
    ap.add_argument("--n_dist_min", type=int, required=True)
    ap.add_argument("--n_dist_max", type=int, required=True)
    pol = ap.add_mutually_exclusive_group(required=True)
    pol.add_argument("--stochastic", dest="deterministic", action="store_false")
    pol.add_argument("--deterministic", dest="deterministic", action="store_true")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate_arenas", type=int, required=True)
    ap.add_argument("--random_agent", action="store_true")
    ap.add_argument("--random_init_seed", type=int, default=0)
    a = ap.parse_args()

    cc = CollectConfig(definition=a.definition, n_trials=a.n_trials,
                       max_steps=a.max_steps, post_steps=a.post_steps,
                       n_dist_min=a.n_dist_min, n_dist_max=a.n_dist_max,
                       deterministic=a.deterministic, seed=a.seed)
    engine = RolloutEngine(a.ckpt, num_arenas=a.num_arenas,
                           random_agent=a.random_agent,
                           random_init_seed=a.random_init_seed,
                           split=a.split, val_seed=a.val_seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.random_agent:
        v = validate(engine, cc, a.validate_arenas)
        (out / "validation.json").write_text(json.dumps(v, indent=1))
    collect(engine, cc, out)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))


if __name__ == "__main__":
    main()
