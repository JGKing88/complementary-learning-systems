"""Controller activity and Hopfield inputs for the dynamics tuning viewer.

    python -m analysis.tuning_viewer.collect_dynamics --out DIR
    python -m analysis.tuning_viewer.collect_dynamics --out DIR --smoke

Model: the Cosyne Agent-HaSH controller, d0_base u725. Six arenas from its
recorded world (three training arenas, three of its validation arenas), each
with the same four goal cells.

Controller. For every arena and goal, fresh trials under the training eval's
contract (``evaluate_navigation``): new RNN state, random non-goal start,
sampled actions, nothing written to memory during the trial, the trial ends on
the first step after which the agent is within the goal radius (or at 200
steps). Two conditions:

    explore   memory empty
    exploit   memory holds the goal pattern only

The trial loop is ``analysis.phase_decoding.collect.run_trial``, which never
honours the policy's store head. Each step's RNN state (after the step's
input) is binned by the cell the agent stood in when that input was read.
Before collecting, the loop is checked against ``evaluate_navigation`` on the
same arenas and goals; a success gap beyond 4 binomial standard errors aborts.
During collection, the Hopfield readout q in every step's policy input is
checked against an exact recomputation for that cell.

Hopfield. Its state depends only on the cell and what is stored, so no runs
are needed. Per arena this writes the encoder outputs on the 21 x 21 patch of
cells the arena's local frames read (cells 0..20 on both axes; the frame at a
cell uses its North and East neighbours), five draws of ten distractor
patterns from ``sample_distractors`` (nested use: d distractors = the first d
of a draw), and reference recalls computed with ``Hopfield.recall_batch`` and
the scaffold's own frame and projection, for the page's self-checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from analysis.phase_decoding.collect import run_trial, signal_slice
from analysis.phase_decoding.rollout import RolloutEngine
from hopfield_nav.evaluation.checkpoint_io import world_spec_for
from hopfield_nav.evaluation.metrics import evaluate_navigation
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world.generate import build_envs

CKPT = ("/orcd/pool/003/jackking/cls_runs/agent_ckpts/"
        "navigate_navp2_d0_base_s42_22133273/navigate_u725.pt")
# Evenly spaced through each recorded list. base_val 0 is skipped because its
# patch overlaps training arena 0 on the scaffold.
ARENAS = (("train", 0), ("train", 7), ("train", 14), ("val", 1), ("val", 3), ("val", 5))
GOALS = ((4, 4), (15, 4), (4, 15), (15, 15))          # (x East, y North)
CONDITIONS = ("explore", "exploit")
N_DRAWS = 5
D_MAX = 10
PATCH = 21                                             # arena cells 0..19 plus the frame's +1 neighbours
CHECK_CELLS = (0, 21, 84, 105, 190, 210, 231, 295, 315, 336, 378, 399)
# (goal index or None, distractors, draw, recall steps)
CHECK_CONFIGS = ((0, 0, 0, 1), (1, 3, 0, 1), (2, 10, 1, 1), (None, 1, 2, 1),
                 (None, 10, 0, 1), (3, 10, 3, 2), (0, 5, 4, 5), (None, 10, 1, 8))


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return {"sha": sha, "dirty": bool(dirty)}


def arena_cells(size: int) -> np.ndarray:
    """(size*size, 2) local cells, x-major: index = x * size + y."""
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.int64)


def exact_recall(engine, off, patterns, steps: int, size: int):
    """Next state after ``steps`` recall steps and the projected q, for every
    arena cell, through the model's own Hopfield, frame and projection."""
    cells = arena_cells(size)
    cues = engine.vh.get_encoded_state(cells, off).astype(np.float32)
    if not patterns:
        return cues, np.zeros_like(cues), np.zeros((len(cells), 2), np.float32)
    hop = engine.make_hopfield()
    for p in patterns:
        hop.input_memory(torch.from_numpy(np.asarray(p)).float())
    cfg = engine.cfg.hopfield
    x = hop.recall_batch(torch.from_numpy(cues).to(engine.device), steps=steps,
                         beta=cfg.beta, alpha=cfg.alpha).cpu().numpy()
    W = engine.vh.gram_schmidt_projection(cells, off)
    q = engine.vh.project_displacement(cues, x, W).astype(np.float32)
    return cues, x, q


def validate(engine, arenas, n_trials: int, seed: int) -> dict:
    """Exploit success and steps, collector vs ``evaluate_navigation``, memory =
    goal only, sampled. Each arena uses goal ``a % len(GOALS)``."""
    sig = signal_slice(engine)
    envs = [a["env"] for a in arenas]
    offs = [a["offset"] for a in arenas]
    for i, env in enumerate(envs):
        env.set_goal(GOALS[i % len(GOALS)])
    ref = evaluate_navigation(engine.agent, envs, engine.vh, offs, engine.cfg, engine.device,
                              num_trials=n_trials, max_steps=200, n_distractors_list=[0],
                              deterministic=False, seed=seed)[0]
    rng = np.random.RandomState(seed + 1)
    torch.manual_seed(seed + 1)
    succ, steps = [], []
    for env, off in zip(envs, offs):
        for _ in range(n_trials):
            hop = engine.make_hopfield()
            hop.input_memory(torch.from_numpy(goal_encoding(engine.vh, off, env.goal_location)).float())
            r = run_trial(engine, env, off, hopfield=hop, mode="end", max_steps=200, post_steps=0,
                          deterministic=False, rng=rng, sig=sig, phase_init=1)
            succ.append(r["reached"])
            if r["reached"]:
                steps.append(r["steps_taken"])
    mine = {"success": float(np.mean(succ)), "steps": float(np.mean(steps)) if steps else None,
            "n": len(succ)}
    p = 0.5 * (mine["success"] + ref["success_rate"])
    se = float(np.sqrt(max(p * (1 - p), 1e-4) * 2 / mine["n"]))
    gap = abs(mine["success"] - ref["success_rate"])
    out = {"collector": mine,
           "reference": {"success": ref["success_rate"], "steps": ref["mean_steps"]},
           "success_gap": gap, "success_gap_se": gap / se}
    print("[validate]", json.dumps(out), flush=True)
    if gap > 4 * se:
        raise RuntimeError(f"collector does not reproduce evaluate_navigation: {out}")
    return out


def collect_arena(engine, arena: dict, goals, n_trials: dict, n_examples: int,
                  rng, sig) -> dict:
    env, off, size = arena["env"], arena["offset"], int(arena["env"].size)
    n_cells, D = size * size, int(engine.cfg.agent.hidden_size)
    G, C = len(goals), len(CONDITIONS)
    act_sum = np.zeros((G, C, n_cells, D), np.float64)
    count = np.zeros((G, C, n_cells), np.int64)
    ex = {k: [] for k in ("goal", "cond", "trial", "pos", "h", "q", "reached")}
    trials = []
    q_err = 0.0
    for g, goal in enumerate(goals):
        env.set_goal(goal)
        goal_pat = goal_encoding(engine.vh, off, goal)
        _, _, q_goal = exact_recall(engine, off, [goal_pat], 1, size)
        for c, cond in enumerate(CONDITIONS):
            q_exact = q_goal if cond == "exploit" else np.zeros_like(q_goal)
            for t in range(n_trials[cond]):
                hop = engine.make_hopfield()
                if cond == "exploit":
                    hop.input_memory(torch.from_numpy(goal_pat).float())
                r = run_trial(engine, env, off, hopfield=hop, mode="end", max_steps=200,
                              post_steps=0, deterministic=False, rng=rng, sig=sig,
                              phase_init=int(cond == "exploit"))
                s = r["steps"]
                cell = np.clip(np.rint(s["pos"]), 0, size - 1).astype(np.int64)
                idx = cell[:, 0] * size + cell[:, 1]
                h = s["h"].astype(np.float64)
                np.add.at(act_sum[g, c], idx, h)
                count[g, c] += np.bincount(idx, minlength=n_cells)
                q_in = s["x"][:, sig]
                q_err = max(q_err, float(np.abs(q_in - q_exact[idx]).max()))
                trials.append({"goal": g, "cond": cond, "trial": t, "reached": bool(r["reached"]),
                               "steps": int(r["steps_taken"]), "start": cell[0].tolist()})
                if t < n_examples:
                    ex["goal"].append(g)
                    ex["cond"].append(c)
                    ex["trial"].append(t)
                    ex["pos"].append(s["pos"].astype(np.float32))
                    ex["h"].append(s["h"].astype(np.float16))
                    ex["q"].append(q_in.astype(np.float32))
                    ex["reached"].append(bool(r["reached"]))
            print(f"[collect] {arena['name']} goal {goal} {cond}: "
                  f"{int(count[g, c].sum())} steps, q err so far {q_err:.2e}", flush=True)
    if q_err > 1e-4:
        raise RuntimeError(f"policy-input q differs from the exact recall by {q_err:.3e}")
    lens = np.array([len(p) for p in ex["pos"]], np.int64)
    return {"act_sum": act_sum.astype(np.float32), "count": count, "trials": trials,
            "q_err": q_err,
            "ex_goal": np.array(ex["goal"], np.int64), "ex_cond": np.array(ex["cond"], np.int64),
            "ex_trial": np.array(ex["trial"], np.int64), "ex_len": lens,
            "ex_reached": np.array(ex["reached"], bool),
            "ex_pos": np.concatenate(ex["pos"]), "ex_h": np.concatenate(ex["h"]),
            "ex_q": np.concatenate(ex["q"])}


def hopfield_inputs(engine, arena: dict, goals, seed: int) -> dict:
    env, off, size = arena["env"], arena["offset"], int(arena["env"].size)
    gx, gy = np.meshgrid(np.arange(PATCH), np.arange(PATCH), indexing="ij")
    patch_cells = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.int64)
    cue_patch = engine.vh.get_encoded_state(patch_cells, off).astype(np.float32)
    rng = np.random.RandomState(seed)
    dist = np.stack([np.stack(sample_distractors(engine.vh, off, size, D_MAX, rng))
                     for _ in range(N_DRAWS)]).astype(np.float32)          # (draws, D_MAX, D)
    chk_q, chk_x = [], []
    for gi, d, draw, steps in CHECK_CONFIGS:
        pats = [goal_encoding(engine.vh, off, goals[gi])] if gi is not None else []
        pats += list(dist[draw, :d])
        _, x, q = exact_recall(engine, off, pats, steps, size)
        chk_q.append(q)
        chk_x.append(x[list(CHECK_CELLS)].astype(np.float32))
    return {"cue_patch": cue_patch.reshape(PATCH, PATCH, -1), "distractors": dist,
            "check_configs": np.array([[-1 if g is None else g, d, r, s]
                                       for g, d, r, s in CHECK_CONFIGS], np.int64),
            "check_cells": np.array(CHECK_CELLS, np.int64),
            "check_q": np.stack(chk_q), "check_x": np.stack(chk_x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n_explore", type=int, default=100)
    ap.add_argument("--n_exploit", type=int, default=500)
    ap.add_argument("--n_examples", type=int, default=8)
    ap.add_argument("--validate_trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="two arenas, 3 explore / 5 exploit trials, 5 validation trials")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arenas_sel, goals = ARENAS, GOALS
    n_trials = {"explore": a.n_explore, "exploit": a.n_exploit}
    if a.smoke:
        arenas_sel = ARENAS[:1] + ARENAS[3:4]
        n_trials = {"explore": 3, "exploit": 5}
        a.validate_trials = 5

    t0 = time.perf_counter()
    engine = RolloutEngine(a.ckpt, num_arenas=6, split="recorded")
    engine.agent.eval()
    spec = world_spec_for(a.ckpt)
    train_envs = build_envs(spec.split.train, engine.cfg.env, engine.cfg.agent.movement_mode)
    lists = {"train": (spec.split.train, train_envs), "val": (spec.split.base_val, engine.val_envs)}
    for s, off in zip(spec.split.base_val, engine.val_offsets):
        assert tuple(s.offset) == tuple(off), (s.offset, off)
    arenas = []
    for kind, i in arenas_sel:
        specs, envs = lists[kind]
        arenas.append({"name": f"{kind}{i}", "kind": kind, "index": i, "env": envs[i],
                       "offset": tuple(int(v) for v in specs[i].offset),
                       "own_goal": [int(v) for v in specs[i].goal],
                       "wall_seed": int(specs[i].wall_seed)})
    print(f"[setup] {time.perf_counter() - t0:.0f}s", flush=True)

    val = validate(engine, arenas, a.validate_trials, a.seed)
    (out / "validation.json").write_text(json.dumps(val, indent=1))

    sig = signal_slice(engine)
    rng = np.random.RandomState(a.seed + 2)
    torch.manual_seed(a.seed + 2)
    summary = []
    for ai, arena in enumerate(arenas):
        res = collect_arena(engine, arena, goals, n_trials, a.n_examples, rng, sig)
        hop = hopfield_inputs(engine, arena, goals, seed=1000 * (a.seed + 1) + ai)
        wall = arena["env"]._wall_code.astype(np.float32)
        np.savez(out / f"arena_{ai}.npz", wall_code=wall, **{k: v for k, v in res.items()
                                                              if k not in ("trials", "q_err")}, **hop)
        summary.append({k: arena[k] for k in ("name", "kind", "index", "offset", "own_goal", "wall_seed")}
                       | {"q_err": res["q_err"], "trials": res["trials"]})
        print(f"[collect] arena {ai + 1}/{len(arenas)} done {time.perf_counter() - t0:.0f}s", flush=True)

    probe = engine.make_hopfield()
    meta = {"ckpt": a.ckpt, "encoder": engine.encoder_path, "args": vars(a),
            "goals": [list(g) for g in goals], "conditions": list(CONDITIONS),
            "hidden_size": int(engine.cfg.agent.hidden_size), "env_size": int(engine.cfg.env.size),
            "goal_radius": float(engine.cfg.env.goal_radius),
            "hopfield": {"beta": float(engine.cfg.hopfield.beta), "alpha": float(engine.cfg.hopfield.alpha),
                         "steps": int(engine.cfg.hopfield.steps), "scale": float(probe.scale),
                         "rule": probe.storage_rule, "zero_diag": bool(probe.zero_diag)},
            "n_draws": N_DRAWS, "d_max": D_MAX, "patch": PATCH,
            "wall_resolution": int(engine.cfg.env.wall_resolution),
            "git": git_state(), "torch": torch.__version__, "numpy": np.__version__,
            "seconds": time.perf_counter() - t0, "arenas": summary}
    (out / "meta.json").write_text(json.dumps(meta))
    print("COLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
