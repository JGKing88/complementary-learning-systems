"""Pre-flight gates C1-C8 for the goal-conditioned control (plan §7.1).

    python scripts/goal_nav_preflight.py [--n_envs 64 --n_val_envs 16 ...]

Builds the configured world exactly as `train_goal_pairs` would, then checks
the assumptions the experiment's reading rests on, and exits non-zero if any
fail. Run once per configuration before A0; nothing downstream is read
until it passes.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CLS_RUNS", "/orcd/pool/003/jackking/cls_runs")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from hopfield_nav.config import EnvConfig, RNNTrainConfig, RNNAgentConfig
from hopfield_nav.evaluation.goal_pairs import (
    EnvTensors, NearestNeighbourDecoder, enumerate_pairs, optimal_action_set,
    pair_inputs, score_pairs, unit_vectors, evaluate_pairs, aggregate_tables)
from hopfield_nav.rollout.oracles import (
    bfs_action_batch_continuous, bfs_action_batch_discrete)
from hopfield_nav.rollout.rnn import build_rnn_input, goal_sensory_vec, grid_state_vec, sensory_vec, xy_vec
from hopfield_nav.training.goal_pairs_setup import agent_cfg_for_mode, build_env_sets
from hopfield_nav.world.generate import toroidal_gap

FAILS: list[str] = []


def gate(cid: str, ok: bool, msg: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{cid}] {tag}  {msg}")
    if not ok:
        FAILS.append(cid)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--n_val_envs", type=int, default=16)
    p.add_argument("--n_same_envs", type=int, default=8)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=120)
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--place_margin", type=int, default=20)
    p.add_argument("--goal_val_frac", type=float, default=0.2)
    p.add_argument("--region_val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size),
        agent=RNNAgentConfig(input_grid_state=True),
        n_envs=args.n_envs, n_val_envs=args.n_val_envs, env_generator=True,
        place_margin=args.place_margin, goal_val_frac=args.goal_val_frac,
        region_val_frac=args.region_val_frac, lambdas=list(args.lambdas),
        fwhm_ratio=args.fwhm_ratio, seed=args.seed)
    train, heldout, same, split, vh, sgb = build_env_sets(
        cfg, np.random.RandomState(args.seed), n_same=args.n_same_envs, keep_field=True)
    cells = split.cell_sets()
    S = args.size
    print(f"world: {len(train)} train / {len(heldout)} heldout / {len(same)} same; "
          f"cells={cells.summary()}\n")

    # ---- C1 split invariants ---------------------------------------------
    arena = frozenset((x, y) for x in range(S) for y in range(S))
    ok = (cells.region <= split.goal_cells_val
          and not (cells.start_train & cells.region)
          and not (cells.goal_train & split.goal_cells_val)
          and (cells.goal_train | cells.goal_heldout | cells.region) == arena)
    train_walls = {s.wall_seed for s in split.train}
    train_boxes = {(s.offset, s.size) for s in split.train}
    hv = split.base_val
    ok &= all(s.wall_seed not in train_walls for s in hv)
    ok &= all((s.offset, s.size) not in train_boxes for s in hv)
    same_specs = split.train[:len(same)]
    ok &= all(s in split.train for s in same_specs)
    gate("C1", ok, f"region⊂goal_val, start∩region=∅, goal partition covers arena, "
                   f"heldout walls/boxes disjoint from train, same⊂train")

    # ---- C2 twin rates ----------------------------------------------------
    def twin_rate(M):
        _, inv, cnt = np.unique(M, axis=0, return_inverse=True, return_counts=True)
        return float((cnt[inv] > 1).mean())
    omni_rates = [twin_rate(t.omni) for t in train.tensors + heldout.tensors]
    gb_rates = [twin_rate(t.gbook) for t in train.tensors + heldout.tensors]
    north_rates = [twin_rate(t.north) for t in train.tensors + heldout.tensors]
    gate("C2", max(omni_rates) == 0.0 and max(gb_rates) == 0.0,
         f"exact-twin rate: omni max={max(omni_rates):.3f}, gbook max={max(gb_rates):.3f} "
         f"(single-north mean={np.mean(north_rates):.3f}, for reference)")

    # ---- C3 teacher agreement ---------------------------------------------
    rng = np.random.RandomState(1)
    p_, g_ = enumerate_pairs(cells, "train", "train")
    idx = rng.choice(len(p_), 10000, replace=False)
    p_, g_ = p_[idx], g_[idx]
    pos = np.stack([p_ // S, p_ % S], axis=1)
    ok = True
    for i in range(0, len(p_), 500):
        gg = (int(g_[i] // S), int(g_[i] % S))
        rows = np.arange(i, min(i + 500, len(p_)))
        mask = g_[rows] == g_[i]
        if not mask.any():
            continue
        r = rows[mask]
        u_ref = bfs_action_batch_continuous(pos[r], gg, rng)
        u_new = unit_vectors(p_[r], g_[r], S)
        ok &= np.allclose(u_ref, u_new, atol=1e-6)
        a_ref = bfs_action_batch_discrete(pos[r], gg, S, rng)
        opt = optimal_action_set(p_[r], g_[r], S)
        ok &= bool(opt[np.arange(len(r)), a_ref].all())
    opt = optimal_action_set(p_, g_, S)
    aligned = (pos[:, 0] == g_ // S) | (pos[:, 1] == g_ % S)
    ok &= bool((opt[aligned].sum(1) == 1).all() and (opt[~aligned].sum(1) == 2).all())
    gate("C3", ok, "unit_vectors == bfs_continuous; bfs_discrete ∈ optimal_set; "
                   "|set|=1 iff aligned else 2")

    # ---- C4 random baseline ------------------------------------------------
    rnd_c = score_pairs(rng.standard_normal((len(p_), 2)).astype(np.float32), p_, g_, S, "continuous")["metric"]
    rnd_d = score_pairs(rng.standard_normal((len(p_), 4)).astype(np.float32), p_, g_, S, "discrete")["metric"]
    exp_d = float(opt.sum(1).mean() / 4)
    gate("C4", abs(rnd_c - 90) < 2 and abs(rnd_d - exp_d) < 0.02,
         f"random: {rnd_c:.1f}° (expect 90), acc {rnd_d:.3f} (expect {exp_d:.3f})")

    # ---- C5 nearest-neighbour decoder -------------------------------------
    ok = True
    msgs = []
    for mode in ("xy", "grid", "regular"):
        for mm in ("continuous", "discrete"):
            acfg = agent_cfg_for_mode(mode, mm)
            t = train.tensors[0]
            dec = NearestNeighbourDecoder(t, acfg, cells, mm)
            pt, gt = enumerate_pairs(cells, "train", "train")
            sc = score_pairs(dec.predict_for_pairs(pt, gt), pt, gt, S, mm)["metric"]
            exact = (sc < 0.1) if mm == "continuous" else (sc > 0.9999)
            ok &= exact
            if mode == "xy":
                pr, gr = enumerate_pairs(cells, "region", "region")
                scr = score_pairs(dec.predict_for_pairs(pr, gr), pr, gr, S, mm)["metric"]
                msgs.append(f"xy/{mm[:4]} region×region nn={scr:.2f}")
    gate("C5", ok, "nn-decoder exact on train×train in every mode; " + ", ".join(msgs))

    # ---- C6 A/B input bridge -------------------------------------------------
    ok = True
    env, off, t = train.envs[0], train.offsets[0], train.tensors[0]
    for mode in ("xy", "grid", "regular"):
        acfg = agent_cfg_for_mode(mode, "continuous")
        pq, gq = enumerate_pairs(cells, "train", "train")
        idx = rng.choice(len(pq), 1000, replace=False)
        pq, gq = pq[idx], gq[idx]
        xa = pair_inputs(t, acfg, pq, gq)
        pos = np.stack([pq // S, pq % S], 1)
        goals = np.stack([gq // S, gq % S], 1)
        xb = build_rnn_input(
            sensory=sensory_vec(env, pos, "omni") if acfg.input_sensory else None,
            prev_action=None, prev_reward=None,
            grid_state=grid_state_vec(pos, off, sgb) if acfg.input_grid_state else None,
            cfg=acfg, device="cpu",
            goal_vec=xy_vec(goals, S) if acfg.goal_channel == "abs" else None,
            xy_state=xy_vec(pos, S) if acfg.input_xy_state else None,
            goal_grid_state=grid_state_vec(goals, off, sgb) if acfg.input_goal_grid_state else None,
            goal_sensory=goal_sensory_vec(env, goals, "omni") if acfg.goal_sensory == "omni" else None,
            sensory_dim=args.observation_size,
        )[:, 0].numpy()
        ok &= xa.shape == xb.shape and np.array_equal(xa, xb)
    gate("C6", ok, "pair_inputs bit-identical to build_rnn_input at prev_action=0, all three modes")

    # ---- C7 grid-code non-invariance -------------------------------------------
    cos = []
    for _ in range(100):
        x1, y1, x2, y2 = rng.randint(0, vh.Npos - 6, size=4)
        d1 = sgb[:, x1, y1] - sgb[:, x1 + 5, y1]
        d2 = sgb[:, x2, y2] - sgb[:, x2 + 5, y2]
        cos.append(float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12)))
    gate("C7", np.mean(cos) < 0.5,
         f"cos(Δgbook@p1, Δgbook@p2) for the same d=(5,0): mean={np.mean(cos):.3f} "
         f"min={np.min(cos):.3f} max={np.max(cos):.3f} — NOT translation-invariant")

    # ---- C8 scaffold margin -------------------------------------------------------
    specs = split.train + split.base_val
    gaps = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            gaps.append(toroidal_gap(a.offset, a.size, b.offset, b.size, split.period))
    gate("C8", min(gaps) >= split.margin,
         f"min pairwise Chebyshev gap over {len(specs)} boxes = {min(gaps)} (margin {split.margin})")

    print()
    if FAILS:
        print(f"PRE-FLIGHT FAILED: {FAILS}")
        sys.exit(1)
    print("PRE-FLIGHT PASSED: C1-C8")


if __name__ == "__main__":
    main()
