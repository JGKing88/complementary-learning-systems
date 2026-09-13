"""Why does one env die in the continual protocol when nav on it is 1.00?

The continual protocol (analysis/continual/agenthash.py) reported ood_corner
u1200's OOD env 2 -- goal (0, 3), offset (1529, 1304) -- at 0/40 in its own
block while eval_all's nav on the same env, goal preloaded, was 16/16. So the
readout works once the goal is in memory; what fails is *discovery* under the
protocol's memory state (envs 0 and 1's goals stored, nothing of env 2's).

Four measurements per env, no squinting:

  1. exploration with an EMPTY memory: per-env coverage and goal-find rate over
     16 deterministic 200-step trials. If env 2 alone is low, the explorer
     never sweeps that goal's neighbourhood (an edge/corner avoidance).
  2. the same explorer with the CONTINUAL memory state (goals 0 and 1 stored,
     via the distractor hook). If env 2 is fine in (1) and dead here, the
     stored patterns are what kill it.
  3. goal-absent ||q|| per env with goals 0 and 1 stored -- the phantom signal
     the agent reads in env 2's block. Inside the gate band means the agent is
     in exploit mode following a phantom rather than exploring
     (DUAL_TRAINING section 9.3; alias_multiplicity.py item 2).
  4. code overlap: max cosine between the env's 400 cell codes and each stored
     goal -- the alias test -- and the readout basin with the env's own goal
     stored once alongside goals 0 and 1.

    python -m analysis.nav_tri.dead_env_probe [--ckpt ...] [--split place=ood]

Two more env sources, for comparing policies on the SAME arenas:

  --envs_from CKPT   mint the set from another run's world record (the scaffold
                     field is a pure function of the encoder, so any policy
                     trained on the same encoder can be run on it) -- e.g.
                     d0_base on the env the corner model failed.
  --legacy_replay    agenthash's pre-record draw (RandomState(cfg.seed), skip
                     envs_per_world*num_worlds, make_env x n, place_envs spread
                     after np.random.seed(0)) -- the envs d0_base's continual
                     figures were made on; the offsets are checked against a
                     history with --check_history.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation import metrics
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_world_for_split, load_agent,
)
from hopfield_nav.rollout.distractors import goal_encoding
from analysis.nav_tri.readout_field import field_over_cells, integrate


def per_env(per_trial, n_envs, size):
    cov = [[] for _ in range(n_envs)]
    found = [[] for _ in range(n_envs)]
    for _nd, env_idx, _t, n_cells, hit, _s in per_trial:
        cov[env_idx].append(n_cells / (size * size))
        found[env_idx].append(hit)
    return [(float(np.mean(c)), float(np.mean(f))) for c, f in zip(cov, found)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/orcd/pool/003/jackking/cls_runs/agent_ckpts/"
                   "navigate_navp2_ood_corner_s42_22629938/navigate_u1200.pt")
    p.add_argument("--split", default="place=ood")
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--n_envs", type=int, default=6)
    p.add_argument("--stored", default="0,1",
                   help="env indices whose goals are in memory for (2)-(4)")
    p.add_argument("--num_trials", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--envs_from", default=None,
                   help="checkpoint whose world record mints the env set "
                        "(default: --ckpt itself)")
    p.add_argument("--legacy_replay", action="store_true",
                   help="agenthash's pre-record env draw instead of a --split")
    p.add_argument("--check_history", default=None,
                   help="continual history .json whose recorded offsets/goals "
                        "the env set must match (asserts)")
    a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = a.n_envs
    enc, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(dev),
                                      getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    if a.legacy_replay:
        from hopfield_nav.training.world_setup import build_field
        from hopfield_nav.world.env import make_env
        from hopfield_nav.world.scaffold import place_envs
        vh = build_field(cfg, enc)
        rng = np.random.RandomState(cfg.seed)
        for _ in range(cfg.envs_per_world * cfg.num_worlds):
            rng.randint(0, 10_000_000)
        envs = [make_env(cfg.env, cfg.agent.movement_mode,
                         seed=int(rng.randint(0, 10_000_000)))
                for _ in range(a.n_envs)]
        np.random.seed(0)     # agenthash draws offsets right after seeding
        offsets = place_envs(a.n_envs, cfg.env.size, vh.Npos, np.random,
                             placement="spread")
        source = "legacy replay"
    else:
        src_ckpt = a.envs_from or a.ckpt
        if a.envs_from:
            # The env set is minted from the OTHER run's record and domains; the
            # field is a pure function of (encoder, lambdas, Npos, fwhm), which
            # both runs share, so this policy can be run on those arenas.
            src_cfg = cfg_from_checkpoint(torch.load(
                src_ckpt, map_location="cpu", weights_only=False)["config"])
            src_cfg.num_val_envs = a.n_envs
            assert src_cfg.encoder_checkpoint == cfg.encoder_checkpoint, (
                "env source trained on a different encoder; its arenas are not "
                "this policy's scaffold")
        else:
            src_cfg = cfg
        envs, vh, offsets = eval_world_for_split(
            src_cfg, enc, str(dev), ckpt_path=src_ckpt, split=a.split,
            val_seed=a.val_seed)
        source = f"{a.split} from {src_ckpt}"
    if a.check_history:
        import json
        ex = json.load(open(a.check_history))["metadata"]["extra"]
        want_o = [tuple(o) for o in ex["env_offsets_per_iter"][0]]
        want_g = [tuple(g) for g in ex["env_goals_per_iter"][0]]
        got_o = [tuple(int(v) for v in o) for o in offsets]
        got_g = [tuple(int(v) for v in e.goal_location) for e in envs]
        assert got_o == want_o and got_g == want_g, (
            f"env set differs from {a.check_history}:\n  offsets {got_o}\n"
            f"  vs      {want_o}\n  goals {got_g}\n  vs    {want_g}")
        print(f"env set matches {a.check_history}")
    print(f"env source: {source}")
    n, size, R = len(envs), envs[0].size, float(cfg.env.goal_radius)
    goals = [goal_encoding(vh, offsets[j], envs[j].goal_location) for j in range(n)]
    stored = [int(s) for s in a.stored.split(",") if s != ""]
    agent = load_agent(cfg, ck["agent_state_dict"], D, dev)
    print(f"ckpt={a.ckpt}\nsplit={a.split}  envs={n}  stored-in-memory={stored}")
    for j in range(n):
        print(f"  env {j}: offset={tuple(offsets[j])} goal={tuple(envs[j].goal_location)}")

    # (1) explorer, empty memory
    pt = []
    metrics.evaluate_exploration(agent, envs, vh, offsets, cfg, dev,
                                 num_trials=a.num_trials, max_steps=200,
                                 n_distractors_list=[0], per_trial=pt)
    empty = per_env(pt, n, size)

    # (2) explorer, continual memory state: hook the distractor sampler so the
    # "distractors" are exactly the stored goals.
    orig = metrics.sample_distractors
    metrics.sample_distractors = lambda vectorhash, off, sz, k, rng: [goals[s] for s in stored][:k]
    try:
        pt = []
        metrics.evaluate_exploration(agent, envs, vh, offsets, cfg, dev,
                                     num_trials=a.num_trials, max_steps=200,
                                     n_distractors_list=[len(stored)], per_trial=pt)
    finally:
        metrics.sample_distractors = orig
    cont = per_env(pt, n, size)

    # (3)+(4) field measurements, no rollouts
    hop = Hopfield(D, beta=cfg.hopfield.beta, device=str(dev))
    for s in stored:
        hop.input_memory(torch.from_numpy(goals[s]).float())
    print(f"\n{'env':>3s} {'cov(empty)':>10s} {'find(empty)':>11s} {'cov(cont)':>10s} {'find(cont)':>10s} "
          f"{'|q| mean':>9s} {'|q| p90':>8s} {'max cos vs stored':>18s} {'basin(own+stored)':>18s}")
    for j in range(n):
        fld = field_over_cells(vh, hop, size, offsets[j], str(dev))
        qn = np.linalg.norm(fld, axis=-1)
        ox, oy = offsets[j]
        codes = np.asarray(vh.encoded_Phi[ox:ox + size, oy:oy + size], dtype=np.float64).reshape(-1, D)
        codes /= np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12
        cos = max(float(np.max(codes @ (goals[s] / (np.linalg.norm(goals[s]) + 1e-12))))
                  for s in stored)
        hop_own = Hopfield(D, beta=cfg.hopfield.beta, device=str(dev))
        for s in stored:
            hop_own.input_memory(torch.from_numpy(goals[s]).float())
        if j not in stored:
            hop_own.input_memory(torch.from_numpy(goals[j]).float())
        fld_own = field_over_cells(vh, hop_own, size, offsets[j], str(dev))
        _e, reached, _p = integrate(fld_own, size, envs[j].goal_location, R)
        tag = " (own goal IS stored)" if j in stored else ""
        print(f"{j:>3d} {empty[j][0]:>10.3f} {empty[j][1]:>11.2f} {cont[j][0]:>10.3f} {cont[j][1]:>10.2f} "
              f"{qn.mean():>9.3f} {np.percentile(qn, 90):>8.3f} {cos:>18.3f} {float(reached.mean()):>18.3f}{tag}")


if __name__ == "__main__":
    main()
