"""What the two controllers are actually handed: Agent-HaSH's q vs idea 1's (d, c).

    python -m analysis.sensory_key.readout_compare \
        --hash_ckpt <task3r distal ckpt> --gmlp_ckpt <task3r gmlp ckpt> --out <json>

Both controllers get a 2-D "goal direction" input and must learn when to follow
it. Whatever makes one of them learn faster has to be in how that input
behaves, so this measures the input itself -- no policy, no rollouts -- at every
cell of the task3r val arenas, for many goals, with 0 and 10 distractors:

1. **Direction**, after the goal is stored: angular error of q and of d against
   the true direction to the goal, by distance.
2. **Scale**: |q| by distance (d is unit everywhere; c is its separate gate).
3. **Gate**: how well the model's own presence signal -- |q| for Agent-HaSH, c
   for idea 1 -- separates "own goal stored" from "only distractors stored"
   (AUC per distance bin; 1 = perfectly separable).

Idea 1 is queried the way the agent queries it: one live view per cell at a
random heading.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint
from hopfield_nav.memory import backend as mb
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world.generate import build_envs
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import WorldSpec
from analysis.nav_tri.signal_separability import q_at
from analysis.sensory_key.retrieval_test import auc

BINS = [("1-2", 0.5, 2.5), ("3-5", 2.5, 5.5), ("6-10", 5.5, 10.5), (">10", 10.5, 1e9)]


def ang(a, b):
    na = np.linalg.norm(a, axis=1)
    cos = (a * b).sum(1) / np.maximum(na * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hash_ckpt", required=True)
    p.add_argument("--gmlp_ckpt", required=True)
    p.add_argument("--goals_per_env", type=int, default=8)
    p.add_argument("--dist", type=int, nargs="+", default=[0, 10])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    hcfg = cfg_from_checkpoint(torch.load(a.hash_ckpt, map_location="cpu", weights_only=False)["config"])
    gcfg = cfg_from_checkpoint(torch.load(a.gmlp_ckpt, map_location="cpu", weights_only=False)["config"])
    world = a.hash_ckpt.rsplit("/", 1)[0] + "/world.json"
    specs = WorldSpec.read(world).split.base_val
    encoder, enc_cfg, gain = load_encoder(hcfg.encoder_checkpoint, str(dev))
    beta = hcfg.hopfield.beta if hcfg.hopfield.beta is not None else float(gain)
    torch.manual_seed(0)
    vh = VectorHash(hcfg.vectorhash)
    vh.build_scaffold()
    vh.precompute_encoded_phi(encoder, hcfg.fwhm_ratio, device=str(dev))
    ro = mb.readout_for(gcfg)
    envs = build_envs(specs, gcfg.env, gcfg.agent.movement_mode)   # panorama on

    rng = np.random.RandomState(0)
    rec = {nd: dict(dist=[], q_err=[], q_mag=[], d_err=[], q_pre=[], q_post=[],
                    c_pre=[], c_post=[]) for nd in a.dist}
    for env, spec in zip(envs, specs):
        S, off = env.size, tuple(spec.offset)
        gx, gy = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        cells = np.stack([gx.ravel(), gy.ravel()], 1)
        for _ in range(a.goals_per_env):
            goal = tuple(int(v) for v in rng.randint(0, S, 2))
            env.set_goal(goal)
            psi = rng.uniform(-np.pi, np.pi, len(cells))
            views = np.stack([env.obs_at(tuple(c), p_) for c, p_ in zip(cells, psi)])
            delta = np.array(goal) - cells
            dist = np.linalg.norm(delta, axis=1)
            away = dist > 0.5
            for nd in a.dist:
                dseed = int(rng.randint(2 ** 31))
                # Agent-HaSH memory: distractors, then + goal
                hop = Hopfield(enc_cfg.out_dim, beta=beta, device=str(dev))
                for pat in sample_distractors(vh, off, S, nd, np.random.RandomState(dseed)):
                    hop.input_memory(torch.from_numpy(pat).float())
                q_pre = q_at(vh, hop, cells, off, dev, ())[0]
                hop.input_memory(torch.from_numpy(goal_encoding(vh, off, goal)).float())
                q_post = q_at(vh, hop, cells, off, dev, ())[0]
                # idea-1 memory: foreign goals, then + goal
                mem = mb.new_kv_memory(gcfg, off, S, nd, np.random.RandomState(dseed))
                _, _, _, c_pre = ro.signal([mem] * len(cells), views, psi, cells, off)
                ro.write_goal(mem, env, goal, off)
                d_post, _, _, c_post = ro.signal([mem] * len(cells), views, psi, cells, off)
                r = rec[nd]
                r["dist"].append(dist[away])
                r["q_err"].append(ang(q_post[away], delta[away]))
                r["q_mag"].append(np.linalg.norm(q_post[away], axis=1))
                r["d_err"].append(ang(d_post[away], delta[away]))
                r["q_pre"].append(np.linalg.norm(q_pre[away], axis=1))
                r["q_post"].append(np.linalg.norm(q_post[away], axis=1))
                r["c_pre"].append(c_pre[away])
                r["c_post"].append(c_post[away])

    out = {}
    for nd, r in rec.items():
        R = {k: np.concatenate(v) for k, v in r.items()}
        rows = []
        print(f"\n=== {nd} distractors ({len(R['dist'])} cell-goal pairs) ===")
        print(f"{'dist':>6} | {'q err':>7} {'q>45':>6} {'|q| med':>8} {'|q| cv':>7} | "
              f"{'d err':>7} {'d>45':>6} | {'gate AUC |q|':>12} {'gate AUC c':>11}")
        for name, lo, hi in BINS:
            m = (R["dist"] >= lo) & (R["dist"] < hi)
            qm = R["q_mag"][m]
            row = dict(bin=name, n=int(m.sum()),
                       q_err=float(R["q_err"][m].mean()), q_bad=float((R["q_err"][m] > 45).mean()),
                       q_mag_med=float(np.median(qm)), q_mag_cv=float(qm.std() / max(qm.mean(), 1e-12)),
                       d_err=float(R["d_err"][m].mean()), d_bad=float((R["d_err"][m] > 45).mean()),
                       gate_auc_q=(auc(R["q_post"][m], R["q_pre"][m]) if nd else float("nan")),
                       gate_auc_c=(auc(R["c_post"][m], R["c_pre"][m]) if nd else float("nan")))
            rows.append(row)
            print(f"{name:>6} | {row['q_err']:7.1f} {row['q_bad']:6.2f} {row['q_mag_med']:8.3g} "
                  f"{row['q_mag_cv']:7.2f} | {row['d_err']:7.1f} {row['d_bad']:6.2f} | "
                  f"{row['gate_auc_q']:12.3f} {row['gate_auc_c']:11.3f}")
        out[nd] = rows
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
