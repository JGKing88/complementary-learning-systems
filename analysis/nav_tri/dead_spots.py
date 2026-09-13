"""How common are dead spots -- goal cells a deterministic explorer never finds?

dead_env_probe.py showed that a "dead env" in the continual protocol is a
visitation hole: the explorer's coverage is normal but the goal cell is never
swept, and mean coverage (a fraction of cells) cannot see which cells. This
counts the holes.

Per env: B deterministic 200-step explore trials from random starts with an
EMPTY memory (evaluation.batched.batched_exploration_trials, the same primitive
`evaluate_exploration` uses). The swept mask (evaluation/swept.py) marks every
sub-pixel within goal_radius of the path, so a goal at integer cell (x, y) is
found by a trial iff that trial's mask is set at sub-pixel (8x, 8y). A DEAD
CELL for k attempts is one no trial among the first k sets. Because goals are
drawn uniformly over cells, the dead fraction at k = 40 (the continual
protocol's primary block) IS the probability that a fresh env is dead.

Reports, over the env set: dead fraction at k in {16, 40, 64} (mean / min /
max across envs), the edge-vs-interior split of dead cells, and an aggregate
20x20 dead-count map. Aggregate statistics, not trajectory pictures.

    python -m analysis.nav_tri.dead_spots --ckpt CK [--split place=ood]
                                          [--envs_from OTHER_CK] [--n_envs 24]
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation import batched
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_world_for_split, load_agent,
)
from hopfield_nav.evaluation.metrics import random_start

_INSTANCES: list = []


class _RecordingSweptArea(batched.SweptArea):
    """SweptArea that keeps itself findable, so the per-trial masks can be read
    after batched_exploration_trials returns only their fractions."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _INSTANCES.append(self)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="place=ood")
    p.add_argument("--envs_from", default=None)
    p.add_argument("--n_envs", type=int, default=24)
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--ks", default="16,40,64")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions from the policy (temperature 1) instead of "
                        "the argmax -- the one-flag test of whether the holes belong "
                        "to the policy or to its argmax (EXPERIMENTS_NAV_P2 section 37.6)")
    a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ks = [int(k) for k in a.ks.split(",")]

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
    src_ckpt = a.envs_from or a.ckpt
    if a.envs_from:
        src_cfg = cfg_from_checkpoint(torch.load(
            src_ckpt, map_location="cpu", weights_only=False)["config"])
        src_cfg.num_val_envs = a.n_envs
        assert src_cfg.encoder_checkpoint == cfg.encoder_checkpoint
    else:
        src_cfg = cfg
    envs, vh, offsets = eval_world_for_split(
        src_cfg, enc, str(dev), ckpt_path=src_ckpt, split=a.split,
        val_seed=a.val_seed)
    agent = load_agent(cfg, ck["agent_state_dict"], D, dev)
    agent.eval()
    n, size, grid = len(envs), envs[0].size, 8
    print(f"policy={a.ckpt}\nenvs: {a.split} from {src_ckpt} ({n} envs, size {size})"
          f"\ntrials/env={a.trials} max_steps={a.max_steps} "
          f"{'SAMPLED (temperature 1)' if a.stochastic else 'deterministic'}, empty memory")

    batched.SweptArea = _RecordingSweptArea
    dead_frac = {k: [] for k in ks}
    dead_edge = {k: 0 for k in ks}
    dead_interior = {k: 0 for k in ks}
    n_edge = 4 * size - 4
    n_interior = size * size - n_edge
    dead_map = np.zeros((size, size), dtype=int)     # at k = 40 (or last k)
    k_map = 40 if 40 in ks else ks[-1]
    cell_union = []
    goal_dead = []
    rng = np.random.RandomState(a.seed)
    xs, ys = np.meshgrid(np.arange(size) * grid, np.arange(size) * grid, indexing="ij")
    for j, (env, off) in enumerate(zip(envs, offsets)):
        goal = tuple(int(v) for v in env.goal_location)
        hops = [Hopfield(D, beta=cfg.hopfield.beta, device=str(dev)) for _ in range(a.trials)]
        starts = [random_start(size, goal, rng) for _ in range(a.trials)]
        _INSTANCES.clear()
        visited, found, _steps, _swept = batched.batched_exploration_trials(
            agent=agent, env=env, env_offset=off, vectorhash=vh, hopfields=hops,
            cfg=cfg, device=dev, starts=starts, max_steps=a.max_steps,
            deterministic=not a.stochastic)
        sa = _INSTANCES[-1]
        M = sa._mask.reshape(a.trials, sa.res, sa.res)
        union_cells = set().union(*visited)
        cell_union.append(len(union_cells) / (size * size))
        for k in ks:
            u = M[:k].any(axis=0)
            hit = u[xs, ys]                       # (size, size): goal cell found by >=1 of k
            dead = ~hit
            dead_frac[k].append(float(dead.mean()))
            edge = np.zeros((size, size), dtype=bool)
            edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
            dead_edge[k] += int((dead & edge).sum())
            dead_interior[k] += int((dead & ~edge).sum())
            if k == k_map:
                dead_map += dead.astype(int)
                goal_dead.append(bool(dead[goal]))
        print(f"  env {j:2d} goal={goal} offset={tuple(int(v) for v in off)}: "
              + "  ".join(f"dead@{k}={dead_frac[k][-1]:.3f}" for k in ks)
              + f"  cell_union={cell_union[-1]:.2f}  found(any of {a.trials})={any(found)}"
              + ("  <-- ITS GOAL IS A DEAD CELL" if goal_dead[-1] else ""), flush=True)

    print("\n=== dead-cell fraction across envs (goal drawn uniformly => P(dead env)) ===")
    for k in ks:
        v = np.asarray(dead_frac[k])
        print(f"  k={k:2d} attempts: mean {v.mean():.3f}  min {v.min():.3f}  max {v.max():.3f}  "
              f"| edge cells dead {dead_edge[k] / (n * n_edge):.3f}  interior cells dead "
              f"{dead_interior[k] / (n * n_interior):.3f}")
    print(f"  envs whose own goal is a dead cell at k={k_map}: {sum(goal_dead)}/{n}")
    print(f"  cell-level union coverage (visited cells, {a.trials} trials): mean {np.mean(cell_union):.3f}")
    print(f"\n=== dead-count map at k={k_map} (rows x=0..{size-1}, cols y=0..{size-1}; "
          f"value = number of the {n} envs in which that cell is dead) ===")
    for x in range(size):
        print("  " + " ".join(f"{dead_map[x, y]:2d}" for y in range(size)))


if __name__ == "__main__":
    main()
