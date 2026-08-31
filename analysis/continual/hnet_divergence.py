"""Did the hypernetwork actually condition on the task, or collapse to one policy?

The failure this exists to catch is the quietest one available. The generator
starts with a deliberately small output layer, so that every task begins at the
warm-started base -- which is what makes the arm comparable to the pretrained
controls. If that output never grows, the generator emits approximately the
same 73k weights whatever embedding it is handed, every environment shares one
policy, and the arm is the naive baseline wearing a hypernetwork's metadata.

It would produce a completely ordinary run. The loss curve would look right,
the penalty would be nonzero and scale with beta, retention would come out low,
and the honest-looking conclusion would be "the hypernetwork does not help
here" when the truth is "the hypernetwork was never switched on". No test of
the method's *mechanics* catches this, because the mechanics are all correct --
it is a question about where optimisation ended up.

Two numbers per block boundary settle it:

    conditioned_frac   ||generator output|| / ||base||, averaged over tasks.
                       Near zero means the task-conditioned part contributes
                       nothing and every task shares the base.
    pairwise_div       mean over task pairs of ||w_i - w_j|| / ||w_i||. This is
                       the one that matters: it is nonzero only if different
                       tasks genuinely get different weights.

    python -m analysis.continual.hnet_divergence --ckpt <pretrain.pt> --beta 10000
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch

from hopfield_nav.config import (
    EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig)
from hopfield_nav.continual.base import build_method
from hopfield_nav.policy.agent_rnn import compute_rnn_input_dim
from hopfield_nav.policy.hypernet import HyperRNNAgent
from hopfield_nav.policy.isolate import warm_start
from hopfield_nav.training.rnn_sequential import run_sequential_blocks
from hopfield_nav.world.env import GridEnv


@torch.no_grad()
def divergence(agent: HyperRNNAgent) -> dict:
    """How task-dependent the generated weights currently are."""
    ws = [agent.generate(t) for t in range(agent.n_tasks)]
    hyper_only = [agent.hyper(t) for t in range(agent.n_tasks)]
    base_norm = (float(agent.base.norm()) if agent.base is not None
                 else float("nan"))
    pairs = list(itertools.combinations(range(agent.n_tasks), 2))
    div = float(np.mean([float((ws[i] - ws[j]).norm() / ws[i].norm())
                         for i, j in pairs])) if pairs else 0.0
    return {
        "conditioned_frac": float(np.mean([float(h.norm()) for h in hyper_only]))
                            / base_norm if base_norm == base_norm else None,
        "pairwise_div": div,
        "weight_norm": float(np.mean([float(w.norm()) for w in ws])),
        "base_norm": base_norm,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--beta", type=float, default=10000.0)
    p.add_argument("--base", default="learned")
    p.add_argument("--n_envs", type=int, default=5)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode="continuous", goal_radius=0.5),
        agent=RNNAgentConfig(hidden_size=args.hidden_size,
                             movement_mode="continuous", init_log_std=0.0),
        bc=RNNBCConfig(lr=args.lr, epochs=1, n_minibatches=1, max_grad_norm=1.0),
        n_envs=args.n_envs, updates_per_env=args.updates, batch_envs=1,
        steps_per_rollout=args.max_steps, eval_max_steps=args.max_steps,
        seed=args.seed,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    envs = [GridEnv(size=cfg.env.size, observation_size=cfg.env.observation_size,
                    seed=int(rng.randint(0, 10_000_000)))
            for _ in range(cfg.n_envs)]

    agent = HyperRNNAgent(
        cfg.agent, compute_rnn_input_dim(cfg.agent, cfg.env.observation_size),
        cfg.n_envs, base=args.base)
    if args.ckpt:
        warm_start(agent, torch.load(args.ckpt, map_location="cpu",
                                     weights_only=False)["agent_state_dict"])
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr)
    method = build_method("hnet", beta=args.beta)

    trace = []
    print(f"{'block':>6}  {'update':>7}  {'cond_frac':>10}  {'pair_div':>10}")
    print("-" * 40)

    def on_update(u):
        if u.update % 50 and u.update != cfg.updates_per_env:
            return
        d = divergence(agent)
        d.update({"block": u.block, "update": u.update})
        trace.append(d)
        cf = d["conditioned_frac"]
        print(f"{u.block:>6}  {u.update:>7}  "
              f"{'n/a' if cf is None else f'{cf:>10.6f}'}  "
              f"{d['pairwise_div']:>10.6f}")

    run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=opt, envs=envs,
        device=torch.device("cpu"), n_eval_trials=1,
        on_update=on_update, method=method)

    final = trace[-1] if trace else {}
    div = final.get("pairwise_div", 0.0)
    print()
    if div < 1e-3:
        print(f"COLLAPSED: pairwise divergence {div:.2e}. Every task is getting "
              "essentially the same weights, so this arm is the naive baseline "
              "with a hypernetwork's metadata. Raise --hnet_init_out_scale, or "
              "the result is uninterpretable.")
    else:
        print(f"CONDITIONED: pairwise divergence {div:.4f}. Different tasks get "
              "genuinely different weights, so a low retention score is a "
              "result about the method rather than about its initialisation.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"trace": trace, "beta": args.beta, "base": args.base,
                       "seed": args.seed}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
