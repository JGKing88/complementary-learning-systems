"""What is the hypernetwork penalty actually worth, against this BC loss?

Wave 2 was run twice because a coefficient range was taken from a paper whose
loss was a cross-entropy of order 1, while this suite's is a Gaussian negative
log-likelihood of order 10. DER++ and CLEAR were both swept over a range where
the regularisation term never reached 3% of the objective, and the conclusion
"the method does not help here" was really "the knob was never turned on".

The cheap way not to do that a third time is to look before sweeping. This runs
the real protocol at a handful of betas for a couple of blocks and prints the
penalty beside the BC loss, so the sweep can be centred where the two are
comparable rather than where a paper put them.

    python -m analysis.continual.calibrate_beta --ckpt <pretrain.pt>

Prints one row per beta: the mean BC loss, the mean penalty once the first
boundary has passed, and their ratio. A beta whose ratio is 1e-3 is off; one
whose ratio is 1e3 has stopped the policy learning at all. The sweep wants the
decades either side of 1.
"""
from __future__ import annotations

import argparse

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


def run_one(beta: float, args, ckpt) -> dict:
    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode="continuous", goal_radius=0.5),
        agent=RNNAgentConfig(hidden_size=args.hidden_size,
                             movement_mode="continuous", init_log_std=0.0),
        bc=RNNBCConfig(lr=args.lr, epochs=1, n_minibatches=1,
                       max_grad_norm=1.0),
        n_envs=args.n_envs, updates_per_env=args.updates,
        batch_envs=1, steps_per_rollout=args.max_steps,
        eval_max_steps=args.max_steps, seed=args.seed,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    envs = [GridEnv(size=cfg.env.size, observation_size=cfg.env.observation_size,
                    seed=int(rng.randint(0, 10_000_000)))
            for _ in range(cfg.n_envs)]
    input_dim = compute_rnn_input_dim(cfg.agent, cfg.env.observation_size)
    agent = HyperRNNAgent(cfg.agent, input_dim, cfg.n_envs,
                          emb_dim=args.emb_dim, chunk_dim=args.chunk_dim,
                          base=args.base)
    if ckpt is not None:
        warm_start(agent, ckpt["agent_state_dict"])
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr)

    seen: list[tuple[int, dict]] = []
    run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=opt, envs=envs,
        device=torch.device("cpu"), n_eval_trials=1,
        on_update=lambda u: seen.append((u.block, u.losses)),
        method=build_method("hnet", beta=beta),
    )
    # Only blocks after the first: before the first boundary there is nothing
    # to regularise and the penalty is absent by construction, so averaging it
    # in would halve every ratio and make every beta look too small.
    post = [d for b, d in seen if b > 0]
    bc = float(np.mean([d["move_loss"] for d in post]))
    pen = float(np.mean([d.get("penalty", 0.0) for d in post]))
    return {"beta": beta, "bc_loss": bc, "penalty": pen,
            "ratio": pen / abs(bc) if bc else float("nan"),
            "n_params": agent.describe()["trainable_params"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.01, 0.1, 1, 10, 100, 1000])
    p.add_argument("--n_envs", type=int, default=3)
    p.add_argument("--updates", type=int, default=25)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--chunk_dim", type=int, default=512)
    p.add_argument("--base", default="learned")
    args = p.parse_args()

    ckpt = (torch.load(args.ckpt, map_location="cpu", weights_only=False)
            if args.ckpt else None)

    print(f"{'beta':>10}  {'bc_loss':>10}  {'penalty':>12}  {'pen/bc':>10}")
    print("-" * 48)
    for beta in args.betas:
        r = run_one(beta, args, ckpt)
        print(f"{r['beta']:>10g}  {r['bc_loss']:>10.4f}  {r['penalty']:>12.6g}  "
              f"{r['ratio']:>10.4g}")
    print(f"\ntrainable params: {r['n_params']}")


if __name__ == "__main__":
    main()
