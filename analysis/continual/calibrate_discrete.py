"""What is each regulariser worth against the *discrete* BC objective?

The continuous suite was run twice because coefficient ranges were taken from
papers whose loss was a cross-entropy of order 1, while that suite's loss was a
Gaussian negative log-likelihood of order 10. DER++ and CLEAR were both swept
over a range where the regularisation term never reached 3% of the objective,
and the conclusion "the method does not help here" was really "the knob was
never turned on". ``calibrate_beta.py`` exists because the hypernetwork nearly
became the third instance.

The discrete suite changes the loss again, and in the opposite direction: a
Categorical(4) cross-entropy sits around ln(4) = 1.39 rather than around 10.
That means two things, and neither is a reason to skip this step:

  * the coefficients that worked in the *continuous* runs are now wrong by
    roughly the ratio of the two loss scales, so porting them would repeat the
    original error with the sign flipped;
  * the papers' published values may finally be close to right, since this is
    the loss geometry they were calibrated on -- but "may be" is not "is", and
    the whole point of the exercise is to look before sweeping.

Run it against the discrete pretraining checkpoint and read the ratio column:

    python -m analysis.continual.calibrate_discrete --ckpt <pretrain_disc.pt>

A coefficient whose ratio is 1e-3 is off. One whose ratio is 1e3 has stopped
the policy learning at all -- watch the bc_loss column climb, which is the
signature of a regulariser that has bought retention by refusing to learn. The
sweep for each method wants the decades bracketing ratio ~1e-1.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield_nav.config import (
    EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig)
from hopfield_nav.continual.base import build_method
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.policy.isolate import warm_start
from hopfield_nav.training.rnn_sequential import run_sequential_blocks
from hopfield_nav.world.env import GridEnv

#: method -> (the coefficient's kwarg name, which loss key it lands in,
#:            candidate values, any fixed kwargs the method needs to function).
#:
#: The loss key matters: EWC and SI add a parameter-space `penalty`, while
#: LwF, CLEAR and DER++ add an output-space `aux_loss`. Reading the wrong key
#: reports a flat zero for half the table, which looks exactly like a
#: coefficient that is too small and would send the sweep the wrong way.
#: The brackets run wider than the continuous suite's on purpose. A dry run at
#: random init put every ratio below 3e-3 even at the top of the published
#: ranges -- which is NOT the calibration (an untrained agent three updates in
#: has barely moved off its anchor, so the penalty is near zero by
#: construction) but it does say which direction the bracket has to reach. A
#: range that turns out to be too wide costs a few minutes of dry run; one that
#: is too narrow costs a whole wave, which is what happened to DER++ and CLEAR.
SPECS: dict[str, tuple[str, str, list[float], dict]] = {
    "online_ewc": ("lam",        "penalty",  [1, 1e2, 1e4, 1e5, 1e6, 1e7], {}),
    "si":         ("lam",        "penalty",  [0.1, 10, 1e3, 1e4, 1e5, 1e6], {}),
    "lwf":        ("alpha",      "aux_loss", [0.1, 1, 10, 100, 1e3, 1e4], {}),
    "clear":      ("clone_coef", "aux_loss", [0.01, 1, 10, 100, 1e3, 1e4],
                   {"replay_batches": 1}),
    "derpp":      ("alpha",      "aux_loss", [0.1, 1, 10, 100, 1e3, 1e4],
                   {"replay_batches": 1}),
}


def run_one(method: str, coef_name: str, coef: float, loss_key: str,
            fixed: dict, args, ckpt) -> dict:
    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode="discrete", goal_radius=0.5),
        agent=RNNAgentConfig(hidden_size=args.hidden_size,
                             movement_mode="discrete"),
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
    agent = RNNAgent(cfg.agent, input_dim)
    if ckpt is not None:
        warm_start(agent, ckpt["agent_state_dict"])
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr)

    seen: list[tuple[int, dict]] = []
    run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=opt, envs=envs,
        device=torch.device("cpu"), n_eval_trials=1,
        on_update=lambda u: seen.append((u.block, u.losses)),
        method=build_method(method, **{coef_name: coef}, **fixed),
    )
    # Only blocks after the first: before the first boundary there is nothing
    # to regularise against and the term is absent by construction, so
    # averaging it in would halve every ratio and make every coefficient look
    # too small.
    post = [d for b, d in seen if b > 0]
    bc = float(np.mean([d["move_loss"] for d in post])) if post else float("nan")
    term = float(np.mean([d.get(loss_key, 0.0) for d in post])) if post else 0.0
    return {"method": method, "coef_name": coef_name, "coef": coef,
            "bc_loss": bc, "term": term, "loss_key": loss_key,
            "ratio": term / abs(bc) if bc else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--methods", nargs="+", default=sorted(SPECS),
                   choices=sorted(SPECS))
    p.add_argument("--n_envs", type=int, default=3)
    p.add_argument("--updates", type=int, default=25)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None,
                   help="Write the table as JSON so the wave scripts can read "
                        "their ranges from it instead of from a log.")
    args = p.parse_args()

    ckpt = (torch.load(args.ckpt, map_location="cpu", weights_only=False)
            if args.ckpt else None)
    if ckpt is not None:
        saved = (ckpt.get("cfg") or {}).get("agent") or {}
        mode = saved.get("movement_mode")
        if mode != "discrete":
            raise SystemExit(
                f"--ckpt is a {mode!r} checkpoint. Calibrating the discrete "
                "objective against continuous-trained weights measures the "
                "wrong loss scale, which is the exact error this script "
                "exists to prevent.")

    rows: list[dict] = []
    for method in args.methods:
        coef_name, loss_key, cands, fixed = SPECS[method]
        print(f"\n=== {method}  ({coef_name}, reported in '{loss_key}') ===")
        print(f"{coef_name:>12}  {'bc_loss':>10}  {'term':>12}  {'term/bc':>10}")
        print("-" * 50)
        for coef in cands:
            r = run_one(method, coef_name, coef, loss_key, fixed, args, ckpt)
            rows.append(r)
            print(f"{r['coef']:>12g}  {r['bc_loss']:>10.4f}  "
                  f"{r['term']:>12.6g}  {r['ratio']:>10.4g}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows,
                       "meta": {"movement_mode": "discrete",
                                "n_envs": args.n_envs,
                                "updates": args.updates,
                                "lr": args.lr, "seed": args.seed,
                                "ckpt": args.ckpt}}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
