"""Is this explore policy using memory, or is it a well-tuned random walk?

`coverage_reference` establishes that nothing *memoryless* beats ~0.56 on a
20x20 arena in 400 steps, and that where a memoryless walker lands is set
almost entirely by two numbers: its stride and how sharply it turns. So a
coverage score on its own cannot distinguish "the policy learned to sweep" from
"the policy learned a good stride" -- both show up as a bigger number.

This separates them. It runs the *exact* evaluation path
(`batched_exploration_trials`, deterministic, distractors only), measures the
policy's realized stride and turn-angle distribution, then simulates a
memoryless walker with those same statistics and reports both coverages side by
side:

    measured 0.52  matched-walk 0.52  excess +0.00   -> no memory in use
    measured 0.68  matched-walk 0.53  excess +0.15   -> the policy is tracking
                                                        where it has been

`excess` is the number to read. It is the part of coverage that the policy's
own movement statistics do not explain, which is the only part any amount of
reward shaping cannot buy.

    python -m hopfield_nav.probes.policy_motion --ckpt <navigate_uN.pt> \\
        --n_envs 4 --trials 32 --max_steps 400 --n_dist 10
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from ..config import TrainConfig
from ..encoder_io import load_encoder
from ..evaluation.batched import batched_exploration_trials
from ..evaluation.checkpoint_io import cfg_from_checkpoint
from ..policy.agent import NavAgent, compute_input_dim
from ..rollout.distractors import sample_distractors
from ..training.world_setup import build_field
from ..world.env import make_env
from ..world.world import build_world
from ..world.walks import (
    random_starts, simulate_coverage, unit_vectors,
)


class MotionRecorder:
    """Accumulates stride, turn angle and blocked-step counts over a run."""

    def __init__(self) -> None:
        self.stride: list[np.ndarray] = []
        self.realized: list[np.ndarray] = []
        self.turn: list[np.ndarray] = []
        self.blocked: list[np.ndarray] = []
        self._prev: np.ndarray | None = None

    def __call__(self, step, pos_before, actions, pos_after) -> None:
        moved = pos_after - pos_before
        self.stride.append(np.linalg.norm(actions, axis=1))
        self.realized.append(np.linalg.norm(moved, axis=1))
        # A step the clip absorbed entirely: the wall ate it. A policy that
        # spends its budget pushing into a wall reads as a low realized stride
        # with a high commanded one, which is a different failure from simply
        # taking small steps.
        self.blocked.append(np.linalg.norm(moved, axis=1) < 1e-9)
        if self._prev is not None:
            a, b = self._prev, actions
            na = np.linalg.norm(a, axis=1).clip(1e-12)
            nb = np.linalg.norm(b, axis=1).clip(1e-12)
            cos = np.clip((a * b).sum(1) / (na * nb), -1.0, 1.0)
            self.turn.append(np.arccos(cos))
        self._prev = actions.copy()

    def summary(self) -> dict:
        stride = np.concatenate(self.stride)
        realized = np.concatenate(self.realized)
        turn = np.concatenate(self.turn) if self.turn else np.zeros(1)
        blocked = np.concatenate(self.blocked)
        return {
            "stride_mean": float(stride.mean()),
            "stride_median": float(np.median(stride)),
            "stride_p90": float(np.percentile(stride, 90)),
            "realized_mean": float(realized.mean()),
            "blocked_frac": float(blocked.mean()),
            "turn_mean_rad": float(turn.mean()),
            # The wrapped-normal width that reproduces this turn distribution:
            # sigma such that E[cos(turn)] = exp(-sigma^2 / 2). This is the
            # `turn_sigma` axis of coverage_reference, so the matched walk can
            # be built directly from it.
            "turn_sigma": float(np.sqrt(max(
                -2.0 * np.log(max(float(np.cos(turn).mean()), 1e-6)), 0.0))),
            "cos_turn_mean": float(np.cos(turn).mean()),
        }


def matched_walk_coverage(stats: dict, size: int, steps: int, trials: int,
                          seed: int = 0) -> float:
    """Coverage of a memoryless walker with this policy's own statistics.

    Correlated-heading family, stride pinned to the policy's mean commanded
    stride and turn width to the wrapped-normal fit. Deliberately memoryless:
    the gap to the measured coverage is the estimate of what memory is buying.
    """
    rng = np.random.RandomState(seed)
    pos = random_starts(trials, size, rng)
    theta = rng.uniform(0, 2 * np.pi, trials)
    stride, sigma = stats["stride_mean"], stats["turn_sigma"]

    def fn(t, blocked):
        nonlocal theta
        theta = theta + rng.normal(0.0, sigma, trials)
        if blocked.any():
            theta[blocked] = rng.uniform(0, 2 * np.pi, int(blocked.sum()))
        return unit_vectors(theta) * stride

    return float(simulate_coverage(pos, size, steps, fn, rng).mean())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default=None,
                   help="default: the checkpoint's own")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--n_dist", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: TrainConfig = cfg_from_checkpoint(ck["config"])
    cfg.device = str(device)
    enc_path = args.encoder or cfg.encoder_checkpoint
    encoder, enc_cfg, gain = load_encoder(enc_path, str(device), cfg.encoder_gain)
    cfg.encoder_gain = gain
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)

    field = build_field(cfg, encoder)
    rng = np.random.RandomState(args.seed)
    envs = [make_env(cfg.env, cfg.agent.movement_mode,
                     seed=int(rng.randint(0, 10_000_000)))
            for _ in range(args.n_envs)]
    world = build_world(field, envs, placement="spread", size=cfg.env.size)
    embed_dim = field.encoded_Phi.shape[2]

    agent = NavAgent(cfg.agent, compute_input_dim(
        cfg.agent, embed_dim, cfg.env.observation_size)).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()

    rec = MotionRecorder()
    covs: list[float] = []
    for env, offset in zip(world.envs, world.offsets):
        hops, starts = [], []
        for _ in range(args.trials):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            for pat in sample_distractors(field, offset, env.size,
                                          args.n_dist, rng):
                hop.input_memory(torch.from_numpy(pat).float())
            hops.append(hop)
            gx, gy = env.goal_location
            while True:
                s = (int(rng.randint(0, env.size)), int(rng.randint(0, env.size)))
                if s != (gx, gy):
                    break
            starts.append(s)
        visited, _found, _s = batched_exploration_trials(
            agent=agent, env=env, env_offset=offset, vectorhash=field,
            hopfields=hops, cfg=cfg, device=device, starts=starts,
            max_steps=args.max_steps, deterministic=True, on_step=rec)
        covs.extend(len(v) / float(env.size * env.size) for v in visited)

    stats = rec.summary()
    measured = float(np.mean(covs))
    matched = matched_walk_coverage(stats, int(cfg.env.size), args.max_steps,
                                    max(len(covs), 128), seed=args.seed)
    out = {
        "ckpt": args.ckpt,
        "measured_coverage": measured,
        "matched_walk_coverage": matched,
        "excess_over_matched_walk": measured - matched,
        **stats,
    }
    print(json.dumps(out, indent=2))
    print(f"\nmeasured {measured:.3f}   matched-walk {matched:.3f}   "
          f"excess {measured - matched:+.3f}")
    print(f"  (stride {stats['stride_mean']:.2f}, turn_sigma "
          f"{stats['turn_sigma']:.2f} rad, blocked {stats['blocked_frac']:.1%} "
          f"of steps)")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
