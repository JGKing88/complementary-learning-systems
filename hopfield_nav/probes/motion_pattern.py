"""What shape is the explore policy's trajectory? Random walk, circle, zipper?

Coverage says how much ground a policy covered and `policy_motion` says whether
that is more than its stride and turn width explain. Neither says what the
motion *is*, and the candidates behave very differently under further training:
a diffusive walk is at its ceiling, a perimeter orbit is a known trap, and a
boustrophedon has almost all the headroom.

So this classifies. It runs the real exploration evaluator, records the
trajectories, and computes features chosen because each one separates families
that coverage cannot:

    msd_slope        d log MSD / d log t. ~1 diffusive, ~2 ballistic,
                     <1 confined -- the standard test for a random walk.
    net_turn_per100  |sum of signed turns| / 2pi per 100 steps. A circle or a
                     perimeter orbit accumulates rotation with a consistent
                     sign; a random walk's turns cancel.
    turn_bias        mean signed turn, i.e. handedness. Near 0 unless circling.
    straight_run     mean length of runs with |turn| < 30 degrees. A zipper's
                     runs are as long as the arena; a random walk's are ~1.
    reversal_rate    fraction of steps turning more than 150 degrees. A zipper
                     reverses once per lane; nothing else does it regularly.
    perimeter_frac   fraction of steps within 1.5 cells of a wall. Uniform
                     coverage of a 20x20 arena gives ~0.28.
    revisit_rate     fraction of steps landing on an already-visited cell.

The same features are computed for every reference family **through the same
simulator**, at the policy's own stride and turn width, so the comparison is
like-for-like rather than against an analytic ideal that ignores the clipping
walls. The policy is then reported next to the family it most resembles, in
standardized-feature distance, with the per-feature table shown so the match
can be judged rather than trusted.

    python -m hopfield_nav.probes.motion_pattern --ckpt <navigate_uN.pt>
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
from ..training.world_setup import build_field, replay_env_seeds
from ..world.env import make_env
from ..world.walks import REFERENCE_FAMILIES, family_positions
from ..world.world import build_world

FEATURES = ("msd_slope", "net_turn_per100", "turn_bias", "straight_run",
            "reversal_rate", "perimeter_frac", "revisit_rate")


def _signed_turns(pos: np.ndarray) -> np.ndarray:
    """(B, T-2) signed heading changes, in radians, wrapped to (-pi, pi]."""
    d = np.diff(pos, axis=1)
    ang = np.arctan2(d[:, :, 1], d[:, :, 0])
    raw = np.diff(ang, axis=1)
    return (raw + np.pi) % (2 * np.pi) - np.pi


def features(pos: np.ndarray, size: int) -> dict[str, float]:
    """Trajectory-shape features for `(B, T+1, 2)` positions."""
    B, T1, _ = pos.shape
    T = T1 - 1

    # Mean squared displacement against lag, on a log-log fit. Lags are capped
    # at T/4 so the estimate is not dominated by the arena's own size, which
    # bends every family toward a slope of 0 once they saturate it.
    lags = np.unique(np.clip(
        np.round(np.geomspace(1, max(T // 4, 2), 12)).astype(int), 1, T))
    msd = np.array([float(np.mean(np.sum(
        (pos[:, lag:] - pos[:, :-lag]) ** 2, axis=2))) for lag in lags])
    ok = msd > 1e-12
    slope = (float(np.polyfit(np.log(lags[ok]), np.log(msd[ok]), 1)[0])
             if ok.sum() >= 2 else 0.0)

    turns = _signed_turns(pos)
    # A step that did not move has an undefined heading; arctan2(0,0) is 0 and
    # would read as "went straight", which is exactly wrong for a policy pinned
    # against a wall.
    moved = np.linalg.norm(np.diff(pos, axis=1), axis=2) > 1e-9
    valid = moved[:, 1:] & moved[:, :-1]
    tv = np.where(valid, turns, np.nan)

    with np.errstate(invalid="ignore"):
        net = np.abs(np.nansum(tv, axis=1)) / (2 * np.pi) / max(T / 100.0, 1e-9)
        bias = np.nanmean(tv)
        straight = np.abs(tv) < np.deg2rad(30)
        reversal = np.abs(tv) > np.deg2rad(150)

    # Mean straight-run length: total straight steps / number of runs.
    runs = []
    for b in range(B):
        s = straight[b]
        s = s[~np.isnan(tv[b])] if np.isnan(tv[b]).any() else s
        if s.size == 0:
            continue
        starts = int(np.sum(s[1:] & ~s[:-1])) + int(s[0])
        runs.append(float(s.sum()) / max(starts, 1))

    edge = 1.5
    near = ((pos[:, :, 0] < edge) | (pos[:, :, 0] > size - 1 - edge)
            | (pos[:, :, 1] < edge) | (pos[:, :, 1] > size - 1 - edge))

    snapped = np.clip(np.rint(pos).astype(int), 0, size - 1)
    revisits = []
    for b in range(B):
        seen: set[tuple[int, int]] = set()
        hit = 0
        for x, y in snapped[b]:
            key = (int(x), int(y))
            if key in seen:
                hit += 1
            seen.add(key)
        revisits.append(hit / float(T1))

    return {
        "msd_slope": slope,
        "net_turn_per100": float(np.nanmean(net)),
        "turn_bias": float(bias),
        "straight_run": float(np.mean(runs)) if runs else 0.0,
        "reversal_rate": float(np.nanmean(reversal)),
        "perimeter_frac": float(near.mean()),
        "revisit_rate": float(np.mean(revisits)),
    }


def classify(policy: dict[str, float], refs: dict[str, dict[str, float]]
             ) -> list[tuple[str, float]]:
    """Rank reference families by standardized distance to the policy.

    Standardized across the reference set, so a feature that barely varies
    between families does not dominate the ranking by accident.

    `straight_run` is compared in logs. It spans 1.2 to 399 across the
    families -- a smooth circle never turns more than 30 degrees in a step, so
    its whole trajectory is one "run" -- and on a linear scale that single
    feature would decide every ranking on its own.
    """
    def vec(d: dict[str, float]) -> np.ndarray:
        return np.array([np.log10(max(d[f], 1e-3)) if f == "straight_run"
                         else d[f] for f in FEATURES], dtype=float)

    ref_mat = np.stack([vec(r) for r in refs.values()])
    scale = ref_mat.std(axis=0)
    scale[scale == 0] = 1.0
    pv = vec(policy)
    return sorted(
        ((name, float(np.sqrt((((pv - vec(r)) / scale) ** 2).sum())))
         for name, r in refs.items()),
        key=lambda kv: kv[1])


class PosRecorder:
    """Collects the continuous trajectory from the evaluator's step hook."""

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []

    def __call__(self, step, pos_before, actions, pos_after) -> None:
        if step == 0:
            self.rows.append(pos_before.copy())
        self.rows.append(pos_after.copy())

    def stack(self) -> np.ndarray:
        return np.stack(self.rows, axis=1)          # (B, T+1, 2)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--n_dist", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="val", choices=("val", "train", "fresh"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda"
                          and torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: TrainConfig = cfg_from_checkpoint(ck["config"])
    cfg.device = str(device)
    if "egocentric_heading" not in ck["config"].get("env", {}):
        cfg.env.egocentric_heading = False
    encoder, _e, gain = load_encoder(cfg.encoder_checkpoint, str(device),
                                     cfg.encoder_gain)
    cfg.encoder_gain = gain
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)

    field = build_field(cfg, encoder)
    np.random.seed(args.seed)
    seeds = replay_env_seeds(cfg, args.split, args.n_envs, args.seed)
    envs = [make_env(cfg.env, cfg.agent.movement_mode, seed=s) for s in seeds]
    world = build_world(field, envs, placement="spread", size=cfg.env.size)
    embed_dim = field.encoded_Phi.shape[2]

    agent = NavAgent(cfg.agent, compute_input_dim(
        cfg.agent, embed_dim, cfg.env.observation_size)).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()

    rng = np.random.RandomState(args.seed)
    size = int(cfg.env.size)
    tracks, covs = [], []
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
        rec = PosRecorder()
        visited, _f, _s = batched_exploration_trials(
            agent=agent, env=env, env_offset=offset, vectorhash=field,
            hopfields=hops, cfg=cfg, device=device, starts=starts,
            max_steps=args.max_steps, deterministic=True, on_step=rec)
        tracks.append(rec.stack())
        covs.extend(len(v) / float(size * size) for v in visited)

    pos = np.concatenate(tracks, axis=0)
    pol = features(pos, size)
    # Reference families at the policy's own movement statistics, so the
    # comparison is about shape rather than speed.
    step = np.linalg.norm(np.diff(pos, axis=1), axis=2)
    stride = float(step[step > 1e-9].mean()) if (step > 1e-9).any() else 1.0
    turns = _signed_turns(pos)
    turn_sigma = float(np.sqrt(max(
        -2.0 * np.log(max(float(np.cos(turns).mean()), 1e-6)), 0.0)))

    refs = {}
    for name in REFERENCE_FAMILIES:
        rp = family_positions(name, 256, size, args.max_steps, stride,
                              turn_sigma, np.random.RandomState(args.seed))
        refs[name] = features(rp, size)

    ranked = classify(pol, refs)
    out = {"ckpt": args.ckpt, "coverage": float(np.mean(covs)),
           "stride": stride, "turn_sigma": turn_sigma,
           "policy": pol, "references": refs,
           "ranking": [{"family": n, "distance": d} for n, d in ranked]}

    hdr = f"{'':<14}" + "".join(f"{f[:12]:>13}" for f in FEATURES)
    print(f"\nstride {stride:.2f}, turn_sigma {turn_sigma:.2f}, "
          f"coverage {np.mean(covs):.3f}\n")
    print(hdr)
    print(f"{'POLICY':<14}" + "".join(f"{pol[f]:>13.3f}" for f in FEATURES))
    for name, _d in ranked:
        print(f"{name:<14}" + "".join(f"{refs[name][f]:>13.3f}"
                                      for f in FEATURES))
    print("\nnearest reference families (standardized distance):")
    for name, d in ranked[:3]:
        print(f"   {name:<12} {d:.2f}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
