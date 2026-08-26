"""Where does the trained explore policy sit on P2's persistence axis?

§6.5 of `docs/EXPERIMENTS_NAV_P2.md` measures how much two sensory cones can
say about the displacement between them, and finds the answer depends heavily
on how straight the agent walks: the cones' contribution rises from +0.027 of
R^2 under uniform random turns to +0.173 when the per-step turn sd is 10 deg,
because the 120 deg aperture means two consecutive views only share a field of
view when the heading barely changed. That sweep was run over *hypothetical*
walks. It left the one number that decides how much of §6.5 matters unmeasured:
**what the trained policy's turn distribution actually is.**

Two knobs pull it in opposite directions and were never weighed against each
other — `PERSISTENCE_BONUS=0.05` pays for going straight, `EPSILON_EXPLORE=0.4`
injects uniform-random actions — so this is not answerable by reading the
launcher.

So: roll out a phase-2 explore policy, take the *realized* displacements, and
report exactly the quantities §6.5's table is indexed by — median |dpsi|
between consecutive steps, median shared cone rays, and the fraction of
consecutive view pairs that are completely disjoint. The output is one row to
drop into that table.

Reuses `analysis.nav_tri.behavior_probe.rollout` rather than reimplementing
the channel assembly: it already mirrors `evaluation/batched.py` step for step,
and a second rollout path is exactly how the train/eval mismatches this project
has already paid for get introduced.

    python -m analysis.nav_p2.policy_turn_stats --ckpt <phase-2 explore ckpt>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from analysis.nav_p2.displacement_decodability import FOVEAL_HALF_ANGLE_DEG
from analysis.nav_tri.behavior_probe import rollout
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, load_agent,
)
from hopfield import Hopfield
from hopfield_nav.rollout.distractors import sample_distractors
from hopfield_nav.evaluation.metrics import random_start


def turn_stats(pos_f, *, obs_size, lags=(1, 2, 4, 8)):
    """|dpsi| and cone overlap from a (T, B, 2) continuous-position trace.

    Heading is `atan2` of the realized displacement (`world/vec_env.py:461`),
    so it is recoverable from positions alone -- which is why this needs no
    extra recording hook. Steps that did not move leave the heading unchanged,
    exactly as the env does; they are carried forward rather than dropped,
    because a clipped step against a wall is a real event and pretending it did
    not happen would bias the turn distribution toward the smooth interior.
    """
    d = np.diff(pos_f, axis=0)                       # (T-1, B, 2)
    moved = np.linalg.norm(d, axis=-1) >= 1e-12
    psi = np.zeros(d.shape[:2])
    cur = np.zeros(d.shape[1])
    for t in range(len(d)):
        cur = np.where(moved[t], np.arctan2(d[t, :, 0], d[t, :, 1]), cur)
        psi[t] = cur
    step = np.deg2rad(2 * FOVEAL_HALF_ANGLE_DEG) / obs_size
    out = {}
    for L in lags:
        if L >= len(psi):
            continue
        dp = np.abs((psi[L:] - psi[:-L] + np.pi) % (2 * np.pi) - np.pi).ravel()
        ov = np.clip(obs_size - np.rint(dp / step), 0, obs_size)
        out[int(L)] = {
            "dpsi_med_deg": float(np.degrees(np.median(dp))),
            "dpsi_p90_deg": float(np.degrees(np.percentile(dp, 90))),
            "overlap_rays_med": float(np.median(ov)),
            "frac_zero_overlap": float((ov == 0).mean()),
            "step_norm_med": float(np.median(
                np.linalg.norm(d, axis=-1)[moved])),
            "frac_still": float(1.0 - moved.mean()),
            "n": int(dp.size),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--envs", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    if args.envs is not None:
        cfg.num_val_envs = args.envs
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device),
                                         ckpt_path=args.ckpt)
    agent = load_agent(cfg, ck["agent_state_dict"], embed_dim, device)

    print(f"{args.ckpt}")
    print(f"envs {len(envs)} x {args.trials} trials x {args.max_steps} steps, "
          f"explore regime, {args.n_distractors} distractors")
    print(f"|a| in [{cfg.env.min_action_norm}, {cfg.env.max_action_norm}], "
          f"persistence_bonus {cfg.hopfield.persistence_bonus}, "
          f"obs_size {cfg.env.observation_size}\n")

    rng = np.random.RandomState(args.seed)
    traces = []
    for i, env in enumerate(envs):
        hops, starts = [], []
        for _ in range(args.trials):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                           device=str(device))
            for pat in sample_distractors(vh, offsets[i], env.size,
                                          args.n_distractors, rng):
                hop.input_memory(torch.from_numpy(pat).float())
            hops.append(hop)
            starts.append(random_start(env.size, env.goal_location, rng))
        rec = rollout(agent=agent, env=env, env_offset=offsets[i],
                       vectorhash=vh, hopfields=hops, cfg=cfg, device=device,
                       starts=starts, max_steps=args.max_steps,
                       ends_on_arrival=False, goal_in_memory=False)
        traces.append(rec["pos_f"])

    pos = np.concatenate(traces, axis=1)
    stats = turn_stats(pos, obs_size=cfg.env.observation_size)
    print("Drop this row into EXPERIMENTS_NAV_P2 §6.5's table:\n")
    print(f"  {'lag':<22s}" + "".join(f"{L:>10d}" for L in stats))
    print(f"  {'median |dpsi| deg':<22s}"
          + "".join(f"{v['dpsi_med_deg']:>10.1f}" for v in stats.values()))
    print(f"  {'p90 |dpsi| deg':<22s}"
          + "".join(f"{v['dpsi_p90_deg']:>10.1f}" for v in stats.values()))
    print(f"  {'median shared rays':<22s}"
          + "".join(f"{v['overlap_rays_med']:>10.0f}" for v in stats.values()))
    print(f"  {'frac DISJOINT views':<22s}"
          + "".join(f"{v['frac_zero_overlap']:>9.1%}" for v in stats.values()))
    s1 = stats[1]
    print(f"\n  median step norm {s1['step_norm_med']:.2f}, "
          f"{s1['frac_still']:.1%} of steps did not move, "
          f"n = {s1['n']} consecutive pairs")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"ckpt": args.ckpt, "stats": stats}, fh, indent=2,
                      default=float)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
