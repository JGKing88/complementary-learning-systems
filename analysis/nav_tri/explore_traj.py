"""Example explore trajectories, matched across checkpoints.

The exploit page (`traj_page.py`) selects on SUCCESS, which explore does not
have -- an explore episode always runs the full horizon. The selection rule
here is COVERAGE, and the layout is a MATCHED PAIR: every checkpoint is rolled
from the identical start with the identical Hopfield contents, so two paths
side by side differ only by the policy.

That matters because §18.4's finding is a spatial one. `p20_e_kcap` reaches
`edge_frac` 0.061 against `p20_e`'s 0.127 -- it under-visits the perimeter --
and a matched pair is the only view in which that is attributable rather than
merely visible. Per `feedback_no_squinting` this page ILLUSTRATES a measured
result; it is not how the result was obtained.

The visited-cell footprint is drawn as well as the path, because coverage is
the metric and a path alone does not show retracing.

  python -m analysis.nav_tri.explore_traj \
      --ckpt A.pt B.pt --labels p20_e p20_e_kcap \
      --envs 3 --trials 12 --n_distractors 0 \
      --json out.json
"""

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_env_set, load_agent,
)
from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.rollout.distractors import sample_distractors
from hopfield_nav.world import generate as gen

from analysis.nav_tri.behavior_probe import rollout


TWO_PI = 2.0 * np.pi
# A single window size is a single loop radius: at speed ~0.96 a circle of
# radius 2 closes in 13 steps and one of radius 6 in 39, so a fixed 40-step
# window cancels the tight ones. Scan scales instead.
LOOP_SCALES = (12, 16, 20, 25, 30, 40, 50, 60, 80)


def loop_stats(path, *, radius=1.0, lag=15):
    """Does this trajectory loop? Two detectors, because they disagree.

    ``return_events``  the path comes back within ``radius`` of somewhere it
                       was >= ``lag`` steps ago. This is what the eye calls a
                       loop, and it is the one to trust.
    ``max_turn`` /     |sum dtheta| inside a window, over many window sizes.
    ``has_full_loop``  Catches a rotation -- but ALSO catches wall bounces,
                       which are large instantaneous turns: a billiard null
                       scores 59% on it. Read it as "rotational character",
                       never on its own as evidence of looping.
    ``sign_flips``     how often the handedness changes. A persistent circler
                       never flips; a path that hairpins back flips often.

    Both are computed from the realized path, which is what gets drawn.
    """
    p = np.asarray(path, dtype=float)
    T = len(p)
    d = p[1:] - p[:-1]
    n = np.linalg.norm(d, axis=-1)
    d = d[n > 1e-6]

    max_turn, n_loops, flips = 0.0, 0, 0
    if len(d) >= 3:
        ang = np.arctan2(d[:, 1], d[:, 0])
        dth = np.arctan2(np.sin(np.diff(ang)), np.cos(np.diff(ang)))
        c = np.concatenate([[0.0], np.cumsum(dth)])
        hit = np.zeros(len(dth), dtype=bool)
        for W in LOOP_SCALES:
            if len(dth) < W:
                continue
            roll = c[W:] - c[:-W]
            max_turn = max(max_turn, float(np.abs(roll).max()))
            for i in np.nonzero(np.abs(roll) >= TWO_PI)[0]:
                hit[i:i + W] = True
            if W == 40:
                s = np.sign(roll)
                s = s[s != 0]
                flips = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0
        if hit.any():
            e = np.diff(np.concatenate([[0], hit.astype(int), [0]]))
            n_loops = int((e == 1).sum())

    close = np.zeros(T, dtype=bool)
    for t in range(lag, T):
        close[t] = bool(
            (np.linalg.norm(p[:t - lag + 1] - p[t], axis=-1) <= radius).any())
    e = np.diff(np.concatenate([[0], close.astype(int), [0]]))
    return {
        "max_turn": float(max_turn),
        "has_full_loop": bool(n_loops > 0),
        "sign_flips": int(flips),
        "return_events": int((e == 1).sum()),
        "near_past_frac": float(close.mean()),
    }


def billiard_path(size, T, speed, rng):
    """Specular reflection off the box -- the null for the loop detectors.

    "The path returns near where it was" is not on its own evidence of looping:
    in a 20x20 box, 200 steps of length ~1 re-cross by geometry alone. Only the
    EXCESS over this null is looping. `coverage_baselines` gives the same
    process as a coverage NUMBER; this returns the path, which is what the
    detectors need.
    """
    p = np.array([rng.uniform(0, size - 1), rng.uniform(0, size - 1)])
    th = rng.uniform(0, TWO_PI)
    v = np.array([np.cos(th), np.sin(th)]) * speed
    out = [p.copy()]
    for _ in range(T - 1):
        p = p + v
        for k in (0, 1):
            if p[k] < 0:
                p[k], v[k] = -p[k], -v[k]
            elif p[k] > size - 1:
                p[k], v[k] = 2 * (size - 1) - p[k], -v[k]
        out.append(p.copy())
    return np.array(out)


def _traj_stats(pos_f, cell, action, q, size):
    """Per-trajectory numbers, matching `behavior_probe`'s definitions.

    Every statistic here is the single-trajectory form of one the aggregate
    probe reports, computed the same way, so a number on this page and the
    same number in §18.4 mean the same thing. The pathology axes -- the ones
    that name an explore FAILURE rather than grade it -- are:

      edge_frac         perimeter orbit. Uniform is 76/400 = 0.19; far above
                        that is `project_hopfield_nav_perimeter_basin`.
      signed_turn_mean  a CIRCLER. A cosine is unsigned, so straightness
                        cannot distinguish a constant-rate circle from an
                        unbiased walk; a large one-signed mean turn can.
      revisit_frac      retracing -- steps landing on an already-seen cell.
      clip_frac         steps the boundary or the norm clamp shortened. Read
                        WITH speed: a policy sitting at the speed cap reads
                        high here with no wall involved.
      chase_q           cos(a, q). Chasing a phantom recall.
    """
    T = len(cell)
    flat = cell[:, 0] * size + cell[:, 1]
    xs, ys = cell[:, 0], cell[:, 1]
    edge = (xs == 0) | (xs == size - 1) | (ys == 0) | (ys == size - 1)

    mag = np.linalg.norm(action, axis=-1)
    a1, a0 = action[1:], action[:-1]
    n1, n0 = mag[1:], mag[:-1]
    ok = (n1 > 1e-6) & (n0 > 1e-6)
    cos = np.sum(a1 * a0, axis=-1) / np.maximum(n1 * n0, 1e-12)

    ang = np.arctan2(action[:, 1], action[:, 0])
    dth = np.arctan2(np.sin(ang[1:] - ang[:-1]), np.cos(ang[1:] - ang[:-1]))

    realized = np.linalg.norm(pos_f[1:] - pos_f[:-1], axis=-1)
    clipped = realized < 0.9 * np.maximum(mag[:-1], 1e-8)

    seen, revisit = set(), np.zeros(T, dtype=bool)
    for t in range(T):
        k = int(flat[t])
        revisit[t] = k in seen
        seen.add(k)

    qn = np.linalg.norm(q, axis=-1)
    okq = (mag > 1e-6) & (qn > 1e-6)
    chase = np.sum(action * q, axis=-1) / np.maximum(mag * qn, 1e-12)

    uniq = np.unique(flat)
    return {
        "coverage": float(len(uniq)) / float(size * size),
        "edge_frac": float(edge.mean()),
        "straightness": float(cos[ok].mean()) if ok.any() else float("nan"),
        "signed_turn_mean": float(dth[ok].mean()) if ok.any() else 0.0,
        "abs_turn_mean": float(np.abs(dth[ok]).mean()) if ok.any() else 0.0,
        "revisit_frac": float(revisit.mean()),
        "clip_frac": float(clipped.mean()),
        "chase_q": float(chase[okq].mean()) if okq.any() else 0.0,
        "speed": float(realized.mean()),
        # unique cells, for the footprint
        "cells": [[int(c // size), int(c % size)] for c in uniq],
    }


def collect(args):
    device = torch.device(args.device)
    cks = [torch.load(c, map_location="cpu", weights_only=False)
           for c in args.ckpt]
    cfg = cfg_from_checkpoint(cks[0]["config"])
    cfg.num_val_envs = args.envs

    # Same guard the probe uses: a shared world is assumed by the matched-pair
    # layout, so anything that would change it has to agree.
    for path, other in zip(args.ckpt[1:], cks[1:]):
        o = cfg_from_checkpoint(other["config"])
        bad = [k for k in ("encoder_checkpoint", "fwhm_ratio")
               if getattr(o, k) != getattr(cfg, k)]
        if bad:
            raise SystemExit(f"{path} does not share a world: {bad} differ.")

    encoder, enc_cfg, gain = load_encoder(
        cfg.encoder_checkpoint, str(device), getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)

    es = eval_env_set(cfg, encoder, str(device), ckpt_path=args.ckpt[0],
                      levels=gen.parse_levels(args.split),
                      val_seed=args.val_seed, n_envs=cfg.num_val_envs)
    envs, vh, offsets = es["envs"], es["field"], es["offsets"]

    # Each agent is built from ITS OWN config, not from cks[0]'s.
    #
    # This was a bug, and a quiet one. Building every agent from the first
    # checkpoint's config gives the others the FIRST run's architecture knobs
    # while loading their own weights -- so `p20_e_kcap`, trained under
    # log_kappa_max=2.5, ran here under `p20_e`'s 5.0 and emitted the kappa its
    # training clamp had been suppressing (measured 45 against a 12.2 ceiling).
    # The WORLD still comes from `cfg`, which is shared and guarded above; only
    # the agent-side knobs are taken per checkpoint.
    #
    # `behavior_probe` has the same shape of guard for world keys and none for
    # agent keys, which is what let this through in both places.
    agents, agent_cfgs = {}, {}
    for lab, ck in zip(args.labels, cks):
        own = cfg_from_checkpoint(ck["config"])
        if own.hopfield.beta is None:
            own.hopfield.beta = float(gain)
        agent_cfgs[lab] = own
        agents[lab] = load_agent(own, ck["agent_state_dict"], embed_dim, device)

    # Surface any agent knob that differs rather than silently handling it: a
    # difference here is a real experimental fact about the comparison.
    base = agent_cfgs[args.labels[0]].agent
    for lab in args.labels[1:]:
        diff = [k for k in vars(base)
                if getattr(agent_cfgs[lab].agent, k, None) != getattr(base, k)]
        if diff:
            print("  NOTE %s differs from %s on agent knobs: %s"
                  % (lab, args.labels[0], ", ".join(sorted(diff))))

    size = cfg.env.size
    trials = []
    for i, env in enumerate(envs):
        off = offsets[i]
        # Drawn ONCE and reused for every checkpoint -- this is what makes the
        # pair matched. Distractors too: explore stores no goal, so the memory
        # holds distractors only and both policies must see the same ones.
        rng = np.random.RandomState(args.seed + i)
        starts, mem = [], []
        for _ in range(args.trials):
            starts.append(random_start(env.size, env.goal_location, rng))
            mem.append(sample_distractors(vh, off, env.size,
                                          args.n_distractors, rng)
                       if args.n_distractors > 0 else [])

        per_ckpt = {}
        for lab, agent in agents.items():
            hops = []
            for pats in mem:
                hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                               device=str(device))
                for pat in pats:
                    hop.input_memory(torch.from_numpy(pat).float())
                hops.append(hop)
    # ...and the ROLLOUT gets that same per-checkpoint config, not the shared
    # one. Building the agent from `own` while handing `rollout` `cfg` fixes
    # only half the bug: `rollout` assembles the policy input from
    # `cfg.agent`, so an arm with a different CHANNEL SET gets the first
    # checkpoint's channels. Comparing p20_e (74 inputs) against p26_abspos
    # (76) would build a 74-wide input for a 76-input agent. The world still
    # comes from the guarded keys, which are identical by construction.
            rec = rollout(
                agent=agent, env=env, env_offset=off, vectorhash=vh,
                hopfields=hops, cfg=agent_cfgs[lab], device=device,
                starts=starts,
                max_steps=args.max_steps,
                ends_on_arrival=False, goal_in_memory=False,
                deterministic=args.deterministic)
            per_ckpt[lab] = rec

        for b in range(args.trials):
            row = {"env": i, "trial": b,
                   "start": [float(x) for x in per_ckpt[args.labels[0]]["pos_f"][0, b]],
                   "by_ckpt": {}}
            for lab in args.labels:
                rec = per_ckpt[lab]
                st = _traj_stats(rec["pos_f"][:, b], rec["cell"][:, b],
                                 rec["action"][:, b], rec["q"][:, b], size)
                # Per-step policy state, for the question of whether a
                # state-dependent kappa is actually USED to switch between a
                # ballistic and a tortuous mode (intermittent search) or just
                # settles on one scale.
                # `in rec` is NOT the right test and cost a job: the probe
                # initialises these keys unconditionally and only FILLS them
                # under a polar head, so a Cartesian checkpoint leaves an
                # EMPTY list -- present, but 1-D -- and a two-index slice of
                # it raises IndexError. Every phase-1 model is Cartesian, so
                # this made explore_traj unable to score any of them.
                # `behavior_probe` guards the same arrays on `.size`; match it.
                for key, prec in (("circ_sd", 5), ("mu_norm", 4)):
                    arr = np.asarray(rec.get(key, []))
                    if arr.ndim == 2 and arr.size:
                        st[key] = [round(float(v), prec) for v in arr[:, b]]
                st["path"] = [[round(float(p[0]), 3), round(float(p[1]), 3)]
                              for p in rec["pos_f"][:, b]]
                row["by_ckpt"][lab] = st
            trials.append(row)
        print(f"env {i}: {args.trials} trials x {len(agents)} ckpts")

    return {"size": size, "max_steps": args.max_steps,
            "n_distractors": args.n_distractors, "split": args.split,
            "labels": list(args.labels), "ckpts": list(args.ckpt),
            "trials": trials}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, nargs="+")
    p.add_argument("--labels", required=True, nargs="+")
    p.add_argument("--envs", type=int, default=3)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--split", default="recorded")
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Act on the distribution MEAN (default, and what every "
                        "P2 number was measured under) or SAMPLE. Sampling is "
                        "the control for the replay finding: noise perturbs "
                        "the state and should break a closed orbit.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", required=True)
    args = p.parse_args()
    if len(args.labels) != len(args.ckpt):
        raise SystemExit("--labels must have one entry per --ckpt")

    out = collect(args)
    with open(args.json, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.json}  ({len(out['trials'])} matched trials)")


if __name__ == "__main__":
    main()
