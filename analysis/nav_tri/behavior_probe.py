"""How is a checkpoint getting its score? Quantitative, not by eye.

`evaluate_exploration` returns a coverage number and `evaluate_navigation`
returns a success rate; neither says what the agent *did*. This module runs the
same two protocols and records the whole trajectory -- continuous position,
executed action, the Hopfield's projected recall `q`, at-goal -- then reduces
them to statistics that name a behaviour class and a failure mode.

Deliberately not a plotting tool. Trajectory PNGs are how this project has gone
wrong before (memory `feedback_no_squinting`): a picture of eight rollouts is a
sample of eight, and every reader sees a different pattern in it. Everything
below is an aggregate over hundreds of trials with a reference value attached.

Explore diagnostics answer "what behaviour class is this?":

  cells_per_step    -- against the reference ladder in
                       `analysis/nav_tri/coverage_baselines.py`:
                       0.36 = uniform random walk, 0.775 = billiard,
                       0.955 = lawnmower, 1.0 = ceiling.
  straightness      -- mean cos(a_t, a_{t-1}). A billiard is ~1.
  edge_frac         -- share of steps on a perimeter cell. Uniform coverage
                       gives 76/400 = 0.19; much above that is the
                       perimeter-orbit basin (memory
                       `project_hopfield_nav_perimeter_basin`).
  clip_frac         -- share of steps whose realized displacement was shorter
                       than the action asked for, i.e. absorbed by the
                       boundary clip. This is the corner-trap signature, and it
                       is invisible in every metric the trainer logs.
  chase_q           -- mean cos(a_t, q_t) with distractors in memory and no
                       goal. Zero means the policy ignores the recall channel,
                       which under `explore_goals_off` is what it should do.

Nav diagnostics separate readout error from policy error, which is the
distinction that decides whether a nav failure is the Hopfield's fault or the
policy's:

  q_accuracy        -- mean cos(q_t, goal - pos_t). Is the RECALL SIGNAL right?
  follow_q          -- mean cos(a_t, q_t). Does the policy follow it?
  align_true        -- mean cos(a_t, goal - pos_t). Does the agent move goalward?

A failure with high `q_accuracy` and low `follow_q` is a policy that will not
follow a correct signal; low `q_accuracy` is a recall/distractor problem. The
two imply opposite fixes.

Usage:
    python -m analysis.nav_tri.behavior_probe \
        --ckpt $CLS_RUNS/agent_ckpts/navtri_.../navigate_u600.pt \
        --mode explore nav --n_distractors 0 10 --trials 32 --max_steps 200 \
        --json out.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, load_agent,
)
from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.policy import channels
from hopfield_nav.rollout import signal as signal_ops
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world import episode
from hopfield_nav.world.env import at_goal
from hopfield_nav.world.vec_env import make_vec


# ---------------------------------------------------------------------------
# The instrumented rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def _rollout(*, agent, env, env_offset, vectorhash, hopfields, cfg, device,
             starts, max_steps, ends_on_arrival, goal_in_memory):
    """One trial per Hopfield, in parallel, recording everything.

    Mirrors `evaluation/batched.py` step for step -- same channel assembly,
    same contract, same freeze-on-arrival -- and additionally keeps the arrays
    the evaluators throw away. Kept as a separate function rather than a flag
    on the evaluators because an evaluator that recorded trajectories would pay
    for it on every training eval.
    """
    B = len(hopfields)
    contract = episode.contract_for(
        "evaluate_navigation" if ends_on_arrival else "evaluate_exploration")
    vec = make_vec(env, B, cfg.agent.movement_mode, cfg.env.continuous_scale,
                   continuous_normalize=cfg.env.continuous_normalize,
                   reset=False)
    vec.set_positions(starts)

    input_specs = channels.channel_specs(
        cfg.agent, vectorhash.encoded_Phi.shape[2], cfg.env.observation_size)
    signal_dim = channels.signal_width(cfg.agent)
    prev_action_dim = channels.prev_action_width(cfg.agent)

    h_rnn = None
    prev_reward_t = torch.zeros(B, 1, device=device)
    prev_action_t = torch.zeros(B, prev_action_dim, device=device)
    steps_to_goal = np.full(B, -1, dtype=np.int64)
    active = np.ones(B, dtype=bool)

    rec = {k: [] for k in ("pos_f", "cell", "action", "q", "at_goal", "alive")}

    for step in range(max_steps):
        positions = vec.positions()
        pos_f = vec.positions_continuous()
        at_g = at_goal(vec)
        current_reward = (
            np.where(at_g, cfg.env.goal_reward, -cfg.env.time_penalty)
            if vec.goals_active
            else np.full(B, -cfg.env.time_penalty, dtype=np.float32)
        ).astype(np.float32)

        embeddings_np = vectorhash.get_encoded_state(positions, env_offset)
        embeddings = torch.from_numpy(embeddings_np).float().to(device)

        q_np = np.zeros((B, 2), dtype=np.float32)
        if cfg.agent.input_hopfield_signal:
            sig_t, q, _mask, _W = signal_ops.hopfield_signal_at(
                vectorhash, cfg, embeddings_np, embeddings, positions,
                env_offset, hopfields, False, device, embeddings.shape[1])
            q_np = np.asarray(q, dtype=np.float32)
            if (cfg.agent.input_hopfield_raw
                    and cfg.agent.hopfield_mode != "discrete"):
                hop_signal = torch.from_numpy(q_np).to(device)
            else:
                hop_signal = sig_t
        else:
            hop_signal = torch.zeros(B, signal_dim, device=device)

        values = {
            "current_reward": torch.from_numpy(current_reward).to(device).unsqueeze(1),
            "prev_reward": prev_reward_t,
            "encoded_state": embeddings,
            "hopfield_signal": hop_signal,
            "prev_action": prev_action_t,
            "goal_in_memory": (torch.ones if goal_in_memory else torch.zeros)(
                B, 1, device=device),
        }
        if cfg.agent.input_sensory:
            values["sensory"] = torch.from_numpy(
                vec.obs_batch()).float().to(device)
        if (cfg.agent.input_hopfield_multistep
                and cfg.agent.hopfield_mode == "continuous"):
            W = vectorhash.gram_schmidt_projection(positions, env_offset)
            msq = signal_ops.multistep_q(
                vectorhash, cfg, embeddings_np, embeddings, hopfields, False,
                W, cfg.agent.input_hopfield_multistep, embeddings.shape[1],
                device)
            for s, q_s in msq.items():
                values[channels.multistep_name(s)] = torch.from_numpy(
                    np.asarray(q_s, dtype=np.float32)).to(device)

        rnn_input = channels.build_policy_input(
            input_specs, values, batch_size=B).unsqueeze(1)
        result = agent.get_action_and_value(rnn_input, h_rnn, deterministic=True)
        h_rnn = result["h_next"]
        actions = result["move_action"].cpu().numpy().reshape(B, -1)
        prev_action_t = result["move_action"].float().view(B, -1)
        prev_reward_t = torch.from_numpy(current_reward).to(device).unsqueeze(1)

        rec["pos_f"].append(pos_f.copy())
        rec["cell"].append(positions.copy())
        rec["action"].append(actions.copy())
        rec["q"].append(q_np.copy())
        rec["at_goal"].append(at_g.copy())
        rec["alive"].append(active.copy())

        idx = np.nonzero(active)[0] if ends_on_arrival else np.arange(B)
        if idx.size == 0:
            break
        vec.step_batch(actions[idx], indices=idx, contract=contract)

        if ends_on_arrival:
            reached = at_goal(vec)
            for b in idx:
                if reached[b]:
                    steps_to_goal[b] = step + 1
                    active[b] = False

    out = {k: np.asarray(v) for k, v in rec.items()}      # (T, B, ...)
    out["steps_to_goal"] = steps_to_goal
    out["final_pos_f"] = vec.positions_continuous()
    return out


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _cos(a, b, eps=1e-8):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    ok = (na > eps) & (nb > eps)
    c = np.zeros(na.shape, dtype=np.float64)
    np.divide((a * b).sum(-1), np.maximum(na * nb, eps), out=c, where=ok)
    return c, ok


def _explore_stats(rec, size, goal):
    """(T, B, ...) trajectory arrays -> the behaviour-class diagnostics."""
    cell = rec["cell"]                        # (T, B, 2)
    act = rec["action"]                       # (T, B, 2)
    pos_f = rec["pos_f"]                      # (T, B, 2)
    T, B = cell.shape[0], cell.shape[1]

    # Coverage, and the curve, from the same definition evaluate_exploration
    # uses: unique snapped cells / size**2.
    flat = cell[..., 0] * size + cell[..., 1]          # (T, B)
    covs, curve = [], {}
    for b in range(B):
        covs.append(len(np.unique(flat[:, b])) / float(size * size))
    for frac in (0.25, 0.5, 0.75, 1.0):
        t = max(1, int(round(frac * T)))
        curve[f"cov@{t}"] = float(np.mean(
            [len(np.unique(flat[:t, b])) for b in range(B)]) / (size * size))
    union = len(np.unique(flat)) / float(size * size)

    mag = np.linalg.norm(act, axis=-1)                  # (T, B)
    straight, ok_s = _cos(act[1:], act[:-1])

    xs, ys = cell[..., 0], cell[..., 1]
    edge = (xs == 0) | (xs == size - 1) | (ys == 0) | (ys == size - 1)

    # Realized displacement vs. requested: the boundary clip is the only thing
    # that can shorten it, so this is the corner-trap fraction.
    realized = np.linalg.norm(pos_f[1:] - pos_f[:-1], axis=-1)
    want = mag[:-1]
    clipped = realized < 0.9 * np.maximum(want, 1e-8)

    revisit = np.zeros((T, B), dtype=bool)
    for b in range(B):
        seen = set()
        for t in range(T):
            k = int(flat[t, b])
            revisit[t, b] = k in seen
            seen.add(k)

    chase, ok_c = _cos(act, rec["q"])
    qmag = np.linalg.norm(rec["q"], axis=-1)

    centre = (size - 1) / 2.0
    return {
        "mean_coverage": float(np.mean(covs)),
        "cells_per_step": float(np.mean(covs) * size * size / T),
        "union_coverage": float(union),
        **{k: float(v) for k, v in curve.items()},
        "step_mag_mean": float(mag.mean()),
        "step_mag_median": float(np.median(mag)),
        "step_mag_p10": float(np.percentile(mag, 10)),
        "step_mag_p90": float(np.percentile(mag, 90)),
        "straightness": float(straight[ok_s].mean()) if ok_s.any() else 0.0,
        "edge_frac": float(edge.mean()),
        "clip_frac": float(clipped.mean()),
        "revisit_frac": float(revisit.mean()),
        "mean_dist_from_centre": float(
            np.linalg.norm(pos_f - centre, axis=-1).mean()),
        "chase_q": float(chase[ok_c].mean()) if ok_c.any() else 0.0,
        "q_mag_mean": float(qmag.mean()),
        "q_present_frac": float((qmag > 1e-6).mean()),
    }


def _nav_stats(rec, size, goal, starts):
    cell = rec["cell"]
    act = rec["action"]
    pos_f = rec["pos_f"]
    alive = rec["alive"]                       # (T, B) bool
    T, B = cell.shape[0], cell.shape[1]
    stg = rec["steps_to_goal"]
    succ = stg > 0

    g = np.asarray(goal, dtype=np.float64)
    to_goal = g[None, None, :] - pos_f                     # (T, B, 2)
    align, ok_a = _cos(act, to_goal)
    follow, ok_f = _cos(act, rec["q"])
    qacc, ok_q = _cos(rec["q"], to_goal)
    mag = np.linalg.norm(act, axis=-1)

    live = alive.astype(bool)
    def m(arr, ok, mask):
        sel = ok & mask & live
        return float(arr[sel].mean()) if sel.any() else float("nan")

    succ_mask = np.broadcast_to(succ[None, :], (T, B))
    fail_mask = ~succ_mask

    start_d = np.linalg.norm(
        np.asarray(starts, dtype=np.float64) - g[None, :], axis=-1)
    final_d = np.linalg.norm(rec["final_pos_f"] - g[None, :], axis=-1)

    # Optimal steps for exactly the starts that succeeded, so path efficiency
    # is not confounded by successes being the nearby ones.
    opt = np.maximum(0.0, start_d - float(getattr(_nav_stats, "_radius", 1.0)))
    eff = (opt[succ] / np.maximum(stg[succ], 1)) if succ.any() else np.array([])

    return {
        "success_rate": float(succ.mean()),
        "mean_steps": float(stg[succ].mean()) if succ.any() else float("nan"),
        "median_steps": float(np.median(stg[succ])) if succ.any() else float("nan"),
        "mean_start_dist": float(start_d.mean()),
        "mean_start_dist_success": float(start_d[succ].mean()) if succ.any() else float("nan"),
        "mean_start_dist_fail": float(start_d[~succ].mean()) if (~succ).any() else float("nan"),
        "path_efficiency": float(eff.mean()) if eff.size else float("nan"),
        "step_mag_mean": float(mag[live].mean()),
        "align_true": m(align, ok_a, np.ones_like(succ_mask)),
        "align_true_success": m(align, ok_a, succ_mask),
        "align_true_fail": m(align, ok_a, fail_mask),
        "follow_q": m(follow, ok_f, np.ones_like(succ_mask)),
        "follow_q_fail": m(follow, ok_f, fail_mask),
        "q_accuracy": m(qacc, ok_q, np.ones_like(succ_mask)),
        "q_accuracy_fail": m(qacc, ok_q, fail_mask),
        "final_dist_fail": float(final_d[~succ].mean()) if (~succ).any() else float("nan"),
        "fail_frac_at_edge": float(
            (((cell[..., 0] == 0) | (cell[..., 0] == size - 1)
              | (cell[..., 1] == 0) | (cell[..., 1] == size - 1))
             & fail_mask & live).sum() / max((fail_mask & live).sum(), 1)),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, nargs="+",
                   help="One or more checkpoints. Several are probed in ONE "
                        "process so the scaffold is built once -- encoded_Phi "
                        "is 12 GB and takes ~15 min to build on CPU, which "
                        "otherwise dominates a multi-checkpoint comparison. "
                        "All of them must share a world (same encoder, "
                        "lambdas, Npos, fwhm, size, wall_resolution); that is "
                        "checked, not assumed.")
    p.add_argument("--mode", nargs="+", default=["explore", "nav"],
                   choices=["explore", "nav"])
    p.add_argument("--n_distractors", type=int, nargs="+", default=[0, 10])
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--envs", type=int, default=None,
                   help="val envs to use; default = the checkpoint's own count")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    p.add_argument("--npos", type=int, default=None,
                   help="Shrink the scaffold. encoded_Phi is Npos^2 x 1024, "
                        "i.e. 12 GB at the real 1716 -- unusable without a GPU "
                        "node. A small value exercises every code path on a "
                        "laptop-sized array. It changes the scaffold geometry, "
                        "so results under it are for VALIDATING THE TOOL, never "
                        "for reading off a number.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cks = [torch.load(c, map_location="cpu", weights_only=False)
           for c in args.ckpt]
    cfg = cfg_from_checkpoint(cks[0]["config"])
    if args.envs is not None:
        cfg.num_val_envs = args.envs
    # Before the --npos override, or the check would compare an overridden Npos
    # against the other checkpoints' real one and reject every pair.
    #
    # The world is shared across checkpoints, so anything that would change it
    # has to agree. Checked rather than assumed: silently scoring checkpoint B
    # against checkpoint A's world is the failure mode this whole shortcut
    # invites, and it produces plausible numbers.
    _WORLD_KEYS = ("encoder_checkpoint", "fwhm_ratio")
    for path, other in zip(args.ckpt[1:], cks[1:]):
        o = cfg_from_checkpoint(other["config"])
        mismatch = [k for k in _WORLD_KEYS if getattr(o, k) != getattr(cfg, k)]
        mismatch += [f"vectorhash.{k}" for k in ("lambdas", "Npos")
                     if getattr(o.vectorhash, k) != getattr(cfg.vectorhash, k)]
        mismatch += [f"env.{k}" for k in ("size", "wall_resolution", "goal_radius")
                     if getattr(o.env, k) != getattr(cfg.env, k)]
        if mismatch:
            raise SystemExit(
                f"{path} does not share a world with {args.ckpt[0]}: "
                f"{', '.join(mismatch)} differ. Probe them separately.")

    if args.npos is not None:
        print(f"  WARNING: --npos {args.npos} overrides the checkpoint's "
              f"scaffold. Tool-validation mode; numbers are not comparable.")
        cfg.vectorhash.Npos = args.npos

    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    # A shrunken scaffold cannot use the recorded world -- the offsets in
    # world.json index the real Npos and eval_field rejects the mismatch, which
    # is the right guard. Validation mode takes the RNG-replay branch instead.
    envs, vh, offsets = build_eval_world(
        cfg, encoder, str(device),
        ckpt_path=(None if args.npos is not None else args.ckpt[0]))
    _nav_stats._radius = cfg.env.goal_radius

    print(f"envs      : {len(envs)}  trials/env: {args.trials}  "
          f"steps: {args.max_steps}  goal_radius: {cfg.env.goal_radius}")
    print(f"trunk     : {cfg.agent.rnn_cell}/{cfg.agent.rnn_nonlinearity} "
          f"h={cfg.agent.hidden_size}")

    all_out = {}
    for path, ck in zip(args.ckpt, cks):
        agent = load_agent(cfg, ck["agent_state_dict"], embed_dim, device)
        print(f"\n================ {path} ================")
        all_out[path] = _probe_one(
            args, cfg, agent, envs, vh, offsets, embed_dim, device)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"max_steps": args.max_steps, "by_ckpt": all_out},
                      fh, indent=2)
        print(f"\nwrote {args.json}")


def _probe_one(args, cfg, agent, envs, vh, offsets, embed_dim, device):
    out: dict = {}
    for mode in args.mode:
        for n_d in args.n_distractors:
            rng = np.random.RandomState(args.seed)
            per_env = []
            for i, env in enumerate(envs):
                goal = env.goal_location
                off = offsets[i]
                hops, starts = [], []
                for _ in range(args.trials):
                    hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                   device=str(device))
                    pats = ([goal_encoding(vh, off, goal)] if mode == "nav"
                            else [])
                    if n_d > 0:
                        pats.extend(sample_distractors(vh, off, env.size,
                                                       n_d, rng))
                    if mode == "nav":
                        rng.shuffle(pats)
                    for pat in pats:
                        hop.input_memory(torch.from_numpy(pat).float())
                    hops.append(hop)
                    starts.append(random_start(env.size, goal, rng))
                rec = _rollout(
                    agent=agent, env=env, env_offset=off, vectorhash=vh,
                    hopfields=hops, cfg=cfg, device=device, starts=starts,
                    max_steps=args.max_steps,
                    ends_on_arrival=(mode == "nav"),
                    goal_in_memory=(mode == "nav"))
                per_env.append(
                    _nav_stats(rec, env.size, goal, starts) if mode == "nav"
                    else _explore_stats(rec, env.size, goal))
            agg = {k: float(np.nanmean([e[k] for e in per_env]))
                   for k in per_env[0]}
            out[f"{mode}_d{n_d}"] = agg
            print(f"\n--- {mode}  n_dist={n_d} ---")
            for k, v in agg.items():
                print(f"  {k:<26s} {v:.4f}")
    return out


if __name__ == "__main__":
    main()
