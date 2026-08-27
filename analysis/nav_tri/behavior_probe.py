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
from analysis.nav_tri.coverage_baselines import billiard_cells_per_step
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, eval_env_set, load_agent,
)
from hopfield_nav.world import generate as gen
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


# NOTE on passing several checkpoints at once: the world and the agent are
# built ONCE from the first checkpoint's config and each set of weights is
# loaded into it, which is what makes the 12 GB scaffold build worth amortizing.
# It also means every checkpoint in the list must share an ARCHITECTURE. Mixing
# a `state_dependent_std` run with a global-sigma one fails with
# "Missing key(s) movement_log_std_head.weight / Unexpected key(s)
# movement_log_std" -- run those as separate jobs rather than one list.


def _circular_sd_np(kappa):
    """Circular sd of a von Mises in RADIANS, ``sqrt(-2 ln(I1/I0))``.

    Duplicated from ``policy/polar_head.py`` in numpy rather than imported,
    because everything else in this module works on numpy arrays already and
    the alternative is a torch round-trip per step. Calibrated so a Cartesian
    arm's ``sigma/||mu||`` and a polar arm's kappa land in the same column:
    section 9.3's 10.56 degrees is kappa 29.4, which reads back as 10.66.
    """
    from scipy.special import i0e, i1e
    r_bar = np.clip(i1e(kappa) / i0e(kappa), 1e-7, 1.0 - 1e-7)
    return np.sqrt(-2.0 * np.log(r_bar))


@torch.no_grad()
def rollout(*, agent, env, env_offset, vectorhash, hopfields, cfg, device,
            starts, max_steps, ends_on_arrival, goal_in_memory,
            q_rescale=None, q_scale=None):
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
    prev_disp_t = torch.zeros(B, 2, device=device)
    steps_to_goal = np.full(B, -1, dtype=np.int64)
    active = np.ones(B, dtype=bool)

    rec = {k: [] for k in ("pos_f", "cell", "action", "q", "at_goal",
                           "alive", "sigma", "mu_norm", "circ_sd")}

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
            q_np = _rescale_q(q_np, q_rescale, q_scale).astype(np.float32)
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
            "prev_displacement": prev_disp_t,
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
                    _rescale_q(np.asarray(q_s, dtype=np.float32),
                               q_rescale, q_scale).astype(np.float32)).to(device)

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
        # The policy's own spread and commanded magnitude. Recorded here rather
        # than recomputed later because sigma under a state-dependent head is a
        # function of the exact inputs the policy saw -- reconstructing them
        # elsewhere is where a hand-rolled version silently differs.
        _ms = result.get("move_std")
        _mm = result.get("move_mean")
        _mk = result.get("move_kappa")
        if _ms is not None and _mm is not None:
            sd = _ms.cpu().numpy().reshape(B, -1)
            if _mk is None:
                rec["sigma"].append(sd.mean(-1))
            else:
                # Polar: stddev is (radial, tangential), so a mean over the two
                # would blend the speed spread with the directional one. Column
                # 0 is the speed sd, which is what `sigma` means in the polar
                # diag() convention and so stays comparable to the Cartesian
                # radial noise.
                rec["sigma"].append(sd[:, 0])
                rec["circ_sd"].append(
                    _circular_sd_np(_mk.cpu().numpy().reshape(B)))
            # Under polar this is the MEAN SPEED, which is exactly the
            # quantity ||mu|| stood for in the Cartesian arms.
            rec["mu_norm"].append(
                np.linalg.norm(_mm.cpu().numpy().reshape(B, -1), axis=-1))

        idx = np.nonzero(active)[0] if ends_on_arrival else np.arange(B)
        if idx.size == 0:
            break
        # After the step, so it is the displacement the env actually
        # produced -- not the action, which the norm clamp and the
        # arena clip both alter.
        vec.step_batch(actions[idx], indices=idx, contract=contract)
        prev_disp_t = torch.from_numpy(
            vec.last_displacement()).float().to(device)

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


@torch.no_grad()
def _warm_vs_cold(*, agent, env, env_offset, vectorhash, hopfields, cfg,
                  device, starts, max_steps):
    """Steps-to-goal for the FIRST reach vs every later reach in one rollout.

    Isolates the one asymmetry between how exploit is trained and how it is
    scored. Training rollouts teleport on arrival and, with
    `reset_state_on_teleport=False`, carry the RNN state across it -- so the
    agent's second and later goals are approached from a *warm* hidden state.
    Navigation eval always measures a *cold* start, and it measures only one
    goal. If the policy has learned to lean on the warm state, reaches 2+ will
    be much faster than reach 1 **for the same policy in the same rollout**,
    which no difference of env, seed or protocol can explain away.

    Uses the TRAINING contract, so this is the training distribution exactly.
    """
    B = len(hopfields)
    contract = episode.contract_for(
        "training_rollout", reset_state=cfg.env.reset_state_on_teleport)
    vec = make_vec(env, B, cfg.agent.movement_mode, cfg.env.continuous_scale,
                   continuous_normalize=cfg.env.continuous_normalize,
                   reset=False)
    vec.set_positions(starts)

    input_specs = channels.channel_specs(
        cfg.agent, vectorhash.encoded_Phi.shape[2], cfg.env.observation_size)
    prev_action_dim = channels.prev_action_width(cfg.agent)
    h_rnn = None
    prev_reward_t = torch.zeros(B, 1, device=device)
    prev_action_t = torch.zeros(B, prev_action_dim, device=device)
    prev_disp_t = torch.zeros(B, 2, device=device)

    last_reach = np.zeros(B, dtype=np.int64)      # step index of the last goal
    first_gap, later_gaps = [], []

    for step in range(max_steps):
        positions = vec.positions()
        at_g = at_goal(vec)
        current_reward = (
            np.where(at_g, cfg.env.goal_reward, -cfg.env.time_penalty)
            if vec.goals_active
            else np.full(B, -cfg.env.time_penalty, dtype=np.float32)
        ).astype(np.float32)
        embeddings_np = vectorhash.get_encoded_state(positions, env_offset)
        embeddings = torch.from_numpy(embeddings_np).float().to(device)

        sig_t, q, _m, _W = signal_ops.hopfield_signal_at(
            vectorhash, cfg, embeddings_np, embeddings, positions,
            env_offset, hopfields, False, device, embeddings.shape[1])
        hop_signal = (torch.from_numpy(np.asarray(q, np.float32)).to(device)
                      if cfg.agent.input_hopfield_raw else sig_t)

        values = {
            "current_reward": torch.from_numpy(current_reward).to(device).unsqueeze(1),
            "prev_reward": prev_reward_t,
            "encoded_state": embeddings,
            "hopfield_signal": hop_signal,
            "prev_action": prev_action_t,
            "prev_displacement": prev_disp_t,
            "goal_in_memory": torch.ones(B, 1, device=device),
        }
        if cfg.agent.input_sensory:
            values["sensory"] = torch.from_numpy(vec.obs_batch()).float().to(device)
        if cfg.agent.input_hopfield_multistep:
            W = vectorhash.gram_schmidt_projection(positions, env_offset)
            for s, q_s in signal_ops.multistep_q(
                    vectorhash, cfg, embeddings_np, embeddings, hopfields,
                    False, W, cfg.agent.input_hopfield_multistep,
                    embeddings.shape[1], device).items():
                values[channels.multistep_name(s)] = torch.from_numpy(
                    np.asarray(q_s, np.float32)).to(device)

        rnn_input = channels.build_policy_input(
            input_specs, values, batch_size=B).unsqueeze(1)
        result = agent.get_action_and_value(rnn_input, h_rnn, deterministic=True)
        h_rnn = result["h_next"]
        actions = result["move_action"].cpu().numpy().reshape(B, -1)
        prev_action_t = result["move_action"].float().view(B, -1)
        prev_reward_t = torch.from_numpy(current_reward).to(device).unsqueeze(1)

        # After the step, so it is the displacement the env actually
        # produced -- not the action, which the norm clamp and the
        # arena clip both alter.
        _r, reached, _p = vec.step_batch(actions, contract=contract)
        prev_disp_t = torch.from_numpy(
            vec.last_displacement()).float().to(device)
        for b in np.nonzero(reached)[0]:
            gap = step + 1 - last_reach[b]
            (first_gap if last_reach[b] == 0 else later_gaps).append(gap)
            last_reach[b] = step + 1

    return {
        "cold_first_reach_steps": float(np.mean(first_gap)) if first_gap else float("nan"),
        "warm_later_reach_steps": float(np.mean(later_gaps)) if later_gaps else float("nan"),
        "n_cold": len(first_gap), "n_warm": len(later_gaps),
        "warm_speedup": (float(np.mean(first_gap) / np.mean(later_gaps))
                         if first_gap and later_gaps else float("nan")),
    }


def _rescale_q(q, target, factor=None):
    """Set ||q|| to `target` (or multiply it by `factor`), preserving direction.

    **`factor` is the sound version and `target` has a confound.** Clamping
    ||q|| to a constant also destroys its variation WITHIN a trajectory -- and
    P0.8 showed that variation is the informative cue: approaching a real goal
    makes ||q|| shrink, approaching a phantom does not. So `target` moves the
    level and deletes the dynamics at the same time, and a behaviour change
    under it cannot be attributed to either. `factor` shifts the level while
    leaving the shape of ||q||(t) intact, which is the intervention the
    magnitude-gating hypothesis actually calls for.

    `target` is kept because the first crossover used it and its result is
    recorded; new work should use `factor`.

    The intervention behind the gating experiment. The hypothesis is that the
    policy decides whether to follow the recall by its MAGNITUDE -- goal-present
    ||q|| is ~0.22 and decoy-only ~0.17 at ten distractors, ~0.27 against ~0.06
    at one -- rather than by anything about the memory's contents. Correlational
    evidence cannot separate those, because in normal rollouts magnitude and
    contents move together.

    Rescaling breaks them apart: feed a DECOY direction at goal-strength, or a
    GOAL direction at decoy-strength, and see which one the behaviour follows.
    Direction is untouched, so `follow_q` / `chase_q` -- both cosines -- measure
    the same thing before and after.

    Applied to the multistep channels with the same factor, not independently:
    they are the same recall iterated, so scaling them apart would create an
    input combination the policy has never seen for a different reason than the
    one under test.
    """
    if factor is not None:
        return q * factor
    if target is None:
        return q
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > 1e-8, q * (target / np.maximum(n, 1e-8)), q)


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

    # Turn statistics, which name the behaviour class in a way `straightness`
    # cannot. A cosine is unsigned, so a policy that circles at a constant rate
    # and one that jitters symmetrically can score the same.
    #
    #   signed_turn_mean  ~0 for any unbiased walk; large and one-signed is a
    #                     CIRCLER, which covers an annulus and nothing else.
    #   abs_turn_mean     0 = ballistic, pi/2 = uniform random walk.
    #   run_len_mean      mean length of a run of near-straight steps. This is
    #                     1/p_turn for a run-and-tumble, and §3.1 puts the
    #                     coverage optimum at ~4 steps.
    ang = np.arctan2(act[..., 1], act[..., 0])
    dth = np.arctan2(np.sin(ang[1:] - ang[:-1]), np.cos(ang[1:] - ang[:-1]))
    moving = (mag[1:] > 1e-6) & (mag[:-1] > 1e-6)
    straight_step = moving & (np.abs(dth) < np.pi / 6)
    runs = []
    for b in range(straight_step.shape[1]):
        n = 0
        for t in range(straight_step.shape[0]):
            if straight_step[t, b]:
                n += 1
            else:
                runs.append(n + 1)
                n = 0
        runs.append(n + 1)

    xs, ys = cell[..., 0], cell[..., 1]
    edge = (xs == 0) | (xs == size - 1) | (ys == 0) | (ys == size - 1)

    # Realized displacement vs. requested. This USED to be a pure corner-trap
    # fraction, when the boundary clip was the only thing that could shorten a
    # step. Phase 2's --max_action_norm also shortens it, and a policy sitting
    # past the clamp reads clip_frac = 1.000 with no wall involved -- so read
    # this together with realized_mag_mean before calling it a corner trap.
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
    cps = float(np.mean(covs) * size * size / T)
    # Billiard reference at the magnitude the agent ACTUALLY MOVES, not the one
    # it commands. Phase 2 added --min/--max_action_norm, and a policy that has
    # drifted past the clamp commands |a| ~ 8 while moving exactly 2.0 every
    # step. Referencing the commanded magnitude then divides by billiard's score
    # for 8-cell strides -- which is terrible, because 8-cell strides skip most
    # of the arena -- and reports a strategy_efficiency near 4.0 for a policy
    # that is merely at the speed limit. The realized displacement is what the
    # coverage came from, so it is what the reference has to match.
    realized_mag = float(realized.mean())
    ref = billiard_cells_per_step(realized_mag, size, T)
    return {
        "mean_coverage": float(np.mean(covs)),
        "cells_per_step": cps,
        # cells_per_step relative to what a perfect billiard gets at THIS
        # step magnitude. 1.0 = the trajectory is as good as it can be given
        # how far the agent moves per step; a low value with a low
        # step_mag_mean means the magnitude is the problem, not the path.
        "strategy_efficiency": cps / max(ref, 1e-8),
        "billiard_ref_cells_per_step": ref,
        # Both magnitudes, because their ratio is the clamp's bite.
        "realized_mag_mean": realized_mag,
        "union_coverage": float(union),
        **{k: float(v) for k, v in curve.items()},
        "step_mag_mean": float(mag.mean()),
        "step_mag_median": float(np.median(mag)),
        "step_mag_p10": float(np.percentile(mag, 10)),
        "step_mag_p90": float(np.percentile(mag, 90)),
        "straightness": float(straight[ok_s].mean()) if ok_s.any() else 0.0,
        "signed_turn_mean": float(dth[moving].mean()) if moving.any() else 0.0,
        "abs_turn_mean": float(np.abs(dth[moving]).mean()) if moving.any() else 0.0,
        "run_len_mean": float(np.mean(runs)) if runs else 0.0,
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

    # q_accuracy BY DISTANCE TO GOAL -- the reconciliation with P1.
    #
    # The aggregate is trajectory-averaged, and EVERY trajectory ends within a
    # cell or two of the goal, where `to_goal` shrinks toward zero and the angle
    # between two short vectors is dominated by noise. P1 measured the readout
    # over all 400 cells uniformly and found lock=goal 98.7-99.4% with dir_cos
    # 0.963 at ten distractors, i.e. an expected mean near 0.95; the trajectory
    # mean reads 0.711. P1 also flagged a mean dir_acc of 0.696 as the
    # artifact behind a retracted "encoder points 46 degrees wrong" headline,
    # which is uncomfortably close.
    #
    # If the far bins recover ~0.95 and only the near ones are low, the low
    # aggregate is GEOMETRY, not recall failure -- and any mode-A diagnosis
    # drawn from the aggregate (or from q_accuracy_fail) is unsafe.
    gdist = np.linalg.norm(to_goal, axis=-1)
    dist_stats = {}
    for lab, lo, hi in (("d0_2", 0.0, 2.0), ("d2_4", 2.0, 4.0),
                        ("d4_8", 4.0, 8.0), ("d8plus", 8.0, 1e9)):
        band = live & (gdist >= lo) & (gdist < hi)
        for name, arr, ok in (("q_accuracy", qacc, ok_q),
                              ("follow_q", follow, ok_f),
                              ("align_true", align, ok_a)):
            sel = ok & band
            dist_stats[f"{name}_{lab}"] = (
                float(arr[sel].mean()) if sel.sum() >= 20 else float("nan"))
        dist_stats[f"n_steps_{lab}"] = float(band.sum())

    # follow_q and align_true by STEP INDEX. The aggregate averages over the
    # whole trajectory including the opening steps, when the RNN starts from a
    # zero hidden state and has seen one observation. With mean_steps ~ 7, two
    # or three badly-aimed opening steps drag the mean down a lot, so a low
    # aggregate is consistent with either "never follows well" or "follows well
    # once settled". These separate the two. Rows that have already reached the
    # goal are frozen out by `live`, so late bins are over the trials still
    # running -- i.e. the slow ones, which is worth remembering when reading a
    # late bin as "it got better".
    def _by_step(arr, ok):
        out = {}
        for t in list(range(6)) + [("6plus", slice(6, None))]:
            if isinstance(t, tuple):
                key, sl = t
            else:
                key, sl = f"t{t}", slice(t, t + 1)
            sel = ok[sl] & live[sl]
            out[key] = float(arr[sl][sel].mean()) if sel.any() else float("nan")
            out[key + "_n"] = int(sel.sum())
        return out

    follow_by_step = _by_step(follow, ok_f)
    align_by_step = _by_step(align, ok_a)

    start_d = np.linalg.norm(
        np.asarray(starts, dtype=np.float64) - g[None, :], axis=-1)
    final_d = np.linalg.norm(rec["final_pos_f"] - g[None, :], axis=-1)

    # Optimal steps for exactly the starts that succeeded, so path efficiency
    # is not confounded by successes being the nearby ones.
    opt = np.maximum(0.0, start_d - float(getattr(_nav_stats, "_radius", 1.0)))
    eff = (opt[succ] / np.maximum(stg[succ], 1)) if succ.any() else np.array([])

    # --- policy spread, conditioned on distance to the goal ----------------
    # The pass/fail for the P9 state-dependent sigma head. The per-update
    # `sigma` in the training log is a BATCH MEAN and so cannot distinguish a
    # lower global sigma from one that varies with state (EXPERIMENTS_NAV_P2
    # section 9.2). Distance to the goal is the sharper axis because there is a
    # prior prediction: P1 measured the readout degrading sharply within about
    # two cells of the goal, so a sigma that tracks how trustworthy the readout
    # is should RISE there. In a global-sigma run these are constant by
    # construction, which is what makes that run a usable control.
    sig_stats = {}
    if rec.get("sigma") is not None and np.asarray(rec["sigma"]).size:
        sg = np.asarray(rec["sigma"])                       # (T, B)
        mn = np.asarray(rec["mu_norm"])                     # (T, B)
        gd = np.linalg.norm(
            rec["cell"].astype(float) - np.asarray(goal, dtype=float), axis=-1)
        m0 = np.asarray(rec["alive"], dtype=bool)
        T = min(sg.shape[0], gd.shape[0], m0.shape[0])
        sg, mn, gd, m0 = sg[:T], mn[:T], gd[:T], m0[:T]
        # Polar: directional spread is the von Mises circular sd, NOT
        # sigma/||mu||. Recorded per step rather than derived, so the ang_*
        # columns mean the same thing in both parameterizations -- which is
        # what lets a polar run and the section 9.3 table share one axis.
        cs = np.asarray(rec["circ_sd"])[:T] if np.asarray(
            rec.get("circ_sd", [])).size else None
        ang = np.degrees(cs) if cs is not None else np.degrees(
            sg / np.maximum(mn, 1e-8))

        sig_stats["sigma_mean"] = float(sg[m0].mean()) if m0.any() else float("nan")
        sig_stats["mu_norm_mean"] = float(mn[m0].mean()) if m0.any() else float("nan")
        if cs is not None:
            sig_stats["ang_mean"] = float(ang[m0].mean()) if m0.any() else float("nan")
        for lab, lo, hi in (("d0_2", 0.0, 2.0), ("d2_4", 2.0, 4.0),
                            ("d4_8", 4.0, 8.0), ("d8plus", 8.0, 1e9)):
            k = m0 & (gd >= lo) & (gd < hi)
            if k.sum() < 20:
                sig_stats[f"sigma_{lab}"] = float("nan")
                sig_stats[f"ang_{lab}"] = float("nan")
                continue
            sig_stats[f"sigma_{lab}"] = float(np.median(sg[k]))
            sig_stats[f"ang_{lab}"] = float(np.median(ang[k]))

    return {
        **sig_stats,
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
        **{f"follow_q_{k}": v for k, v in follow_by_step.items()
           if not k.endswith("_n")},
        **{f"align_true_{k}": v for k, v in align_by_step.items()
           if not k.endswith("_n")},
        **{f"n_steps_{k[:-2]}": v for k, v in follow_by_step.items()
           if k.endswith("_n")},
        "follow_q_fail": m(follow, ok_f, fail_mask),
        "q_accuracy": m(qacc, ok_q, np.ones_like(succ_mask)),
        "q_accuracy_fail": m(qacc, ok_q, fail_mask),
        **dist_stats,
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
                   choices=["explore", "nav", "warmcold"])
    p.add_argument("--n_distractors", type=int, nargs="+", default=[0, 10])
    p.add_argument("--split", default="recorded",
                   help="Which validation envs to probe. 'recorded' (default) "
                        "is the run's own base_val -- the set it was scored "
                        "against every eval, never trained on but not fresh. "
                        "Otherwise 'trait=level' pairs over place/wall/goal at "
                        "same|held_out|ood, unnamed traits defaulting to "
                        "held_out, e.g. --split place=held_out. Same grammar "
                        "and same minting path as eval_all's --split.")
    p.add_argument("--val_seed", type=int, default=0,
                   help="Seed for minting a split env set; changing it draws a "
                        "different set at the same levels.")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--envs", type=int, default=None,
                   help="val envs to use; default = the checkpoint's own count")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    p.add_argument("--q_rescale", type=float, default=None,
                   help="Force ||q|| to this value before the policy sees it, "
                        "preserving direction (multistep channels scaled the "
                        "same way). The controlled test of whether the policy "
                        "gates on magnitude: run nav mode at the decoy-level "
                        "norm and explore mode at the goal-level norm, and see "
                        "whether following tracks the magnitude or the memory "
                        "contents. Off by default.")
    p.add_argument("--q_scale", type=float, default=None,
                   help="MULTIPLY ||q|| by this factor before the policy sees "
                        "it, preserving both direction and the shape of "
                        "||q||(t). The sound form of the magnitude "
                        "intervention -- unlike --q_rescale it does not also "
                        "destroy the within-trajectory dynamics that P0.8 "
                        "identified as the real cue.")
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
    levels = gen.parse_levels(args.split)
    if levels is not None and args.npos is not None:
        raise SystemExit(
            "--split needs the recorded world.json to mint from, and --npos "
            "builds a shrunken scaffold that cannot use it. Pick one.")
    if levels is None:
        # A shrunken scaffold cannot use the recorded world -- the offsets in
        # world.json index the real Npos and eval_field rejects the mismatch,
        # which is the right guard. Validation mode takes the RNG-replay branch.
        envs, vh, offsets = build_eval_world(
            cfg, encoder, str(device),
            ckpt_path=(None if args.npos is not None else args.ckpt[0]))
    else:
        # Same entry point every eval CLI uses, so `--split place=ood` means one
        # thing project-wide rather than one thing per driver.
        es = eval_env_set(
            cfg, encoder, str(device), ckpt_path=args.ckpt[0], levels=levels,
            val_seed=args.val_seed, n_envs=cfg.num_val_envs)
        envs, vh, offsets = es["envs"], es["field"], es["offsets"]
    _nav_stats._radius = cfg.env.goal_radius

    # Printed because it is the difference between "how did this checkpoint do
    # on the set it was scored against all run" and "does that survive a fresh
    # draw" -- two numbers that look identical in a table and answer different
    # questions. Every behaviour probe before 2026-08-27 was `recorded` and did
    # not say so.
    print(f"split     : {args.split}"
          f"{' (the run OWN validation envs, not a fresh draw)' if levels is None else ' (minted fresh)'}")
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
                    goal_in_mem = mode in ("nav", "warmcold")
                    pats = ([goal_encoding(vh, off, goal)] if goal_in_mem
                            else [])
                    if n_d > 0:
                        pats.extend(sample_distractors(vh, off, env.size,
                                                       n_d, rng))
                    if goal_in_mem:
                        rng.shuffle(pats)
                    for pat in pats:
                        hop.input_memory(torch.from_numpy(pat).float())
                    hops.append(hop)
                    starts.append(random_start(env.size, goal, rng))
                if mode == "warmcold":
                    per_env.append(_warm_vs_cold(
                        agent=agent, env=env, env_offset=off, vectorhash=vh,
                        hopfields=hops, cfg=cfg, device=device, starts=starts,
                        max_steps=args.max_steps))
                    continue
                rec = rollout(
                    agent=agent, env=env, env_offset=off, vectorhash=vh,
                    hopfields=hops, cfg=cfg, device=device, starts=starts,
                    max_steps=args.max_steps,
                    ends_on_arrival=(mode == "nav"),
                    goal_in_memory=(mode == "nav"),
                    q_rescale=args.q_rescale,
                    q_scale=args.q_scale)
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
