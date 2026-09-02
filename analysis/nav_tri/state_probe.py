"""Is the agent storing USEFUL information in its recurrent state?

Two halves, and both are needed, because P2 §27 established that **content
without use is possible**. The auxiliary-visitation head drove
``aux_visited_loss`` from 0.632 to 0.367 -- the trunk demonstrably learned to
represent local visitation -- while the policy, reading that same hidden
vector, went on ignoring it (replay ratio 0.115 against the control's 0.125).
A diagnostic that only decodes would have called that a success.

    CONTENT   Ridge probes from the hidden state h to quantities an explorer
              would want: where am I, where did I start, how long have I been
              going, how much have I covered, have I been just over there.
              Scored as **delta R^2 over an observation-only probe** on
              held-out TRIALS. The raw R^2(h) is nearly meaningless on its own:
              the observation already carries heading, the recall signal, and
              (under `input_abs_position`) position outright, and the trunk
              passes its input forward, so h decodes those whether or not it is
              remembering anything. Only what h adds BEYOND obs is memory.

    USE       Hold the observation fixed and splice in an h from a different
              episode; measure how far the deterministic action moves. If it
              barely moves, the state is decorative no matter how much is
              decodable from it. Reported against the natural spread of the
              action -- how far it moves when the agent is somewhere else
              entirely -- so the number is a fraction, not an action unit.

The two failure modes this separates
------------------------------------
    high content, ~zero use   the trunk represents history and the policy
                              readout ignores it. This is §27's lever B, and
                              the fix is in the policy/optimization, not the
                              representation.
    ~zero content             nothing is being stored. The fix is upstream:
                              the input, the horizon, or the objective.

Caveats that are part of the reading
------------------------------------
* The cross-episode donor h may be off-manifold for this observation, so the
  action could move for reasons that are not "the state is being used".
  `state_influence` is therefore an UPPER bound -- which is the useful
  direction when the finding is that it is near zero. The lag curve is the
  on-manifold version: the donor is this episode's own h from tau steps ago,
  so it is always a state the agent could actually have been in here.
* `swap_same_t` draws the donor from the same STEP INDEX of another episode,
  which controls out the clock. A state that is only a step counter scores
  high on the plain swap and near zero on this one; that difference is the
  reason to run both.
* Probes are fit and scored on DISJOINT TRIALS. Splitting on timesteps leaks
  badly -- consecutive h are near-identical -- and would report memory that is
  not there.

Usage
-----
    python -m analysis.nav_tri.state_probe \
        --ckpt $CLS_RUNS/agent_ckpts/<run>/navigate_u700.pt \
        --mode explore --trials 16 --max_steps 200 --json state.json

Any checkpoint, any channel set, any mode: the probe reads whatever the policy
input happens to be and whatever hidden size the trunk happens to have.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

from hopfield import Hopfield
from analysis.nav_tri.behavior_probe import rollout
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, eval_env_set, load_agent,
)
from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.rollout.visited import VisitedProbe
from hopfield_nav.world import generate as gen

ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
CHUNK = 8192           # forward-pass batch for the splice
EPS = 1e-8


# ---------------------------------------------------------------------------
# CONTENT: ridge probes, fit and scored on disjoint trials
# ---------------------------------------------------------------------------


def _ridge_w(X, Y, alpha):
    """Closed-form ridge on standardised X and centred Y. No intercept."""
    A = X.T @ X
    A[np.diag_indices_from(A)] += alpha
    return np.linalg.solve(A, X.T @ Y)


def _fit_score(Xtr, Ytr, Xte, Yte, alpha):
    """Out-of-sample R^2, one value per target column.

    SStot is taken about the TRAIN mean, so a probe that learned nothing scores
    0 rather than being flattered by the test set's own mean.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ym = Ytr.mean(0)
    w = _ridge_w((Xtr - mu) / sd, Ytr - ym, alpha)
    pred = ((Xte - mu) / sd) @ w + ym
    sse = ((Yte - pred) ** 2).sum(0)
    sst = ((Yte - ym) ** 2).sum(0)
    return 1.0 - sse / np.maximum(sst, EPS)


def _split_trials(trial, frac, rng):
    """Boolean (train, test) masks splitting on TRIAL, never on timestep."""
    ids = np.unique(trial)
    n_held = min(max(1, int(round(frac * len(ids)))), max(len(ids) - 1, 1))
    held = rng.choice(ids, n_held, replace=False)
    te = np.isin(trial, held)
    return ~te, te


def _best_alpha(X, Y, trial, rng):
    """Pick alpha on an inner trial split of the training set."""
    tr, va = _split_trials(trial, 0.25, rng)
    if tr.sum() == 0 or va.sum() == 0:
        return ALPHAS[len(ALPHAS) // 2]
    scores = [float(np.mean(_fit_score(X[tr], Y[tr], X[va], Y[va], a)))
              for a in ALPHAS]
    return ALPHAS[int(np.argmax(scores))]


def content_probes(obs, h, targets, trial, rng, test_frac=0.3) -> dict:
    """delta R^2 = R^2([obs, h]) - R^2(obs), per named target block.

    All three columns are reported because they answer different questions:
    R^2(obs) is what the current observation alone gives you, R^2(h) is what
    the state gives you (inflated, since the trunk sees the observation), and
    the delta is the only one that means "stored".
    """
    tr, te = _split_trials(trial, test_frac, rng)
    both = np.concatenate([obs, h], axis=1)
    out = {}
    for name, Y in targets.items():
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        row = {}
        for key, X in (("obs", obs), ("h", h), ("both", both)):
            a = _best_alpha(X[tr], Y[tr], trial[tr], rng)
            r2 = _fit_score(X[tr], Y[tr], X[te], Y[te], a)
            row[key] = float(np.mean(r2))
            row[key + "_alpha"] = float(a)
        row["delta"] = row["both"] - row["obs"]
        row["dim"] = int(Y.shape[1])
        # A target that never changes within an episode has only as many
        # independent samples as there are trials, not as there are steps.
        # `start_pos` is the case that matters: 67 effective samples against a
        # 1024-unit hidden state, which is the lowest-powered row in the table
        # and must not be read as a proven null.
        const = all(np.ptp(Y[trial == t], axis=0).max() < 1e-9
                    for t in np.unique(trial)[:8])
        row["eff_n"] = int(len(np.unique(trial))) if const else int(len(Y))
        out[name] = row
    out["_n_train_trials"] = int(len(np.unique(trial[tr])))
    out["_n_test_trials"] = int(len(np.unique(trial[te])))
    out["_n_samples"] = int(obs.shape[0])
    return out


# ---------------------------------------------------------------------------
# USE: hold the observation fixed, swap the state, watch the action
# ---------------------------------------------------------------------------


@torch.no_grad()
def _act(agent, obs, h, n_layers, device):
    """Deterministic action for (obs, h) pairs. (N, D), (N, H) -> (N, 2).

    The deterministic action is the right quantity: it is what the evaluation
    protocol executes, and it is unaffected by kappa (§20.1), so a difference
    here is a difference in behaviour rather than in spread.
    """
    outs = []
    for i in range(0, obs.shape[0], CHUNK):
        x = torch.as_tensor(obs[i:i + CHUNK], dtype=torch.float32,
                            device=device).unsqueeze(1)
        n = x.shape[0]
        hh = torch.as_tensor(h[i:i + CHUNK], dtype=torch.float32,
                             device=device)
        hh = hh.view(n, n_layers, -1).permute(1, 0, 2).contiguous()
        r = agent.get_action_and_value(x, hh, deterministic=True)
        outs.append(r["move_action"].reshape(n, -1).float().cpu().numpy())
    return np.concatenate(outs, 0)


def _donor(trial, rng, step=None):
    """Index permutation drawing each row's donor from a DIFFERENT trial.

    With ``step`` given the donor sits at the same step index, which holds the
    clock fixed so only episode-specific history can move the action.
    """
    n = len(trial)
    idx = np.arange(n)
    groups = ([(np.arange(n), np.arange(n))] if step is None
              else [(np.nonzero(step == s)[0],) * 2 for s in np.unique(step)])
    for rows, cand in groups:
        if len(cand) < 2:
            continue                      # nothing to swap with; stays a no-op
        pick = rng.choice(cand, size=len(rows))
        for _ in range(20):
            bad = trial[pick] == trial[rows]
            if not bad.any():
                break
            pick[bad] = rng.choice(cand, size=int(bad.sum()))
        idx[rows] = pick
    return idx


def use_probes(agent, obs, h, trial, step, n_layers, device, rng,
               lags=(1, 2, 5, 10, 20, 50)) -> dict:
    """How much of the action is driven by state rather than observation."""
    base = _act(agent, obs, h, n_layers, device)
    scale = float(np.linalg.norm(base, axis=1).mean())

    def d(a):
        return float(np.linalg.norm(a - base, axis=1).mean())

    any_i = _donor(trial, rng)
    same_i = _donor(trial, rng, step=step)
    # Swapping BOTH halves is the natural spread of the action: how far it
    # moves when the agent is in a different episode entirely. Everything else
    # is a fraction of it, so the numbers compare across agents whose action
    # scales differ.
    d_both = d(_act(agent, obs[any_i], h[any_i], n_layers, device))
    d_state = d(_act(agent, obs, h[any_i], n_layers, device))
    d_obs = d(_act(agent, obs[any_i], h, n_layers, device))
    d_same_t = d(_act(agent, obs, h[same_i], n_layers, device))
    d_zero = d(_act(agent, obs, np.zeros_like(h), n_layers, device))

    lag_curve = {}
    span = int(step.max()) + 1
    key = trial.astype(np.int64) * span + step.astype(np.int64)
    pos = {int(k): i for i, k in enumerate(key)}
    for L in lags:
        ok = step >= L
        if ok.sum() < 32:
            continue
        want = (trial[ok].astype(np.int64) * span
                + (step[ok].astype(np.int64) - L))
        src = np.array([pos.get(int(w), -1) for w in want])
        ok2 = src >= 0
        if ok2.sum() < 32:
            continue
        rows = np.nonzero(ok)[0][ok2]
        a = _act(agent, obs[rows], h[src[ok2]], n_layers, device)
        lag_curve[str(L)] = float(
            np.linalg.norm(a - base[rows], axis=1).mean() / max(d_both, EPS))

    return {
        "action_norm": scale,
        "d_both": d_both,
        "d_state": d_state,
        "d_obs": d_obs,
        "d_same_t": d_same_t,
        "d_zero": d_zero,
        "state_influence": d_state / max(d_both, EPS),
        "obs_influence": d_obs / max(d_both, EPS),
        "same_t_influence": d_same_t / max(d_both, EPS),
        "zero_influence": d_zero / max(d_both, EPS),
        "state_share": d_state / max(d_state + d_obs, EPS),
        "lag_curve": lag_curve,
    }


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def build_targets(rec, size: int, radius: float) -> dict:
    """Per-step quantities an explorer would need to have stored.

    ``pos`` and ``heading`` are POSITIVE CONTROLS for the observation probe --
    heading is in `prev_action` outright, so R^2(obs) near 1 says the machinery
    works. ``start_pos`` is the opposite control: it is constant within an
    episode and appears in no channel, so anything above 0 there is path
    integration.
    """
    pos = np.asarray(rec["pos_f"], dtype=np.float64)          # (T, B, 2)
    cell = np.asarray(rec["cell"], dtype=np.int64)            # (T, B, 2)
    act = np.asarray(rec["action"], dtype=np.float64)         # (T, B, 2)
    T, B, _ = pos.shape
    norm = 2.0 * pos / max(size - 1, 1) - 1.0

    start = np.repeat(norm[:1], T, axis=0)
    elapsed = np.repeat(
        (np.arange(T, dtype=np.float64) / max(T - 1, 1))[:, None], B, axis=1)

    # Coverage so far: unique snapped cells seen up to and including t.
    seen = np.zeros((B, size, size), dtype=bool)
    cov = np.zeros((T, B))
    for t in range(T):
        seen[np.arange(B), cell[t, :, 0], cell[t, :, 1]] = True
        cov[t] = seen.reshape(B, -1).sum(1) / float(size * size)

    # The same 8-bit vector the aux head was trained on, replayed in the same
    # read-then-mark order, so it is the quantity §27 measured.
    vp = VisitedProbe(size, radius, B)
    vis = np.stack([vp.read(cell[t]) for t in range(T)])       # (T, B, 8)

    prev = np.concatenate([np.zeros((1, B, 2)), act[:-1]], axis=0)
    head = prev / np.maximum(np.linalg.norm(prev, axis=-1, keepdims=True), EPS)

    def flat(a):
        return np.asarray(a, dtype=np.float64).reshape(T * B, -1)

    return {"pos": flat(norm), "start_pos": flat(start),
            "elapsed": flat(elapsed[..., None]),
            "coverage": flat(cov[..., None]),
            "visited8": flat(vis), "heading": flat(head)}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect(*, agent, cfg, envs, vh, offsets, embed_dim, device, args, mode,
            n_distractors):
    """Roll out every env and stack (obs, h, targets, trial, step)."""
    rng = np.random.RandomState(args.seed)
    obs, hid, hcon, trial, step = [], [], [], [], []
    targets: dict = {}
    next_trial = 0
    for i, env in enumerate(envs):
        goal, off = env.goal_location, offsets[i]
        hops, starts = [], []
        for _ in range(args.trials):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                           device=str(device))
            goal_in_mem = mode == "nav"
            pats = [goal_encoding(vh, off, goal)] if goal_in_mem else []
            if n_distractors > 0:
                pats.extend(sample_distractors(vh, off, env.size,
                                               n_distractors, rng))
            if goal_in_mem:
                rng.shuffle(pats)
            for pat in pats:
                hop.input_memory(torch.from_numpy(pat).float())
            hops.append(hop)
            starts.append(random_start(env.size, goal, rng))

        rec = rollout(
            agent=agent, env=env, env_offset=off, vectorhash=vh,
            hopfields=hops, cfg=cfg, device=device, starts=starts,
            max_steps=args.max_steps, ends_on_arrival=(mode == "nav"),
            goal_in_memory=(mode == "nav"), record_state=True,
            deterministic=not args.sampled_rollout)

        o = np.asarray(rec["obs"], dtype=np.float64)           # (T, B, D)
        # `h_in` for USE always: the action at t is f(obs_t, h_t), so splicing
        # anything else answers a question nobody asked. CONTENT can take
        # either, and `--content_h out` is the one that matches a supervised
        # head bolted onto the trunk, which reads `features` = h_{t+1}.
        hh = np.asarray(rec["h_in"], dtype=np.float64)         # (T, B, H)
        hc = (np.asarray(rec["h_out"], dtype=np.float64)
              if args.content_h == "out" else hh)
        T, B, _ = o.shape
        tg = build_targets(rec, env.size,
                           getattr(cfg.agent, "aux_visited_radius", 3.0))
        # A nav agent frozen on the goal contributes identical rows that would
        # otherwise dominate every mean.
        keep = (np.asarray(rec["alive"], dtype=bool).reshape(T * B)
                if mode == "nav" else np.ones(T * B, dtype=bool))
        obs.append(o.reshape(T * B, -1)[keep])
        hid.append(hh.reshape(T * B, -1)[keep])
        hcon.append(hc.reshape(T * B, -1)[keep])
        trial.append((next_trial + np.tile(np.arange(B), T))[keep])
        step.append(np.repeat(np.arange(T), B)[keep])
        for k, v in tg.items():
            targets.setdefault(k, []).append(v[keep])
        next_trial += B

    return (np.concatenate(obs), np.concatenate(hid), np.concatenate(hcon),
            {k: np.concatenate(v) for k, v in targets.items()},
            np.concatenate(trial), np.concatenate(step))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

STORED = 0.05          # deltaR^2 above this is real content, not probe noise
USED = 0.15            # state_influence above this moves the action


def report(res: dict, label: str) -> None:
    c, u = res["content"], res["use"]
    print(f"\n================ {label} ================")
    print(f"  {c['_n_samples']} steps, {c['_n_train_trials']} train / "
          f"{c['_n_test_trials']} test trials, hidden {res['hidden']}, "
          f"obs {res['obs_dim']}")

    h_at = res.get("content_h", "in")
    print("\n  CONTENT -- out-of-sample R^2 on held-out trials, h_%s (%s)"
          % (h_at, "h_t, the state the action read" if h_at == "in"
             else "h_{t+1}, what an aux head sees"))
    print("    %-12s %4s %8s %8s %8s %9s %9s"
          % ("target", "dim", "eff_n", "R2(obs)", "R2(h)", "R2(both)",
             "deltaR2"))
    for k, v in c.items():
        if k.startswith("_"):
            continue
        print("    %-12s %4d %8d %8.3f %8.3f %9.3f %9.3f"
              % (k, v["dim"], v.get("eff_n", c["_n_samples"]), v["obs"],
                 v["h"], v["both"], v["delta"]))
    print("    deltaR2 is the only memory number: what h adds beyond the "
          "current observation.")
    print("    eff_n is INDEPENDENT samples: a target constant within an "
          "episode has only")
    print("    as many as there are trials, so a 0.000 there is weak evidence, "
          "not a null.")

    print("\n  USE -- deterministic action displacement under a splice")
    print("    mean |action|                    %.4f" % u["action_norm"])
    print("    swap BOTH (the scale)            %.4f  = 1.000" % u["d_both"])
    print("    swap STATE only                  %.4f  = %.3f of scale"
          % (u["d_state"], u["state_influence"]))
    print("    swap STATE, same step index      %.4f  = %.3f"
          % (u["d_same_t"], u["same_t_influence"]))
    print("    swap OBSERVATION only            %.4f  = %.3f"
          % (u["d_obs"], u["obs_influence"]))
    print("    zero the state                   %.4f  = %.3f"
          % (u["d_zero"], u["zero_influence"]))
    print("    state_share = state/(state+obs)  %.3f" % u["state_share"])
    if u["lag_curve"]:
        ks = sorted(u["lag_curve"], key=int)
        print("    own-episode donor tau steps back (fraction of scale):")
        print("      tau  " + "".join("%8s" % k for k in ks))
        print("      d    " + "".join("%8.3f" % u["lag_curve"][k] for k in ks))

    best_k, best = "", 0.0
    for k, v in c.items():
        if not k.startswith("_") and v["delta"] > best:
            best_k, best = k, v["delta"]
    stored, used = best > STORED, u["state_influence"] > USED
    print("\n  VERDICT: content %s (best deltaR2 %.3f on %r), use %s "
          "(state_influence %.3f)"
          % ("YES" if stored else "no", best, best_k or "-",
             "YES" if used else "no", u["state_influence"]))
    if stored and not used:
        print("  -> the trunk represents history and the POLICY IGNORES IT. "
              "The §27 lever-B failure; the fix is in the readout, not the "
              "representation.")
    elif not stored:
        print("  -> nothing is being stored. The fix is upstream of the "
              "trunk: input, horizon, or objective.")
    else:
        print("  -> the state carries history AND changes the action.")


BASELINE_SPREAD = 0.15   # R^2(obs) varying more than this across arms


def _short(name: str) -> str:
    """A column heading: the run directory without the boilerplate."""
    d = name.split("/")[-2] if "/" in name else name
    return re.sub(r"_s\d+_\d+$", "", re.sub(r"^navigate_", "", d))


def cross_report(out: dict) -> dict:
    """Compare arms -- and refuse to let deltaR^2 be compared naively.

    Found by running it: R^2(obs) for `pos` was 0.067 on `p20_e` and 0.728 on
    `p24_aux` **with the same 74 input channels**. The observation baseline is
    not a property of the channel set; it is a property of the states the agent
    actually visits, and a narrow, orbit-like distribution makes position
    linearly decodable from the sensory code in a way a broad one does not.

    So where the baseline moves, a deltaR^2 difference is a HEADROOM
    difference, not a storage difference, and R^2(both) -- the total
    decodability -- is the honest cross-arm column. Where the baseline is flat
    near zero (elapsed, coverage, visited8 on the arms that do not take it as
    input) deltaR^2 compares cleanly. This flags which is which rather than
    leaving it to be noticed.
    """
    names = list(out)
    short = [_short(n) for n in names]
    targets = [k for k in out[names[0]]["content"] if not k.startswith("_")]
    flagged = {}
    print("\n\n================ ACROSS CHECKPOINTS ================")
    for title, col in (("deltaR2 -- what h adds beyond obs", "delta"),
                       ("R2(both) -- total decodability", "both")):
        print(f"\n  {title}")
        print("    %-12s" % "target" + "".join("%17s" % s[:16] for s in short))
        for t in targets:
            base = [out[n]["content"][t]["obs"] for n in names]
            spread = max(base) - min(base)
            flagged[t] = bool(spread > BASELINE_SPREAD)
            mark = " [!]" if flagged[t] and col == "delta" else ""
            print("    %-12s" % t
                  + "".join("%17.3f" % out[n]["content"][t][col]
                            for n in names) + mark)
    bad = [t for t, f in flagged.items() if f]
    if bad:
        print(f"\n    [!] {', '.join(bad)}: R^2(obs) itself differs across "
              f"these arms by")
        print("        more than %.2f, so the deltaR2 gap is headroom, not "
              "storage." % BASELINE_SPREAD)
        print("        Read R2(both) for those rows.")

    print("\n  USE")
    print("    %-16s" % "" + "".join("%17s" % s[:16] for s in short))
    for k, lab in (("state_influence", "state_influence"),
                   ("same_t_influence", "same-step swap"),
                   ("obs_influence", "obs_influence"),
                   ("state_share", "state_share")):
        print("    %-16s" % lab
              + "".join("%17.3f" % out[n]["use"][k] for n in names))
    lags = sorted({int(L) for n in names for L in out[n]["use"]["lag_curve"]})
    if lags:
        print("\n  lag curve (own-episode donor tau steps back)")
        print("    %-16s" % "tau" + "".join("%17s" % s[:16] for s in short))
        for L in lags:
            print("    %-16d" % L
                  + "".join("%17.3f"
                            % out[n]["use"]["lag_curve"].get(str(L), np.nan)
                            for n in names))
    return {"baseline_confounded": bad}


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, nargs="+",
                   help="One or more checkpoints, probed in ONE process so the "
                        "12 GB scaffold is built once. They must share a "
                        "world; that is checked, not assumed.")
    p.add_argument("--mode", default="explore", choices=["explore", "nav"])
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--split", default="recorded",
                   help="Which validation envs to probe; same grammar as "
                        "behavior_probe and eval_all.")
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--envs", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--sampled_rollout", action="store_true",
                   help="Collect states under SAMPLED actions. The splice "
                        "still compares deterministic means -- this only "
                        "changes which states get visited, which matters "
                        "because a deterministic rollout can sit in a much "
                        "narrower part of the state space (§20.4).")
    p.add_argument("--content_h", default="in", choices=["in", "out"],
                   help="Which hidden state the CONTENT probes read. 'in' "
                        "(default) is h_t, the state the action at t was "
                        "computed from -- causally matched to the USE half, "
                        "which always uses h_t. 'out' is h_{t+1}, the state "
                        "AFTER obs_t, which is what a supervised head bolted "
                        "onto the trunk sees (`features`); use it to compare "
                        "against an aux-head loss, since a probe on h_t and a "
                        "head on h_{t+1} are not the same quantity (§30.7).")
    p.add_argument("--lags", type=int, nargs="*",
                   default=[1, 2, 5, 10, 20, 50])
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    p.add_argument("--npos", type=int, default=None,
                   help="Shrink the scaffold for tool validation on a laptop. "
                        "Changes the world; numbers under it are not readable.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cks = [torch.load(c, map_location="cpu", weights_only=False)
           for c in args.ckpt]
    cfg = cfg_from_checkpoint(cks[0]["config"])
    if args.envs is not None:
        cfg.num_val_envs = args.envs

    _WORLD_KEYS = ("encoder_checkpoint", "fwhm_ratio")
    for path, other in zip(args.ckpt[1:], cks[1:]):
        o = cfg_from_checkpoint(other["config"])
        bad = [k for k in _WORLD_KEYS if getattr(o, k) != getattr(cfg, k)]
        bad += [f"vectorhash.{k}" for k in ("lambdas", "Npos")
                if getattr(o.vectorhash, k) != getattr(cfg.vectorhash, k)]
        bad += [f"env.{k}" for k in ("size", "wall_resolution", "goal_radius")
                if getattr(o.env, k) != getattr(cfg.env, k)]
        if bad:
            raise SystemExit(f"{path} does not share a world with "
                             f"{args.ckpt[0]}: {', '.join(bad)} differ.")

    if args.npos is not None:
        print(f"  WARNING: --npos {args.npos} overrides the scaffold. "
              f"Tool-validation mode; numbers are not comparable.")
        cfg.vectorhash.Npos = args.npos

    encoder, enc_cfg, gain = load_encoder(
        cfg.encoder_checkpoint, str(device), getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)

    levels = gen.parse_levels(args.split)
    if levels is not None and args.npos is not None:
        raise SystemExit("--split needs the recorded world; --npos builds a "
                         "shrunken one. Pick one.")
    if levels is None:
        envs, vh, offsets = build_eval_world(
            cfg, encoder, str(device),
            ckpt_path=(None if args.npos is not None else args.ckpt[0]))
    else:
        es = eval_env_set(cfg, encoder, str(device), ckpt_path=args.ckpt[0],
                          levels=levels, val_seed=args.val_seed,
                          n_envs=cfg.num_val_envs)
        envs, vh, offsets = es["envs"], es["field"], es["offsets"]

    print(f"split     : {args.split}"
          f"{' (the run OWN validation envs)' if levels is None else ' (minted fresh)'}")
    print(f"envs      : {len(envs)}  trials/env: {args.trials}  "
          f"steps: {args.max_steps}  mode: {args.mode} "
          f"n_dist={args.n_distractors}")
    print(f"rollout   : "
          f"{'SAMPLED' if args.sampled_rollout else 'deterministic'}")

    out = {}
    for path, ck in zip(args.ckpt, cks):
        # Its OWN config. Building every agent from cks[0] hands the others the
        # first run's architecture knobs while loading their weights -- the bug
        # that made every §18.4 number for `p20_e_kcap` come off an agent the
        # run never produced.
        own = cfg_from_checkpoint(ck["config"])
        if own.hopfield.beta is None:
            # `collect` builds this checkpoint's Hopfields from `own`, so the
            # beta resolved onto cfg above would not reach them.
            own.hopfield.beta = float(gain)
        diff = [k for k in vars(cfg.agent)
                if getattr(own.agent, k, None) != getattr(cfg.agent, k)]
        if diff:
            print(f"  NOTE {path} differs on agent knobs: "
                  f"{', '.join(sorted(diff))} -- built from its own.")
        agent = load_agent(own, ck["agent_state_dict"], embed_dim, device)
        obs, hid, hcon, targets, trial, step = collect(
            agent=agent, cfg=own, envs=envs, vh=vh, offsets=offsets,
            embed_dim=embed_dim, device=device, args=args, mode=args.mode,
            n_distractors=args.n_distractors)
        rng = np.random.RandomState(args.seed)
        res = {
            "hidden": int(hid.shape[1]),
            "obs_dim": int(obs.shape[1]),
            "content_h": args.content_h,
            "content": content_probes(obs, hcon, targets, trial, rng,
                                      args.test_frac),
            "use": use_probes(agent, obs, hid, trial, step,
                              own.agent.num_rnn_layers, device, rng,
                              tuple(args.lags)),
        }
        report(res, path)
        out[path] = res

    cross = cross_report(out) if len(out) > 1 else {}

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"mode": args.mode, "max_steps": args.max_steps,
                       "cross": cross, "by_ckpt": out}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
