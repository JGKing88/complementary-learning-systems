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
GRID = 4               # coarse occupancy blocks per side (16 cells of a 20x20)
POS_LAGS = (5, 10, 20)  # "where was I k steps ago"


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


def _flow_basis(pos, env, n_feat=96, seed=0, scales=(1.0, 3.0, 8.0)):
    """Random Fourier features of position, in a separate block per env.

    The proper null for "the past is a function of the present". §22 found the
    policy is a deterministic vector field, so pos(t-k) is recoverable from
    pos(t) by running the flow backwards -- and that map is SMOOTH BUT
    NONLINEAR, and different in every environment because the walls are. A
    linear position column (the `anchor` rung) cannot represent it, so a state
    that merely encodes position in a rich basis scores as memory against it.
    This basis can, which makes `delta_flow` the column that means trajectory.
    """
    pos = np.asarray(pos, dtype=np.float64)
    rng = np.random.RandomState(seed)
    per = max(n_feat // (2 * len(scales)), 1)
    W = np.concatenate([rng.randn(pos.shape[1], per) * s for s in scales], 1)
    b = rng.rand(W.shape[1]) * 2.0 * np.pi
    z = np.concatenate([np.cos(pos @ W + b), np.sin(pos @ W + b)], axis=1)
    ids = np.unique(env)
    out = np.zeros((len(pos), z.shape[1] * len(ids)))
    for j, e in enumerate(ids):
        m = env == e
        out[m, j * z.shape[1]:(j + 1) * z.shape[1]] = z[m]
    return out


class _Ridge:
    """`_fit_score` with the expensive part hoisted out of the alpha loop.

    The Gram matrix is 1098x1098 built from 13k rows and does not depend on
    alpha or on the target, so the naive form recomputes it 350 times for one
    table. This standardises and forms it once per (regressor, split); alphas
    then cost a solve. `test_it_matches_the_reference_fit` pins it against
    `_fit_score`, which stays the readable definition.
    """

    def __init__(self, Xtr, Xte):
        self.mu = Xtr.mean(0)
        sd = Xtr.std(0)
        self.sd = np.where(sd < 1e-8, 1.0, sd)
        self.Ztr = (Xtr - self.mu) / self.sd
        self.Zte = (Xte - self.mu) / self.sd
        self.G = self.Ztr.T @ self.Ztr

    def score(self, Ytr, Yte, alpha):
        ym = Ytr.mean(0)
        A = self.G.copy()
        A[np.diag_indices_from(A)] += alpha
        w = np.linalg.solve(A, self.Ztr.T @ (Ytr - ym))
        pred = self.Zte @ w + ym
        sse = ((Yte - pred) ** 2).sum(0)
        sst = ((Yte - ym) ** 2).sum(0)
        return 1.0 - sse / np.maximum(sst, EPS)


def content_probes(obs, h, targets, trial, rng, test_frac=0.3,
                   clock=None, anchor=None, env=None) -> dict:
    """delta R^2 = R^2([obs, h]) - R^2(obs), per named target block.

    All the columns are reported because they answer different questions:
    R^2(obs) is what the current observation alone gives you, R^2(h) is what
    the state gives you (inflated, since the trunk sees the observation), and
    the delta is the only one that means "stored".

    ``clock`` and ``anchor`` build a LADDER OF BASELINES, each rung ruling out
    a cheaper explanation for the one before it:

    ``delta``      beyond the observation.
    ``delta_clk``  beyond the observation AND a perfect clock. §30.6 found the
                   state is partly a clock, and coverage-so-far is nearly a
                   function of elapsed time, so `delta` scores a pure clock as
                   spatial content. This rung is what caught that.
    ``delta_anc``  beyond those AND the agent's current position. Needed
                   because §22 established the policy is a deterministic
                   vector field, and under a deterministic flow **the past is
                   a function of the present** -- so decoding "where was I 20
                   steps ago" from h is not evidence of a trajectory memory
                   until knowing where you are now has been ruled out.

    The rung a target is itself part of is degenerate there (`elapsed` has
    delta_clk 0, `pos` has delta_anc 0) and that is the self-check.
    """
    tr, te = _split_trials(trial, test_frac, rng)
    regs = {"obs": obs, "h": h, "both": np.concatenate([obs, h], axis=1)}
    base = obs
    if clock is not None:
        base = np.concatenate(
            [base, np.asarray(clock, np.float64).reshape(-1, 1)], axis=1)
        regs["obsclk"] = base
        regs["bothclk"] = np.concatenate([base, h], axis=1)
    if anchor is not None:
        base = np.concatenate([base, np.asarray(anchor, np.float64)], axis=1)
        regs["obsanc"] = base
        regs["bothanc"] = np.concatenate([base, h], axis=1)
        if env is not None:
            base = np.concatenate([base, _flow_basis(anchor, env)], axis=1)
            regs["obsflow"] = base
            regs["bothflow"] = np.concatenate([base, h], axis=1)

    # One inner split, shared by every target and regressor. Previously each
    # (target, regressor) drew its own, which added variance across a table
    # whose rows are meant to be compared with each other.
    itr, iva = _split_trials(trial[tr], 0.25, rng)
    outer = {k: _Ridge(X[tr], X[te]) for k, X in regs.items()}
    inner = {k: _Ridge(X[tr][itr], X[tr][iva]) for k, X in regs.items()}

    out = {}
    for name, Y in targets.items():
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        Ytr = Y[tr]
        row = {}
        for key in regs:
            sc = [float(np.mean(inner[key].score(Ytr[itr], Ytr[iva], a)))
                  for a in ALPHAS]
            a = ALPHAS[int(np.argmax(sc))]
            row[key] = float(np.mean(outer[key].score(Ytr, Y[te], a)))
            row[key + "_alpha"] = float(a)
        row["delta"] = row["both"] - row["obs"]
        if clock is not None:
            row["delta_clk"] = row["bothclk"] - row["obsclk"]
        if anchor is not None:
            row["delta_anc"] = row["bothanc"] - row["obsanc"]
            if env is not None:
                row["delta_flow"] = row["bothflow"] - row["obsflow"]
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


def _shuffle_units(h, rng):
    """Permute each unit independently across samples.

    Column-wise, so every unit's marginal distribution is preserved exactly
    and only the cross-unit structure is destroyed. That is what makes it a
    fair null for a ReLU trunk, whose activity is non-negative and sparse:
    matching a Gaussian to h's mean and sd would put half its mass below zero,
    somewhere the state can never be, and the policy's response to THAT says
    nothing about how much the real state matters.
    """
    idx = np.argsort(rng.random((h.shape[0], h.shape[1])), axis=0)
    return np.take_along_axis(h, idx, axis=0)


def _donor_matched(trial, pos, rng, cell=1.0, heading=None, n_oct=8):
    """Donor from a DIFFERENT episode standing in the same place.

    The plain donor comes from wherever that episode happened to be, so
    splicing the POSITION subspace leaves the state saying "I am at B" while
    the held-fixed observation still says "I am at A" -- a contradiction the
    policy has never seen. Crucially that is asymmetric across subspaces: the
    observation carries no visitation signal, so swapping the occupancy
    directions is merely uninformative, while swapping the position directions
    is self-contradictory. Comparing the two as-is therefore flatters position.

    Matching the donor on position removes the contradiction and turns the
    splice into the question actually worth asking: two agents in the same
    place with different histories -- does the policy tell them apart?

    Rows with no same-cell partner from another episode keep their own index
    and are reported, not silently counted as "no effect".
    """
    p = np.asarray(pos, dtype=np.float64)
    key = np.rint(p / float(cell)).astype(np.int64)
    key = key[:, 0] * 100003 + key[:, 1]
    if heading is not None:
        # Matching position alone was not enough, and the failure was
        # informative: the "position subspace" still moved the action ~10x
        # between two agents in the SAME CELL. Those 2 directions are the ones
        # position is most decodable FROM, not position itself, and they carry
        # whatever else correlates with it -- heading first among them, which
        # the observation also encodes (prev_action, prev_displacement). So a
        # heading mismatch reintroduces exactly the contradiction the position
        # match was added to remove.
        hd = np.asarray(heading, dtype=np.float64)
        ang = np.arctan2(hd[:, 1], hd[:, 0])
        oct_ = (np.floor((ang + np.pi) / (2.0 * np.pi) * n_oct)
                .astype(np.int64) % n_oct)
        key = key * 16 + oct_
    idx = np.arange(len(trial))
    buckets: dict = {}
    for i, k in enumerate(key):
        buckets.setdefault(int(k), []).append(i)
    unmatched = 0
    for i, k in enumerate(key):
        cand = [j for j in buckets[int(k)] if trial[j] != trial[i]]
        if not cand:
            unmatched += 1
            continue
        idx[i] = cand[rng.randint(len(cand))]
    return idx, unmatched


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


def _readout_subspace(h, Y, alpha=100.0):
    """Orthonormal basis for the directions of h that code Y.

    The ridge decoder's weight matrix (H, k) spans the subspace a linear
    readout of Y would look at; its left singular vectors are that subspace's
    basis. Fit on TRAIN rows only, so the splice is evaluated on states the
    subspace was not chosen against.
    """
    Z = h - h.mean(0)
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += alpha
    W = np.linalg.solve(A, Z.T @ (Y - Y.mean(0)))
    Q, s, _ = np.linalg.svd(W, full_matrices=False)
    return Q[:, s > s[0] * 1e-6] if s[0] > 0 else Q[:, :1]


def _orth_against(Q, R):
    """Q with the span of R projected out, re-orthonormalised.

    The USE-side counterpart of the content ladder. The occupancy, visited8 and
    position decoders are not orthogonal -- visitation near a wall is partly a
    statement about where you are -- so a target subspace can inherit its
    causal punch from the position directions, which every arm reads at ~10x
    random. This asks what the subspace does once position is taken out.
    """
    P = Q - R @ (R.T @ Q)
    U, s, _ = np.linalg.svd(P, full_matrices=False)
    return U[:, s > max(float(s[0]), EPS) * 1e-6] if s.size else P


def subspace_splice(agent, obs, h, trial, Q, n_layers, device, rng,
                    n_rand=8, pos=None, heading=None, cell=1.0) -> dict:
    """Swap ONLY the component of h inside Q, keeping the rest intact.

    The whole-state swap in `use_probes` replaces position, clock and map at
    once, so it says the state matters -- never WHICH content matters. This
    replaces one readout subspace and leaves its orthogonal complement alone,
    which is the question "does the policy read *this*".

    The random-subspace control is not optional: any k-dimensional
    perturbation of h moves the action somewhat, so `d_sub` alone is
    uninterpretable. `ratio` above 1 is the claim.
    """
    base = _act(agent, obs, h, n_layers, device)
    don = _donor(trial, rng)

    def swap_with(B, src):
        """(action displacement, size of the edit made to h).

        The second number is the control the ratio was missing. A readout
        subspace for something the trunk encodes strongly is a HIGH-VARIANCE
        subspace, while a random 2-plane in 1024 dimensions captures about
        2/1024 of the state's variance -- so `d_sub / d_random` partly
        measures "this subspace is bigger", not "the policy weights it more".
        Dividing each by the perturbation it actually applied turns both into
        a sensitivity, action-change per unit of state-change, and the two
        become comparable.
        """
        g = h - (h @ B) @ B.T + (h[src] @ B) @ B.T
        d_act = float(np.linalg.norm(
            _act(agent, obs, g, n_layers, device) - base, axis=1).mean())
        d_h = float(np.linalg.norm(g - h, axis=1).mean())
        return d_act, d_h

    def swap(B):
        return swap_with(B, don)

    d_sub, dh_sub = swap(Q)
    rand = [swap(np.linalg.qr(rng.randn(h.shape[1], Q.shape[1]))[0])
            for _ in range(n_rand)]
    d_rand = float(np.mean([r[0] for r in rand]))
    dh_rand = float(np.mean([r[1] for r in rand]))
    # Mean of the per-draw sensitivities, NOT the ratio of the means. A random
    # subspace's action-change and edit-size are strongly correlated across
    # draws (both scale with how much of h's variance the draw happens to
    # catch), so a ratio of means is dominated by whichever draw was largest
    # and swings badly at small n_rand.
    _sens = [r[0] / max(r[1], EPS) for r in rand]
    sens_rand = float(np.mean(_sens))
    # Spread of the random baseline across draws. Without it a gap like
    # 2.48 vs 2.17 between two arms cannot be called a difference at all, and
    # this project has already over-read one such gap.
    sens_sd = float(np.std(_sens, ddof=1)) if len(_sens) > 1 else float("nan")
    d_full = float(np.linalg.norm(
        _act(agent, obs, h[don], n_layers, device) - base, axis=1).mean())
    out = {
        "rank": int(Q.shape[1]),
        "d_sub": d_sub,
        "d_rand": d_rand,
        "d_full": d_full,
        "dh_sub": dh_sub,
        "dh_rand": dh_rand,
        "ratio": d_sub / max(d_rand, EPS),
        # Action-change per unit of state-change, against the same for a
        # random subspace. This is the ratio with subspace SIZE divided out.
        "ratio_sens": ((d_sub / max(dh_sub, EPS)) / max(sens_rand, EPS)),
        "size_vs_random": dh_sub / max(dh_rand, EPS),
        # 1-sigma band on ratio_sens induced by the random baseline alone.
        "ratio_sens_sd": ((d_sub / max(dh_sub, EPS)) * sens_sd
                          / max(sens_rand ** 2, EPS)),
        "n_rand": int(n_rand),
        "frac_of_full": d_sub / max(d_full, EPS),
    }
    if pos is not None:
        # Same place, different episode. Removes the state/observation
        # contradiction that only the position subspace suffers, so the
        # subspaces become comparable to each other.
        don_m, unmatched = _donor_matched(trial, pos, rng, cell=cell,
                                          heading=heading)
        m_sub, mh_sub = swap_with(Q, don_m)
        mr = [swap_with(np.linalg.qr(rng.randn(h.shape[1], Q.shape[1]))[0],
                        don_m) for _ in range(n_rand)]
        m_rand = float(np.mean([r[0] for r in mr]))
        mh_rand = float(np.mean([r[1] for r in mr]))
        msens_rand = float(np.mean([r[0] / max(r[1], EPS) for r in mr]))
        out["d_sub_matched"] = m_sub
        out["d_rand_matched"] = m_rand
        out["ratio_matched"] = m_sub / max(m_rand, EPS)
        out["ratio_matched_sens"] = ((m_sub / max(mh_sub, EPS))
                                     / max(msens_rand, EPS))
        out["unmatched_frac"] = float(unmatched) / max(len(trial), 1)
    return out


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
    # The null the whole-state swap never had. The targeted splice scores each
    # subspace against a random subspace of the same RANK, but at full rank
    # that is the whole space, so the control degenerates into the thing it
    # controls for. Instead: shuffle each unit independently across samples.
    # Every unit keeps its exact marginal -- scale, and for a ReLU trunk its
    # sparsity and non-negativity, which a Gaussian null would violate outright
    # -- while the joint pattern across units is destroyed. Same activity,
    # wrongly configured.
    d_shuffle = d(_act(agent, obs, _shuffle_units(h, rng), n_layers, device))

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
        "d_shuffle": d_shuffle,
        "state_influence": d_state / max(d_both, EPS),
        "obs_influence": d_obs / max(d_both, EPS),
        "same_t_influence": d_same_t / max(d_both, EPS),
        "zero_influence": d_zero / max(d_both, EPS),
        "shuffle_influence": d_shuffle / max(d_both, EPS),
        # < 1 means a real foreign state disturbs the policy LESS than
        # structureless activity of the same marginals: the states the agent
        # actually visits lie on a manifold it is comparatively robust to.
        "state_vs_shuffle": d_state / max(d_shuffle, EPS),
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

    # A coarse occupancy MAP -- which G x G blocks have been entered so far.
    # `start_pos` cannot settle the episodic-memory question because it is
    # constant within a trial and so has only n_trials independent samples
    # (§30.6); this varies every step, so it is scored on all T*B of them.
    # §29.2 named it as the signal that was never tested.
    blk = np.minimum((cell * GRID) // size, GRID - 1)
    occ = np.zeros((T, B, GRID * GRID))
    seen_b = np.zeros((B, GRID * GRID), dtype=bool)
    for t in range(T):
        seen_b[np.arange(B), blk[t, :, 0] * GRID + blk[t, :, 1]] = True
        occ[t] = seen_b
    # ... and where the agent WAS. Beyond what the observation says, knowing
    # position k steps back is path history and cannot be anything else.
    # Clamped at the episode start for t < k, which is 10% of rows at k=20.
    lagged = {f"pos_lag{k}": norm[np.maximum(np.arange(T) - k, 0)]
              for k in POS_LAGS}

    prev = np.concatenate([np.zeros((1, B, 2)), act[:-1]], axis=0)
    head = prev / np.maximum(np.linalg.norm(prev, axis=-1, keepdims=True), EPS)

    def flat(a):
        return np.asarray(a, dtype=np.float64).reshape(T * B, -1)

    out = {"pos": flat(norm), "start_pos": flat(start),
           "elapsed": flat(elapsed[..., None]),
           "coverage": flat(cov[..., None]),
           "occupancy": flat(occ),
           "visited8": flat(vis), "heading": flat(head)}
    out.update({k: flat(v) for k, v in lagged.items()})
    return out


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect(*, agent, cfg, envs, vh, offsets, embed_dim, device, args, mode,
            n_distractors):
    """Roll out every env and stack (obs, h, targets, trial, step)."""
    rng = np.random.RandomState(args.seed)
    obs, hid, hcon, trial, step, envi = [], [], [], [], [], []
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
        envi.append(np.full(T * B, i)[keep])
        for k, v in tg.items():
            targets.setdefault(k, []).append(v[keep])
        next_trial += B

    return (np.concatenate(obs), np.concatenate(hid), np.concatenate(hcon),
            {k: np.concatenate(v) for k, v in targets.items()},
            np.concatenate(trial), np.concatenate(step), np.concatenate(envi))


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
    print("    %-12s %4s %8s %8s %8s %9s %9s %10s %10s %10s"
          % ("target", "dim", "eff_n", "R2(obs)", "R2(h)", "R2(both)",
             "deltaR2", "delta_clk", "delta_anc", "delta_flow"))
    for k, v in c.items():
        if k.startswith("_"):
            continue
        print("    %-12s %4d %8d %8.3f %8.3f %9.3f %9.3f %10s %10s %10s"
              % (k, v["dim"], v.get("eff_n", c["_n_samples"]), v["obs"],
                 v["h"], v["both"], v["delta"],
                 "-" if "delta_clk" not in v else "%.3f" % v["delta_clk"],
                 "-" if "delta_anc" not in v else "%.3f" % v["delta_anc"],
                 "-" if "delta_flow" not in v else "%.3f" % v["delta_flow"]))
    print("    A LADDER: each column adds a cheaper explanation to the "
          "baseline and asks")
    print("    what h still adds beyond it.")
    print("    deltaR2   beyond the current observation.")
    print("    delta_clk beyond that AND a perfect clock -- separates spatial "
          "memory from")
    print("              elapsed time (§30.6).")
    print("    delta_anc beyond those AND current position, LINEARLY. Under a "
          "deterministic")
    print("              vector field (§22) THE PAST IS A FUNCTION OF THE "
          "PRESENT, so decoding")
    print("              where you were 20 steps ago is not memory until this "
          "is ruled out.")
    print("    delta_flow beyond those AND any SMOOTH function of position "
          "within this env")
    print("              (random Fourier features x env). The backward flow is "
          "nonlinear and")
    print("              wall-dependent, so this is the rung that actually "
          "means trajectory.")
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
    print("    SHUFFLE units (the null)         %.4f  = %.3f"
          % (u["d_shuffle"], u["shuffle_influence"]))
    print("    state / shuffle                  %.3f"
          % u["state_vs_shuffle"])
    print("      < 1: a real foreign state disturbs the policy LESS than")
    print("      structureless activity with the same per-unit marginals.")
    print("    state_share = state/(state+obs)  %.3f" % u["state_share"])
    if u["lag_curve"]:
        ks = sorted(u["lag_curve"], key=int)
        print("    own-episode donor tau steps back (fraction of scale):")
        print("      tau  " + "".join("%8s" % k for k in ks))
        print("      d    " + "".join("%8.3f" % u["lag_curve"][k] for k in ks))

    # Scored on delta_clk where it exists: a state that is only a clock is not
    # storing anything an explorer can use, and deltaR2 alone would call it
    # content. `elapsed` itself is excluded -- the clock regressor contains it,
    # so its delta_clk is 0 by construction and means nothing.
    sp_ = res.get("splice") or {}
    if sp_:
        print("\n  TARGETED SPLICE -- swap ONE readout subspace, keep the rest")
        print("    %-12s %5s %8s %8s %16s %13s %11s"
              % ("subspace", "rank", "ratio", "size/rnd", "ratio-sens",
                 "matched-sens", "frac_full"))
        for t, v in sp_.items():
            print("    %-12s %5d %8.2f %8.2f %16s %13s %11.3f"
                  % (t, v["rank"], v["ratio"], v.get("size_vs_random", 0.0),
                     "%.2f +/- %.2f" % (v.get("ratio_sens", 0.0),
                                        v.get("ratio_sens_sd", float("nan"))),
                     "-" if "ratio_matched_sens" not in v
                     else "%.2f" % v["ratio_matched_sens"],
                     v["frac_of_full"]))
        print("    size/rnd  how much BIGGER the edit is than a random "
              "subspace's. A readout")
        print("              subspace for something strongly encoded is "
              "high-variance, and a")
        print("              random 2-plane in 1024 dims holds ~2/1024 of the "
              "variance, so the")
        print("              plain ratio partly measures size, not weighting.")
        print("    *-sens    the same ratios with that divided out: action "
              "change per unit of")
        print("              state change. THIS is 'does the policy weight "
              "these directions'.")
        um = [v.get("unmatched_frac") for v in sp_.values()
              if v.get("unmatched_frac") is not None]
        if um:
            print("    ratio-matched draws the donor from a DIFFERENT episode "
                  "at the SAME position.")
            print("    The plain donor is wherever that episode happened to "
                  "be, so splicing the")
            print("    POSITION directions leaves the state saying \"I am at "
                  "B\" while the held-fixed")
            print("    observation still says \"I am at A\". No other subspace "
                  "suffers that, because")
            print("    the observation carries no visitation signal -- so the "
                  "plain ratio flatters")
            print("    position specifically. Matched, the question becomes: "
                  "same place, different")
            print("    history, does the policy tell them apart? "
                  "(%.1f%% of rows had no partner)" % (100.0 * max(um)))
        print("    ratio vs a RANDOM subspace of the same rank. Any "
              "k-dimensional")
        print("    perturbation moves the action, so ratio > 1 is the claim, "
              "not d_sub.")
        print("    ratio-pos repeats it with the POSITION directions projected "
              "out. The")
        print("    decoders overlap -- visitation near a wall is partly a "
              "statement about")
        print("    where you are -- and every arm reads position at ~10x, so a "
              "subspace can")
        print("    inherit its punch from position. This is the number that "
              "does not.")

    rows = {k: v for k, v in c.items() if not k.startswith("_")}
    col, skip = "delta", set()
    if any("delta_clk" in v for v in rows.values()):
        col, skip = "delta_clk", {"elapsed"}
    if any("delta_anc" in v for v in rows.values()):
        col, skip = "delta_anc", {"elapsed", "pos"}
    if any("delta_flow" in v for v in rows.values()):
        col, skip = "delta_flow", {"elapsed", "pos"}
    best_k, best = "", 0.0
    for k, v in rows.items():
        # A target that IS a baseline rung scores 0 there by construction.
        if k in skip:
            continue
        if v[col] > best:
            best_k, best = k, v[col]
    stored, used = best > STORED, u["state_influence"] > USED
    print("\n  VERDICT: content %s (best %s %.3f on %r), use %s "
          "(state_influence %.3f)"
          % ("YES" if stored else "no", col, best, best_k or "-",
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
                       ("delta_clk -- beyond obs AND a perfect clock",
                        "delta_clk"),
                       ("delta_anc -- beyond those AND current position",
                        "delta_anc"),
                       ("delta_flow -- beyond any smooth f(position) in-env",
                        "delta_flow"),
                       ("R2(both) -- total decodability", "both")):
        if not all(col in out[n]["content"][t]
                   for n in names for t in targets):
            continue
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
                   ("shuffle_influence", "shuffle (null)"),
                   ("state_vs_shuffle", "state / shuffle"),
                   ("obs_influence", "obs_influence"),
                   ("state_share", "state_share")):
        # .get, not [k]: a dump from before a metric existed should still
        # render the rows it does have rather than failing to render at all.
        if not any(k in out[n]["use"] for n in names):
            continue
        print("    %-16s" % lab
              + "".join("%17.3f" % out[n]["use"].get(k, np.nan)
                        for n in names))
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
    p.add_argument("--splice_targets", nargs="*",
                   default=["occupancy", "visited8", "pos"],
                   help="TARGETED SPLICE. The whole-state swap says the state "
                        "matters, never WHICH content matters -- it replaces "
                        "position, clock and map at once. For each target "
                        "named here the probe fits a linear readout subspace "
                        "of h, swaps only that subspace for a donor "
                        "episode's, and leaves the orthogonal complement "
                        "intact. Scored against a RANDOM subspace of the same "
                        "rank, without which the number is uninterpretable. "
                        "Empty to skip.")
    p.add_argument("--match_heading", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Match the splice donor on HEADING as well as "
                        "position. Matching position alone left the position "
                        "subspace moving the action ~10x between two agents "
                        "in the same cell -- those directions carry heading "
                        "too, which the observation also encodes, so a "
                        "heading mismatch reintroduces the very contradiction "
                        "the match exists to remove. With both matched, what "
                        "differs between donor and recipient is HISTORY, "
                        "which is the only thing the observation does not "
                        "carry.")
    p.add_argument("--match_cell", type=float, default=1.0,
                   help="Arena units per matching bin. Larger finds more "
                        "partners and matches them more loosely; the "
                        "unmatched fraction is reported either way.")
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
        obs, hid, hcon, targets, trial, step, envi = collect(
            agent=agent, cfg=own, envs=envs, vh=vh, offsets=offsets,
            embed_dim=embed_dim, device=device, args=args, mode=args.mode,
            n_distractors=args.n_distractors)
        rng = np.random.RandomState(args.seed)
        res = {
            "hidden": int(hid.shape[1]),
            "obs_dim": int(obs.shape[1]),
            "content_h": args.content_h,
            "content": content_probes(obs, hcon, targets, trial, rng,
                                      args.test_frac,
                                      clock=targets["elapsed"],
                                      anchor=targets["pos"], env=envi),
            "use": use_probes(agent, obs, hid, trial, step,
                              own.agent.num_rnn_layers, device, rng,
                              tuple(args.lags)),
        }
        # Targeted splice: fit the readout subspace on TRAIN trials, then swap
        # only that subspace on the HELD-OUT ones, so the directions were not
        # chosen against the states they are tested on.
        if args.splice_targets:
            str_, ste = _split_trials(trial, args.test_frac,
                                      np.random.RandomState(args.seed))
            res["splice"] = {}
            qpos = _readout_subspace(hid[str_], targets["pos"][str_])
            for t in args.splice_targets:
                if t not in targets:
                    print(f"  NOTE no target {t!r}; skipping its splice.")
                    continue
                Q = _readout_subspace(hid[str_], targets[t][str_])
                row = subspace_splice(
                    agent, obs[ste], hid[ste], trial[ste], Q,
                    own.agent.num_rnn_layers, device,
                    np.random.RandomState(args.seed),
                    pos=targets["pos"][ste],
                    heading=(targets["heading"][ste]
                             if args.match_heading else None),
                    cell=args.match_cell)
                if t != "pos":
                    # ... and again with the position directions removed.
                    Qr = _orth_against(Q, qpos)
                    r = subspace_splice(
                        agent, obs[ste], hid[ste], trial[ste], Qr,
                        own.agent.num_rnn_layers, device,
                        np.random.RandomState(args.seed))
                    row["rank_resid"] = r["rank"]
                    row["d_sub_resid"] = r["d_sub"]
                    row["ratio_resid"] = r["ratio"]
                    row["frac_resid"] = r["frac_of_full"]
                res["splice"][t] = row
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
