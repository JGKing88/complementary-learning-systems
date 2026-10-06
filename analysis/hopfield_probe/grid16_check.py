"""The 16-condition grid: encoder x saturation x storage x readout (EXPERIMENTS_IDEAL_ENCODER Sec 18).

Memory conditions (8, plus 4 side rows):

    encoder   ideal r=32 (``ideal:r=32,n_freq=512,seed=0,gain=100``) | att0.5 s42
    sat       unsat = production (encoder gain 100, recall beta = 100; the ideal
              code has no tanh) | sat = encoder output binarised AND recall
              saturated (att0.5: gain = beta = 1e6, "arm B"; ideal: binary=1,
              beta = 1e6) | side row "rsat" = recall-only saturation (beta 1e6,
              encoder unsaturated)
    storage   hebb | proj

Navigation conditions: x readout {(a) q = basis.(recalled - current),
(b) central-difference gradient of s(p) = <zhat, z(p)> (THEORY Sec 7.1
"(iii-c)", the readout of ``potential_readout_check.py``)} x recall timing
{one step at alpha = 1 (production), the converged walk at the condition's
best alpha}.

One task = one (memory condition, layout, K). Layout is the whole arena or the
500-cell region ``--world_region 0 0 500``; memory mode is multi_env_goals.

What each part measures, and the decisions behind it:

1. **alpha sweep** (2 worlds x 5 scored envs x every non-goal cell). Iterate
   the recall at each alpha for up to ``T_MAX`` steps (early stop once every
   cue moves < 1e-6 per step), decode each step to its nearest bank cell (the
   K envs' cells + 5000 alias cells), and record decoded distance to the goal
   and cos(state, that cell's code). Per cue the *inbound* segment runs from
   the start to the first step at which the decoded distance is minimal; the
   drift after arrival is the fixed-point question (2-3), not this one.
   ``interpolates`` = >= 90% of cues monotone non-increasing on the inbound
   segment AND mean (over cues) of the inbound min cos >= 0.95 AND the median
   largest single-step drop <= 50% of the distance covered (a one-step jump is
   not an interpolation) AND mean closest approach <= 1 cell. ``snap`` = median
   largest drop >= 80% of the path, taken after >= 2 steps, with an inbound
   cos dip below 0.95. Best alpha: see ``pick_alpha`` -- among interpolating
   alphas if any (tag "interpolating"), else the alpha closest to
   interpolating, i.e. the least snap (tag "closest").
2. **fixed point**: ``self_fixed_point`` at alpha = 1 and at the best alpha,
   8 worlds x K stored goals: does self-recall STOP (cos of the last two
   states > 0.99999 within T_MAX), and is the stopped state the goal's own
   cell (decoded against the cell bank)? If not: cells off and cos to the
   goal's code. ``attractor.fixed_point_probe``'s cos-to-self at steps 1, 5,
   15, 30 is kept alongside.
3. **correct fixed point and basin (ii)**: every scaffold cell within
   ``BASIN_R`` of the goal is a cue, iterated at the best alpha to convergence,
   decoded over the disc + the other stored goals (``attractor.basin_bank``).
   Near-goal starts are d <= 2; the fixed point proper is the start at the goal
   cell. Basin = ``first_failure_radius`` at 100% and 95%. One-step alpha = 1
   is recorded alongside.
4. **navigation**: per env (8 worlds x 5 envs) acc45, mean |err| and the
   ``flow.continuous_flow`` reach rate (arrival within 0.5 cell) of each
   readout. Basin (iii) cannot be read off a 20-cell env (censored at its
   diagonal), so it is measured on a goal-centred (2*BOX_R+1)^2 box of scaffold
   cells, flowing with the same ``continuous_flow`` (box edge = wall), as the
   first-failure radius of "reached" over start distance <= BOX_R.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from analysis.hopfield_probe.attractor import (basin_bank,
                                               first_failure_radius,
                                               fixed_point_probe)
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.flow import continuous_flow, discrete_flow
from analysis.hopfield_probe.harness import (ProbeConfig, build_cell_bank,
                                             build_memory, load_probe_encoder,
                                             local_cells, sample_worlds,
                                             scored_envs)
from analysis.hopfield_probe.qfield import project_q
from analysis.hopfield_probe.stats import wrap_to_pi

PROD = ("/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/"
        "000_att0.5_seed=42/encoder_final.pt")
IDEAL = "ideal:r=32,n_freq=512,seed=0,gain=100"

ENCODERS = {
    # name: {sat: (path, encoder gain, beta)}
    "ideal": {"unsat": (IDEAL, 100.0, 100.0),
              "sat": (IDEAL + ",binary=1", 100.0, 1e6),
              "rsat": (IDEAL, 100.0, 1e6)},
    "att0.5": {"unsat": (PROD, 100.0, 100.0),
               "sat": (PROD, 1e6, 1e6),
               "rsat": (PROD, 100.0, 1e6)},
}
CONDITIONS = [(e, s, st) for s in ("unsat", "sat", "rsat")
              for e in ("ideal", "att0.5") for st in ("hebb", "proj")]
LAYOUTS = {"whole": None, "region": (0, 0, 500)}
KS = (5, 20)
TASKS = [(c, lay, k) for c in CONDITIONS for lay in LAYOUTS for k in KS]

ALPHAS = (1.0, 0.95, 0.9, 0.8, 0.5, 0.2, 0.05, 0.01, 0.003, 0.001)
T_MAX = 600
CONV_TOL = 1e-6
DECODE = sorted(set(range(1, 101)) | set(range(102, 201, 2))
                | set(range(205, 401, 5)) | set(range(410, T_MAX + 1, 10)))
SHOW = (1, 2, 3, 5, 8, 12, 20, 30, 60, 100, 200, 400, 600)
FP_STEPS = (1, 5, 15, 30)
BASIN_R = 64
BOX_R = 40
N_BOX = 8
FP_COS = 0.99999        # "stopped": cos between the last two states
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))      # (East, North) order


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-30)


# ---------------------------------------------------------------------------
# Recall with early stopping (same update as Hopfield.recall_batch_trajectory)
# ---------------------------------------------------------------------------

def iterate(mem, X0: np.ndarray, alpha: float, snaps, *, chunk: int = 8192,
            callback=None):
    """Recall at ``alpha`` with snapshots at ``snaps``.

    Returns ``{step: state}`` for every snapshot (unless ``callback(step, a, b,
    X, repeat)`` is given, which then receives each snapshot of rows ``a:b``
    instead of storing it; ``repeat`` is True for snapshots after the early
    stop, which all equal the last computed state), plus ``"conv"`` (bool per cue: moved < CONV_TOL on the last
    step taken) and ``"steps_run"``.

    The update is exactly ``Hopfield.recall_batch_trajectory``'s,
    ``x <- normalize((1-a) x + a tanh(beta W x))``. Once every cue in a chunk
    moves less than ``CONV_TOL`` the chunk stops and its remaining snapshots
    repeat the last state -- the only approximation, at float32 resolution.
    """
    hop = mem.hopfield
    W, beta = hop.W, float(hop.beta)
    snaps = sorted(set(int(s) for s in snaps))
    top = snaps[-1]
    want = set(snaps)
    out = {} if callback else {s: np.empty((X0.shape[0], X0.shape[1]),
                                           dtype=np.float32) for s in snaps}

    def emit(s, a, b, X, repeat=False):
        if callback:
            callback(s, a, b, X, repeat)
        else:
            out[s][a:b] = X

    conv = np.zeros(X0.shape[0], dtype=bool)
    ran = 0
    for a in range(0, X0.shape[0], chunk):
        b = min(a + chunk, X0.shape[0])
        X = torch.from_numpy(np.ascontiguousarray(X0[a:b])).float()
        t, move = 0, None
        for t in range(1, top + 1):
            Xn = (1 - alpha) * X + alpha * torch.tanh(beta * (X @ W.T))
            Xn = F.normalize(Xn, dim=-1)
            move = (Xn - X).norm(dim=-1)
            X = Xn
            if t in want:
                emit(t, a, b, X.numpy())
            if float(move.max()) < CONV_TOL:
                break
        conv[a:b] = (move < CONV_TOL).numpy()
        ran = max(ran, t)
        Xf = X.numpy()
        for s in snaps:
            if s > t:
                emit(s, a, b, Xf, True)
    out["conv"] = conv
    out["steps_run"] = ran
    return out


class Decoder:
    """Nearest bank row and its cosine, with the bank normalised once."""

    def __init__(self, Z: np.ndarray):
        self.B = torch.from_numpy(_unit(Z).astype(np.float32))

    def __call__(self, X: np.ndarray, chunk: int = 4096):
        idx = np.empty(X.shape[0], dtype=np.int64)
        val = np.empty(X.shape[0], dtype=np.float32)
        for a in range(0, X.shape[0], chunk):
            x = torch.from_numpy(_unit(X[a:a + chunk]).astype(np.float32))
            c = x @ self.B.T
            v, i = c.max(dim=1)
            idx[a:a + chunk] = i.numpy()
            val[a:a + chunk] = v.numpy()
        return idx, val


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def make_field(path, gain, cfg):
    enc, ecfg, _g, fwhm, header = load_probe_encoder(path, fwhm_fallback=0.25)
    enc.gain = float(gain)
    field = Field(enc, list(ecfg.lambdas), fwhm, float(gain), cfg.Npos)
    return field, header


def make_cfg(beta, storage, region, k, n_worlds=8, alpha=1.0, steps=(1,)):
    return ProbeConfig(n_worlds=n_worlds, n_envs_per_world=20, env_size=20,
                       Npos=1716, k_values=(k,), steps=tuple(steps), seed=0,
                       basin_radius=BASIN_R, storage_rule=storage,
                       beta_override=float(beta), world_region=region,
                       n_alias=5000, alpha=float(alpha))


# ---------------------------------------------------------------------------
# 1. alpha sweep
# ---------------------------------------------------------------------------

def alpha_walk(field, worlds, cfg, k, alpha):
    dists, coss, inenv, start, final_cos_goal, conv = [], [], [], [], [], []
    steps_run = 0
    col = {s: j for j, s in enumerate(DECODE)}
    cells = local_cells(cfg.env_size)
    for w in worlds:
        rng = np.random.RandomState(w.seed * 31 + k)
        mem = build_memory(field, w, k, cfg, rng)
        bank = build_cell_bank(field, w, k, cfg, rng)
        dec = Decoder(bank.Z)
        for e in scored_envs(cfg, k):
            off = w.specs[e].offset
            goal = np.array(w.specs[e].goal)
            gz = mem.Z[mem.owner == e][0]
            d0 = np.sqrt(((cells - goal) ** 2).sum(1))
            keep = d0 > 0
            cues = field.encode(cells[keep, 0] + off[0], cells[keep, 1] + off[1])
            n = cues.shape[0]
            D = np.empty((n, len(DECODE)))
            C = np.empty_like(D)
            E = np.empty((n, len(DECODE)), dtype=bool)
            fin = np.empty(n)
            last = {}

            def cb(s, a, b, X, repeat):
                if repeat and a in last:          # converged: copy the decode
                    j0 = col[last[a]]
                    D[a:b, col[s]], C[a:b, col[s]], E[a:b, col[s]] = \
                        D[a:b, j0], C[a:b, j0], E[a:b, j0]
                else:
                    idx, val = dec(X)
                    r_env, r_x, r_y = bank.decode(idx)
                    same = r_env == e
                    d = np.sqrt((r_x - goal[0]) ** 2 + (r_y - goal[1]) ** 2)
                    D[a:b, col[s]] = np.where(same, d, np.inf)
                    C[a:b, col[s]] = val
                    E[a:b, col[s]] = same
                    last[a] = s
                if s == DECODE[-1]:
                    fin[a:b] = _unit(X) @ gz

            tr = iterate(mem, cues, alpha, DECODE, callback=cb)
            steps_run = max(steps_run, tr["steps_run"])
            dists.append(D)
            coss.append(C)
            inenv.append(E)
            start.append(d0[keep])
            final_cos_goal.append(fin)
            conv.append(tr["conv"])
    return walk_stats(np.concatenate(dists), np.concatenate(coss),
                      np.concatenate(inenv), np.concatenate(start),
                      np.concatenate(final_cos_goal), np.concatenate(conv),
                      steps_run, alpha)


def walk_stats(D, C, E, d0, cos_goal_final, conv, steps_run, alpha):
    n, T = D.shape
    full = np.concatenate([d0[:, None], D], axis=1)        # step 0 = the cue
    tstar = np.argmin(full, axis=1)                          # first minimum
    dmin = full[np.arange(n), tstar]
    cols = np.arange(T + 1)[None, :]
    inb = cols <= tstar[:, None]
    diffs = np.diff(full, axis=1)                            # (n, T)
    inb_d = inb[:, 1:]
    with np.errstate(invalid="ignore"):
        bad = (diffs > 1e-6) & inb_d
    monotone = ~bad.any(1)
    drops = np.where(inb_d, -diffs, -np.inf)
    drops = np.where(np.isfinite(drops), drops, -np.inf)
    big = drops.max(1)
    path = np.maximum(d0 - dmin, 1e-12)
    jump = np.where(np.isfinite(dmin) & (d0 - dmin > 0), big / path, np.nan)
    jump_at = np.array(DECODE)[np.argmax(drops, axis=1)]
    Cin = np.where(inb_d, C, np.inf)
    mincos_in = np.where(tstar > 0, Cin.min(1), np.nan)
    final_d = D[:, -1]
    traj_d = [float(D[np.isfinite(D[:, j]), j].mean())
              if np.isfinite(D[:, j]).any() else None for j in range(T)]
    traj_c = C.mean(0)
    first_hit = np.where((D == 0).any(1), np.array(DECODE)[np.argmax(D == 0, 1)],
                         -1)
    st = {
        "alpha": alpha,
        "n_cues": int(n),
        "start_mean": float(d0.mean()),
        "frac_monotone_in": float(monotone.mean()),
        "mincos_in_mean": float(np.nanmean(mincos_in)),
        "mincos_in_p10": float(np.nanpercentile(mincos_in, 10)),
        "mincos_whole_mean": float(C.min(1).mean()),
        "jump_median": float(np.nanmedian(jump)),
        "jump_mean": float(np.nanmean(np.clip(jump, 0, 1))),
        "jump_step_median": float(np.median(jump_at[np.isfinite(jump)]))
        if np.isfinite(jump).any() else None,
        "dmin_mean": float(np.mean(np.where(np.isfinite(dmin), dmin, 99.0))),
        "frac_reach_goal": float(np.mean(dmin == 0)),
        "first_hit_median": float(np.median(first_hit[first_hit > 0]))
        if (first_hit > 0).any() else None,
        "final_exact": float(np.mean(final_d == 0)),
        "final_dist_mean": float(np.mean(final_d[np.isfinite(final_d)]))
        if np.isfinite(final_d).any() else None,
        "final_in_env": float(np.mean(np.isfinite(final_d))),
        "final_cos_goal": float(np.mean(cos_goal_final)),
        "frac_converged": float(conv.mean()),
        "steps_run": int(steps_run),
        "traj": {str(s): {"dist": traj_d[DECODE.index(s)],
                          "cos": float(traj_c[DECODE.index(s)]),
                          "in_env": float(E[:, DECODE.index(s)].mean())}
                 for s in SHOW if s in DECODE},
    }
    st["interpolates"] = bool(st["frac_monotone_in"] >= 0.9
                              and st["mincos_in_mean"] >= 0.95
                              and st["jump_median"] <= 0.5
                              and st["dmin_mean"] <= 1.0)
    st["snap"] = bool(alpha < 1 and st["jump_median"] >= 0.8
                      and (st["jump_step_median"] or 0) >= 2
                      and st["mincos_in_mean"] < 0.95)
    return st


def pick_alpha(sweep: list[dict]) -> tuple[dict, str]:
    """``(best, how)``, ``how`` in {"interpolating", "closest"}.

    (a) Some alpha interpolates: among those, the most final states exactly on
    the goal, then the higher inbound min cos, then the larger alpha.
    (b) None does: the alpha that comes CLOSEST to interpolating -- the least
    snap, i.e. the smallest median largest-single-step fraction, then the
    smallest mean of it (the median is 1.00 for every saturated alpha; the mean
    still separates a walk that sometimes takes two steps from a one-step
    jump), then the fraction ending exactly on the goal, then min cos.
    Candidates are restricted to alphas whose walk gets in (mean closest
    approach <= 1 cell) when any do: one that never leaves the cue has no
    large step either, and is not "close to interpolating". alpha = 1 wins
    only if it genuinely jumps least -- not, as under the old rule, because a
    one-step jump has inbound min cos 1.000.
    """
    interp = [s for s in sweep if s["interpolates"]]
    if interp:
        return max(interp, key=lambda s: (round(s["final_exact"], 2),
                                          round(s["mincos_in_mean"], 3),
                                          s["alpha"])), "interpolating"
    arrive = [s for s in sweep if s["dmin_mean"] <= 1.0] or sweep

    def j(v):
        return 1.0 if v is None or not np.isfinite(v) else v

    return min(arrive, key=lambda s: (round(j(s["jump_median"]), 2),
                                      round(j(s.get("jump_mean")), 3),
                                      -round(s["final_exact"], 2),
                                      -round(s["mincos_in_mean"], 3))), "closest"


# ---------------------------------------------------------------------------
# 3. disc probe: correct fixed point + basin (ii)
# ---------------------------------------------------------------------------

def self_fixed_point(field, worlds, cfg, k, alpha):
    """Self-recall from every stored goal, run to T_MAX (early stop).

    ``exists``: the state has stopped -- cos between the last two states
    > FP_COS (the early stop at per-step change < CONV_TOL implies it).
    ``correct``: that end state decodes exactly to its own goal cell, against
    the world's cell bank (the K envs' cells + 5000 alias cells). For the rest,
    ``off`` is the decoded distance from the goal (cells, same env only),
    ``other_env`` the fraction decoding outside the goal's env, and
    ``cos_goal`` the end state's cosine to the goal's code.
    """
    exists, correct, off, other, cosg = [], [], [], [], []
    for w in worlds:
        rng = np.random.RandomState(w.seed * 31 + k)
        mem = build_memory(field, w, k, cfg, rng)
        bank = build_cell_bank(field, w, k, cfg, rng)
        tr = iterate(mem, mem.Z, alpha, (T_MAX - 1, T_MAX))
        x0, x1 = _unit(tr[T_MAX - 1]), _unit(tr[T_MAX])
        exists.append((x0 * x1).sum(1) > FP_COS)
        idx, _v = Decoder(bank.Z)(x1)
        r_env, r_x, r_y = bank.decode(idx)
        goals = w.goals()[:k]
        owner = mem.owner
        same = r_env == owner
        d = np.hypot(r_x - goals[owner, 0], r_y - goals[owner, 1])
        correct.append(same & (d == 0))
        off.append(np.where(same, d, np.nan))
        other.append(~same)
        cosg.append((x1 * mem.Z).sum(1))
    exists, correct = np.concatenate(exists), np.concatenate(correct)
    off, other, cosg = (np.concatenate(off), np.concatenate(other),
                        np.concatenate(cosg))
    bad = ~correct
    return {
        "n": int(exists.size),
        "exists": float(exists.mean()),
        "correct": float(correct.mean()),
        "off_mean_wrong": float(np.nanmean(off[bad]))
        if np.isfinite(off[bad]).any() else None,
        "off_mean_all": float(np.nanmean(off)) if np.isfinite(off).any() else None,
        "other_env": float(other.mean()),
        "cos_goal": float(cosg.mean()),
        "cos_goal_wrong": float(cosg[bad].mean()) if bad.any() else None,
    }


def disc_probe(field, w, env, mem, cfg, alpha):
    R = BASIN_R
    off, goal = w.specs[env].offset, w.specs[env].goal
    gx0, gy0 = goal[0] + off[0], goal[1] + off[1]
    a = np.arange(-R, R + 1)
    dx, dy = np.meshgrid(a, a, indexing="ij")
    dx, dy = dx.ravel(), dy.ravel()
    d = np.hypot(dx, dy)
    cx, cy = gx0 + dx, gy0 + dy
    keep = ((d <= R) & (cx >= 0) & (cx < cfg.Npos) & (cy >= 0)
            & (cy < cfg.Npos))
    dx, dy, d, cx, cy = dx[keep], dy[keep], d[keep], cx[keep], cy[keep]
    cells = field.encode(cx, cy)
    bank = basin_bank(cells, mem, env)
    goal_row = int(np.flatnonzero(d == 0)[0])
    zg = _unit(cells[goal_row])
    dec = Decoder(bank)

    res = {}
    for tag, al, steps in (("one_step", 1.0, (1,)), ("converged", alpha, (T_MAX,))):
        tr = iterate(mem, cells, al, steps)
        x = tr[steps[-1]]
        won, val = dec(x)
        hit = won == goal_row
        on_cell = won < cells.shape[0]
        land = np.where(on_cell, d[np.clip(won, 0, len(d) - 1)], np.nan)
        cg = _unit(x) @ zg
        near = d <= 2
        res[tag] = {
            "r_exact_all": first_failure_radius(hit, d, 1.0, max_r=R),
            "r_exact_95": first_failure_radius(hit, d, 0.95, max_r=R),
            "exact_frac_disc": float(hit.mean()),
            "near_exact": float(hit[near].mean()),
            "near_land_dist": float(np.nanmean(land[near]))
            if np.isfinite(land[near]).any() else None,
            "near_other_goal": float(np.mean(~on_cell[near])),
            "near_cos_goal": float(cg[near].mean()),
            "self_exact": bool(hit[goal_row]),
            "self_land_dist": None if not on_cell[goal_row]
            else float(land[goal_row]),
            "self_cos_goal": float(cg[goal_row]),
            "self_cos_nearest": float(val[goal_row]),
            "frac_converged": float(tr["conv"].mean()),
            "steps_run": int(tr["steps_run"]),
        }
    return res


# ---------------------------------------------------------------------------
# 4. readouts and flows
# ---------------------------------------------------------------------------

def readouts(field, cells, off, cues, zhat):
    """(a) the production q and (b) the (iii-c) central difference.

    Both copied from ``potential_readout_check.one_encoder`` (which itself uses
    ``qfield.project_q`` for (a)); (b) is THEORY Sec 7.1 step 5 verbatim.
    """
    zhat = _unit(zhat)
    q_a = project_q(field.local_basis(cells, off), cues, zhat)
    sn = {n: np.einsum("id,id->i", zhat,
                       _unit(field.encoded_state(cells + np.array(n), off)))
          for n in NEIGHBOURS}
    q_b = 0.5 * np.stack([sn[(1, 0)] - sn[(-1, 0)],
                          sn[(0, 1)] - sn[(0, -1)]], axis=1)
    return {"a": q_a, "b": q_b}


def _angle_err(q, cells, goal):
    dx = goal[0] - cells[:, 0]
    dy = goal[1] - cells[:, 1]
    keep = (dx != 0) | (dy != 0)
    err = np.abs(wrap_to_pi(np.arctan2(q[keep, 1], q[keep, 0])
                            - np.arctan2(dy[keep], dx[keep])))
    return np.degrees(err)


def nav_envs(field, worlds, cfg, k, alpha):
    size = cfg.env_size
    cells = local_cells(size)
    acc = {(r, t): {"err": [], "reach": [], "reach_disc": [], "r_env": []}
           for r in "ab" for t in ("one_step", "converged")}
    for w in worlds:
        rng = np.random.RandomState(w.seed * 13 + k)
        mem = build_memory(field, w, k, cfg, rng)
        for e in scored_envs(cfg, k):
            off, goal = w.specs[e].offset, w.specs[e].goal
            cues = field.encoded_state(cells, off)
            recalls = {"one_step": iterate(mem, cues, 1.0, (1,))[1],
                       "converged": iterate(mem, cues, alpha, (T_MAX,))[T_MAX]}
            d0 = np.sqrt(((cells - np.array(goal)) ** 2).sum(1))
            for t, zhat in recalls.items():
                for r, q in readouts(field, cells, off, cues, zhat).items():
                    A = acc[(r, t)]
                    A["err"].append(_angle_err(q, cells, goal))
                    c = continuous_flow(q, size, goal, cfg)
                    A["reach"].append(c["reach_rate"])
                    A["reach_disc"].append(discrete_flow(q, size, goal)
                                           ["reach_rate"])
                    A["r_env"].append(first_failure_radius(c["reached"], d0, 1.0))
    out = {}
    for (r, t), A in acc.items():
        err = np.concatenate(A["err"])
        out[f"{r}|{t}"] = {"acc45": float(np.mean(err < 45)),
                           "abs_err": float(err.mean()),
                           "reach": float(np.mean(A["reach"])),
                           "reach_disc": float(np.mean(A["reach_disc"])),
                           "r_env_mean": float(np.mean(A["r_env"])),
                           "n_envs": len(A["reach"])}
    return out


def nav_box(field, w, env, mem, cfg, alpha):
    """Basin (iii) on a goal-centred box of scaffold cells."""
    size = 2 * BOX_R + 1
    off0, goal0 = w.specs[env].offset, w.specs[env].goal
    gx, gy = goal0[0] + off0[0], goal0[1] + off0[1]
    off = (gx - BOX_R, gy - BOX_R)
    if min(off) < 2 or max(off) + size > cfg.Npos - 2:
        return None
    goal = (BOX_R, BOX_R)
    cells = local_cells(size)
    d0 = np.sqrt(((cells - np.array(goal)) ** 2).sum(1))
    cues = field.encoded_state(cells, off)
    bcfg = dataclasses.replace(cfg, env_size=size)
    recalls = {"one_step": iterate(mem, cues, 1.0, (1,))[1],
               "converged": iterate(mem, cues, alpha, (T_MAX,))[T_MAX]}
    disc = d0 <= BOX_R
    out = {}
    for t, zhat in recalls.items():
        for r, q in readouts(field, cells, off, cues, zhat).items():
            c = continuous_flow(q, size, goal, bcfg)
            reached = c["reached"]
            out[f"{r}|{t}"] = {
                "r_reach_all": first_failure_radius(reached[disc], d0[disc],
                                                    1.0, max_r=BOX_R),
                "r_reach_95": first_failure_radius(reached[disc], d0[disc],
                                                   0.95, max_r=BOX_R),
                "reach_disc_frac": float(reached[disc].mean()),
            }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_task(i: int, out_dir: str, alphas=ALPHAS) -> str:
    (enc_name, sat, storage), layout, k = TASKS[i]
    path, gain, beta = ENCODERS[enc_name][sat]
    region = LAYOUTS[layout]
    label = f"{enc_name}|{sat}|{storage}|{layout}|K={k}"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {label}: {msg}", flush=True)

    cfg = make_cfg(beta, storage, region, k)
    field, header = make_field(path, gain, cfg)
    worlds = sample_worlds(cfg)
    res = {"label": label, "encoder": enc_name, "sat": sat, "storage": storage,
           "layout": layout, "k": k, "path": path, "gain": gain, "beta": beta,
           "header": header, "T_MAX": T_MAX, "BASIN_R": BASIN_R, "BOX_R": BOX_R}

    sweep = []
    for a in alphas:
        st = alpha_walk(field, worlds[:2], cfg, k, a)
        sweep.append(st)
        log(f"alpha {a:<6g} interp={st['interpolates']} snap={st['snap']} "
            f"mono {st['frac_monotone_in']:.2f} mincos {st['mincos_in_mean']:.3f} "
            f"jump {st['jump_median']:.2f} dmin {st['dmin_mean']:.2f} "
            f"final exact {st['final_exact']:.3f} conv {st['frac_converged']:.2f}"
            f" steps {st['steps_run']}")
    res["sweep"] = sweep
    best, how = pick_alpha(sweep)
    alpha = best["alpha"]
    res["best_alpha"] = alpha
    res["best_alpha_how"] = how
    log(f"best alpha {alpha} ({how})")

    res["fixed_point"] = {}
    for tag, al, steps in (("alpha1", 1.0, FP_STEPS),
                           ("best", alpha, FP_STEPS + (T_MAX,))):
        c = make_cfg(beta, storage, region, k, alpha=al, steps=steps)
        pooled = {s: [] for s in steps}
        for w in worlds:
            mem = build_memory(field, w, k, c, np.random.RandomState(w.seed * 31 + k))
            fp = fixed_point_probe(mem, c)
            for s in steps:
                pooled[s].append(fp[str(s)]["cos_self_mean"])
        res["fixed_point"][tag] = {str(s): float(np.mean(v))
                                   for s, v in pooled.items()}
    log(f"fixed point {res['fixed_point']}")

    res["self_fixed"] = {}
    for tag, al in (("alpha1", 1.0), ("best", alpha)):
        res["self_fixed"][tag] = self_fixed_point(field, worlds, cfg, k, al)
        log(f"self fixed point {tag} (alpha {al}): {res['self_fixed'][tag]}")

    disc = []
    for w in worlds[:4]:
        mem = build_memory(field, w, k, cfg, np.random.RandomState(w.seed * 31 + k))
        for e in scored_envs(cfg, k)[:2]:
            disc.append(disc_probe(field, w, e, mem, cfg, alpha))
    res["disc"] = disc
    log("disc " + " ".join(f"{t}: r_all {np.mean([d[t]['r_exact_all'] for d in disc]):.1f}"
                           f" near_exact {np.mean([d[t]['near_exact'] for d in disc]):.2f}"
                           for t in ("one_step", "converged")))

    res["nav_env"] = nav_envs(field, worlds, cfg, k, alpha)
    log(f"nav env {json.dumps(res['nav_env'])}")

    # Up to N_BOX goals, scanning worlds and scored envs in order and skipping
    # boxes that do not fit inside the scaffold. (The first run took only
    # worlds[:4] x envs[:2]; in the 500-cell region only 1 of those 8 fit.)
    box = []
    for w in worlds:
        if len(box) >= N_BOX:
            break
        mem = build_memory(field, w, k, cfg, np.random.RandomState(w.seed * 13 + k))
        for e in scored_envs(cfg, k):
            if len(box) >= N_BOX:
                break
            b = nav_box(field, w, e, mem, cfg, alpha)
            if b is not None:
                box.append(b)
    res["nav_box"] = box
    log("box " + " ".join(f"{key}: {np.mean([b[key]['r_reach_all'] for b in box]):.1f}"
                          for key in box[0]) if box else "box: none fit")

    os.makedirs(out_dir, exist_ok=True)
    slug = label.replace("|", "__").replace("=", "")
    p = os.path.join(out_dir, f"{slug}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"wrote {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, required=True,
                    help=f"task 0..{len(TASKS) - 1}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    run_task(args.index, args.out, tuple(args.alphas))


if __name__ == "__main__":
    main()
