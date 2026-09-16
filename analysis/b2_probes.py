"""Representation probes for the lattice-randomised agents (plan sec 6.2).

    python -m analysis.b2_probes --probe all --runs frozen_s0,S1,S2,B3-2,raw,rec
    python -m analysis.b2_probes --probe delta --runs frozen_s0,dist90,A1

Synthetic lattices only (`gbook_at`; no scaffold): every row of a lifetime
gets its own orientation theta and translation, as in training. Four probes:

  delta   static. Linear (ridge) readout from the memoryless features -- the
          encoder's output, or the trunk of a pairs model -- to the code-frame
          displacement direction R_theta (g - p), to the env-frame direction
          (g - p) (not recoverable from two codes: the control), and to the
          per-module wrapped phase differences. Fit at |Delta| <= 19, then
          tested by |Delta| beyond it. Also from the GRU state, by episode,
          from the lifetimes of `theta`: where the un-rotated direction lives.
  theta   lifetimes at theta ~ U(-180, 180): ridge from the GRU state to
          (cos theta, sin theta), fit on the states of later episodes of half
          the rows, read on the other half by step of episode 0 and by
          episode -- the frame estimate, if it is explicit, and its
          sharpening. Trained band and the held-out band |theta| < 15 deg
          separately.
  swap    lifetimes at theta_1; at the start of episode E the code switches
          to theta_2 = theta_1 + 90 deg, h kept. Signed error by step after
          the switch: a decoupled estimator keeps un-rotating by theta_1, so
          the action is off by the swap angle until it re-measures.
  act     the previous-action channel is rotated by +90 deg for ONE step
          (episode 0, step t, for t in a few values; and once in a late
          episode). Signed error at and after that step against unperturbed
          rows with the same seed, starts, goals and lattices.

The rollouts are sampled (the evaluation convention); the readouts are
linear so what is reported is what a linear head could use.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from cls_paths import checkpoints_dir
from gridcode.lattice import gbook_at, module_phases, rotation, wrapped_phase_diff
from hopfield_nav.config import EnvConfig
from hopfield_nav.evaluation.lifetime import evaluate_lifetime_direction
from hopfield_nav.policy.agent_rnn import RNNAgent
from hopfield_nav.policy.pair_regressor import PairRegressor
from hopfield_nav.policy.recurrent import EncodedRecurrentCore
from hopfield_nav.rollout.rnn import table_gather
from hopfield_nav.training.goal_pairs_setup import ARMS, agent_cfg_for_mode
from hopfield_nav.world.env import make_env
from hopfield_nav.world.spec import CellSets

RUNS = {
    "frozen_s0": ("goal_lifetimes_b2frz_full_s0_polar-night-19", "life_final.pt"),
    "frozen_s1": ("goal_lifetimes_b2frz_full_s1_absurd-breeze-22", "life_final.pt"),
    "S1": ("goal_lifetimes_b2scratch_lr4_full_s0_toasty-monkey-23", "life_final.pt"),
    "S2": ("goal_lifetimes_b2scratch_lr4det_full_s0_grateful-valley-24", "life_final.pt"),
    "B3-2": ("goal_lifetimes_b3_scratch_corner_s0_ruby-lake-26", "life_final.pt"),
    "raw": ("goal_lifetimes_b2_full_s0_quiet-sun-9", "life_final.pt"),
    "rec": ("goal_lifetimes_b2frz_rec_s0_spring-serenity-21", "life_final.pt"),
    "dist90": ("goal_lifetimes_b2diag_dist_anchor90_ethereal-elevator-12", "life_u1000.pt"),
    "A1": ("goal_pairs_a1c_cont_l5h768_u8k_step_s0_misty-feather-21", "pairs_final.pt"),
}
LAM = [11, 12, 13]
FWHM = 0.25
SIZE = 20
PERIOD = int(np.prod(LAM))
HOLDOUT_DEG = 15.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Loaded:
    """A checkpoint with a uniform face: `agent` (lifetimes) or `pairs`, and
    `features(x)` -- the memoryless features a linear head could read."""

    def __init__(self, name: str, device):
        run, file = RUNS[name]
        path = os.path.join(checkpoints_dir(), run, file)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["argv"]
        self.name, self.run, self.argv, self.device = name, run, a, device
        self.pairs = None
        self.agent = None
        if "model_state_dict" in ck:
            m = PairRegressor(ck["input_dim"], a["hidden_size"], a["num_layers"], a["movement_mode"],
                              nonlinearity=a["nonlinearity"], dropout=a.get("dropout", 0.0))
            m.load_state_dict(ck["model_state_dict"])
            self.pairs = m.to(device).eval()
            self.arm = "pairs"
            self.bypass = 0
            return
        self.arm = a["arm"]
        arm = ARMS[self.arm]
        self.bypass = 0 if not arm["input_prev_action"] else (4 if a["movement_mode"] == "discrete" else 2)
        extra = dict(arm, init_log_std=a.get("init_log_std", 0.0),
                     input_encoder_layers=a.get("encoder_layers", 0),
                     input_encoder_hidden=a.get("encoder_hidden", 768),
                     input_encoder_norm=a.get("encoder_norm", True),
                     input_encoder_detach=a.get("encoder_detach", False),
                     input_encoder_bypass=self.bypass)
        nonlin = a["nonlinearity"] if arm["rnn_cell"] == "mlp" else "tanh"
        acfg = agent_cfg_for_mode(a["mode"], a["movement_mode"], hidden_size=a["hidden_size"],
                                  num_rnn_layers=a["num_layers"], rnn_nonlinearity=nonlin, **extra)
        agent = RNNAgent(acfg, ck["input_dim"])
        agent.load_state_dict(ck["agent_state_dict"])
        self.agent = agent.to(device).eval()
        self.input_dim = int(ck["input_dim"])

    @property
    def has_encoder(self) -> bool:
        return self.agent is not None and isinstance(self.agent.rnn, EncodedRecurrentCore)

    @torch.no_grad()
    def features(self, codes: np.ndarray) -> np.ndarray:
        """Memoryless features for `(N, 2 Ng)` code pairs `[gbook(p), gbook(g)]`."""
        x = torch.from_numpy(codes.astype(np.float32)).to(self.device)
        if self.pairs is not None:
            return self.pairs.features(x).cpu().numpy()
        rnn = self.agent.rnn
        if isinstance(rnn, EncodedRecurrentCore):
            z, _ = rnn.encoder(x[:, None, :], None)
            return z[:, 0].cpu().numpy()
        if self.bypass:
            x = torch.cat([torch.zeros(len(x), self.bypass, device=x.device), x], 1)
        f, _ = rnn(x[:, None, :], None)        # a memoryless trunk, or one GRU step from h = 0
        return f[:, 0].cpu().numpy()

    def describe(self) -> str:
        if self.pairs is not None:
            return f"{self.name}: pairs model l{self.argv['num_layers']}h{self.argv['hidden_size']}"
        a = self.argv
        enc = f"encoder {a.get('encoder_layers', 0)}x{a.get('encoder_hidden', 768)}" if self.has_encoder else "no encoder"
        return f"{self.name}: {self.arm} arm, {a['num_layers']}x{a['hidden_size']}, {enc}, update {a.get('n_updates')}"


# ---------------------------------------------------------------------------
# Synthetic codes
# ---------------------------------------------------------------------------

CELLS = np.array([(x, y) for x in range(SIZE) for y in range(SIZE)], dtype=float)


def row_tables(thetas: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """`(B, S*S, Ng)` code tables, one lattice per row, on a 20x20 env at offset 0."""
    return np.stack([gbook_at(CELLS, LAM, FWHM, float(th), 1.0, sh).astype(np.float32)
                     for th, sh in zip(thetas, shifts)])


def code_pairs(rng, n: int, max_abs: int = 19, exact: int | None = None):
    """Random `(p, g)` on random lattices. Returns the code pair matrix, the env-frame
    displacement, the code-frame displacement, the per-module wrapped phase
    differences (cos, sin over 3 modules x 2 axes) and theta."""
    thetas = rng.uniform(-np.pi, np.pi, size=n)
    shifts = rng.uniform(0, PERIOD, size=(n, 2))
    p = rng.uniform(0, PERIOD, size=(n, 2))
    if exact is None:
        d = rng.randint(-max_abs, max_abs + 1, size=(n, 2)).astype(float)
        zero = np.abs(d).max(1) == 0
        d[zero, 0] = 1.0
    else:
        ax = rng.randint(0, 2, size=n)
        sgn = rng.choice([-1, 1], size=n)
        d = np.zeros((n, 2))
        d[np.arange(n), ax] = sgn * exact
        d[np.arange(n), 1 - ax] = rng.randint(-exact, exact + 1, size=n)
    # A per-pair theta and shift, vectorised: the code at (P, theta, shift) is
    # the standard code at R_theta P + shift, so rotate the positions here and
    # call `gbook_at` once at theta = 0.
    lam = np.array(LAM, float)[:, None]
    qp = rotate(p, thetas) + shifts
    qg = rotate(p + d, thetas) + shifts
    codes = np.concatenate([gbook_at(qp, LAM, FWHM), gbook_at(qg, LAM, FWHM)], 1).astype(np.float32)
    dprime = rotate(d, thetas)
    dphase = wrapped_phase_diff(module_phases(qg, LAM), module_phases(qp, LAM), lam)
    ang = 2 * np.pi * dphase / lam[None]
    phase_feats = np.concatenate([np.cos(ang).reshape(n, -1), np.sin(ang).reshape(n, -1)], 1)
    return codes, d, dprime, phase_feats, thetas


def rotate(v: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """`R_theta v` row by row, `rotation`'s convention."""
    c, s = np.cos(thetas), np.sin(thetas)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def ang_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((unit(pred) * unit(true)).sum(-1), -1.0, 1.0)))


def signed_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Angle from `true` to `pred`, degrees in (-180, 180]; positive = counter-clockwise."""
    u, v = unit(true), unit(pred)
    return np.degrees(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1)))


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

class Ridge:
    """Centred inputs, no per-dimension scaling (a near-constant hidden unit
    would otherwise be blown up on states it was not fit on); the penalty is
    `alpha x N x mean variance`, so it means the same at 20k static pairs and
    100k lifetime states whatever the feature scale."""

    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Ridge":
        self.mu = X.mean(0)
        self.ymu = Y.mean(0)
        Z = X - self.mu
        lam = self.alpha * len(Z) * float(Z.var(0).mean())
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        self.W = np.linalg.solve(A, Z.T @ (Y - self.ymu))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) @ self.W + self.ymu

    @staticmethod
    def r2(pred: np.ndarray, Y: np.ndarray) -> float:
        ss = ((Y - Y.mean(0)) ** 2).sum()
        return float(1.0 - ((Y - pred) ** 2).sum() / max(ss, 1e-12))


# ---------------------------------------------------------------------------
# P-delta (static)
# ---------------------------------------------------------------------------

def probe_delta(model: Loaded, rng, n_train: int = 20000, n_test: int = 5000) -> dict:
    codes, d, dp, ph, _ = code_pairs(rng, n_train)
    F = model.features(codes)
    fits = {
        "code_frame_dir": Ridge().fit(F, unit(dp)),
        "env_frame_dir": Ridge().fit(F, unit(d)),
        "phase_diffs": Ridge().fit(F, ph),
    }
    out = {}
    codes, d, dp, ph, _ = code_pairs(rng, n_test)
    F = model.features(codes)
    out["code_frame_dir"] = {"r2": Ridge.r2(fits["code_frame_dir"].predict(F), unit(dp)),
                             "deg": float(ang_err(fits["code_frame_dir"].predict(F), dp).mean())}
    out["env_frame_dir"] = {"r2": Ridge.r2(fits["env_frame_dir"].predict(F), unit(d)),
                            "deg": float(ang_err(fits["env_frame_dir"].predict(F), d).mean())}
    out["phase_diffs"] = {"r2": Ridge.r2(fits["phase_diffs"].predict(F), ph)}
    # Beyond the trained range: the same readout, pairs at exactly d.
    by_range = {}
    for dd in (5, 10, 15, 19, 22, 25, 30, 40):
        codes, d, dp, ph, _ = code_pairs(rng, 2000, exact=dd)
        F = model.features(codes)
        by_range[dd] = {"code_frame_deg": float(ang_err(fits["code_frame_dir"].predict(F), dp).mean()),
                        "phase_r2": Ridge.r2(fits["phase_diffs"].predict(F), ph)}
    out["by_range"] = by_range
    out["feature_dim"] = int(F.shape[1])
    return out


# ---------------------------------------------------------------------------
# Lifetimes with a probe attached
# ---------------------------------------------------------------------------

class Recorder:
    """Keeps every live step: (row, episode, step, h_top, direction, action, score)."""

    def __init__(self, hidden_index: int = -1):
        self.rows, self.eps, self.ts, self.hs, self.ds, self.acts, self.scores = [], [], [], [], [], [], []
        self.hidden_index = hidden_index

    def record(self, *, ep_idx, steps_in_ep, live, positions, goals, h, x, action, score):
        idx = np.where(live)[0]
        if len(idx) == 0:
            return
        self.rows.append(idx)
        self.eps.append(ep_idx[idx])
        self.ts.append(steps_in_ep[idx])
        if h is not None:
            self.hs.append(h[self.hidden_index][idx].detach().cpu().numpy().astype(np.float32))
        self.ds.append((goals[idx] - positions[idx]).astype(np.float32))
        self.acts.append(action[idx].astype(np.float32))
        self.scores.append(score[idx].astype(np.float32))

    def arrays(self) -> dict:
        out = {"row": np.concatenate(self.rows), "ep": np.concatenate(self.eps), "t": np.concatenate(self.ts),
               "d": np.concatenate(self.ds), "act": np.concatenate(self.acts), "score": np.concatenate(self.scores)}
        if self.hs:
            out["h"] = np.concatenate(self.hs)
        return out


class LatticeSource:
    """Per-row code tables; `swap_at_episode` switches rows to the second table
    from the start of that episode on (h is never touched)."""

    def __init__(self, table1: np.ndarray, table2: np.ndarray | None = None, swap_at_episode: int | None = None):
        self.t1, self.t2, self.swap_ep = table1, table2, swap_at_episode

    def grid_at(self, cells, ep_idx, steps_in_ep):
        if self.t2 is None or self.swap_ep is None:
            return table_gather(self.t1, cells, SIZE)
        a = table_gather(self.t1, cells, SIZE)
        b = table_gather(self.t2, cells, SIZE)
        use2 = (ep_idx >= self.swap_ep)[:, None]
        return np.where(use2, b, a).astype(np.float32)


class ActionPerturb:
    """Rotates the previous-action channel by `deg` at (episode, step) for the given rows."""

    def __init__(self, episode: int, step: int, rows: np.ndarray, deg: float = 90.0):
        self.ep, self.t, self.rows, self.R = episode, step, rows, rotation(np.radians(deg))
        self.fired = 0

    def prev_action(self, prev, ep_idx, steps_in_ep):
        if prev is None:
            return prev
        hit = np.zeros(len(ep_idx), bool)
        hit[self.rows] = True
        hit &= (ep_idx == self.ep) & (steps_in_ep == self.t)
        if not hit.any():
            return prev
        out = np.array(prev, dtype=np.float32, copy=True)
        out[hit] = out[hit] @ self.R.T
        self.fired += int(hit.sum())
        return out


class Probe:
    def __init__(self, source: LatticeSource, recorder: Recorder | None = None, perturb: ActionPerturb | None = None):
        self.grid_at = source.grid_at
        if recorder is not None:
            self.record = recorder.record
        if perturb is not None:
            self.prev_action = perturb.prev_action


def run_lifetimes(model: Loaded, probe: Probe, n_lifetimes: int, n_episodes: int, max_steps: int,
                  seed: int, wall_seed: int) -> dict:
    env = make_env(EnvConfig(size=SIZE, observation_size=120, movement_mode="continuous"), "continuous", wall_seed)
    arena = frozenset((x, y) for x in range(SIZE) for y in range(SIZE))
    cells = CellSets(size=SIZE, start_train=arena, goal_train=arena, goal_heldout=frozenset(), region=frozenset())
    return evaluate_lifetime_direction(env, model.agent, cells=cells, n_lifetimes=n_lifetimes,
                                       n_episodes=n_episodes, max_steps=max_steps, device=model.device,
                                       deterministic=False, seed=seed, probe=probe)


def by_bins(values: np.ndarray, keys: np.ndarray, bins) -> list:
    return [float(values[keys == k].mean()) if (keys == k).any() else float("nan") for k in bins]


# ---------------------------------------------------------------------------
# P-theta (and P-delta on the GRU state)
# ---------------------------------------------------------------------------

def probe_theta(model: Loaded, rng, n_lifetimes: int, n_episodes: int, max_steps: int, seed: int) -> dict:
    thetas = rng.uniform(-np.pi, np.pi, size=n_lifetimes)
    # A third of the rows in the held-out band; the rest, uniform over the
    # trained band, fit the readout and read it.
    k = n_lifetimes // 3
    thetas[:k] = rng.uniform(-np.radians(HOLDOUT_DEG), np.radians(HOLDOUT_DEG), size=k)
    shifts = rng.uniform(0, PERIOD, size=(n_lifetimes, 2))
    tables = row_tables(thetas, shifts)
    rec = Recorder()
    res = run_lifetimes(model, Probe(LatticeSource(tables), rec), n_lifetimes, n_episodes, max_steps, seed, wall_seed=seed)
    A = rec.arrays()
    H = A["h"]
    th = thetas[A["row"]]
    d = A["d"]
    dp = rotate(d, th)                       # the code-frame displacement, what the decode sees
    # The readout is fit on trained-band rows only: they are uniform over
    # ~330 deg, so an uncertain state decodes to an arbitrary angle (~90 deg
    # error) rather than to the held-out band's centre, which half the rows
    # sit in. Half of those rows fit, the other half and every held-out row read.
    trained_rows = np.where(np.abs(thetas) >= np.radians(HOLDOUT_DEG))[0]
    train_rows = rng.permutation(trained_rows)[: len(trained_rows) // 2]
    is_train = np.isin(A["row"], train_rows)
    late = A["ep"] >= 2
    fit_theta = Ridge().fit(H[is_train & late], np.stack([np.cos(th), np.sin(th)], 1)[is_train & late])
    fit_env = Ridge().fit(H[is_train & late], unit(d)[is_train & late])
    fit_code = Ridge().fit(H[is_train & late], unit(dp)[is_train & late])
    test = ~is_train
    pred = fit_theta.predict(H[test])
    th_hat = np.arctan2(pred[:, 1], pred[:, 0])
    err = np.degrees(np.abs(np.angle(np.exp(1j * (th_hat - th[test])))))
    held = np.abs(th[test]) < np.radians(HOLDOUT_DEG)
    ep, t = A["ep"][test], A["t"][test]
    steps = list(range(0, 11)) + [15, 20]
    eps = list(range(0, min(n_episodes, 20)))
    # A second readout fit on episode-0 states only: is theta there early, in
    # a subspace the settled readout does not see?
    early = A["ep"] == 0
    fit0 = Ridge().fit(H[is_train & early], np.stack([np.cos(th), np.sin(th)], 1)[is_train & early])
    pred0 = fit0.predict(H[test])
    err0 = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pred0[:, 1], pred0[:, 0]) - th[test])))))
    out = {
        "theta_deg_ep0_by_step_ep0fit": {"trained": by_bins(err0[~held], t[~held] * (ep[~held] == 0) + 999 * (ep[~held] != 0), steps),
                                         "heldout": by_bins(err0[held], t[held] * (ep[held] == 0) + 999 * (ep[held] != 0), steps)},
        "theta_deg_ep0_by_step": {"trained": by_bins(err[~held], t[~held] * (ep[~held] == 0) + 999 * (ep[~held] != 0), steps),
                                  "heldout": by_bins(err[held], t[held] * (ep[held] == 0) + 999 * (ep[held] != 0), steps)},
        "theta_deg_by_episode": {"trained": by_bins(err[~held], ep[~held], eps),
                                 "heldout": by_bins(err[held], ep[held], eps)},
        "theta_r2_late": Ridge.r2(pred[ep >= 2], np.stack([np.cos(th[test]), np.sin(th[test])], 1)[ep >= 2]),
        "env_dir_from_h": {"deg_late": float(ang_err(fit_env.predict(H[test][ep >= 2]), d[test][ep >= 2]).mean()),
                           "deg_ep0_by_step": by_bins(ang_err(fit_env.predict(H[test]), d[test]),
                                                      t * (ep == 0) + 999 * (ep != 0), steps),
                           "r2_late": Ridge.r2(fit_env.predict(H[test][ep >= 2]), unit(d[test][ep >= 2]))},
        "code_dir_from_h": {"deg_late": float(ang_err(fit_code.predict(H[test][ep >= 2]), dp[test][ep >= 2]).mean()),
                            "r2_late": Ridge.r2(fit_code.predict(H[test][ep >= 2]), unit(dp[test][ep >= 2]))},
        "policy_deg_ep0_by_step": {"trained": by_bins(A["score"][test][~held], t[~held] * (ep[~held] == 0) + 999 * (ep[~held] != 0), steps),
                                   "heldout": by_bins(A["score"][test][held], t[held] * (ep[held] == 0) + 999 * (ep[held] != 0), steps)},
        "policy_deg_by_episode": {"trained": by_bins(A["score"][test][~held], ep[~held], eps),
                                  "heldout": by_bins(A["score"][test][held], ep[held], eps)},
        "lifetime_deg": float(np.nanmean(res["by_episode"])),
        "n_states": int(len(H)),
    }
    return out


# ---------------------------------------------------------------------------
# P-swap
# ---------------------------------------------------------------------------

def probe_swap(model: Loaded, rng, n_lifetimes: int, n_episodes: int, max_steps: int, seed: int,
               swap_ep: int = 10, swap_deg: float = 90.0) -> dict:
    thetas = rng.uniform(-np.pi, np.pi, size=n_lifetimes)
    shifts = rng.uniform(0, PERIOD, size=(n_lifetimes, 2))
    t1 = row_tables(thetas, shifts)
    t2 = row_tables(thetas + np.radians(swap_deg), shifts)
    rec = Recorder()
    run_lifetimes(model, Probe(LatticeSource(t1, t2, swap_ep), rec), n_lifetimes, n_episodes, max_steps, seed, wall_seed=seed)
    A = rec.arrays()
    se = signed_err(A["act"], A["d"])
    ep, t = A["ep"], A["t"]
    steps = list(range(0, 11)) + [15, 20]
    eps = list(range(0, n_episodes))
    return {
        "swap_ep": swap_ep, "swap_deg": swap_deg,
        "signed_deg_swap_ep_by_step": by_bins(se, t * (ep == swap_ep) + 999 * (ep != swap_ep), steps),
        "abs_deg_swap_ep_by_step": by_bins(A["score"], t * (ep == swap_ep) + 999 * (ep != swap_ep), steps),
        "signed_deg_by_episode": by_bins(se, ep, eps),
        "abs_deg_by_episode": by_bins(A["score"], ep, eps),
        "signed_deg_before_swap_by_step": by_bins(se, t * (ep == swap_ep - 1) + 999 * (ep != swap_ep - 1), steps),
    }


# ---------------------------------------------------------------------------
# P-act
# ---------------------------------------------------------------------------

def probe_act(model: Loaded, rng, n_lifetimes: int, n_episodes: int, max_steps: int, seed: int,
              deg: float = 90.0) -> dict:
    thetas = rng.uniform(-np.pi, np.pi, size=n_lifetimes)
    shifts = rng.uniform(0, PERIOD, size=(n_lifetimes, 2))
    tables = row_tables(thetas, shifts)
    steps = list(range(0, 11))
    out = {"deg": deg}
    base = Recorder()
    run_lifetimes(model, Probe(LatticeSource(tables), base), n_lifetimes, n_episodes, max_steps, seed, wall_seed=seed)
    B = base.arrays()
    seb = signed_err(B["act"], B["d"])
    out["baseline"] = {"signed_ep0_by_step": by_bins(seb, B["t"] * (B["ep"] == 0) + 999 * (B["ep"] != 0), steps),
                       "abs_ep0_by_step": by_bins(B["score"], B["t"] * (B["ep"] == 0) + 999 * (B["ep"] != 0), steps)}
    for ep_hit, t_hit in ((0, 1), (0, 3), (0, 6), (5, 2)):
        rec = Recorder()
        pert = ActionPerturb(ep_hit, t_hit, np.arange(n_lifetimes), deg)
        run_lifetimes(model, Probe(LatticeSource(tables), rec, pert), n_lifetimes, n_episodes, max_steps, seed, wall_seed=seed)
        A = rec.arrays()
        se = signed_err(A["act"], A["d"])
        key = f"ep{ep_hit}_t{t_hit}"
        sel = A["ep"] == ep_hit
        out[key] = {"fired": pert.fired,
                    "signed_by_step": by_bins(se[sel], A["t"][sel], steps),
                    "abs_by_step": by_bins(A["score"][sel], A["t"][sel], steps)}
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(xs, w=5):
    return " ".join(f"{x:{w}.0f}" if np.isfinite(x) else " " * (w - 1) + "-" for x in xs)


def print_delta(name, r):
    print(f"  [{name}] P-delta  features {r['feature_dim']}d:  code-frame dir R2 {r['code_frame_dir']['r2']:.3f} "
          f"({r['code_frame_dir']['deg']:.1f} deg)   env-frame dir R2 {r['env_frame_dir']['r2']:.3f} "
          f"({r['env_frame_dir']['deg']:.1f} deg)   phase diffs R2 {r['phase_diffs']['r2']:.3f}")
    print("           by |Delta|: " + "  ".join(f"{k}:{v['code_frame_deg']:.0f}deg/{v['phase_r2']:.2f}" for k, v in r["by_range"].items()))


def print_theta(name, r):
    print(f"  [{name}] P-theta  (theta from h; R2 late {r['theta_r2_late']:.3f}; {r['n_states']} states; lifetime {r['lifetime_deg']:.1f} deg)")
    print(f"           ep0 by step (0..10,15,20)  trained: {fmt(r['theta_deg_ep0_by_step']['trained'])}")
    print(f"                                      heldout: {fmt(r['theta_deg_ep0_by_step']['heldout'])}")
    print(f"           by episode (0..)           trained: {fmt(r['theta_deg_by_episode']['trained'][:12])}")
    print(f"                                      heldout: {fmt(r['theta_deg_by_episode']['heldout'][:12])}")
    print(f"           ep0 by step, ep0-fit       trained: {fmt(r['theta_deg_ep0_by_step_ep0fit']['trained'])}")
    print(f"                                      heldout: {fmt(r['theta_deg_ep0_by_step_ep0fit']['heldout'])}")
    print(f"           policy ep0 by step         trained: {fmt(r['policy_deg_ep0_by_step']['trained'])}")
    print(f"                                      heldout: {fmt(r['policy_deg_ep0_by_step']['heldout'])}")
    print(f"           policy by episode          trained: {fmt(r['policy_deg_by_episode']['trained'][:12])}")
    print(f"                                      heldout: {fmt(r['policy_deg_by_episode']['heldout'][:12])}")
    e, c = r["env_dir_from_h"], r["code_dir_from_h"]
    print(f"           env-frame dir from h: late {e['deg_late']:.1f} deg (R2 {e['r2_late']:.3f}); ep0 by step {fmt(e['deg_ep0_by_step'])}")
    print(f"           code-frame dir from h: late {c['deg_late']:.1f} deg (R2 {c['r2_late']:.3f})")


def print_swap(name, r):
    print(f"  [{name}] P-swap  +{r['swap_deg']:.0f} deg at episode {r['swap_ep']}")
    print(f"           signed err, episode before, by step: {fmt(r['signed_deg_before_swap_by_step'])}")
    print(f"           signed err, swap episode,   by step: {fmt(r['signed_deg_swap_ep_by_step'])}")
    print(f"           abs err,    swap episode,   by step: {fmt(r['abs_deg_swap_ep_by_step'])}")
    print(f"           abs err by episode: {fmt(r['abs_deg_by_episode'])}")
    print(f"           signed by episode:  {fmt(r['signed_deg_by_episode'])}")


def print_act(name, r):
    print(f"  [{name}] P-act  prev_action rotated +{r['deg']:.0f} deg for one step")
    print(f"           baseline ep0 signed by step: {fmt(r['baseline']['signed_ep0_by_step'])}   abs: {fmt(r['baseline']['abs_ep0_by_step'])}")
    for k, v in r.items():
        if k.startswith("ep"):
            print(f"           {k:7s} (fired {v['fired']:4d}) signed: {fmt(v['signed_by_step'])}   abs: {fmt(v['abs_by_step'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default="all", choices=["all", "delta", "theta", "swap", "act"])
    ap.add_argument("--runs", default="frozen_s0,frozen_s1,S1,S2,B3-2,raw,rec")
    ap.add_argument("--n_lifetimes", type=int, default=128)
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device(args.device)
    results = {}
    for name in args.runs.split(","):
        model = Loaded(name, device)
        print(f"== {model.describe()}", flush=True)
        r = {}
        t0 = time.time()
        if args.probe in ("all", "delta"):
            r["delta"] = probe_delta(model, np.random.RandomState(args.seed))
            print_delta(name, r["delta"])
        if model.agent is not None and model.arm != "dist":
            if args.probe in ("all", "theta"):
                r["theta"] = probe_theta(model, np.random.RandomState(args.seed + 1), args.n_lifetimes,
                                         args.n_episodes, args.max_steps, args.seed)
                print_theta(name, r["theta"])
            if args.probe in ("all", "swap"):
                r["swap"] = probe_swap(model, np.random.RandomState(args.seed + 2), args.n_lifetimes,
                                       args.n_episodes, args.max_steps, args.seed)
                print_swap(name, r["swap"])
            if args.probe in ("all", "act") and model.arm == "full":
                r["act"] = probe_act(model, np.random.RandomState(args.seed + 3), args.n_lifetimes,
                                     min(args.n_episodes, 8), args.max_steps, args.seed)
                print_act(name, r["act"])
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
        results[name] = {"run": model.run, **r}
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
