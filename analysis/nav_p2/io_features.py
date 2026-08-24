"""The P3 cue statistics, in one implementation.

`docs/EXPERIMENTS_NAV_P2.md` §7.2 lists four groups of candidate statistics.
They are computed twice -- once on scripted probe trajectories
(`ideal_observer.py`) and once on a trained policy's own trajectories
(`ideal_observer_score.py`, §7.3 item 5) -- and the second is only meaningful
if it is the *same* statistic as the first. `rollout/signal.py` carries the
scar of the alternative: two copies of the direction readout that were
identical only by accident. So the per-step update lives here, in a class the
caller drives one step at a time, and both callers import it.

**Everything here is causal by construction.** A running statistic may only see
steps <= t. That is not a stylistic preference: the natural way to write these
is to slice a stored trajectory, and an off-by-one in that slice is a leak from
the future that would inflate every AUC in §7 without changing anything
visible. `StepFeatures` is advanced by the caller and has no access to the
trajectory it is not yet on.

**Nothing here touches the goal.** The feature update is handed `q` at three
recall depths, the commanded action and the realized displacement, and four
per-cell scalars derived from the recalled *pattern*. The goal position enters
this module in exactly one place, `cell_tables`, and only to compute the
`dir_cos` LABEL, which is returned under a name no feature set can address.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FEATURES = [
    # --- A: magnitude and its dynamics, from the REALIZED displacement -----
    "a1_qnorm", "a1_logq", "a2_dq", "a3_resid", "a4_resid_mean",
    "a4_resid_std", "a5_q_mean", "a5_q_std", "a6_q_min", "a6_q_max",
    "a7_q_rel",
    # --- B: direction consistency, from the REALIZED displacement ----------
    "b1_cos", "b1_cos_mean", "b2_spread", "b3_drift", "b3_drift_mean",
    "b4_dev",
    # --- C: recall-depth structure, as the POLICY sees it (projected) ------
    "c1_q12", "c1_q12_rel", "c2_cos13", "c3_q3_rel",
    # --- D: sensory consistency -------------------------------------------
    "d1_valid1", "d1_valid3", "d1_selfcos", "d1_chart", "d2_clip",
    "d2_clipcos", "d2_clip_mean",
    # --- A'/B': the same two sharpest cues from the COMMANDED action -------
    #     (§7.3 item 6 -- the channel ablation, not part of any cue group)
    "x_a3_cmd", "x_b2_cmd",
]
FEAT_INDEX = {n: i for i, n in enumerate(FEATURES)}
GROUPS = {
    "A": [n for n in FEATURES if n.startswith("a")],
    "B": [n for n in FEATURES if n.startswith("b")],
    "C": [n for n in FEATURES if n.startswith("c")],
    "D": [n for n in FEATURES if n.startswith("d")],
    "X": [n for n in FEATURES if n.startswith("x")],
}
# The headline ideal observer sees A u B u C: every one is a function of `q` at
# the three depths already fed to the policy, plus `prev_action` and
# `prev_displacement`, both policy inputs as of §4 B3. Group D is NOT in it --
# `d1` needs a pattern decoder fitted inside the env, which the agent would
# have to learn rather than be handed -- so it is reported as a separate, more
# permissive set.
POLICY_GROUPS = ["A", "B", "C"]

# Oracle channels: NOT policy-visible, used only as the ceiling control.
# `o_cos_goal` / `o_cos_dmax` additionally read the stored patterns, so they
# are the label for Q_step by construction and are never in a headline fit.
ORACLE = ["o_c1D", "o_c2D", "o_cos_goal", "o_cos_dmax"]
ORACLE_INDEX = {n: i for i, n in enumerate(ORACLE)}

# Names `cell_tables` returns that are LABELS, not features. Kept explicit so
# a future feature set cannot address one by accident.
LABEL_KEYS = ("dir_cos", "cos_goal", "cos_dmax")

_EPS = 1e-8


def all_cells(size: int) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.int32)


def snap(pos_f: np.ndarray, size: int) -> np.ndarray:
    """`world/vec_env.py:_update_snapped` -- round, then clip into the grid."""
    return np.clip(np.round(pos_f), 0, size - 1).astype(np.int64)


# ---------------------------------------------------------------------------
# Group D scaffolding: two ways to ask "is this pattern from this env?"
# ---------------------------------------------------------------------------

def fit_obs_decoder(emb: np.ndarray, obs: np.ndarray, ridge: float,
                    rng: np.random.RandomState):
    """Ridge map pattern -> North-facing cone, fitted on 3/4 of the env's cells.

    The env's wall code is a fresh random draw per arena (`world/env.py:329`),
    so this decoder is a fingerprint of *this* env: it is the concrete form of
    §7.2's `d1`. Fitted on 300 cells rather than 400 so the returned held-out
    score is a generalization number and not an interpolation.

    Returns ``(A, b, holdout_mean_abs)`` with ``A`` of shape (D, n_rays).
    """
    n = emb.shape[0]
    idx = rng.permutation(n)
    tr, te = idx[: (n * 3) // 4], idx[(n * 3) // 4:]
    X, Y = emb[tr].astype(np.float64), obs[tr].astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    yb = Y.mean(0)
    K = Xc @ Xc.T
    lam = ridge * np.trace(K) / K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Y - yb)
    A = Xc.T @ alpha
    b = yb - mu @ A
    hold = float(np.abs(emb[te].astype(np.float64) @ A + b).mean())
    return A.astype(np.float32), b.astype(np.float32), hold


def chart_basis(emb: np.ndarray, k: int = 64) -> np.ndarray:
    """Top-`k` right singular vectors of this env's encoded patterns.

    The env's patterns lie on a smooth 2-D chart embedded in D = 1024, so they
    occupy a low-dimensional subspace; a pattern from elsewhere in the scaffold
    does not. The decoder-free limit of the group-D question.
    """
    X = emb.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k].astype(np.float32)


def chart_residual(P: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Fraction of each row of `P` the chart subspace cannot explain."""
    c = P @ basis.T
    return np.sqrt(np.maximum(
        1.0 - (c * c).sum(1) / np.maximum((P * P).sum(1), _EPS), 0.0))


# ---------------------------------------------------------------------------
# Per-cell tables
# ---------------------------------------------------------------------------

@torch.no_grad()
def cell_tables(vh, hop, emb_np, emb_t, W, g_pat, d_pats, goal_vec,
                A_dec, b_dec, obs_pred_self, basis, depths=(1, 2, 3)):
    """Everything that depends only on (cell, memory), as a dict of arrays.

    Every embedding the agent ever sees is `encoded_Phi` at its *snapped* cell,
    so for a fixed memory the entire readout is a function of the cell. That is
    what lets a trajectory be an index sequence rather than a recomputation.

    `goal_vec` is (n_cell, 2) = goal - cell and is used for one thing only: the
    `dir_cos` LABEL.
    """
    n_cell = emb_np.shape[0]
    out = {}
    if hop.num_memories == 0:
        # Empty memory: nothing to recall. Matches `rollout/signal.py`, which
        # zero-fills rows whose Hopfield is empty.
        for k in ("q1", "q2", "q3"):
            out[k] = np.zeros((n_cell, 2), dtype=np.float32)
        for k in ("cos_goal", "cos_dmax", "c1D", "c2D", "d1_valid1",
                  "d1_valid3", "d1_selfcos", "d1_chart", "dir_cos"):
            out[k] = np.zeros(n_cell, dtype=np.float32)
        out["dir_cos"][:] = np.nan
        out["cos_dmax"][:] = -1.0
        return out

    traj = hop.recall_batch_trajectory(emb_t, list(depths), beta=hop.beta,
                                       alpha=1.0)
    r = {s: F.normalize(traj[s], dim=-1) for s in depths}
    r_np = {s: r[s].cpu().numpy() for s in depths}
    for i, s in enumerate(depths, start=1):
        out[f"q{i}"] = vh.project_displacement(
            emb_np, r_np[s], W).astype(np.float32)

    g = F.normalize(torch.from_numpy(g_pat).float().to(emb_t.device), dim=0)
    out["cos_goal"] = (r[depths[0]] @ g).cpu().numpy().astype(np.float32)
    if len(d_pats):
        P = F.normalize(torch.from_numpy(np.stack(d_pats)).float()
                        .to(emb_t.device), dim=-1)
        out["cos_dmax"] = (r[depths[0]] @ P.T).max(dim=1).values.cpu() \
            .numpy().astype(np.float32)
    else:
        out["cos_dmax"] = np.full(n_cell, -1.0, dtype=np.float32)

    # Group C in pattern space -- the oracle version of what the projection
    # hands the policy.
    out["c1D"] = np.linalg.norm(r_np[1] - r_np[2], axis=1).astype(np.float32)
    out["c2D"] = (r_np[1] * r_np[3]).sum(1).astype(np.float32)

    # Group D. `log1p` because the decoder does not shrink an out-of-span
    # pattern, it EXTRAPOLATES on it: the measured floor is ~30 against a
    # ceiling of ~1, a dynamic range no linear classifier could use raw.
    oh1 = r_np[1] @ A_dec + b_dec
    oh3 = r_np[3] @ A_dec + b_dec
    out["d1_valid1"] = np.log1p(np.abs(oh1).mean(1)).astype(np.float32)
    out["d1_valid3"] = np.log1p(np.abs(oh3).mean(1)).astype(np.float32)
    num = (oh1 * obs_pred_self).sum(1)
    den = np.linalg.norm(oh1, axis=1) * np.linalg.norm(obs_pred_self, axis=1)
    out["d1_selfcos"] = (num / np.maximum(den, _EPS)).astype(np.float32)
    out["d1_chart"] = chart_residual(r_np[1], basis).astype(np.float32)

    # LABEL ONLY.
    q1 = out["q1"]
    den = np.linalg.norm(q1, axis=1) * np.linalg.norm(goal_vec, axis=1)
    dc = np.full(n_cell, np.nan, dtype=np.float32)
    ok = den > 1e-8
    dc[ok] = ((q1[ok] * goal_vec[ok]).sum(1) / den[ok]).astype(np.float32)
    out["dir_cos"] = dc
    return out


# ---------------------------------------------------------------------------
# The causal per-step feature stream
# ---------------------------------------------------------------------------

class StepFeatures:
    """Running per-episode statistics, advanced one step at a time.

    Usage, per step, in this order:

        f = sf.observe(q1, q2, q3, d_valid1, d_valid3, d_selfcos, d_chart)
        ...                                    # save f if t is a checkpoint
        sf.act(action, realized_displacement)  # then move

    `observe` may only use what is known before acting; `act` records the
    action and the move it produced. Calling them out of order would make `a3`
    compare a change in `q` against a step not yet taken.
    """

    def __init__(self, n: int):
        z = np.zeros(n)
        self.n = n
        self.n_r = z.copy(); self.s_r = z.copy(); self.s_r2 = z.copy()
        self.n_q = z.copy(); self.s_q = z.copy(); self.s_q2 = z.copy()
        self.q_min = np.full(n, np.inf); self.q_max = np.full(n, -np.inf)
        self.n_b = z.copy(); self.s_b = z.copy()
        self.n_dr = z.copy(); self.s_dr = z.copy()
        self.n_cl = z.copy(); self.s_cl = z.copy()
        self.n_g = z.copy()
        self.sg = np.zeros((n, 2)); self.sg2 = np.zeros((n, 2))
        self.sgc = np.zeros((n, 2)); self.sgc2 = np.zeros((n, 2))
        self.prev_qn = z.copy(); self.prev_qh = np.zeros((n, 2))
        self.prev_gh = np.zeros((n, 2))
        self.last_a = np.zeros((n, 2)); self.last_d = np.zeros((n, 2))
        self.sum_d = np.zeros((n, 2)); self.sum_a = np.zeros((n, 2))
        self.have_prev = np.zeros(n, dtype=bool)
        self.path_len = z.copy()
        self.clip_seen = np.zeros(n, dtype=bool)
        self.q_hat = np.zeros((n, 2))
        self._f = np.zeros((n, len(FEATURES)))

    @staticmethod
    def _mean(s, n):
        return s / np.maximum(n, 1.0)

    @staticmethod
    def _std(s, s2, n):
        m = s / np.maximum(n, 1.0)
        return np.sqrt(np.maximum(s2 / np.maximum(n, 1.0) - m * m, 0.0))

    def observe(self, q1, q2, q3, dv1, dv3, dsc, dch) -> np.ndarray:
        qn = np.linalg.norm(q1, axis=1)
        qh = q1 / np.maximum(qn, _EPS)[:, None]
        self.q_hat = qh
        m = self.have_prev.astype(float)

        # ---- A ------------------------------------------------------------
        dq = np.where(self.have_prev, qn - self.prev_qn, 0.0)
        a3 = np.where(self.have_prev,
                      dq + (self.last_d * self.prev_qh).sum(1), 0.0)
        a3c = np.where(self.have_prev,
                       dq + (self.last_a * self.prev_qh).sum(1), 0.0)
        self.n_r += m; self.s_r += a3 * m; self.s_r2 += a3 * a3 * m
        self.n_q += 1.0; self.s_q += qn; self.s_q2 += qn * qn
        self.q_min = np.minimum(self.q_min, qn)
        self.q_max = np.maximum(self.q_max, qn)

        # ---- B ------------------------------------------------------------
        b1 = np.where(self.have_prev, (qh * self.prev_qh).sum(1), 0.0)
        self.n_b += m; self.s_b += b1 * m
        gh = self.sum_d + q1
        ghc = self.sum_a + q1
        self.n_g += 1.0
        self.sg += gh; self.sg2 += gh * gh
        self.sgc += ghc; self.sgc2 += ghc * ghc
        b3 = np.where(self.have_prev,
                      np.linalg.norm(gh - self.prev_gh, axis=1), 0.0)
        self.n_dr += m; self.s_dr += b3 * m
        gmean = self.sg / self.n_g[:, None]
        spread = np.maximum(self.sg2 / self.n_g[:, None] - gmean * gmean,
                            0.0).sum(1)
        gmc = self.sgc / self.n_g[:, None]
        spread_c = np.maximum(self.sgc2 / self.n_g[:, None] - gmc * gmc,
                              0.0).sum(1)
        b4 = np.linalg.norm(gh - gmean, axis=1)

        # ---- C ------------------------------------------------------------
        c1 = np.linalg.norm(q1 - q2, axis=1)
        n3 = np.linalg.norm(q3, axis=1)
        c2 = (q1 * q3).sum(1) / np.maximum(qn * n3, _EPS)

        # ---- D ------------------------------------------------------------
        dclip = np.linalg.norm(self.last_a - self.last_d, axis=1)
        na = np.linalg.norm(self.last_a, axis=1)
        nd = np.linalg.norm(self.last_d, axis=1)
        dcc = (self.last_a * self.last_d).sum(1) / np.maximum(na * nd, _EPS)
        self.n_cl += m; self.s_cl += dclip * m

        f = self._f
        f[:, FEAT_INDEX["a1_qnorm"]] = qn
        f[:, FEAT_INDEX["a1_logq"]] = np.log(qn + 1e-6)
        f[:, FEAT_INDEX["a2_dq"]] = dq
        f[:, FEAT_INDEX["a3_resid"]] = a3
        f[:, FEAT_INDEX["a4_resid_mean"]] = self._mean(self.s_r, self.n_r)
        f[:, FEAT_INDEX["a4_resid_std"]] = self._std(self.s_r, self.s_r2,
                                                     self.n_r)
        f[:, FEAT_INDEX["a5_q_mean"]] = self._mean(self.s_q, self.n_q)
        f[:, FEAT_INDEX["a5_q_std"]] = self._std(self.s_q, self.s_q2, self.n_q)
        f[:, FEAT_INDEX["a6_q_min"]] = self.q_min
        f[:, FEAT_INDEX["a6_q_max"]] = self.q_max
        f[:, FEAT_INDEX["a7_q_rel"]] = qn / np.maximum(
            self._mean(self.s_q, self.n_q), _EPS)
        f[:, FEAT_INDEX["b1_cos"]] = b1
        f[:, FEAT_INDEX["b1_cos_mean"]] = self._mean(self.s_b, self.n_b)
        f[:, FEAT_INDEX["b2_spread"]] = np.log1p(spread)
        f[:, FEAT_INDEX["b3_drift"]] = b3
        f[:, FEAT_INDEX["b3_drift_mean"]] = self._mean(self.s_dr, self.n_dr)
        f[:, FEAT_INDEX["b4_dev"]] = b4
        f[:, FEAT_INDEX["c1_q12"]] = c1
        f[:, FEAT_INDEX["c1_q12_rel"]] = c1 / np.maximum(qn, _EPS)
        f[:, FEAT_INDEX["c2_cos13"]] = c2
        f[:, FEAT_INDEX["c3_q3_rel"]] = n3 / np.maximum(qn, _EPS)
        f[:, FEAT_INDEX["d1_valid1"]] = dv1
        f[:, FEAT_INDEX["d1_valid3"]] = dv3
        f[:, FEAT_INDEX["d1_selfcos"]] = dsc
        f[:, FEAT_INDEX["d1_chart"]] = dch
        f[:, FEAT_INDEX["d2_clip"]] = dclip
        f[:, FEAT_INDEX["d2_clipcos"]] = dcc
        f[:, FEAT_INDEX["d2_clip_mean"]] = self._mean(self.s_cl, self.n_cl)
        f[:, FEAT_INDEX["x_a3_cmd"]] = a3c
        f[:, FEAT_INDEX["x_b2_cmd"]] = np.log1p(spread_c)

        self._pending = (qn, qh, gh)
        return f

    def act(self, action: np.ndarray, realized: np.ndarray) -> None:
        qn, qh, gh = self._pending
        self.path_len += np.linalg.norm(realized, axis=1)
        self.clip_seen |= np.linalg.norm(action - realized, axis=1) > 1e-9
        self.sum_d += realized
        self.sum_a += action
        self.last_a, self.last_d = action, realized
        self.prev_qn, self.prev_qh, self.prev_gh = qn, qh, gh
        self.have_prev = np.ones(self.n, dtype=bool)


def frame_self_test(vh, env, offset, cells) -> tuple[float, float]:
    """`q` and the realized displacement must live in the same 2-D frame.

    `a3` adds `<d, q_hat>` to a change in `‖q‖` and `b2` adds `q` to a sum of
    displacements: both are nonsense if the tangent basis orders its axes
    (East, North) while the env's `(dx, dy)` is (North, East). The two cannot
    be told apart by eye -- `gram_schmidt_2d_batch` is handed `d_forward` first
    and returns it *second* -- so this checks it against a target with a known
    answer. Oracle `q` at the goal pattern must point at the goal (§5.8: 0.96).
    Raises if the axis-swapped version is not far worse.
    """
    from hopfield_nav.rollout.distractors import goal_encoding
    emb_np = vh.get_encoded_state(cells, offset)
    W = vh.gram_schmidt_projection(cells, offset)
    g = goal_encoding(vh, offset, env.goal_location)
    q = vh.project_displacement(emb_np, np.broadcast_to(g, emb_np.shape), W)
    tgt = np.asarray(env.goal_location, float)[None, :] - cells.astype(float)
    den = np.linalg.norm(q, axis=1) * np.linalg.norm(tgt, axis=1)
    ok = den > 1e-8

    def _c(v):
        return float(np.mean((v[ok] * tgt[ok]).sum(1) / den[ok]))
    straight, swapped = _c(q), _c(q[:, ::-1])
    if not (straight > 0.85 and straight - swapped > 0.3):
        raise AssertionError(
            f"tangent basis / displacement frames disagree: {straight:.4f} vs "
            f"{swapped:.4f} swapped. a3 and b2 would be meaningless.")
    return straight, swapped


__all__ = [
    "FEATURES", "FEAT_INDEX", "GROUPS", "POLICY_GROUPS", "ORACLE",
    "ORACLE_INDEX", "LABEL_KEYS", "StepFeatures", "cell_tables",
    "chart_basis", "chart_residual", "fit_obs_decoder", "frame_self_test",
    "all_cells", "snap",
]
