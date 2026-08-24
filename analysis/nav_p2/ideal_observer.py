"""P3 stage 1 -- generate the ideal observer's feature/label tensors.

`docs/EXPERIMENTS_NAV_P2.md` §7. The question is *how much information about
whether the goal is in memory is present in what the policy can already see*,
and how that grows with steps observed and with what the agent does while
observing. This module does not answer it; it manufactures the evidence, and
`ideal_observer_fit.py` fits classifiers to it.

Three targets, per §7.1:

  * ``y_ep``    -- is a pattern from *this* env in the memory at all (the
                   explore/exploit regime question)?
  * ``y_step``  -- is the current recall the goal rather than a foreign
                   pattern (``lock == goal``)?
  * ``y_trust`` -- is this recall's *direction* usable right now
                   (``cos(q, goal - x) >= trust_thresh``)? The headline: §5.2
                   established that direction is the only thing the policy
                   consumes and that a recall which locked on the wrong pattern
                   still usually points roughly the right way, so "is this the
                   goal" and "should I follow it" are different questions.

**Why it can be done by table lookup.** Every embedding the agent ever sees is
``encoded_Phi`` at its *snapped* cell (``scaffold.get_encoded_state`` clips and
indexes; ``vec_env._update_snapped`` rounds), so for a fixed memory the whole
per-step readout -- ``q`` at every recall depth, the recall's fidelity, its
decoded observation -- is a function of the cell alone. So the expensive part
is computed once per (env, distractor draw, regime) as a 400-cell table, and a
trajectory is an index sequence into it. That is what makes seven probe
policies x seven distractor levels x eight draws affordable.

**What would falsify the whole measurement.** Four self-checks run inline and
print, and any of them failing means the numbers below are not about what they
claim:

  1. **The analytic anchor** (§7.3 item 0). §5.8 predicts goal-absent ``‖q‖``
     at ``sqrt(2/D) * sqrt(2) = 0.0625`` against a measured 0.0670, with
     goal-present at 0.3006. Those are printed back. A group-A AUC far below
     what that separation implies means the instrument is broken, not the
     signal.
  2. **Label leakage.** No feature may be computed from the goal position or
     from the true displacement to it. The feature builder is handed positions
     and displacements only; ``goal`` enters exactly one expression, the label
     ``dir_cos``. A permutation control in the fitting stage is the empirical
     half of this.
  3. **The observation decoder** (group D) is fitted on 300 of the env's 400
     cells and its validity score is reported on the 100 held out (ceiling) and
     on the distractor patterns (floor). If those two do not separate, ``d1``
     measures nothing and should read as such in the ablation.
  4. **Non-empty sets.** Every (level, regime) block asserts it wrote as many
     rows as it promised; a silently empty block would show up as a suspiciously
     clean AUC.

**Degenerate condition, flagged rather than hidden.** At ``n_dist = 0`` the
goal-absent memory is empty and ``q = 0`` exactly, which is a perfect cue for a
reason that has nothing to do with the signal being measured. It is a real
training condition so it is generated, but it is tagged and the fitting stage
reports it separately (§7.3 item 1).

    python -m analysis.nav_p2.ideal_observer --ckpt <any nav ckpt> \
        --envs 48 --draws 8 --starts 2 --steps 64 \
        --out results/nav_p2/io_probe.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world.env import raycast_codes

PROBES = ["still", "straight", "billiard", "along_q", "perp_q", "anti_q",
          "random"]

# Feature names, in the order the (T, N, F) array stores them. Grouped by the
# §7.2 cue families so the ablation can slice by prefix.
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
# The headline ideal observer sees A u B u C: every one of those is a function
# of `q` at the three depths already fed to the policy, plus `prev_action` and
# `prev_displacement`, both of which are policy inputs as of §4 B3. Group D is
# NOT in it -- `d1` needs a pattern decoder fitted inside the env, which the
# agent would have to learn rather than be handed -- so it is reported as a
# separate, more permissive set. See the module docstring.
POLICY_GROUPS = ["A", "B", "C"]
# Oracle channels: NOT policy-visible. Reported as the ceiling control, never
# pooled into the headline. `cos_goal`/`cos_dist_max` additionally use the
# stored patterns, so they are the label for `y_step` by construction.
ORACLE = ["o_c1D", "o_c2D", "o_cos_goal", "o_cos_dmax"]
ORACLE_INDEX = {n: i for i, n in enumerate(ORACLE)}

_EPS = 1e-8


def _all_cells(size: int) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.int32)


def _snap(pos_f: np.ndarray, size: int) -> np.ndarray:
    """`world/vec_env.py:_update_snapped` -- round, then clip into the grid."""
    return np.clip(np.round(pos_f), 0, size - 1).astype(np.int64)


def _fit_obs_decoder(emb: np.ndarray, obs: np.ndarray, ridge: float,
                     rng: np.random.RandomState):
    """Ridge map pattern -> North-facing cone, fitted on 3/4 of the env's cells.

    Group D asks whether a recalled pattern is *from this env*. The env's wall
    code is a fresh random draw per arena (`world/env.py:329`), so a decoder
    fitted here predicts a saturated +/-1 barcode for a pattern that belongs to
    this chart and mush for one that does not -- ``mean |o_hat|`` is the
    fingerprint check §7.1 says group D can be.

    Fitted on 300 cells rather than all 400 so the returned held-out score is a
    real generalization number: a decoder that only interpolates its training
    cells would give a validity score that means nothing on the recalls.

    Returns ``(A, b, holdout_validity, )`` with ``A`` (D, n_rays).
    """
    n = emb.shape[0]
    idx = rng.permutation(n)
    tr, te = idx[: (n * 3) // 4], idx[(n * 3) // 4:]
    X, Y = emb[tr].astype(np.float64), obs[tr].astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    yb = Y.mean(0)
    Yc = Y - yb
    # Dual form: 300 samples, 1024 dims.
    K = Xc @ Xc.T
    lam = ridge * np.trace(K) / K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yc)
    A = Xc.T @ alpha                                     # (D, n_rays)
    b = yb - mu @ A
    hold = float(np.abs(emb[te].astype(np.float64) @ A + b).mean())
    return A.astype(np.float32), b.astype(np.float32), hold


def _chart_basis(emb: np.ndarray, k: int = 64) -> np.ndarray:
    """Top-`k` right singular vectors of this env's 400 encoded patterns.

    The env's patterns lie on a smooth 2-D chart embedded in D = 1024, so they
    occupy a low-dimensional subspace; a pattern drawn from elsewhere in the
    scaffold does not. This is the decoder-free limit of the group-D question.
    """
    X = emb.astype(np.float64) - emb.astype(np.float64).mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k].astype(np.float32)


def _chart_residual(P: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Fraction of each row of `P` that the chart subspace cannot explain."""
    c = P @ basis.T
    keep = (c * c).sum(1)
    tot = (P * P).sum(1)
    return np.sqrt(np.maximum(1.0 - keep / np.maximum(tot, _EPS), 0.0))


@torch.no_grad()
def _cell_tables(vh, hop, emb_np, emb_t, W, g_pat, d_pats, goal_vec,
                 A_dec, b_dec, obs_pred_self, chart_basis, depths=(1, 2, 3)):
    """Everything that depends only on (cell, memory), as a dict of arrays.

    `goal_vec` is (n_cell, 2) = goal - cell and is used for exactly one thing:
    the ``dir_cos`` LABEL. Nothing returned under a feature name touches it.
    """
    n_cell = emb_np.shape[0]
    D = emb_np.shape[1]
    out = {}
    if hop.num_memories == 0:
        # Empty memory: the network has nothing to return. Matches
        # `rollout/signal.py`, which zero-fills rows whose Hopfield is empty.
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

    g = torch.from_numpy(g_pat).float().to(emb_t.device)
    g = F.normalize(g, dim=0)
    out["cos_goal"] = (r[depths[0]] @ g).cpu().numpy().astype(np.float32)
    if len(d_pats):
        P = F.normalize(torch.from_numpy(np.stack(d_pats)).float()
                        .to(emb_t.device), dim=-1)
        out["cos_dmax"] = (r[depths[0]] @ P.T).max(dim=1).values.cpu() \
            .numpy().astype(np.float32)
    else:
        out["cos_dmax"] = np.full(n_cell, -1.0, dtype=np.float32)

    # Group C, the D-dimensional (oracle) version: the residual and the
    # multistep agreement in pattern space rather than after the 2-D
    # projection the policy actually receives.
    out["c1D"] = np.linalg.norm(r_np[1] - r_np[2], axis=1).astype(np.float32)
    out["c2D"] = (r_np[1] * r_np[3]).sum(1).astype(np.float32)

    # Group D: is the recalled pattern one this env's chart can explain?
    # `log1p` because the decoder does not shrink an out-of-span pattern, it
    # *extrapolates* on it -- the measured floor is ~36 against a ceiling of
    # ~1 (see the controls printed at the end), a dynamic range no linear
    # classifier could use raw.
    oh1 = r_np[1] @ A_dec + b_dec
    oh3 = r_np[3] @ A_dec + b_dec
    out["d1_valid1"] = np.log1p(np.abs(oh1).mean(1)).astype(np.float32)
    out["d1_valid3"] = np.log1p(np.abs(oh3).mean(1)).astype(np.float32)
    num = (oh1 * obs_pred_self).sum(1)
    den = (np.linalg.norm(oh1, axis=1) * np.linalg.norm(obs_pred_self, axis=1))
    out["d1_selfcos"] = (num / np.maximum(den, _EPS)).astype(np.float32)
    # The clean limit of the same idea, with no decoder in the way: how far the
    # recalled pattern lies from the linear span of THIS env's patterns.
    out["d1_chart"] = _chart_residual(r_np[1], chart_basis).astype(np.float32)

    # LABEL ONLY.
    q1 = out["q1"]
    den = np.linalg.norm(q1, axis=1) * np.linalg.norm(goal_vec, axis=1)
    dc = np.full(n_cell, np.nan, dtype=np.float32)
    ok = den > 1e-8
    dc[ok] = ((q1[ok] * goal_vec[ok]).sum(1) / den[ok]).astype(np.float32)
    out["dir_cos"] = dc
    return out


def _probe_actions(probe: str, q_hat: np.ndarray, theta: np.ndarray,
                   mag: float, rng: np.random.RandomState,
                   t: int) -> tuple[np.ndarray, np.ndarray]:
    """(action, new_heading) for one batched step of a scripted probe.

    ``q_hat`` is the unit-normalized ``q`` at the current cell, which is
    exactly what the policy's direction channel carries -- so the ``*_q``
    probes are behaviours the policy could execute, not oracles.
    """
    n = q_hat.shape[0]
    if probe == "still":
        # Minimal motion: a fixed axis walked forward then back, so path
        # length accrues at ``mag`` per step while net displacement stays ~0.
        # The information floor with the smallest possible parallax.
        s = 1.0 if (t % 2 == 0) else -1.0
        a = s * mag * np.stack([np.cos(theta), np.sin(theta)], 1)
        return a, theta
    if probe in ("straight", "billiard"):
        a = mag * np.stack([np.cos(theta), np.sin(theta)], 1)
        return a, theta
    if probe == "random":
        th = rng.uniform(-np.pi, np.pi, size=n)
        return mag * np.stack([np.cos(th), np.sin(th)], 1), th
    # q-relative probes. Where q is exactly zero (empty memory) there is no
    # direction to be relative to, so those rows fall back to random -- the
    # honest behaviour, and it keeps the probe from silently becoming `still`.
    nz = np.linalg.norm(q_hat, axis=1) > 1e-8
    base = np.where(nz[:, None], q_hat,
                    np.stack([np.cos(theta), np.sin(theta)], 1))
    if probe == "along_q":
        d = base
    elif probe == "anti_q":
        d = -base
    elif probe == "perp_q":
        d = np.stack([-base[:, 1], base[:, 0]], 1)
    else:
        raise ValueError(probe)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), _EPS)
    return mag * d, np.arctan2(d[:, 1], d[:, 0])


class _Running:
    """Per-episode running statistics, advanced one step at a time.

    Kept as an explicit object because every one of these is a *causal*
    statistic -- it may only see steps <= t. Computing them by slicing a stored
    trajectory is where an off-by-one becomes a leak from the future.
    """

    def __init__(self, n: int):
        z = np.zeros(n, dtype=np.float64)
        self.n_r = z.copy(); self.s_r = z.copy(); self.s_r2 = z.copy()
        self.n_q = z.copy(); self.s_q = z.copy(); self.s_q2 = z.copy()
        self.q_min = np.full(n, np.inf); self.q_max = np.full(n, -np.inf)
        self.n_b = z.copy(); self.s_b = z.copy()
        self.n_dr = z.copy(); self.s_dr = z.copy()
        self.n_cl = z.copy(); self.s_cl = z.copy()
        # Allocentric goal estimate g_hat = sum(d) + q, accumulated two ways.
        self.n_g = z.copy()
        self.sg = np.zeros((n, 2)); self.sg2 = np.zeros((n, 2))
        self.n_gc = z.copy()
        self.sgc = np.zeros((n, 2)); self.sgc2 = np.zeros((n, 2))

    @staticmethod
    def _mean(s, n):
        return s / np.maximum(n, 1.0)

    @staticmethod
    def _std(s, s2, n):
        m = s / np.maximum(n, 1.0)
        v = s2 / np.maximum(n, 1.0) - m * m
        return np.sqrt(np.maximum(v, 0.0))


def _frame_self_test(vh, env, offset, cells, size) -> None:
    """`q` and the realized displacement must live in the same 2-D frame.

    ``a3`` adds ``<d, q_hat>`` to a change in ``‖q‖`` and ``b2`` adds ``q`` to
    a sum of displacements: both are nonsense if the tangent basis orders its
    axes (East, North) while the env's ``(dx, dy)`` is (North, East). The two
    conventions cannot be told apart by eye -- ``gram_schmidt_2d_batch`` is
    handed ``d_forward`` first and returns it *second* -- so this checks it,
    against a target that has a known answer.

    Oracle ``q`` at the goal pattern must point at the goal. The unswapped
    order should score ~0.96 (§5.8); the swapped order is what a transposed
    basis would produce and must score far worse. Raises if it does not.
    """
    from hopfield_nav.rollout.distractors import goal_encoding as _ge
    emb_np = vh.get_encoded_state(cells, offset)
    W = vh.gram_schmidt_projection(cells, offset)
    g = _ge(vh, offset, env.goal_location)
    q = vh.project_displacement(emb_np, np.broadcast_to(g, emb_np.shape), W)
    tgt = np.asarray(env.goal_location, float)[None, :] - cells.astype(float)
    d = np.linalg.norm(q, axis=1) * np.linalg.norm(tgt, axis=1)
    ok = d > 1e-8

    def _c(v):
        return float(np.mean((v[ok] * tgt[ok]).sum(1) / d[ok]))
    straight, swapped = _c(q), _c(q[:, ::-1])
    print(f"   FRAME SELF-TEST  cos(oracle q, goal-x) = {straight:.4f}   "
          f"axis-swapped = {swapped:.4f}")
    if not (straight > 0.85 and straight - swapped > 0.3):
        raise AssertionError(
            f"tangent basis / displacement frames disagree: {straight:.4f} vs "
            f"{swapped:.4f} swapped. a3 and b2 would be meaningless.")


def _build_world(cfg, encoder, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return build_eval_world(cfg, encoder, str(device), ckpt_path=None)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="read only for config; no policy is evaluated here")
    p.add_argument("--envs", type=int, default=48)
    p.add_argument("--draws", type=int, default=8,
                   help="independent distractor draws per (env, level). §5.2.1: "
                        "draw count is the larger variance term, larger than "
                        "the between-world one.")
    p.add_argument("--starts", type=int, default=2,
                   help="episodes per (env, level, draw, regime, probe)")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--n_distractors", type=int, nargs="+",
                   default=[0, 1, 2, 3, 5, 7, 10])
    p.add_argument("--probes", nargs="+", default=PROBES)
    p.add_argument("--step_norm", type=float, default=1.0,
                   help="|a| for every probe. The trained policies walk at "
                        "roughly one cell per step, and the comparison between "
                        "probes is only fair at a common speed.")
    p.add_argument("--checkpoints", type=int, nargs="+",
                   default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],
                   help="steps-observed values at which features are saved")
    p.add_argument("--trust_thresh", type=float, default=0.5,
                   help="cos(q, goal-x) at or above which following q is "
                        "called usable. §5.2 scores direction failure at 0.5.")
    p.add_argument("--lock_thresh", type=float, default=0.9)
    p.add_argument("--obs_ridge", type=float, default=1e-3)
    p.add_argument("--chart_k", type=int, default=64,
                   help="rank of the per-env pattern subspace used by d1_chart")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = args.envs
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D = enc_cfg.out_dim
    envs, vh, offsets = _build_world(cfg, encoder, device, args.seed)
    size = envs[0].size
    cells = _all_cells(size)
    n_cell = cells.shape[0]
    n_ray = cfg.env.observation_size

    probes = list(args.probes)
    levels = list(args.n_distractors)
    ckpts = [c for c in args.checkpoints if c <= args.steps]
    n_env, n_lvl, n_draw, n_start, n_probe = (
        len(envs), len(levels), args.draws, args.starts, len(probes))
    n_reg = 2
    # Rows per (env, level): draws x regimes x starts x probes, all simulated
    # in one batch because they share the per-cell tables.
    B = n_draw * n_reg * n_start * n_probe
    n_ck = len(ckpts)
    n_feat = len(FEATURES)

    print(f"encoder   : {cfg.encoder_checkpoint}")
    print(f"scaffold  : Npos={vh.Npos}  D={D}  beta={cfg.hopfield.beta:.4f}  "
          f"size={size}  rays={n_ray}")
    print(f"grid      : {n_env} envs x {n_lvl} levels x {n_draw} draws x "
          f"{n_reg} regimes x {n_start} starts x {n_probe} probes "
          f"= {n_env * n_lvl * B:,} episodes of {args.steps} steps")
    print(f"saved at t = {ckpts}")
    print(f"predicted goal-absent |q| = sqrt(2/D)*sqrt(2) = "
          f"{np.sqrt(2.0 / D) * np.sqrt(2.0):.4f}  (§5.8 measured 0.0670; "
          f"goal-present 0.3006)\n", flush=True)

    tot = n_env * n_lvl * B
    Xf = np.zeros((n_ck, tot, n_feat), dtype=np.float32)
    Xo = np.zeros((n_ck, tot, len(ORACLE)), dtype=np.float32)
    y_ep = np.zeros((n_ck, tot), dtype=np.int8)
    y_step = np.zeros((n_ck, tot), dtype=np.int8)
    y_trust = np.zeros((n_ck, tot), dtype=np.int8)
    valid = np.zeros((n_ck, tot), dtype=bool)
    valid_tr = np.zeros((n_ck, tot), dtype=bool)
    # Bookkeeping, all per row: which env / level / draw / regime / probe, and
    # the two motion costs the per-unit-distance comparison needs.
    meta = {k: np.zeros((tot,), dtype=np.int32)
            for k in ("env", "level", "draw", "regime", "probe")}
    path_len = np.zeros((n_ck, tot), dtype=np.float32)
    net_disp = np.zeros((n_ck, tot), dtype=np.float32)
    wall_d = np.zeros((n_ck, tot), dtype=np.float32)
    clip_any = np.zeros((n_ck, tot), dtype=bool)
    # Anchors (§7.3 item 0) and the group-D controls.
    anch_present, anch_absent = [], []
    d_hold, d_floor, d_self = [], [], []
    d_chart_self, d_chart_floor = [], []

    _frame_self_test(vh, envs[0], offsets[0], cells, size)

    rng = np.random.RandomState(args.seed + 1)
    trng = np.random.RandomState(args.seed + 7919)
    t0 = time.time()
    cur = 0

    for ei, env in enumerate(envs):
        off, goal = offsets[ei], env.goal_location
        emb_np = vh.get_encoded_state(cells, off)
        emb_t = torch.from_numpy(emb_np).float().to(device)
        W = vh.gram_schmidt_projection(cells, off)
        goal_vec = (np.asarray(goal, dtype=np.float64)[None, :]
                    - cells.astype(np.float64))
        goal_dist = np.linalg.norm(goal_vec, axis=1)
        wall_dist = np.minimum(cells.min(1),
                               (size - 1) - cells.max(1)).astype(np.float32)

        # North-facing cone at every cell -- the heading-free canonical view.
        obs_N = raycast_codes(env._wall_code, size, cells[:, 0], cells[:, 1],
                              np.zeros(n_cell), n_ray, env.wall_resolution)
        A_dec, b_dec, hold = _fit_obs_decoder(emb_np, obs_N, args.obs_ridge,
                                              rng)
        obs_pred_self = emb_np @ A_dec + b_dec
        chart_basis = _chart_basis(emb_np, args.chart_k)
        d_chart_self.append(float(_chart_residual(emb_np, chart_basis).mean()))
        d_hold.append(hold)
        d_self.append(float(np.abs(obs_N).mean()))
        g_pat = goal_encoding(vh, off, goal)

        for li, n_d in enumerate(levels):
            tabs = {}
            for di in range(n_draw):
                d_pats = (sample_distractors(vh, off, size, n_d, rng)
                          if n_d > 0 else [])
                if d_pats:
                    dps = np.stack(d_pats)
                    d_floor.append(float(np.abs(dps @ A_dec + b_dec).mean()))
                    d_chart_floor.append(
                        float(_chart_residual(dps, chart_basis).mean()))
                for reg in (0, 1):
                    hop = Hopfield(D, beta=cfg.hopfield.beta, device=str(device))
                    pats = ([g_pat] + list(d_pats)) if reg else list(d_pats)
                    if pats:
                        order = rng.permutation(len(pats))
                        for j in order:
                            hop.input_memory(torch.from_numpy(pats[j]).float())
                    tabs[(di, reg)] = _cell_tables(
                        vh, hop, emb_np, emb_t, W, g_pat, d_pats, goal_vec,
                        A_dec, b_dec, obs_pred_self, chart_basis)
            for di in range(n_draw):
                anch_present.append(
                    (n_d, float(np.median(
                        np.linalg.norm(tabs[(di, 1)]["q1"], axis=1)))))
                anch_absent.append(
                    (n_d, float(np.median(
                        np.linalg.norm(tabs[(di, 0)]["q1"], axis=1)))))

            # ---- one batched simulation over draws x regimes x starts x probes
            b_draw = np.repeat(np.arange(n_draw), n_reg * n_start * n_probe)
            b_reg = np.tile(np.repeat(np.arange(n_reg), n_start * n_probe),
                            n_draw)
            b_start = np.tile(np.repeat(np.arange(n_start), n_probe),
                              n_draw * n_reg)
            b_probe = np.tile(np.arange(n_probe), n_draw * n_reg * n_start)
            assert len(b_draw) == B

            # Stack the (draw, regime) tables so a row indexes with one gather.
            def _stack(key, extra=()):
                arr = np.zeros((n_draw, n_reg, n_cell) + extra,
                               dtype=np.float32)
                for (di, reg), tb in tabs.items():
                    arr[di, reg] = tb[key]
                return arr
            Q1 = _stack("q1", (2,))
            Q2 = _stack("q2", (2,))
            Q3 = _stack("q3", (2,))
            CG = _stack("cos_goal"); CD = _stack("cos_dmax")
            C1D = _stack("c1D"); C2D = _stack("c2D")
            DV1 = _stack("d1_valid1"); DV3 = _stack("d1_valid3")
            DSC = _stack("d1_selfcos"); DIR = _stack("dir_cos")

            # Start cells: uniform over the arena excluding the goal cell,
            # matching `ContinuousVecEnv.reset_all`.
            ok = np.where(goal_dist > 0)[0]
            c0 = ok[trng.randint(0, len(ok), size=B)]
            pos_f = cells[c0].astype(np.float64)
            theta = trng.uniform(-np.pi, np.pi, size=B)
            cell = c0.copy()

            run = _Running(B)
            prev_qn = np.zeros(B); prev_qh = np.zeros((B, 2))
            last_a = np.zeros((B, 2)); last_d = np.zeros((B, 2))
            have_prev = np.zeros(B, dtype=bool)
            sum_d = np.zeros((B, 2)); sum_a = np.zeros((B, 2))
            prev_gh = np.zeros((B, 2)); have_gh = np.zeros(B, dtype=bool)
            plen = np.zeros(B); clip_seen = np.zeros(B, dtype=bool)
            feat = np.zeros((B, n_feat), dtype=np.float64)
            oracle = np.zeros((B, len(ORACLE)), dtype=np.float64)
            ck_i = 0

            for t in range(1, args.steps + 1):
                q1 = Q1[b_draw, b_reg, cell]
                q2 = Q2[b_draw, b_reg, cell]
                q3 = Q3[b_draw, b_reg, cell]
                qn = np.linalg.norm(q1, axis=1)
                qh = q1 / np.maximum(qn, _EPS)[:, None]

                # ---- group A -------------------------------------------
                dq = np.where(have_prev, qn - prev_qn, 0.0)
                # a3: how much |q| SHOULD have shrunk if the target were fixed.
                proj_d = (last_d * prev_qh).sum(1)
                proj_a = (last_a * prev_qh).sum(1)
                a3 = np.where(have_prev, dq + proj_d, 0.0)
                a3c = np.where(have_prev, dq + proj_a, 0.0)
                m = have_prev.astype(float)
                run.n_r += m; run.s_r += a3 * m; run.s_r2 += a3 * a3 * m
                run.n_q += 1.0; run.s_q += qn; run.s_q2 += qn * qn
                run.q_min = np.minimum(run.q_min, qn)
                run.q_max = np.maximum(run.q_max, qn)

                # ---- group B -------------------------------------------
                b1 = np.where(have_prev, (qh * prev_qh).sum(1), 0.0)
                run.n_b += m; run.s_b += b1 * m
                gh = sum_d + q1
                ghc = sum_a + q1
                run.n_g += 1.0; run.sg += gh; run.sg2 += gh * gh
                run.n_gc += 1.0; run.sgc += ghc; run.sgc2 += ghc * ghc
                b3 = np.where(have_gh, np.linalg.norm(gh - prev_gh, axis=1),
                              0.0)
                run.n_dr += m; run.s_dr += b3 * m
                gmean = run.sg / run.n_g[:, None]
                gvar = (run.sg2 / run.n_g[:, None] - gmean * gmean)
                spread = np.maximum(gvar, 0.0).sum(1)
                gmc = run.sgc / run.n_gc[:, None]
                gvc = (run.sgc2 / run.n_gc[:, None] - gmc * gmc)
                spread_c = np.maximum(gvc, 0.0).sum(1)
                b4 = np.linalg.norm(gh - gmean, axis=1)

                # ---- group C (projected: what the policy is fed) --------
                c1 = np.linalg.norm(q1 - q2, axis=1)
                n3 = np.linalg.norm(q3, axis=1)
                c2 = ((q1 * q3).sum(1)
                      / np.maximum(qn * n3, _EPS))

                # ---- group D -------------------------------------------
                dclip = np.linalg.norm(last_a - last_d, axis=1)
                na = np.linalg.norm(last_a, axis=1)
                nd = np.linalg.norm(last_d, axis=1)
                dcc = (last_a * last_d).sum(1) / np.maximum(na * nd, _EPS)
                run.n_cl += m; run.s_cl += dclip * m

                f = feat
                f[:, FEAT_INDEX["a1_qnorm"]] = qn
                f[:, FEAT_INDEX["a1_logq"]] = np.log(qn + 1e-6)
                f[:, FEAT_INDEX["a2_dq"]] = dq
                f[:, FEAT_INDEX["a3_resid"]] = a3
                f[:, FEAT_INDEX["a4_resid_mean"]] = run._mean(run.s_r, run.n_r)
                f[:, FEAT_INDEX["a4_resid_std"]] = run._std(run.s_r, run.s_r2,
                                                            run.n_r)
                f[:, FEAT_INDEX["a5_q_mean"]] = run._mean(run.s_q, run.n_q)
                f[:, FEAT_INDEX["a5_q_std"]] = run._std(run.s_q, run.s_q2,
                                                        run.n_q)
                f[:, FEAT_INDEX["a6_q_min"]] = run.q_min
                f[:, FEAT_INDEX["a6_q_max"]] = run.q_max
                f[:, FEAT_INDEX["a7_q_rel"]] = qn / np.maximum(
                    run._mean(run.s_q, run.n_q), _EPS)
                f[:, FEAT_INDEX["b1_cos"]] = b1
                f[:, FEAT_INDEX["b1_cos_mean"]] = run._mean(run.s_b, run.n_b)
                f[:, FEAT_INDEX["b2_spread"]] = np.log1p(spread)
                f[:, FEAT_INDEX["b3_drift"]] = b3
                f[:, FEAT_INDEX["b3_drift_mean"]] = run._mean(run.s_dr,
                                                              run.n_dr)
                f[:, FEAT_INDEX["b4_dev"]] = b4
                f[:, FEAT_INDEX["c1_q12"]] = c1
                f[:, FEAT_INDEX["c1_q12_rel"]] = c1 / np.maximum(qn, _EPS)
                f[:, FEAT_INDEX["c2_cos13"]] = c2
                f[:, FEAT_INDEX["c3_q3_rel"]] = n3 / np.maximum(qn, _EPS)
                f[:, FEAT_INDEX["d1_valid1"]] = DV1[b_draw, b_reg, cell]
                f[:, FEAT_INDEX["d1_valid3"]] = DV3[b_draw, b_reg, cell]
                f[:, FEAT_INDEX["d1_selfcos"]] = DSC[b_draw, b_reg, cell]
                f[:, FEAT_INDEX["d2_clip"]] = dclip
                f[:, FEAT_INDEX["d2_clipcos"]] = dcc
                f[:, FEAT_INDEX["d2_clip_mean"]] = run._mean(run.s_cl, run.n_cl)
                f[:, FEAT_INDEX["x_a3_cmd"]] = a3c
                f[:, FEAT_INDEX["x_b2_cmd"]] = np.log1p(spread_c)

                oracle[:, ORACLE_INDEX["o_c1D"]] = C1D[b_draw, b_reg, cell]
                oracle[:, ORACLE_INDEX["o_c2D"]] = C2D[b_draw, b_reg, cell]
                oracle[:, ORACLE_INDEX["o_cos_goal"]] = CG[b_draw, b_reg, cell]
                oracle[:, ORACLE_INDEX["o_cos_dmax"]] = CD[b_draw, b_reg, cell]

                if ck_i < n_ck and t == ckpts[ck_i]:
                    sl = slice(cur, cur + B)
                    Xf[ck_i, sl] = f
                    Xo[ck_i, sl] = oracle
                    y_ep[ck_i, sl] = b_reg.astype(np.int8)
                    dcv = DIR[b_draw, b_reg, cell]
                    cg = CG[b_draw, b_reg, cell]
                    cd = CD[b_draw, b_reg, cell]
                    y_step[ck_i, sl] = ((cg >= args.lock_thresh) & (cg >= cd)) \
                        .astype(np.int8)
                    y_trust[ck_i, sl] = (dcv >= args.trust_thresh).astype(np.int8)
                    # A step standing on the goal has no defined direction to
                    # it, and the training env would have ended the episode
                    # there. Masked symmetrically in BOTH regimes, so the mask
                    # cannot itself carry the label.
                    valid[ck_i, sl] = goal_dist[cell] >= cfg.env.goal_radius
                    # `dir_cos` is additionally undefined where q == 0, which
                    # happens for one reason only: an EMPTY memory, i.e. the
                    # goal-absent regime at n_dist = 0. Folding that into the
                    # shared mask would silently delete every negative row of
                    # the degenerate condition and leave a one-class problem
                    # reading as a perfect AUC. Q_trust gets its own mask.
                    valid_tr[ck_i, sl] = valid[ck_i, sl] & np.isfinite(dcv)
                    path_len[ck_i, sl] = plen
                    net_disp[ck_i, sl] = np.linalg.norm(sum_d, axis=1)
                    wall_d[ck_i, sl] = wall_dist[cell]
                    clip_any[ck_i, sl] = clip_seen
                    ck_i += 1

                # ---- act -----------------------------------------------
                a, theta = _probe_actions_batch(probes, b_probe, qh, theta,
                                                args.step_norm, trng, t)
                new_f = np.clip(pos_f + a, 0.0, float(size - 1))
                d_real = new_f - pos_f
                if probes and "billiard" in probes:
                    bm = b_probe == probes.index("billiard")
                    if bm.any():
                        hit_x = ((new_f[:, 0] <= 0.0)
                                 | (new_f[:, 0] >= size - 1)) & bm
                        hit_y = ((new_f[:, 1] <= 0.0)
                                 | (new_f[:, 1] >= size - 1)) & bm
                        theta = np.where(hit_x, np.pi - theta, theta)
                        theta = np.where(hit_y, -theta, theta)
                pos_f = new_f
                cell_xy = _snap(pos_f, size)
                cell = cell_xy[:, 0] * size + cell_xy[:, 1]
                plen += np.linalg.norm(d_real, axis=1)
                clip_seen |= np.linalg.norm(a - d_real, axis=1) > 1e-9
                sum_d += d_real
                sum_a += a
                last_a, last_d = a, d_real
                prev_qn, prev_qh = qn, qh
                prev_gh, have_gh = gh, np.ones(B, dtype=bool)
                have_prev = np.ones(B, dtype=bool)

            assert ck_i == n_ck, f"saved {ck_i} checkpoints, expected {n_ck}"
            sl = slice(cur, cur + B)
            meta["env"][sl] = ei
            meta["level"][sl] = n_d
            meta["draw"][sl] = b_draw
            meta["regime"][sl] = b_reg
            meta["probe"][sl] = b_probe
            cur += B

        el = time.time() - t0
        print(f"  env {ei + 1}/{n_env}  ({el / (ei + 1):.1f} s/env, "
              f"{el / (ei + 1) * (n_env - ei - 1) / 60:.1f} min left)",
              flush=True)

    assert cur == tot, f"filled {cur} of {tot} rows"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    np.savez_compressed(
        args.out,
        X=Xf, X_oracle=Xo, y_ep=y_ep, y_step=y_step, y_trust=y_trust,
        valid=valid, valid_trust=valid_tr,
        path_len=path_len, net_disp=net_disp, wall_dist=wall_d,
        clip_any=clip_any,
        features=np.array(FEATURES), oracle_features=np.array(ORACLE),
        probes=np.array(probes), checkpoints=np.asarray(ckpts),
        levels=np.asarray(levels), embed_dim=np.asarray(D),
        trust_thresh=np.asarray(args.trust_thresh),
        lock_thresh=np.asarray(args.lock_thresh),
        step_norm=np.asarray(args.step_norm),
        **{f"meta_{k}": v for k, v in meta.items()})
    print(f"\nwrote {args.out}")

    # ---- the self-checks, printed from the saved arrays ------------------
    print("\n0. ANALYTIC ANCHOR (§7.3 item 0) -- per-env medians of |q|")
    print(f"   §5.8: goal-present 0.3006, goal-absent 0.0670, predicted "
          f"{np.sqrt(2.0 / D) * np.sqrt(2.0):.4f}")
    print(f"   {'n_d':>4} {'present':>9} {'absent':>9} {'ratio':>8}")
    ap = np.array(anch_present); aa = np.array(anch_absent)
    for n_d in levels:
        pv = ap[ap[:, 0] == n_d, 1]
        av = aa[aa[:, 0] == n_d, 1]
        r = (np.median(pv) / np.median(av)) if np.median(av) > 1e-9 else np.inf
        print(f"   {n_d:>4} {np.median(pv):>9.4f} {np.median(av):>9.4f} "
              f"{r:>8.2f}")

    print("\n   GROUP D CONTROLS -- ceiling and floor for both variants")
    print(f"   obs decoder, mean |o_hat|: real obs (ceiling) "
          f"{np.mean(d_self):.4f}   held-out in-env cells "
          f"{np.mean(d_hold):.4f}   distractor patterns "
          f"{np.mean(d_floor) if d_floor else float('nan'):.4f}")
    print(f"   chart residual (0 = fully explained by this env's subspace): "
          f"in-env cells {np.mean(d_chart_self):.4f}   distractor patterns "
          f"{np.mean(d_chart_floor) if d_chart_floor else float('nan'):.4f}")

    print("\n   ROW COUNTS BY (level, regime), at the last checkpoint")
    for n_d in levels:
        for reg in (1, 0):
            m = (meta["level"] == n_d) & (meta["regime"] == reg)
            v = valid[-1] & m
            vt = valid_tr[-1] & m
            yt = float(y_trust[-1][vt].mean()) if vt.any() else float("nan")
            ys = float(y_step[-1][v].mean()) if v.any() else float("nan")
            print(f"   n_d={n_d:>2}  {'present' if reg else 'absent':>7}"
                  f"  rows={int(m.sum()):>7}  valid={int(v.sum()):>7}"
                  f"  valid_trust={int(vt.sum()):>7}"
                  f"  P(y_trust)={yt:.3f}  P(y_step)={ys:.3f}")


def _probe_actions_batch(probes, b_probe, qh, theta, mag, rng, t):
    """Dispatch `_probe_actions` per probe over one batched step."""
    a = np.zeros_like(qh)
    th = theta.copy()
    for pi, name in enumerate(probes):
        m = b_probe == pi
        if not m.any():
            continue
        ai, ti = _probe_actions(name, qh[m], theta[m], mag, rng, t)
        a[m] = ai
        th[m] = ti
    return a, th


if __name__ == "__main__":
    main()
