"""P3 stage 1 -- generate the ideal observer's feature/label tensors.

`docs/EXPERIMENTS_NAV_P2.md` §7. The question is *how much information about
whether the goal is in memory is present in what the policy can already see*,
and how that grows with steps observed and with what the agent does while
observing. This module does not answer it; it manufactures the evidence, and
`ideal_observer_fit.py` fits classifiers to it. The cue statistics themselves
live in `io_features.py`, shared with `ideal_observer_score.py` so the probe
pass and the trained-agent pass cannot drift apart.

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

**Seven scripted probes** (§7.3 item 4) run side by side over identical
memories and identical per-cell tables, so the comparison between them is a
comparison of *behaviour* and of nothing else. `still` oscillates on a fixed
axis so its path length accrues while its net displacement stays ~0 -- the
information floor with the least possible parallax.

**Why it can be done by table lookup.** Every embedding the agent ever sees is
``encoded_Phi`` at its *snapped* cell, so for a fixed memory the entire
per-step readout is a function of the cell. The expensive part is computed once
per (env, draw, regime) as a 400-cell table and a trajectory is an index
sequence into it. That is what makes seven probes x seven distractor levels x
eight draws x 48 envs affordable.

**What would falsify this measurement.** Five checks run inline and print:

  1. **The frame self-test.** `q` and the realized displacement must share a
     2-D frame or `a3` and `b2` are noise. Checked against oracle `q`, which
     has a known answer; raises rather than warns.
  2. **The analytic anchor** (§7.3 item 0). §5.8 predicts goal-absent ``‖q‖``
     at ``sqrt(2/D)*sqrt(2) = 0.0625`` against a measured 0.0670, with
     goal-present at 0.3006. Both are printed per distractor level. A group-A
     AUC far below what that separation implies means the instrument is broken,
     not the signal.
  3. **Label leakage.** No feature is computed from the goal position or from
     the true displacement to it: `io_features.cell_tables` is the only place
     the goal appears and it is used only for the `dir_cos` label. The
     empirical half is the permutation control in the fitting stage.
  4. **The group-D controls.** The observation decoder's validity score is
     reported on held-out in-env cells (ceiling) and on distractor patterns
     (floor), and the chart residual likewise. If those do not separate, `d1`
     measures nothing and must read as such in the ablation.
  5. **Non-empty sets.** Row counts and class balance are printed for every
     (level, regime); an empty negative set is what makes a broken measurement
     look like a perfect AUC.

**Degenerate condition, flagged rather than hidden.** At ``n_dist = 0`` the
goal-absent memory is empty and ``q = 0`` exactly, a perfect cue for a reason
that has nothing to do with the signal. `dir_cos` is then undefined, so
`valid_trust` excludes those rows while `valid` keeps them -- one shared mask
would have silently deleted every negative row of the degenerate condition and
left a one-class problem reading as AUC 1.0.

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

from analysis.nav_p2.io_features import (
    FEATURES, ORACLE, ORACLE_INDEX, StepFeatures, all_cells, cell_tables,
    chart_basis, chart_residual, fit_obs_decoder, frame_self_test, snap,
)
from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world.env import raycast_codes

PROBES = ["still", "straight", "billiard", "along_q", "perp_q", "anti_q",
          "random"]
_EPS = 1e-8


def probe_step(probe: str, q_hat: np.ndarray, theta: np.ndarray, mag: float,
               rng: np.random.RandomState, t: int):
    """(action, new_heading) for one batched step of a scripted probe.

    ``q_hat`` is the unit-normalized ``q`` at the current cell -- exactly the
    policy's direction channel -- so the ``*_q`` probes are behaviours a policy
    could execute, not oracles.
    """
    n = q_hat.shape[0]
    if probe == "still":
        # Minimal motion: one fixed axis, walked forward then back. Path length
        # accrues at `mag` per step while net displacement stays ~0, which is
        # the information floor the comparison needs.
        s = 1.0 if (t % 2 == 0) else -1.0
        return s * mag * np.stack([np.cos(theta), np.sin(theta)], 1), theta
    if probe in ("straight", "billiard"):
        return mag * np.stack([np.cos(theta), np.sin(theta)], 1), theta
    if probe == "random":
        th = rng.uniform(-np.pi, np.pi, size=n)
        return mag * np.stack([np.cos(th), np.sin(th)], 1), th
    # q-relative probes. Where q is exactly zero (empty memory) there is no
    # direction to be relative to, so those rows keep their heading -- the
    # honest fallback, and it stops the probe silently becoming `still`.
    hd = np.stack([np.cos(theta), np.sin(theta)], 1)
    nz = np.linalg.norm(q_hat, axis=1) > 1e-8
    base = np.where(nz[:, None], q_hat, hd)
    perp = np.stack([-base[:, 1], base[:, 0]], 1)
    if probe == "along_q":
        d = base
    elif probe == "anti_q":
        d = -base
    elif probe == "perp_q":
        d = perp
    elif probe.startswith("mix"):
        # `mix_<along>_<perp>_<persist>` -- the parametric family the three
        # named probes above are corners of. §7.3 item 4's learned prober is a
        # search over these three weights: it asks whether any *combination* of
        # following q, circling it and persisting beats the menu, without
        # pretending a two-minute RL run would be a better answer.
        wa, wp, wh = (float(v) for v in probe.split("_")[1:4])
        d = wa * base + wp * perp + wh * hd
        z = np.linalg.norm(d, axis=1) < 1e-8
        d = np.where(z[:, None], hd, d)
    else:
        raise ValueError(probe)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), _EPS)
    return mag * d, np.arctan2(d[:, 1], d[:, 0])


def probe_step_batch(probes, b_probe, qh, theta, mag, rng, t):
    a = np.zeros_like(qh)
    th = theta.copy()
    for pi, name in enumerate(probes):
        m = b_probe == pi
        if not m.any():
            continue
        ai, ti = probe_step(name, qh[m], theta[m], mag, rng, t)
        a[m] = ai
        th[m] = ti
    return a, th


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
                   help="|a| for every probe. The comparison between probes is "
                        "only fair at a common speed.")
    p.add_argument("--checkpoints", type=int, nargs="+",
                   default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],
                   help="steps-observed values at which features are saved")
    p.add_argument("--trust_thresh", type=float, default=0.5,
                   help="cos(q, goal-x) at or above which following q is "
                        "called usable. §5.2 scores direction failure at 0.5.")
    p.add_argument("--lock_thresh", type=float, default=0.9)
    p.add_argument("--obs_ridge", type=float, default=1e-3)
    p.add_argument("--chart_k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = args.envs
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
        getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D = enc_cfg.out_dim
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device),
                                         ckpt_path=None)
    size = envs[0].size
    cells = all_cells(size)
    n_cell = cells.shape[0]
    n_ray = cfg.env.observation_size

    probes = list(args.probes)
    levels = list(args.n_distractors)
    ckpts = [c for c in args.checkpoints if c <= args.steps]
    n_env, n_draw, n_start, n_probe = (len(envs), args.draws, args.starts,
                                       len(probes))
    n_lvl, n_reg = len(levels), 2
    B = n_draw * n_reg * n_start * n_probe
    n_ck, n_feat = len(ckpts), len(FEATURES)

    print(f"encoder   : {cfg.encoder_checkpoint}")
    print(f"scaffold  : Npos={vh.Npos}  D={D}  beta={cfg.hopfield.beta:.4f}  "
          f"size={size}  rays={n_ray}  goal_radius={cfg.env.goal_radius}")
    print(f"grid      : {n_env} envs x {n_lvl} levels x {n_draw} draws x "
          f"{n_reg} regimes x {n_start} starts x {n_probe} probes "
          f"= {n_env * n_lvl * B:,} episodes of {args.steps} steps")
    print(f"saved at t = {ckpts}")
    st, sw = frame_self_test(vh, envs[0], offsets[0], cells)
    print(f"FRAME SELF-TEST  cos(oracle q, goal-x) = {st:.4f}   "
          f"axis-swapped = {sw:.4f}   (§5.8 expects ~0.96)\n", flush=True)

    tot = n_env * n_lvl * B
    Xf = np.zeros((n_ck, tot, n_feat), dtype=np.float32)
    Xo = np.zeros((n_ck, tot, len(ORACLE)), dtype=np.float32)
    y_ep = np.zeros((n_ck, tot), dtype=np.int8)
    y_step = np.zeros((n_ck, tot), dtype=np.int8)
    y_trust = np.zeros((n_ck, tot), dtype=np.int8)
    valid = np.zeros((n_ck, tot), dtype=bool)
    valid_tr = np.zeros((n_ck, tot), dtype=bool)
    meta = {k: np.zeros(tot, dtype=np.int32)
            for k in ("env", "level", "draw", "regime", "probe")}
    path_len = np.zeros((n_ck, tot), dtype=np.float32)
    net_disp = np.zeros((n_ck, tot), dtype=np.float32)
    wall_d = np.zeros((n_ck, tot), dtype=np.float32)
    clip_any = np.zeros((n_ck, tot), dtype=bool)
    dir_cos_v = np.full((n_ck, tot), np.nan, dtype=np.float32)
    anch_p, anch_a = [], []
    d_hold, d_floor, d_self, d_ch_self, d_ch_floor = [], [], [], [], []

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

        obs_N = raycast_codes(env._wall_code, size, cells[:, 0], cells[:, 1],
                              np.zeros(n_cell), n_ray, env.wall_resolution)
        A_dec, b_dec, hold = fit_obs_decoder(emb_np, obs_N, args.obs_ridge, rng)
        obs_pred_self = emb_np @ A_dec + b_dec
        basis = chart_basis(emb_np, args.chart_k)
        d_hold.append(hold)
        d_self.append(float(np.abs(obs_N).mean()))
        d_ch_self.append(float(chart_residual(emb_np, basis).mean()))
        g_pat = goal_encoding(vh, off, goal)

        for li, n_d in enumerate(levels):
            tabs = {}
            for di in range(n_draw):
                d_pats = (sample_distractors(vh, off, size, n_d, rng)
                          if n_d > 0 else [])
                if d_pats:
                    dps = np.stack(d_pats)
                    d_floor.append(float(np.abs(dps @ A_dec + b_dec).mean()))
                    d_ch_floor.append(float(chart_residual(dps, basis).mean()))
                for reg in (0, 1):
                    hop = Hopfield(D, beta=cfg.hopfield.beta,
                                   device=str(device))
                    pats = ([g_pat] + list(d_pats)) if reg else list(d_pats)
                    for j in rng.permutation(len(pats)):
                        hop.input_memory(torch.from_numpy(pats[j]).float())
                    tabs[(di, reg)] = cell_tables(
                        vh, hop, emb_np, emb_t, W, g_pat, d_pats, goal_vec,
                        A_dec, b_dec, obs_pred_self, basis)
                anch_p.append((n_d, float(np.median(
                    np.linalg.norm(tabs[(di, 1)]["q1"], axis=1)))))
                anch_a.append((n_d, float(np.median(
                    np.linalg.norm(tabs[(di, 0)]["q1"], axis=1)))))

            b_draw = np.repeat(np.arange(n_draw), n_reg * n_start * n_probe)
            b_reg = np.tile(np.repeat(np.arange(n_reg), n_start * n_probe),
                            n_draw)
            b_probe = np.tile(np.arange(n_probe), n_draw * n_reg * n_start)
            assert len(b_draw) == B, (len(b_draw), B)

            def _stack(key, extra=()):
                arr = np.zeros((n_draw, n_reg, n_cell) + extra,
                               dtype=np.float32)
                for (di, reg), tb in tabs.items():
                    arr[di, reg] = tb[key]
                return arr

            Q1, Q2, Q3 = (_stack("q1", (2,)), _stack("q2", (2,)),
                          _stack("q3", (2,)))
            CG, CD = _stack("cos_goal"), _stack("cos_dmax")
            C1D, C2D = _stack("c1D"), _stack("c2D")
            DV1, DV3 = _stack("d1_valid1"), _stack("d1_valid3")
            DSC, DCH = _stack("d1_selfcos"), _stack("d1_chart")
            DIR = _stack("dir_cos")

            # Start cells: uniform over the arena excluding the goal cell,
            # matching `ContinuousVecEnv.reset_all`.
            okc = np.where(goal_dist > 0)[0]
            cell = okc[trng.randint(0, len(okc), size=B)]
            pos_f = cells[cell].astype(np.float64)
            theta = trng.uniform(-np.pi, np.pi, size=B)

            sf = StepFeatures(B)
            oracle = np.zeros((B, len(ORACLE)))
            ck_i = 0
            for t in range(1, args.steps + 1):
                f = sf.observe(Q1[b_draw, b_reg, cell],
                               Q2[b_draw, b_reg, cell],
                               Q3[b_draw, b_reg, cell],
                               DV1[b_draw, b_reg, cell],
                               DV3[b_draw, b_reg, cell],
                               DSC[b_draw, b_reg, cell],
                               DCH[b_draw, b_reg, cell])
                oracle[:, ORACLE_INDEX["o_c1D"]] = C1D[b_draw, b_reg, cell]
                oracle[:, ORACLE_INDEX["o_c2D"]] = C2D[b_draw, b_reg, cell]
                oracle[:, ORACLE_INDEX["o_cos_goal"]] = CG[b_draw, b_reg, cell]
                oracle[:, ORACLE_INDEX["o_cos_dmax"]] = CD[b_draw, b_reg, cell]

                if ck_i < n_ck and t == ckpts[ck_i]:
                    sl = slice(cur, cur + B)
                    Xf[ck_i, sl] = f
                    Xo[ck_i, sl] = oracle
                    dcv = DIR[b_draw, b_reg, cell]
                    cg = CG[b_draw, b_reg, cell]
                    cd = CD[b_draw, b_reg, cell]
                    y_ep[ck_i, sl] = b_reg.astype(np.int8)
                    y_step[ck_i, sl] = ((cg >= args.lock_thresh)
                                        & (cg >= cd)).astype(np.int8)
                    y_trust[ck_i, sl] = (dcv >= args.trust_thresh).astype(np.int8)
                    # A step standing on the goal has no defined direction to
                    # it and the training env would have ended the episode
                    # there. Masked symmetrically in BOTH regimes, so the mask
                    # cannot itself carry the label.
                    valid[ck_i, sl] = goal_dist[cell] >= cfg.env.goal_radius
                    # dir_cos is additionally undefined where q == 0, which
                    # happens for one reason only: an empty memory, i.e. the
                    # goal-absent regime at n_dist = 0. Folding that into the
                    # shared mask would delete every negative row of the
                    # degenerate condition and leave a one-class problem
                    # reading as a perfect AUC.
                    valid_tr[ck_i, sl] = valid[ck_i, sl] & np.isfinite(dcv)
                    dir_cos_v[ck_i, sl] = dcv
                    path_len[ck_i, sl] = sf.path_len
                    net_disp[ck_i, sl] = np.linalg.norm(sf.sum_d, axis=1)
                    wall_d[ck_i, sl] = wall_dist[cell]
                    clip_any[ck_i, sl] = sf.clip_seen
                    ck_i += 1

                a, theta = probe_step_batch(probes, b_probe, sf.q_hat, theta,
                                            args.step_norm, trng, t)
                new_f = np.clip(pos_f + a, 0.0, float(size - 1))
                d_real = new_f - pos_f
                if "billiard" in probes:
                    bm = b_probe == probes.index("billiard")
                    hx = ((new_f[:, 0] <= 0.0) | (new_f[:, 0] >= size - 1)) & bm
                    hy = ((new_f[:, 1] <= 0.0) | (new_f[:, 1] >= size - 1)) & bm
                    theta = np.where(hx, np.pi - theta, theta)
                    theta = np.where(hy, -theta, theta)
                pos_f = new_f
                cxy = snap(pos_f, size)
                cell = cxy[:, 0] * size + cxy[:, 1]
                sf.act(a, d_real)

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
        valid=valid, valid_trust=valid_tr, dir_cos=dir_cos_v,
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

    print("\n0. ANALYTIC ANCHOR (§7.3 item 0) -- medians of |q| over draws")
    print(f"   §5.8: goal-present 0.3006, goal-absent 0.0670, predicted "
          f"{np.sqrt(2.0 / D) * np.sqrt(2.0):.4f}")
    ap, aa = np.array(anch_p), np.array(anch_a)
    print(f"   {'n_d':>4} {'present':>9} {'absent':>9} {'ratio':>8}")
    for n_d in levels:
        pv = ap[ap[:, 0] == n_d, 1]
        av = aa[aa[:, 0] == n_d, 1]
        r = np.median(pv) / np.median(av) if np.median(av) > 1e-9 else np.inf
        print(f"   {n_d:>4} {np.median(pv):>9.4f} {np.median(av):>9.4f} "
              f"{r:>8.2f}")

    print("\n   GROUP D CONTROLS -- ceiling and floor for both variants")
    print(f"   obs decoder, mean |o_hat|: real observations (ceiling) "
          f"{np.mean(d_self):.4f}   held-out in-env cells {np.mean(d_hold):.4f}"
          f"   distractor patterns "
          f"{np.mean(d_floor) if d_floor else float('nan'):.4f}")
    print(f"   chart residual (0 = fully inside this env's subspace): "
          f"in-env cells {np.mean(d_ch_self):.4f}   distractor patterns "
          f"{np.mean(d_ch_floor) if d_ch_floor else float('nan'):.4f}")

    print("\n   ROW COUNTS BY (level, regime), at the last checkpoint")
    for n_d in levels:
        for reg in (1, 0):
            m = (meta["level"] == n_d) & (meta["regime"] == reg)
            v, vt = valid[-1] & m, valid_tr[-1] & m
            yt = float(y_trust[-1][vt].mean()) if vt.any() else float("nan")
            ys = float(y_step[-1][v].mean()) if v.any() else float("nan")
            print(f"   n_d={n_d:>2}  {'present' if reg else 'absent':>7}"
                  f"  rows={int(m.sum()):>7}  valid={int(v.sum()):>7}"
                  f"  valid_trust={int(vt.sum()):>7}"
                  f"  P(y_trust)={yt:.3f}  P(y_step)={ys:.3f}")
    print("   In the goal-ABSENT regime `q` points at a foreign pattern, whose "
          "direction in this env's frame is unrelated to the goal, so "
          "P(y_trust) there has a chance value of 1/3 for a cos >= 0.5 "
          "threshold. Read the absent rows against that, not against 0.")


if __name__ == "__main__":
    main()
