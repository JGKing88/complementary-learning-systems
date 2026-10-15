"""Input-history baselines and ||q|| residualisation on synthetic data."""
from __future__ import annotations

import numpy as np

from analysis.phase_decoding import history as hs

KW = dict(k_train=[3], n_rep=4, n_test=6, n_jobs=1,
          spline_cfg={"n_knots": 6, "degree": 3}, ridge_alpha=1e-3, C=1.0)


def test_window_and_since_start_bookkeeping():
    trial = np.array([0, 0, 0, 1, 1])
    t = np.array([0, 1, 2, 0, 1])
    f = np.array([[1.], [2.], [3.], [10.], [20.]], np.float32)
    w = hs.window(f, trial, t, 2, np.arange(5))
    # columns: lag0, lag1, mask0, mask1
    np.testing.assert_array_equal(w[:, 0], [1, 2, 3, 10, 20])
    np.testing.assert_array_equal(w[:, 1], [0, 1, 2, 0, 10])
    np.testing.assert_array_equal(w[:, 3], [0, 1, 1, 0, 1])
    s = hs.since_start(f, trial, t)
    np.testing.assert_allclose(s[:, 0], [1, 1.5, 2, 10, 15])     # running mean
    np.testing.assert_allclose(s[:, 1], [1, 2, 3, 10, 20])       # running max


def _trials(rng, n_arenas=10, n_trials=12, steps=8):
    """Per step: q (phase-dependent at lag 3 only, matched now), phase, arena."""
    rows = {k: [] for k in ("q", "phase", "arena", "trial", "t")}
    tid = 0
    for a in range(n_arenas):
        for ph in (0, 1):
            for _ in range(n_trials):
                q = rng.uniform(0, 1, steps)
                if ph == 1:
                    q[3] += 2.0          # a burst three steps... at step 3
                rows["q"].append(q)
                rows["phase"].append(np.full(steps, ph))
                rows["arena"].append(np.full(steps, a))
                rows["trial"].append(np.full(steps, tid))
                rows["t"].append(np.arange(steps))
                tid += 1
    return {k: np.concatenate(v) for k, v in rows.items()}


def test_window_finds_phase_at_a_past_lag():
    rng = np.random.default_rng(0)
    d = _trials(rng)
    rows = np.flatnonzero(d["t"] == 6)            # the burst is 3 steps back
    q = d["q"][:, None].astype(np.float32)
    now = q[rows]
    win = hs.window(q, d["trial"], d["t"], 5, rows)
    y, ar = d["phase"][rows], d["arena"][rows]
    a_now = hs.cross_arena(now, y, ar, reg=None, mode="plain", rng=rng, **KW)["3"]["mean"]
    a_win = hs.cross_arena(win, y, ar, reg=None, mode="plain", rng=rng, **KW)["3"]["mean"]
    assert a_now < 0.65 and a_win > 0.95


def test_lag_resolved_tracks_a_trace_that_fades():
    """B-style trials: arrival at step 20, then 40 post steps. z carries a
    post-arrival signal for 10 steps only."""
    rng = np.random.default_rng(3)
    rows = {k: [] for k in ("trial", "t", "arena", "phase", "at_goal_in")}
    tid = 0
    for a in range(10):
        for _ in range(10):
            T = 61
            t = np.arange(T)
            rows["trial"].append(np.full(T, tid))
            rows["t"].append(t)
            rows["arena"].append(np.full(T, a))
            rows["phase"].append((t > 20).astype(int))
            rows["at_goal_in"].append(t == 20)
            tid += 1
    d = {k: np.concatenate(v) for k, v in rows.items()}
    lag = hs.steps_since_arrival(d)
    keep = ~d["at_goal_in"]
    z = rng.standard_normal((len(lag), 5)).astype(np.float32)
    z[:, 0] += 3.0 * ((lag >= 1) & (lag <= 10))
    r = hs.lag_resolved(z[keep], d["phase"][keep], d["arena"][keep], lag[keep],
                        bins=[[1, 10], [21, 40]], k=5, n_rep=3, n_test=5, rng=rng,
                        n_jobs=1, C=1.0)
    early, late = r["balanced_mean"]
    assert early > late + 0.25


def _h(rng, shift):
    """h = smooth function of q + noise (+ an independent phase axis in
    h_extra). q = U[0,1] + shift * phase, so ``shift`` sets how much q alone
    settles the phase."""
    n_ar, per = 10, 200
    arena = np.repeat(np.arange(n_ar), per)
    phase = np.tile(np.r_[np.zeros(per // 2), np.ones(per // 2)], n_ar).astype(int)
    q = rng.uniform(0, 1, len(phase)) + shift * phase
    F = 30
    W = rng.standard_normal((3, F))
    basis = np.stack([q, np.sin(3 * q), q ** 2], 1)
    u = rng.standard_normal(F)
    u /= np.linalg.norm(u)
    h_q_only = basis @ W + 0.3 * rng.standard_normal((len(q), F))
    h_extra = h_q_only + 1.5 * phase[:, None] * u
    return (h_q_only.astype(np.float32), h_extra.astype(np.float32), phase, arena,
            q[:, None].astype(np.float32))


def _acc(z, y, ar, reg, mode, rng):
    return hs.cross_arena(z, y, ar, reg=reg, mode=mode, rng=rng, **KW)["3"]["mean"]


def test_residual_removes_what_q_carries_and_keeps_an_independent_axis():
    # q only weakly informative: most rows are ambiguous given q.
    rng = np.random.default_rng(1)
    h_q_only, h_extra, y, ar, q = _h(rng, shift=0.3)
    assert _acc(h_q_only, y, ar, None, "plain", rng) > 0.6   # q leaks into h
    assert _acc(h_q_only, y, ar, q, "residual", rng) < 0.58  # and is removed
    assert _acc(h_extra, y, ar, q, "residual", rng) > 0.85   # independent axis survives


def test_residual_is_diluted_where_q_settles_phase_but_incremental_is_not():
    # q decisive on most rows: the residual of the independent axis is zero
    # there, so residual accuracy falls toward chance; the incremental
    # comparison still shows that h adds nothing (q_only) or adds (extra).
    rng = np.random.default_rng(2)
    h_q_only, h_extra, y, ar, q = _h(rng, shift=0.8)
    assert _acc(h_extra, y, ar, q, "residual", rng) < 0.7
    base = _acc(h_extra, y, ar, q, "basis_only", rng)
    assert _acc(h_extra, y, ar, q, "basis_plus", rng) > base + 0.04
    assert abs(_acc(h_q_only, y, ar, q, "basis_plus", rng) - base) < 0.03
