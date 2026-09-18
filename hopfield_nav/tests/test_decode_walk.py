"""Phase 1 (plan sec 6.4): random walks, the replay buffer and the odometry target."""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import EnvConfig
from hopfield_nav.train_decode_walk import HEADINGS8, Buffer, Walkers, targets_for
from hopfield_nav.world.env import make_env


def test_walkers_take_unit_steps_and_never_teleport():
    envs = [make_env(EnvConfig(size=8, observation_size=40), "discrete", seed=s) for s in (1, 2)]
    w = Walkers(envs, walkers=4, seed=0)
    seg = w.segment(T=25)
    assert seg.shape == (2, 4, 26, 2)
    step = np.abs(np.diff(seg, axis=2)).max(-1)          # Chebyshev size of every step
    assert step.max() <= 1                                # a unit step or a blocked one, never a jump
    assert (step == 1).mean() > 0.5                       # and it does move
    assert seg.min() >= 0 and seg.max() <= 7
    # A second segment continues from where the first ended.
    seg2 = w.segment(T=3)
    assert np.array_equal(seg2[:, :, 0], seg[:, :, -1])


def _walk_segments(rng, n_envs, W, T, n_seg):
    """Continuous random walks cut into segments whose first position repeats the previous last."""
    pos = rng.randint(5, 15, size=(n_envs, W, 2))
    segs = []
    for _ in range(n_seg):
        seg = np.zeros((n_envs, W, T + 1, 2), dtype=np.int64)
        seg[:, :, 0] = pos
        for t in range(T):
            step = rng.randint(-1, 2, size=(n_envs, W, 2))
            pos = np.clip(pos + step, 0, 19)
            seg[:, :, t + 1] = pos
        segs.append(seg)
    return segs


def test_buffer_pairs_are_odometry_and_within_range():
    rng = np.random.RandomState(0)
    n_envs, W, T = 3, 2, 10
    buf = Buffer(n_envs, W, T, n_updates=4)
    segs = _walk_segments(rng, n_envs, W, T, 6)
    for seg in segs:                                       # ring holds 4 * T + 1 positions
        buf.add(seg)
    assert buf.n == 4 * T + 1
    # The stored history is the tail of the true continuous walk.
    full = np.concatenate([segs[0]] + [s[:, :, 1:] for s in segs[1:]], axis=2)   # (n_envs, W, 6T+1, 2)
    for i in range(buf.n):
        assert np.array_equal(buf._at(np.arange(n_envs), np.zeros(n_envs, int), i), full[:, 0, full.shape[2] - buf.n + i])
    envs, p, g, d = buf.sample(rng, 500, k_max=25, max_abs=3)
    assert len(envs) == 500 and (np.abs(d).max(1) >= 1).all() and (np.abs(d).max(1) <= 3).all()
    assert np.array_equal(d, (g - p).astype(np.float32))
    # Every pair is two positions of one stored walk, at most k_max apart.
    for e, pp, gg in zip(envs[:60], p[:60], g[:60]):
        hit = False
        for w in range(W):
            path = full[e, w, full.shape[2] - buf.n:]
            ip = np.where((path == pp).all(1))[0]
            ig = np.where((path == gg).all(1))[0]
            if any(0 < j - i <= 25 for i in ip for j in ig):
                hit = True
        assert hit


def test_buffer_balance_flattens_the_displacement_sizes():
    rng = np.random.RandomState(1)
    n_envs, W, T = 4, 4, 50
    buf = Buffer(n_envs, W, T, n_updates=8)
    for seg in _walk_segments(rng, n_envs, W, T, 8):
        buf.add(seg)
    _, _, _, d = buf.sample(rng, 4000, k_max=300, max_abs=6)
    r = np.abs(d).max(1).astype(int)
    plain = np.bincount(r, minlength=7)[1:]
    _, _, _, d = buf.sample(rng, 4000, k_max=300, max_abs=6, balance=True)
    r = np.abs(d).max(1).astype(int)
    bal = np.bincount(r, minlength=7)[1:]
    assert plain.max() / plain.min() > 1.5                # a walk's own sizes are uneven
    assert bal.max() / bal.min() < 1.3                    # balanced within 30%


def test_targets_direction_and_heading8():
    d = np.array([[3.0, 0.0], [0.0, -2.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    u = targets_for(d, "direction")
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.allclose(u[0], [1, 0]) and np.allclose(u[1], [0, -1])
    h = targets_for(d, "heading8")
    assert np.allclose(np.linalg.norm(h, axis=1), 1.0)
    assert np.allclose(h[2], HEADINGS8[1])                # 45 deg
    assert np.allclose(h[3], HEADINGS8[1])                # 26.6 deg is nearer 45 than 0


def test_encoder_walk_masks_and_readout():
    """Phase 1, encoder side: odometry masks are the coordinate masks restricted to
    a walker's own moments; the projected readout recovers direction exactly on a
    linear embedding."""
    from hopfield_nav.train_encoder_walk import near_far_masks, readout_direction, angular_error
    rng = np.random.RandomState(0)
    e = np.repeat(np.arange(3), 8)
    w = np.tile(np.repeat(np.arange(2), 4), 3)
    pos = rng.randint(0, 20, size=(24, 2))
    near_o, far_o = near_far_masks(e, w, pos, 6.0, "odometry")
    near_c, far_c = near_far_masks(e, w, pos, 6.0, "coords")
    same_walker = (e[:, None] == e[None, :]) & (w[:, None] == w[None, :])
    assert np.array_equal(near_o, near_c & same_walker)
    assert np.array_equal(far_o, far_c & same_walker)
    assert not near_o.diagonal().any() and not far_o.diagonal().any()
    assert not (near_c & (e[:, None] != e[None, :])).any()
    # Linear embedding z = A [x, y] with orthogonal rows (the frame the harness's
    # Gram-Schmidt assumes): the projection of z(g) - z(p) is exactly (dx, dy)
    # up to scale.
    S = 12
    A = np.linalg.qr(rng.randn(16, 2))[0].T * 3.0
    cells = np.array([(x, y) for x in range(S) for y in range(S)], dtype=float)
    Z = cells @ A
    p = rng.randint(0, S, size=(300, 2))
    g = rng.randint(0, S, size=(300, 2))
    ok = (np.abs(g - p).max(1) >= 1)
    err = angular_error(readout_direction(Z, S, p[ok], g[ok]), (g - p)[ok].astype(float))
    assert err.max() < 1e-3


def test_visited_buffer_pairs_are_visited_cells_of_one_walker():
    from hopfield_nav.train_decode_walk import VisitedBuffer
    rng = np.random.RandomState(0)
    n_envs, W, T, S = 3, 2, 30, 12
    buf = VisitedBuffer(n_envs, W, S)
    pos = rng.randint(0, S, size=(n_envs, W, 2))
    for _ in range(3):
        seg = np.zeros((n_envs, W, T + 1, 2), dtype=np.int64)
        seg[:, :, 0] = pos
        for t in range(T):
            pos = np.clip(pos + rng.randint(-1, 2, size=(n_envs, W, 2)), 0, S - 1)
            seg[:, :, t + 1] = pos
        buf.add(seg)
    envs, p, g, d = buf.sample(rng, 300, k_max=0, max_abs=5, balance=True)
    assert len(envs) == 300 and (np.abs(d).max(1) >= 1).all() and (np.abs(d).max(1) <= 5).all()
    # both ends visited by some walker of that env (the walker id is not returned; check the env's union)
    for e_, pp, gg in zip(envs, p, g):
        assert buf.visited[e_, :, pp[0] * S + pp[1]].any() and buf.visited[e_, :, gg[0] * S + gg[1]].any()


def test_balanced_rows_are_endpoints_of_balanced_pairs_of_one_walker():
    from hopfield_nav.train_encoder_walk import balanced_rows
    rng = np.random.RandomState(0)
    n_envs, W, T = 4, 3, 40
    buf = Buffer(n_envs, W, T, n_updates=6)
    for seg in _walk_segments(rng, n_envs, W, T, 6):
        buf.add(seg)
    chosen = rng.randint(0, W, size=n_envs)
    rows = balanced_rows(buf, rng, chosen, per_walker=64, k_max=200, max_abs=6)
    assert rows.shape == (n_envs * 64, 2)
    for i in range(n_envs):
        r = rows[i * 64:(i + 1) * 64]
        p, g = r[:32], r[32:]
        d = np.abs(g - p).max(1)
        assert (d >= 1).all() and (d <= 6).all()
        hist = np.bincount(d, minlength=7)[1:]
        assert hist.max() - hist.min() <= 3                    # balanced up to top-up
        hist_walk = buf.pos[i, chosen[i], :buf.n]
        for q in r:                                            # every row is a moment of that walker
            assert (hist_walk == q).all(1).any()
