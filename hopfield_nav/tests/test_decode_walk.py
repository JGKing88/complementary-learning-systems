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


def test_buffer_pairs_are_odometry_and_within_range():
    rng = np.random.RandomState(0)
    n_envs, W, T = 3, 2, 10
    buf = Buffer(n_envs, W, T, n_updates=4)
    for _ in range(6):                                     # ring wraps after 4
        seg = rng.randint(0, 20, size=(n_envs, W, T + 1, 2))
        buf.add(seg)
    assert buf.n == 4
    envs, p, g, d = buf.sample(rng, 500, k_max=5, max_abs=3)
    assert len(envs) == 500 and (np.abs(d).max(1) >= 1).all() and (np.abs(d).max(1) <= 3).all()
    assert np.array_equal(d, (g - p).astype(np.float32))
    # Every pair really is two positions of one stored walk.
    for e, pp, gg in zip(envs[:50], p[:50], g[:50]):
        walks = buf.pos[:buf.n, e]                         # (n, W, T+1, 2)
        hit = False
        for s in range(buf.n):
            for w in range(W):
                path = walks[s, w]
                ip = np.where((path == pp).all(1))[0]
                ig = np.where((path == gg).all(1))[0]
                if any(0 < j - i <= 5 for i in ip for j in ig):
                    hit = True
        assert hit


def test_targets_direction_and_heading8():
    d = np.array([[3.0, 0.0], [0.0, -2.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    u = targets_for(d, "direction")
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.allclose(u[0], [1, 0]) and np.allclose(u[1], [0, -1])
    h = targets_for(d, "heading8")
    assert np.allclose(np.linalg.norm(h, axis=1), 1.0)
    assert np.allclose(h[2], HEADINGS8[1])                # 45 deg
    assert np.allclose(h[3], HEADINGS8[1])                # 26.6 deg is nearer 45 than 0
