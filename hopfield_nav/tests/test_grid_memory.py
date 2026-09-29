"""The idea-1 memory parts (``hopfield_nav/memory``) against ground truth.

1. ``SmoothedGridCode`` equals the smoothed codebook column, bit for bit.
2. The frozen grid MLP points from one cell to another in a size-20 arena at
   random scaffold offsets -- the regime the navigator uses it in.
3. ``read_batch`` reproduces the offline single-view retrieval (plan §2.4).
4. ``GridMemoryReadout`` returns a direction toward the stored goal, zeros and
   ``c = 0`` with nothing stored, and ``c · d`` under ``scale_q_by_c``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from gridcode.codebook import gen_gbook_2d
from gridcode.smoothing import smooth_gbook
from hopfield_nav.config import EnvConfig
from hopfield_nav.memory.grid_code import SmoothedGridCode
from hopfield_nav.memory.readout import GridMemoryReadout
from hopfield_nav.memory.sensory_kv import SensoryKVMemory, omni_key, read_batch
from hopfield_nav.world.env import make_env
from hopfield_nav.world.vec_env import VecEnv

GRID_MLP = ("/orcd/pool/003/jackking/cls_runs/agent_ckpts/"
            "goal_pairs_p1_grid64_bal_s0/pairs_final.pt")
LAMBDAS = [11, 12, 13]


@pytest.mark.parametrize("lambdas,fwhm", [([3, 4], 0.25), (LAMBDAS, 0.25), ([5, 7], 0.5)])
def test_grid_code_equals_the_smoothed_book(lambdas, fwhm):
    Npos = 40
    Ng = sum(l * l for l in lambdas)
    sgb = smooth_gbook(gen_gbook_2d(lambdas, Ng, Npos), lambdas, fwhm)
    gc = SmoothedGridCode(lambdas, fwhm, Npos)
    gx, gy = np.meshgrid(np.arange(Npos), np.arange(Npos), indexing="ij")
    got = gc.at(gx.ravel(), gy.ravel())
    assert np.array_equal(got, sgb[:, gx.ravel(), gy.ravel()].T.astype(np.float32))


def test_grid_code_clips_like_grid_state_vec():
    gc = SmoothedGridCode([3, 4], 0.25, 12)
    assert np.array_equal(gc.at([-5, 30], [0, 11]), gc.at([0, 11], [0, 11]))


def _env(seed, amp=1.0, size=20):
    cfg = EnvConfig(size=size, observation_size=60, wall_resolution=4,
                    distal_amp=amp, goal_radius=1.0)
    return make_env(cfg, "continuous", seed=seed)


def test_read_batch_reproduces_offline_retrieval():
    rng = np.random.RandomState(0)
    envs = [_env(1000 + i) for i in range(30)]
    mem = SensoryKVMemory()
    for e in envs:
        mem.write(omni_key(e, e._goal), np.eye(30, dtype=np.float32)[len(mem.keys)])
    correct, own_s = [], []
    for i, e in enumerate(envs):
        cells = rng.randint(0, 20, size=(64, 2))
        psi = rng.uniform(-np.pi, np.pi, 64)
        views = np.stack([e.obs_at(tuple(c), p) for c, p in zip(cells, psi)])
        vals, s, has = read_batch([mem] * 64, views, psi, 30)
        assert has.all()
        correct.append(vals.argmax(1) == i)
        own_s.append(s[vals.argmax(1) == i])
    acc = np.concatenate(correct).mean()
    assert acc > 0.95, acc                       # offline §2.4: 0.985 at N = 30
    assert abs(np.concatenate(own_s).mean() - 0.5) < 0.05


def test_empty_memory_reads_zero():
    vals, s, has = read_batch([SensoryKVMemory()] * 3, np.ones((3, 60)),
                              np.zeros(3), 7)
    assert not has.any() and not vals.any() and not s.any()


def test_omni_key_is_one_value_per_direction_slice():
    e = _env(5)
    k = omni_key(e, e._goal)
    assert k.shape == (180,)
    # Without near walls the key IS the panorama.
    e._wall_code[:] = 0.0
    e._codebook = e._build_sensory_codebook(60)
    assert np.allclose(omni_key(e, (3, 3)), e._panorama)


needs_mlp = pytest.mark.skipif(not os.path.exists(GRID_MLP), reason="grid MLP checkpoint not on this machine")


@needs_mlp
def test_grid_mlp_points_across_an_arena():
    from hopfield_nav.memory.grid_mlp import load_grid_mlp
    mlp = load_grid_mlp(GRID_MLP)
    gc = SmoothedGridCode(LAMBDAS, 0.25, 1716)
    rng = np.random.RandomState(0)
    errs = []
    for _ in range(20):
        off = rng.randint(0, 1716 - 20, size=2)
        p = rng.randint(0, 20, size=(256, 2))
        g = rng.randint(0, 20, size=(256, 2))
        keep = (p != g).any(1)
        p, g = p[keep], g[keep]
        d = mlp.direction(torch.from_numpy(gc.at_local(p, off)),
                          torch.from_numpy(gc.at_local(g, off))).numpy()
        true = (g - p) / np.linalg.norm(g - p, axis=1, keepdims=True)
        errs.append(np.degrees(np.arccos(np.clip((d * true).sum(1), -1, 1))))
    errs = np.concatenate(errs)
    assert errs.mean() < 2.0, errs.mean()        # Phase-1 held-out: 0.48 deg
    assert np.percentile(errs, 99) < 15.0


@needs_mlp
def test_readout_points_at_the_stored_goal():
    from types import SimpleNamespace
    from hopfield_nav.memory.grid_mlp import load_grid_mlp
    gc = SmoothedGridCode(LAMBDAS, 0.25, 1716)
    ro = GridMemoryReadout(gc, load_grid_mlp(GRID_MLP))
    e, off = _env(7), (300, 900)
    vec = VecEnv(e, batch_size=16)
    vec.reset_all()
    mems = [SensoryKVMemory() for _ in range(16)]
    ch, q, has, c = ro.signal(mems, vec.obs_batch(), vec._heading_rad, vec.positions(), off)
    assert not has.any() and not ch.any() and not c.any()
    for m in mems:
        ro.write_goal(m, e, e._goal, off)
    ch, q, has, c = ro.signal(mems, vec.obs_batch(), vec._heading_rad, vec.positions(), off)
    pos = vec.positions()
    away = (pos != np.array(e._goal)).any(1)
    true = np.array(e._goal) - pos[away]
    true = true / np.linalg.norm(true, axis=1, keepdims=True)
    assert has.all() and np.allclose(q, ch)
    assert (np.degrees(np.arccos(np.clip((ch[away] * true).sum(1), -1, 1))) < 10).all()
    assert np.allclose(np.linalg.norm(ch[away], axis=1), 1.0, atol=1e-4)
    ro.scale_q_by_c = True
    ch2, *_ , c2 = ro.signal(mems, vec.obs_batch(), vec._heading_rad, vec.positions(), off)
    assert np.allclose(ch2, ch * c2[:, None])
