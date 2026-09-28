"""Distal panorama (``EnvConfig.distal_amp``): a skyline at infinity.

Four claims are pinned here:

1. **Off is off.** ``distal_amp=0`` builds no panorama and every view is
   bit-identical to the no-knob env; turning it on leaves the wall code and the
   goal draw untouched (the panorama has its own seed stream).
2. **Position-invariant.** The panorama's contribution to a view depends on the
   ray's absolute angle only -- the same from every cell -- and the overlapping
   parts of two cardinal cones read the same values.
3. **One definition.** The codebook, a live off-cardinal cast, and both vec
   envs agree, so no path sees a different world.
4. **It does its job.** The four-heading view at the goal is an env key that
   argmax retrieves from every cell, which the near walls alone cannot do.
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import (
    CARDINAL_RADIANS, PANORAMA_BINS, cone_offsets, make_env, panorama_bins,
    raycast_codes,
)
from hopfield_nav.world.vec_env import ContinuousVecEnv, VecEnv

OBS = 60


def _env(amp, seed=11, size=10, mode="discrete"):
    cfg = EnvConfig(size=size, observation_size=OBS, wall_resolution=4,
                    distal_amp=amp)
    return make_env(cfg, mode, seed=seed)


def test_zero_amp_is_bit_identical_and_has_no_panorama():
    off = _env(0.0)
    assert off._panorama is None
    size = off.size
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    ref = raycast_codes(off._wall_code, size, np.repeat(gx.ravel(), 4),
                        np.repeat(gy.ravel(), 4), np.tile(CARDINAL_RADIANS, size * size),
                        OBS, 4).reshape(off._codebook.shape)
    assert np.array_equal(off._codebook, ref)


def test_panorama_leaves_walls_and_goal_draw_alone():
    off, on = _env(0.0), _env(1.0)
    assert np.array_equal(off._wall_code, on._wall_code)
    assert off._goal == on._goal and off._pos == on._pos
    assert on._panorama.shape == (PANORAMA_BINS,)
    assert set(np.unique(on._panorama)) <= {-1.0, 1.0}


def test_panorama_differs_between_envs_and_scales_with_amp():
    a, b = _env(1.0, seed=11), _env(1.0, seed=12)
    assert not np.array_equal(a._panorama, b._panorama)
    assert np.allclose(_env(0.5, seed=11)._panorama, 0.5 * a._panorama)


def test_contribution_is_the_same_from_every_cell():
    off, on = _env(0.0), _env(1.0)
    delta = on._codebook - off._codebook                # (S, S, 4, OBS)
    assert np.allclose(delta, delta[0, 0][None, None])
    ang = CARDINAL_RADIANS[:, None] + cone_offsets(OBS)[None, :]
    assert np.allclose(delta[0, 0], on._panorama[panorama_bins(ang)])


def test_overlapping_cones_read_the_same_direction_alike():
    on = _env(1.0)
    ang = np.mod(np.rad2deg(CARDINAL_RADIANS[:, None] + cone_offsets(OBS)[None, :]), 360)
    vals = on._panorama[panorama_bins(np.deg2rad(ang))]
    seen = {}
    for a, v in zip(np.round(ang.ravel(), 6), vals.ravel()):
        assert seen.setdefault(a, v) == v
    assert len(seen) == PANORAMA_BINS    # the omni view covers the whole skyline


def test_live_offcardinal_cast_includes_the_panorama():
    off, on = _env(0.0), _env(1.0)
    psi = 0.7
    ang = psi + cone_offsets(OBS)
    got = on.obs_at((3, 4), psi) - off.obs_at((3, 4), psi)
    assert np.allclose(got, on._panorama[panorama_bins(ang)])


def test_vec_envs_see_the_panorama():
    on = _env(1.0)
    vec = VecEnv(on, batch_size=4)
    vec.reset_all()
    for i in range(4):
        assert np.array_equal(vec.obs_batch()[i],
                              on.obs_at(tuple(vec._pos[i]), vec._heading_rad[i]))
    con = _env(1.0, mode="continuous")
    cvec = ContinuousVecEnv(con, batch_size=6, scale=1.0)
    cvec.reset_all()
    cvec.step_batch(np.random.RandomState(0).randn(6, 2))
    obs = cvec.obs_batch()
    for i in range(6):
        assert np.allclose(obs[i], con.obs_at(tuple(cvec._pos[i]), cvec._heading_rad[i]))


def test_goal_view_is_an_env_key_from_every_cell():
    def keys_and_views(amp):
        envs = [_env(amp, seed=s, size=20) for s in range(100, 108)]
        views = np.stack([e.omni_obs_all().reshape(400, -1) for e in envs])
        keys = np.stack([e.omni_obs_at(e._goal) for e in envs])
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        sims = unit(views) @ unit(keys).T                # (E, cells, E)
        return (sims.argmax(-1) == np.arange(len(envs))[:, None]).mean()
    assert keys_and_views(1.0) == 1.0
    assert keys_and_views(0.0) < 0.5
