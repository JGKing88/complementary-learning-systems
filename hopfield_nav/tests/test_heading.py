"""Egocentric heading: the cone turns with the agent, and nothing else moves.

Heading is a continuous angle ψ, radians clockwise from North, and it follows
the direction the agent *actually* travelled. The only thing it changes anywhere
in the system is the sensory vector -- actions stay world-frame, the policy gets
no extra channel, and the sensory channel keeps its width.

Three claims are pinned here:

1. **Geometry.** The vectorized ray-caster reproduces the pre-heading scalar
   implementation exactly at ψ=0, and the precomputed cardinal codebook agrees
   with a live cast at every cardinal. The codebook is a speed path, not a
   second definition, so the two must never disagree.
2. **Dynamics.** ψ tracks realized displacement -- exactly cardinal after a
   discrete step, ``atan2(dx, dy)`` after a continuous one, and unchanged when a
   step is clipped by a wall.
3. **Confinement.** With ``egocentric_heading=False`` every observation is the
   North one, which is what makes the flag a faithful reproduction of every run
   from before headings were wired up.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import (
    CARDINAL_ACTIONS, CARDINAL_RADIANS, FOVEAL_HALF_ANGLE_DEG, GridEnv,
    cardinal_index, make_env, nearest_heading, raycast_codes,
)
from hopfield_nav.world.vec_env import ContinuousVecEnv, VecEnv

OBS = 12


# ---------------------------------------------------------------------------
# 1. Geometry
# ---------------------------------------------------------------------------

def _scalar_segment_code(wall_code, size, cx, cy, dx, dy):
    """The pre-heading ``GridEnv._raycast_segment_code``, kept verbatim.

    This is a frozen reference, not live code: it is what the sensory codebook
    meant before the cone could turn. Do not "fix" or refactor it to match the
    vectorized version -- the whole point is that it was written independently.
    """
    best_t, best_wall, best_seg = np.inf, -1, 0
    if dy > 0.0:                                    # N: y = size - 0.5
        t = (size - 0.5 - cy) / dy
        if 0.0 <= t < best_t:
            x_hit = cx + t * dx
            if -0.5 <= x_hit <= size - 0.5:
                best_t, best_wall = t, 0
                best_seg = int(np.clip(np.floor(x_hit + 0.5), 0, size - 1))
    if dx > 0.0:                                    # E: x = size - 0.5
        t = (size - 0.5 - cx) / dx
        if 0.0 <= t < best_t:
            y_hit = cy + t * dy
            if -0.5 <= y_hit <= size - 0.5:
                best_t, best_wall = t, 1
                best_seg = int(np.clip(np.floor(y_hit + 0.5), 0, size - 1))
    if dy < 0.0:                                    # S: y = -0.5
        t = (-0.5 - cy) / dy
        if 0.0 <= t < best_t:
            x_hit = cx + t * dx
            if -0.5 <= x_hit <= size - 0.5:
                best_t, best_wall = t, 2
                best_seg = int(np.clip(np.floor(x_hit + 0.5), 0, size - 1))
    if dx < 0.0:                                    # W: x = -0.5
        t = (-0.5 - cx) / dx
        if 0.0 <= t < best_t:
            y_hit = cy + t * dy
            if -0.5 <= y_hit <= size - 0.5:
                best_t, best_wall = t, 3
                best_seg = int(np.clip(np.floor(y_hit + 0.5), 0, size - 1))
    if best_wall < 0:
        return 0.0
    return float(wall_code[best_wall, best_seg])


def _scalar_codebook(wall_code, size, n_rays, psi=0.0):
    """Every cell's view at one heading, via the frozen scalar reference."""
    half = np.deg2rad(FOVEAL_HALF_ANGLE_DEG)
    angles = psi + (-half + (np.arange(n_rays) + 0.5) * (2 * half / n_rays))
    sin_t, cos_t = np.sin(angles), np.cos(angles)
    cb = np.zeros((size, size, n_rays), dtype=np.float32)
    for cx in range(size):
        for cy in range(size):
            for i in range(n_rays):
                cb[cx, cy, i] = _scalar_segment_code(
                    wall_code, size, cx, cy, sin_t[i], cos_t[i])
    return cb


@pytest.mark.parametrize("size,n_rays", [(4, 8), (6, 12), (8, 60)])
def test_north_slab_reproduces_the_pre_heading_codebook(size, n_rays):
    """ψ=0 is bit-for-bit what the fixed-North cone produced.

    Not "close" -- identical. The rotation is an added term on the ray angle and
    adding 0.0 is exact, so any drift here is a real change in the geometry.
    """
    env = GridEnv(size=size, observation_size=n_rays, seed=0)
    ref = _scalar_codebook(env._wall_code, size, n_rays)
    assert np.array_equal(ref, env._codebook[:, :, 0, :])


@pytest.mark.parametrize("k", range(4))
def test_each_cardinal_slab_matches_an_independent_cast(k):
    env = GridEnv(size=6, observation_size=OBS, seed=7)
    ref = _scalar_codebook(env._wall_code, 6, OBS, float(CARDINAL_RADIANS[k]))
    assert np.array_equal(ref, env._codebook[:, :, k, :])


@pytest.mark.parametrize("psi", [0.3, 1.0, -2.2, 2.9])
def test_offcardinal_cast_matches_the_scalar_reference(psi):
    """Angles between the cardinals are cast live, and must agree too."""
    env = GridEnv(size=7, observation_size=OBS, seed=11)
    ref = _scalar_codebook(env._wall_code, 7, OBS, psi)
    got = np.stack([[env.obs_at((x, y), psi) for y in range(7)]
                    for x in range(7)])
    assert np.array_equal(ref, got)


def test_codebook_gather_and_live_cast_agree_at_cardinals():
    """The fast path is a speed choice, never a behavioral one."""
    env = GridEnv(size=6, observation_size=OBS, seed=3)
    for k, psi in enumerate(CARDINAL_RADIANS):
        for pos in [(0, 0), (3, 2), (5, 5)]:
            gathered = env._codebook[pos[0], pos[1], k]
            cast = raycast_codes(env._wall_code, env.size, pos[0], pos[1],
                                 float(psi), OBS)[0]
            assert np.array_equal(gathered, cast), (k, pos)


def test_cardinal_index_resolves_exactly_the_cardinals():
    assert [cardinal_index(p) for p in CARDINAL_RADIANS] == [0, 1, 2, 3]
    # atan2 returns (-π, π]; the modulo has to absorb that.
    assert cardinal_index(-np.pi / 2) == 3
    assert cardinal_index(3 * np.pi / 2) == 3
    for psi in (0.1, np.pi / 3, -1.0):
        assert cardinal_index(psi) == -1


def test_nearest_heading_snaps_and_reports_stillness():
    for k, a in enumerate(CARDINAL_ACTIONS):
        assert nearest_heading(a) == k
        assert nearest_heading((3 * a[0], 3 * a[1])) == k   # speed > 1
    assert nearest_heading((0, 0)) == -1
    assert nearest_heading((0.9, 0.1)) == 1                 # ENE -> East
    assert np.array_equal(nearest_heading([(0, 1), (0, 0), (-1, 0)]),
                          [0, -1, 3])


# ---------------------------------------------------------------------------
# 2. Dynamics
# ---------------------------------------------------------------------------

def test_discrete_step_faces_exactly_the_direction_it_moved():
    """Exactly, so the observation stays on the codebook gather."""
    env = make_env(EnvConfig(size=6, observation_size=OBS), "discrete", seed=5)
    for k, a in enumerate(CARDINAL_ACTIONS):
        env.set_position((3, 3))
        env.step(a)
        assert cardinal_index(env.heading) == k
        assert env.heading == float(CARDINAL_RADIANS[k])


def test_blocked_step_leaves_heading_alone():
    """Heading follows realized displacement, so a wall cannot spin the agent."""
    env = make_env(EnvConfig(size=6, observation_size=OBS), "discrete", seed=5)
    env.set_position((0, 5))                 # NW corner
    env.step(CARDINAL_ACTIONS[1])            # face East by moving there
    facing = env.heading
    obs_before = env.obs()
    env.step(CARDINAL_ACTIONS[0])            # North: clipped by the top wall
    assert env.heading == facing
    assert np.array_equal(env.obs(), obs_before)


def test_continuous_step_faces_atan2_of_realized_displacement():
    cfg = EnvConfig(size=6, observation_size=OBS, movement_mode="continuous")
    env = make_env(cfg, "continuous", seed=5)
    env.set_position((3, 3))
    env.step(np.array([0.6, -0.8]))
    assert env.heading == pytest.approx(float(np.arctan2(0.6, -0.8)))
    # ...and that is genuinely off-cardinal, i.e. heading really is continuous.
    assert cardinal_index(env.heading) == -1


def test_continuous_vec_env_tracks_heading_per_episode():
    """ContinuousVecEnv had no heading state at all before this."""
    cfg = EnvConfig(size=6, observation_size=OBS, movement_mode="continuous")
    base = make_env(cfg, "continuous", seed=5)
    vec = ContinuousVecEnv(base, batch_size=8, scale=1.0)
    vec.reset_all()
    assert np.array_equal(vec._heading_rad, np.zeros(8))     # North at reset

    acts = np.random.RandomState(0).randn(8, 2)
    vec.step_batch(acts)
    expected = np.arctan2(acts[:, 0], acts[:, 1])
    # Episodes whose step was clipped at a wall keep their old heading, so
    # compare only the ones that actually moved freely.
    free = np.isclose(vec._heading_rad, expected)
    assert free.sum() >= 5, vec._heading_rad
    assert (cardinal_index(vec._heading_rad) < 0).any(), "no continuous heading"


def test_vec_env_heading_is_cardinal_and_obs_follows():
    env = make_env(EnvConfig(size=6, observation_size=OBS), "discrete", seed=5)
    vec = VecEnv(env, batch_size=8)
    vec.reset_all()
    before = vec.obs_batch().copy()
    vec.step_batch(np.full(8, 1, dtype=np.int32))            # all East
    assert set(np.unique(cardinal_index(vec._heading_rad))) <= {1}
    assert not np.array_equal(before, vec.obs_batch())


# ---------------------------------------------------------------------------
# 3. What heading does, and does not, change
# ---------------------------------------------------------------------------

def _views_by_heading(env, pos=(3, 3)):
    """The same CELL seen from four headings -- position held fixed.

    Stepping would also move the agent, which would compare four different
    cells and prove nothing about heading.
    """
    out = []
    env.set_position(pos)
    for psi in CARDINAL_RADIANS:
        env._heading_rad = float(psi)
        out.append(tuple(env.obs()))
    return out


def test_egocentric_cell_looks_different_from_each_heading():
    env = make_env(EnvConfig(size=6, observation_size=OBS), "discrete", seed=5)
    assert len(set(_views_by_heading(env))) == 4


def test_fixed_heading_pins_every_view_to_north():
    cfg = EnvConfig(size=6, observation_size=OBS, egocentric_heading=False)
    env = make_env(cfg, "discrete", seed=5)
    views = _views_by_heading(env)
    assert len(set(views)) == 1

    ego = make_env(EnvConfig(size=6, observation_size=OBS), "discrete", seed=5)
    assert views[0] == _views_by_heading(ego)[0]     # and it IS the North view


def test_observation_width_is_unchanged_by_heading():
    """No new channel, no wider sensory vector -- checkpoint shapes are safe."""
    for ego in (True, False):
        cfg = EnvConfig(size=6, observation_size=OBS, egocentric_heading=ego)
        env = make_env(cfg, "discrete", seed=0)
        assert env.obs().shape == (OBS,)
        vec = VecEnv(env, batch_size=4)
        vec.reset_all()
        assert vec.obs_batch().shape == (4, OBS)


@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_egocentric_heading_reaches_env_and_vec(movement_mode):
    """EnvConfig -> make_env -> VecEnv, the plumbing test_env_config pins."""
    for value in (True, False):
        cfg = EnvConfig(size=6, observation_size=OBS,
                        movement_mode=movement_mode,
                        egocentric_heading=value)
        env = make_env(cfg, movement_mode, seed=0)
        assert env.egocentric_heading == value
        vec = (ContinuousVecEnv(env, batch_size=4) if movement_mode == "continuous"
               else VecEnv(env, batch_size=4))
        assert vec.egocentric_heading == value


def test_omni_obs_is_the_four_cardinal_views_concatenated():
    """The scaffold's stand-in -- see fit_env_assoc, which explains the patch."""
    env = GridEnv(size=5, observation_size=OBS, seed=2)
    omni = env.omni_obs_at((2, 3))
    assert omni.shape == (4 * OBS,)
    for k in range(4):
        assert np.array_equal(omni[k * OBS:(k + 1) * OBS],
                              env._codebook[2, 3, k])
    assert np.array_equal(env.omni_obs_all()[2, 3], omni)
