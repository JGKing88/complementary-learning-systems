"""Tests for the env generator: domains, separation, and the split entry points.

The properties worth pinning here are the ones that make a split *mean*
something. A generator that quietly returns overlapping envs, or that produces
different envs on a second invocation, would look fine at every call site and be
wrong everywhere downstream.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from hopfield_nav.config import EnvConfig, VectorHashConfig
from hopfield_nav.world import domains as dom
from hopfield_nav.world import generate as gen
from hopfield_nav.world.env import GridEnv
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import EnvSpec, GeneratedSplit, TraitDomains

LAMBDAS = [3, 4, 5]          # Npos = 60, small enough to build per session
SIZE = 6
OBS = 12


@pytest.fixture(scope="module")
def field():
    vh = VectorHash(VectorHashConfig(lambdas=LAMBDAS, static_vectorhash=True))
    vh.build_scaffold()
    torch.manual_seed(0)
    enc = torch.nn.Linear(vh.Ng, 16)
    enc.eval()
    vh.precompute_encoded_phi(enc, 0.25, device="cpu")
    return vh


@pytest.fixture
def env_cfg():
    return EnvConfig(size=SIZE, observation_size=OBS)


@pytest.fixture
def domains():
    return TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                        goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def test_stable_hash_is_stable_across_processes():
    """The one that catches an accidental `hash()`.

    Python salts `hash()` per interpreter run, so a generator seeded from it
    would silently produce a different world on every launch -- and would pass
    any single-process test.
    """
    here = dom.stable_hash("place", 7, "train")
    out = subprocess.run(
        [sys.executable, "-c",
         "from hopfield_nav.world.domains import stable_hash;"
         "print(stable_hash('place', 7, 'train'))"],
        capture_output=True, text=True, check=True,
        env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin",
             "HOME": "/tmp", "PYTHONPATH": "."},
    )
    assert int(out.stdout.strip()) == here


def test_trait_streams_are_independent():
    """Changing one trait's tick must not move another trait's draws."""
    a = dom.trait_rng(5, "place", tick=0).randint(0, 10_000, size=8)
    b = dom.trait_rng(5, "place", tick=0).randint(0, 10_000, size=8)
    assert np.array_equal(a, b)
    # A different tick on `goal` leaves `place` alone -- streams are derived per
    # key, not advanced from a shared one.
    c = dom.trait_rng(5, "place", tick=0).randint(0, 10_000, size=8)
    dom.trait_rng(5, "goal", tick=3).randint(0, 10_000, size=8)
    assert np.array_equal(a, c)
    assert not np.array_equal(a, dom.trait_rng(5, "place", tick=1)
                              .randint(0, 10_000, size=8))


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

def test_goal_complement_is_an_involution():
    for d in (dom.Ring(1), dom.Interior(2), dom.Cells(frozenset({(0, 0), (1, 1)}))):
        assert d.complement().complement() == d


def test_ring_and_interior_partition_the_arena():
    ring, interior = dom.Ring(1).cells(SIZE), dom.Interior(1).cells(SIZE)
    assert ring | interior == frozenset((x, y) for x in range(SIZE)
                                        for y in range(SIZE))
    assert not (ring & interior)


@pytest.mark.parametrize("domain, why", [
    (dom.Anywhere(), "whole scaffold"),
    (dom.AnyCells(), "every cell"),
    (dom.SeedRange(0, 10), "wall patterns"),
    (dom.Sizes((20,)), "env size"),
])
def test_unbounded_domains_refuse_to_complement(domain, why):
    """OOD is only meaningful where the universe is bounded.

    Asking for place-OOD on a model trained everywhere, or wall-OOD at all, has
    to fail where the mistake is rather than yield a silently empty set.
    """
    with pytest.raises(ValueError):
        domain.complement()


def test_quadrant_complement_needs_a_size():
    with pytest.raises(ValueError, match="arena size"):
        dom.Quadrant(0).complement()
    assert dom.complement_for(dom.Quadrant(0), SIZE).cells(SIZE) == (
        frozenset((x, y) for x in range(SIZE) for y in range(SIZE))
        - dom.Quadrant(0).cells(SIZE))


def test_rect_contains_the_footprint_not_just_the_corner():
    """`Rect` means the env fits inside, which is what "place envs here" means."""
    r = dom.Rect(10, 10, 20, 20)
    assert r.contains((10, 10), size=SIZE, Npos=60)
    assert r.contains((24, 24), size=SIZE, Npos=60)       # 24+6 == 30, just fits
    assert not r.contains((25, 25), size=SIZE, Npos=60)   # would poke out


def test_domain_json_round_trip():
    for d in (dom.Anywhere(), dom.Rect(1, 2, 3, 4),
              dom.Rect(1, 2, 30, 40).complement(5), dom.AnyCells(),
              dom.Ring(2), dom.Interior(1), dom.Quadrant(3),
              dom.Cells(frozenset({(1, 2)})), dom.SeedRange(0, 9),
              dom.Sizes((6, 8))):
        assert dom.from_json(json.loads(json.dumps(d.to_json()))) == d


# ---------------------------------------------------------------------------
# Wall bits -- the structural claim from EVAL_SPLITS_DESIGN §1.7
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [4, 6, 8])
def test_south_wall_is_perceptually_dead_under_fixed_heading(size):
    """With the cone pinned North, no ray can ever hit wall 2.

    Every ray has dy = cos(theta) >= 0.5 > 0 at +/-60 deg, so a Hamming distance
    over all 4*size bits would count size silent no-ops -- a quarter of the
    "identity" of an env that nothing can observe.
    """
    live = np.array(gen.live_wall_bits(size, OBS, False), dtype=bool)
    assert live.shape == (4, size)
    assert live[2].sum() == 0, "South wall should contribute no live bits"
    for w in (0, 1, 3):
        assert live[w].all(), f"wall {w} has dead bits: {live[w]}"
    assert live.sum() == 3 * size


@pytest.mark.parametrize("size", [4, 6, 8])
def test_egocentric_heading_makes_every_wall_live(size):
    """Once the cone turns with the agent, the South wall is observable.

    Env identity goes from 3*size bits to the full 4*size. That can only *raise*
    the Hamming distance between two envs, so a split that was separated under
    the fixed-heading count stays separated under this one -- which is why the
    change needs no re-derivation of margins.
    """
    live = np.array(gen.live_wall_bits(size, OBS, True), dtype=bool)
    assert live.shape == (4, size)
    assert live.all(), f"dead bits under egocentric heading: {live}"
    assert live.sum() == 4 * size

    fixed = np.array(gen.live_wall_bits(size, OBS, False), dtype=bool)
    assert (live >= fixed).all(), "egocentric must not kill a bit fixed could see"


def test_wall_hamming_counts_live_bits_only():
    for ego, cap in ((False, 3 * SIZE), (True, 4 * SIZE)):
        d = gen.wall_hamming(1, 2, SIZE, OBS, ego)
        assert 0 <= d <= cap
        assert gen.wall_hamming(7, 7, SIZE, OBS, ego) == 0

    # Turning the cone can only reveal differences, never hide them.
    assert (gen.wall_hamming(1, 2, SIZE, OBS, True)
            >= gen.wall_hamming(1, 2, SIZE, OBS, False))


def test_wall_code_matches_what_the_env_builds():
    """wall_code_for must reproduce GridEnv's first RNG draw exactly."""
    env = GridEnv(size=SIZE, observation_size=OBS, seed=1234)
    assert np.array_equal(gen.wall_code_for(1234, SIZE), env._wall_code)


# ---------------------------------------------------------------------------
# Toroidal separation
# ---------------------------------------------------------------------------

def test_gap_is_negative_when_footprints_overlap():
    assert gen.toroidal_gap((0, 0), 6, (0, 0), 6, 60) < 0
    assert gen.toroidal_gap((0, 0), 6, (3, 3), 6, 60) < 0


def test_gap_separates_on_a_single_axis():
    """Boxes far apart in x are separated however they line up in y."""
    assert gen.toroidal_gap((0, 0), 6, (20, 0), 6, 60) == 14


def test_gap_wraps_at_the_seam():
    """The scaffold is a torus: x~0 and x~Npos-1 are the same coordinate.

    A flat check calls the measured worst case -- (1715,987) vs (4,989) at
    cos +0.972 -- 1711 cells apart. This is the check that catches that.
    """
    period = 60
    near = gen.toroidal_gap((58, 0), 2, (0, 0), 2, period)
    assert near == 0, f"seam-adjacent envs should touch, got gap {near}"
    assert gen.toroidal_gap((55, 0), 2, (0, 0), 2, period) == 3


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generate_split_is_deterministic(field, env_cfg, domains):
    a = gen.generate_split(field, env_cfg, domains, 4, 2, seed=3, margin=6)
    b = gen.generate_split(field, env_cfg, domains, 4, 2, seed=3, margin=6)
    assert a.train == b.train and a.base_val == b.base_val


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_split_separates_on_every_trait(field, env_cfg, domains, seed):
    split = gen.generate_split(field, env_cfg, domains, 4, 2, seed=seed, margin=6)
    gen.verify_split(split, env_cfg)          # raises on any violation
    train_goals = {t.goal for t in split.train}
    train_offs = {t.offset for t in split.train}
    train_seeds = {t.wall_seed for t in split.train}
    for v in split.base_val:
        assert v.goal not in train_goals
        assert v.offset not in train_offs
        assert v.wall_seed not in train_seeds
        for t in split.train:
            assert gen.toroidal_gap(v.offset, v.size, t.offset, t.size,
                                    split.period) >= split.margin


def test_capacity_error_names_the_knob(field, env_cfg):
    """A region too small must fail fast, not spin in rejection sampling."""
    tight = TraitDomains(place=dom.Rect(0, 0, 12, 12), wall=dom.SeedRange(0, 1000),
                         goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    with pytest.raises(ValueError, match="lower the margin|holds"):
        gen.generate_split(field, env_cfg, tight, n_train=20, n_val=8,
                           seed=0, margin=20)


def test_goal_partition_when_refresh_is_on(field, env_cfg, domains):
    """Refresh would exhaust the grid, so the train cell set is capped up front."""
    split = gen.generate_split(field, env_cfg, domains, 4, 2, seed=0,
                               margin=6, refresh_goal=True, val_frac=0.25)
    all_cells = dom.AnyCells().cells(SIZE)
    assert split.goal_cells_train | split.goal_cells_val == all_cells
    assert not (split.goal_cells_train & split.goal_cells_val)
    assert len(split.goal_cells_val) == round(0.25 * len(all_cells))


def test_goal_val_is_the_complement_when_refresh_is_off(field, env_cfg, domains):
    split = gen.generate_split(field, env_cfg, domains, 4, 2, seed=0, margin=6)
    assert not (split.goal_cells_train & split.goal_cells_val)
    for v in split.base_val:
        assert v.goal in split.goal_cells_val


def test_spec_round_trip_rebuilds_identical_envs(field, env_cfg, domains):
    """Equal seeds is not enough -- the wall *arrays* must match."""
    split = gen.generate_split(field, env_cfg, domains, 3, 2, seed=11, margin=6)
    rebuilt = GeneratedSplit.from_json(json.loads(json.dumps(split.to_json())))
    assert rebuilt.train == split.train
    assert rebuilt.goal_cells_val == split.goal_cells_val
    a = gen.build_envs(split.base_val, env_cfg, "discrete")
    b = gen.build_envs(rebuilt.base_val, env_cfg, "discrete")
    for ea, eb, spec in zip(a, b, split.base_val):
        assert np.array_equal(ea._wall_code, eb._wall_code)
        assert np.array_equal(ea._codebook, eb._codebook)
        assert ea.goal_location == eb.goal_location == spec.goal


def test_build_envs_applies_the_spec_goal(field, env_cfg, domains):
    split = gen.generate_split(field, env_cfg, domains, 3, 2, seed=2, margin=6)
    for env, spec in zip(gen.build_envs(split.train, env_cfg, "discrete"),
                         split.train):
        assert env.goal_location == spec.goal
        assert env.size == spec.size


def test_set_goal_rejects_out_of_bounds():
    env = GridEnv(size=SIZE, observation_size=OBS, seed=0)
    with pytest.raises(ValueError, match="out of bounds"):
        env.set_goal((SIZE, 0))


# ---------------------------------------------------------------------------
# derive_margin
# ---------------------------------------------------------------------------

class _SyntheticField:
    """A field whose embedding decorrelates at a known length.

    Random Fourier features with Gaussian frequencies of scale ``1/length``, so
    cosine similarity follows ``exp(-d^2 / 2 length^2)`` -- a field that actually
    decorrelates, unlike a sum of fixed cosines, which stays quasi-periodic
    forever and would make this test assert on the wrong thing.
    """

    def __init__(self, Npos=200, length=20, dim=512, seed=0):
        self.Npos, self.lambdas = Npos, [Npos]
        rng = np.random.RandomState(seed)
        w = rng.standard_normal((2, dim)).astype(np.float32) / float(length)
        b = rng.uniform(0, 2 * np.pi, size=dim).astype(np.float32)
        xs, ys = np.meshgrid(np.arange(Npos), np.arange(Npos), indexing="ij")
        coords = np.stack([xs, ys], -1).astype(np.float32)
        self.encoded_Phi = np.cos(coords @ w + b).astype(np.float32)


@pytest.mark.parametrize("length, lo, hi", [(10, 10, 60), (25, 30, 140)])
def test_derive_margin_tracks_the_correlation_length(length, lo, hi):
    """A longer correlation length must yield a larger margin."""
    f = _SyntheticField(length=length)
    m = gen.derive_margin(f, np.random.RandomState(0), quantile=0.99,
                          threshold=0.15, n_pairs=1024)
    assert lo <= m <= hi, f"length={length} gave margin {m}"


def test_derive_margin_raises_when_the_field_never_decorrelates():
    """fwhm_ratio=0.5 plateaus around +0.12; a hardcoded margin would hide it."""
    f = _SyntheticField(length=10_000)      # correlated everywhere
    with pytest.raises(RuntimeError, match="does not decorrelate"):
        gen.derive_margin(f, np.random.RandomState(0), n_pairs=256)


# ---------------------------------------------------------------------------
# make_val_set
# ---------------------------------------------------------------------------

def test_same_level_reuses_training_values(field, env_cfg, domains):
    split = gen.generate_split(field, env_cfg, domains, 6, 2, seed=4, margin=6)
    vs = gen.make_val_set(split, 4, {"place": "same", "wall": "same",
                                     "goal": "same"}, seed=9)
    assert {v.offset for v in vs} <= split.used["place"]
    assert {v.wall_seed for v in vs} <= split.used["wall"]
    assert {v.goal for v in vs} <= split.used["goal"]


def test_held_out_level_avoids_everything_training_used(field, env_cfg, domains):
    split = gen.generate_split(field, env_cfg, domains, 4, 2, seed=5, margin=6)
    vs = gen.make_val_set(split, 3, {"place": "held_out", "wall": "held_out",
                                     "goal": "held_out"}, seed=9)
    for v in vs:
        assert v.wall_seed not in split.used["wall"]
        assert v.goal not in split.used["goal"]
        for o in split.used["place"]:
            assert gen.toroidal_gap(v.offset, v.size, o, v.size,
                                    split.period) >= split.margin


def test_mix_and_match_is_per_trait(field, env_cfg, domains):
    """'train scaffold locations, val goals and patterns' -- the isolation case."""
    split = gen.generate_split(field, env_cfg, domains, 6, 2, seed=6, margin=6)
    vs = gen.make_val_set(split, 3, {"place": "same", "wall": "held_out",
                                     "goal": "held_out"}, seed=9)
    for v in vs:
        assert v.offset in split.used["place"]
        assert v.wall_seed not in split.used["wall"]
        assert v.goal not in split.used["goal"]


def test_place_ood_lands_outside_the_declared_region(field, env_cfg):
    region = dom.Rect(0, 0, 30, 30)
    domains = TraitDomains(place=region, wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, env_cfg, domains, 3, 2, seed=7, margin=4)
    for t in split.train:
        assert region.contains(t.offset, SIZE, field.Npos)
    vs = gen.make_val_set(split, 3, {"place": "ood", "wall": "held_out",
                                     "goal": "held_out"}, seed=9)
    for v in vs:
        assert not region.contains(v.offset, SIZE, field.Npos)


def test_goal_ood_uses_the_region_complement(field, env_cfg):
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.Ring(1), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, env_cfg, domains, 4, 2, seed=8, margin=6)
    for t in split.train:
        assert t.goal in dom.Ring(1).cells(SIZE)
    vs = gen.make_val_set(split, 3, {"place": "held_out", "wall": "held_out",
                                     "goal": "ood"}, seed=9)
    interior = dom.Interior(1).cells(SIZE)
    for v in vs:
        assert v.goal in interior


def test_ood_on_an_unrestricted_trait_is_an_error(field, env_cfg, domains):
    split = gen.generate_split(field, env_cfg, domains, 3, 2, seed=0, margin=6)
    with pytest.raises(ValueError, match="nothing outside"):
        gen.make_val_set(split, 2, {"place": "ood"}, seed=1)
    with pytest.raises(ValueError, match="no bounded universe"):
        gen.make_val_set(split, 2, {"wall": "ood"}, seed=1)
    with pytest.raises(ValueError, match="only level 'same'"):
        gen.make_val_set(split, 2, {"size": "held_out"}, seed=1)


def test_size_ood_is_named_outright(field, env_cfg, domains):
    """Size has no bounded universe, so an OOD size is passed, not derived."""
    split = gen.generate_split(field, env_cfg, domains, 3, 2, seed=0, margin=6)
    vs = gen.make_val_set(split, 2, {"place": "held_out", "wall": "held_out",
                                     "goal": "held_out"}, seed=1, size=4)
    assert all(v.size == 4 for v in vs)
    for env in gen.build_envs(vs, env_cfg, "discrete"):
        assert env.size == 4


def test_unknown_level_is_rejected(field, env_cfg, domains):
    split = gen.generate_split(field, env_cfg, domains, 3, 2, seed=0, margin=6)
    with pytest.raises(ValueError, match="unknown level"):
        gen.make_val_set(split, 2, {"place": "novel"}, seed=1)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_report_but_do_not_gate(field, env_cfg, domains):
    """A high cosine must not change which envs come out.

    The split is derivable from coordinates alone; the cosine numbers are
    evidence about it, never an input to it.
    """
    a = gen.generate_split(field, env_cfg, domains, 4, 2, seed=12, margin=6,
                           diagnostics=True)
    b = gen.generate_split(field, env_cfg, domains, 4, 2, seed=12, margin=6,
                           diagnostics=False)
    assert a.train == b.train and a.base_val == b.base_val
    assert b.diagnostics == {}
    c = a.diagnostics["cosine"]
    assert len(c["per_env_max"]) == len(a.base_val)
    assert -1.0 <= c["max"] <= 1.0
    assert a.diagnostics["min_place_gap"] >= a.margin


# ---------------------------------------------------------------------------
# Dense packing -- the case unit tests with room to spare kept missing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wide_field():
    """Npos=132, big enough that a margin can actually bind."""
    vh = VectorHash(VectorHashConfig(lambdas=[11, 12], static_vectorhash=True))
    vh.build_scaffold()
    torch.manual_seed(0)
    enc = torch.nn.Linear(vh.Ng, 16)
    enc.eval()
    vh.precompute_encoded_phi(enc, 0.25, device="cpu")
    return vh


def test_dense_packing_still_places_everything(wide_field):
    """15 envs at margin 20 on Npos=132 -- most of the scaffold is excluded.

    Every placed env blocks a (size + 2*margin)^2 zone, so twelve of them cover
    27648 cells of a 17424-cell scaffold. Uniform rejection sampling cannot find
    the remaining slots; the lattice fallback has to.
    """
    env_cfg = EnvConfig(size=8, observation_size=OBS)
    domains = TraitDomains(dom.Anywhere(), dom.SeedRange(0, 10_000_000),
                           dom.AnyCells(), dom.Sizes((8,)))
    split = gen.generate_split(wide_field, env_cfg, domains, 12, 3,
                               seed=42, margin=20)
    assert len(split.train) == 12 and len(split.base_val) == 3
    gen.verify_split(split, env_cfg)


def test_lattice_slots_clear_the_torus_seam():
    """A lattice on a torus must not let its last slot touch its first.

    At Npos=132, pitch=41 the naive slots are 0/41/82/123 -- and 123 is 9 cells
    from 0, not 123. That single wrapped pair silently violates the spacing the
    lattice exists to guarantee.
    """
    naive = list(range(0, 125, 41))
    assert naive == [0, 41, 82, 123]
    assert gen.axis_separation(123, 8, 0, 8, 132) == 1      # the seam
    slots = gen._axis_slots(0, 124, 41, size=8, spacing=20, period=132)
    assert slots == [0, 41, 82]
    for a in slots:
        for b in slots:
            if a != b:
                assert gen.axis_separation(a, 8, b, 8, 132) >= 20


def test_held_out_val_is_disjoint_on_every_trait(wide_field):
    """The property the whole design exists for, stated once, at scale."""
    env_cfg = EnvConfig(size=8, observation_size=OBS)
    domains = TraitDomains(dom.Anywhere(), dom.SeedRange(0, 10_000_000),
                           dom.AnyCells(), dom.Sizes((8,)))
    split = gen.generate_split(wide_field, env_cfg, domains, 12, 3,
                               seed=42, margin=12)
    vs = gen.make_val_set(split, 5, {"place": "held_out", "wall": "held_out",
                                     "goal": "held_out"}, seed=99)
    assert not {v.wall_seed for v in vs} & split.used["wall"]
    assert not {v.offset for v in vs} & split.used["place"]
    assert not {v.goal for v in vs} & split.used["goal"]
    for v in vs:
        for o in split.used["place"]:
            assert gen.toroidal_gap(v.offset, v.size, o, v.size,
                                    split.period) >= split.margin


def test_margin_spaces_every_pair(wide_field):
    """Margin is spacing, and it applies to all three pair types.

    train<->train, train<->val and val<->val all get the same clearance. The
    generated set must not be able to stack envs on one patch of scaffold just
    because they happen to be on the same side of the split.
    """
    env_cfg = EnvConfig(size=8, observation_size=OBS)
    domains = TraitDomains(dom.Anywhere(), dom.SeedRange(0, 10_000_000),
                           dom.AnyCells(), dom.Sizes((8,)))
    split = gen.generate_split(wide_field, env_cfg, domains, 8, 4,
                               seed=7, margin=12)
    everything = split.train + split.base_val
    for i, a in enumerate(everything):
        for b in everything[i + 1:]:
            assert gen.toroidal_gap(a.offset, a.size, b.offset, b.size,
                                    split.period) >= split.margin

    # ...and a set minted later is spaced internally too, not just from train.
    vs = gen.make_val_set(split, 4, {"place": "held_out", "wall": "held_out",
                                     "goal": "held_out"}, seed=3)
    for i, a in enumerate(vs):
        for b in vs[i + 1:]:
            assert gen.toroidal_gap(a.offset, a.size, b.offset, b.size,
                                    split.period) >= split.margin
