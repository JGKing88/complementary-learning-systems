"""Per-trait env refresh: what moves, what does not, and what gets recorded.

The property everything else rests on is that **a refresh tick folds its values
into ``split.used``**. That union is what ``make_val_set`` excludes against, so a
tick that moved an env without recording it would let a later ``held_out``
validation env be placed exactly where training was -- with nothing raising and
every other test still green. `test_the_union_is_what_a_later_val_set_avoids` is
the one that would catch it.

The rest pin the reasons per-trait refresh is worth having at all: that the
traits really are independent (a place tick must not move goals), that the
train/val separation survives every draw and not just the first, and that a goal
refresh stays inside the partition rather than eating the arena.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import EnvConfig, TrainConfig, VectorHashConfig
from hopfield_nav.training.refresh import Cadence, Refresher
from hopfield_nav.world import domains as dom
from hopfield_nav.world import generate as gen
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import EnvSpec, GeneratedSplit, TraitDomains
from hopfield_nav.world.world import build_world

SIZE = 8
OBS = 12


@pytest.fixture(scope="module")
def field():
    """Npos=132 -- wide enough that a margin can actually bind on placement."""
    vh = VectorHash(VectorHashConfig(lambdas=[11, 12], static_vectorhash=True))
    vh.build_scaffold()
    torch.manual_seed(0)
    enc = torch.nn.Linear(vh.Ng, 16)
    enc.eval()
    vh.precompute_encoded_phi(enc, 0.25, device="cpu")
    return vh


@pytest.fixture
def env_cfg():
    return EnvConfig(size=SIZE, observation_size=OBS)


def _domains(place=None, goal=None, sizes=(SIZE,)):
    return TraitDomains(place=place or dom.Anywhere(),
                        wall=dom.SeedRange(0, 10_000_000),
                        goal=goal or dom.AnyCells(), size=dom.Sizes(sizes))


def _setup(field, env_cfg, cadence, *, n_train=8, n_val=3, margin=12,
           n_worlds=1, seed=0, domains=None, refresh_goal=None):
    """A split, the worlds built from it, and a Refresher over both."""
    domains = domains or _domains()
    split = gen.generate_split(
        field, env_cfg, domains, n_train, n_val, seed=seed, margin=margin,
        refresh_goal=(cadence.goal is not None if refresh_goal is None
                      else refresh_goal),
        diagnostics=False)
    envs = gen.build_envs(split.train, env_cfg, "discrete")
    per = n_train // n_worlds
    worlds = [build_world(field, envs[w * per:(w + 1) * per],
                          offsets=[s.offset
                                   for s in split.train[w * per:(w + 1) * per]])
              for w in range(n_worlds)]
    return split, worlds, Refresher(cadence, split, worlds, env_cfg,
                                    "discrete", seed)


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------

def test_cadence_fires_on_multiples_of_its_period():
    cad = Cadence(place=5, goal=1)
    assert cad.due(1) == ("goal",)
    assert cad.due(4) == ("goal",)
    assert cad.due(5) == ("place", "goal")
    assert Cadence().due(7) == ()
    assert not Cadence()
    assert bool(cad)


def test_refresh_without_the_generator_is_a_startup_error():
    """The legacy path declares no domains, so there is nothing to re-draw from.

    A silent no-op here would be the worst outcome: the run trains on fixed envs
    while its config, its wandb page and its world.json all say it refreshed.
    """
    cfg = TrainConfig()
    cfg.refresh_place = 10
    with pytest.raises(ValueError, match="needs --env_generator"):
        Cadence.from_config(cfg)
    cfg.env_generator = True
    assert Cadence.from_config(cfg).place == 10


def test_a_zero_cadence_is_rejected():
    cfg = TrainConfig()
    cfg.env_generator = True
    cfg.refresh_wall = 0
    with pytest.raises(ValueError, match="must be >= 1"):
        Cadence.from_config(cfg)


def test_no_cadence_means_no_refresher():
    """Off by default, and off is the same code path runs have always taken."""
    assert not Cadence.from_config(TrainConfig())


# ---------------------------------------------------------------------------
# The recording invariant
# ---------------------------------------------------------------------------

def test_the_union_grows_with_every_tick(field, env_cfg):
    split, _, refresher = _setup(field, env_cfg, Cadence(place=1, wall=1))
    n_train = len(split.train)
    assert len(split.used["place"]) == n_train
    for tick in range(1, 5):
        assert refresher.maybe_refresh(tick) == ("place", "wall")
    assert len(split.used["wall"]) == 5 * n_train, "wall seeds must all be new"
    assert len(split.used["place"]) > n_train, (
        "placements moved but the union did not grow -- a later held_out val "
        "env could be placed exactly where training was")


def test_the_union_is_what_a_later_val_set_avoids(field, env_cfg):
    """The end-to-end statement: mint a val set *after* refresh, and it clears
    everything training ever touched -- not just the startup draw.

    Checked against the **live worlds**, not against ``split.used``. Comparing a
    val set to the union it was built from is circular: a tick that applies a
    refresh without recording it makes the union *smaller*, so the val set
    trivially clears it while sitting exactly where training just was. Reading
    the envs the agent actually rolled out in is the only statement with teeth.
    """
    split, worlds, refresher = _setup(field, env_cfg,
                                      Cadence(place=1, wall=1, goal=1),
                                      n_train=6, n_val=3, margin=12)
    seen_place, seen_wall, seen_goal = set(), set(), set()

    def observe():
        for w in worlds:
            for env, off in zip(w.envs, w.offsets):
                seen_place.add(tuple(off))
                seen_wall.add(int(env.seed))
                seen_goal.add(tuple(env.goal_location))

    observe()
    for tick in range(1, 4):
        refresher.maybe_refresh(tick)
        observe()
    assert len(seen_wall) == 4 * len(split.train)   # every tick was distinct

    vs = gen.make_val_set(split, 4, {"place": "held_out", "wall": "held_out",
                                     "goal": "held_out"}, seed=99)
    for v in vs:
        assert v.wall_seed not in seen_wall
        assert v.goal not in seen_goal
        for off in seen_place:
            assert gen.toroidal_gap(v.offset, v.size, off, v.size,
                                    split.period) >= split.margin


def test_split_train_tracks_the_live_worlds(field, env_cfg):
    """`split.train` is what gets written to world.json, so it has to be true."""
    split, worlds, refresher = _setup(field, env_cfg,
                                      Cadence(place=1, wall=1, goal=1))
    refresher.maybe_refresh(1)
    live = [(env.seed, env.size, off, env.goal_location)
            for w in worlds for env, off in zip(w.envs, w.offsets)]
    assert live == [(s.wall_seed, s.size, s.offset, s.goal) for s in split.train]


def test_a_world_that_does_not_match_the_split_is_refused(field, env_cfg):
    """Refresh slices split.train back into worlds by position.

    If the two ever disagree, env i would be handed env j's traits -- silently,
    since both are perfectly legal envs.
    """
    split, worlds, _ = _setup(field, env_cfg, Cadence(place=1))
    worlds[0].offsets = list(reversed(worlds[0].offsets))
    with pytest.raises(ValueError, match="do not match"):
        Refresher(Cadence(place=1), split, worlds, env_cfg, "discrete", 0)


# ---------------------------------------------------------------------------
# Traits are independent
# ---------------------------------------------------------------------------

def test_a_place_tick_moves_only_placements(field, env_cfg):
    """The point of per-trait cadences, and of the derived RNG streams.

    If refreshing one trait perturbed another, "move the envs around but hold
    the wall patterns fixed" would not be an experiment anyone could run.
    """
    split, _, refresher = _setup(field, env_cfg, Cadence(place=1))
    before = list(split.train)
    refresher.maybe_refresh(1)
    assert [s.offset for s in split.train] != [s.offset for s in before]
    assert [s.wall_seed for s in split.train] == [s.wall_seed for s in before]
    assert [s.goal for s in split.train] == [s.goal for s in before]
    assert [s.size for s in split.train] == [s.size for s in before]


def test_a_goal_tick_moves_only_goals(field, env_cfg):
    split, worlds, refresher = _setup(field, env_cfg, Cadence(goal=1))
    before = list(split.train)
    env_ids = [id(e) for w in worlds for e in w.envs]
    refresher.maybe_refresh(1)
    assert [s.offset for s in split.train] == [s.offset for s in before]
    assert [s.wall_seed for s in split.train] == [s.wall_seed for s in before]
    # The envs themselves are not rebuilt -- a goal is set on the live object,
    # so the wall code and codebook the agent has been seeing are untouched.
    assert [id(e) for w in worlds for e in w.envs] == env_ids


def test_another_traits_cadence_cannot_move_this_ones_values(field, env_cfg):
    """Streams are derived per (trait, tick), not advanced from a shared one."""
    _, _, a = _setup(field, env_cfg, Cadence(place=1), seed=3)
    _, _, b = _setup(field, env_cfg, Cadence(place=1, wall=1), seed=3)
    a.maybe_refresh(1)
    b.maybe_refresh(1)
    assert ([s.offset for s in a.split.train]
            == [s.offset for s in b.split.train])


def test_refresh_is_deterministic_and_tick_dependent(field, env_cfg):
    _, _, a = _setup(field, env_cfg, Cadence(place=1, goal=1), seed=11)
    _, _, b = _setup(field, env_cfg, Cadence(place=1, goal=1), seed=11)
    a.maybe_refresh(3)
    b.maybe_refresh(3)
    assert a.split.train == b.split.train

    _, _, c = _setup(field, env_cfg, Cadence(place=1, goal=1), seed=11)
    c.maybe_refresh(4)
    assert c.split.train != a.split.train


# ---------------------------------------------------------------------------
# Separation survives every draw, not just the first
# ---------------------------------------------------------------------------

def test_refreshed_placements_stay_clear_of_validation(field, env_cfg):
    split, _, refresher = _setup(field, env_cfg, Cadence(place=1), margin=12)
    for tick in range(1, 6):
        refresher.maybe_refresh(tick)
        for t in split.train:
            for v in split.base_val:
                assert gen.toroidal_gap(t.offset, t.size, v.offset, v.size,
                                        split.period) >= split.margin
        # ...and from each other, same as the generated draw.
        for i, a in enumerate(split.train):
            for b in split.train[i + 1:]:
                assert gen.toroidal_gap(a.offset, a.size, b.offset, b.size,
                                        split.period) >= split.margin


def test_refreshed_walls_never_collide_with_validation(field, env_cfg):
    split, _, refresher = _setup(field, env_cfg, Cadence(wall=1))
    val_seeds = {v.wall_seed for v in split.base_val}
    for tick in range(1, 6):
        refresher.maybe_refresh(tick)
        assert not {t.wall_seed for t in split.train} & val_seeds


def test_wall_refresh_never_reissues_a_seed_it_already_used(field, env_cfg):
    """Only visible on a narrow seed range.

    At the default 10M-wide range two ticks never collide by chance, so dropping
    the exclusion changes nothing anyone would notice -- and then on a range
    small enough to matter, `used["wall"]` quietly stops growing and a
    `held_out` val set treats a pattern training has already seen as fresh.
    """
    narrow = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 30),
                          goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split, _, refresher = _setup(field, env_cfg, Cadence(wall=1), n_train=6,
                                 n_val=2, margin=12, domains=narrow, seed=4)
    seen = ({v.wall_seed for v in split.base_val}
            | {t.wall_seed for t in split.train})
    for tick in range(1, 4):
        refresher.maybe_refresh(tick)
        fresh = {t.wall_seed for t in split.train}
        assert not fresh & seen, f"tick {tick} reissued {fresh & seen}"
        seen |= fresh
    assert len(seen) == 8 + 3 * 6

    # And when the range really is used up, it says so rather than looping.
    with pytest.raises(ValueError, match="cannot yield"):
        refresher.maybe_refresh(4)


def test_goal_refresh_stays_inside_the_train_partition(field, env_cfg):
    """Drawing from the domain minus what was used would eat the arena.

    64 cells, 8 envs a tick: the grid is gone in eight updates and no legal
    held-out goal is left. The partition is what makes an unbounded cadence
    possible at all.
    """
    split, _, refresher = _setup(field, env_cfg, Cadence(goal=1), n_train=8)
    assert split.goal_cells_train and split.goal_cells_val
    for tick in range(1, 20):
        refresher.maybe_refresh(tick)
        for t in split.train:
            assert t.goal in split.goal_cells_train
    assert not split.used["goal"] & split.goal_cells_val


def test_goal_refresh_switches_on_the_partition_in_training_setup():
    """`setup_worlds_declared` has to pass refresh_goal, or the cap never runs."""
    import inspect
    from hopfield_nav.training import world_setup
    src = inspect.getsource(world_setup.setup_worlds_declared)
    assert "refresh_goal=" in src, (
        "generate_split's goal-partition branch is unreachable, so goal "
        "refresh would consume every cell in the arena")


# ---------------------------------------------------------------------------
# Rebuilds
# ---------------------------------------------------------------------------

def test_a_wall_tick_rebuilds_the_envs(field, env_cfg):
    """A wall code is fixed at construction from (seed, size), so a new seed
    means a new env object -- not a mutated one."""
    split, worlds, refresher = _setup(field, env_cfg, Cadence(wall=1))
    before = [(id(e), e.seed, e._wall_code.copy(), e._codebook.copy())
              for e in worlds[0].envs]
    refresher.maybe_refresh(1)
    for (old_id, old_seed, old_code, old_book), env, spec in zip(
            before, worlds[0].envs, split.train):
        assert id(env) != old_id and env.seed != old_seed
        assert env.seed == spec.wall_seed
        assert not np.array_equal(env._wall_code, old_code)
        assert np.array_equal(env._wall_code,
                              gen.wall_code_for(spec.wall_seed, spec.size))
        # What the agent sees has to follow the new walls, not just the seed.
        assert not np.array_equal(env._codebook, old_book)


def test_a_rebuilt_env_carries_its_recorded_goal(field, env_cfg):
    split, worlds, refresher = _setup(field, env_cfg, Cadence(wall=1, goal=1))
    refresher.maybe_refresh(1)
    for env, spec in zip(worlds[0].envs, split.train):
        assert env.goal_location == spec.goal


def test_multiple_worlds_are_sliced_in_order(field, env_cfg):
    split, worlds, refresher = _setup(field, env_cfg,
                                      Cadence(place=1, wall=1, goal=1),
                                      n_train=8, n_worlds=2)
    refresher.maybe_refresh(1)
    flat = [(e.seed, off) for w in worlds for e, off in zip(w.envs, w.offsets)]
    assert flat == [(s.wall_seed, s.offset) for s in split.train]


def test_the_exploit_memory_follows_a_refreshed_goal(field, env_cfg):
    """The regime derives its pattern from `encoded_Phi[goal + offset]`, and a
    refresh moves both. A cached memory would go on pointing at the old cell
    while the reward still fired at the real goal -- teaching the agent that
    following recall does not pay."""
    from hopfield_nav.training.exploit import ExploitRegime
    from hopfield_nav.training.stages import Knobs
    split, worlds, refresher = _setup(field, env_cfg, Cadence(place=1, goal=1))
    cfg = TrainConfig()
    cfg.hopfield.beta = 1.0
    regime = ExploitRegime(cfg, field.encoded_Phi.shape[-1],
                           torch.device("cpu"), np.random.RandomState(0))
    knobs = Knobs(lr=1e-4, empty_frac=0.0, novelty=0.0, eps=0.0, dist_min=0,
                  dist_max=0, emp_dist_min=0, emp_dist_max=0)
    world = worlds[0]

    def memory():
        env, off = world.envs[0], world.offsets[0]
        return regime.spec(0, world, 0, env, off, knobs).hop.W.clone()

    before = memory()
    refresher.maybe_refresh(1)
    assert not torch.allclose(memory(), before)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------

def test_size_refresh_needs_more_than_one_declared_size(field, env_cfg):
    with pytest.raises(ValueError, match="more than one declared env size"):
        _setup(field, env_cfg, Cadence(size=1))


def test_size_refresh_drags_place_and_goal_with_it(field, env_cfg):
    """A new footprint invalidates the packing and the arena at once.

    Offsets were spaced for the old size, and a goal cell can fall outside the
    new arena outright -- so a run asking only for --refresh_size must not be
    able to produce an env whose goal is out of bounds.
    """
    domains = _domains(sizes=(4, 8))
    # generate_split refuses mixed sizes, so the starting split is built by
    # hand at one of them; refresh is what introduces the second.
    base = gen.generate_split(field, env_cfg, _domains(sizes=(8,)), 6, 3,
                              seed=1, margin=12, diagnostics=False)
    split = GeneratedSplit(
        domains=domains, train=base.train, base_val=base.base_val,
        goal_cells_train=base.goal_cells_train,
        goal_cells_val=base.goal_cells_val, margin=base.margin,
        period=base.period, Npos=base.Npos)
    split.record_used(base.train)
    envs = gen.build_envs(split.train, env_cfg, "discrete")
    worlds = [build_world(field, envs, offsets=[s.offset for s in split.train])]
    refresher = Refresher(Cadence(size=1), split, worlds, env_cfg,
                          "discrete", 5)

    seen = set()
    for tick in range(1, 12):
        traits = refresher.maybe_refresh(tick)
        assert traits == ("place", "goal", "size"), (
            "a size tick has to carry place and goal, or an env can end up "
            "with a goal outside its own arena")
        size = split.train[0].size
        seen.add(size)
        for spec, env in zip(split.train, worlds[0].envs):
            assert env.size == spec.size == size
            assert spec.goal[0] < size and spec.goal[1] < size
            assert env.goal_location == spec.goal
        for t in split.train:
            for v in split.base_val:
                assert gen.toroidal_gap(t.offset, t.size, v.offset, v.size,
                                        split.period) >= split.margin
    assert seen == {4, 8}, f"both declared sizes should come up, saw {seen}"


def test_a_val_set_after_a_size_refresh_must_be_told_which_size(field, env_cfg):
    """`used["size"]` is a set; picking from it arbitrarily would make the val
    env size depend on iteration order rather than on anything asked for."""
    split = gen.generate_split(field, env_cfg, _domains(), 4, 2, seed=0,
                               margin=12, diagnostics=False)
    split.record_used([EnvSpec(1, 4, (0, 0), (0, 0))])
    with pytest.raises(ValueError, match="pass make_val_set"):
        gen.make_val_set(split, 2, {"place": "held_out"}, seed=1)


# ---------------------------------------------------------------------------
# Dense packing -- the case unit tests with room to spare kept missing
# ---------------------------------------------------------------------------

def test_refresh_survives_a_packing_with_no_room_to_spare(field, env_cfg):
    """15 envs at margin 20 on Npos=132: the lattice fallback, every tick.

    Rejection sampling cannot find the free slots when the region is nearly
    full, and a refresh has to solve that problem repeatedly rather than once.
    Phase 3's placement bugs all hid behind tests that had room to spare.
    """
    split, _, refresher = _setup(field, env_cfg, Cadence(place=1),
                                 n_train=12, n_val=3, margin=20, seed=42)
    for tick in range(1, 6):
        assert refresher.maybe_refresh(tick) == ("place",)
        assert len(split.train) == 12
        everything = split.train + split.base_val
        for i, a in enumerate(everything):
            for b in everything[i + 1:]:
                assert gen.toroidal_gap(a.offset, a.size, b.offset, b.size,
                                        split.period) >= split.margin


def test_a_region_too_small_after_a_size_change_names_the_cause(field, env_cfg):
    """The preflight has to say that the *size* refresh broke the bound, since
    the same numbers passed at startup."""
    domains = _domains(place=dom.Rect(0, 0, 60, 60), sizes=(8, 40))
    base = gen.generate_split(field, env_cfg, _domains(place=dom.Rect(0, 0, 60, 60)),
                              4, 2, seed=0, margin=6, diagnostics=False)
    split = GeneratedSplit(
        domains=domains, train=base.train, base_val=base.base_val,
        goal_cells_train=base.goal_cells_train,
        goal_cells_val=base.goal_cells_val, margin=base.margin,
        period=base.period, Npos=base.Npos)
    split.record_used(base.train)
    envs = gen.build_envs(split.train, env_cfg, "discrete")
    worlds = [build_world(field, envs, offsets=[s.offset for s in split.train])]
    refresher = Refresher(Cadence(size=1), split, worlds, env_cfg, "discrete", 0)
    with pytest.raises(ValueError, match="size refresh is what broke it"):
        for tick in range(1, 30):
            refresher.maybe_refresh(tick)


# ---------------------------------------------------------------------------
# The report that reaches world.json
# ---------------------------------------------------------------------------

def test_the_report_says_whether_the_world_stood_still(field, env_cfg):
    split, _, refresher = _setup(field, env_cfg, Cadence(place=2, goal=1))
    for tick in range(1, 5):
        refresher.maybe_refresh(tick)
    rep = refresher.report()
    assert rep["cadence"] == {"place": 2, "wall": None, "goal": 1, "size": None}
    assert rep["ticks"] == 4
    assert rep["counts"] == {"place": 2, "wall": 0, "goal": 4, "size": 0}
    assert rep["n_used"]["place"] >= len(split.train)


# ---------------------------------------------------------------------------
# The startup preflight
# ---------------------------------------------------------------------------

def test_draw_only_mode_records_without_building_anything(field, env_cfg):
    """`worlds=None` draws and records; it must not touch envs or offsets.

    This is what lets the preflight replay 300 ticks in a second -- the env
    rebuild is the whole cost of a wall tick -- and, more to the point, what lets
    it run `_draw` itself rather than a reimplementation that could drift.
    """
    split, worlds, _ = _setup(field, env_cfg, Cadence(place=1, wall=1))
    before = [(e.seed, off) for w in worlds for e, off in zip(w.envs, w.offsets)]
    dry = Refresher(Cadence(place=1, wall=1), split, None, env_cfg, "discrete", 0)
    for tick in range(1, 4):
        dry.maybe_refresh(tick)
    assert len(split.used["place"]) > len(before)      # recorded
    assert [(e.seed, off) for w in worlds
            for e, off in zip(w.envs, w.offsets)] == before   # not applied


def test_the_preflight_matches_what_the_run_actually_leaves(field, env_cfg):
    """The prediction and the run must agree, because they share `_draw`.

    A preflight that reimplemented the draw would be a second source of truth
    for the thing it exists to check -- and would agree right up until someone
    changed one of them.
    """
    from hopfield_nav.training.refresh import preflight
    split, worlds, refresher = _setup(field, env_cfg, Cadence(place=1, wall=1),
                                      n_train=6, n_val=3, seed=5)
    rep = preflight(split, Cadence(place=1, wall=1), 5, env_cfg, "discrete", 5,
                    n_val_envs=3)
    for tick in range(1, 6):
        refresher.maybe_refresh(tick)
    assert rep["used_at_end"]["place"] == len(split.used["place"])
    assert rep["used_at_end"]["wall"] == len(split.used["wall"])
    assert rep["ticks"] == 5


def test_the_preflight_leaves_the_real_split_alone(field, env_cfg):
    """It simulates on a copy; a run must not train on the preflight's draws."""
    from hopfield_nav.training.refresh import preflight
    split, _, _ = _setup(field, env_cfg, Cadence(place=1), n_train=6, n_val=3)
    before_train = list(split.train)
    before_used = {k: set(v) for k, v in split.used.items()}
    preflight(split, Cadence(place=1), 10, env_cfg, "discrete", 0, n_val_envs=3)
    assert split.train == before_train
    assert {k: set(v) for k, v in split.used.items()} == before_used


def _run_and_preflight(field, env_cfg, seed, n, ticks=6):
    from hopfield_nav.training.refresh import preflight
    split, _, refresher = _setup(field, env_cfg, Cadence(place=1), n_train=8,
                                 n_val=3, margin=12, seed=seed)
    rep = preflight(split, Cadence(place=1), ticks, env_cfg, "discrete", seed,
                    n_val_envs=n)
    for tick in range(1, ticks + 1):
        refresher.maybe_refresh(tick)
    return split, rep


def _val_seed_outcomes(split, n):
    out = []
    for val_seed in (1, 7, 42, 1234):
        try:
            out.append(len(gen.make_val_set(
                split, n, {"place": "held_out", "wall": "held_out",
                           "goal": "held_out"}, seed=val_seed)))
        except RuntimeError:
            out.append(-1)
    return out


@pytest.mark.parametrize("seed, n", [(s, n) for s in range(6) for n in (3, 4)])
def test_an_ok_verdict_holds_for_whichever_val_seed_is_used_later(field, env_cfg,
                                                                  seed, n):
    """`ok` must survive the eval's own draw order, not just one lucky order.

    Greedy packing depends on the order it sees candidates in, and
    `make_val_set` shuffles with the *val* seed -- which the preflight cannot
    know. A verdict taken from a single order over-promises: at seed 1 and seed 4
    with n=4, one order reports room and `make_val_set` then fails for two of
    four val seeds. Taking the worst of several orders is what closes that, and
    these parameters are the ones where the two disagree.
    """
    split, rep = _run_and_preflight(field, env_cfg, seed, n)
    if not rep["ok"]:
        return                       # covered by the conservative-direction test
    assert _val_seed_outcomes(split, n) == [n] * 4, (
        f"preflight said {n} held-out place envs were available at seed {seed}, "
        f"and make_val_set could not deliver for every val seed")


@pytest.mark.parametrize("seed, n", [(1, 4), (4, 4)])
def test_a_single_draw_order_would_have_said_yes(field, env_cfg, seed, n):
    """The two configurations that make the multi-order check load-bearing.

    Pinned explicitly so that dropping back to one order, or taking the best
    instead of the worst, fails here rather than passing everywhere by slack.
    """
    import hopfield_nav.training.refresh as R
    split, rep = _run_and_preflight(field, env_cfg, seed, n)
    assert rep["ok"] is False, "the honest verdict at this boundary is 'no'"
    assert -1 in _val_seed_outcomes(split, n), "expected a val seed to fail here"

    orig = R._PREFLIGHT_ORDERS
    try:
        R._PREFLIGHT_ORDERS = 1
        _, optimistic = _run_and_preflight(field, env_cfg, seed, n)
    finally:
        R._PREFLIGHT_ORDERS = orig
    assert optimistic["ok"] is True, (
        "this case no longer distinguishes one order from several; find another")


def test_the_verdict_errs_toward_saying_no(field, env_cfg):
    """Under-promising is the safe direction, and it does happen.

    At seed 5, n=3 the worst of five orders finds no room while every val seed
    tried actually succeeds. That is the intended failure direction: a spurious
    warning costs a line of output, a spurious all-clear costs a re-run.
    """
    split, rep = _run_and_preflight(field, env_cfg, 5, 3)
    assert rep["ok"] is False
    assert _val_seed_outcomes(split, 3) == [3] * 4


def test_a_shrinking_ceiling_is_reported_not_enforced(field, env_cfg):
    """Records and proceeds. A run using only --split recorded is fine with a
    tight union, and that is not the trainer's call to veto."""
    from hopfield_nav.training.refresh import format_preflight, preflight
    split, _, _ = _setup(field, env_cfg, Cadence(place=1), n_train=10, n_val=3,
                         margin=12, seed=1)
    tight = preflight(split, Cadence(place=1), 40, env_cfg, "discrete", 1,
                      n_val_envs=99)               # more than any run can hold
    assert tight["ok"] is False
    msg = format_preflight(tight)
    assert "WARNING" in msg and "the run continues" in msg
    assert "base_val are unaffected" in msg

    roomy = preflight(split, Cadence(place=1), 2, env_cfg, "discrete", 1,
                      n_val_envs=1)
    assert roomy["ok"] is True
    assert "WARNING" not in format_preflight(roomy)


def test_a_narrow_seed_range_shows_up_as_wall_headroom(field, env_cfg):
    """Place is not the only trait a long refresh can exhaust.

    40 seeds, 6 per tick: after 5 ticks 36 are spoken for and 4 are left, which
    is not enough for a 5-env held-out wall set.
    """
    from hopfield_nav.training.refresh import format_preflight, preflight
    narrow = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 40),
                          goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split, _, _ = _setup(field, env_cfg, Cadence(wall=1), n_train=6, n_val=2,
                         margin=12, domains=narrow, seed=4)
    rep = preflight(split, Cadence(wall=1), 5, env_cfg, "discrete", 4,
                    n_val_envs=5)
    assert rep["refresh_dies_at_update"] is None
    assert rep["wall_seeds_left"] == 4 and rep["ok"] is False
    assert "wall seeds unused" in format_preflight(rep)


def test_a_domain_that_runs_dry_mid_run_is_named_before_the_run_starts(
        field, env_cfg):
    """The failure the preflight is most worth having.

    A trait whose domain empties does not degrade -- it raises, hours in, at a
    tick fixed before the run began. Catching the exception during the replay
    turns four wasted hours into two seconds, and the tick it names is exact
    because nothing about the draw depends on training.
    """
    from hopfield_nav.training.refresh import format_preflight, preflight
    narrow = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 40),
                          goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split, _, refresher = _setup(field, env_cfg, Cadence(wall=1), n_train=6,
                                 n_val=2, margin=12, domains=narrow, seed=4)
    rep = preflight(split, Cadence(wall=1), 50, env_cfg, "discrete", 4,
                    n_val_envs=2)
    assert rep["refresh_dies_at_update"] == 6
    assert "cannot yield" in rep["refresh_dies_of"]
    assert rep["ok"] is False
    msg = format_preflight(rep)
    assert "will not finish" in msg and "update 6 of 50" in msg

    # ...and the real run dies exactly there, which is what makes it predictable.
    for tick in range(1, 6):
        refresher.maybe_refresh(tick)
    with pytest.raises(ValueError):
        refresher.maybe_refresh(6)
