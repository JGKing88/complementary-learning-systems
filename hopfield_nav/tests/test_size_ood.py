"""Evaluating a checkpoint on arenas of a size it never trained on.

Every property here is one that fails *silently* without it. The architecture
takes a differing size without complaint -- the observation is a ray count, not
a grid width -- so nothing raises; the numbers just come back computed against
the wrong arena. Measured before the fix: an env.size=12 val set scored under a
size-6 config reported mean_coverage against a denominator of 36 instead of 144,
four times the truth, with no error anywhere.

The gate for the whole phase is `test_val_size_equal_to_the_trained_size_is_a_noop`:
it says the size plumbing changed nothing for every run that does not use it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import (EnvConfig, HopfieldConfig, TrainConfig,
                                 VectorHashConfig)
from hopfield_nav.evaluation import checkpoint_io as cio
from hopfield_nav.policy import channels
from hopfield_nav.rollout.distractors import sample_distractors
from hopfield_nav.training.world_setup import build_field
from hopfield_nav.world import domains as dom
from hopfield_nav.world import generate as gen
from hopfield_nav.world.spec import TraitDomains, WorldSpec

LAMBDAS = [7, 11]       # Npos = 77 -- room for arenas well above SIZE
SIZE = 4                # what "training" ran at
OBS = 16
LEVELS = {"place": "held_out", "wall": "held_out", "goal": "held_out"}


def _cfg() -> TrainConfig:
    return TrainConfig(
        env=EnvConfig(size=SIZE, observation_size=OBS),
        vectorhash=VectorHashConfig(lambdas=LAMBDAS, Np=40,
                                    static_vectorhash=True),
        # Set outright: production fills it from the encoder gain at startup,
        # and recall multiplies by it.
        hopfield=HopfieldConfig(beta=1.0),
        envs_per_world=3, num_worlds=1, num_val_envs=2, device="cpu",
    )


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    e = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    e.eval()
    return e


@pytest.fixture(scope="module")
def field(enc):
    return build_field(_cfg(), enc)


@pytest.fixture
def spec(field):
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, EnvConfig(size=SIZE, observation_size=OBS),
                               domains, 2, 2, seed=0, margin=2)
    return WorldSpec(scaffold={"Npos": field.Npos}, generator="declared",
                     split=split)


# ---------------------------------------------------------------------------
# The gate: passing the size you already have changes nothing
# ---------------------------------------------------------------------------

def test_val_size_equal_to_the_trained_size_is_a_noop(spec, enc, field):
    """The gate for the phase. Same key, same envs, same report.

    If this ever drifts, every run that does not use `--val_size` has silently
    changed too, and the size work has bled into results it was never meant to
    touch.
    """
    cfg = _cfg()
    plain = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                             val_seed=3, spec=spec, field=field)
    sized = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                             val_seed=3, spec=spec, field=field, size=SIZE)
    assert sized["key"] == plain["key"] == gen.levels_key(LEVELS)
    assert sized["offsets"] == plain["offsets"]
    assert sized["report"] == plain["report"]
    for a, b in zip(sized["envs"], plain["envs"]):
        assert (a.size, a.goal_location, a.seed) == (b.size, b.goal_location,
                                                     b.seed)


def test_a_differing_size_says_so_in_the_key(spec, enc, field):
    """So a size sweep is a column in the same table `--split` produces, and
    two sizes cannot collide into one entry of `results["splits"]`."""
    got = cio.eval_env_set(_cfg(), enc, "cpu", ckpt_path=None, levels=LEVELS,
                           val_seed=3, spec=spec, field=field, size=SIZE + 4)
    assert got["key"] == f"{gen.levels_key(LEVELS)},size={SIZE + 4}"
    assert {e.size for e in got["envs"]} == {SIZE + 4}


def test_val_size_on_the_recorded_set_is_refused(spec, enc, field):
    """`recorded` is a fixed list of envs; there is no size-N version of it.
    Silently ignoring the flag is the failure mode this phase exists to end."""
    with pytest.raises(SystemExit, match="needs a minted split"):
        cio.eval_env_set(_cfg(), enc, "cpu", ckpt_path=None, levels=None,
                         val_seed=3, spec=spec, field=field, size=SIZE + 4)


# ---------------------------------------------------------------------------
# 6.2 -- the excluded envs are the size they were placed at
# ---------------------------------------------------------------------------

def test_legal_offsets_matches_the_predicate_across_sizes():
    """`legal_mask` against ground truth, when candidate and excluded env are
    different sizes -- the case `_forbidden_span`'s asymmetry exists for."""
    period = Npos = 132
    margin, train_size, val_size = 10, 6, 20
    boxes = [((10, 10), train_size), ((60, 90), train_size),
             ((120, 30), train_size)]
    domain = dom.Anywhere()

    truth = {(x, y) for x in range(period) for y in range(period)
             if domain.contains((x, y), val_size, Npos)
             and all(gen.toroidal_gap((x, y), val_size, o, s, period) >= margin
                     for o, s in boxes)}
    got = set(gen.legal_offsets(boxes, domain, size=val_size, Npos=Npos,
                                period=period, margin=margin))
    assert got == truth

    # Labelling the training boxes with the *validation* size -- what
    # `make_val_set` did before -- throws away legal ground. It is a subset, so
    # nothing unsafe was ever placed; the cost is room, and a val set that
    # cannot be drawn at all.
    mislabelled = set(gen.legal_offsets([(o, val_size) for o, _ in boxes],
                                        domain, size=val_size, Npos=Npos,
                                        period=period, margin=margin))
    assert mislabelled < truth
    assert len(truth) - len(mislabelled) > 2000, (
        "the over-exclusion this test guards against is not being exercised")


def test_make_val_set_excludes_the_train_envs_at_their_own_size(field):
    """The call site, not just the predicate.

    A `--val_size` of 30 against four size-4 training envs at margin 3: the
    boxes as recorded leave 421 legal offsets and place both val envs, and the
    same boxes relabelled with the validation size leave **zero**. So this is a
    val set that exists or does not depending only on the label -- the sharpest
    form of the §6.2 bug, and what makes the fix load-bearing rather than tidy.
    """
    env_cfg = EnvConfig(size=SIZE, observation_size=OBS)
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, env_cfg, domains, 4, 1, seed=1, margin=3)

    vs = gen.make_val_set(split, 2, LEVELS, seed=0, size=30)
    assert len(vs) == 2 and {v.size for v in vs} == {30}
    for v in vs:                       # and they really are clear of training
        for o, s in split.used_boxes():
            assert gen.toroidal_gap(v.offset, v.size, o, s,
                                    split.period) >= split.margin

    mislabelled = [(o, 30) for o, _ in split.used_boxes()]
    assert gen.legal_offsets(mislabelled, dom.Anywhere(), size=30,
                             Npos=split.Npos, period=split.period,
                             margin=split.margin) == []


def test_used_place_keeps_the_size_it_was_placed_at(spec):
    """The representation, not just the call site. `used["place"]` is boxes."""
    split = spec.split
    assert split.used_boxes() == sorted(
        (s.offset, s.size) for s in split.train)
    assert split.used_offsets() == {s.offset for s in split.train}
    # And it survives a round trip through world.json.
    back = type(split).from_json(split.to_json())
    assert back.used_boxes() == split.used_boxes()


def test_a_pre_phase6_world_json_pairs_offsets_with_the_recorded_size(spec):
    """Files written before this carry [r, c] with no size. They still load,
    paired with the size the run recorded -- not left unlabelled."""
    d = spec.split.to_json()
    d["used"]["place"] = [[r, c] for r, c, _ in d["used"]["place"]]
    back = type(spec.split).from_json(d)
    assert back.used_boxes() == spec.split.used_boxes()


def test_a_world_json_written_before_phase_6_still_loads(spec):
    """Including its hash.

    `from_json` used to verify by re-rendering the object, which makes the hash
    a claim about the current serializer rather than about the file. Changing
    one field's shape would then reject every world.json ever written -- with a
    message blaming the reader for editing it. Hashing the payload as read fixes
    that for this format change and every later one.
    """
    from hopfield_nav.world.spec import _hash_payload

    d = WorldSpec(scaffold={"Npos": 77}, generator="declared",
                  split=spec.split).to_json()
    d["spec_version"] = 1
    d["split"]["used"]["place"] = [[r, c] for r, c, _
                                   in d["split"]["used"]["place"]]
    d["spec_hash"] = _hash_payload({k: v for k, v in d.items()
                                    if k != "spec_hash"})

    back = WorldSpec.from_json(d)
    assert back.spec_version == 1
    assert back.split.used_boxes() == spec.split.used_boxes()


def test_a_hand_edited_world_json_is_still_caught(spec):
    """The guard the hash exists for, kept alive by the same change."""
    d = WorldSpec(scaffold={"Npos": 77}, generator="declared",
                  split=spec.split).to_json()
    d["split"]["train"][0]["goal"] = [99, 99]
    with pytest.raises(ValueError, match="hash mismatch"):
        WorldSpec.from_json(d)


# ---------------------------------------------------------------------------
# 6.1 -- what the metrics are computed against
# ---------------------------------------------------------------------------

def test_coverage_denominator_is_the_val_sets_not_the_configs(spec, enc, field):
    """The 4x error, driven rather than restated.

    This is the number the whole phase exists to not get wrong, so it goes
    through the real evaluator: coverage must be `cells / val_size**2`. Under
    the pre-fix expression -- `cfg.env.size**2` -- the very same rollouts report
    `(big/SIZE)**2` times as much, which the second assertion rules out.
    """
    from hopfield_nav.evaluation.metrics import evaluate_exploration

    cfg = _cfg()
    big = SIZE + 8      # 12, comfortably placeable at margin 2
    es = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                          val_seed=1, spec=spec, field=field, size=big)
    assert {int(e.size) for e in es["envs"]} == {big} != {cfg.env.size}

    agent = cio.load_agent(cfg, None, field.encoded_Phi.shape[2],
                           torch.device("cpu"))
    per_trial: list = []
    got = evaluate_exploration(
        agent, es["envs"], field, es["offsets"], cfg, torch.device("cpu"),
        num_trials=1, max_steps=6, n_distractors_list=[0], per_trial=per_trial)
    cells = float(np.mean([r[3] for r in per_trial]))
    assert cells > 0, "no cell was ever visited; the test is vacuous"
    assert got[0]["mean_coverage"] == pytest.approx(cells / (big * big))
    assert got[0]["mean_coverage"] != pytest.approx(cells / (SIZE * SIZE))
    # The union is a fraction of the same arena. Against the config's size it
    # would exceed 1.0 here, which is what a fraction cannot do.
    assert 0.0 < got[0]["union_coverage"] <= 1.0
    assert got[0]["union_coverage"] == pytest.approx(
        got[0]["union_per_rollout"] * 1)      # num_trials=1

    # And what the run records about itself says the val size, not the config's.
    layout = cio.scaffold_layout_dict(cfg, field, es["envs"], es["offsets"])
    assert layout["env_size"] == big
    assert {e["size"] for e in layout["envs"]} == {big}


@pytest.mark.parametrize("evaluator", ["exploration", "navigation", "discovery"])
def test_distractors_are_excluded_at_the_arenas_own_size(spec, enc, field,
                                                         monkeypatch, evaluator):
    """A distractor drawn from inside the env's own footprint is a neighbouring
    cell of the same arena, not a false memory -- the distinction the whole
    Hopfield eval rests on. `sample_distractors` guarantees it for the size it
    is *given*, so what has to be pinned is the size each evaluator gives it.
    """
    from hopfield_nav.evaluation import metrics

    cfg = _cfg()
    big = SIZE + 8
    es = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                          val_seed=2, spec=spec, field=field, size=big)
    seen: list[int] = []
    real = metrics.sample_distractors

    def spy(vh, offset, env_size, n, rng):
        seen.append(int(env_size))
        return real(vh, offset, env_size, n, rng)

    monkeypatch.setattr(metrics, "sample_distractors", spy)
    agent = cio.load_agent(cfg, None, field.encoded_Phi.shape[2],
                           torch.device("cpu"))
    fn = {"exploration": metrics.evaluate_exploration,
          "navigation": metrics.evaluate_navigation,
          "discovery": metrics.evaluate_goal_discovery}[evaluator]
    fn(agent, es["envs"], field, es["offsets"], cfg, torch.device("cpu"),
       num_trials=1, max_steps=4, n_distractors_list=[2])

    assert seen, f"{evaluator} never drew a distractor; the test is vacuous"
    assert set(seen) == {big}, (
        f"{evaluator} excluded a {sorted(set(seen))}-cell box around a "
        f"{big}-cell arena, so its own cells were eligible as distractors")


def test_the_observation_width_does_not_depend_on_the_arena(spec, enc, field):
    """The architecture claim, asserted rather than assumed.

    `input_dim` sums over embed_dim and the ray count; the ray count is
    `observation_size`, not the grid width. If this ever stops holding, size OOD
    stops being an eval-only change and needs a re-architected policy.
    """
    cfg = _cfg()
    small = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                             val_seed=4, spec=spec, field=field)
    big = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                           val_seed=4, spec=spec, field=field, size=SIZE + 8)
    assert {e.size for e in small["envs"]} != {e.size for e in big["envs"]}
    widths = set()
    for es in (small, big):
        env = es["envs"][0]
        env.set_position((0, 0))
        sensory = int(np.asarray(env.obs()).ravel().size)
        widths.add((sensory, channels.input_dim(cfg.agent, 8, sensory)))
    assert len(widths) == 1, (
        f"the policy input width depends on the arena size: {widths}")
    assert next(iter(widths))[0] == cfg.env.observation_size


# ---------------------------------------------------------------------------
# 6.3 -- wall codes of two sizes are not in the same space
# ---------------------------------------------------------------------------

def test_wall_codes_of_different_sizes_do_not_compare():
    """The premise for reporting `None`: there is no distance to report."""
    a = gen.wall_code_for(1234, 6)
    b = gen.wall_code_for(1234, 20)
    assert a.shape != b.shape
    with pytest.raises(ValueError, match="broadcast"):
        _ = a != b
    # Nor is the bigger code an extension of the smaller: same seed, unrelated
    # draw. So there is no honest way to line them up either.
    assert not np.array_equal(a, b[:, :6])


def test_the_report_says_why_the_hamming_margin_is_missing(spec, enc, field):
    """Silence would read as "not measured"; a number would be a fiction --
    computed against a wall the training env never had."""
    cfg = _cfg()
    same = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                            val_seed=5, spec=spec, field=field)
    assert same["report"]["min_wall_hamming_vs_train"] is not None
    assert same["report"]["wall_hamming_note"] is None

    big = cio.eval_env_set(cfg, enc, "cpu", ckpt_path=None, levels=LEVELS,
                           val_seed=5, spec=spec, field=field, size=SIZE + 8)
    assert big["report"]["min_wall_hamming_vs_train"] is None
    note = big["report"]["wall_hamming_note"]
    assert note and str(SIZE + 8) in note and "seed disjointness" in note
    # Novelty on this axis is still asserted, by the means that does survive.
    assert big["report"]["n_wall_seeds_shared"] == 0


# ---------------------------------------------------------------------------
# 6.4 -- a size the scaffold has no room for
# ---------------------------------------------------------------------------

def test_no_room_at_the_new_size_names_the_size_change(field):
    """The exclusion-set explanation is true and sends the reader to the wrong
    knob: the same numbers passed at the training size."""
    env_cfg = EnvConfig(size=SIZE, observation_size=OBS)
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, env_cfg, domains, 3, 1, seed=1, margin=3)
    # Size 40 on a 77-wide scaffold: the same draw succeeds at the training
    # size, which is what makes the clause true rather than merely plausible.
    with pytest.raises(RuntimeError, match="size change is the cause"):
        gen.make_val_set(split, 2, LEVELS, seed=0, size=40)


def test_it_does_not_blame_the_size_when_the_size_is_not_to_blame(field):
    """The clause is measured, not asserted: it re-resolves the draw at the
    excluded envs' own size and only fires if that would have worked."""
    env_cfg = EnvConfig(size=SIZE, observation_size=OBS)
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, env_cfg, domains, 3, 1, seed=1, margin=3)
    # Far more envs than the scaffold packs at *either* size, so the size
    # change is a real difference that is nonetheless not what went wrong.
    with pytest.raises(RuntimeError) as exc:
        gen.make_val_set(split, 400, LEVELS, seed=0, size=SIZE + 4)
    assert "size change is the cause" not in str(exc.value)
    assert "size change is not the cause" in str(exc.value)


# ---------------------------------------------------------------------------
# 6.5 -- held-out goals reach the new geometry
# ---------------------------------------------------------------------------

def test_held_out_goals_reach_outside_the_training_footprint(field):
    """Every cell in `goal_cells_val` was drawn at the training size, so
    filtering to them would confine every goal to the original arena and make
    "size OOD" quietly test something narrower than it claims."""
    env_cfg = EnvConfig(size=SIZE, observation_size=OBS)
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, env_cfg, domains, 2, 1, seed=2, margin=2)
    big = SIZE + 8      # 12, comfortably placeable at margin 2
    outside = 0
    for seed in range(24):
        for v in gen.make_val_set(split, 2, LEVELS, seed=seed, size=big):
            assert v.goal not in split.used["goal"]        # still held out
            assert max(v.goal) < big                       # still in the arena
            outside += int(v.goal[0] >= SIZE or v.goal[1] >= SIZE)
    assert outside > 0, (
        "no held-out goal ever landed outside the training footprint, so the "
        "outer band of the bigger arena is never tested")


def test_a_smaller_val_size_keeps_only_cells_that_fit(field):
    """The other direction: there is no new region, and the filter is the whole
    of the rule."""
    env_cfg = EnvConfig(size=SIZE + 6, observation_size=OBS)
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE + 6,)))
    split = gen.generate_split(field, env_cfg, domains, 2, 1, seed=3, margin=2)
    for v in gen.make_val_set(split, 3, LEVELS, seed=0, size=SIZE):
        assert max(v.goal) < SIZE
        assert v.goal in split.goal_cells_val


# ---------------------------------------------------------------------------
# 6.6 -- coverage at a fixed budget confounds capability with arena size
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6.7 -- the sweep is one table, not one run per size
# ---------------------------------------------------------------------------

def test_sizes_cross_with_splits_and_recorded_keeps_its_own():
    """`--val_size` is repeatable and crossed with `--split`, or a size sweep
    would be several runs whose numbers no table puts side by side.

    `recorded` is paired with `None` alone: it is a fixed list of envs, so
    there is no size of it to vary, and crossing it would make an invocation
    that plainly makes sense -- "the run's own set, plus held_out at 12" --
    raise.
    """
    from hopfield_nav.eval_all import split_size_combos

    combos = split_size_combos([None, LEVELS], [4, 12])
    assert combos == [(None, None), (LEVELS, 4), (LEVELS, 12)]

    # No sizes asked for: unchanged from what `--split` alone always did.
    assert split_size_combos([None, LEVELS], None) == [(None, None),
                                                       (LEVELS, None)]


def test_a_repeated_combination_is_evaluated_once():
    """Two identical combinations would each cost a full evaluation and then
    overwrite one another in `results["splits"]` -- paying twice for one row."""
    from hopfield_nav.eval_all import split_size_combos

    assert split_size_combos([LEVELS, dict(LEVELS)], [8, 8]) == [(LEVELS, 8)]


def test_val_size_with_only_recorded_splits_is_refused():
    """Otherwise the flag would be accepted and do nothing at all."""
    from hopfield_nav.eval_all import split_size_combos

    with pytest.raises(SystemExit, match="every --split is 'recorded'"):
        split_size_combos([None], [12])


def test_the_scaled_budget_is_absent_when_there_is_nothing_to_scale():
    """Same size, same budget: a second identical run of two evaluators would be
    pure cost, and would make a same-size run differ from one without the flag."""
    from hopfield_nav.eval_all import step_budget

    class E:
        def __init__(self, s):
            self.size = s

    assert step_budget([E(SIZE), E(SIZE)], SIZE, 200) is None
    assert step_budget([E(SIZE)], None, 200) is None


def test_the_scaled_budget_grows_as_the_metric_does():
    """Coverage is a fraction of cells (size^2); a path is a length (size)."""
    from hopfield_nav.eval_all import step_budget

    class E:
        def __init__(self, s):
            self.size = s

    b = step_budget([E(12), E(12)], 6, 200)
    assert b["explore"] == 800        # 200 * (12/6)^2
    assert b["nav"] == 400            # 200 * (12/6)
    assert b["fixed"] == 200
    assert (b["trained_size"], b["val_size"]) == (6, 12)
