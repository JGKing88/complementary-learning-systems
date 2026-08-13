"""The checkpoint compatibility surface, pinned.

309 agent run directories are only readable through `config.py`'s dataclass
*field names*: a checkpoint is `asdict(cfg)`, and rebuilding it is a lookup by
name. Two renames have happened so far and both are carried by
`coerce_legacy_cfg` rather than by touching the dataclasses. If either clause is
dropped, every pre-rename checkpoint stops loading -- with a `TypeError` deep
inside a dataclass constructor, months after the change.

`build_eval_world` is not tested here: it needs a real encoder and a real
scaffold, and `test_smoke_train.py` already drives it end to end through four
entry points.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from hopfield_nav.policy.agent import compute_input_dim
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, coerce_legacy_cfg, load_agent,
)
from hopfield_nav.tests.fixtures import make_stub_cfg


def test_coerce_legacy_cfg_carries_both_renames():
    cd = coerce_legacy_cfg({
        "val_envs_per_world": 7,
        "vectorhash": {"gbook_only": True, "Np": 40},
    })
    assert cd["num_val_envs"] == 7
    assert "val_envs_per_world" not in cd
    assert cd["vectorhash"]["static_vectorhash"] is True
    assert "gbook_only" not in cd["vectorhash"]


def test_coerce_legacy_cfg_leaves_current_schema_alone():
    """A checkpoint written after the renames must pass through untouched."""
    cd = coerce_legacy_cfg({
        "num_val_envs": 3,
        "vectorhash": {"static_vectorhash": False, "Np": 40},
    })
    assert cd == {"num_val_envs": 3,
                  "vectorhash": {"static_vectorhash": False, "Np": 40}}


def test_coerce_legacy_cfg_prefers_the_new_name_when_both_are_present():
    cd = coerce_legacy_cfg({
        "val_envs_per_world": 7, "num_val_envs": 3,
        "vectorhash": {"gbook_only": True, "static_vectorhash": False},
    })
    assert cd["num_val_envs"] == 3
    assert cd["vectorhash"]["static_vectorhash"] is False


def test_cfg_from_checkpoint_round_trips_a_current_config():
    cfg = make_stub_cfg(movement_mode="continuous", input_sensory=True)
    assert asdict(cfg_from_checkpoint(asdict(cfg))) == asdict(cfg)


def test_cfg_from_checkpoint_round_trips_a_legacy_config():
    """The same config as an old checkpoint would have stored it."""
    cfg = make_stub_cfg()
    saved = asdict(cfg)
    saved["val_envs_per_world"] = saved.pop("num_val_envs")
    saved["vectorhash"]["gbook_only"] = saved["vectorhash"].pop(
        "static_vectorhash")
    assert asdict(cfg_from_checkpoint(saved)) == asdict(cfg)


def test_cfg_from_checkpoint_defaults_fields_the_checkpoint_predates():
    """A field added after a run must read as its default, not fail the load."""
    cfg = make_stub_cfg()
    saved = asdict(cfg)
    saved["env"].pop("allow_offcell_store")
    saved.pop("union_cov_trials")
    rebuilt = cfg_from_checkpoint(saved)
    assert rebuilt.env.allow_offcell_store is type(cfg.env)().allow_offcell_store
    assert rebuilt.union_cov_trials == type(cfg)().union_cov_trials


def test_cfg_from_checkpoint_ignores_keys_the_schema_dropped():
    cfg = make_stub_cfg()
    saved = asdict(cfg)
    saved["a_flag_that_no_longer_exists"] = 42
    rebuilt = cfg_from_checkpoint(saved)
    assert not hasattr(rebuilt, "a_flag_that_no_longer_exists")
    assert asdict(rebuilt) == asdict(cfg)


@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_load_agent_builds_the_width_the_channels_imply(movement_mode):
    cfg = make_stub_cfg(movement_mode=movement_mode, input_sensory=True,
                        input_prev_action=True, input_encoded_state=True)
    embed_dim = 8
    agent = load_agent(cfg, None, embed_dim, torch.device("cpu"))
    expected = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    assert agent.rnn.input_size == expected


def test_load_agent_round_trips_a_state_dict_and_leaves_eval_mode():
    cfg = make_stub_cfg()
    device = torch.device("cpu")
    source = load_agent(cfg, None, 8, device)
    loaded = load_agent(cfg, source.state_dict(), 8, device)
    assert loaded.training is False
    for k, v in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)


def test_load_agent_without_a_state_dict_keeps_the_fresh_init():
    """The random-agent control in phase_decoding_v2 depends on this branch.

    Its reproducibility rests on nothing in `load_agent` consuming the RNG
    before `NavAgent.__init__` does -- so seeding outside the call is enough.
    """
    cfg = make_stub_cfg()
    device = torch.device("cpu")
    torch.manual_seed(1234)
    a = load_agent(cfg, None, 8, device)
    torch.manual_seed(1234)
    b = load_agent(cfg, None, 8, device)
    for k, v in a.state_dict().items():
        assert torch.equal(b.state_dict()[k], v)

    torch.manual_seed(4321)
    c = load_agent(cfg, None, 8, device)
    assert not torch.equal(c.state_dict()["rnn.weight_ih_l0"],
                           a.state_dict()["rnn.weight_ih_l0"])


def test_agent_can_store_is_coerced_to_allow_store():
    """Every checkpoint written before 2026-08 carries the old field name.

    `cfg_from_checkpoint` builds `HopfieldConfig(**cd["hopfield"])`, so an
    unrecognised key is a TypeError, not a warning -- without the coercion the
    rename would make all 355 recorded run dirs unloadable.
    """
    from dataclasses import asdict
    from hopfield_nav.config import TrainConfig
    from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint

    saved = asdict(TrainConfig())
    saved["hopfield"]["agent_can_store"] = False
    del saved["hopfield"]["allow_store"]

    cfg = cfg_from_checkpoint(saved)
    assert cfg.hopfield.allow_store is False


def test_a_deleted_randomize_goal_flag_is_reported_not_dropped(capsys):
    """Two recorded runs set it, and its removal changes what they would do.

    Unlike the renames above this one cannot be mapped: `--refresh_goal` draws
    from the declared train partition rather than uniformly over the arena, and
    it demands `--env_generator`, which those runs never had. An unknown
    top-level key is silently ignored by `cfg_from_checkpoint`, so without this
    the behavior would vanish from a resumed run with nothing said.
    """
    from dataclasses import asdict
    from hopfield_nav.config import TrainConfig
    from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint

    saved = asdict(TrainConfig())
    saved["randomize_goal_per_rollout"] = True
    cfg = cfg_from_checkpoint(saved)
    assert not hasattr(cfg, "randomize_goal_per_rollout")
    assert "--refresh_goal" in capsys.readouterr().out

    # A run that had it off has nothing to report.
    saved["randomize_goal_per_rollout"] = False
    cfg_from_checkpoint(saved)
    assert "refresh_goal" not in capsys.readouterr().out


def test_a_current_checkpoint_still_round_trips():
    from dataclasses import asdict
    from hopfield_nav.config import TrainConfig
    from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint

    cfg = TrainConfig()
    cfg.hopfield.allow_store = False
    cfg.freeze_store = False
    back = cfg_from_checkpoint(asdict(cfg))
    assert back.hopfield.allow_store is False
    assert back.freeze_store is False
