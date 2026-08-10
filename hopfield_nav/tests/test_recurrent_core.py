"""The recurrent trunk: one factory, three cells, four contracts.

`policy/recurrent.py` is load-bearing in a way that is easy to under-test,
because two of the things it must guarantee produce no exception when they
break. The GRU path must stay bit-identical or every prior checkpoint silently
becomes a different model; and a T-step call must equal T single-step calls
carrying `h`, because the rollout collects one step at a time while the PPO
update re-runs the whole sequence -- if those disagree, nothing crashes, the
importance ratio is just wrong.

The rest of the file pins the four contracts everything downstream reads the
trunk through: `input_size`, `parameters()`, the `(num_layers, B, hidden)`
hidden state, and that chunking equivalence.
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hopfield_nav.config import (
    AgentConfig, EnvConfig, HopfieldConfig, PPOConfig, RNNAgentConfig,
    TrainConfig, VectorHashConfig, validate_recurrent_core,
    validate_train_config,
)
from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint
from hopfield_nav.policy.agent import NavAgent, compute_input_dim
from hopfield_nav.policy.agent_rnn import RNNAgent
from hopfield_nav.policy.recurrent import SoftplusRNN, build_recurrent_core

IN_DIM = 6
HIDDEN = 12

CELLS = [
    pytest.param(("gru", "tanh"), id="gru"),
    pytest.param(("rnn", "tanh"), id="rnn-tanh"),
    pytest.param(("rnn", "relu"), id="rnn-relu"),
    pytest.param(("rnn", "softplus"), id="rnn-softplus"),
]


def _cfg(cell="gru", nonlinearity="tanh", layers=1, dropout=0.0):
    return AgentConfig(
        hidden_size=HIDDEN, num_rnn_layers=layers, dropout=dropout,
        rnn_cell=cell, rnn_nonlinearity=nonlinearity,
    )


def _core(cell="gru", nonlinearity="tanh", layers=1, dropout=0.0, seed=0):
    torch.manual_seed(seed)
    return build_recurrent_core(_cfg(cell, nonlinearity, layers, dropout), IN_DIM)


# ---------------------------------------------------------------------------
# The factory dispatches, and the default is the historical GRU
# ---------------------------------------------------------------------------

def test_default_config_still_builds_a_gru():
    """Every checkpoint in the tree was trained under this branch."""
    assert isinstance(build_recurrent_core(AgentConfig(), IN_DIM), nn.GRU)
    assert isinstance(build_recurrent_core(RNNAgentConfig(), IN_DIM), nn.GRU)


def test_gru_path_is_unchanged():
    """Same construction arguments as the hand-written block it replaced, so
    the same seed gives the same weights and the same forward."""
    torch.manual_seed(0)
    built = build_recurrent_core(_cfg("gru", "tanh", layers=2, dropout=0.3), IN_DIM)
    torch.manual_seed(0)
    direct = nn.GRU(IN_DIM, HIDDEN, num_layers=2, batch_first=True, dropout=0.3)

    assert sorted(built.state_dict()) == sorted(direct.state_dict())
    for k, v in direct.state_dict().items():
        assert torch.equal(built.state_dict()[k], v)
    built.eval(), direct.eval()
    x = torch.randn(3, 5, IN_DIM)
    assert torch.allclose(built(x)[0], direct(x)[0])


def test_tanh_and_relu_are_plain_nn_rnn():
    """These stay on cuDNN; only softplus pays for a Python recurrence."""
    for nonlinearity in ("tanh", "relu"):
        core = _core("rnn", nonlinearity)
        assert isinstance(core, nn.RNN) and not isinstance(core, SoftplusRNN)
        assert core.nonlinearity == nonlinearity
    assert isinstance(_core("rnn", "softplus"), SoftplusRNN)


def test_single_layer_ignores_dropout_like_the_original():
    """The replaced block passed dropout=0.0 whenever num_layers == 1, because
    torch warns and ignores it there."""
    assert _core("gru", "tanh", layers=1, dropout=0.5).dropout == 0.0
    assert _core("rnn", "softplus", layers=1, dropout=0.5).dropout == 0.0
    assert _core("gru", "tanh", layers=2, dropout=0.5).dropout == 0.5


# ---------------------------------------------------------------------------
# Softplus correctness
# ---------------------------------------------------------------------------

def test_softplus_matches_a_reference_recurrence():
    """Against the recurrence written out longhand, no hoisting, no fusing."""
    core = _core("rnn", "softplus", layers=2)
    core.eval()
    x = torch.randn(4, 7, IN_DIM)
    out, h_n = core(x)

    layer_in = x
    ref_h_last = []
    for layer in range(core.num_layers):
        w_ih = getattr(core, f"weight_ih_l{layer}")
        w_hh = getattr(core, f"weight_hh_l{layer}")
        b_ih = getattr(core, f"bias_ih_l{layer}")
        b_hh = getattr(core, f"bias_hh_l{layer}")
        h = torch.zeros(x.shape[0], HIDDEN)
        steps = []
        for t in range(x.shape[1]):
            h = F.softplus(layer_in[:, t] @ w_ih.t() + b_ih + h @ w_hh.t() + b_hh)
            steps.append(h)
        layer_in = torch.stack(steps, dim=1)
        ref_h_last.append(h)

    assert torch.allclose(out, layer_in, atol=1e-6)
    assert torch.allclose(h_n, torch.stack(ref_h_last, dim=0), atol=1e-6)


def test_softplus_output_is_strictly_positive():
    """The property that distinguishes it from tanh, and the one an accidental
    fallback to the base class's forward would break."""
    core = _core("rnn", "softplus")
    core.eval()
    out, h_n = core(torch.randn(4, 9, IN_DIM) * 5.0)
    assert (out > 0).all()
    assert (h_n > 0).all()


def test_softplus_state_does_not_diverge_over_a_long_rollout():
    """Softplus is unbounded above, so unlike tanh nothing structurally caps
    the state. At the inherited init it settles rather than growing; this pins
    that, since a divergence would surface downstream as a dead value head
    rather than as an exception here.
    """
    torch.manual_seed(0)
    core = build_recurrent_core(
        AgentConfig(hidden_size=128, rnn_cell="rnn", rnn_nonlinearity="softplus"),
        IN_DIM)
    core.eval()
    with torch.no_grad():
        out, _ = core(torch.randn(8, 400, IN_DIM))
    norms = out.norm(dim=-1).mean(dim=0)
    assert torch.isfinite(norms).all()
    # Late-sequence norm must not be a multiple of the early-sequence norm.
    assert norms[-1] < 2.0 * norms[10]


def test_softplus_gradients_are_finite():
    core = _core("rnn", "softplus")
    out, _ = core(torch.randn(3, 64, IN_DIM))
    out.pow(2).mean().backward()
    for name, p in core.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_softplus_rejects_an_unbatched_sequence():
    with pytest.raises(ValueError, match="B, T, input_size"):
        _core("rnn", "softplus")(torch.randn(5, IN_DIM))


# ---------------------------------------------------------------------------
# The four downstream contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("layers", [1, 2])
def test_hidden_state_layout(cell, layers):
    """`(num_layers, B, hidden)` -- what evaluation/protocols.py slices with
    `h_rnn[-1, 0]` and what the collector carries between steps."""
    core = _core(*cell, layers=layers)
    core.eval()
    out, h_n = core(torch.randn(5, 8, IN_DIM))
    assert out.shape == (5, 8, HIDDEN)
    assert h_n.shape == (layers, 5, HIDDEN)


@pytest.mark.parametrize("cell", CELLS)
def test_input_size_attribute(cell):
    """`rollout/collector.py` sizes its observation buffer off this."""
    assert _core(*cell).input_size == IN_DIM


@pytest.mark.parametrize("cell", CELLS)
def test_parameters_are_registered(cell):
    """`training/world_setup.rnn_params` freezes the trunk through this."""
    core = _core(*cell)
    assert len(list(core.parameters())) > 0
    assert all(p.requires_grad for p in core.parameters())


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("layers", [1, 2])
def test_chunked_and_whole_sequence_agree(cell, layers):
    """The rollout takes T single steps carrying `h`; the PPO update re-runs
    the whole sequence from `h=None`. PPO's importance ratio is only defined
    if those are the same function -- and a divergence here raises nothing.
    """
    core = _core(*cell, layers=layers)
    core.eval()
    x = torch.randn(4, 16, IN_DIM)
    whole, h_whole = core(x)

    h = None
    steps = []
    for t in range(x.shape[1]):
        out_t, h = core(x[:, t:t + 1], h)
        steps.append(out_t)
    assert torch.allclose(torch.cat(steps, dim=1), whole, atol=1e-5)
    assert torch.allclose(h, h_whole, atol=1e-5)


def test_inter_layer_dropout_is_wired_and_train_mode_only():
    core = _core("rnn", "softplus", layers=2, dropout=0.5)
    x = torch.randn(4, 6, IN_DIM)

    core.eval()
    assert torch.allclose(core(x)[0], core(x)[0])

    core.train()
    torch.manual_seed(1)
    a = core(x)[0]
    torch.manual_seed(2)
    b = core(x)[0]
    assert not torch.allclose(a, b), "dropout had no effect in train mode"


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------

def test_softplus_and_tanh_share_a_state_dict():
    """Same parameter names and shapes, so the nonlinearity is an ablation
    axis: a tanh run's weights load into a softplus model and back."""
    softplus = _core("rnn", "softplus", layers=2, dropout=0.1)
    tanh = _core("rnn", "tanh", layers=2, dropout=0.1, seed=1)
    tanh.load_state_dict(softplus.state_dict())
    softplus.load_state_dict(tanh.state_dict())
    for k, v in softplus.state_dict().items():
        assert torch.equal(tanh.state_dict()[k], v)


def test_gru_weights_do_not_silently_load_into_a_vanilla_cell():
    """A GRU's weight_ih_l0 is 3H x D against a vanilla cell's H x D. The
    mismatch must fail loudly rather than partially load."""
    gru = _core("gru", "tanh")
    rnn = _core("rnn", "tanh")
    with pytest.raises(RuntimeError, match="size mismatch"):
        rnn.load_state_dict(gru.state_dict())


def test_new_fields_survive_a_checkpoint_roundtrip():
    cfg = TrainConfig(
        env=EnvConfig(), vectorhash=VectorHashConfig(), hopfield=HopfieldConfig(),
        agent=_cfg("rnn", "softplus"), ppo=PPOConfig(),
    )
    restored = cfg_from_checkpoint(asdict(cfg))
    assert restored.agent.rnn_cell == "rnn"
    assert restored.agent.rnn_nonlinearity == "softplus"


def test_checkpoint_predating_the_fields_reads_as_a_gru():
    """309 run dirs were written before these fields existed."""
    saved = asdict(TrainConfig(
        env=EnvConfig(), vectorhash=VectorHashConfig(), hopfield=HopfieldConfig(),
        agent=AgentConfig(), ppo=PPOConfig(),
    ))
    saved["agent"].pop("rnn_cell")
    saved["agent"].pop("rnn_nonlinearity")
    restored = cfg_from_checkpoint(saved)
    assert restored.agent.rnn_cell == "gru"
    assert restored.agent.rnn_nonlinearity == "tanh"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_gru_with_a_nonlinearity_is_an_error_not_a_no_op():
    with pytest.raises(ValueError, match="no selectable nonlinearity"):
        validate_recurrent_core("gru", "softplus")


@pytest.mark.parametrize("cell,nonlinearity", [
    ("lstm", "tanh"), ("GRU", "tanh"), ("rnn", "sigmoid"), ("rnn", "Softplus"),
])
def test_unknown_values_are_rejected(cell, nonlinearity):
    with pytest.raises(ValueError):
        validate_recurrent_core(cell, nonlinearity)


def test_the_factory_itself_validates():
    """`train_rnn` builds an RNNTrainConfig, which no validate_train_config
    ever sees -- so the factory has to be the backstop."""
    with pytest.raises(ValueError):
        build_recurrent_core(_cfg("gru", "relu"), IN_DIM)


def test_train_config_validation_catches_it_at_startup():
    cfg = TrainConfig(
        env=EnvConfig(), vectorhash=VectorHashConfig(), hopfield=HopfieldConfig(),
        agent=_cfg("gru", "softplus"), ppo=PPOConfig(),
    )
    with pytest.raises(ValueError, match="no selectable nonlinearity"):
        validate_train_config(cfg)
    cfg.agent = _cfg("rnn", "softplus")
    validate_train_config(cfg)  # should not raise


# ---------------------------------------------------------------------------
# The agents that own a trunk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", CELLS)
def test_nav_agent_forward_and_heads(cell):
    cfg = _cfg(*cell)
    cfg.input_encoded_state = False
    cfg.input_hopfield_signal = True
    agent = NavAgent(cfg, compute_input_dim(cfg, embed_dim=4, sensory_dim=0))
    agent.eval()
    x = torch.randn(3, 5, agent.rnn.input_size)
    move_dist, store_dist, values, h_n, features = agent(x, return_features=True)
    assert values.shape == (3, 5)
    assert features.shape == (3, 5, HIDDEN)
    assert h_n.shape == (1, 3, HIDDEN)
    assert move_dist.logits.shape == (3, 5, 4)
    assert store_dist.logits.shape == (3, 5)


@pytest.mark.parametrize("cell", CELLS)
def test_rnn_agent_act(cell):
    cfg = RNNAgentConfig(hidden_size=HIDDEN, rnn_cell=cell[0],
                         rnn_nonlinearity=cell[1])
    agent = RNNAgent(cfg, IN_DIM)
    out = agent.act(torch.randn(3, 1, IN_DIM))
    assert out["move_action"].shape == (3,)
    assert out["h_next"].shape == (1, 3, HIDDEN)


@pytest.mark.parametrize("cell", CELLS)
def test_trunk_freeze_covers_every_cell(cell):
    """`set_phase_freeze` reaches the trunk through `rnn_params`."""
    from hopfield_nav.training.world_setup import rnn_params, set_requires_grad
    cfg = _cfg(*cell)
    agent = NavAgent(cfg, compute_input_dim(cfg, embed_dim=4, sensory_dim=0))
    params = rnn_params(agent)
    assert len(params) > 0
    set_requires_grad(params, False)
    assert not any(p.requires_grad for p in agent.rnn.parameters())


@pytest.mark.parametrize("cell", CELLS)
def test_trunk_actually_trains(cell):
    """A step of SGD moves every trunk parameter -- catches a forward that
    reads the weights through a path the autograd graph does not reach."""
    core = _core(*cell)
    before = copy.deepcopy(core.state_dict())
    opt = torch.optim.SGD(core.parameters(), lr=0.5)
    out, _ = core(torch.randn(4, 12, IN_DIM))
    out.pow(2).mean().backward()
    opt.step()
    for name, tensor in core.state_dict().items():
        assert not torch.equal(before[name], tensor), f"{name} did not move"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_navigate_cli_reaches_the_agent_config():
    """The flag has to be in `CFG_FIELDS` too, not just on the parser -- a
    missing table entry is silent, and silently a GRU."""
    from hopfield_nav.train_navigate import CFG_FIELDS, build_parser
    assert CFG_FIELDS["rnn_cell"] == ("agent.rnn_cell",)
    assert CFG_FIELDS["rnn_nonlinearity"] == ("agent.rnn_nonlinearity",)
    args = build_parser().parse_args(
        ["--encoder_checkpoint", "stub",
         "--rnn_cell", "rnn", "--rnn_nonlinearity", "softplus"])
    assert args.rnn_cell == "rnn"
    assert args.rnn_nonlinearity == "softplus"


def test_navigate_cli_defaults_to_the_gru():
    from hopfield_nav.train_navigate import build_parser
    args = build_parser().parse_args(["--encoder_checkpoint", "stub"])
    assert args.rnn_cell == "gru"
    assert args.rnn_nonlinearity == "tanh"


def test_cli_rejects_an_unknown_cell():
    from hopfield_nav.train_navigate import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--encoder_checkpoint", "stub", "--rnn_cell", "lstm"])


def test_every_flag_in_the_table_exists_on_the_parser():
    """Guards the reverse drift: a table entry whose flag was renamed."""
    from hopfield_nav.train_navigate import CFG_FIELDS, build_parser
    dests = {a.dest for a in build_parser()._actions}
    missing = sorted(set(CFG_FIELDS) - dests)
    assert not missing, f"CFG_FIELDS names flags the parser does not have: {missing}"


def test_finetune_restores_the_cell_from_the_checkpoint():
    """`rnn_cell` changes parameter shapes, so `restore_arch_from_ckpt` has to
    carry it or a softplus finetune rebuilds a GRU and fails to load."""
    from hopfield_nav.config import RNNTrainConfig
    from hopfield_nav.training.rnn_setup import restore_arch_from_ckpt
    cfg = RNNTrainConfig()
    assert cfg.agent.rnn_cell == "gru"
    restore_arch_from_ckpt(
        cfg, {"cfg": {"agent": {"rnn_cell": "rnn",
                                "rnn_nonlinearity": "softplus"}}})
    assert cfg.agent.rnn_cell == "rnn"
    assert cfg.agent.rnn_nonlinearity == "softplus"


def test_finetune_of_a_legacy_checkpoint_stays_a_gru():
    from hopfield_nav.config import RNNTrainConfig
    from hopfield_nav.training.rnn_setup import restore_arch_from_ckpt
    cfg = RNNTrainConfig()
    restore_arch_from_ckpt(cfg, {"cfg": {"agent": {"hidden_size": 64}}})
    assert cfg.agent.rnn_cell == "gru"
    assert cfg.agent.rnn_nonlinearity == "tanh"
