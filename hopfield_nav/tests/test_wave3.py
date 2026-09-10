"""Behavioural tests for the parameter-isolation family (Wave 3).

Same standard as `test_continual_methods.py`, and for the same reason: every
failure mode in this family is silent. A hypernetwork whose regulariser
contributes no gradient trains exactly like one with no regulariser at all and
reports a sensible-looking penalty the whole time -- which is precisely the
defect that let DER++ run as plain Experience Replay for two waves. A
multi-head policy that is never told which task it is in evaluates every env
under head 0 and produces a curve indistinguishable from catastrophic
forgetting. A gating mask that is built but not applied changes nothing and
crashes nothing.

So the tests here check gradients, not loss values; identity of parameters, not
their existence; and equivalence against the baseline policy wherever the new
code is supposed to reproduce it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import (
    EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig)
from hopfield_nav.continual.base import CONTINUAL_METHODS, build_method
from hopfield_nav.continual.hypernet import HypernetOutputReg
from hopfield_nav.policy.agent_rnn import (
    RNNAgent, compute_rnn_input_dim, set_agent_task)
from hopfield_nav.policy.hypernet import ChunkedHypernet, HyperRNNAgent
from hopfield_nav.policy.isolate import (
    MultiHeadRNNAgent, XdGRNNAgent, warm_start)
from hopfield_nav.training.rnn_sequential import run_sequential_blocks
from hopfield_nav.world.env import GridEnv

OBS = 16
SIZE = 6
HID = 8


def _cfg(**kw) -> RNNAgentConfig:
    base = dict(hidden_size=HID, movement_mode="continuous",
                init_log_std=-1.0, freeze_log_std=False)
    base.update(kw)
    return RNNAgentConfig(**base)


def _hyper(n_tasks=3, chunk_dim=32, base="learned", cfg=None, **kw):
    cfg = cfg or _cfg()
    return HyperRNNAgent(cfg, compute_rnn_input_dim(cfg, OBS), n_tasks,
                         chunk_dim=chunk_dim, base=base, emb_dim=8,
                         hyper_hidden=(16,), **kw)


def _x(b=2, t=4):
    return torch.randn(b, t, OBS)


# ===========================================================================
# ChunkedHypernet
# ===========================================================================

def test_chunked_output_is_exactly_out_dim():
    """The last chunk is partial and must be trimmed, not padded or dropped."""
    h = ChunkedHypernet(3, out_dim=1000, emb_dim=8, chunk_dim=64, hidden=(16,))
    assert h.n_chunks == 16                       # 15*64=960 < 1000 <= 16*64
    assert h(0).shape == (1000,)


def test_chunked_and_unchunked_both_produce_the_target_size():
    for chunk in (0, 7, 64, 4096):
        h = ChunkedHypernet(2, out_dim=500, emb_dim=8, chunk_dim=chunk,
                            hidden=(16,))
        assert h(0).shape == (500,), f"chunk_dim={chunk}"


def test_different_tasks_generate_different_weights():
    """The whole method rests on this. If the generator learned to ignore its
    task embedding, every arm would collapse to the shared-weights baseline
    while still reporting a hypernetwork in its metadata."""
    h = ChunkedHypernet(3, out_dim=200, emb_dim=8, chunk_dim=32, hidden=(16,))
    a, b = h(0), h(1)
    assert not torch.allclose(a, b)


def test_task_index_out_of_range_is_an_error():
    h = ChunkedHypernet(2, out_dim=50, emb_dim=4, chunk_dim=16, hidden=(8,))
    with pytest.raises(IndexError):
        h(2)
    with pytest.raises(IndexError):
        h(-1)


def test_init_out_scale_shrinks_the_generated_weights():
    torch.manual_seed(0)
    big = ChunkedHypernet(2, 200, emb_dim=8, chunk_dim=32, hidden=(16,),
                          init_out_scale=1.0)
    torch.manual_seed(0)
    small = ChunkedHypernet(2, 200, emb_dim=8, chunk_dim=32, hidden=(16,),
                            init_out_scale=0.01)
    assert small(0).abs().mean() < 0.05 * big(0).abs().mean()


# ===========================================================================
# HyperRNNAgent: equivalence with the baseline policy
# ===========================================================================

def test_generated_forward_reproduces_the_baseline_rnn_exactly():
    """The load-bearing test for `functional_call`.

    With the generator's output zeroed and the base set to an `RNNAgent`'s own
    weights, the hypernetwork agent is that `RNNAgent` -- so any difference here
    is a difference in the forward pass itself, which would show up in every
    Wave 3 number as a method effect that was really an implementation gap.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    plain = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))
    hyp = _hyper(base="frozen", cfg=cfg)
    hyp.warm_start_from(plain.state_dict())
    with torch.no_grad():
        hyp.hyper.head.weight.zero_()
        hyp.hyper.head.bias.zero_()

    x = _x()
    hyp.set_task(0)
    d_ref, h_ref = plain(x)
    d_gen, h_gen = hyp(x)
    assert torch.allclose(d_ref.mean, d_gen.mean, atol=1e-6)
    assert torch.allclose(d_ref.stddev, d_gen.stddev, atol=1e-6)
    assert torch.allclose(h_ref, h_gen, atol=1e-6)


def test_warm_start_puts_every_task_at_the_pretrained_policy():
    """Comparability with the pretrained controls depends on this: if task 3
    started somewhere else, its block-3 curve would be measuring the warm start
    rather than the method."""
    torch.manual_seed(0)
    cfg = _cfg()
    plain = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))
    hyp = _hyper(base="learned", cfg=cfg, init_out_scale=1e-4)
    hyp.warm_start_from(plain.state_dict())
    x = _x()
    ref = plain(x)[0].mean
    for t in range(hyp.n_tasks):
        hyp.set_task(t)
        assert torch.allclose(ref, hyp(x)[0].mean, atol=1e-2), f"task {t}"


def test_warm_start_refuses_when_there_is_no_base():
    hyp = _hyper(base="none")
    cfg = _cfg()
    plain = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))
    with pytest.raises(ValueError, match="no base vector"):
        hyp.warm_start_from(plain.state_dict())


def test_warm_start_refuses_a_checkpoint_that_is_not_this_policy():
    hyp = _hyper()
    with pytest.raises(KeyError, match="missing"):
        hyp.warm_start_from({"something.else": torch.zeros(3)})


def test_template_parameters_are_not_the_agents_parameters():
    """The template exists for its forward pass and its shapes. If it were
    registered, the optimiser would carry state for 74k tensors that can never
    move and every parameter count in the metadata would be double."""
    hyp = _hyper()
    names = {n for n, _ in hyp.named_parameters()}
    assert not any(n.startswith("_tmpl") for n in names)
    assert not any(n.startswith("rnn.") for n in names)
    assert "base" in names and any(n.startswith("hyper.") for n in names)


def test_acting_without_a_task_raises_rather_than_defaulting():
    """Defaulting to task 0 would evaluate every env under one task's weights,
    which is exactly what catastrophic forgetting looks like."""
    hyp = _hyper()
    with pytest.raises(RuntimeError, match="no active task"):
        hyp(_x())


def test_freeze_log_std_is_refused_rather_than_ignored():
    """log_std is generated like every other weight, so there is no leaf to
    freeze. This project has already paid for one silently-ignored freeze flag."""
    cfg = _cfg(freeze_log_std=True)
    with pytest.raises(ValueError, match="freeze_log_std"):
        HyperRNNAgent(cfg, compute_rnn_input_dim(cfg, OBS), 2)


def test_frozen_base_is_a_buffer_and_never_receives_gradient():
    hyp = _hyper(base="frozen")
    hyp.set_task(0)
    hyp(_x())[0].mean.sum().backward()
    assert not isinstance(hyp.base, torch.nn.Parameter)
    assert hyp.base.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in hyp.hyper.parameters())


def test_learned_base_does_receive_gradient():
    hyp = _hyper(base="learned")
    hyp.set_task(0)
    hyp(_x())[0].mean.sum().backward()
    assert hyp.base.grad is not None and hyp.base.grad.abs().sum() > 0


# ===========================================================================
# HyperRNNAgent: the weight cache
# ===========================================================================

def test_cache_does_not_survive_a_parameter_update():
    """The cache saves regenerating 74k weights at every environment step. It is
    only safe because a training forward clears it; if it were not, evaluation
    after an update would score the *previous* policy and every curve would lag
    one update behind."""
    hyp = _hyper()
    hyp.set_task(0)
    with torch.no_grad():
        before = hyp(_x())[0].mean.clone()

    opt = torch.optim.SGD(hyp.parameters(), lr=1.0)
    hyp(_x())[0].mean.sum().backward()        # grad-enabled: clears the cache
    opt.step()

    with torch.no_grad():
        after = hyp(_x())[0].mean
    assert not torch.allclose(before, after)


def test_set_task_invalidates_the_cache():
    hyp = _hyper()
    x = _x()
    with torch.no_grad():
        hyp.set_task(0)
        a = hyp(x)[0].mean.clone()
        hyp.set_task(1)
        b = hyp(x)[0].mean.clone()
    assert not torch.allclose(a, b)


def test_cache_is_used_when_grad_is_off():
    hyp = _hyper()
    hyp.set_task(0)
    with torch.no_grad():
        hyp(_x())
        cached = hyp._cache
        hyp(_x())
        assert hyp._cache is cached          # same objects, not regenerated


# ===========================================================================
# HypernetOutputReg
# ===========================================================================

def _end_block(method, hyp, block):
    method.on_block_end(block, hyp, [])
    method.on_block_start(block + 1, hyp, [])


def test_no_penalty_before_any_block_ends():
    m = HypernetOutputReg(beta=1.0)
    hyp = _hyper()
    m.on_block_start(0, hyp, [])
    assert m.penalty(hyp) is None


def test_penalty_is_zero_at_the_snapshot_and_grows_with_drift():
    m = HypernetOutputReg(beta=1.0)
    hyp = _hyper()
    _end_block(m, hyp, 0)
    assert float(m.penalty(hyp).detach()) == pytest.approx(0.0, abs=1e-12)

    with torch.no_grad():
        for p in hyp.hyper.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert float(m.penalty(hyp).detach()) > 0.0


def test_penalty_carries_gradient_to_the_generator():
    """The DER++ defect, checked directly. A penalty computed against a detached
    generator output is nonzero, scales correctly with beta, and moves nothing;
    every value-based assertion in this file would still pass."""
    m = HypernetOutputReg(beta=1.0)
    hyp = _hyper()
    _end_block(m, hyp, 0)
    with torch.no_grad():
        for p in hyp.hyper.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    pen = m.penalty(hyp)
    assert pen.requires_grad, "penalty is detached from the generator"
    hyp.zero_grad()
    pen.backward()
    total = sum(float(p.grad.abs().sum()) for _, p in hyp.generator_parameters()
                if p.grad is not None)
    assert total > 0.0, "penalty produced no gradient on the generator"


def test_penalty_scales_with_beta():
    hyp = _hyper()
    ms = [HypernetOutputReg(beta=b) for b in (1.0, 10.0)]
    for m in ms:
        _end_block(m, hyp, 0)
    with torch.no_grad():
        for p in hyp.hyper.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert float(ms[1].penalty(hyp).detach()) == pytest.approx(
        10.0 * float(ms[0].penalty(hyp).detach()), rel=1e-5)


def test_beta_zero_is_no_penalty_at_all():
    m = HypernetOutputReg(beta=0.0)
    hyp = _hyper()
    _end_block(m, hyp, 0)
    assert m.penalty(hyp) is None


def test_penalty_covers_every_task_seen_so_far():
    m = HypernetOutputReg(beta=1.0)
    hyp = _hyper(n_tasks=3)
    _end_block(m, hyp, 0)
    assert set(m._targets) == {0}
    _end_block(m, hyp, 1)
    assert set(m._targets) == {0, 1}


def test_targets_are_the_weights_as_of_the_boundary():
    """Not the weights as of task 0's start, and not live ones. If the target
    tracked the live generator the penalty would be identically zero forever."""
    m = HypernetOutputReg(beta=1.0)
    hyp = _hyper()
    with torch.no_grad():
        for p in hyp.hyper.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    _end_block(m, hyp, 0)
    expected = hyp.generate(0).detach()
    assert torch.allclose(m._targets[0], expected, atol=1e-6)

    with torch.no_grad():
        for p in hyp.hyper.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert torch.allclose(m._targets[0], expected, atol=1e-6)


def test_large_beta_actually_restrains_the_past_task_weights():
    """The mutation check: the penalty has to change the optimisation, not just
    appear in it. Same seed, same target, two betas.

    The task-1 objective is a regression on the *generated weights* rather than
    a policy loss, because it needs to pull hard and in a known direction. Every
    task's weights come out of one shared generator, so dragging task 1's
    somewhere new is exactly what drags task 0's along with it -- which is the
    forgetting this regulariser exists to resist.
    """
    def drift(beta):
        torch.manual_seed(0)
        hyp = _hyper()
        m = HypernetOutputReg(beta=beta)
        _end_block(m, hyp, 0)
        w0 = m._targets[0].clone()
        torch.manual_seed(1)
        target = torch.randn(hyp.out_dim)
        opt = torch.optim.SGD(hyp.parameters(), lr=0.05)
        for _ in range(20):
            loss = (hyp.generate(1) - target).pow(2).mean()
            pen = m.penalty(hyp)
            if pen is not None:
                loss = loss + pen
            opt.zero_grad()
            loss.backward()
            # As the real update does. Without it a large beta simply diverges,
            # which would make this test pass for the wrong reason.
            torch.nn.utils.clip_grad_norm_(hyp.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            return float((hyp.generate(0) - w0).norm())

    drifts = [drift(b) for b in (0.0, 10.0, 1e3)]
    assert all(np.isfinite(d) for d in drifts), drifts
    # Monotone in beta, rather than one threshold: a single cut-off can be met
    # by a penalty that merely destabilises the optimisation, and the gradient
    # clipping above caps how hard any beta can pull, so the *ordering* is the
    # claim worth making. Measured here: 0.47 -> 0.28 -> 0.12.
    assert drifts[0] > drifts[1] > drifts[2], drifts
    assert drifts[2] < 0.5 * drifts[0], drifts


def test_snapshot_size_does_not_grow_with_the_number_of_tasks():
    """The reason the targets are recomputed from a snapshot rather than kept:
    a method whose memory grows per task sits somewhere else on the frontier."""
    m = HypernetOutputReg(beta=1.0)
    hyp = _hyper(n_tasks=4)
    _end_block(m, hyp, 0)
    one = m.state_bytes()
    _end_block(m, hyp, 1)
    _end_block(m, hyp, 2)
    assert m.state_bytes() == one > 0


def test_frozen_base_is_not_counted_in_the_snapshot():
    """It cannot move, so a copy of it would be constant overhead nothing reads."""
    learned = HypernetOutputReg(beta=1.0)
    frozen = HypernetOutputReg(beta=1.0)
    _end_block(learned, _hyper(base="learned"), 0)
    _end_block(frozen, _hyper(base="frozen"), 0)
    assert frozen.state_bytes() < learned.state_bytes()


def test_method_declares_both_costs():
    m = build_method("hnet", beta=2.0)
    d = m.describe()
    assert d["needs_task_boundaries"] is True
    assert d["needs_task_id"] is True
    assert d["beta"] == 2.0


def test_hnet_is_registered_and_coerces_its_args():
    assert "hnet" in CONTINUAL_METHODS
    m = build_method("hnet", beta="1e3", normalize="false")
    assert m.beta == 1000.0 and m.normalize is False


def test_hnet_rejects_a_policy_with_no_generator():
    """A plain RNN plus --method hnet is a run that would otherwise train
    naively and be filed under the headline method."""
    cfg = _cfg()
    plain = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))
    with pytest.raises(TypeError, match="no generator"):
        HypernetOutputReg(beta=1.0).on_block_start(0, plain, [])


def test_normalize_false_is_the_papers_sum_form():
    hyp = _hyper()
    mean_m = HypernetOutputReg(beta=1.0, normalize=True)
    sum_m = HypernetOutputReg(beta=1.0, normalize=False)
    for m in (mean_m, sum_m):
        _end_block(m, hyp, 0)
    with torch.no_grad():
        for p in hyp.hyper.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    n = hyp.out_dim
    assert float(sum_m.penalty(hyp).detach()) == pytest.approx(
        n * float(mean_m.penalty(hyp).detach()), rel=1e-4)


# ===========================================================================
# MultiHeadRNNAgent
# ===========================================================================

def _multi(n_tasks=3):
    cfg = _cfg()
    return MultiHeadRNNAgent(cfg, compute_rnn_input_dim(cfg, OBS), n_tasks)


def test_multihead_uses_the_selected_head():
    m = _multi()
    x = _x()
    m.set_task(0)
    a = m(x)[0].mean.clone()
    m.set_task(1)
    assert not torch.allclose(a, m(x)[0].mean)


def test_multihead_trains_only_the_active_head():
    """The isolation claim, checked as a gradient rather than as an intention."""
    m = _multi()
    m.set_task(1)
    m(_x())[0].mean.sum().backward()
    grads = [any(p.grad is not None and p.grad.abs().sum() > 0
                 for p in head.parameters()) for head in m.heads]
    assert grads == [False, True, False]


def test_multihead_warm_start_copies_one_head_to_all():
    torch.manual_seed(0)
    cfg = _cfg()
    plain = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))
    m = _multi()
    warm_start(m, plain.state_dict())
    x = _x()
    ref = plain(x)[0].mean
    for t in range(m.n_tasks):
        m.set_task(t)
        assert torch.allclose(ref, m(x)[0].mean, atol=1e-6), f"task {t}"


def test_multihead_warm_start_rejects_a_headless_checkpoint():
    m = _multi()
    with pytest.raises(KeyError, match="no movement head"):
        warm_start(m, {"rnn.weight_ih_l0": torch.zeros(3)})


def test_multihead_without_a_task_raises():
    with pytest.raises(RuntimeError, match="no active task"):
        _multi()(_x())


def test_multihead_head_prefixes_leave_something_trainable():
    """`freeze_trunk` looks for `movement_*` at the top level by default, which
    matches nothing here and would freeze the entire policy."""
    from analysis.continual.baseline import freeze_trunk_params
    m = _multi()
    n_frozen, n_trainable = freeze_trunk_params(m)
    assert n_frozen > 0 and n_trainable > 0
    assert all(not p.requires_grad for p in m.rnn.parameters())
    assert all(p.requires_grad for p in m.heads.parameters())


# ===========================================================================
# XdGRNNAgent
# ===========================================================================

def _xdg(n_tasks=3, gating=0.5, seed=0):
    cfg = _cfg()
    return XdGRNNAgent(cfg, compute_rnn_input_dim(cfg, OBS), n_tasks,
                       gating=gating, seed=seed)


def test_xdg_masks_have_the_requested_density_and_differ_per_task():
    x = _xdg(gating=0.5)
    assert x.masks.shape == (3, HID)
    assert set(x.masks.unique().tolist()) <= {0.0, 1.0}
    assert all(int(row.sum()) == HID // 2 for row in x.masks)
    assert not torch.equal(x.masks[0], x.masks[1])


def test_xdg_masks_are_a_function_of_the_seed_alone():
    """Drawn from a dedicated generator, so two runs differing only in an
    unrelated flag get the same masks."""
    assert torch.equal(_xdg(seed=7).masks, _xdg(seed=7).masks)
    assert not torch.equal(_xdg(seed=7).masks, _xdg(seed=8).masks)


def test_xdg_gating_zero_keeps_every_unit():
    assert float(_xdg(gating=0.0).masks.sum()) == 3 * HID


def test_xdg_always_leaves_at_least_one_unit():
    x = _xdg(gating=0.99)
    assert all(int(row.sum()) >= 1 for row in x.masks)


def test_xdg_state_is_masked_inside_the_recurrence_not_just_at_the_readout():
    """The difference between XdG and a masked readout. Checked against a manual
    step-wise reference: if the mask were applied only to the features, the
    hidden state fed back in would be unmasked and this would diverge."""
    a = _xdg(gating=0.5)
    a.set_task(1)
    x = _x(b=2, t=5)
    _, h_out = a(x)

    mask = a.masks[1].view(1, 1, -1)
    h = torch.zeros(1, 2, HID) * mask
    with torch.no_grad():
        for t in range(5):
            _, h = a.rnn(x[:, t:t + 1], h)
            h = h * mask
    assert torch.allclose(h, h_out, atol=1e-6)


def test_xdg_hidden_state_stays_inside_the_mask():
    a = _xdg(gating=0.5)
    a.set_task(2)
    _, h = a(_x())
    off = (a.masks[2] == 0)
    assert float(h[0][:, off].abs().max().detach()) == 0.0


def test_xdg_different_tasks_give_different_policies():
    a = _xdg(gating=0.5)
    x = _x()
    a.set_task(0)
    first = a(x)[0].mean.clone()
    a.set_task(1)
    assert not torch.allclose(first, a(x)[0].mean)


def test_xdg_rejects_a_gating_fraction_of_one():
    with pytest.raises(ValueError, match="gating"):
        _xdg(gating=1.0)


def test_xdg_describe_reports_the_active_width():
    d = _xdg(gating=0.75).describe()
    assert d["arch"] == "xdg"
    assert d["units_active_per_task"] == HID // 4
    assert d["gating"] == 0.75


# ===========================================================================
# driver plumbing
# ===========================================================================

def test_set_agent_task_distinguishes_the_two_kinds_of_policy():
    cfg = _cfg()
    assert set_agent_task(RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS)), 0) is False
    assert set_agent_task(_multi(), 1) is True
    assert set_agent_task(_hyper(), 1) is True
    assert set_agent_task(_xdg(), 1) is True


def _tiny_cfg(n_envs, updates):
    return RNNTrainConfig(
        env=EnvConfig(size=SIZE, observation_size=OBS,
                      movement_mode="continuous", goal_radius=0.5),
        agent=_cfg(),
        bc=RNNBCConfig(lr=1e-2, epochs=1, n_minibatches=1),
        n_envs=n_envs, updates_per_env=updates,
        batch_envs=2, steps_per_rollout=6, eval_max_steps=6,
    )


def _drive(agent, method, n_envs=2, updates=2):
    cfg = _tiny_cfg(n_envs, updates)
    envs = [GridEnv(size=SIZE, observation_size=OBS, seed=s)
            for s in range(n_envs)]
    seen: list[dict] = []
    run_sequential_blocks(
        cfg=cfg, agent=agent,
        optimizer=torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr),
        envs=envs, device=torch.device("cpu"), n_eval_trials=1,
        on_update=lambda u: seen.append(u.losses), method=method)
    return seen


def test_driver_runs_the_hypernetwork_end_to_end_and_applies_the_penalty():
    torch.manual_seed(0)
    seen = _drive(_hyper(n_tasks=2), build_method("hnet", beta=1.0))
    assert "penalty" not in seen[0], "penalised before any boundary"
    assert "penalty" in seen[-1], "no penalty after the first boundary"


def test_driver_refuses_to_replay_into_a_task_conditioned_policy():
    """One forward runs under one task's parameters, so replayed trajectories
    from other blocks would be trained through this block's head -- destroying
    the isolation while still producing a plausible curve."""
    with pytest.raises(RuntimeError, match="task-conditioned"):
        _drive(_multi(n_tasks=2),
               build_method("er", replay_batches=1, buffer_size=float("inf")))


def test_driver_leaves_the_agent_on_the_training_task_each_update():
    """Evaluation walks every env seen so far and sets the task as it goes, so
    the next rollout would otherwise be collected under whichever env was
    evaluated last."""
    agent = _multi(n_tasks=2)
    tasks: list[int] = []
    orig = agent.set_task

    def spy(t):
        tasks.append(t)
        orig(t)
    agent.set_task = spy
    _drive(agent, None, n_envs=2, updates=1)
    # Block 1: task set to 1 for the rollout, then 0 and 1 for the two evals.
    assert tasks[-3:] == [1, 0, 1]


def test_driver_runs_every_architecture():
    """A smoke test with teeth: each of these has to survive rollout,
    minibatched BC and multi-env evaluation, and all three call the policy in a
    different shape."""
    for agent in (_multi(n_tasks=2), _xdg(n_tasks=2), _hyper(n_tasks=2)):
        seen = _drive(agent, None, n_envs=2, updates=1)
        assert len(seen) == 2, type(agent).__name__
        assert all(np.isfinite(s["move_loss"]) for s in seen)


# ===========================================================================
# the collapse detector
# ===========================================================================

def test_task_divergence_is_zero_when_the_generator_is_silenced():
    """The failure it exists to detect: if the generator emits nothing, every
    task shares the base and the arm is the naive baseline in disguise."""
    hyp = _hyper(base="frozen")
    with torch.no_grad():
        hyp.hyper.head.weight.zero_()
        hyp.hyper.head.bias.zero_()
    d = hyp.task_divergence()
    assert d["pairwise_divergence"] == pytest.approx(0.0, abs=1e-9)
    assert d["hyper_norm"] == pytest.approx(0.0, abs=1e-9)


def test_task_divergence_grows_with_the_generator_output():
    hyp = _hyper(base="frozen")
    small = hyp.task_divergence()["pairwise_divergence"]
    with torch.no_grad():
        hyp.hyper.head.weight.mul_(50.0)
        hyp.hyper.head.bias.mul_(50.0)
    assert hyp.task_divergence()["pairwise_divergence"] > small


def test_task_divergence_reports_the_conditioned_fraction_only_with_a_base():
    assert "conditioned_frac" in _hyper(base="learned").task_divergence()
    assert "conditioned_frac" in _hyper(base="frozen").task_divergence()
    assert "conditioned_frac" not in _hyper(base="none").task_divergence()


def test_describe_carries_the_divergence_so_every_history_does():
    """Recorded per run rather than investigated separately, because a
    collapsed generator produces an entirely ordinary-looking history."""
    d = _hyper().describe()
    assert "pairwise_divergence" in d and "trainable_params" in d
