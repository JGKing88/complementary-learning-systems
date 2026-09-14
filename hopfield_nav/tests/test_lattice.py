"""Gates for the lattice-randomised grid code and its scripted reader (plan sec 4B, 5.12).

- B2-C1  `gbook_at(theta = 0, scale = 1)` equals the scaffold's smoothed book,
         gathered at the same global positions, exactly -- and the one-hot at
         fwhm 0. This is what pins the smoothing convention.
- B2-C2  the per-module centroid shift for a unit step equals `R_theta a / s`
         to 0.05 cells over 100 random (theta, scale, position).
- the CRT decode recovers the rotated displacement anywhere inside an arena.
- `ScriptedFrameAgent` recovers theta to 1 deg and the direction to 1 deg on
  a synthetic lifetime, and survives a clipped first step.
- the oracle channel appends last, `pair_inputs` still equals `build_rnn_input`.
- `EnvSet.with_lattice(0, 1)` is the identity; the collector and the lifetime
  evaluator on `gbook_table` at (0, 1) reproduce the `sgb` path bit for bit.
- `LatticeSampler` never draws inside the held-out band.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gridcode.codebook import gen_gbook_2d
from gridcode.lattice import (
    code_phases, crt_displacement, gbook_at, module_phases, rotation, wrapped_phase_diff)
from gridcode.smoothing import smooth_gbook
from hopfield_nav.config import EnvConfig, RNNAgentConfig, RNNTrainConfig
from hopfield_nav.evaluation.goal_pairs import EnvTensors, pair_inputs
from hopfield_nav.evaluation.scripted_frame import ScriptedFrameAgent
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim, rnn_input_layout
from hopfield_nav.rollout.rnn import build_rnn_input, table_gather
from hopfield_nav.training.goal_pairs_setup import (
    LatticeSampler, agent_cfg_for_mode, build_env_sets, parse_thetas)

LAM = [11, 12, 13]
NG = sum(l * l for l in LAM)
SIZE, OBS = 8, 12


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def test_b2_c1_gbook_at_reproduces_smooth_gbook():
    Npos = 60
    gb = gen_gbook_2d(LAM, NG, Npos)
    sgb = smooth_gbook(gb, LAM, 0.25)
    rng = np.random.RandomState(0)
    pos = rng.randint(0, Npos, size=(300, 2))
    ref = sgb[:, pos[:, 0], pos[:, 1]].T
    assert np.abs(ref - gbook_at(pos, LAM, 0.25)).max() < 1e-5
    ref0 = gb[:, pos[:, 0], pos[:, 1]].T
    assert np.abs(ref0 - gbook_at(pos, LAM, 0.0)).max() == 0.0


def test_gbook_at_is_a_real_code_at_every_lattice():
    rng = np.random.RandomState(1)
    pos = rng.uniform(0, 500, size=(50, 2))
    for th, s in [(0.3, 1.0), (2.0, 0.8), (-1.2, 1.3)]:
        code = gbook_at(pos, LAM, 0.25, th, s)
        assert code.shape == (50, NG) and code.dtype == np.float32
        # Every module's block peaks at ~1 (a bump on the torus), never empty.
        off = 0
        for lam in LAM:
            blk = code[:, off:off + lam * lam]
            assert blk.max(axis=1).min() > 0.8
            off += lam * lam
        # Phases recovered from the code are the phases put in.
        want = module_phases(pos, LAM, th, s)
        got = code_phases(code, LAM)
        err = np.abs(wrapped_phase_diff(got, want, np.array(LAM, dtype=float)[:, None]))
        assert err.max() < 1e-3


def test_b2_c2_centroid_shift_is_the_rotated_unit_step():
    rng = np.random.RandomState(2)
    worst = 0.0
    for _ in range(100):
        th = rng.uniform(0, 2 * np.pi); s = rng.uniform(0.7, 1.4)
        p = rng.uniform(0, 1000, size=2)
        a = np.array([0.0, 1.0])
        code = gbook_at(np.array([p, p + a]), LAM, 0.25, th, s)
        ph = code_phases(code, LAM)
        shift = wrapped_phase_diff(ph[1], ph[0], np.array(LAM, dtype=float)[:, None]).mean(axis=0)
        want = rotation(th) @ a / s
        worst = max(worst, float(np.abs(shift - want).max()))
    assert worst < 0.05, worst


def test_crt_decode_recovers_rotated_displacement():
    rng = np.random.RandomState(3)
    for _ in range(200):
        th = rng.uniform(0, 2 * np.pi); p = rng.uniform(0, 1000, size=2)
        d = rng.uniform(-19, 19, size=2)
        code = gbook_at(np.array([p, p + d]), LAM, 0.25, th, 1.0)
        ph = code_phases(code, LAM)
        dph = wrapped_phase_diff(ph[1], ph[0], np.array(LAM, dtype=float)[:, None])[None]
        dec = crt_displacement(dph, LAM, 20 * np.sqrt(2))[0]
        assert np.abs(dec - rotation(th) @ d).max() < 1e-3


# ---------------------------------------------------------------------------
# The scripted estimator
# ---------------------------------------------------------------------------

def _x(cfg, pos, goal, th, s=1.0, obs=OBS):
    """Grid-mode input rows for positions/goals (N, 2) under one lattice."""
    gp = gbook_at(pos, LAM, 0.25, th, s)
    gg = gbook_at(goal, LAM, 0.25, th, s)
    prev = np.zeros((len(pos), 2), dtype=np.float32) if cfg.input_prev_action else None
    return build_rnn_input(None, prev, None, gp, cfg, "cpu", goal_grid_state=gg)


@pytest.mark.parametrize("prev_action", [False, True])
def test_scripted_agent_recovers_frame_and_direction(prev_action):
    cfg = agent_cfg_for_mode("grid", "continuous", rnn_cell="mlp", input_prev_action=prev_action)
    agent = ScriptedFrameAgent(cfg, OBS, LAM, 20)
    rng = np.random.RandomState(4)
    B = 32
    th = rng.uniform(0.3, 6.0, size=B); s = np.where(rng.uniform(size=B) < 0.5, 1.0, 1.25)
    base = rng.uniform(100, 900, size=(B, 2))
    pos = base + rng.randint(0, 20, size=(B, 2))
    goal = base + rng.randint(0, 20, size=(B, 2))
    goal[(goal == pos).all(axis=1)] += [3, 0]
    agent.begin_lifetimes(B)
    # Step 0 and 1: the agent measures. Apply its action in the arena frame.
    for _ in range(2):
        x = torch.cat([_x(cfg, pos[b:b + 1], goal[b:b + 1], th[b], s[b]) for b in range(B)])
        out = agent.act(x)
        a = out["move_action"].numpy().astype(np.float64)
        assert np.allclose(np.abs(a).sum(axis=1), 1.0)     # axis moves only
        pos = pos + a
    # Step 2: navigate.
    x = torch.cat([_x(cfg, pos[b:b + 1], goal[b:b + 1], th[b], s[b]) for b in range(B)])
    a = agent.act(x)["move_action"].numpy()
    fr = agent.frame()
    assert fr["measured"].all()
    dth = np.degrees(np.abs(np.mod(fr["theta"] - th + np.pi, 2 * np.pi) - np.pi))
    assert dth.max() < 1.0, dth.max()
    assert np.abs(fr["scale"] - s).max() < 0.02
    d = goal - pos
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((a * u).sum(axis=1), -1, 1)))
    assert ang.max() < 1.0, ang.max()


def test_scripted_agent_retries_a_clipped_step():
    cfg = agent_cfg_for_mode("grid", "continuous", rnn_cell="mlp")
    agent = ScriptedFrameAgent(cfg, OBS, LAM, 20)
    th = 1.1
    pos = np.array([[500.0, 500.0]]); goal = np.array([[507.0, 512.0]])
    agent.begin_lifetimes(1)
    a0 = agent.act(_x(cfg, pos, goal, th))["move_action"].numpy()[0]
    assert np.allclose(a0, [1, 0])
    # The arena clips the step: same code again. The agent must flip.
    a1 = agent.act(_x(cfg, pos, goal, th))["move_action"].numpy()[0]
    assert np.allclose(a1, [-1, 0])
    pos = pos + a1
    a2 = agent.act(_x(cfg, pos, goal, th))["move_action"].numpy()[0]
    assert np.allclose(a2, [0, 1])
    pos = pos + a2
    a3 = agent.act(_x(cfg, pos, goal, th))["move_action"].numpy()[0]
    assert agent.frame()["measured"][0]
    d = (goal - pos)[0]; u = d / np.linalg.norm(d)
    assert np.degrees(np.arccos(np.clip(a3 @ u, -1, 1))) < 1.0


# ---------------------------------------------------------------------------
# The oracle channel and the world plumbing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import os
    os.environ.setdefault("CLS_RUNS", str(tmp_path_factory.mktemp("runs")))
    cfg = RNNTrainConfig(
        env=EnvConfig(size=SIZE, observation_size=OBS),
        agent=RNNAgentConfig(input_grid_state=True),
        n_envs=3, n_val_envs=2, env_generator=True, place_margin=4,
        goal_val_frac=0.25, region_val_frac=0.1, lambdas=[5, 6, 7], seed=3)
    train, heldout, same, split, vh, sgb = build_env_sets(
        cfg, np.random.RandomState(3), n_same=1, keep_field=True)
    return dict(cfg=cfg, train=train, heldout=heldout, split=split, vh=vh, sgb=sgb,
                cells=split.cell_sets())


def test_oracle_channel_appends_last_and_pair_inputs_agree(world):
    cfg = agent_cfg_for_mode("grid", "continuous", rnn_cell="mlp", input_lattice_oracle=True)
    layout = rnn_input_layout(cfg, OBS, world["vh"].Ng)
    assert layout[-1] == ("lattice_oracle", 2)
    t = world["train"].tensors[0]
    t45 = world["train"].with_lattice(np.radians(45)).tensors[0]
    p = np.array([3, 17]); g = np.array([40, 5])
    x = pair_inputs(t45, cfg, p, g)
    assert x.shape[1] == compute_rnn_input_dim(cfg, OBS, world["vh"].Ng)
    assert np.allclose(x[:, -2:], [np.cos(np.radians(45)), np.sin(np.radians(45))], atol=1e-6)
    y = build_rnn_input(None, None, None, t45.gbook[p], cfg, "cpu", goal_grid_state=t45.gbook[g],
                        lattice_oracle=np.tile(t45.lattice_oracle, (2, 1))).squeeze(1).numpy()
    assert np.allclose(x, y)
    with pytest.raises(KeyError):
        build_rnn_input(None, None, None, t.gbook[p], cfg, "cpu", goal_grid_state=t.gbook[g])


def test_with_lattice_zero_is_identity_and_rotation_changes_code(world):
    tr = world["train"]
    same = tr.with_lattice(0.0, 1.0)
    for a, b in zip(tr.tensors, same.tensors):
        assert np.abs(a.gbook - b.gbook).max() < 1e-5
        assert a.omni is b.omni
    rot = tr.with_lattice(np.radians(45))
    assert rot.name == "train@45"
    assert np.abs(rot.tensors[0].gbook - tr.tensors[0].gbook).max() > 0.5
    assert rot.tensors[0].theta == pytest.approx(np.radians(45))


def test_collector_and_evaluator_gbook_table_path_matches_sgb(world):
    from hopfield_nav.evaluation.lifetime import evaluate_lifetime_direction
    from hopfield_nav.rollout.rnn import collect_rollout_rnn
    from hopfield_nav.world.vec_env import make_vec
    tr, sgb, cells = world["train"], world["sgb"], world["cells"]
    env, off = tr.envs[0], tr.offsets[0]
    cfg = agent_cfg_for_mode("grid", "continuous", rnn_cell="gru", hidden_size=8,
                             input_prev_action=True)
    torch.manual_seed(0)
    agent = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS, world["vh"].Ng))
    table = tr.tensors[0].gbook
    outs = []
    for tab in (None, table):
        vec = make_vec(env, 4, "continuous")
        vec._rng = np.random.RandomState(9)
        vec.set_goal_pool(cells.goal_train); vec.reset_all()
        torch.manual_seed(1)
        r = collect_rollout_rnn(vec, agent, cfg, 3, "cpu", sgb=sgb, env_offset=off,
                                carry_across_episodes=True, episode_max_steps=10,
                                gbook_table=tab)
        outs.append(r.obs.numpy())
    assert np.allclose(outs[0], outs[1])
    assert np.allclose(table_gather(table, np.array([[2, 3]]), SIZE)[0], table[2 * SIZE + 3])
    res = []
    for tab in (None, table):
        torch.manual_seed(2)
        res.append(evaluate_lifetime_direction(
            env, agent, cells=cells, n_lifetimes=4, n_episodes=2, max_steps=6, device="cpu",
            sgb=sgb, env_offset=off, seed=5, gbook_table=tab))
    assert np.allclose(np.array(res[0]["table"], dtype=float), np.array(res[1]["table"], dtype=float),
                       equal_nan=True)
    assert len(res[0]["ep0_by_step"]) == 7


def test_lattice_sampler_respects_holdout_band():
    ls = LatticeSampler(np.random.RandomState(0), holdout_deg=15, scale_range=(0.7, 1.4))
    for _ in range(2000):
        th, s, _ = ls.draw()
        assert not ls.in_holdout(th)
        assert 0.7 <= s <= 1.4 and not (0.95 <= s <= 1.05)
    assert ls.in_holdout(np.radians(7)) and ls.in_holdout(np.radians(-7)) and not ls.in_holdout(np.radians(20))
    mix = LatticeSampler(np.random.RandomState(0), holdout_deg=15, mix_standard_frac=0.5)
    zeros = sum(1 for _ in range(400) if mix.draw()[:2] == (0.0, 1.0))
    assert 150 < zeros < 250
    assert parse_thetas("0,7,45") == pytest.approx([0.0, np.radians(7), np.radians(45)])


# ---------------------------------------------------------------------------
# Translation (the fix for the memorised-offset route, 2026-09-13)
# ---------------------------------------------------------------------------

def test_translation_moves_absolute_phases_but_not_differences():
    rng = np.random.RandomState(6)
    pos = rng.uniform(0, 1000, size=(40, 2))
    th = 0.9
    a = code_phases(gbook_at(pos, LAM, 0.25, th), LAM)
    b = code_phases(gbook_at(pos, LAM, 0.25, th, 1.0, shift=(123.4, 567.8)), LAM)
    lam = np.array(LAM, dtype=float)[:, None]
    # Absolute phases differ by the shift (mod lambda) in every module ...
    d = wrapped_phase_diff(b, a, lam)
    assert np.allclose(d, d[0:1], atol=1e-3)          # the same shift for every position
    for m, l in enumerate(LAM):
        assert abs(wrapped_phase_diff(d[0, m, 0], 123.4, l)) < 1e-3
        assert abs(wrapped_phase_diff(d[0, m, 1], 567.8, l)) < 1e-3
    # ... and pairwise differences are untouched.
    da = wrapped_phase_diff(a[1:], a[:-1], lam)
    db = wrapped_phase_diff(b[1:], b[:-1], lam)
    assert np.abs(da - db).max() < 1e-3


def test_scripted_agent_is_unaffected_by_translation():
    cfg = agent_cfg_for_mode("grid", "continuous", rnn_cell="mlp")
    agent = ScriptedFrameAgent(cfg, OBS, LAM, 20)
    th, shift = 2.2, (901.1, 44.4)
    pos = np.array([[300.0, 300.0]]); goal = np.array([[311.0, 295.0]])
    agent.begin_lifetimes(1)

    def x(p):
        gp = gbook_at(p, LAM, 0.25, th, 1.0, shift); gg = gbook_at(goal, LAM, 0.25, th, 1.0, shift)
        return build_rnn_input(None, None, None, gp, cfg, "cpu", goal_grid_state=gg)
    for _ in range(2):
        pos = pos + agent.act(x(pos))["move_action"].numpy()
    a = agent.act(x(pos))["move_action"].numpy()[0]
    assert abs(np.degrees(np.mod(agent.frame()["theta"][0] - th + np.pi, 2 * np.pi) - np.pi)) < 1.0
    d = (goal - pos)[0]; u = d / np.linalg.norm(d)
    assert np.degrees(np.arccos(np.clip(a @ u, -1, 1))) < 1.0


def test_lattice_sampler_translate_is_uniform_over_the_period():
    ls = LatticeSampler(np.random.RandomState(0), holdout_deg=15, translate=True, period=1716.0)
    draws = [ls.draw() for _ in range(500)]
    sh = np.array([d.shift for d in draws])
    assert sh.min() >= 0 and sh.max() < 1716 and sh.std() > 400
    assert all(not ls.in_holdout(d.theta) for d in draws)
    off = LatticeSampler(np.random.RandomState(0), holdout_deg=15, translate=False)
    assert off.draw().shift == (0.0, 0.0)


def test_table_gather_per_row_and_collector_on_per_row_tables(world):
    from hopfield_nav.rollout.rnn import collect_rollout_rnn
    from hopfield_nav.world.vec_env import make_vec
    tr, cells = world["train"], world["cells"]
    env, off = tr.envs[0], tr.offsets[0]
    B = 4
    ls = LatticeSampler(np.random.RandomState(0), holdout_deg=15, translate=True, period=210.0)
    lats = [ls.draw() for _ in range(B)]
    tables = np.stack([tr.lattice_gbook(0, l.theta, l.scale, l.shift) for l in lats])
    assert tables.shape == (B, SIZE * SIZE, world["vh"].Ng)
    pos = np.array([[1, 2], [3, 4], [5, 6], [7, 0]])
    got = table_gather(tables, pos, SIZE)
    for b in range(B):
        assert np.allclose(got[b], tables[b, pos[b, 0] * SIZE + pos[b, 1]])
    with pytest.raises(ValueError):
        table_gather(tables, pos[:2], SIZE)
    cfg = agent_cfg_for_mode("grid", "continuous", rnn_cell="gru", hidden_size=8)
    agent = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS, world["vh"].Ng))
    vec = make_vec(env, B, "continuous")
    vec.set_goal_pool(cells.goal_train); vec.reset_all()
    p0 = vec.positions().copy(); g0 = vec._goals.copy()
    r = collect_rollout_rnn(vec, agent, cfg, 1, "cpu", carry_across_episodes=True,
                            episode_max_steps=10, gbook_table=tables)
    Ng = world["vh"].Ng
    x = r.obs[:, 0].numpy()
    for b in range(B):
        assert np.allclose(x[b, :Ng], tables[b, p0[b, 0] * SIZE + p0[b, 1]], atol=1e-6)
        assert np.allclose(x[b, Ng:2 * Ng], tables[b, g0[b, 0] * SIZE + g0[b, 1]], atol=1e-6)


def test_encoded_recurrent_core_contracts():
    from hopfield_nav.policy.recurrent import EncodedRecurrentCore, build_recurrent_core
    cfg = RNNAgentConfig(rnn_cell="gru", hidden_size=16, num_rnn_layers=2,
                         input_encoder_layers=3, input_encoder_hidden=24)
    core = build_recurrent_core(cfg, 40)
    assert isinstance(core, EncodedRecurrentCore)
    assert core.input_size == 40 and core.hidden_size == 16 and core.num_layers == 2
    x = torch.randn(5, 7, 40)
    f, h = core(x, None)
    assert core.feature_size == 16 + 24
    assert f.shape == (5, 7, 40) and h.shape == (2, 5, 16)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in core.encoder.net)
    plain = build_recurrent_core(RNNAgentConfig(rnn_cell="gru", hidden_size=16, num_rnn_layers=2,
                                                input_encoder_layers=3, input_encoder_hidden=24,
                                                input_encoder_skip=False, input_encoder_norm=False), 40)
    assert plain.feature_size == 16 and plain(x, None)[0].shape == (5, 7, 16)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in plain.encoder.net)
    # T steps == T single steps, carrying the CORE's state.
    hs, outs = None, []
    for t in range(7):
        o, hs = core(x[:, t:t + 1], hs)
        outs.append(o)
    assert torch.allclose(torch.cat(outs, 1), f, atol=1e-5)
    # Off by default: a plain GRU.
    assert not isinstance(build_recurrent_core(RNNAgentConfig(rnn_cell="gru", hidden_size=16), 40),
                          EncodedRecurrentCore)
    # The agent runs on it end to end.
    agent = RNNAgent(cfg, 40)
    out = agent.act(x[:, :1], None)
    assert out["h_next"].shape == (2, 5, 16)


def test_encoder_bypass_and_detach():
    from hopfield_nav.policy.recurrent import EncodedRecurrentCore, build_recurrent_core
    cfg = RNNAgentConfig(rnn_cell="gru", hidden_size=16, num_rnn_layers=1, input_prev_action=True,
                         input_encoder_layers=2, input_encoder_hidden=24, input_encoder_bypass=2,
                         input_encoder_detach=True)
    core = build_recurrent_core(cfg, 42)
    assert isinstance(core, EncodedRecurrentCore)
    assert core.encoder.input_size == 40 and core.input_size == 42
    assert core.core.input_size == 24 + 2
    x = torch.randn(3, 5, 42, requires_grad=True)
    f, h = core(x, None)
    assert f.shape == (3, 5, 16 + 24)
    # The bypass columns reach the core but not the encoder: perturbing them
    # changes the recurrent half of the features and not the skip half.
    x2 = x.detach().clone(); x2[..., :2] += 1.0
    f2, _ = core(x2, None)
    assert not torch.allclose(f2[..., :16], f[..., :16])
    assert torch.allclose(f2[..., 16:], f[..., 16:])
    # Detach: the encoder gets gradient only through the skip half.
    f[..., :16].sum().backward(retain_graph=True)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in core.encoder.parameters())
    f[..., 16:].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.encoder.parameters())
