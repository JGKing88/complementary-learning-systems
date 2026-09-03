"""The state-usefulness probe: content, use, and the controls for both.

Following `test_exploit_diag`, the heavy rollout is not exercised here -- it is
`behavior_probe.rollout`, covered elsewhere. What is tested is the arithmetic
that turns a rollout into a claim, because that is where a diagnostic goes
quietly wrong: a probe that leaks across trials, a donor that draws from the
episode it is meant to contrast with, or an influence number normalised by
something that is not the action's natural spread.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.nav_tri import state_probe as sp


# ---------------------------------------------------------------------------
# The trial split -- the leak that would invent memory
# ---------------------------------------------------------------------------


class TestSplit:

    def test_no_trial_appears_on_both_sides(self):
        """Consecutive hidden states are near-identical, so a split on
        TIMESTEPS lets the probe memorise the test set. Splitting on trials is
        the whole reason the R^2 means anything."""
        rng = np.random.RandomState(0)
        trial = np.repeat(np.arange(20), 50)
        tr, te = sp._split_trials(trial, 0.3, rng)
        assert not (set(trial[tr]) & set(trial[te]))
        assert tr.sum() + te.sum() == len(trial)

    def test_holds_out_roughly_the_requested_fraction(self):
        rng = np.random.RandomState(0)
        trial = np.repeat(np.arange(20), 10)
        _, te = sp._split_trials(trial, 0.3, rng)
        assert len(np.unique(trial[te])) == 6

    def test_never_holds_out_everything(self):
        """A two-trial probe must still have something to fit on."""
        rng = np.random.RandomState(0)
        trial = np.repeat(np.arange(2), 10)
        tr, te = sp._split_trials(trial, 1.0, rng)
        assert tr.sum() > 0 and te.sum() > 0


# ---------------------------------------------------------------------------
# The ridge itself
# ---------------------------------------------------------------------------


class TestFitScore:

    def test_recovers_a_linear_map(self):
        rng = np.random.RandomState(0)
        X = rng.randn(400, 5)
        Y = X @ rng.randn(5, 2)
        r2 = sp._fit_score(X[:300], Y[:300], X[300:], Y[300:], 1e-2)
        assert r2.min() > 0.99

    def test_noise_scores_about_zero(self):
        """The floor has to be 0, not 'whatever a 128-dim fit gives you'."""
        rng = np.random.RandomState(1)
        X = rng.randn(600, 40)
        Y = rng.randn(600, 1)
        r2 = sp._fit_score(X[:400], Y[:400], X[400:], Y[400:], 1e3)
        assert r2.max() < 0.1

    def test_predicting_the_train_mean_scores_exactly_zero(self):
        """SStot is taken about the TRAIN mean, which makes "0 = learned
        nothing" true no matter what the test set looks like. About the TEST
        mean this same case scores about -25, because the test set is shifted
        away from what the probe was fit on -- a floor that moves with the
        split is not a floor."""
        X = np.zeros((100, 3))                 # no signal: pred is the mean
        Ytr = np.zeros((60, 1))
        Yte = (np.arange(40, dtype=float)[:, None] % 3) + 5.0
        r2 = sp._fit_score(X[:60], Ytr, X[60:], Yte, 1.0)
        assert r2[0] == pytest.approx(0.0)
        about_test_mean = 1.0 - (((Yte - 0.0) ** 2).sum()
                                 / ((Yte - Yte.mean()) ** 2).sum())
        assert about_test_mean < -20.0


# ---------------------------------------------------------------------------
# CONTENT
# ---------------------------------------------------------------------------


def _panel(n_trials=24, n_steps=20, seed=0):
    rng = np.random.RandomState(seed)
    trial = np.repeat(np.arange(n_trials), n_steps)
    obs = rng.randn(n_trials * n_steps, 6)
    return rng, trial, obs


class TestContent:

    def test_state_that_adds_nothing_scores_delta_zero(self):
        """h is a copy of obs, so it carries no information obs does not.
        deltaR^2 must be ~0 even though R^2(h) is ~1 -- which is exactly why
        the raw R^2(h) column cannot be the headline."""
        rng, trial, obs = _panel()
        target = obs[:, :1] * 2.0
        got = sp.content_probes(obs, obs.copy(), {"t": target}, trial, rng)
        assert got["t"]["h"] > 0.9          # decodable from h ...
        assert abs(got["t"]["delta"]) < 0.05  # ... and yet nothing is stored

    def test_state_carrying_something_new_scores_delta_positive(self):
        rng, trial, obs = _panel()
        secret = np.repeat(np.random.RandomState(7).randn(24), 20)[:, None]
        h = np.concatenate([obs, secret], axis=1)
        got = sp.content_probes(obs, h, {"t": secret}, trial, rng)
        assert got["t"]["obs"] < 0.2
        assert got["t"]["delta"] > 0.7

    def test_a_one_dim_target_is_accepted(self):
        rng, trial, obs = _panel()
        got = sp.content_probes(obs, obs.copy(), {"t": obs[:, 0]}, trial, rng)
        assert got["t"]["dim"] == 1

    def test_it_records_the_split_it_used(self):
        rng, trial, obs = _panel()
        got = sp.content_probes(obs, obs.copy(), {"t": obs[:, :1]}, trial, rng)
        assert got["_n_train_trials"] + got["_n_test_trials"] == 24
        assert got["_n_samples"] == len(trial)


# ---------------------------------------------------------------------------
# The donor
# ---------------------------------------------------------------------------


class TestDonor:

    def test_never_draws_from_the_same_trial(self):
        """A donor from the same episode is a near-copy of the state it
        replaces, so the splice would measure nothing and read as 'the state
        is ignored'."""
        rng = np.random.RandomState(0)
        trial = np.repeat(np.arange(10), 30)
        idx = sp._donor(trial, rng)
        assert (trial[idx] != trial).all()

    def test_same_step_donor_matches_the_step(self):
        rng = np.random.RandomState(0)
        trial = np.repeat(np.arange(10), 30)
        step = np.tile(np.arange(30), 10)
        idx = sp._donor(trial, rng, step=step)
        assert (step[idx] == step).all()
        assert (trial[idx] != trial).all()

    def test_a_lone_row_at_some_step_stays_a_no_op(self):
        """One trial reaching step 5 alone has nobody to swap with; it must
        stay put rather than borrow from another step index."""
        rng = np.random.RandomState(0)
        trial = np.array([0, 0, 1])
        step = np.array([0, 5, 0])
        idx = sp._donor(trial, rng, step=step)
        assert idx[1] == 1


# ---------------------------------------------------------------------------
# USE
# ---------------------------------------------------------------------------


class _ToyAgent:
    """action = w_obs . obs + w_h . h, so the two influences are dialable."""

    def __init__(self, obs_dim, h_dim, w_obs, w_h, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(obs_dim, 2, generator=g) * w_obs
        self.B = torch.randn(h_dim, 2, generator=g) * w_h

    def get_action_and_value(self, x, h, deterministic=True):
        obs = x.squeeze(1).double()
        # h is (layers, B, hidden); the probe's job is to hand it over in that
        # layout, and flattening it back here is what checks that it did.
        hh = h.permute(1, 0, 2).reshape(obs.shape[0], -1).double()
        a = obs @ self.A.double() + hh @ self.B.double()
        return {"move_action": a.unsqueeze(1)}


def _use(w_obs, w_h, n_trials=20, n_steps=25, h_dim=8, obs_dim=6, seed=0):
    rng = np.random.RandomState(seed)
    n = n_trials * n_steps
    obs = rng.randn(n, obs_dim)
    h = rng.randn(n, h_dim)
    trial = np.repeat(np.arange(n_trials), n_steps)
    step = np.tile(np.arange(n_steps), n_trials)
    agent = _ToyAgent(obs_dim, h_dim, w_obs, w_h)
    return sp.use_probes(agent, obs, h, trial, step, 1,
                         torch.device("cpu"), rng, lags=(1, 5))


class TestUse:

    def test_a_memoryless_policy_has_zero_state_influence(self):
        """The §22 prediction for the explore arms: the action is a function
        of the observation alone, so splicing the state must do nothing."""
        u = _use(w_obs=1.0, w_h=0.0)
        assert u["state_influence"] < 1e-6
        assert u["state_share"] < 1e-6
        assert u["obs_influence"] == pytest.approx(1.0, abs=0.15)

    def test_a_state_only_policy_has_zero_obs_influence(self):
        u = _use(w_obs=0.0, w_h=1.0)
        assert u["obs_influence"] < 1e-6
        assert u["state_share"] > 0.999

    def test_a_balanced_policy_lands_near_a_half(self):
        u = _use(w_obs=1.0, w_h=1.0, h_dim=6)
        assert 0.35 < u["state_share"] < 0.65

    def test_zeroing_the_state_moves_nothing_when_it_is_unused(self):
        u = _use(w_obs=1.0, w_h=0.0)
        assert u["zero_influence"] < 1e-6

    def test_the_lag_curve_is_reported_as_a_fraction_of_the_scale(self):
        u = _use(w_obs=1.0, w_h=1.0)
        assert set(u["lag_curve"]) == {"1", "5"}
        assert all(v >= 0.0 for v in u["lag_curve"].values())

    def test_an_unused_state_gives_a_flat_zero_lag_curve(self):
        u = _use(w_obs=1.0, w_h=0.0)
        assert max(u["lag_curve"].values()) < 1e-6

    def test_the_scale_is_the_both_swap_not_the_action_norm(self):
        """Normalising by |action| would make the influences depend on how fast
        the agent moves, which is not what is being asked."""
        u = _use(w_obs=1.0, w_h=1.0)
        assert u["state_influence"] == pytest.approx(
            u["d_state"] / u["d_both"], rel=1e-9)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def _rec(T=12, B=3, size=20, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.cumsum(rng.rand(T, B, 2), axis=0) + 2.0
    pos = np.clip(pos, 0, size - 1)
    return {"pos_f": pos,
            "cell": np.rint(pos).astype(int),
            "action": rng.randn(T, B, 2)}


class TestTargets:

    def test_start_pos_is_constant_within_a_trial(self):
        """It appears in no input channel and never changes, so decoding it
        from h is path integration and nothing else."""
        T, B = 12, 3
        tg = sp.build_targets(_rec(T, B), 20, 3.0)
        s = tg["start_pos"].reshape(T, B, 2)
        assert np.allclose(s, s[:1])

    def test_start_pos_is_the_first_position(self):
        rec = _rec()
        tg = sp.build_targets(rec, 20, 3.0)
        first = 2.0 * rec["pos_f"][0] / 19.0 - 1.0
        assert np.allclose(tg["start_pos"].reshape(12, 3, 2)[0], first)

    def test_coverage_is_non_decreasing(self):
        T, B = 12, 3
        tg = sp.build_targets(_rec(T, B), 20, 3.0)
        c = tg["coverage"].reshape(T, B)
        assert (np.diff(c, axis=0) >= 0).all()

    def test_coverage_counts_the_first_cell(self):
        tg = sp.build_targets(_rec(), 20, 3.0)
        assert tg["coverage"].reshape(12, 3)[0].min() == pytest.approx(
            1.0 / 400.0)

    def test_elapsed_runs_zero_to_one(self):
        T, B = 12, 3
        tg = sp.build_targets(_rec(T, B), 20, 3.0)
        e = tg["elapsed"].reshape(T, B)
        assert e[0].max() == 0.0
        assert e[-1].min() == pytest.approx(1.0)

    def test_visited_is_empty_at_the_first_step(self):
        """Read-before-mark: at t=0 nothing has been visited, so a non-zero
        row would mean the replay is off by one against the aux head's target
        and the two would not be the same quantity."""
        tg = sp.build_targets(_rec(), 20, 3.0)
        assert tg["visited8"].reshape(12, 3, 8)[0].sum() == 0.0

    def test_heading_is_the_previous_action_and_starts_at_zero(self):
        rec = _rec()
        tg = sp.build_targets(rec, 20, 3.0)
        h = tg["heading"].reshape(12, 3, 2)
        assert np.allclose(h[0], 0.0)
        want = rec["action"][0] / np.linalg.norm(
            rec["action"][0], axis=-1, keepdims=True)
        assert np.allclose(h[1], want)

    def test_every_block_is_flattened_the_same_way(self):
        T, B = 12, 3
        tg = sp.build_targets(_rec(T, B), 20, 3.0)
        assert all(v.shape[0] == T * B for v in tg.values())
        assert tg["visited8"].shape[1] == 8
        assert tg["pos"].shape[1] == 2


# ---------------------------------------------------------------------------
# The rollout hook
# ---------------------------------------------------------------------------


class TestRecordState:

    def test_rollout_takes_record_state_and_defaults_it_off(self):
        """Off by default because it is (T, B, obs+hidden) of extra memory that
        no other caller of `rollout` wants."""
        import inspect

        from analysis.nav_tri.behavior_probe import rollout
        p = inspect.signature(rollout).parameters["record_state"]
        assert p.default is False

    def test_content_h_defaults_to_the_state_the_action_read(self):
        """h_t, not h_{t+1}: the action at t is f(obs_t, h_t), so the default
        keeps CONTENT causally matched to USE. `out` exists only because an
        aux head reads `features` = h_{t+1}, and comparing a probe on one
        against a head on the other is comparing two quantities (§30.7)."""
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-m", "analysis.nav_tri.state_probe", "--help"],
            capture_output=True, text=True)
        assert "--content_h" in r.stdout
        assert "{in,out}" in r.stdout


# ---------------------------------------------------------------------------
# The cross-arm comparison, and the confound it exists to flag
# ---------------------------------------------------------------------------


def _arm(obs_r2, delta, infl=0.2, lag=None):
    return {"hidden": 8, "obs_dim": 4,
            "content": {"pos": {"obs": obs_r2, "h": 0.5,
                                "both": obs_r2 + delta, "delta": delta,
                                "dim": 2, "eff_n": 100},
                        "_n_samples": 100, "_n_train_trials": 7,
                        "_n_test_trials": 3},
            "use": {"state_influence": infl, "same_t_influence": infl,
                    "obs_influence": 0.9, "state_share": 0.2,
                    "shuffle_influence": 0.3, "state_vs_shuffle": 0.7,
                    "lag_curve": lag or {"1": 0.05, "20": 0.3}}}


def test_cross_report_renders_a_dump_missing_a_newer_metric():
    """Result JSONs on disk predate the shuffle null; rendering one must drop
    the row it cannot fill, not fail to render at all."""
    out = {"a/one/u700.pt": _arm(0.02, 0.1), "a/two/u700.pt": _arm(0.02, 0.1)}
    for v in out.values():
        del v["use"]["shuffle_influence"]
        del v["use"]["state_vs_shuffle"]
    assert sp.cross_report(out)["baseline_confounded"] == []


class TestCrossReport:

    def test_it_flags_a_target_whose_obs_baseline_moved(self):
        """The confound the real run exposed: R^2(obs) for position was 0.067
        on one arm and 0.728 on another with the SAME 74 channels, because the
        baseline tracks which states the agent visits. A deltaR2 gap there is
        headroom, not storage."""
        out = {"a/one/u700.pt": _arm(0.07, 0.59),
               "a/two/u700.pt": _arm(0.73, 0.06)}
        assert sp.cross_report(out)["baseline_confounded"] == ["pos"]

    def test_a_stable_baseline_is_not_flagged(self):
        out = {"a/one/u700.pt": _arm(0.02, 0.59),
               "a/two/u700.pt": _arm(0.05, 0.06)}
        assert sp.cross_report(out)["baseline_confounded"] == []

    def test_it_survives_arms_with_different_lag_grids(self):
        """Short episodes drop long lags, so the union has holes; the table
        must print them rather than raise."""
        out = {"a/one/u700.pt": _arm(0.02, 0.1, lag={"1": 0.05}),
               "a/two/u700.pt": _arm(0.02, 0.1, lag={"1": 0.06, "50": 0.2})}
        assert sp.cross_report(out)["baseline_confounded"] == []


class TestRidgeHoist:

    def test_it_matches_the_reference_fit(self):
        """`_Ridge` exists only to hoist the Gram matrix out of the alpha
        loop; if it ever stops agreeing with `_fit_score`, the readable
        definition is the one that is right."""
        rng = np.random.RandomState(3)
        X, Y = rng.randn(300, 12), rng.randn(300, 3)
        r = sp._Ridge(X[:200], X[200:])
        for a in (0.1, 10.0, 1e4):
            assert r.score(Y[:200], Y[200:], a) == pytest.approx(
                sp._fit_score(X[:200], Y[:200], X[200:], Y[200:], a))

    def test_constant_columns_do_not_blow_up(self):
        """A channel that never varies has sd 0; it must be passed through as
        a zero column rather than dividing by nothing."""
        rng = np.random.RandomState(4)
        X = np.concatenate([rng.randn(200, 3), np.ones((200, 1))], axis=1)
        r = sp._Ridge(X[:150], X[150:])
        assert np.isfinite(r.score(rng.randn(150, 1), rng.randn(50, 1),
                                   1.0)).all()


class TestClockBaseline:

    def test_a_pure_clock_target_adds_nothing_beyond_the_clock(self):
        """delta_clk exists because deltaR2 counts a state that is only a
        clock as content. Handed the clock, it must score that at zero."""
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        t = np.tile(np.arange(20, dtype=float) / 19.0, 24)[:, None]
        h = np.concatenate([obs, t], axis=1)          # h knows only the time
        got = sp.content_probes(obs, h, {"clockish": t * 3.0}, trial, rng,
                                clock=t)
        assert got["clockish"]["delta"] > 0.9         # ... looks like content
        assert abs(got["clockish"]["delta_clk"]) < 0.05   # ... and is not

    def test_spatial_content_survives_the_clock_baseline(self):
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        t = np.tile(np.arange(20, dtype=float) / 19.0, 24)[:, None]
        secret = rng.randn(24 * 20, 1)
        h = np.concatenate([obs, secret], axis=1)
        got = sp.content_probes(obs, h, {"spatial": secret}, trial, rng,
                                clock=t)
        assert got["spatial"]["delta_clk"] > 0.7

    def test_it_is_absent_when_no_clock_is_given(self):
        rng, trial, obs = _panel()
        got = sp.content_probes(obs, obs.copy(), {"t": obs[:, :1]}, trial, rng)
        assert "delta_clk" not in got["t"]
        assert "delta_anc" not in got["t"]


class TestAnchorBaseline:

    def test_a_past_that_is_a_function_of_the_present_scores_zero(self):
        """§22 established the policy is a deterministic vector field, and
        under a deterministic flow the past is recoverable from the present.
        So `pos_lag20` decodable from h is not memory until current position
        has been ruled out -- which is what this rung does."""
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        pos = rng.randn(24 * 20, 2)
        past = pos * 0.8                       # past = f(present), exactly
        t = np.tile(np.arange(20, dtype=float) / 19.0, 24)[:, None]
        h = np.concatenate([obs, pos], axis=1)   # h knows only where it IS
        got = sp.content_probes(obs, h, {"past": past}, trial, rng,
                                clock=t, anchor=pos)
        assert got["past"]["delta"] > 0.9          # ... looks like memory
        assert abs(got["past"]["delta_anc"]) < 0.05   # ... and is not

    def test_a_nonlinear_flow_defeats_the_linear_anchor_and_not_the_flow_rung(
            self):
        """The reason delta_flow exists. The backward flow of a deterministic
        field is SMOOTH BUT NONLINEAR, so a state that merely encodes position
        richly scores as trajectory memory against a linear position column.
        Here the past is a nonlinear function of the present and nothing is
        remembered: delta_anc must be fooled and delta_flow must not."""
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        n = 24 * 20
        pos = rng.uniform(-1, 1, (n, 2))
        past = np.stack([np.sin(3.0 * pos[:, 0]) * np.cos(2.0 * pos[:, 1]),
                         np.cos(4.0 * pos[:, 0] + pos[:, 1])], axis=1)
        t = np.tile(np.arange(20, dtype=float) / 19.0, 24)[:, None]
        env = np.zeros(n, dtype=int)
        # h holds position in a rich basis -- and no history whatsoever.
        h = np.concatenate([obs, sp._flow_basis(pos, env, seed=1)], axis=1)
        got = sp.content_probes(obs, h, {"past": past}, trial, rng,
                                clock=t, anchor=pos, env=env)
        assert got["past"]["delta_anc"] > 0.3      # the linear rung is fooled
        assert abs(got["past"]["delta_flow"]) < 0.1   # this one is not

    def test_genuine_history_survives_the_flow_rung(self):
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        n = 24 * 20
        pos = rng.uniform(-1, 1, (n, 2))
        past = rng.randn(n, 2)                  # unrelated to position
        t = np.tile(np.arange(20, dtype=float) / 19.0, 24)[:, None]
        env = np.zeros(n, dtype=int)
        h = np.concatenate([obs, pos, past], axis=1)
        got = sp.content_probes(obs, h, {"past": past}, trial, rng,
                                clock=t, anchor=pos, env=env)
        assert got["past"]["delta_flow"] > 0.7

    def test_the_flow_basis_keeps_environments_apart(self):
        """The walls differ per env, so the flow does too; one shared basis
        would let env A's geometry explain env B's trajectory."""
        pos = np.zeros((4, 2))
        z = sp._flow_basis(pos, np.array([0, 0, 1, 1]))
        half = z.shape[1] // 2
        assert not z[:2, half:].any()      # env 0 writes only its own block
        assert not z[2:, :half].any()

    def test_genuine_history_survives_the_anchor(self):
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        pos = rng.randn(24 * 20, 2)
        past = rng.randn(24 * 20, 2)           # unrelated to the present
        t = np.tile(np.arange(20, dtype=float) / 19.0, 24)[:, None]
        h = np.concatenate([obs, pos, past], axis=1)
        got = sp.content_probes(obs, h, {"past": past}, trial, rng,
                                clock=t, anchor=pos)
        assert got["past"]["delta_anc"] > 0.7


class TestNewTargets:

    def test_occupancy_is_a_map_that_fills_in(self):
        """The row `start_pos` could not carry: it varies every step, so it is
        scored on all T*B samples rather than on n_trials (§30.6)."""
        T, B = 12, 3
        tg = sp.build_targets(_rec(T, B), 20, 3.0)
        occ = tg["occupancy"].reshape(T, B, sp.GRID * sp.GRID)
        assert occ.shape[2] == 16
        assert (np.diff(occ, axis=0) >= 0).all()      # blocks never un-visit
        assert occ[0].sum(axis=1).max() == 1.0        # one block at the start

    def test_pos_lag_is_the_position_k_steps_back(self):
        rec = _rec(T=30, B=3)
        tg = sp.build_targets(rec, 20, 3.0)
        want = tg["pos"].reshape(30, 3, 2)
        for k in sp.POS_LAGS:
            got = tg[f"pos_lag{k}"].reshape(30, 3, 2)
            assert np.allclose(got[k:], want[:-k])

    def test_pos_lag_clamps_at_the_episode_start(self):
        """Before step k there is no t-k, and the start is the honest stand-in;
        it is 10% of rows at k=20 and the docstring says so."""
        tg = sp.build_targets(_rec(T=30, B=3), 20, 3.0)
        got = tg["pos_lag20"].reshape(30, 3, 2)
        assert np.allclose(got[:20], got[0])

    def test_every_new_block_flattens_like_the_rest(self):
        T, B = 12, 3
        tg = sp.build_targets(_rec(T, B), 20, 3.0)
        assert all(v.shape[0] == T * B for v in tg.values())


class _SubspaceAgent:
    """action = (h . d) * v, so ONLY the direction `d` of h reaches the action.

    The ground truth the targeted splice has to recover: swapping `d` must move
    the action and swapping anything orthogonal to it must not.
    """

    def __init__(self, d, obs_dim):
        self.d = torch.as_tensor(d, dtype=torch.float64)
        self.obs_dim = obs_dim

    def get_action_and_value(self, x, h, deterministic=True):
        hh = h.permute(1, 0, 2).reshape(x.shape[0], -1).double()
        a = (hh @ self.d).unsqueeze(-1) * torch.tensor([1.0, -1.0],
                                                       dtype=torch.float64)
        return {"move_action": a.unsqueeze(1)}


class TestShuffleNull:
    """The null the whole-state swap never had. At full rank a 'random
    subspace of the same rank' is the whole space, so the targeted splice's
    control degenerates; shuffling units is the version that does not."""

    def test_every_unit_keeps_its_exact_marginal(self):
        """Which is what makes it fair for a ReLU trunk: non-negativity and
        sparsity survive, where a Gaussian null would put half its mass
        somewhere the state can never be."""
        rng = np.random.RandomState(0)
        h = np.maximum(rng.randn(200, 12), 0.0)      # ReLU-like: >= 0, sparse
        s = sp._shuffle_units(h, rng)
        assert np.allclose(np.sort(h, axis=0), np.sort(s, axis=0))
        assert (s >= 0).all()
        assert (s == 0).sum() == (h == 0).sum()

    def test_it_destroys_cross_unit_structure(self):
        rng = np.random.RandomState(0)
        base = rng.randn(400, 1)
        h = np.hstack([base, base * 2.0])            # perfectly correlated
        s = sp._shuffle_units(h, rng)
        assert abs(np.corrcoef(h[:, 0], h[:, 1])[0, 1]) > 0.99
        assert abs(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) < 0.2

    def test_units_are_permuted_independently(self):
        rng = np.random.RandomState(0)
        h = np.arange(60, dtype=float).reshape(20, 3)
        s = sp._shuffle_units(h, rng)
        # a single row permutation would keep each row's values together
        rows_intact = sum(bool(np.isin(r, h).all()
                               and (h == r).all(axis=1).any()) for r in s)
        assert rows_intact < 20

    def test_it_is_reported_against_the_same_scale(self):
        u = _use(w_obs=1.0, w_h=1.0)
        assert u["shuffle_influence"] == pytest.approx(
            u["d_shuffle"] / u["d_both"], rel=1e-9)
        assert u["state_vs_shuffle"] == pytest.approx(
            u["d_state"] / u["d_shuffle"], rel=1e-9)

    def test_a_policy_ignoring_the_state_moves_for_neither(self):
        u = _use(w_obs=1.0, w_h=0.0)
        assert u["d_shuffle"] < 1e-6
        assert u["shuffle_influence"] < 1e-6


class TestTargetedSplice:

    def _setup(self, seed=0, H=64):
        # H must sit well above the rank or the random control is not a
        # control: a random line in 12 dims already captures 1/12 of any
        # direction, capping the ratio near 3.5. The real trunk is 1024.
        rng = np.random.RandomState(seed)
        n_trials, n_steps = 20, 25
        n = n_trials * n_steps
        h = rng.randn(n, H)
        d = np.zeros(H)
        d[0] = 1.0                       # the only direction the agent reads
        return (rng, rng.randn(n, 5), h, d,
                np.repeat(np.arange(n_trials), n_steps))

    def test_the_read_subspace_moves_the_action(self):
        rng, obs, h, d, trial = self._setup()
        agent = _SubspaceAgent(d, 5)
        got = sp.subspace_splice(agent, obs, h, trial, d[:, None].copy(),
                                 1, torch.device("cpu"), rng)
        assert got["ratio"] > 5.0
        assert got["frac_of_full"] == pytest.approx(1.0, abs=0.05)

    def test_an_orthogonal_subspace_does_not(self):
        """The control that makes the result mean something: a direction the
        agent provably ignores must score ~0 even though it is a real, equally
        large perturbation of h."""
        rng, obs, h, d, trial = self._setup()
        agent = _SubspaceAgent(d, 5)
        other = np.zeros((len(d), 1))
        other[3, 0] = 1.0
        got = sp.subspace_splice(agent, obs, h, trial, other, 1,
                                 torch.device("cpu"), rng)
        assert got["d_sub"] < 1e-9

    def test_the_splice_leaves_the_complement_untouched(self):
        rng = np.random.RandomState(0)
        h = rng.randn(50, 8)
        don = rng.permutation(50)
        B = np.linalg.qr(rng.randn(8, 2))[0]
        g = h - (h @ B) @ B.T + (h[don] @ B) @ B.T
        # orthogonal component identical, in-subspace component the donor's
        assert np.allclose(g - (g @ B) @ B.T, h - (h @ B) @ B.T)
        assert np.allclose(g @ B, h[don] @ B)


class TestMatchedDonor:
    """The control for the asymmetry the plain donor creates: splicing the
    POSITION subspace contradicts the held-fixed observation, and no other
    subspace does, because the observation carries no visitation signal."""

    def test_the_donor_stands_in_the_same_place(self):
        rng = np.random.RandomState(0)
        pos = np.repeat(np.arange(10, dtype=float), 20)[:, None]
        pos = np.hstack([pos, np.zeros_like(pos)])
        trial = np.tile(np.arange(20), 10)
        idx, unmatched = sp._donor_matched(trial, pos, rng)
        assert unmatched == 0
        assert np.allclose(pos[idx], pos)

    def test_the_donor_is_a_different_episode(self):
        rng = np.random.RandomState(0)
        pos = np.zeros((200, 2))
        trial = np.repeat(np.arange(10), 20)
        idx, _ = sp._donor_matched(trial, pos, rng)
        assert (trial[idx] != trial).all()

    def test_a_row_alone_in_its_cell_keeps_itself_and_is_counted(self):
        """Silently treating it as 'no effect' would push the ratio toward
        zero for reasons that have nothing to do with the policy."""
        rng = np.random.RandomState(0)
        pos = np.array([[0.0, 0.0], [0.0, 0.0], [9.0, 9.0]])
        trial = np.array([0, 1, 2])
        idx, unmatched = sp._donor_matched(trial, pos, rng)
        assert unmatched == 1
        assert idx[2] == 2

    def test_matching_heading_too_splits_the_buckets(self):
        """Position alone was not enough: two agents in one cell heading
        opposite ways are not comparable, and the observation encodes heading,
        so the mismatch reintroduces the contradiction the match removes."""
        rng = np.random.RandomState(0)
        pos = np.zeros((40, 2))                       # all in one cell
        head = np.zeros((40, 2))
        head[:20, 0] = 1.0                            # east
        head[20:, 0] = -1.0                           # west
        trial = np.arange(40) % 8
        idx, unmatched = sp._donor_matched(trial, pos, rng, heading=head)
        assert unmatched == 0
        assert (np.sign(head[idx][:, 0]) == np.sign(head[:, 0])).all()
        assert (trial[idx] != trial).all()

    def test_without_heading_the_buckets_ignore_it(self):
        rng = np.random.RandomState(0)
        pos = np.zeros((40, 2))
        head = np.zeros((40, 2))
        head[:20, 0], head[20:, 0] = 1.0, -1.0
        trial = np.arange(40) % 8
        idx, _ = sp._donor_matched(trial, pos, rng)
        assert not (np.sign(head[idx][:, 0]) == np.sign(head[:, 0])).all()

    def test_a_position_only_state_barely_moves_under_a_matched_donor(self):
        """The point of the control. An agent reading only position should be
        strongly disturbed by an arbitrary donor and hardly at all by one
        standing in the same place."""
        rng = np.random.RandomState(0)
        n, H = 600, 64
        trial = np.repeat(np.arange(30), 20)
        pos = np.stack([np.tile(np.arange(20, dtype=float), 30),
                        np.zeros(600)], axis=1)
        h = np.zeros((n, H))
        h[:, 0] = pos[:, 0]                    # unit 0 IS the position code
        h[:, 1:] = rng.randn(n, H - 1)
        obs = rng.randn(n, 5)
        d = np.zeros(H)
        d[0] = 1.0
        agent = _SubspaceAgent(d, 5)
        Q = np.eye(H)[:, :1]
        got = sp.subspace_splice(agent, obs, h, trial, Q, 1,
                                 torch.device("cpu"), rng, pos=pos)
        assert got["ratio"] > 3.0              # arbitrary donor: big effect
        assert got["d_sub_matched"] < 1e-9     # same place: none at all


class TestOrthAgainst:

    def test_it_removes_the_shared_directions(self):
        """A target subspace that is ENTIRELY position inherits position's
        causal punch; after residualising there is nothing left of it."""
        R = np.eye(6)[:, :2]
        Q = np.eye(6)[:, :2]                    # identical to R
        assert sp._orth_against(Q, R).shape[1] == 0

    def test_it_keeps_the_part_that_is_its_own(self):
        R = np.eye(6)[:, :1]
        Q = np.eye(6)[:, [0, 3]]                # one shared, one its own
        out = sp._orth_against(Q, R)
        assert out.shape[1] == 1
        assert abs(out[3, 0]) > 0.99
        assert abs(float(R[:, 0] @ out[:, 0])) < 1e-9

    def test_the_residual_is_still_orthonormal(self):
        rng = np.random.RandomState(0)
        R = np.linalg.qr(rng.randn(20, 2))[0]
        Q = np.linalg.qr(rng.randn(20, 5))[0]
        out = sp._orth_against(Q, R)
        assert np.allclose(out.T @ out, np.eye(out.shape[1]), atol=1e-8)

    def test_a_subspace_read_only_via_position_loses_its_effect(self):
        """End to end: the agent reads direction 0 only, and the 'target'
        subspace is direction 0 plus an ignored one. Raw splice looks causal;
        with position projected out it is not."""
        rng = np.random.RandomState(0)
        # H must be well above the rank or the control is not a control: a
        # random 2-plane in 12 dims already captures 1/6 of any direction, so
        # the ratio saturates near 2.4 there. The real trunk is 1024.
        n, H = 400, 64
        h, obs = rng.randn(n, H), rng.randn(n, 5)
        trial = np.repeat(np.arange(20), 20)
        d = np.zeros(H)
        d[0] = 1.0
        agent = _SubspaceAgent(d, 5)
        Q = np.eye(H)[:, [0, 5]]
        raw = sp.subspace_splice(agent, obs, h, trial, Q, 1,
                                 torch.device("cpu"), rng)
        res = sp.subspace_splice(agent, obs, h, trial,
                                 sp._orth_against(Q, d[:, None].copy()), 1,
                                 torch.device("cpu"), rng)
        assert raw["ratio"] > 3.0
        assert res["d_sub"] < 1e-9


class TestSizeControl:
    """`ratio` conflates 'the policy weights these directions' with 'these
    directions are bigger'. A readout subspace for something the trunk encodes
    strongly is high-variance; a random 2-plane in 1024 dims holds ~2/1024 of
    the variance. `ratio_sens` divides that out."""

    def test_a_high_variance_subspace_inflates_the_plain_ratio(self):
        rng = np.random.RandomState(0)
        n, H = 500, 64
        h = rng.randn(n, H) * 0.05
        h[:, 0] *= 200.0                      # one enormous direction
        obs = rng.randn(n, 5)
        trial = np.repeat(np.arange(25), 20)
        # the agent weights EVERY direction identically
        agent = _SubspaceAgent(np.ones(H) / np.sqrt(H), 5)
        got = sp.subspace_splice(agent, obs, h, trial, np.eye(H)[:, :1], 1,
                                 torch.device("cpu"), rng, n_rand=48)
        assert got["size_vs_random"] > 5.0    # the edit really is much bigger
        assert got["ratio"] > 3.0             # ... which the plain ratio reads
        # ... and with size divided out, an equally-weighted agent scores ~1
        assert got["ratio_sens"] == pytest.approx(1.0, abs=0.35)

    def test_a_genuinely_favoured_direction_survives_the_size_control(self):
        rng = np.random.RandomState(1)
        n, H = 500, 64
        h = rng.randn(n, H)                   # all directions equal variance
        obs = rng.randn(n, 5)
        trial = np.repeat(np.arange(25), 20)
        d = np.zeros(H)
        d[0] = 1.0                            # read direction 0 and nothing else
        agent = _SubspaceAgent(d, 5)
        got = sp.subspace_splice(agent, obs, h, trial, np.eye(H)[:, :1], 1,
                                 torch.device("cpu"), rng, n_rand=48)
        assert got["size_vs_random"] == pytest.approx(1.0, abs=0.25)
        assert got["ratio_sens"] > 3.0

    def test_the_sensitivity_ratio_is_the_stated_quotient(self):
        rng = np.random.RandomState(2)
        n, H = 300, 32
        h, obs = rng.randn(n, H), rng.randn(n, 5)
        trial = np.repeat(np.arange(15), 20)
        agent = _SubspaceAgent(np.ones(H) / np.sqrt(H), 5)
        g = sp.subspace_splice(agent, obs, h, trial, np.eye(H)[:, :2], 1,
                               torch.device("cpu"), rng)
        # mean of per-draw sensitivities, not a ratio of means
        assert g["ratio_sens"] > 0.0
        assert g["size_vs_random"] == pytest.approx(
            g["dh_sub"] / g["dh_rand"], rel=1e-9)


class TestReadoutSubspace:

    def test_it_finds_the_direction_that_codes_the_target(self):
        rng = np.random.RandomState(0)
        h = rng.randn(400, 10)
        Y = h[:, [2]] * 2.0                       # only unit 2 codes Y
        Q = sp._readout_subspace(h, Y, alpha=1e-3)
        assert Q.shape[1] == 1
        assert abs(Q[2, 0]) > 0.95                # ... and it is recovered

    def test_the_basis_is_orthonormal(self):
        rng = np.random.RandomState(1)
        h = rng.randn(400, 10)
        Q = sp._readout_subspace(h, rng.randn(400, 3))
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)

    def test_rank_never_exceeds_the_target_width(self):
        rng = np.random.RandomState(2)
        h = rng.randn(400, 32)
        assert sp._readout_subspace(h, rng.randn(400, 4)).shape[1] <= 4


class TestEffectiveN:

    def test_a_within_trial_constant_target_reports_trial_count(self):
        """`start_pos` has as many independent samples as there are trials --
        24, not 480 -- which is why a 0.000 there is weak evidence against a
        1024-unit state, not a proven null."""
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        const = np.repeat(np.arange(24, dtype=float), 20)[:, None]
        got = sp.content_probes(obs, obs.copy(), {"c": const}, trial, rng)
        assert got["c"]["eff_n"] == 24

    def test_a_varying_target_reports_every_step(self):
        rng, trial, obs = _panel(n_trials=24, n_steps=20)
        got = sp.content_probes(obs, obs.copy(), {"v": obs[:, :1]}, trial, rng)
        assert got["v"]["eff_n"] == 24 * 20
