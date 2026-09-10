"""The goal-PRESENT ||q||, and the live mask that makes it mean anything.

`_explore_stats` has recorded the goal-absent ||q|| since the regime work
started; `_nav_stats` recorded no magnitude at all. That asymmetry is why the
exploit side of the q_scale dose response was quoted from a design comment
rather than measured -- the number simply was not computed anywhere.

The arithmetic is one line, so what is worth testing is the part that is easy
to get wrong: nav episodes END ON ARRIVAL and the record is padded after that.
An unmasked mean over (T, B) would average the real magnitudes together with
whatever the padding holds, and would drift with the horizon rather than with
the model. Every case below is about the mask.
"""
from __future__ import annotations

import numpy as np

from analysis.nav_tri.behavior_probe import _nav_stats


def _rec(T=6, B=3, q=None, alive=None, stg=None):
    """A minimal rollout record with the keys `_nav_stats` reads."""
    if q is None:
        q = np.zeros((T, B, 2))
    if alive is None:
        alive = np.ones((T, B), dtype=bool)
    if stg is None:
        stg = np.full(B, T, dtype=float)
    return {
        "cell": np.zeros((T, B, 2), dtype=int),
        "action": np.ones((T, B, 2)) * 0.5,
        "pos_f": np.zeros((T, B, 2)),
        "alive": alive,
        "steps_to_goal": stg,
        "q": q,
        "final_pos_f": np.zeros((B, 2)),
        "sigma": None,
        "mu_norm": None,
    }


def _run(rec, size=20, goal=(5, 5), starts=None):
    B = rec["alive"].shape[1]
    if starts is None:
        starts = np.zeros((B, 2))
    return _nav_stats(rec, size, goal, starts)


class TestGoalPresentMagnitude:

    def test_reports_the_mean_norm_when_every_step_is_live(self):
        T, B = 6, 3
        q = np.zeros((T, B, 2))
        q[..., 0] = 0.3                       # every ||q|| is exactly 0.3
        out = _run(_rec(T, B, q=q))
        assert np.isclose(out["q_mag_mean"], 0.3)
        assert np.isclose(out["q_mag_median"], 0.3)

    def test_norm_is_euclidean_not_a_component(self):
        """A 3-4-5 vector: reading one component would give 3 or 4, not 5."""
        T, B = 4, 2
        q = np.zeros((T, B, 2))
        q[..., 0], q[..., 1] = 0.3, 0.4
        out = _run(_rec(T, B, q=q))
        assert np.isclose(out["q_mag_mean"], 0.5)

    def test_dead_steps_are_excluded(self):
        """THE case. Half the steps are post-arrival padding at ||q||=0; an
        unmasked mean would report 0.15 for a trajectory whose every real
        step carried 0.3."""
        T, B = 6, 2
        q = np.zeros((T, B, 2))
        q[:3, :, 0] = 0.3                     # live steps
        q[3:, :, 0] = 0.0                     # padding after arrival
        alive = np.zeros((T, B), dtype=bool)
        alive[:3] = True
        out = _run(_rec(T, B, q=q, alive=alive))
        assert np.isclose(out["q_mag_mean"], 0.3), "padding leaked into the mean"

    def test_padding_that_is_nonzero_also_stays_out(self):
        """The mask must be the alive flag, not a ||q||>0 test -- otherwise a
        record whose padding holds stale values silently contaminates it."""
        T, B = 6, 2
        q = np.zeros((T, B, 2))
        q[:3, :, 0] = 0.3
        q[3:, :, 0] = 9.0                     # stale, would dominate
        alive = np.zeros((T, B), dtype=bool)
        alive[:3] = True
        out = _run(_rec(T, B, q=q, alive=alive))
        assert np.isclose(out["q_mag_mean"], 0.3)

    def test_percentiles_bracket_the_mean(self):
        rng = np.random.RandomState(0)
        T, B = 20, 5
        q = rng.rand(T, B, 2) * 0.4
        out = _run(_rec(T, B, q=q))
        assert out["q_mag_p10"] <= out["q_mag_median"] <= out["q_mag_p90"]
        assert out["q_mag_p10"] <= out["q_mag_mean"] <= out["q_mag_p90"]

    def test_all_dead_gives_nan_not_zero(self):
        """Zero would read as 'the readout is empty', which is a claim. NaN
        reads as 'not measured', which is the truth."""
        T, B = 4, 2
        out = _run(_rec(T, B, alive=np.zeros((T, B), dtype=bool)))
        assert np.isnan(out["q_mag_mean"])

    def test_keys_match_the_explore_side_naming(self):
        """_explore_stats emits `q_mag_mean`; a differently-named nav key
        would defeat the point of being able to compare the two halves."""
        out = _run(_rec())
        for k in ("q_mag_mean", "q_mag_median", "q_mag_p10", "q_mag_p90"):
            assert k in out

    def test_does_not_disturb_the_existing_cosines(self):
        """The magnitude is additive: follow_q and q_accuracy are cosines and
        must be unchanged by adding a norm alongside them."""
        T, B = 5, 2
        q = np.zeros((T, B, 2))
        q[..., 0] = 2.0                       # same direction, big magnitude
        big = _run(_rec(T, B, q=q))
        q2 = q * 0.01                         # same direction, small magnitude
        small = _run(_rec(T, B, q=q2))
        assert np.isclose(big["follow_q"], small["follow_q"])
        assert np.isclose(big["q_accuracy"], small["q_accuracy"])
        assert big["q_mag_mean"] > small["q_mag_mean"] * 100 - 1e-9
