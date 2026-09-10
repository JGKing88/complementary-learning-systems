"""Parsing and arithmetic behind the three-metric training curve.

The plot is not tested; the two things that would silently produce a wrong
CURVE are.

`path_optimality` divides by `mean_steps`, which `evaluation/metrics.py`
computes as steps_sum / total_successes -- a mean over successes only. It
reports 0.0, not NaN, when nothing succeeded, so a naive division yields inf
and draws a spike off the top of the axis exactly where the model is worst.

`first_reliable` decides which stretch of that curve is comparable at all. It
has to be the LAST crossing, not the first: a run that reaches 1.0, dips, and
recovers is only comparable from the recovery onward, and taking the first
crossing would certify the dip as trustworthy.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.nav_tri import training_curve as tc


class TestPathOptimality:

    def test_one_cell_per_step_is_one(self):
        """max_action_norm is 1.0, so covering (d - R) in (d - R) steps is the
        cap and must read exactly 1.0."""
        assert tc.path_optimality(9.0, start_dist=10.0, goal_radius=1.0) == 1.0

    def test_half_speed_is_one_half(self):
        assert tc.path_optimality(18.0, 10.0, 1.0) == 0.5

    def test_zero_successes_is_nan_not_inf(self):
        """metrics.py emits mean_steps 0.0 when total_successes is 0. Dividing
        by it would put an infinite spike on the plot precisely where the model
        has not learned anything yet."""
        assert np.isnan(tc.path_optimality(0.0, 10.0, 1.0))

    def test_negative_steps_is_nan(self):
        assert np.isnan(tc.path_optimality(-3.0, 10.0, 1.0))

    def test_nan_steps_propagates(self):
        assert np.isnan(tc.path_optimality(float("nan"), 10.0, 1.0))

    def test_start_inside_the_capture_ball_clips_at_zero(self):
        """start_dist below the radius would give a negative optimality, which
        is not a fraction. Clip rather than emit it."""
        assert tc.path_optimality(5.0, 0.5, 1.0) == 0.0

    def test_matches_the_probe_within_the_stated_tolerance(self):
        """The docstring claims agreement within 0.02 from u400 on. These are
        the measured pairs (exact per-episode path_efficiency vs this ratio of
        means) at d=10 on d0_base; if the arithmetic drifts, this catches it."""
        for steps, exact in ((14.00, 0.7029), (13.48, 0.7222),
                             (12.50, 0.7715), (12.14, 0.7933)):
            approx = tc.path_optimality(steps, 10.8714, 1.0)
            assert abs(approx - exact) <= 0.021, (steps, approx, exact)


class TestParse:

    def _log(self, tmp_path, body):
        p = tmp_path / "run.out"
        p.write_text(body)
        return str(p)

    def test_pairs_nav_and_expl_by_update(self, tmp_path):
        body = (
            "  [navigate_u25] nav={0: {'success_rate': 0.5, 'mean_steps': 40.0}}\n"
            "  [navigate_u25] expl={0: {'swept_coverage': 0.17}}\n"
            "  [navigate_u50] nav={0: {'success_rate': 0.9, 'mean_steps': 20.0}}\n"
            "  [navigate_u50] expl={0: {'swept_coverage': 0.24}}\n"
        )
        log = tc.parse_log(self._log(tmp_path, body))
        assert sorted(log) == [25, 50]
        assert log[50]["nav"][0]["success_rate"] == 0.9

    def test_an_update_missing_one_half_is_dropped(self, tmp_path):
        """A checkpoint evaluated on nav but not explore cannot contribute a
        point to a plot that shows both, and carrying it forward would silently
        misalign the two series."""
        body = (
            "  [navigate_u25] nav={0: {'success_rate': 0.5, 'mean_steps': 40.0}}\n"
            "  [navigate_u50] nav={0: {'success_rate': 0.9, 'mean_steps': 20.0}}\n"
            "  [navigate_u50] expl={0: {'swept_coverage': 0.24}}\n"
        )
        log = tc.parse_log(self._log(tmp_path, body))
        assert list(log) == [50]

    def test_updates_come_back_sorted(self, tmp_path):
        body = ""
        for u in (100, 25, 50):
            body += ("  [navigate_u%d] nav={0: {'success_rate': 1.0, "
                     "'mean_steps': 12.0}}\n" % u)
            body += "  [navigate_u%d] expl={0: {'swept_coverage': 0.5}}\n" % u
        log = tc.parse_log(self._log(tmp_path, body))
        assert list(log) == [25, 50, 100]

    def test_empty_log_gives_empty_dict(self, tmp_path):
        assert tc.parse_log(self._log(tmp_path, "nothing here\n")) == {}


class TestSeries:

    def _log(self, succ, steps, swept, nd=10):
        return {u: {"nav": {nd: {"success_rate": s, "mean_steps": k}},
                    "expl": {nd: {"swept_coverage": c}}}
                for u, s, k, c in zip(range(25, 25 + 25 * len(succ), 25),
                                      succ, steps, swept)}

    def test_uses_the_measured_constant_for_the_level(self):
        s = tc.series(self._log([1.0], [9.8714], [0.5]), 10, 1.0)
        assert s["start_dist"] == pytest.approx(10.8714)
        assert s["optimality"][0] == pytest.approx(1.0)

    def test_explicit_start_dist_overrides(self):
        s = tc.series(self._log([1.0], [9.0], [0.5]), 10, 1.0, start_dist=10.0)
        assert s["optimality"][0] == pytest.approx(1.0)

    def test_unknown_level_without_start_dist_raises(self):
        with pytest.raises(ValueError, match="start_dist"):
            tc.series(self._log([1.0], [12.0], [0.5], nd=7), 7, 1.0)

    def test_levels_absent_from_an_eval_are_skipped(self):
        log = self._log([1.0, 1.0], [12.0, 11.0], [0.5, 0.6])
        del log[50]["expl"][10]
        s = tc.series(log, 10, 1.0)
        assert list(s["u"]) == [25.0]


class TestFirstReliable:

    def _s(self, succ):
        return {"u": np.arange(25, 25 + 25 * len(succ), 25, dtype=float),
                "success": np.array(succ, float)}

    def test_finds_the_first_update_that_holds(self):
        assert tc.first_reliable(self._s([0.5, 0.9, 1.0, 1.0]), 0.99) == 75.0

    def test_takes_the_LAST_crossing_when_the_run_dips(self):
        """A run that hits 1.0, dips to 0.9 and recovers is comparable only
        from the recovery. Returning the first crossing would certify the dip."""
        assert tc.first_reliable(self._s([1.0, 0.9, 1.0, 1.0]), 0.99) == 75.0

    def test_none_when_it_never_holds(self):
        assert tc.first_reliable(self._s([0.5, 0.6, 0.7]), 0.99) is None

    def test_reliable_from_the_start_returns_the_first_update(self):
        assert tc.first_reliable(self._s([1.0, 1.0]), 0.99) == 25.0
