"""The speed-matched swept billiard reference, and the discrepancy it exposed.

`swept_billiard` exists because swept numbers from DIFFERENT WORLDS are not
comparable but their ratios against a common reference are -- which is not
hypothetical: `w6_pers` (phase 1) and `p20_e` (phase 2) were trained on
different encoders, so `explore_traj` refuses to roll them together at all
("does not share a world"). A billiard in an empty box does not depend on the
encoder, so dividing each model's swept by the billiard at ITS OWN realized
speed puts them on one axis.

Dividing is also the only honest way to read a swept number: §19.2 established
that swept coverage is monotone in speed, so a model that sweeps more because
it moves faster has not explored better.

**The discrepancy these tests pin.** This reference does not reproduce the
table in `hopfield_nav/evaluation/swept.py`'s docstring. The CELL column
disagrees too and the gap grows with speed, so it is a difference between two
billiard implementations rather than a bug in the swept reduction, and it is
PRE-EXISTING: `behavior_probe.strategy_efficiency` already divides by this same
billiard. Pinned here so it is documented rather than lurking.
"""
from __future__ import annotations

import pytest

from analysis.nav_tri.coverage_baselines import (
    billiard_cells_per_step, swept_billiard,
)

SIZE, STEPS, R = 20, 200, 1.0


class TestShape:

    def test_monotone_in_speed(self):
        """§19.2's headline: swept is monotone increasing in speed, where cell
        coverage is flat. That is the whole reason the metric changed."""
        got = [swept_billiard(s, SIZE, STEPS, R, trials=128)
               for s in (0.5, 1.0, 1.5, 2.0, 3.0)]
        assert got == sorted(got), got

    def test_cell_coverage_is_NOT_monotone_in_speed(self):
        """The contrast that motivated the metric change: a long stride sweeps
        the same corridor but lands on fewer cell CENTRES."""
        cell = [billiard_cells_per_step(s, SIZE, STEPS, trials=128) * STEPS
                / (SIZE * SIZE) for s in (1.0, 2.0, 3.0)]
        assert cell != sorted(cell), cell

    def test_bounded(self):
        for s in (0.5, 1.0, 3.0):
            v = swept_billiard(s, SIZE, STEPS, R, trials=64)
            assert 0.0 < v < 1.0

    def test_a_bigger_goal_is_easier_to_find(self):
        """Swept is defined against goal_radius and cannot be radius-free
        (§19.4). A bigger disc must sweep more."""
        small = swept_billiard(1.0, SIZE, STEPS, 0.5, trials=64)
        big = swept_billiard(1.0, SIZE, STEPS, 2.0, trials=64)
        assert big > small

    def test_more_steps_sweep_more(self):
        """Swept is CUMULATIVE, so the horizon is part of the number."""
        assert (swept_billiard(1.0, SIZE, 400, R, trials=64)
                > swept_billiard(1.0, SIZE, 100, R, trials=64))

    def test_cached(self):
        a = swept_billiard(1.0, SIZE, STEPS, R, trials=64)
        b = swept_billiard(1.0, SIZE, STEPS, R, trials=64)
        assert a == b


class TestTheKnownDiscrepancy:
    """`swept.py`'s docstring table vs this billiard. Systematic, not noise."""

    # speed -> (cell in swept.py, swept in swept.py)
    DOC = {0.50: (0.246, 0.391), 1.00: (0.383, 0.633),
           2.00: (0.384, 0.839), 3.00: (0.397, 0.881)}

    def test_the_cell_column_disagrees_too(self):
        """This is the load-bearing one. If only the SWEPT column disagreed,
        the reduction would be suspect. The cell column disagreeing says the
        two TRACKS differ, and the swept reduction is exonerated."""
        deltas = {}
        for s, (dc, _) in self.DOC.items():
            cell = (billiard_cells_per_step(s, SIZE, STEPS, trials=256)
                    * STEPS / (SIZE * SIZE))
            deltas[s] = cell - dc
        assert all(d < 0 for d in deltas.values()), deltas
        # and it grows with speed
        assert deltas[3.0] < deltas[0.5]

    def test_it_does_not_shrink_with_more_trials(self):
        """Checked at 64 / 256 / 1024 when this was found. Sampling noise
        would shrink; this does not, so it is a real implementation gap."""
        at_2 = [swept_billiard(2.0, SIZE, STEPS, R, trials=n)
                for n in (64, 256)]
        assert all(v < self.DOC[2.0][1] - 0.02 for v in at_2), at_2

    def test_strategy_efficiency_uses_THIS_billiard(self):
        """Why the discrepancy is pre-existing rather than introduced here:
        the probe's own efficiency already divides by this reference, so a
        fast policy's efficiency is inflated by the same gap."""
        src = open("analysis/nav_tri/behavior_probe.py").read()
        assert "billiard_cells_per_step" in src
        assert "strategy_efficiency" in src
