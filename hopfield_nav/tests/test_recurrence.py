"""The recurrence curve, the primary orbit diagnostic (P2 doc §28).

It replaces three statistics that each had a blind spot, and each blind spot
cost a wrong call in §18.6/§27. These tests pin that it sees what those missed:
a PRECESSING orbit (which cell-revisit lag cannot see) and a slow curl (which
`straightness` reads as straight), while not firing on a straight run.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.nav_tri.recurrence import (
    orbit_stats, recurrence_curve, summarise,
)


def _circle(period, radius=8.0, centre=(10.0, 10.0), n=200, drift=0.0):
    """A closed orbit, optionally precessing outward by `drift` per step."""
    t = np.arange(n)
    th = 2 * np.pi * t / period
    r = radius + drift * t
    return np.stack([centre[0] + r * np.cos(th),
                     centre[1] + r * np.sin(th)], axis=1)


def _straight(n=200, speed=0.95):
    x = np.clip(2.0 + speed * np.arange(n), 0, 19)
    return np.stack([x, np.full(n, 10.0)], axis=1)


class TestItSeesAnOrbit:

    def test_period_is_recovered(self):
        c = recurrence_curve(_circle(50))
        s = orbit_stats(c)
        assert s["orbits"]
        assert s["period"] == pytest.approx(50, abs=3)

    def test_depth_is_the_diameter_scale(self):
        """A radius-8 orbit puts the agent ~16 cells away at half a period and
        back at ~0 after a full one, so the dip is large."""
        s = orbit_stats(recurrence_curve(_circle(50, radius=8.0)))
        assert s["depth"] > 10.0

    def test_a_PRECESSING_orbit_is_still_found(self):
        """The failure that produced §27's false negative. The agent returns to
        the REGION but not the same snapped cell, so cell-revisit lag misses
        it; the recurrence curve must not."""
        s = orbit_stats(recurrence_curve(_circle(50, drift=0.02)))
        assert s["orbits"], s
        assert s["period"] == pytest.approx(50, abs=6)

    def test_a_slow_curl_is_found_though_it_reads_as_straight(self):
        """6.9 deg/step is below any sane 'is it turning' threshold and is
        exactly what `straightness` missed on p20_e_kcap."""
        period = 2 * np.pi / np.deg2rad(6.9)          # ~52 steps
        s = orbit_stats(recurrence_curve(_circle(period)))
        assert s["orbits"]
        assert s["period"] == pytest.approx(period, abs=6)


class TestItDoesNotFireOnNonOrbits:

    def test_a_straight_run_has_no_dip(self):
        s = orbit_stats(recurrence_curve(_straight()))
        assert not s["orbits"], s

    def test_a_random_walk_has_no_strong_dip(self):
        rng = np.random.RandomState(0)
        p = np.clip(np.cumsum(rng.uniform(-1, 1, size=(200, 2)), axis=0) + 10,
                    0, 19)
        s = orbit_stats(recurrence_curve(p))
        assert s["depth"] < 8.0, s


class TestSummary:

    def test_aggregates_over_trajectories(self):
        paths = [_circle(50, centre=(10 + i * 0.1, 10)) for i in range(6)]
        s = summarise(paths, "circles")
        assert s["n"] == 6
        assert s["n_orbiting"] == 6
        assert s["median_period"] == pytest.approx(50, abs=4)

    def test_mixed_population_is_split(self):
        paths = [_circle(50) for _ in range(4)] + [_straight() for _ in range(4)]
        s = summarise(paths, "mixed")
        assert s["n_orbiting"] == 4

    def test_short_paths_do_not_crash(self):
        s = summarise([_straight(n=8)], "tiny")
        assert s["n"] == 1
