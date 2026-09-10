"""Position aliasing -- making position INSUFFICIENT so memory is necessary.

Every lever through §35 changed what the state contains and left what the
policy does with it alone: occupancy's absolute influence on the action sat at
~0.025 across five arms regardless of intervention. The reason memory stays
optional is that the arena is uniquely coded, so a position field is a complete
policy and history buys nothing. Aliasing removes that.

Unlike `place_dropout`, this applies at TRAINING AND EVALUATION alike -- it is
a property of the sensor, not a perturbation of it, and aliasing one but not
the other would score a policy on inputs it never trained against.
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import HopfieldConfig
from hopfield_nav.rollout.visited import alias_positions


class TestAliasPositions:

    def test_off_by_default(self):
        assert HopfieldConfig().alias_mod == 0

    def test_zero_and_negative_are_exact_no_ops(self):
        p = np.array([[3, 17], [11, 4]])
        assert alias_positions(p, 0) is p
        assert alias_positions(p, -1) is p

    def test_it_folds_the_arena(self):
        """size 20, mod 10: the four quadrants collapse onto one another."""
        got = alias_positions(np.array([[3, 4], [13, 4], [3, 14], [13, 14]]), 10)
        assert (got == np.array([3, 4])).all()

    def test_distinct_places_really_do_collide(self):
        """The whole point: no function of the aliased code can separate them,
        so only how the agent GOT there can."""
        a = alias_positions(np.array([[2, 7]]), 10)
        b = alias_positions(np.array([[12, 17]]), 10)
        assert (a == b).all()

    def test_within_a_quadrant_nothing_changes(self):
        p = np.array([[0, 0], [9, 9], [4, 6]])
        assert (alias_positions(p, 10) == p).all()

    def test_it_does_not_mutate_its_input(self):
        p = np.array([[13, 14]])
        alias_positions(p, 10)
        assert (p == np.array([[13, 14]])).all()


class TestWiring:

    def test_every_place_code_site_aliases(self):
        """Six call sites: two in the collector, two in the batched
        evaluators, two in behavior_probe. A site that forgets would train or
        score against an un-aliased world and silently break the comparison."""
        for path, n in (("hopfield_nav/rollout/collector.py", 2),
                        ("hopfield_nav/evaluation/batched.py", 2),
                        ("analysis/nav_tri/behavior_probe.py", 2)):
            src = open(path).read()
            assert src.count("get_encoded_state") == n, path
            assert src.count("alias_positions") == n, path

    def test_wired_through_every_layer(self):
        from hopfield_nav import train_navigate as tn
        assert tn.CFG_FIELDS["alias_mod"] == ("hopfield.alias_mod",)
        assert "--alias_mod" in open("hopfield_nav/train_navigate.py").read()
        assert "ALIAS_MOD" in open("hopfield_nav/navigate_job.sh").read()

    def test_it_is_not_a_training_only_perturbation(self):
        """place_dropout is training-only by design; aliasing must NOT be, or
        evaluation measures a policy on a world it never saw."""
        ev = open("hopfield_nav/evaluation/batched.py").read()
        assert "alias_positions" in ev
        assert "place_dropout" not in ev
