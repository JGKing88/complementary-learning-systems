"""One-sided persistence: reward smooth motion, stop paying not to turn."""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import AgentConfig, HopfieldConfig


class TestOneSidedPersistence:

    def test_off_by_default(self):
        assert HopfieldConfig().persistence_one_sided is False

    def test_the_clamp_keeps_forward_motion_and_drops_the_turn_penalty(self):
        """The whole point. Two-sided, a 180 deg reversal costs -bonus, so the
        swing across a wall turn is 2*bonus = 0.4 at bonus=0.2 -- more than the
        0.3 a fresh cell pays, which is why ploughing straight over covered
        ground beat the lawnmower turn."""
        cos = np.array([1.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32)
        two_sided = 0.2 * cos
        one_sided = 0.2 * np.maximum(cos, 0.0)
        assert np.allclose(one_sided[:3], two_sided[:3])   # forward unchanged
        assert (one_sided[3:] == 0.0).all()                # turns cost nothing
        assert two_sided[-1] == -0.2

    def test_it_flips_which_move_wins_at_the_wall(self):
        """novelty 0.3, persistence 0.2, scale 1 early in an episode."""
        straight_over_old = 0.0 + 0.2 * 1.0
        turn_onto_new_two_sided = 0.3 + 0.2 * (-1.0)
        turn_onto_new_one_sided = 0.3 + 0.2 * max(-1.0, 0.0)
        assert turn_onto_new_two_sided < straight_over_old   # the bug
        assert turn_onto_new_one_sided > straight_over_old   # the fix

    def test_the_flag_is_wired_all_the_way_through(self):
        """Config field, CFG_FIELDS entry, CLI flag and launcher passthrough --
        a knob that exists in only three of the four is a silent no-op, which
        this project has shipped before (--freeze_log_std did nothing for an
        entire lineage of runs)."""
        from hopfield_nav import train_navigate as tn
        assert "persistence_one_sided" in tn.CFG_FIELDS
        assert tn.CFG_FIELDS["persistence_one_sided"] == (
            "hopfield.persistence_one_sided",)
        src = open("hopfield_nav/train_navigate.py").read()
        assert '--persistence_one_sided' in src
        sh = open("hopfield_nav/navigate_job.sh").read()
        assert "PERSISTENCE_ONE_SIDED" in sh

    def test_the_collector_applies_the_clamp(self):
        src = open("hopfield_nav/rollout/collector.py").read()
        assert 'persistence_one_sided' in src
        assert 'np.maximum(cos_np_r, 0.0)' in src
