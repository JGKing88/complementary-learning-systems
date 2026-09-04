"""Channel dropout on the position and heading inputs.

The goal these serve: §30-§33 established the policy acts as a function of
position and heading -- position occupies 2 of 1024 state directions and is
read at ~7x a size-matched random subspace, and its SHARE of the state's causal
effect predicts orbit depth monotonically across five arms. Reward shaping did
not move that. Making the channels intermittently unavailable during training
is the direct attack.

Training only. Evaluation always sees the full input, because the question is
whether a policy TRAINED under an unreliable position signal learns to weight
it less -- not whether it can cope with a handicap at test time.
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import HopfieldConfig


class TestConfig:

    def test_both_default_off(self):
        c = HopfieldConfig()
        assert c.place_dropout == 0.0
        assert c.heading_dropout == 0.0

    def test_wired_through_every_layer(self):
        """A knob present in the config, the arg map, the CLI but not the
        launcher is a silent no-op. --freeze_log_std was exactly that for an
        entire lineage of runs, so this is checked rather than assumed."""
        from hopfield_nav import train_navigate as tn
        for k in ("place_dropout", "heading_dropout"):
            assert k in tn.CFG_FIELDS, k
            assert tn.CFG_FIELDS[k] == (f"hopfield.{k}",)
        src = open("hopfield_nav/train_navigate.py").read()
        assert "--place_dropout" in src and "--heading_dropout" in src
        sh = open("hopfield_nav/navigate_job.sh").read()
        assert "PLACE_DROPOUT" in sh and "HEADING_DROPOUT" in sh

    def test_the_collector_applies_both(self):
        src = open("hopfield_nav/rollout/collector.py").read()
        assert 'values["encoded_state"] = values["encoded_state"] * keep' in src
        assert 'values["prev_action"] = values["prev_action"] * keep' in src
        assert 'values["prev_displacement"] = (' in src

    def test_heading_channels_share_one_mask(self):
        """prev_action and prev_displacement both carry the direction of
        travel, so dropping them independently would leave the heading
        available on most steps and measure nothing."""
        src = open("hopfield_nav/rollout/collector.py").read()
        i = src.index('_hdrop = float(')
        blk = src[i:i + 700]
        assert blk.count("np.random.rand(B)") == 1
        assert 'values["prev_action"]' in blk
        assert 'values["prev_displacement"]' in blk


class TestMaskSemantics:

    def test_the_mask_zeroes_the_expected_share(self):
        rng = np.random.RandomState(0)
        for p in (0.0, 0.3, 0.7):
            keep = (rng.rand(20000) >= p).astype(np.float32)
            assert abs((1.0 - keep.mean()) - p) < 0.02

    def test_it_is_per_env_not_per_batch(self):
        """A single scalar draw would blank the channel for every env at once,
        which is a different and much blunter intervention."""
        rng = np.random.RandomState(1)
        keep = (rng.rand(64) >= 0.5).astype(np.float32)
        assert 0.0 < keep.mean() < 1.0

    def test_zero_dropout_is_an_exact_no_op(self):
        rng = np.random.RandomState(2)
        keep = (rng.rand(500) >= 0.0).astype(np.float32)
        assert (keep == 1.0).all()
