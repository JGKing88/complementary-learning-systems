"""The trunk-step counter, and the two cost shapes it exists to tell apart.

The frontier says every method spends "200 gradient steps per environment",
which is true and misleading: the work inside one step ranges over 5x across
the suite. Deriving that from configs got it wrong twice in a row -- reading
`replay_batches` suggests CLEAR costs what ER costs, when CLEAR additionally
forwards every replayed batch again in `aux_loss`; and EWC looks free per
update while running one backward per stored trajectory at each boundary.

So these tests pin the two shapes rather than the absolute numbers:

  * replay is FLAT after a short ramp while its buffer fills, and
  * EWC is FLAT WITH STEPS at the block boundaries.

A regression that silently stopped counting replayed batches would leave both
curves flat and identical, which is exactly the reading the counter exists to
prevent.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from hopfield_nav.continual.cost import COUNTER, TrunkCounter

REPO = "/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite"
PY = sys.executable


# --------------------------------------------------------------------------
# the counter itself
# --------------------------------------------------------------------------

def test_counts_batch_times_time():
    c = TrunkCounter()
    c.add(torch.zeros(3, 40, 7), backward=True)
    assert c.fwd == 120 and c.bwd == 120


def test_no_grad_pass_is_forward_only():
    """A frozen teacher's forward costs a forward and no backward. Charging it
    to both would make LwF and CLEAR look twice as expensive as they are."""
    c = TrunkCounter()
    c.add(torch.zeros(2, 10, 3), backward=False)
    assert c.fwd == 20 and c.bwd == 0


def test_malformed_shapes_are_ignored_not_raised():
    """An instrument that can halt a 14-hour wave because it met an unexpected
    shape is worse than one that under-reports."""
    c = TrunkCounter()
    c.add(torch.zeros(5), backward=True)
    c.add(None, backward=True)
    c.add(object(), backward=True)
    assert c.fwd == 0 and c.bwd == 0


def test_reset_clears():
    c = TrunkCounter()
    c.add(torch.zeros(1, 2, 3), backward=True)
    c.reset()
    assert c.fwd == 0 and c.bwd == 0


def test_module_counter_is_the_shared_one():
    assert isinstance(COUNTER, TrunkCounter)


# --------------------------------------------------------------------------
# end to end, through a real (tiny) sequential run
# --------------------------------------------------------------------------

def _run(tmp_path, name, method, method_args, steps=20, iters=4, n_envs=3):
    out = tmp_path / f"{name}.json"
    cmd = [
        PY, "-m", "analysis.continual.baseline",
        "--out", str(out), "--run_name", name,
        "--n_envs", str(n_envs), "--iters_per_block", str(iters),
        "--max_steps", str(steps), "--size", "20", "--observation_size", "60",
        "--movement_mode", "discrete", "--goal_radius", "0.5",
        "--seed", "1", "--num_full_iters", "1",
        "--hidden_size", "32", "--num_rnn_layers", "1",
        "--lr", "1e-3", "--epochs", "1", "--n_minibatches", "1",
        "--batch_envs", "1", "--steps_per_rollout", str(steps),
        "--max_grad_norm", "1.0", "--device", "cpu",
        "--method", method,
    ]
    if method_args:
        cmd += ["--method_args", method_args]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.load(open(out))


def _deltas(hist):
    cost = hist["cost"]
    assert cost, "history carries no cost trace"
    return [b[1] - a[1] for a, b in zip(cost, cost[1:])]


@pytest.mark.slow
def test_history_carries_the_cost_trace(tmp_path):
    h = _run(tmp_path, "none", "none", None)
    assert len(h["cost"]) == len(h["trace"])
    steps = [c[0] for c in h["cost"]]
    assert steps == sorted(steps), "cost trace must be in update order"
    fwd = [c[1] for c in h["cost"]]
    assert fwd == sorted(fwd), "cumulative counts must be non-decreasing"


@pytest.mark.slow
def test_replay_ramps_then_goes_flat(tmp_path):
    """ER samples `min(replay_batches, len(buffer))`, so the per-update cost
    climbs while the buffer fills and is constant forever after."""
    h = _run(tmp_path, "er", "er",
             "buffer_size=inf,replay_batches=4,sampling=balanced",
             steps=20, iters=5, n_envs=3)
    d = _deltas(h)
    assert d[0] < d[1] < d[2], f"expected a ramp while the buffer fills: {d[:4]}"
    tail = d[4:]
    assert len(set(tail)) == 1, f"expected flat cost after the ramp, got {tail}"


@pytest.mark.slow
def test_ewc_is_flat_with_steps_at_boundaries(tmp_path):
    """The Fisher pass is one backward per stored trajectory, at `on_block_end`
    only -- so EWC's curve is flat within a block and jumps between them."""
    h = _run(tmp_path, "ewc", "online_ewc",
             "lam=1000,gamma=1.0,fisher=true,fisher_trajectories=4",
             steps=20, iters=4, n_envs=3)
    d = _deltas(h)
    base = min(d)
    spikes = [x for x in d if x > base]
    assert spikes, f"no boundary spike in EWC's cost trace: {d}"
    # One per boundary the run actually crossed.
    assert len(spikes) == len(h["blocks"]) - 1, f"{spikes} vs {h['blocks']}"
    # Each spike is the ordinary update plus 4 trajectories of 20 steps.
    for x in spikes:
        assert x == base + 4 * 20, f"spike {x} != {base} + 80"
