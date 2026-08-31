"""`analysis/continual/metrics.py` against histories whose answers are known.

Every number on the results page comes out of this module, so a subtle error
here does not produce a crash -- it produces a plausible figure that is wrong.
The histories below are hand-built so each scalar has one arithmetically
correct value, worked out in the test's own comment.

(Tests are exempt from the layering rules, which is why importing `analysis`
from under `hopfield_nav/tests/` is fine.)
"""
from __future__ import annotations

import math

import pytest

from analysis.continual import metrics as M


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _history(per_step, blocks, n_iters=1):
    """per_step: {step: {env: value_or_list}} -> a history dict."""
    trace = []
    for step in sorted(per_step):
        env_of_block = next(b[2] for b in blocks if b[0] <= step <= b[1])
        inner = {str(j): {"reached": v} for j, v in per_step[step].items()}
        trace.append([step, env_of_block, inner])
    return {"metadata": {"run_name": "t", "n_envs": len(blocks),
                         "num_full_iters": n_iters},
            "blocks": [list(b) for b in blocks], "trace": trace}


def _clean_forgetting_history():
    """Three envs, 10 updates each. Each env is solved during its own block and
    lost completely as soon as the stream moves on.

    With tail_frac=0.2 the window on block (1,10) starts at
    1 + int(0.8*9) = 8, so steps 8-10 are what the tail sees.

        p[0] = {0: 1.0}
        p[1] = {0: 0.0, 1: 1.0}
        p[2] = {0: 0.0, 1: 0.0, 2: 1.0}
    """
    blocks = [(1, 10, 0), (11, 20, 1), (21, 30, 2)]
    per = {}
    for s in range(1, 11):
        per[s] = {0: 1.0 if s >= 8 else 0.0}
    for s in range(11, 21):
        per[s] = {0: 0.0, 1: 1.0 if s >= 18 else 0.0}
    for s in range(21, 31):
        per[s] = {0: 0.0, 1: 0.0, 2: 1.0 if s >= 28 else 0.0}
    return _history(per, blocks)


# ---------------------------------------------------------------------------
# performance matrix
# ---------------------------------------------------------------------------

def test_performance_matrix_uses_the_block_tail():
    p = M.performance_matrix(_clean_forgetting_history())
    assert p[0] == {0: pytest.approx(1.0)}
    assert p[1] == {0: pytest.approx(0.0), 1: pytest.approx(1.0)}
    assert p[2] == {0: pytest.approx(0.0), 1: pytest.approx(0.0),
                    2: pytest.approx(1.0)}


def test_performance_matrix_ignores_steps_outside_the_window():
    """A run that is solved early in a block and lost by the end must not score
    on the early part -- the tail is the point."""
    blocks = [(1, 10, 0)]
    per = {s: {0: 1.0 if s <= 5 else 0.0} for s in range(1, 11)}
    p = M.performance_matrix(_history(per, blocks))
    assert p[0][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the scalars, against hand-computed values
# ---------------------------------------------------------------------------

def test_retained_average_excludes_the_current_env():
    """The headline number. Including the env being trained on would dilute
    exactly the difference the figure is about -- every method solves it."""
    p = M.performance_matrix(_clean_forgetting_history())
    assert M.retained_average(p) == pytest.approx(0.0)     # mean(0.0, 0.0)
    assert M.current_env_score(p) == pytest.approx(1.0)
    assert M.final_average(p) == pytest.approx(1 / 3)      # mean(0, 0, 1)


def test_forgetting_is_peak_minus_final():
    # env 0 peaked at 1.0 and ended at 0.0; env 1 the same. mean = 1.0
    p = M.performance_matrix(_clean_forgetting_history())
    assert M.forgetting(p) == pytest.approx(1.0)


def test_forgetting_is_floored_at_zero():
    """A env that *improved* after the stream left it contributes 0, not a
    negative -- that is backward transfer's job, not forgetting's."""
    blocks = [(1, 10, 0), (11, 20, 1)]
    per = {}
    for s in range(1, 11):
        per[s] = {0: 0.0}
    for s in range(11, 21):
        per[s] = {0: 1.0, 1: 1.0}
    p = M.performance_matrix(_history(per, blocks))
    assert M.forgetting(p) == pytest.approx(0.0)
    assert M.backward_transfer(p) == pytest.approx(1.0)    # 1.0 - 0.0


def test_backward_transfer_is_negative_under_forgetting():
    # env 0: 0.0 - 1.0 = -1;  env 1: 0.0 - 1.0 = -1;  mean = -1
    p = M.performance_matrix(_clean_forgetting_history())
    assert M.backward_transfer(p) == pytest.approx(-1.0)


def test_perfect_retention_gives_zero_forgetting():
    """The Hopfield agent's shape: everything stays solved."""
    blocks = [(1, 10, 0), (11, 20, 1), (21, 30, 2)]
    per = {}
    for s in range(1, 31):
        envs = [0] if s <= 10 else ([0, 1] if s <= 20 else [0, 1, 2])
        per[s] = {j: 1.0 for j in envs}
    p = M.performance_matrix(_history(per, blocks))
    assert M.retained_average(p) == pytest.approx(1.0)
    assert M.forgetting(p) == pytest.approx(0.0)
    assert M.backward_transfer(p) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# stability gap
# ---------------------------------------------------------------------------

def test_stability_gap_sees_a_transient_dip_that_final_forgetting_misses():
    """The whole reason it is measured separately: this history ends with env 0
    fully recovered, so `forgetting` is 0 -- but env 0 was destroyed for the
    first half of block 1 and a reader of the final number would never know."""
    blocks = [(1, 10, 0), (11, 20, 1)]
    per = {}
    for s in range(1, 11):
        per[s] = {0: 1.0}
    for s in range(11, 21):
        # collapse at the boundary, recover by the end of the block
        per[s] = {0: 0.0 if s <= 15 else 1.0, 1: 1.0}
    hist = _history(per, blocks)
    p = M.performance_matrix(hist)

    assert M.forgetting(p) == pytest.approx(0.0), \
        "final forgetting should be clean in this history"
    assert M.stability_gap(hist, window=5) == pytest.approx(1.0), \
        "but the transient collapse must still be reported"


def test_stability_gap_is_zero_when_nothing_dips():
    blocks = [(1, 10, 0), (11, 20, 1)]
    per = {}
    for s in range(1, 11):
        per[s] = {0: 1.0}
    for s in range(11, 21):
        per[s] = {0: 1.0, 1: 1.0}
    hist = _history(per, blocks)
    assert M.stability_gap(hist, window=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# episodes to criterion
# ---------------------------------------------------------------------------

def test_episodes_to_criterion_counts_from_the_block_start():
    """At batch_envs=1 an update is an episode, so this reads directly as
    episodes of experience -- the axis where the store's advantage is largest."""
    blocks = [(1, 20, 0)]
    # solved from update 6 onward; with smooth=3 the running mean crosses 0.9
    # at the third consecutive 1.0, i.e. update 8.
    per = {s: {0: 1.0 if s >= 6 else 0.0} for s in range(1, 21)}
    hist = _history(per, blocks)
    got = M.episodes_to_criterion(hist, threshold=0.9, smooth=3)
    assert got == pytest.approx(8.0)
    assert M.episodes_to_criterion_censored(hist, smooth=3) == pytest.approx(0.0)


def test_episodes_to_criterion_reports_censoring():
    """An env never solved returns the block length, which is a floor on the
    true value -- so the censored fraction has to travel with it or the mean
    reads as a real average when it is not."""
    blocks = [(1, 20, 0)]
    per = {s: {0: 0.0} for s in range(1, 21)}
    hist = _history(per, blocks)
    assert M.episodes_to_criterion(hist, smooth=3) == pytest.approx(20.0)
    assert M.episodes_to_criterion_censored(hist, smooth=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# schema handling
# ---------------------------------------------------------------------------

def test_both_history_schemas_agree():
    """`merge_histories` writes a scalar when num_full_iters == 1 and a list of
    per-iteration values otherwise. Both are in the recorded histories, so every
    reader has to cope with both -- and give the same answer."""
    blocks = [(1, 10, 0), (11, 20, 1)]
    per_scalar, per_list = {}, {}
    for s in range(1, 11):
        per_scalar[s] = {0: 1.0}
        per_list[s] = {0: [1.0, 1.0, 1.0]}
    for s in range(11, 21):
        per_scalar[s] = {0: 0.0, 1: 1.0}
        per_list[s] = {0: [0.0, 0.0, 0.0], 1: [1.0, 1.0, 1.0]}

    p_s = M.performance_matrix(_history(per_scalar, blocks))
    p_l = M.performance_matrix(_history(per_list, blocks, n_iters=3))
    assert p_s == p_l


def test_list_schema_averages_across_iterations():
    blocks = [(1, 10, 0)]
    per = {s: {0: [1.0, 0.0, 0.0, 0.0]} for s in range(1, 11)}
    p = M.performance_matrix(_history(per, blocks, n_iters=4))
    assert p[0][0] == pytest.approx(0.25)


def test_none_entries_are_skipped_not_counted_as_zero():
    """A missing iteration contributes None, and treating it as 0 would drag
    the mean down by exactly the fraction of runs that failed to record."""
    blocks = [(1, 10, 0)]
    per = {s: {0: [1.0, None, 1.0]} for s in range(1, 11)}
    p = M.performance_matrix(_history(per, blocks, n_iters=3))
    assert p[0][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# forward transfer
# ---------------------------------------------------------------------------

def test_forward_transfer_is_zero_against_itself():
    hist = _clean_forgetting_history()
    assert M.forward_transfer(hist, hist) == pytest.approx(0.0)


def test_forward_transfer_is_positive_when_learning_is_faster():
    """The metric that scores pretraining, and the one the existing figures
    never had -- so whether pretraining helps had never been measured."""
    blocks = [(1, 10, 0)]
    slow = _history({s: {0: 1.0 if s >= 9 else 0.0} for s in range(1, 11)}, blocks)
    fast = _history({s: {0: 1.0 if s >= 3 else 0.0} for s in range(1, 11)}, blocks)
    ft = M.forward_transfer(fast, slow)
    assert ft > 0.0
    assert M.forward_transfer(slow, fast) < 0.0


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------

def test_empty_history_returns_nan_not_a_crash():
    empty = {"metadata": {}, "blocks": [], "trace": []}
    p = M.performance_matrix(empty)
    assert p == {}
    assert math.isnan(M.retained_average(p))
    assert math.isnan(M.forgetting(p))
    assert math.isnan(M.stability_gap(empty))


def test_summarize_carries_the_cost_axes():
    hist = _clean_forgetting_history()
    hist["metadata"]["method"] = "er"
    hist["metadata"]["method_detail"] = {
        "state_bytes": 12345, "needs_task_boundaries": False,
        "needs_task_id": False,
    }
    s = M.summarize(hist)
    assert s["method"] == "er"
    assert s["state_bytes"] == 12345
    assert s["needs_task_boundaries"] is False
    assert s["retained"] == pytest.approx(0.0)
    assert s["current_env"] == pytest.approx(1.0)
