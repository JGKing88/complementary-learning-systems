"""analysis/explore_first/series.py: the exploit criterion and explore retention.

The log parser has to read the trainer's own eval lines (with `nan`), the
criterion has to be the task line's (first crossing, at d=0), and retention
has to be a delta against the run's OWN u0 row -- absent u0, no delta.
"""
from __future__ import annotations

from analysis.explore_first import series as S

LOG = """\
=== navigate: task:4 (4 updates) ===
  [navigate_u0] samples={'episodes': 0, 'env_steps': 0}
  [navigate_u0] nav={0: {'success_rate': 0.1, 'mean_steps': 90.0}, 10: {'success_rate': 0.1, 'mean_steps': 95.0}}
  [navigate_u0] expl={0: {'swept_coverage': 0.600, 'mean_coverage': 0.5}, 10: {'swept_coverage': 0.580, 'mean_coverage': 0.48}}
  [navigate_u0] task={0: {'found_rate': 0.9, 'steps_first': 40.0, 'revisit_found': 0.3, 'revisit_steps_first': 80.0, 'cos_aq_post': 0.05}, 10: {'found_rate': 0.85, 'steps_first': 45.0, 'revisit_found': 0.3, 'revisit_steps_first': 85.0, 'cos_aq_post': 0.02}}
  [navigate_u0] eval_seconds=10.0 scope=task
  u1(task): mean_r=0.1
  [navigate_u25] samples={'episodes': 6400, 'env_steps': 1280000}
  [navigate_u25] expl={0: {'swept_coverage': 0.540, 'mean_coverage': 0.45}, 10: {'swept_coverage': 0.500, 'mean_coverage': 0.4}}
  [navigate_u25] task={0: {'found_rate': 0.88, 'steps_first': 42.0, 'revisit_found': 0.96, 'revisit_steps_first': 19.0, 'cos_aq_post': 0.81}, 10: {'found_rate': 0.8, 'steps_first': 50.0, 'revisit_found': 0.9, 'revisit_steps_first': 25.0, 'cos_aq_post': 0.7}}
  [navigate_u50] samples={'episodes': 12800, 'env_steps': 2560000}
  [navigate_u50] nav={0: {'success_rate': 1.0, 'mean_steps': 18.0}, 10: {'success_rate': 0.95, 'mean_steps': 19.0}}
  [navigate_u50] expl={0: {'swept_coverage': 0.300, 'mean_coverage': 0.25}, 10: {'swept_coverage': 0.290, 'mean_coverage': nan}}
  [navigate_u50] task={0: {'found_rate': 0.5, 'steps_first': nan, 'revisit_found': 1.0, 'revisit_steps_first': 15.0, 'cos_aq_post': 0.9}, 10: {'found_rate': 0.5, 'steps_first': 70.0, 'revisit_found': 1.0, 'revisit_steps_first': 18.0, 'cos_aq_post': 0.85}}
"""


def _write(tmp_path, text=LOG):
    p = tmp_path / "nav_p2_1.out"
    p.write_text(text)
    return str(p)


def test_parse_reads_every_kind_and_nan(tmp_path):
    s = S.parse_log(_write(tmp_path))
    assert sorted(s) == [0, 25, 50]
    assert s[0]["samples"]["episodes"] == 0
    assert s[25]["samples"]["episodes"] == 6400
    assert s[0]["nav"][0]["success_rate"] == 0.1
    assert s[50]["expl"][10]["mean_coverage"] is None      # nan -> None
    assert S._get(s[50], "task", 0, "steps_first") is None


def test_criterion_is_the_task_lines_at_d0(tmp_path):
    s = S.parse_log(_write(tmp_path))
    assert not S.meets_criterion(s[0])
    assert S.meets_criterion(s[25])                          # 0.96 @ 19.0, cos 0.81
    assert not S.meets_criterion(s[25], dist=10)             # 0.9 @ 25
    assert S.meets_criterion(s[50])


def test_first_crossing_not_last(tmp_path):
    s = S.parse_log(_write(tmp_path))
    assert S.first_criterion(s) == (25, 6400)
    s25 = dict(s); s25.pop(25)
    assert S.first_criterion(s25) == (50, 12800)
    s0 = {0: s[0]}
    assert S.first_criterion(s0) is None


def test_each_threshold_binds():
    row = {"task": {0: {"revisit_found": 0.96, "revisit_steps_first": 19.0, "cos_aq_post": 0.81}}}
    assert S.meets_criterion(row)
    for k, bad in (("revisit_found", 0.94), ("revisit_steps_first", 20.5), ("cos_aq_post", 0.79)):
        r = {"task": {0: dict(row["task"][0], **{k: bad})}}
        assert not S.meets_criterion(r), k


def test_retention_is_a_delta_against_u0(tmp_path):
    s = S.parse_log(_write(tmp_path))
    r = S.retention(s, 0)
    assert [u for u, *_ in r] == [0, 25, 50]
    u, v, d, f = r[2]
    assert (v, round(d, 3), round(f, 3)) == (0.3, -0.3, 0.5)
    assert S.worst_retention(s, 0) == 0.5
    assert round(S.worst_retention(s, 10), 3) == 0.5


def test_no_u0_means_no_delta(tmp_path):
    text = "\n".join(l for l in LOG.splitlines() if "navigate_u0" not in l) + "\n"
    s = S.parse_log(_write(tmp_path, text))
    assert 0 not in s
    assert all(d is None and f is None for _, _, d, f in S.retention(s, 0))
    assert S.worst_retention(s, 0) is None
    table = S.arm_table("x", s)
    assert "(+" not in table and "(-" not in table


def test_tables_render_and_mark_the_crossing(tmp_path):
    s = S.parse_log(_write(tmp_path))
    arm = S.arm_table("E0", s)
    assert "| 25 ✓ |" in arm and "| 0 |" in arm
    assert "0.300 (-0.300)" in arm
    summ = S.summary_table({"E0": s, "empty": {}})
    assert "| E0 | 3 (last u50) | 25 | 6400 |" in summ
    assert "0.600 → 0.300 (0.50)" in summ
    assert "| empty | 0 |" in summ


def test_last_occurrence_wins(tmp_path):
    text = LOG + "  [navigate_u50] task={0: {'found_rate': 0.7, 'steps_first': 1.0, 'revisit_found': 1.0, 'revisit_steps_first': 10.0, 'cos_aq_post': 0.95}}\n"
    s = S.parse_log(_write(tmp_path, text))
    assert s[50]["task"][0]["found_rate"] == 0.7


def test_path_optimality_and_exploit_columns(tmp_path):
    assert S.path_optimality(10.0, 11.0) == 1.0
    assert S.path_optimality(20.0, 11.0) == 0.5
    assert S.path_optimality(5.0, 0.5) == 0.0          # clipped at 0
    assert S.path_optimality(None, 11.0) is None
    assert S.path_optimality(10.0, None) is None
    s = S.parse_log(_write(tmp_path))
    arm = S.arm_table("E0", s, {0: 10.0, 10: 10.5})
    assert "| 0.10 / 0.10 | 0.10 / 0.10 |" in arm       # u0: sr 0.1, steps 90 / 95
    arm_blank = S.arm_table("E0", s)
    assert "| 0.10 / 0.10 | — / — |" in arm_blank
    summ = S.summary_table({"E0": s}, {0: 10.0, 10: 10.5})
    assert summ.splitlines()[-1].endswith("| 1.00 / 0.95 | 0.50 / 0.50 |")
