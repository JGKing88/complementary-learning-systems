# Published run pages — source

These are the HTML sources for the charts published as Artifacts during phase 2.
They were authored in a session scratch directory that is deleted with the job,
so they live here to survive it. Nothing in the codebase imports them.

**`nav_p2_runs.html` is the one that matters.** It is a self-contained hub —
hash-routed index plus a detail view for each of the nine runs — and it
supersedes the nine standalone pages beside it. Every metric on it is computed
by a single `derive()` function, so the cards, the stat rows and the comparison
table cannot disagree with one another. That was a real failure in the
standalone pages: each had a hand-written stat row, and the two P12 pages
happened to omit the zero-distractor success card that every other page led
with, which made those runs look as though they never reached 1.000. They do —
96/96.

| file | run | published |
|---|---|---|
| `nav_p2_runs.html` | **all nine, one page** | [c7beee82](https://claude.ai/code/artifact/c7beee82-a27a-49d8-b44f-c750df466dd7) |
| `p10_pol_v1.html` | frozen speed, exploit | [3bc9ad4e](https://claude.ai/code/artifact/3bc9ad4e-0655-43ca-b870-0516f4487bdc) |
| `p10_pol.html` | learned speed, exploit | [388023ce](https://claude.ai/code/artifact/388023ce-a725-4253-a53b-c9979a77baf2) |
| `p10_e_pol.html` | learned speed, explore | [00bd7fd3](https://claude.ai/code/artifact/00bd7fd3-bb60-4e22-a968-c62822c5cdb3) |
| `p10_e_pol_v1.html` | frozen speed, explore | [8fd3ecf0](https://claude.ai/code/artifact/8fd3ecf0-c429-40d6-bde6-008ca25b5a40) |
| `p11_cur.html` | distractor curriculum | [4de8dfa7](https://claude.ai/code/artifact/4de8dfa7-9403-43c8-b4f9-b14669ae603e) |
| `p11_tp.html` | cheaper failure | [4dbbe6e9](https://claude.ai/code/artifact/4dbbe6e9-c8e3-41fb-9b38-36a1443bf420) |
| `p11_cur_tp.html` | both P11 treatments | [6c3a0503](https://claude.ai/code/artifact/6c3a0503-dba1-405a-a90c-d33c491ee5b2) |
| `p12_lo.html` | speed capped at 1.0 | [6b09232a](https://claude.ai/code/artifact/6b09232a-bcd2-4609-9c1d-97d9757d0f5a) |
| `p12_lo_curtp.html` | capped + both treatments | [835846df](https://claude.ai/code/artifact/835846df-d30d-46f7-b979-3fe41fdfff7e) |
| `memory_mechanism.html` | the recall-mechanism explainer (§5.3–5.9) | [30c2ddd6](https://claude.ai/code/artifact/30c2ddd6-691a-4e4b-a3d4-95ac0bf70f07) |
| `where_q_fails.html` | P1 readout failure map | — |

**The data is embedded, not read from disk.** Each page carries its own eval
series as literal arrays, extracted from the run logs under
`$CLS_RUNS/logs/nav_p2_<job>.out`. That makes them permanent and portable, and
it means a page cannot be refreshed by re-running it — editing a number means
editing the array. The job ids are on each page.

To republish after an edit, pass the artifact URL as `url` so it updates in
place rather than creating a second copy.

**All numbers are on the `recorded` split** — each run's own six validation
environments. Never trained on, but the set it was scored against at every eval,
not a fresh draw. See EXPERIMENTS_NAV_P2 §9.8.2.
