# Task-faithful navigation training — experiment log

Plan: `docs/TASK_FAITHFUL_PLAN.md`. Started 2026-09-16. Newest entries at the
bottom of each section; §0 is the running summary and is rewritten as
results land.

Conventions: every arm is `run_nav_p2.sh` variant `task*` (§3.6 of the plan),
200-step rollouts, batch 64, SE optimizer, polar head, goal 2.0, distractors
U[0,10] per trajectory. "steps" are mean steps to the goal; `follow_q` is the
held-out behavior-probe number (`--mode nav`); "CL" is the continual protocol
(`run_cl_ood.sh`, STOCHASTIC=1, 200 iterations per arena). Reference points
from the fixed-goal line (`EXPERIMENTS_SAMPLE_EFF.md` §8): `fix3_h128`
follow_q 0.83, exploit 1.25/1.29/1.35 (d = 0/5/10, steps ÷ shortest path);
`fix1_h128 s43 u3000` follow_q 0.89–0.91, exploit 1.14/1.14/1.19, CL revisits
3000/3000 at 11.6 steps; `d0_base u725` exploit 1.19/1.17/1.37, explore
0.949 / 0.926.

## 0. Summary

| arm | job | status | found_frac | steps_first | steps_per_reach | revisit steps | held-out follow_q | CL |
|---|---|---|---|---|---|---|---|---|
| task3_k1_h128 | 22864186 | queued | | | | | | |
| task3_k2_h128 | 22864187 | queued | | | | | | |
| task3_k1_h1024 | 22864188 | queued | | | | | | |
| task3_k2_h1024 | 22864189 | queued | | | | | | |
| task1_k2_h128 s43 | 22864192 | queued | | | | | | |
| task1r_k2_h128 | 22864190 | queued | | | | | | |

## 1. Protocol (as run)

Search → oracle store of the goal cell at the first touch (once per
trajectory, never off-goal, the store head never writes) → teleport → keep
going; memory and state reset per rollout; one Hopfield per trajectory with
U[0,10] outside-arena distractors; explore rewards (novelty 0.3 remaining-
scaled, ε 0.1→0 over 200 updates) until the store fires, exploit rewards
(goal 2.0, teleport) after; wall −0.1 / persistence +0.2 / time −0.05
throughout. `visits=K` keeps each trajectory's memory for K consecutive
rollouts of its env with the state reset each rollout. No
`input_goal_in_memory`, ever.

## 2. Implementation log

- 2026-09-16 — plan written and approved (Jack: 3 fixed-goal envs primary,
  revisits worth trying, never the oracle bit). Implementation started.
- 2026-09-16 — implemented (plan §3): `task` stage kind + `visits` key
  (`training/stages.py`); `TaskRegime` (`training/task.py`, B Hopfields per
  slot, own distractor draw each); collector `task_mode` (oracle write at the
  first touch only, head masked, novelty/revisit/ε pre-store only, flags
  carried via `store_fired_init` / `store_fired_final`, phase-split
  diagnostics `diag` / `diag_post`, `TaskTracker`); `rollout/task_stats.py`
  (merge + print fragment, unit-checked); composer visit sequences in
  `train_navigate.py` (`n_reps % visits == 0` enforced, refuses
  `input_goal_in_memory`); `evaluate_task` on the collector path, eval
  scope `task`; launcher family `task3_* / task1_* / task1r_*` with `k1/k2`
  and `h128/h1024`. Smoke: job 22863755 (`task3_k2_h128`, `task:6,visits=2`).

## 3. Wave 1

- 2026-09-16 — smoke `task3_k2_h128` `task:6,visits=2` (job 22863755):
  COMPLETED in 7 min; 192 trajectories/update, 12.2 s/update (h128), eval
  at scope `task` 61 s; untrained numbers as expected (found 1–4 %, a few
  step-0 touches from spawns inside the goal ball).
- 2026-09-16 — wave 1 launched, seed 42 (task1 seed 43 = interior goal
  (11,6)), `task:4000,visits=K`, ENV_REPEATS 2, batch 32 (task3) / 64
  (task1*), eval every 50, 12 h walls on ou_bcs_normal: 22864186
  task3_k1_h128, 22864187 task3_k2_h128, 22864188 task3_k1_h1024, 22864189
  task3_k2_h1024, 22864190 task1r_k2_h128, 22864192 task1_k2_h128. Expect
  ~15 h for h128 at 4000 updates → one `--continue_from` leg after the wall.
- 2026-09-16 22:15 — **wave-1 finding: the goal is a penalty.** Every arm
  shows reward per step rising monotonically while train `found_frac` stalls
  or falls after an early peak, with coverage-at-touch flat and post-store
  steps per touch not improving in the h128 arms. Clearest series,
  task3_k1_h1024 (per 20 updates): reward 0.09 → 0.33, found 0.33 / 0.67 /
  0.77 (u120) / 0.44 / 0.31 / 0.26 / 0.14 (u200), cov_first 0.15–0.17
  throughout, std 0.123 → 0.053. task3_k1_h128: found 0.61 (u120) → 0.11–0.33;
  task3_k2_h128: 0.61 → 0.26–0.45; task1r_k2_h128 (goal redrawn, nothing to
  memorise): 0.22–0.66 with post-store 50–90 steps per touch. Mechanism:
  novelty is remaining-scaled (0.3 × 400/remaining, cap 10 → up to 3.0 per
  new cell late in a rollout) against 2.0 per touch, and under the wave-1
  rule a touch switches novelty OFF for the rest of the rollout — so until
  exploit is ≤ ~10 steps per touch, finding the goal lowers a trajectory's
  return, and on a fixed-goal arena the agent can sweep around the goal it
  knows. Pure-explore eval (empty memory) confirms search itself is learning
  slowly rather than avoiding an unknown goal (task3_k1_h128 coverage 0.05 /
  0.10 / 0.08 / 0.10 at u50–u200; d0_base's explorer reaches 0.31).
  Held-out at u150–u200 for the record: task3_k1_h1024 found 31–43 %,
  revisits 83–88 % at 74–83 steps, follow_q 0.21–0.26; task3_k2_h1024 found
  35–39 %, revisits 97 % at 41 steps, follow_q 0.26 (my u100 "position map"
  read of it was premature — it generalises as well as K=1); h128 arms
  found 10–19 %, no q-following.
- 2026-09-16 22:18 — cancelled 22864186 task3_k1_h128, 22864190
  task1r_k2_h128, 22864192 task1_k2_h128 (pending) to free slots; kept
  22864187 task3_k2_h128, 22864188 task3_k1_h1024, 22864189 task3_k2_h1024
  running to the wall as the does-it-self-correct record (K=2 h1024 is the
  best exploiter on train: 8 touches at 18 steps, revisits 100 % at 20).

## 4. Wave 2 — novelty stays on after the store

One reward rule for the whole rollout (`--task_novelty_after_store`,
launcher lever `_nv`): novelty for every new cell whenever it happens, +2.0
and a teleport for every touch, wall / persistence / time throughout. A touch
never costs anything; the teleport lands the agent somewhere new, which pays
novelty too. `_c1` additionally flattens novelty (`NOVELTY_SCALE_CAP=1`, 0.3
per cell with no remaining-scaling) to test the directness concern — with
the cap at 10, late-rollout sweeping (up to 3.0 per cell) can still outpay a
beeline through visited cells (2.0 per ~10 steps).

| arm | job | status |
|---|---|---|
| task3_k2_h1024_nv | 22866165 | queued 22:19 |
| task3_k2_h128_nv | 22866166 | queued |
| task3_k1_h128_nv | 22866167 | queued |
| task3_k2_h1024_nv_c1 | 22866168 | queued |
| task1r_k2_h128_nv | 22866169 | queued |

- 2026-09-16 22:19 — wave 2 launched, seed 42, 12 h walls, ou_bcs_normal
  (8 of 8 slots with the three surviving wave-1 arms).
