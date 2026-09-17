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
| task3_k2_h1024_nv | ~~22866165~~ (s42) / ~~22869638~~ (s43) | **2/2 seeds hit the polar fixed point** (s42 from u60, s43 from u20: approx_kl 0.74 at u10, then kappa 11.7 vs cap 12.2, sigma 0.027, KL = clip = 0). Not a seed artefact under this reward; parked |
| task3_k2_h128_nv | 22866166 | queued |
| task3_k1_h128_nv | 22866167 | queued |
| task3_k2_h1024_nv_c1 | 22866168 | queued |
| task1r_k2_h128_nv | ~~22866169~~ → 22869636 (mit_normal_gpu 6 h) | moved 00:20 |

- 2026-09-16 22:19 — wave 2 launched, seed 42, 12 h walls, ou_bcs_normal
  (8 of 8 slots with the three surviving wave-1 arms).
- 2026-09-16 22:58 — first wave-2 points at u100: **task3_k2_h128_nv train
  found 0.73** (wave-1 twin: 0.30 at u100), cov_first 0.15, revisits 0.82 at
  79 steps — the one-rule reward searches as it should. task3_k2_h1024_nv
  s42 hit the polar-head fixed point (std frozen at 0.045, approx_kl =
  clip_frac = 0.000 from u60, reward flat at 0.063/step) — cancelled and
  re-seeded as s43 (22866716). Wave-1 avoidance, now unambiguous:
  task3_k1_h1024 at u300 finds the goal on HELD-OUT arenas 49–52 % but on
  its own train arenas 20 %, with more coverage at touch on train
  (0.16–0.21) — it sweeps the arenas whose goal it knows without touching
  it. Wave-1 held-out at u300–u350 for the record: task3_k2_h128 found
  0.36–0.40, revisits 1.00 at 34 steps, follow_q 0.35; task3_k2_h1024 found
  0.38–0.44, revisits 1.00 at 53, follow_q 0.31; task3_k1_h1024 revisits
  0.72 at 96, follow_q 0.27, std 0.032 and falling.
- 2026-09-16 23:25 — wave-1 h1024 arms cancelled at u400 (22864188
  task3_k1_h1024: train found 0.16 vs held-out 0.53–0.56; 22864189
  task3_k2_h1024: train found 0.02, revisits 95 steps — avoidance complete
  and exploit decaying). Their slots go to the pending wave-2 arms. 22864187
  task3_k2_h128 kept as the wave-1 survivor (u400 held-out found 0.31–0.33,
  revisits 0.97 at 42 steps, follow_q 0.37). Wave-2 task3_k2_h128_nv at
  u150: train found 0.86 (rising), 3.4 post-store touches at 36 steps,
  revisits 1.00; held-out found 0.36–0.47, revisits 0.83.
- 2026-09-17 00:20 — wave 2 at u250–u450. **Exploit generalises, search
  does not.** task3_k2_h128_nv held-out: revisits 1.00 at 26–29 steps,
  post-store 20 steps/touch, follow_q 0.47–0.49 (best of any arm) — but
  held-out found slides 0.48 (u250) → 0.42 → 0.38 → 0.29 (u450) while train
  found is 0.90–0.95 with the first touch at ~46–50 steps and 9–10 %
  coverage: on fixed-goal arenas the search phase becomes a walk to the
  known goal, so sweeping stops improving. task3_k1_h128_nv (K=1) same
  shape at u250: train 0.97 at 42 steps / 8 % coverage; held-out found
  0.42, revisits 1.00 at 47, follow_q 0.31. The wave-1 survivor, forced to
  keep sweeping by its own avoidance, is the better held-out searcher
  (found 0.56–0.66 at u550–u700, revisits 0.96–1.00 at 35–42, follow_q
  0.35–0.42). The redraw control decides whether search transfers once the
  shortcut is removed: task1r_k2_h128_nv and the h1024_nv re-seed moved to
  mit_normal_gpu (6 h walls) as 22869636 / 22869638 because ou_bcs_normal's
  backlog was holding them; `_c1` (22866168) still queued there.
- 2026-09-17 00:10 — task3_k2_h1024_nv s43 (22869638) collapsed too: u10
  approx_kl 0.74 (one blow-up update; wave-1 h1024 arms sat at 0.02–0.05),
  u20 onward kappa 11.7 against the 12.2 cap, sigma 0.027, approx_kl =
  clip_frac = 0.000. 2/2 seeds → the one-rule reward's higher-variance
  returns (novelty and goal in the same steps) plus the 1024 trunk take a
  step the polar head cannot recover from. Cancelled; h1024 parked under
  `_nv` (the `_c1` arm, flat novelty, is the lower-variance test). h128 —
  the better trunk in §8 anyway — carries the line. task3_k2_h128_nv
  held-out u500: revisits 1.00 at **19 steps**, post-store 20 steps/touch,
  follow_q 0.52, found 0.54 / 0.33 (d 0 / 10). task1r_k2_h128_nv u100: train
  found 0.49 at 13 % coverage, held-out 0.25–0.34, revisits 0.38.
- 2026-09-17 00:25 — the two `nv` settings have complementary halves.
  Fixed goals (task3_k2_h128_nv, u600): held-out revisits 1.00 at 29 steps,
  follow_q 0.49, post-store 22 steps/touch — but held-out found 0.22–0.24
  (from 0.54 at u500; train 0.95 at 50 steps / 10 % coverage). Redrawn goal
  (task1r_k2_h128_nv, u350): held-out found 0.58 / 0.67 (d 0 / 10) and
  rising — but revisits 0.89 at 62 steps, 55 steps/touch, follow_q 0.09.
  Reading: without a position map there is nothing cheap to bootstrap
  q-following from, and with novelty on post-store and remaining-scaled
  (up to 3.0/cell late in a rollout) sweeping out-earns a 50-step beeline,
  so exploit stays slow. Flat novelty (`_c1`, 0.3/cell) makes a 20-step
  touch (0.1/step) beat late sweeping (≤ 0.3 × ~0.1 new cells/step) — the
  directness lever from plan §7, now aimed at the redraw arm.

## 5. Wave 3 — redraw + flat novelty

| arm | job | status |
|---|---|---|
| task1r_k2_h128_nv_c1 | 22872240 (mit_normal_gpu 6 h) | launched 00:27 |
| task3r_k2_h128_nv_c1 | 22872241 (ou_bcs 12 h) | queued 00:27 |
| task1r_k4_h128_nv_c1 | 22874710 (ou_bcs 12 h) | running (started ~03:15) |
| task1r_k4_h128_nv_c1 s43 | 22882203 (mit_normal_gpu 6 h) | launched 05:20 — replicate |

- 2026-09-17 04:40 — **K=4 densifies exploit under redraw.**
  task1r_k4_h128_nv_c1 held-out u500: found 0.60–0.66, revisits 0.98 at
  **34 steps**, 37 steps/touch, follow_q 0.26 — against the K=2 redraw arms
  at u500 (revisits 0.82–0.96 at 62–64, 52–55 steps/touch, follow_q
  0.10–0.13). task3r_k2_h128_nv_c1 u500: found 0.47–0.48, revisits 0.98
  at 44, 39 steps/touch, follow_q 0.29 — three arenas' post-store data
  helps too. task1r_k2_h128_nv finished (22869636, 4000 updates, 4 h 43):
  held-out found 0.50–0.60, revisits 0.93 at 64, 53 steps/touch, follow_q
  0.13 — search transferred, exploit never left the sparse-signal regime.
  task1r_k2_h128_nv_c1 at u2500: same shape (0.45–0.57 / 0.82 at 45 / 55 /
  0.13). Fixed-goal arms at u1300–u1500: task3_k2_h128_nv revisits 1.00
  at 22–27, 17 steps/touch, follow_q 0.54–0.70, found 0.29–0.35;
  task3_k1_h128_nv revisits 0.93 at 64, 21 steps/touch, follow_q 0.43,
  found 0.38–0.44; wave-1 survivor task3_k2_h128 u2000: found 0.59–0.63,
  revisits 1.00 at 40, 28 steps/touch, follow_q 0.36.
- 2026-09-17 05:20 — task1r_k4_h128_nv_c1 held-out u1000: found 0.57–0.61,
  revisits 1.00 at 30 steps, 33 steps/touch, follow_q 0.34 — both halves
  improving together, the first arm to do that. Seed 43 replicate launched
  (22882203, mit_normal_gpu 6 h).

`task3r_*` added to the launcher (3 arenas, goal redrawn per visit
sequence). h128 only (h1024 collapses under `_nv`).

- 2026-09-17 00:55 — `_c1` does not separate: task1r_k2_h128_nv_c1
  held-out u300 found 0.59–0.61, revisits 0.90 at 62 steps, 50 steps/touch,
  follow_q 0.09, against the cap-10 twin's 0.52–0.57 / 0.74 at 63 / 50 /
  0.01 at u300. Flat novelty is not what unlocks exploit under redraw.
  Reading: bootstrapping. The fixed-goal arms reach ~10 touches per rollout
  early through the position map, so q and reward co-occur densely and
  q-following is learned as the generalising shortcut; the redraw arm gets
  1–2 touches per rollout, so the q signal is sparse. The task-faithful way
  to densify it: more visits per memory (K=4 — three of four rollouts in a
  sequence start with the goal stored). `k4` added to the launcher
  (ENV_REPEATS 4); `task1r_k4_h128_nv_c1` replaces the low-priority h1024
  `_c1` arm (22866168 cancelled unstarted).
