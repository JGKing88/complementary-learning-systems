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

Held-out numbers are the in-training task eval (6 unseen arenas × 16
sampled trials) at the update shown; probe / CL columns from §7.

| arm | job | status | held-out found | revisit steps | steps/touch | follow_q (eval) | probe × opt d0/5/10 | CL revisits |
|---|---|---|---|---|---|---|---|---|
| **task3r_k2_h128** (wave-1 rule, 3 redrawn, K=2) | 22883646 | running | 0.61 / 0.46 (u1000) | 14.6 | 12.7 | 0.89 | 1.40/1.38/1.46 (u1000) | 3000/3000 at 15.4, Δ +0.00 |
| **task1r_k4_h128** (wave-1 rule, 1 redrawn, K=4) | 22883645 | running | 0.52–0.54 (u1500) | 15.9 | 14.4 | 0.87 | 1.32/1.31/1.50 (u1500) | 2997/3000 at 15.9, Δ −0.01 |
| **task3r_k2_h128_nv_c1** (one rule, 3 redrawn) | 22872241 | running | 0.42–0.52 (u2500) | 12.4 | 12.5 | 0.91 | **1.17/1.20/1.29** (u2500) | – |
| task1r_k4_h128_nv_c1 (one rule, K=4) | 22874710 | done u4000 | 0.46–0.49 | 13.5 | 13.4 | 0.89 (after a u1150–u2000 trough) | – | – |
| task1r_k4_h128_nv_c1_g5 | 22883647 | running | 0.33 (u1500) | 16.3 | 16.0 | 0.80 | – | – |
| task3_k2_h128_nv (fixed goals) | 22866166 | running | 0.17–0.20 (u3000) | 25 | 15 | 0.66 | 2.44/2.25/2.67 (u2500, position map) | – |
| task3_k1_h128_nv (fixed goals) | 22866167 | running | 0.43–0.49 (u3000) | 37 | 17 | 0.56 | – | – |
| task3_k2_h128 (wave-1 rule, fixed goals) | 22864187 | timed out u3400 | 0.57 (u3000) | 42 | 24 | 0.20 | – | – |
| task1r_k2_h128_nv / _c1 | 22869636 / 22872240 | done / timed out | 0.50–0.60 | 64 | 53 | 0.13 | – | – |
| task3_k1_h1024, task3_k2_h1024 (wave-1 rule) | 22864188/89 | cancelled u400 | 0.53–0.56 / 0.49 | – | – | – | avoidance on train | – |
| task3_k2_h1024_nv (s42, s43) | 22866165 / 22869638 | cancelled | polar fixed point 2/2 | | | | | |
| task1_k2_h128 s43, task1r_k2_h128 (wave-1 rule) | 22864192 / 22864190 | cancelled (unstarted / u300) | | | | | | |
| d0_base u725 (reference) | 22133273 | – | – | 11.6 (CL) | – | – | 1.20/1.21/1.31 | 3000/3000 (§8) |

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
- 2026-09-17 05:50 — **the K=4 exploit decays after u1100: a reward tie.**
  Held-out (d 0) per 50 updates: follow_q 0.25 (u450) → 0.38 (u850) →
  0.37 (u1050) → 0.25 / 0.20 / 0.14 / 0.14 / 0.14 / 0.12 / 0.13 / 0.10
  (u1150–u1500); steps/touch 30 → 47; revisits 1.00 at 28 → 0.85 at 42;
  found rises to 0.74. Train side the same: post-store touches 3–4 at
  24–30 steps (u840–u1140) → 1.0 at 50 (u1290–u1490) while **reward per
  step is flat at 0.34–0.36** and std 0.021–0.024. At ~27 steps/touch a
  touch pays 2.0/27 ≈ 0.074 per step — about what flat novelty (0.3/cell)
  pays a freshly-teleported sweeper — so beelining and sweeping tie and the
  narrow policy drifts along the reward-neutral direction. The fixed-goal
  arms escape because their beeline is 14 steps (0.14/step ≫ novelty).
  task3r_k2_h128_nv_c1 u1000: found 0.54–0.59, revisits 1.00 at 28, 27
  steps/touch, follow_q 0.41 — watch for the same decay.

## 6. Wave 4 — break the post-store tie under redraw

Two task-faithful ways, both run:
- **Wave-1 rule under redraw** (novelty off after the store). The
  avoidance failure of wave 1 needs a *known* goal; redraw removes it.
  Visits 2..K are then pure exploit on the agent's own memory.
- **Goal reward 5.0** with the one-rule reward (`_g5`, launcher lever):
  0.19/step at 27 steps/touch, well above novelty.

| arm | job | status |
|---|---|---|
| task1r_k4_h128 (wave-1 rule, K=4) | 22883645 | queued 05:53 (ou_bcs 12 h) |
| task3r_k2_h128 (wave-1 rule, 3 redrawn arenas) | 22883646 | queued 05:53 |
| task1r_k4_h128_nv_c1_g5 | 22883647 | queued 05:53 |
| task3r_k2_h128 s43 (replicate) | 22897775 (mit_normal_gpu 6 h) | launched 11:05 |

- 2026-09-17 11:05 — task3r_k2_h128_nv_c1 s42 u3000: found 0.51 / 0.51,
  revisits 1.00 at **11.3**, 12.0 steps/touch, 10.3 touches, follow_q
  0.92. Its s43 twin at u2000 is still in the `nv` trough (54 steps/touch,
  follow_q 0.16) — the one-rule route's timing is seed-dependent; the
  wave-1-rule arms show no trough. Timed out at their walls (records
  complete): 22866166 task3_k2_h128_nv (~u3500), 22866167 task3_k1_h128_nv
  (~u3500), 22882203 K=4 nv s43 (~u3100, mid-recovery). Replicate of the
  winning recipe launched: task3r_k2_h128 s43 (22897775).

- 2026-09-17 06:45 — **task3r_k2_h128_nv_c1 (3 redrawn arenas) is the first
  arm strong on both halves**: held-out u1500 found 0.66–0.68, revisits
  1.00 at **20 steps**, 21 steps/touch, follow_q 0.54; u1000 was 0.54–0.59
  / 1.00 at 28 / 27 / 0.41 — still improving, no decay yet. Seed 43
  replicate launched (22884953, mit_normal_gpu 6 h). K=4 s42 at u2000:
  found 0.64, revisits 0.95 at 34, 45 steps/touch, follow_q 0.17 (partial
  recovery from u1500, well below its u1000 peak). K=4 s43 u500: found
  0.40, revisits 1.00 at 44, follow_q 0.16. task1r_k2_h128_nv_c1 hit its
  6 h wall at ~u3600 (22872240): exploit never left ~50 steps/touch.
- 2026-09-17 07:20 — **wave-1 rule under redraw: task1r_k4_h128 at u500**
  (held-out): found 0.52–0.59, revisits 1.00 at **18.9 steps**, 17.6
  steps/touch, 7.6 touches per rollout, follow_q **0.80**. The exploit
  level the fixed-goal `nv` arms needed ~2000 updates for, at u500, with
  honest search and no avoidance possible (the goal moves). Novelty off
  after the store gives an unambiguous exploit signal; the wave-1 failure
  was the *known* goal, not the rule. For the record at the same hour:
  task3_k2_h128_nv u2500 14.4 steps/touch, revisits 1.00 at 19, follow_q
  0.79, found 0.25–0.38; K=4 s43 u1000 revisits 1.00 at 31, follow_q 0.28.
- 2026-09-17 08:50 — wave 4 at u1000, held-out: **task1r_k4_h128** (wave-1
  rule) found 0.53–0.57, revisits 1.00 at 18 steps, 15 steps/touch,
  follow_q 0.87 — holding, no decay; **task3r_k2_h128** (wave-1 rule) u500
  revisits 1.00 at 17.3, 18.4 steps/touch, follow_q 0.69, found 0.41–0.47;
  `_g5` u1000 revisits 1.00 at 18, follow_q 0.73 but found 0.29–0.32 — the
  big goal reward starves the sweep. The one-rule (`nv`) tie is a slow
  oscillation, not a dead end: K=4 nv s42 recovered by u3000 (revisits
  1.00 at 14.7, follow_q 0.44, found 0.52–0.60) after its u1500 trough; s43
  shows the same trough at u1500 (45 steps/touch, follow_q 0.19).
  task3r_k2_h128_nv_c1 s42 u2000: revisits 1.00 at **12.8**, 13.7
  steps/touch, follow_q 0.84, found 0.43–0.50 (from 0.66–0.68 at u1500).
  Fixed-goal task3_k2_h128_nv u3000: held-out found 0.17–0.20, exploit
  intact (14–15 steps/touch) — the shortcut keeps eroding sweep.
- 2026-09-17 09:55 — **task3r_k2_h128 (wave-1 rule, 3 redrawn arenas, K=2)
  at u1000**: found 0.61 / 0.46 (d 0 / 10), revisits 1.00 at 14.6, 12.7
  steps/touch, follow_q 0.89 — both halves at u1000; task1r_k4_h128 u1500
  0.52–0.54 / 1.00 at 15.9 / 14.4 / 0.87; task3r_k2_h128_nv_c1 s42 u2500
  0.42–0.52 / 1.00 at **12.4** / 12.5 / 0.91; K=4 nv s42 u3500 recovered
  (1.00 at 14.7, 13.6, 0.83, found 0.42–0.53). Wave-1 survivor 22864187
  timed out at u3400 (record complete). Verdicts submitted: held-out probe
  22893595 (`run_se_probe.sh`, TAG task_r1: task3r_k2 u1000, task1r_k4
  u1500, task3r_nv_c1 u2500, task3_nv u2500 + d0_base u725) and the
  continual protocol (200 iterations per arena, sampled) on task3r_k2_h128
  u1000 (22893661) and task1r_k4_h128 u1500 (22893662). Note: job 22891316
  `xfB0_task1r_k4_h1024` is another session's run of this launcher, not
  part of this log.

## 7. Verdicts (held-out probe, 22893595, `se_task_r1_*`)

Six unseen arenas; exploit = pre-stored goal, deterministic, 32 trials × 6
envs, "× optimal" = steps / (start_dist − 1); explore = 144 sampled trials,
swept at 200 steps, swept_eff = swept / billiard at the model's own speed.

| checkpoint | success d 0/5/10 | × optimal | align_true | swept d0/d10 | speed | swept_eff d0 |
|---|---|---|---|---|---|---|
| task3r_k2_h128 u1000 (wave-1 rule) | 1.00/1.00/0.99 | 1.40/1.38/1.46 | 0.83/0.83/0.73 | 0.49/0.48 | 0.85 | 0.82 |
| task1r_k4_h128 u1500 (wave-1 rule) | 1.00/1.00/0.98 | 1.32/1.31/1.50 | 0.86/0.83/0.59 | 0.48/0.47 | 0.81 | 0.82 |
| **task3r_k2_h128_nv_c1 u2500** | 1.00/1.00/0.99 | **1.17/1.20/1.29** | **0.93/0.91/0.75** | 0.54/0.53 | 0.93 | 0.85 |
| task3_k2_h128_nv u2500 (fixed goals) | 1.00/1.00/1.00 | 2.44/2.25/2.67 | 0.45/0.48/0.39 | 0.26/0.37 | 0.52 | 0.58 |
| d0_base u725 | 1.00/1.00/1.00 | 1.20/1.21/1.31 | 0.91/0.90/0.81 | 0.61/0.59 | 0.96 | 0.95 |

Reading: the three-redrawn-arena one-rule arm at u2500 exploits as well
as d0_base on unseen arenas (and matches fix1_h128 s43's 1.14/1.14/1.19
without ever seeing a fixed goal); the wave-1-rule arms are 10–20 % less
direct at 1000–1500 updates and still improving; all redraw arms sweep at
82–85 % of d0_base's efficiency and move more slowly (0.81–0.93 vs 0.96),
which is most of the raw swept gap. The fixed-goal arm's deterministic
exploit is a position-map walk (2.4× optimal, align 0.45) although sampled
it reaches the goal in ~15 steps — the two evals disagree exactly when the
policy is not following q. Collapsed-tail fraction 0.29–0.32 for every arm
including d0_base (0.32).

**Continual protocol** (`run_cl_ood.sh`, six unseen arenas, 200 iterations
per block, 200-step cap, one oracle store per arena, Hopfield never reset,
sampled; outputs `results/nav_tri_probe/cl_i200/cl_task*`):

| checkpoint | job | first visits (1200) | locked-store revisits (3000) | worst retention Δ |
|---|---|---|---|---|
| task3r_k2_h128 u1000 | 22893661 | 0.9925 at 15.2 steps | **3000/3000 at 15.4** | +0.00 |
| task1r_k4_h128 u1500 | 22893662 | 0.9875 at 17.0 | 2997/3000 at 15.9 | −0.01 (one arena 0.98 in the last block) |
| fix1_h128 s43 u3000 (§8.15, for reference) | 22859040 | 0.993 at 11.8 | 3000/3000 at 11.6 | +0.00 |

No dead arena in either. The task-trained models reproduce the zero-
forgetting result of the fixed-goal line with no fixed goal ever seen, at
1000–1500 updates; their revisits are ~4 steps longer than fix1_h128's,
consistent with the probe's × optimal (1.3–1.4 vs 1.14).

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
