# Sample efficiency — d0_base's quality on a fraction of its data

Started 2026-09-13. Jack: *"train a model that is just as good as
`navigate_navp2_d0_base_s42_22133273/navigate_u725.pt` but trained on as few
samples as possible. Less envs or less rollouts or both (probably both). And
you can change all sorts of rewards and train configs (batching, PPO configs,
etc.). I really want to push the number of samples down dramatically."*

Branch `worktree-sample-eff`, worktree `.claude/worktrees/sample-eff`.
Launcher: `VARIANT=se_* hopfield_nav/run_nav_p2.sh` (the `se_*` block inherits
the `d0_base` recipe verbatim and layers the sample knobs on top).

Read §0 to resume.

---

## 0. Where I am

| | |
|---|---|
| **Target** | `d0_base` u725 on the held-out probe (§1). Exploit: success ≥ 0.995, ≤ ~1.19× optimal at d = 10. Explore, sampled: `swept_eff` ≥ 0.93, collapsed tail ≤ 0.02. |
| **Baseline cost** | **928,000 episodes / ~103.5 M env-steps** at u725 (1,280 episodes per update). Its own training-eval window first clears the §2 screen at **u625 = 800k episodes / ~91 M env-steps**. |
| **Where the samples went** | 16 gradient steps per update on ~64k-transition minibatches, every sample touched 4× — ~60× more data per gradient step than a textbook PPO update. That is the lever wave S1 pulls. |
| **Running** | Wave S1 (§3) — six arms, `ou_bcs_normal`, 24 h. |
| **Tools** | `analysis/nav_tri/sample_eff_curve.py` (eval series vs cumulative samples, window means, first-clear of the screen); the wave-1 probe pipeline `hopfield_nav/run_wave1_final.sh` pattern for the verdict. Trainer now logs exact `episodes` / realized `env_steps` per update and per eval, and writes both into every checkpoint. |

---

## 1. The bar — what "as good as d0_base u725" means

Numbers from `w1_final_22177955.out` (place=held_out, 6 envs; explore 144
sampled trials, exploit 192 trials), the same probe every candidate is scored
with:

| half | d | metric | d0_base u725 |
|---|---|---|---|
| exploit | 0 | success / steps / × optimal | 1.000 / 11.74 / 1.15 |
| exploit | 10 | success / steps / × optimal | 0.995 / 12.14 / 1.19 |
| explore | 0 | swept / `swept_eff` / frac collapsed | 0.603 / 0.948 / 0.000 |
| explore | 10 | swept / `swept_eff` / frac collapsed | 0.589 / 0.938 / 0.021 |

Two things about reading it, both from the wave-1 write-up
(`DUAL_TRAINING` §9.0.1 caveat 5): the probe's tail is not reproducible across
different `--ckpt` list lengths (a < 2× tail difference is noise), and
`d0_base` u725 must be **in the same probe process** as any candidate so the
envs, starts and memory contents match.

**A candidate passes when, in one probe run beside d0_base u725, it is within
noise of every row above.** Success within 0.01, × optimal within 0.03,
`swept_eff` within 0.02, tail not more than 2× — and the training-eval window
(§2) has held for ≥ 4 evals, per `feedback_eval_point_threshold`.

## 2. The screen — the training eval, as a window mean

Every run evaluates 6 val envs × 16 trials at d = 0 and 10, both halves, every
25 updates. The eval swings 30+ points between points, so the screen is the
mean over the **last 4 evals**, and the bar is what d0_base's own last four
(u650–u725) read:

`succ0 ≥ 0.99 · succ10 ≥ 0.99 · steps0 ≤ 13.0 · steps10 ≤ 14.0 · swept0 ≥ 0.55 · swept10 ≥ 0.50`

`sample_eff_curve.py --at_target` prints the first window that clears it and
the samples spent by then. A screen pass says "probe this checkpoint", nothing
more.

## 3. Wave S1 — the learning rate, under many steps, on 1/8–1/16 of the pool

### 3.1 What was measured first — how far one gradient step moves this policy

`analysis/nav_tri/ppo_step_probe.py` (job 22701002): one update's pool built
exactly as `run_navigate` builds it (20 envs × 8 = 160 trajectories, 32,000
realized env-steps), then `ppo_update` from that pool under a grid, from
d0_base u725's weights with (a) a fresh Adam, (b) **d0_base's own Adam
moments from `resume/latest.pt`**, and (c) a fresh init. Approx-KL is the k3
estimate on the scored steps against the rollout policy; clip is the fraction
outside ±0.15.

| condition | optimizer | after 1 step | after 16 steps | after 80 steps |
|---|---|---|---|---|
| u725, fresh Adam | 3e-4 | KL 0.056, clip 0.52 | 0.07 / 0.48 | 0.03–0.07 / 0.35 (saturated) |
| u725, **real Adam** | 3e-4 | 0.001 | **0.015 / 0.22** (= d0_base's update) | **0.06 / 0.50** ← fitting noise |
| u725, real Adam | 1e-4 | 0.000 | 0.007 / 0.11 | 0.010 / 0.15 |
| u725, real Adam | 3e-5 | 0.000 | 0.003 / 0.04 | 0.005 / 0.09 |
| **fresh init**, fresh Adam | 3e-4 | **0.19–1.5, clip 0.66–0.93** | 0.44 / 0.75 | 0.18 / 0.64 |
| fresh init | 1e-4 | 0.21 / 0.76 | 0.05 / 0.44 | 0.055 / 0.49 |
| fresh init | 3e-5 | 0.02 / 0.38 | 0.011 / 0.21 | 0.020 / 0.34 |

Three things this settles before a single arm runs:

1. **Eighty steps at 3e-4 overfit the pool.** On a *converged* policy — where
   the true gradient is ~0 — they drift it to KL 0.06 with half the pool
   clipped. So "more epochs at the same lr" is not an arm, and `target_kl`
   exists to catch exactly this (0.1 fires at step 33 there; never fires at
   1e-4 or 3e-5 in steady state).
2. **d0_base's first ~100 updates ran on a policy reshuffled every step.**
   From init, one 3e-4 step is KL 0.4–1.5 — the polar head's direction
   vector has norm 0.09 at init, so a small parameter change swings the
   heading by tens of degrees. That is 128k episodes spent before the
   optimizer is in a regime PPO's clip can govern. At 3e-5 the same step is
   0.01–0.02 and 80 of them accumulate *coherently* with two thirds of the
   pool still inside the clip region.
3. **An update is entirely rollout time.** 20 envs × 200 serial steps =
   30–36 s (≈8 ms per step, batch-independent); PPO < 1 s even at 80 steps.
   Gradient steps are free in wall-clock; wall-clock scales with envs × T.
   24 h buys ~2,500 updates for a 20-env arm, ~6,500 for 10 envs.

`target_kl=0.02`, the textbook value, stops every update at step 2 here
(smokes 22700949/951) — the KL scale of a von Mises heading at κ≈12 is not
a Gaussian's. Dropped in favour of 0.1 as a safety.

### 3.2 The arms

All arms are the `d0_base` recipe (w52 encoder, gain = β = 100, polar, κ cap
2.5, speed [0.5, 1.0], `interleave, empty_frac=0.5`, shuffle, ε 0.1/200,
goal 2.0, persistence 0.20, depth-1 Hopfield input), seed 42, schedule
`interleave:4000`, on `ou_bcs_normal` a100s for 24 h. Only these rows differ.

| arm | envs × batch | traj/update | epochs × mb | steps/u | lr | target_kl | episodes/u vs d0_base |
|---|---|---|---|---|---|---|---|
| `d0_base` | 20 × 64 | 1280 | 4 × 4 | 16 | 3e-4 | — | 1× |
| `se_b8` | 20 × 8 | 160 | 4 × 4 | 16 | 3e-4 | — | **1/8** — control: d0_base's optimizer on 1/8 of the data |
| `se_b8_lr1` | 20 × 8 | 160 | 10 × 8 | 80 | 1e-4 | 0.1 | 1/8 |
| `se_b8_lr03` | 20 × 8 | 160 | 20 × 8 | 160 | 3e-5 | 0.1 | 1/8 |
| `se_b4_lr1` | 20 × 4 | 80 | 10 × 8 | 80 | 1e-4 | 0.1 | **1/16** |
| `se_n10_b8_lr1` | 10 × 8 | 80 | 10 × 8 | 80 | 1e-4 | 0.1 | 1/16, half the serial calls — diversity vs pool size |
| `se_b8_lr1_h100` | 20 × 8, T = 100 | 160 | 10 × 8 | 80 | 1e-4 | 0.1 | 1/8 episodes, ~1/16 env-steps |

Samples-to-target is read off the first checkpoint that passes §1, never off
the end of the run. TIMEOUT at 24 h is the expected end.

**Predictions on record, before any eval landed.**

1. `se_b8` reaches the exploit lock (success pinned at 1.000) in *more*
   updates than d0_base's u125 but far fewer episodes; if it needs more than
   8× the updates, the pool is the constraint and only the lr arms can help.
2. `se_b8_lr1` and `se_b8_lr03` beat `se_b8` on updates-to-lock, because the
   early phase stops being a random walk (§3.1 point 2). `lr03` may be slower
   in updates late, when the policy is converged and a small trust-region
   move per update is the cost.
3. The explore half separates the arms: its slowest quantity is the collapsed
   tail (d0_base closed it between u250 and u725), and a small pool sees
   fewer of the rare high-‖q‖ goal-absent draws per update. That may cap the
   gain at ~2–4× on *episodes* regardless of optimizer, which would point
   wave S2 at the distractor distribution rather than the optimizer.
4. `h100` matches `se_b8_lr1` on exploit and loses on 200-step explore,
   because coverage over the second 100 steps is never rewarded.
5. `n10` ≈ `b4` on samples (same pool) — env diversity per update is not the
   constraint when the regimes are reshuffled every update anyway.

## 4. Results

*(window means of the training eval, `sample_eff_curve.py`; the verdict is
§1's probe, `hopfield_nav/run_se_probe.sh`)*

### 4.0 Launch record

| arm | job | node | s/update (roll / ppo) | u1 approx_kl / clip / steps |
|---|---|---|---|---|
| `se_b8` | 22701297 | node2702 | 28.6 (27.8 / 0.8) | **1.47 / 0.81** / 16 — the init random walk, live |
| `se_b8_lr1` | 22701298 | node3811 | 27.3 (26.8 / 0.5) | 0.083 / 0.35 / 2 (KL stop at step 2, fresh Adam) |
| `se_b8_lr03` | 22701299 | node2703 | 28.4 (25.4 / 3.0) | 0.034 / 0.39 / 160 |
| `se_b4_lr1` | 22701300 | node3911 | 28.3 (27.5 / 0.8) | 0.049 / 0.33 / 3 |
| `se_n10_b8_lr1` | 22701302 | node3810 | 13.2 (12.4 / 0.8) | u10: 0.052 / 0.31 / 66 |
| `se_b8_lr1_h100` | 22701304 | node3811 | 13.7 (13.1 / 0.6) | u10: 0.059 / 0.44 / 31 |

### 4.1 Digest at ~1.4 h (window means of the last ≤4 evals; episodes exact)

| arm | u | episodes | succ0 / succ10 | steps0 / steps10 | swept0 / swept10 |
|---|---|---|---|---|---|
| `se_b8` (3e-4, 4×4) | 75 | 12,000 | 0.51 / 0.54 | 66 / 66 | 0.17 / 0.17 |
| `se_b8_lr1` | 75 | 12,000 | 0.54 / 0.51 | 47 / 43 | 0.15 / 0.17 |
| `se_b8_lr03` | 75 | 12,000 | **0.96 / 0.95** | 53 / 53 | 0.28 / 0.23 |
| `se_b4_lr1` | 75 | 6,000 | 0.73 / 0.72 | 68 / 65 | 0.25 / 0.24 |
| `se_n10_b8_lr1` | 175 | 14,000 | **0.995 / 0.990** | **25 / 28** | 0.28 / 0.27 |
| `se_b8_lr1_h100` | 150 | 24,000 | 0.995 / 0.995 | 30 / 29 | 0.14 / 0.17 |
| `se_b8_akl` | 50 | 8,000 | 0.81 / 0.80 | 68 / 71 | 0.20 / 0.18 |
| d0_base (ref) | 125 | 160,000 | 1.00 / 1.00 | 34 / 34 | 0.20 / 0.21 |
| d0_base (ref) | 150 | 192,000 | 1.00 / 1.00 | 22 / 23 | 0.26 / 0.34 |

- **The exploit lock costs 12–14k episodes in `se_n10_b8_lr1`** (success
  ≥0.99 over a 4-eval window from u125–175) against d0_base's 160k — **~12×**
  — and its steps/swept at that point match what d0_base had at u150 (192k).
- The per-sample leader is the arm with the most, smallest updates (80
  trajectories, 175 of them), not the arm with the best per-update curve
  (`lr03`). Prediction 5 (n10 ≈ b4 on samples) is **wrong so far**: n10 is
  well ahead of b4 at the same pool size, so the number of policy moves is
  what the small pool buys, and 10 envs' worth of diversity per update is
  enough. → `se_n5_b8_lr1` (40 trajectories/update) launched as the 8th arm.
- `se_b8` — d0_base's optimizer on 1/8 the data — is the worst arm: a 3e-4
  step from a 160-trajectory pool is a noisy step of the same size, exactly
  what §3.1 predicted. The KL-adaptive arm settled at lr 2e-5 by u20 on its
  own (`kl_final` 0.016 in its 0.01–0.04 band).
- `h100` locks exploit at 24k but its explore half is the weakest (swept0
  0.14) — prediction 4 holding.

### 4.2 Digest at ~2.4 h

| arm | u | episodes | env-steps | succ0 / succ10 | steps0 / steps10 | swept0 / swept10 |
|---|---|---|---|---|---|---|
| `se_b8` (3e-4) | 175 | 28,000 | 5.6M | 0.90 / 0.92 | 44 / 44 | 0.21 / 0.22 |
| `se_b8_lr1` | 175 | 28,000 | 5.6M | 1.00 / 0.997 | 22.5 / 24.6 | 0.38 / 0.38 |
| `se_b8_lr03` | 175 | 28,000 | 5.6M | 1.00 / 1.00 | 21.0 / 21.2 | 0.41 / 0.35 |
| `se_b4_lr1` | 175 | 14,000 | 2.8M | 0.997 / 1.00 | 24.7 / 28.8 | 0.30 / 0.30 |
| **`se_n10_b8_lr1`** | **375** | **30,000** | **6.0M** | **1.00 / 0.997** | **14.7 / 15.7** | **0.50 / 0.48** |
| `se_b8_lr1_h100` | 350 | 56,000 | 5.6M | 1.00 / 1.00 | 13.6 / 13.9 | 0.43 / 0.44 |
| `se_b8_akl` | 175 | 28,000 | 5.6M | 0.995 / 0.98 | 40 / 42 | 0.27 / 0.29 |
| `se_n5_b8_lr1` | 325 | 13,000 | 2.6M | 0.88 / 0.85 | 47 / 52 | 0.21 / 0.21 |
| d0_base (ref) | 400 | 512,000 | ~60M | 1.00 / 1.00 | 15.5 / 17.8 | 0.52 / 0.48 |

- `n10` at 30k episodes reads what d0_base read at u400 (512k episodes):
  **~17× fewer episodes, ~10× fewer env-steps** for that quality, and it is
  one notch from the §2 screen (steps ≤13/14, swept ≥0.55/0.50).
- Every arm at lr ≤1e-4 locks exploit by 14–28k episodes; `se_b8` (3e-4)
  only reaches 0.90/0.92 at 28k. The optimizer, not the pool, was d0_base's
  bottleneck in the early phase.
- `n5` (40 trajectories) is *worse* per sample than `n10` (0.85 vs 0.99
  success at 12k episodes): 5-trajectory minibatches trip the KL safety at
  7–48 steps and the steps are noisier. ~80 trajectories per update is the
  floor for this recipe; the win is in the number of updates, not in
  shrinking the pool further.
- `h100` is on the screen for exploit (13.6/13.9) at 56k episodes but its
  explore half trails (0.43/0.44 vs n10's 0.50/0.48 at half the episodes).
- `akl` settled at lr 2e-5 and is the slowest of the small-lr arms on steps
  (40): its 0.01–0.04 band is set too low for the late phase.

### 4.3 Digest at ~3.4 h — the first screen clear

| arm | u | episodes | env-steps | succ0 / succ10 | steps0 / steps10 | swept0 / swept10 | screen |
|---|---|---|---|---|---|---|---|
| **`se_b8_lr1_h100`** | **575** | **92,000** | **9.2M** | 1.00 / 1.00 | 12.1 / 12.5 | 0.57 / 0.52 | **YES (first)** |
| `se_n10_b8_lr1` | 625 | 50,000 | 10.0M | 0.99 / 0.98 | 14.0 / 15.3 | 0.54 / 0.53 | close |
| `se_b8_lr03` | 325 | 52,000 | 10.4M | 1.00 / 1.00 | 14.1 / 14.5 | 0.54 / 0.47 | close |
| `se_b8_lr1` | 300 | 48,000 | 9.6M | 1.00 / 1.00 | 14.3 / 15.8 | 0.45 / 0.44 | |
| `se_b4_lr1` | 300 | 24,000 | 4.8M | 1.00 / 1.00 | 15.0 / 15.3 | 0.45 / 0.40 | |
| `se_b8` (3e-4) | 325 | 52,000 | 10.4M | 1.00 / 0.997 | 17.0 / 18.9 | 0.44 / 0.42 | |
| `se_b8_akl` | 350 | 56,000 | 11.2M | 1.00 / 0.995 | 27.0 / 27.2 | 0.42 / 0.39 | |
| `se_n5_b8_lr1` | 825 | 33,000 | 6.6M | 0.98 / 0.98 | 21.1 / 23.6 | 0.39 / 0.39 | **cancelled** |
| d0_base (ref) | 625 | 800,000 | ~91M | 1.00 / 0.997 | 12.6 / 13.9 | 0.55 / 0.53 | YES (first) |

- **`h100` clears the screen at 92k episodes / 9.2M env-steps — 8.7× / 10×
  below d0_base's own screen clear (800k / ~91M), 10× / 11× below u725.**
  Prediction 4 (h100 loses on 200-step explore) is **wrong**: its swept at
  200 eval steps is on the bar. Coverage over the second hundred steps is
  apparently what a memoryless vector field does anyway.
- Per env-step, `h100` and `n10` are the same pool (16k transitions per
  update) and reach the bar region at the same env-step count (~9–10M);
  `n10` has half the episodes. Both are ~600 updates in, which is d0_base's
  update count — **the small pool did not cost updates**.
- `n5` cancelled at u825 to free a GPU for the probe: at 33k episodes it
  trails `n10`-at-33k on every column (steps 21 vs ~14.5). 40 trajectories is
  below the floor. Its checkpoints are kept.
- **Probe round 1 launched** (job 22707156, GPU): h100 u550/u575/u600 and
  n10 u625 beside d0_base u725.

Timing smokes before the wave (8 updates each, 3e-4, target_kl 0.02):
22700947 (`se_b8`) 36 s/u, 22700949 (20 envs, 10×8) 30 s/u with the KL stop
firing at step 2 every update, 22700951 (10 envs, 10×8) 12.4 s/u. The
per-update line now carries `(roll= ppo=)`, `eps_cum= steps_cum=`,
`approx_kl clip_frac epochs_run grad_steps`.

## 5. Accounting notes

- **Episodes** are exact: envs × batch per update, every update.
- **Env-steps** are realized transitions (`alive_mask.sum()`), logged since
  2026-09-13 as `eps_cum= steps_cum=` on the per-10-update line and as
  `[navigate_uN] samples={...}` beside every eval; both are in each
  `navigate_uN.pt` as `cum_episodes` / `cum_env_steps`. For d0_base, whose
  log predates this, `sample_eff_curve.py` reconstructs env-steps from the
  eval's `mean_steps` and lands at 104.4 M vs the 103.5 M the run's own
  rollout diagnostics gave — good to 1%.
- A `--continue_from` carries both counters, so a resumed run reports the run's
  total, not the segment's.
