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
| **Baseline cost** | **928,000 episodes / 185.6 M env-steps** at u725 (1,280 episodes × 200 steps per update; no rollout in this recipe ends early — see §6). Its own training-eval window first clears the §2 screen at **u625 = 800k episodes / 160 M env-steps**. |
| **Where the samples went** | 16 gradient steps per update on ~64k-transition minibatches, every sample touched 4× — ~60× more data per gradient step than a textbook PPO update. That is the lever wave S1 pulls. |
| **Runs** | All on `ou_bcs_normal`. Completed their 4000-update schedules: `se_n10_b8_lr1` (22701302), `se_b8_lr1_h100` (22701304), `se_n10_b8_lr1_h100` (22707208). Ran to the wall: `se_b8_lr1` (22701298, the delivered arm), `se_h100_d5` (22707437), `se_lr1_e75` (22715546), `se_lr1_d5_e75` (22715547). Cancelled with checkpoints kept: `se_n5_b8_lr1` (40 traj too few), `se_b8_akl` (band too low), `se_b4_lr1` (n10 ≥ b4), `se_b8` (d0_base's optimizer on 1/8 data ≈ d0_base at matched updates), `se_b8_lr03` (3e-5 too slow late). |
| **RESULT** | **§5.5: `se_b8_lr1` u1500 at 240k episodes / 48M env-steps passes every held-out probe row against d0_base u725 (928k / 185.6M) — 3.9× fewer samples; explore efficiency better at both levels.** u1000 at 160k equals d0_base u600 on every row (4.8×) and u725 on all but a 2×-within-noise tail (5.8×); the exploit half alone is matched at 80–92k (10×; 20× on env-steps with 100-step rollouts). Delivered: `agent_ckpts/navigate_navp2_se_b8_lr1_s42_22701298/navigate_u1500.pt`. Recipe change: `batch_envs` 64 → 8, PPO lr 1e-4 × 10 epochs × 8 minibatches, `target_kl` 0.1. |
| **Curves page** | [15cc014f](https://claude.ai/code/artifact/15cc014f-fc15-4432-9f7f-8f7e41be1bc7) — success / path optimality / swept coverage for `se_b8_lr1` (deterministic and sampled) and `d0_base` (sampled), by update, episodes and env-steps. Sources `results/nav_tri_probe/*_training_curve.png`; sampled rows from `reeval_series.py`. |
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

### 4.1 Digest at ~45 min (arms started 20:01) (window means of the last ≤4 evals; episodes exact)

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

### 4.2 Digest at ~1.3 h

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
| d0_base (ref) | 400 | 512,000 | 102M | 1.00 / 1.00 | 15.5 / 17.8 | 0.52 / 0.48 |

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

### 4.3 Digest at ~2.3 h — the first screen clear

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
| d0_base (ref) | 625 | 800,000 | 160M | 1.00 / 0.997 | 12.6 / 13.9 | 0.55 / 0.53 | YES (first) |

- **`h100` clears the screen at 92k episodes / 9.2M env-steps — 8.7× / 17×
  below d0_base's own screen clear (800k / 160M), 10× / 20× below u725.**
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

### 4.4 Digest at 4.5 h — three arms through the screen

| arm | screen clear | episodes | env-steps | latest window (u · steps0/10 · swept0/10) |
|---|---|---|---|---|
| `se_b8_lr1` | **u475** | **76,000** | 15.2M | u575 · 12.0/12.5 · 0.57/0.53 |
| `se_b8_lr1_h100` | u575 | 92,000 | **9.2M** | u1100 · 11.9/13.3 · 0.55/0.52 |
| `se_n10_b8_lr1_h100` | not yet | 74,000 so far | 7.4M | u925 · 13.7/15.6 · 0.55/0.53 |
| `se_n10_b8_lr1` | not yet | 92,000 so far | 18.4M | u1150 · 12.9/15.0 · 0.58/0.55 — steps10 sits at ~15 |
| `se_b8_lr03` | not yet | 104,000 so far | 20.8M | u650 · 14.5/15.9 · 0.57/0.53 |
| `se_b8` (3e-4) | not yet | 116,000 so far | 23.2M | u725 · 13.2/13.4 · 0.50/0.45 |
| `se_h100_d5` | not yet | 64,000 so far | 6.4M | u400 · 15.2/17.0 · 0.48/0.44 |
| d0_base | u625 | 800,000 | 160M | u725 · 12.0/13.1 · 0.585/0.557 |

- The 10-env arms lock exploit earliest per sample but their d=10 `steps`
  settle ~1–2 above the 20-env arms (15 vs 13): with 5 exploit envs per
  update the d=10 directness signal is thinner. Per env-step the combo arm is
  still the cheapest at the bar's edge (7.4M).
- `se_b8` at matched updates (u725) trails d0_base by 1.2 steps at d=0 and
  0.08 swept — the 1/8 pool at 3e-4 costs a little per update and ~8× less
  data; the lr arms cost nothing per update.
- `h100_d5`'s exploit lock came late (u~300) but it is now on the same
  trajectory as h100 was; its d=10 tail is what round 3 will read.

Timing smokes before the wave (8 updates each, 3e-4, target_kl 0.02):
22700947 (`se_b8`) 36 s/u, 22700949 (20 envs, 10×8) 30 s/u with the KL stop
firing at step 2 every update, 22700951 (10 envs, 10×8) 12.4 s/u. The
per-update line now carries `(roll= ppo=)`, `eps_cum= steps_cum=`,
`approx_kl clip_frac epochs_run grad_steps`.

## 5. Verdicts — the held-out probe

### 5.1 Round 1 (job 22707156, ~25 min on one a100): exploit passes at 92k, the d=10 tail does not

`run_se_probe.sh`, place=held_out, 6 envs; explore 144 sampled trials per
level, exploit 192 trials per level, d0_base u725 in the same process.

**Exploit** (success d=0/5/10 · steps d=0/d=10 · × optimal at d=10 =
steps ÷ ((start − 1) / realized speed)):

| checkpoint | episodes | env-steps | success | steps | × opt | `align_true` d10 |
|---|---|---|---|---|---|---|
| h100 u550 | 88k | 8.8M | 1.000 / 0.995 / 1.000 | 11.90 / 12.49 | 1.20 | 0.875 |
| **h100 u575** | **92k** | **9.2M** | 1.000 / 0.995 / 0.995 | 11.73 / 12.46 | 1.20 | 0.879 |
| h100 u600 | 96k | 9.6M | 1.000 / 0.995 / 1.000 | 11.55 / 12.42 | 1.19 | 0.887 |
| n10 u625 | 50k | 10.0M | 1.000 / 1.000 / 1.000 | 12.28 / 13.01 | 1.25 | 0.850 |
| d0_base u725 | 928k | 185.6M | 1.000 / 0.990 / 1.000 | 11.71 / 12.35 | 1.21 | 0.880 |

**Explore** (`swept_eff` = swept ÷ billiard at own speed; tail = frac
collapsed below ½ billiard; `chase_t` = chase_q inside the tail):

| checkpoint | d=0 swept / eff / tail | d=10 swept / eff / **tail** / chase_t |
|---|---|---|
| h100 u550 | 0.543 / 0.871 / 0.035 | 0.477 / 0.789 / **0.188** / 0.58 |
| h100 u575 | 0.578 / 0.919 / 0.000 | 0.511 / 0.858 / **0.139** / 0.59 |
| h100 u600 | 0.557 / 0.890 / 0.042 | 0.497 / 0.830 / **0.160** / 0.48 |
| n10 u625 | 0.481 / 0.783 / 0.090 | 0.508 / 0.824 / **0.104** / 0.56 |
| d0_base u725 | 0.606 / 0.944 / 0.000 | 0.589 / 0.931 / **0.021** / 0.56 |

1. **The exploit half is matched at 92k episodes / 9.2M env-steps — 10× / 20×
   below d0_base u725** (h100 u575 and u600: success ≥0.995 at every level,
   steps within 0.1 at d=0 and d=10, × optimal 1.19–1.20 vs 1.21,
   `align_true` equal). n10 u625 passes success at 50k and is 0.6 step / 0.04×
   behind on directness.
2. **Explore at d=0 is within noise at u575** (eff 0.919 vs 0.944, tail 0).
3. **Explore at d=10 is not.** The collapsed tail is 0.10–0.19 against 0.021,
   with `chase_t` ≈0.5–0.6 — the corner trap (chasing a phantom recall), the
   thing d0_base closed between u250 (12.5%) and u725 (1.4%). These
   candidates are at u575–625 and have the tail d0_base had at ~u300.
   Prediction 3 of §3.2 — that the tail is the quantity a small pool cannot
   buy cheaply — is the live hypothesis. Two ways it can resolve: the arms
   keep running (the tail closes with UPDATES → h100 passes at ~u1000–1500 =
   160–240k episodes, ~4–6×), or it needs EPISODES with many stored patterns
   (→ the distractor floor arm below).
4. The u550/u575/u600 spread (tail 0.19 / 0.14 / 0.16, eff 0.79 / 0.86 / 0.83)
   is the probe's own noise plus checkpoint-to-checkpoint swing; per the wave-1
   caveat a < 2× tail difference is not a ranking.

**Action.** `se_h100_d5` launched: h100 with the explore-regime distractor
floor raised to 5 (U[5,10]; exploit unchanged). Half of the U[0,10] explore
episodes carry ≤5 patterns and teach nothing about the tail. `b4` cancelled
to free the slot (its question is answered: n10 ≥ b4 per sample at half the
wall-clock). Probe round 2 at u~1000 of h100 / n10 / the combo arm.

### 5.2 Round 2 (job 22710570): `se_b8_lr1` u500 at 80k episodes is one row short

Same protocol. × optimal = steps ÷ ((start − 1) ÷ realized speed).

**Exploit** (d = 0 / 5 / 10):

| checkpoint | episodes / env-steps | success | steps | × opt | `align_true` |
|---|---|---|---|---|---|
| **lr1 u500** | **80k / 16.0M** | 1.000 / 0.990 / 1.000 | 11.79 / 12.43 / 12.16 | 1.17 / 1.18 / 1.19 | 0.90 / 0.90 / 0.89 |
| h100 u800 | 128k / 12.8M | 1.000 / 0.995 / 1.000 | 12.12 / 12.94 / 12.75 | 1.18 / 1.20 / 1.22 | 0.89 / 0.88 / 0.87 |
| h100 u1000 | 160k / 16.0M | 1.000 / 1.000 / 1.000 | 12.52 / 13.23 / 13.49 | 1.22 / 1.22 / 1.29 | 0.87 / 0.86 / 0.83 |
| n10 u1000 | 80k / 16.0M | 1.000 / 1.000 / 1.000 | 11.87 / 12.49 / 12.57 | 1.16 / 1.16 / 1.21 | 0.91 / 0.91 / 0.88 |
| d0_base u725 | 928k / 185.6M | 1.000 / 0.995 / 0.995 | 11.66 / 13.29 / 11.99 | 1.16 / 1.26 / 1.18 | 0.91 / 0.85 / 0.90 |

**Explore**:

| checkpoint | d=0 swept / eff / tail | d=10 swept / eff / **tail** / chase_t |
|---|---|---|
| **lr1 u500** | **0.609 / 0.974 / 0.000** | **0.584 / 0.939 / 0.056** (8 of 144) / 0.57 |
| h100 u800 | 0.589 / 0.946 / 0.000 | 0.546 / 0.888 / 0.090 / 0.45 |
| h100 u1000 | 0.563 / 0.910 / 0.000 | 0.547 / 0.883 / 0.042 / 0.53 |
| n10 u1000 | 0.538 / 0.856 / 0.000 | 0.480 / 0.782 / 0.076 / 0.59 |
| d0_base u725 | 0.604 / 0.950 / 0.000 | 0.592 / 0.937 / 0.021 (3 of 144) / 0.46 |

1. **`se_b8_lr1` u500 — 80k episodes, 11.6× below d0_base u725 — matches or
   beats it on every row but one.** Exploit: steps within 0.2 at d=0 and
   d=10 and 0.9 better at d=5, × optimal 1.17–1.19 vs 1.16–1.26,
   `align_true` equal. Explore: efficiency 0.974 / 0.939 against
   0.950 / 0.937, d=0 tail 0. The one open row is the **d=10 collapsed
   tail: 8 episodes of 144 against 3** (0.056 vs 0.021, Fisher p ≈ 0.12;
   the wave-1 caveat puts a < 2× tail difference inside probe noise and this
   is 2.7×). Not declared; the arm is still training.
2. **The tail does close with updates.** h100 went 0.139 (u575) → 0.090
   (u800) → 0.042 (u1000) — but its d=10 directness went the other way
   (× optimal 1.20 → 1.29, `align_true` 0.88 → 0.83), the same
   clean-memory-vs-distractor trade `d0_base` showed between u600 and u725.
   h100's 100-step rollouts appear to under-train the d=10 exploit half late.
3. **The 10-env arm has the best exploit half of any candidate** (1.16 /
   1.16 / 1.21, `align_true` 0.91 / 0.91 / 0.88 — d0_base's numbers) at 80k
   episodes, and the weakest explore (eff 0.856 / 0.782). Five explore envs
   per update is enough for exploit and not for coverage.
4. Round 3 at lr1 u600 / u700, `h100_d5` u600 / u700 and the combo arm at
   u1200 / u1400, when they exist.

### 5.3 Round 3 (job 22714339): the tail is real, ~5–8% for every small-pool arm at ~100k

| checkpoint | episodes / env-steps | exploit steps d0/5/10 | × opt | explore d0 eff / tail | explore d10 eff / **tail** (n of 144) |
|---|---|---|---|---|---|
| lr1 u600 | 96k / 19.2M | 11.9 / 12.6 / 12.4 | 1.17 / 1.18 / 1.21 | 0.895 / 0 | 0.822 / **0.153** (22) |
| **lr1 u700** | **112k / 22.4M** | 12.0 / 12.5 / 12.5 | 1.19 / 1.18 / 1.22 | **0.972 / 0** | **0.927 / 0.069** (10) |
| h100_d5 u600 | 96k / 9.6M | 12.3 / 13.2 / 12.9 | 1.22 / 1.25 / 1.26 | 0.854 / 0.014 | 0.870 / 0.083 (12) |
| h100_d5 u700 | 112k / 11.2M | 12.2 / 12.8 / 12.1 | 1.21 / 1.21 / 1.19 | 0.934 / 0 | 0.892 / 0.056 (8) |
| combo u1200 | 96k / 9.6M | 12.5 / 13.4 / 13.0 | 1.24 / 1.26 / 1.27 | 0.927 / 0 | 0.901 / 0.083 (12) |
| combo u1400 | 112k / 11.2M | 12.1 / 13.6 / 12.7 | 1.20 / 1.28 / 1.23 | 0.926 / 0.007 | 0.900 / 0.083 (12) |
| d0_base u725 | 928k / 185.6M | 11.7 / 12.3 / 12.4 | 1.16 / 1.17 / 1.21 | 0.945 / 0 | 0.939 / **0.014** (2) |

1. **Every small-pool arm at ~100k episodes matches d0_base u725 on the
   exploit half and on explore efficiency at both levels** — `lr1` u700 is
   at 0.972 / 0.927 against 0.945 / 0.939 — **and every one carries a d=10
   collapsed tail of 5–8%** (8–12 of 144) against d0_base's 2–3. Three
   rounds, eleven candidate checkpoints, all above; this is not probe noise.
   `chase_t` 0.55–0.75 in every tail: the corner trap, as in §5.1.
2. The tail is the **noisiest quantity between adjacent checkpoints** of one
   run (`lr1` u500 / u600 / u700 = 8 / 22 / 10), which is why the training
   eval's mean coverage cannot screen it and why a single checkpoint's tail
   cannot be ranked against another's below ~2×.
3. **The distractor floor did not close it** (`h100_d5` u700: 8, against
   `lr1`'s 10 and `h100`'s 6 at u1000). Exposure to many-pattern memories
   per episode is not the limiting factor.
4. What is left on the table is exactly d0_base's own slowest-converging
   quantity (`DUAL_TRAINING` §9.2: "the explore tail had not plateaued when
   the 6 h wall stopped the run at u730"). Its value at d0_base u500 / u600
   was never probed; round 4 does that, because if d0_base-u600 (768k
   episodes) also reads ~5%, `lr1` at 112k matches *it* outright and the
   residual is d0_base's last 160k episodes of tail closure.

**Action.** Exploit locks at 12k episodes and explore is the bottleneck, so
the interleave *mix* is the untried lever on the sample axis: `se_lr1_e75`
(`empty_frac` 0.75, 15 explore + 5 exploit envs per update) and
`se_lr1_d5_e75`, launched at ~02:15 in place of `se_b8` (conclusion: d0_base's
optimizer on 1/8 the data ≈ d0_base at matched updates, i.e. the pool was
never the constraint) and `se_b8_lr03` (3e-5 is too slow late: steps 14.6 /
17.1 at u825). Round 4 at `lr1` u1000 with d0_base u500 / u600.

### 5.4 Round 4 (job 22717265): calibrated against d0_base's own trajectory — `lr1` u1000 IS d0_base u600

Same protocol, with d0_base's u500 and u600 probed as candidates beside u725.

**Exploit** (d = 0 / 5 / 10):

| checkpoint | episodes / env-steps | success | steps | × opt | `align_true` |
|---|---|---|---|---|---|
| lr1 u800 | 128k / 25.6M | 1.000 / 1.000 / 1.000 | **11.43 / 12.26 / 12.02** | **1.14 / 1.16 / 1.18** | **0.92 / 0.91 / 0.90** |
| **lr1 u1000** | **160k / 32.0M** | 1.000 / 1.000 / 0.995 | 11.51 / 13.33 / 11.99 | 1.14 / 1.25 / 1.17 | 0.92 / 0.85 / 0.90 |
| d0_base u500 | 640k / 128M | 1.000 / 0.995 / 1.000 | 13.31 / 14.39 / 13.59 | 1.30 / 1.34 / 1.31 | 0.83 / 0.80 / 0.82 |
| d0_base u600 | 768k / 153.6M | 1.000 / 0.995 / 1.000 | 11.98 / 13.09 / 12.62 | 1.18 / 1.23 / 1.22 | 0.89 / 0.86 / 0.87 |
| d0_base u725 | 928k / 185.6M | 1.000 / 0.995 / 0.995 | 11.66 / 13.29 / 11.99 | 1.16 / 1.26 / 1.18 | 0.91 / 0.85 / 0.90 |

**Explore**:

| checkpoint | d=0 swept / eff / tail | d=10 swept / eff / **tail** (n of 144) / chase_t |
|---|---|---|
| lr1 u800 | 0.532 / 0.853 / 0.049 | 0.510 / 0.825 / 0.139 (20) / 0.46 |
| **lr1 u1000** | **0.583 / 0.927 / 0.000** | **0.568 / 0.907 / 0.042 (6)** / 0.57 |
| d0_base u500 | 0.589 / 0.928 / 0.000 | 0.564 / 0.893 / 0.049 (7) / 0.56 |
| d0_base u600 | 0.593 / 0.931 / 0.000 | 0.574 / 0.923 / **0.042 (6)** / 0.43 |
| d0_base u725 | 0.602 / 0.942 / 0.000 | 0.589 / 0.931 / 0.021 (3) / 0.44 |

1. **`se_b8_lr1` u1000 — 160,000 episodes / 32.0 M env-steps — is d0_base u600
   (768,000 episodes / 153.6 M) on every row of the probe, tail included**:
   exploit steps 11.5 / 13.3 / 12.0 vs 12.0 / 13.1 / 12.6, × optimal
   1.14 / 1.25 / 1.17 vs 1.18 / 1.23 / 1.22, explore efficiency
   0.927 / 0.907 vs 0.931 / 0.923, d=10 tail **6 of 144 vs 6 of 144**. That is
   **4.8× fewer episodes and env-steps** for the checkpoint the wave-1
   write-up itself called "the cheaper alternative" to u725.
2. **Against u725 it matches every row but the tail, and the tail is 2.0×
   (6 vs 3)** — the exact boundary the wave-1 caveat draws for probe noise
   ("a tail difference smaller than ~2× should not be read as real"). Exploit
   is equal or better on every level (× optimal 1.14 / 1.25 / 1.17 vs
   1.16 / 1.26 / 1.18); explore efficiency within 0.015 / 0.024.
   **5.8× fewer episodes and env-steps.**
3. **d0_base's own tail trajectory is 0.049 → 0.042 → 0.021 over u500 →
   u600 → u725 (128 → 154 → 186 M env-steps).** The small-pool arms' 5–8% at
   ~100k is the same stage of the same curve; its last factor of two cost
   d0_base 160,000 episodes. `lr1` u1000 is on that curve at u1000; where it
   sits at u1500–2000 (240–320k) is what the running arm will show.
4. `lr1` u800 has the **best exploit half of anything probed** (× optimal
   1.14 / 1.16 / 1.18, `align_true` 0.92 / 0.91 / 0.90 — better than d0_base
   u725 on every level) and a bad explore checkpoint (tail 20). The two
   halves do not peak together within one run any more than they did in
   d0_base (§9.5 of `DUAL_TRAINING`); select by the joint probe, not either
   half.

**Headline, as of round 4:** the d0_base recipe with `batch_envs` 8 instead of
64 and PPO at lr 1e-4 × 10 epochs × 8 minibatches (`target_kl` 0.1) reaches
**d0_base u600 on every probe metric at 160k episodes (4.8×)** and
**d0_base u725 on every metric but a 2×-within-noise collapsed tail at the
same 160k (5.8×)**; the exploit half alone is matched at 80–92k (10×, or 20×
on env-steps with 100-step rollouts).

### 5.5 Round 5 (job 22748125): the FULL pass — `lr1` u1500 at 240k episodes

Ten checkpoints in one process; this run's d0_base u725 tail read 4 of 144.

**Exploit** (d = 0 / 5 / 10):

| checkpoint | episodes / env-steps | success | steps | × opt | `align_true` |
|---|---|---|---|---|---|
| lr1 u1250 | 200k / 40M | 1.000 / 1.000 / 1.000 | 12.10 / 13.42 / 12.96 | 1.20 / 1.26 / 1.26 | 0.89 / 0.85 / 0.85 |
| **lr1 u1500** | **240k / 48M** | 1.000 / 0.990 / 0.995 | 12.07 / 12.42 / 12.38 | **1.20 / 1.18 / 1.22** | 0.88 / 0.90 / 0.87 |
| lr1 u2000 | 320k / 64M | 1.000 / 1.000 / 1.000 | 11.70 / 13.05 / 12.20 | 1.15 / 1.22 / 1.18 | 0.92 / 0.88 / 0.89 |
| lr1 u2500 | 400k / 80M | 1.000 / 0.995 / 1.000 | 12.26 / 13.42 / 12.91 | 1.19 / 1.24 / 1.23 | 0.89 / 0.86 / 0.86 |
| e75 u800 | 128k / 25.6M | 1.000 / 1.000 / 1.000 | 12.70 / 13.54 / 13.83 | 1.24 / 1.26 / 1.33 | 0.85 / 0.84 / 0.80 |
| e75 u1200 | 192k / 38.4M | 1.000 / 0.990 / 0.995 | 12.58 / 14.03 / 14.85 | 1.22 / 1.30 / **1.42** | 0.87 / 0.83 / **0.76** |
| e75 u1450 | 232k / 46.4M | 1.000 / 0.995 / 1.000 | 11.98 / 14.21 / 12.82 | 1.17 / 1.33 / 1.23 | 0.90 / 0.80 / 0.86 |
| d5e75 u800 | 128k / 25.6M | 1.000 / 1.000 / 0.990 | 15.94 / 17.36 / 17.53 | 1.57 / 1.63 / 1.71 | 0.67 / 0.65 / 0.63 |
| d5e75 u1250 | 200k / 40M | 1.000 / 0.995 / 1.000 | 13.98 / 15.21 / 14.80 | 1.36 / 1.41 / 1.42 | 0.78 / 0.75 / 0.74 |
| d0_base u725 | 928k / 185.6M | 1.000 / 0.990 / 1.000 | 11.71 / 12.30 / 12.35 | 1.16 / 1.17 / 1.21 | 0.91 / 0.90 / 0.88 |

**Explore**:

| checkpoint | d=0 swept / eff / tail | d=10 swept / eff / **tail** (n of 144) / chase_t |
|---|---|---|
| lr1 u1250 | 0.607 / 0.942 / 0.000 | 0.591 / 0.945 / 0.021 (3) / 0.77 |
| **lr1 u1500** | **0.616 / 0.969 / 0.000** | **0.601 / 0.961 / 0.035 (5)** / 0.68 |
| lr1 u2000 | 0.617 / 0.968 / 0.000 | 0.597 / 0.961 / 0.028 (4) / 0.60 |
| lr1 u2500 | 0.610 / 0.975 / 0.000 | 0.599 / 0.950 / **0.007 (1)** / 0.74 |
| e75 u800 | 0.549 / 0.877 / 0.021 | 0.518 / 0.837 / 0.062 (9) / 0.42 |
| e75 u1200 | 0.621 / 0.971 / 0.000 | 0.605 / 0.958 / **0.000 (0)** / — |
| e75 u1450 | 0.595 / 0.938 / 0.000 | 0.577 / 0.928 / 0.035 (5) / 0.28 |
| d5e75 u800 | 0.568 / 0.909 / 0.028 | 0.552 / 0.882 / 0.062 (9) / 0.22 |
| d5e75 u1250 | 0.597 / 0.954 / 0.000 | 0.584 / 0.921 / 0.042 (6) / 0.39 |
| d0_base u725 | 0.609 / 0.953 / 0.000 | 0.593 / 0.945 / 0.028 (4) / 0.41 |

1. **`se_b8_lr1` u1500 — 240,000 episodes / 48.0 M env-steps — passes every
   row of §1 within noise against d0_base u725: 3.9× fewer episodes and
   env-steps.** Success 1.000 / 0.990 / 0.995 vs 1.000 / 0.990 / 1.000;
   × optimal 1.20 / 1.18 / 1.22 vs 1.16 / 1.17 / 1.21 (within 0.04);
   explore efficiency **0.969 / 0.961 against 0.953 / 0.945** — better at
   both levels; d=10 collapsed tail 5 vs 4 of 144. u2000 (320k, 2.9×)
   passes with margin on every row and u2500's tail is 1 of 144 — the
   tail keeps closing along d0_base's own curve (§5.4 point 3).
2. **u1250 (200k, 4.6×) passes the whole explore half** — tail 3 of 144,
   efficiency 0.942 / 0.945 — and misses exploit only on directness at
   d = 5 / 10 (× optimal 1.26 / 1.26 vs 1.17 / 1.21, past the 0.03 rule).
   Between u1000 (§5.4: every row but a 2× tail) and u1500 the two halves
   take turns; the first checkpoint on the 250-update probe grid that holds
   both is u1500.
3. **The explore-heavy mix closes the tail and pays for it on exploit.**
   `e75` u1200 has a **zero** tail and the best explore of anything probed
   (0.971 / 0.958) at 192k episodes, and × optimal 1.42 at d = 10 with
   `align_true` 0.76 — with 5 exploit envs per update the directness never
   converges. `d5e75` is worse still (1.4–1.7×). Tri finding 13's frontier
   again: `empty_frac` trades the halves rather than buying both. A mix
   *schedule* — 0.5 until the exploit lock, 0.75 after — is the obvious
   follow-up and was not run.
4. The tail as read here for d0_base u725 is 4 of 144 (0.028) against 2–3
   in rounds 2–4: the ten-checkpoint list shifts the sampled stream, as the
   wave-1 caveat says. Rank tails within a round only.

**Final headline.** The d0_base recipe with `batch_envs` 8 instead of 64 and
PPO at lr 1e-4 × 10 epochs × 8 minibatches (`target_kl` 0.1) — nothing else
changed — reaches **d0_base u725 on every held-out probe metric at 240k
episodes / 48M env-steps (3.9× fewer)**, reaches **d0_base u600 on every
metric at 160k (4.8× vs u600; 5.8× vs u725 with a 2×-within-noise tail)**,
and matches the **exploit half alone at 80–92k (10×; 20× on env-steps with
100-step rollouts)**. The residual factor between 160k and 240k is the
d=10 collapsed tail, the slowest-converging quantity for d0_base too. The
delivered checkpoint is
`agent_ckpts/navigate_navp2_se_b8_lr1_s42_22701298/navigate_u1500.pt`
(u2000 for margin; u1000 for the 5.8× near-pass).

**Wall-clock caveat, stated because it is the opposite sign to the sample
result:** an update's cost is the serial rollout loop (envs × T × ~8 ms),
independent of batch, so `lr1` u1500 took ~11 h on one a100 against
d0_base's 6 h on an H200 for u725. Fewer samples cost more time here. The
lever for both at once is a multi-env `VecEnv` (batch across envs, not
within one), which was not attempted.

## 6. Accounting notes

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
