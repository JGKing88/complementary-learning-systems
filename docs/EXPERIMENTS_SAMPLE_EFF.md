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
| **RESULT** | **`se_b8_lr1` — d0_base with `batch_envs` 64→8 and PPO lr 1e-4 × 10 epochs × 8 minibatches (`target_kl` 0.1) — is on its plateau on every mean-level metric by u600–1000 = 96–160k episodes, 6–10× below d0_base's 928k (curves page [15cc014f](https://claude.ai/code/artifact/15cc014f-fc15-4432-9f7f-8f7e41be1bc7)). On the held-out probe u1000 (160k / 32M env-steps, 5.8×) matches d0_base u725 on every row but the d=10 collapsed tail, 6 vs 3 of 144 — a rare-event count the probe cannot resolve below ~2× (the same d0_base checkpoint read 2–4 across rounds). u1500 (240k, 3.9×) passes even that. Exploit half alone at 80–92k (10×). Delivered: `agent_ckpts/navigate_navp2_se_b8_lr1_s42_22701298/navigate_u1000.pt` (u1500 if the tail count must also match). |
| **Curves page** | [15cc014f](https://claude.ai/code/artifact/15cc014f-fc15-4432-9f7f-8f7e41be1bc7) — Part I: `se_b8_lr1` and `d0_base`, deterministic and sampled, by update / episodes / env-steps. Part II (v4): one-arena overlays and rows (`one_k4_g` s43 held-out det/sampled + training arena det/sampled, `one_k4_g_e75`, `one_k2_b16_g`, the fixed-goal `one_k2` held-out and on its own arena), both probe rounds, the fixed-goal `follow_q` finding. Sources `results/nav_tri_probe/*_training_curve.png`; sampled rows from `reeval_series.py`. |
| **ONE ARENA (§7, 2026-09-14/15)** | `envs_per_world` 1 with `--env_repeats K` (the one env collected K times per update so both regimes share the PPO update) and `--redraw_goal_per_rollout` (each env otherwise holds ONE goal for the run — the strict one-goal reading learns a position map, never follows `q`, and is coin-flip on new arenas; §7.10). **Exploit transfers from one arena at d0_base level in every redraw arm; explore needs explore data** (held-out coverage orders by explore trajectories/update: 32 → 0.25, 64 → 0.35, 128 → 0.5, 192 → 0.55). Training-eval screen first cleared at **221–250k episodes by the 3:1 K=4 arms** (d0_base 800k). Held-out probe (§7.8, §7.11): the 1:1 K=4 arm (`one_k4_g` s43 u1900, 486k) has d0_base's exploit — × optimal 1.17/1.17/1.23 vs 1.20/1.16/1.28 — and d=0 explore within 0.05, with a d=10 collapsed tail 21 vs 5 of 144 (distractor capture on unseen offsets); the 3:1 arm (`one_k4_g_e75` s42 u2150, 550k) matches explore completely (tail 5 vs 5) and pays in directness (1.61 at d=10). No single checkpoint passes every row yet; the mix schedule (`one_k4_g_sched`, 1:1 → 3:1 at u400) is a hair off the screen at 307k and resumes after maintenance. Delivered: `agent_ckpts/navigate_navp2_one_k4_g_s43_22757447/navigate_u1900.pt`; explore-complete alternative `..._one_k4_g_e75_s42_22763086/navigate_u2150.pt`. Page v4 Part II. |
| **Tools** | `analysis/nav_tri/compare_curves.py` (several runs on one 3-panel figure); `reeval_series --which train` (score the training arena); `analysis/nav_tri/sample_eff_curve.py` (eval series vs cumulative samples, window means, first-clear of the screen); the wave-1 probe pipeline `hopfield_nav/run_wave1_final.sh` pattern for the verdict. Trainer now logs exact `episodes` / realized `env_steps` per update and per eval, and writes both into every checkpoint. |

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

**Final headline (revised 2026-09-14 after the curves, Jack: "se_b8_lr1
honestly stops improving at like 1000 updates").** The d0_base recipe with
`batch_envs` 8 instead of 64 and PPO at lr 1e-4 × 10 epochs × 8 minibatches
(`target_kl` 0.1) — nothing else changed — is on its plateau on every
mean-level metric (success, steps, × optimal, explore efficiency at d=0 and
d=10, on the training eval and on the probe) by **u600–1000 = 96–160k
episodes, 6–10× below d0_base's 928k**, and is flat from there to 416k. On
the probe, **u1000 (160k / 32M env-steps, 5.8×) matches d0_base u725 on every
row but the d=10 collapsed tail** — 6 vs 3 of 144, a rare-event count the
probe cannot resolve below ~2× (d0_base u725 itself read 2, 3 and 4 across
rounds; Fisher p ≈ 0.5). **u1500 (240k, 3.9×) passes even that row** (5 vs 4),
and the exploit half alone is matched at 80–92k (10×; 20× on env-steps with
100-step rollouts). The 3.9× in §5.5 was the pre-registered "tail ≤ 2×" rule
applied past what the instrument supports; **5.8× at 160k is the defensible
headline**, 3.9× the reading where even the tail count is within 1.25×.
Delivered checkpoint:
`agent_ckpts/navigate_navp2_se_b8_lr1_s42_22701298/navigate_u1000.pt`
(u1500 if the tail count must also match; u2000 for margin).

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

---

## 7. ONE env — the best model a single training env can give

Started 2026-09-14 16:00. Jack: *"get a model trained on just one environment
to be as good as possible. then add to that page ... if you get a model
trained on one env as good as those d0, then see how few samples you can
train with."*

### 7.1 What "one env" is, and what had to change

An env is a wall barcode (the ±1 code the 60 raycasts read, drawn from the
env's seed), a scaffold offset (which place codes the agent sees), and a
goal drawn per exploit rollout from its 20×20 cells. Distractors come from
the scaffold OUTSIDE the env's rectangle, so they are as diverse at 1 env as
at 20. Val is the run's own 6 held-out envs as always — the curve is
**generalization from one env**; `reeval_series --which train` scores the
training env itself.

The trainer assigns the explore/exploit regime **per env**: at
`empty_frac=0.5`, `round(1 × 0.5) = 0` explore slots — a one-env run would
have been exploit-only every update, and any rounding fix would still put
one regime per update instead of both in the same PPO update (the tri line's
whole point). **`--env_repeats K`** (commit 6853202): each train env is
collected K times per update, the regime draw (`shuffle`) is over envs × K
slots, and the K rollouts of the one env — some exploit with a fresh goal
memory + distractors, some explore with an empty memory — land in one PPO
update exactly as d0_base's 20 envs do. K=1 is the historical loop
byte-for-byte.

Cost model: the rollout loop is serial in slots (K × 200 steps × ~8 ms),
batch-independent, so a one-env update at K=4, batch 64 is ~6.4 s of
rollouts + PPO for 256 episodes — about d0_base's 20 s/update ÷ 3.

### 7.2 Wave 1 (2026-09-14 16:05; ou_bcs_normal, 7.5 h wall — the monthly
maintenance blocks 09-15 00:00–21:00, so these TIMEOUT at ~u2500–4000 and
resume with `--continue_from` if still climbing)

All: `ENVS_PER_WORLD=1`, d0_base recipe, `interleave:4000,empty_frac=0.5`,
eval/ckpt every 25, the SE optimizer (lr 1e-4 × 10 epochs × 8 minibatches,
`target_kl` 0.1) unless noted.

| arm | seed | K | batch | eps/update | job |
|---|---|---|---|---|---|
| `one_k4` | 42 | 4 (2+2) | 64 | 256 | 22756994 |
| `one_k2` | 42 | 2 (1+1) | 64 | 128 | 22756995 |
| `one_k8_b32` | 42 | 8 (4+4) | 32 | 256 | 22756996 |
| `one_k2_lr3` | 42 | 2 | 64 | 128 | 22756997 — d0_base's optimizer (3e-4, 4×4) as the control |
| `one_k4` | 43 | 4 | 64 | 256 | 22757000 — a different single env: how much is env identity |

Smoke (6 updates, 40-step rollouts): 22756982.

**16:05 — one env has ONE goal.** `VecEnv` shares `base_env._goal` and every
reset keeps it; `reset_goal` is only ever called by the legacy `train.py`.
So d0_base's 20 envs were 20 goal cells, and the exploit half of a one-env
run sees a single cell (s42's is [17, 0], on the edge; s43's [11, 6]). The
evaluator has the same convention — one goal per val env, trials vary the
start — so the wave-1 arms are the strict reading: one arena, one goal.
Added `--redraw_goal_per_rollout` (commit after 6853202): a fresh goal cell
from the env's own RNG before every rollout slot, both regimes. Same arena,
goals everywhere in it — the "as good as possible" reading. Cancelled
`one_k8_b32` (22756996) and `one_k2_lr3` (22756997) at u25 — second-order
questions the SE line already answered — and `se_lr1_d5_e75` (22715547,
past u2700, verdict taken at u1200) to make room:

| arm | seed | K | batch | eps/update | job |
|---|---|---|---|---|---|
| `one_k4_g` | 42 | 4 | 64 | 256 | see launch record below |
| `one_k4_g` | 43 | 4 | 64 | 256 | |
| `one_k2_g` | 42 | 2 | 64 | 128 | |

First evals (u25, 6 held-out envs, deterministic): `one_k4` s42 success
0.64/0.59 (d=0/10), swept 0.18; `one_k2` 0.32/0.17; the 3e-4 control 0.82/0.72
(the usual fast start of the large step). Timing: K=4 8.2 s/update (7.4
rollout + 0.8 PPO), eval 23.5 s per 25 → ~3000 updates before the wall.

### 7.3 Digest at 16:38 (~35 min; window means of the last 4 evals, held-out, deterministic)

| arm | u | episodes | succ 0/10 | steps 0/10 | swept 0/10 |
|---|---|---|---|---|---|
| `one_k2` (fixed goal) | 425 | 54k | 0.84/0.84 | 46/52 | 0.28/0.31 |
| `one_k4` (fixed) s42 | 250 | 64k | 0.70/0.69 | 38/38 | 0.25/0.29 |
| `one_k4` (fixed) s43 | 125 | 32k | 0.74/0.72 | 57/50 | 0.17/0.18 |
| **`one_k2_g`** (redraw) | 375 | 48k | **1.00/0.99** | **15.1/17.2** | 0.24/0.24 |
| `one_k4_g` s43 | 200 | 51k | 0.98/0.97 | 29/33 | 0.21/0.21 |
| `one_k4_g` s42 | 175 | 45k | 0.41/0.32 | 67/62 | 0.17/0.16 |

Fixed-goal arms peak EARLY on held-out (`one_k2` 0.99/0.99 at u100,
`one_k4` s43 0.90/0.88 at u25) and slide to ~0.8 with mean steps 40–50 as
the policy specializes to its one goal cell — the strict reading is a
worse model the longer it trains. The goal-redraw arm is at d0_base-level
exploit by u300–375 (38–48k episodes; d0_base u725: 11.7/12.1 steps). Explore
is the laggard (swept 0.24 vs the 0.55/0.50 bar; d0_base was ~0.25 at u200,
0.5 at u400–500). Seed spread is large — `one_k4_g` s42 vs s43 — the price
of one arena. Same env for both s42 arms (seed → wall 6423388, val set), so
K=2 vs K=4 there is an init/draw difference, not an env difference.

**Action (16:39):** the strict reading is answered by `one_k2` s42 +
`one_k4` s43; cancelled `one_k4` s42 (22756994) and the finished
`se_lr1_e75` (22715546, u2800+) to launch two sample-lean redraw arms while
the pre-maintenance window is open — Jack's second ask, on the reading that
works: `one_k2_b32_g` (64 traj/update) and `one_k2_b16_g` (32 traj/update),
both s42, 7 h wall. The SE line put the usable pool floor at ~80–160
trajectories per update with 20 envs; these ask where it is with one.

### 7.4 16:55 — the one-env failure mode is EXPLORE, and it is a landmark map

`one_k2_g` held-out swept coverage by eval: u100 0.40, u200 0.31, u300 0.17,
u400 0.37, u500 0.15, u600 0.10 — sliding while its exploit holds at
d0_base level (u600: 1.00/0.99, 14.5/16.2 steps). The two halves read
different inputs: exploit follows `q`, the local-chart displacement from the
Hopfield recall, which is the same object in every env; explore has only
the 60-ray wall code and its own path integration (`prev_disp`), and with
ONE barcode the wall code is a landmark map — every view names a position —
so the cheapest explore policy is a memorised sweep keyed on views that no
held-out env has. Twenty barcodes forced d0_base onto the barcode-agnostic
cue (run-length structure of the rays = distance to wall); one does not.

Two checks queued: CPU re-eval 22761980 (held-out SAMPLED — explore's
convention — and the TRAIN env, det + sampled, u100–600) — if the train env
scores ~0.55 while held-out slides, the diagnosis stands.

Remedy within one env: `--obs_dropout p` (commit after ae1bd36) — training-
only dropout on the sensory channel, per entry per step; eval is clean. The
surviving rays keep the run-length cue; the landmark identity gets noisy.
Cancelled `one_k4` s43 (22757000; strict reading covered by `one_k2` s42,
same slide) and `one_k4_g` s42 (22757441; K=4 covered by s43) for
`one_k2_g_od3` and `one_k2_g_od5` (K=2, batch 64, redraw, p = 0.3 / 0.5),
6 h 50 wall.

Also at 16:55: `one_k2_b32_g` (64 traj/update) collapsed at u200 to success
0.09/0.06 from 0.51/0.43 at u100 — the SE line's pool floor (~80–160
traj/update) again, or a transient; `one_k2_b16_g` u100 0.66/0.74.

### 7.5 17:20 — the re-eval (CPU job 22762054): a generalization gap AND an oscillating explore half

`one_k2_g` s42, swept coverage d=0 / d=10 by checkpoint:

| u | held-out det (trainer) | held-out sampled | TRAIN env det | exploit (held-out det) |
|---|---|---|---|---|
| 100 | 0.40 | 0.48/0.43 | 0.44/0.43 | 0.98/0.95, 36 steps |
| 200 | 0.31 | 0.42/0.41 | **0.55/0.50** | 1.00/0.98, 19.5 |
| 300 | 0.17 | 0.35/0.37 | **0.52/0.47** | 1.00/0.99, 15.9/17.0 |
| 400 | 0.37 | 0.46/0.40 | 0.51/0.42 | 1.00/1.00, 13.8/17.1 |
| 500 | 0.15 | 0.19/0.18 | 0.26/0.17 | 1.00/0.98, 13.8/15.1 |
| 600 | 0.10 | 0.13/0.26 | 0.29/0.32 | 1.00/0.99, 14.5/16.2 |
| 700–900 | 0.28, 0.33, 0.20 | | | 1.00/0.99, 12.7–13.0 / 14.2–14.7 |

Two effects, both real. (1) The training arena scores explore ~0.5 where
held-out scores 0.2–0.3 at the same checkpoint (u200–400): a landmark
component — the explore policy is partly a view-keyed sweep of its one
barcode. (2) Explore also DIPS on the training arena (u500: 0.52 → 0.26),
and the explore half's own training reward dips with it (100-update means
of `emp=`: 0.35 → 0.28 → **0.20** → 0.32 → 0.34 → **0.23** → 0.32 over
u301–1000) while exploit's `pre=` sits flat at 0.24–0.28. The explore half
oscillates; it has 64 trajectories per update against d0_base's 640.
Exploit is d0_base-level from u300 (bar 13/14 steps: u700 12.96/14.7,
u800 12.7/14.2, u900 12.9/14.4) and stays there.

`one_k2_g_od5` (obs dropout 0.5) is DEAD: reward flat at 0.05 from u50,
`approx_kl=0.000 clip_frac=0.000` — a fixed point PPO cannot leave.
`od3` learns, slower (u100 0.49/0.48 vs 0.98/0.95 clean). Cancelled od5
(22762040) and `one_k2` fixed-goal (22756995; u900: exploit 0.52/0.50,
explore 0.48 — the strict reading's shape is on record) for
**`one_k4_g_e75`** on s42 and s43: K=4 with `empty_frac` 0.75 = 3 explore +
1 exploit slots, 192 explore trajectories per update. 6 h 30 wall.

Sample-lean redraw arms at 17:20: `b32_g` u500 (32k episodes) 1.00/0.98,
14.0/16.2 steps, swept 0.46; `b16_g` u400 (12.8k) 1.00/0.99, 17.6/18.2,
swept 0.29. Neither is below the pool floor on one env.

### 7.6 18:30 — explore follows explore DATA per update; the 3:1 mix is the arm

| arm | u | episodes | succ 0/10 | steps 0/10 | swept 0/10 |
|---|---|---|---|---|---|
| **`one_k4_g_e75` s42** | 675 | 173k | 1.00/0.98 | 14.0/16.4 | **0.55/0.52** |
| `one_k4_g_e75` s43 | 400 | 102k | 1.00/0.98 | 17.5/18.6 | 0.46/0.43 |
| `one_k4_g` s43 (1:1) | 1000 | 256k | 1.00/0.99 | **11.3/13.6** | 0.51/0.49 |
| `one_k2_g` (1:1) | 1875 | 240k | 1.00/0.99 | 13.6/15.1 | 0.32/0.23 |
| `one_k2_b32_g` | 1400 | 90k | 1.00/1.00 | 12.6/13.5 | 0.26/0.26 |
| `one_k2_b16_g` | 1400 | 45k | 1.00/1.00 | 13.9/15.3 | 0.29/0.24 |
| `one_k2_g_od3` | 525 | 67k | 1.00/1.00 | 13.6/15.0 | 0.12/0.23 |

Ordering by explore trajectories per update — 32 (b16), 64 (k2, b32),
128 (k4 1:1), 192 (e75) — is the ordering of the explore window: 0.25,
0.25–0.35, 0.51, 0.55. The exploit half is indifferent (every redraw arm
reaches ≤13/14 steps, the 32-trajectory pool included). Obs dropout 0.3
did nothing for explore (0.12/0.23 at u525): the landmark story was not
the binding one — the trunk interference / explore pool size was.

The overlay (`results/nav_tri_probe/one_vs_d0_by_episodes.png`) shows the
mirror image on the fixed-goal arm: its explore generalizes (0.45–0.48 at
u700–900) while its exploit collapses. Explore rollouts are built the same
way in both, so the difference is what the exploit half does to the shared
trunk: a strong general follow-`q` captures explore; a goal-specific one
leaves it alone (and does not transfer).

`e75` s42 clears the explore bar at 173k with exploit two steps short and
falling; `k4_g` s43 has the exploit bar and is one point short on explore.
Both have ~5 h to the wall. Cancelled `od3` (22762039) and the plateaued
`b32_g` (22761191) for two sample-lean 3:1 arms: `one_k4_b16_g_e75` (64
episodes/update, 48 explore) and `one_k4_b32_g_e75` (128, 96), 5 h 20 wall.

### 7.7 19:40 — probe round 1 and the page rows

Digest at 19:03: `one_k4_g` s43 u1225 (314k) **11.4/12.6 steps, 1.00/0.99,
swept 0.53/0.49** — d0_base's own eval at u725 read 11.7/12.1 and
~0.55–0.6: a d0_base-class checkpoint from one arena at 3× fewer episodes.
`e75` s42 u800 13.7/16.3, swept 0.47/0.43 (its u675 window 0.55/0.52 was a
peak, not a plateau); `one_k2_g` u2325 13.0/14.6, 0.45/0.38 — explore
climbs slowly with updates in every arm; `b16_g` u2025 (65k) 12.8/13.8,
0.34/0.31.

Verdict probe round 1 (job 22767338, `run_se_probe.sh`, d0_base u725 in
process): `one_k4_g_s43` u1200 + u1450, `one_k4_g_e75_s42` u1250,
`one_k4_g_e75_s43` u1000, `one_k2_g` u2800, `one_k2_b16_g` u2500. CPU
re-evals for the page (22767322: `one_k4_g_s43` held-out sampled + train
arena det/sampled to u1300; 22767323: the fixed-goal arm's train arena).
All must land before the 00:00 maintenance; the arms TIMEOUT at ~23:50
with checkpoints, and `--continue_from` resumes tomorrow if anything is
still climbing.

Other sessions' GPU jobs (`cc_scatter*`, `corner_check`) now share the
partition; `se_b8_lr1` (22701298) reached its 24 h wall at 20:01.

### 7.8 Probe round 1 (job 22767338, 20:00): one arena = d0_base on every row but the d=10 explore tail

Seven checkpoints in one process; this run's d0_base u725 tail read 6 of 144.

**Exploit** (d = 0 / 5 / 10):

| checkpoint | episodes | success | steps | × opt | `align_true` |
|---|---|---|---|---|---|
| **k4_g s43 u1450** | **371k** | 1.000 / 1.000 / 1.000 | 11.59 / 12.47 / 12.31 | **1.22 / 1.19 / 1.26** | 0.93 / 0.94 / 0.91 |
| k4_g s43 u1200 | 307k | 1.000 / 1.000 / 0.995 | 11.52 / 12.47 / 12.54 | 1.21 / 1.19 / 1.29 | 0.94 / 0.94 / 0.82 |
| e75 s42 u1250 | 320k | 1.000 / 1.000 / 1.000 | 12.65 / 13.35 / 13.93 | 1.33 / 1.28 / **1.43** | 0.83 / 0.86 / 0.78 |
| e75 s43 u1000 | 256k | 1.000 / 1.000 / 1.000 | 12.82 / 13.37 / 13.27 | 1.35 / 1.28 / 1.36 | 0.84 / 0.88 / 0.83 |
| k2_g u2800 | 358k | 1.000 / 1.000 / 1.000 | 11.83 / 12.98 / 12.96 | 1.24 / 1.24 / 1.33 | 0.95 / 0.95 / 0.89 |
| b16_g u2500 | 80k | 1.000 / 1.000 / 0.995 | 11.61 / 12.90 / 13.43 | 1.22 / 1.23 / 1.37 | 0.95 / 0.94 / 0.79 |
| d0_base u725 | 928k | 1.000 / 1.000 / 1.000 | 11.40 / 12.14 / 12.48 | 1.20 / 1.16 / 1.28 | 0.92 / 0.93 / 0.86 |

**Explore** (144 sampled trials, held-out):

| checkpoint | d=0 swept / eff / tail | d=10 swept / eff / **tail** (n) / chase_t |
|---|---|---|
| **k4_g s43 u1450** | **0.596 / 0.951 / 0.000** | 0.514 / 0.879 / **0.132 (19)** / 0.67 |
| k4_g s43 u1200 | 0.535 / 0.854 / 0.000 | 0.481 / 0.824 / 0.160 (23) / 0.64 |
| e75 s42 u1250 | 0.593 / 0.935 / 0.007 | 0.541 / 0.870 / 0.097 (14) / 0.64 |
| e75 s43 u1000 | 0.497 / 0.810 / 0.056 | 0.465 / 0.804 / 0.139 (20) / 0.59 |
| k2_g u2800 | 0.518 / 0.876 / 0.014 | 0.464 / 0.830 / 0.153 (22) / 0.76 |
| b16_g u2500 | 0.269 / 0.504 / 0.493 | 0.134 / 0.356 / 0.840 (121) / 0.68 |
| d0_base u725 | 0.606 / 0.944 / 0.000 | 0.577 / 0.922 / **0.042 (6)** / 0.60 |

1. **`one_k4_g` s43 u1450 — 371k episodes, one arena — equals d0_base u725
   on every §1 row except the d=10 explore collapsed tail**: success 1.000
   everywhere, × optimal within 0.03 at all three levels, explore
   efficiency at d=0 0.951 vs 0.944, tail 0 vs 0. At d=10 its tail is 19
   of 144 vs 6 (efficiency 0.879 vs 0.922), and the collapsed trials chase
   `q` — distractor capture: the ||q|| gate learned at one offset transfers
   imperfectly to arenas at other offsets. 2.5× fewer episodes than
   d0_base for everything but that row.
2. The 3:1 mix buys the smaller one-env tail (14) at the usual exploit
   price (× optimal 1.43 at d=10) — the same frontier as §5.5 point 3.
3. **Explore needs the data on one arena.** `b16_g` (32 episodes/update) at
   80k has exploit at d0_base level but its SAMPLED explore is 0.27 / 0.13
   swept (deterministic eval read 0.34 / 0.31 — the one case here where
   sampling hurt), tail 121 of 144 at d=10. One arena's exploit is cheap;
   its explore is not.
4. Round 2 at ~22:30 on the last pre-maintenance checkpoints (k4_g s43
   ~u2400, e75 s42 ~u2000, the sample-lean 3:1 arms) asks whether the
   tail closes with updates as d0_base's did (§5.4 point 3).

**Page v3 (20:50):** [15cc014f](https://claude.ai/code/artifact/15cc014f-fc15-4432-9f7f-8f7e41be1bc7)
Part II — overlays `one_vs_d0_by_{episodes,update}.png`, per-run rows for
`one_k4_g` s43 (held-out det / sampled; TRAINING arena det / sampled),
`one_k4_g_e75` s42, `one_k2_b16_g`, and the fixed-goal `one_k2` (held-out
and its own arena). Re-eval logs `results/nav_tri_probe/reeval_one_k4_g_s43_{stoch,train,train_stoch}.log`,
`reeval_one_k2_s42_fixed_{stoch,train,train_stoch}.log`. Training arena for
`one_k4_g` s43: exploit 8.8–9.0 steps from u800 (its own goal), swept
0.55–0.60 from u1000 — the level held-out reaches sampled.

### 7.9 21:20 — the §2 screen on one arena: first clear at 221–250k episodes

`sample_eff_curve` window means (4 evals), first window meeting succ ≥0.99,
steps ≤13/14, swept ≥0.55/0.50:

| arm | eps/update | first clear | episodes | latest window |
|---|---|---|---|---|
| `one_k4_g_e75` s43 | 256 | u975 | **250k** | u1825: 12.9/13.5 · 0.56/0.54 |
| **`one_k4_b32_g_e75`** | 128 | u1725 | **221k** | u1775: 12.6/13.9 · 0.51/0.49 |
| `one_k4_g_e75` s42 | 256 | u1775 | 454k | u2175: 12.5/16.3 · 0.57/0.55 |
| `one_k4_g` s43 (1:1) | 256 | — (swept0 0.54) | — | u1800: 11.1/13.3 · 0.54/0.51 |
| `one_k4_b16_g_e75` | 64 | — (steps0 14.3) | — | swept 0.55–0.58 at 77–115k; dip at u2000 |
| `one_k2_g` (COMPLETED u4000) | 128 | — | 512k | 12.5/14.1 · 0.39/0.36 |
| `one_k2_b16_g` (COMPLETED u4000) | 32 | — | 128k | 12.3/14.2 · 0.33/0.18 |

d0_base first cleared this screen at u625 = 800k (§0), `se_b8_lr1` at
~100k (§4.3). On one arena the 3:1 K=4 recipe clears it at 221–250k —
3.2–3.6× below d0_base — and the 1:1 K=4 arm sits a point under the
swept0 line with the best exploit of anything. Explore trajectories per
update is the currency: 64/update (`b16_g_e75`) has the coverage at
77k but not the directness; 32/update (`b16_g`, 1:1) never gets the
coverage in 4000 updates.

Probe round 2 (job 22772411): `k4_g` s43 u1900, `e75` s42 u2150, `e75`
s43 u1825 + u1000 (its first clear), `b32_g_e75` u1750, `b16_g_e75` u1800,
`k2_g` u4000 — vs d0_base u725.

### 7.10 Why the fixed-goal exploit fails on new arenas — it never follows `q` (job 22772812, 21:55)

`behavior_probe --mode nav`, held-out arenas, 32 trials × 6, d = 0 / 10:

| checkpoint | success | steps | **follow_q** | align_true | follow_q t0 / t1 / t2 / t6+ |
|---|---|---|---|---|---|
| fixed u100 | 0.92 / 0.91 | 38 / 43 | **0.13 / 0.09** | 0.12 / 0.08 | 0.34 / 0.36 / 0.48 / 0.09 |
| fixed u500 | 0.96 / 0.96 | 44 / 45 | 0.21 / 0.16 | 0.19 / 0.14 | 0.46 / 0.37 / 0.42 / 0.16 |
| fixed u900 | 0.83 / 0.84 | 59 / 59 | **0.05 / 0.09** | 0.04 / 0.09 | 0.76 / 0.69 / 0.53 / 0.00 |
| redraw u900 | 1.00 / 1.00 | 12 / 13 | **0.90 / 0.87** | 0.90 / 0.87 | 0.66 / 0.82 / 0.91 / 0.92 |

With one goal cell, "position → heading to (17, 0)" — readable from the
one barcode and from path integration — fits every training rollout at
least as well as "follow `q`", without `q`'s distractor noise, and it is
what PPO learns from the start: on new arenas the fixed-goal policy's
actions are uncorrelated with the recall at every checkpoint (`follow_q`
0.05–0.2 vs 0.9 for the redraw policy), its successes are search (40–60
steps; u900's 0.5 success in the trainer's eval ≈ its explore coverage),
and on failures it moves away from the goal (`align_true_fail` −0.1 to
−0.4). `q` appears only in the first 2–3 steps of a late-training episode
(0.76 → 0.53 → 0) before wall views arrive and the map takes over. The
fixed goal does not make a `q`-follower that drifts; it never makes one.
Redrawing the goal removes the redundancy — only `q` predicts the target —
and d0_base's 20 goals do the same at 20 envs (the D3 identity-gating note
in `run_nav_p2.sh` is the same failure one level up). The explore-side
corollary — a goal-keyed map leaves the trunk freer for sweeping than a
general follow-`q` — fits the curves but is NOT measured. Output:
`results/nav_tri_probe/fixed_goal_nav_heldout.json`.

### 7.11 Probe round 2 (job 22772411, 21:35): the two halves come from different mixes

d0_base u725 tail read 5 of 144 this run.

| checkpoint | episodes | exploit × opt (d=0/5/10) | success | explore d=0 swept / eff / tail | explore d=10 swept / eff / **tail** (n) |
|---|---|---|---|---|---|
| d0_base u725 | 928k | 1.20 / 1.16 / 1.28 | 1 / 1 / 1 | 0.607 / 0.953 / 0 | 0.578 / 0.925 / **0.035 (5)** |
| **k4_g s43 u1900** (1:1) | 486k | **1.17 / 1.17 / 1.23** | 1 / 1 / 1 | 0.564 / 0.898 / 0 | 0.517 / 0.845 / 0.146 (21) |
| e75 s42 u2150 (3:1) | 550k | 1.39 / 1.30 / **1.61** | 0.995 / 0.984 / 0.984 | 0.615 / 0.957 / 0 | **0.592 / 0.954 / 0.035 (5)** |
| e75 s43 u1825 (3:1) | 467k | 1.30 / 1.30 / 1.43 | 1 / 1 / 1 | 0.599 / 0.957 / 0 | 0.556 / 0.889 / 0.076 (11) |
| e75 s43 u1000 | 256k | 1.35 / 1.28 / 1.36 | 1 / 1 / 1 | 0.486 / 0.800 / 0.083 | 0.452 / 0.762 / 0.194 (28) |
| b32_g_e75 u1750 (3:1, 128/u) | 224k | 1.26 / 1.21 / 1.28 | 1 / 1 / 0.995 | 0.589 / 0.928 / 0 | 0.523 / 0.852 / 0.118 (17) |
| b16_g_e75 u1800 (3:1, 64/u) | 115k | 1.35 / 1.36 / 1.47 | 1 / 1 / 1 | 0.506 / 0.824 / 0.111 | 0.467 / 0.796 / 0.167 (24) |
| k2_g u4000 (1:1) | 512k | 1.19 / 1.23 / 1.28 | 1 / 1 / 1 | 0.568 / 0.924 / 0 | 0.462 / 0.812 / 0.201 (29) |

1. **No single one-arena checkpoint passes every row, and the miss is a
   mix, not a data, question.** The 1:1 K=4 arm has d0_base's exploit
   (× optimal 1.17/1.17/1.23 — better) and d=0 explore within 0.05, with
   a distractor tail of 15% that did NOT close from u1450 (19) to u1900
   (21). The 3:1 arm (s42 u2150) matches d0_base's explore completely —
   tail 5 vs 5, efficiency 0.954 vs 0.925 at d=10 — and pays in directness
   (1.61 at d=10, `align_true` 0.55). Same frontier as §5.5 point 3.
2. The sample-lean 3:1 arm at 224k sits on the frontier too (exploit
   within 0.06, d=0 explore passes, tail 17): explore data per update
   sets how fast the explore half arrives, not which half you get.
3. **Delivered:** `agent_ckpts/navigate_navp2_one_k4_g_s43_22757447/navigate_u1900.pt`
   (486k episodes; exploit ✓, explore d=0 ≈, d=10 tail ✗) and, for the
   explore-complete alternative, `..._one_k4_g_e75_s42_22763086/navigate_u2150.pt`.
4. **Next arm:** a mix SCHEDULE — 1:1 until the exploit lock (~u400), then
   3:1 — or a 2:1 mix (K=3, `empty_frac` 0.67; K=6, 4+2). The SE line's
   untried follow-up, now with the same shape on one arena.

**21:37 — launched on the two freed slots, 2 h 05 wall (TIMEOUT at 23:42,
resume tomorrow with `--continue_from`):** `one_k4_g_sched` (22772984; K=4,
`interleave:400,empty_frac=0.5 ; interleave:3600,empty_frac=0.75` = 2+2
until the exploit lock, then 3+1) and `one_k3_g_e67` (22772985; K=3,
`empty_frac` 0.67 = 2 explore + 1 exploit, 192 episodes/update).

**22:40:** `one_k3_g_e67` (22772985) is DEAD the way `od5` was — movement
std 0.125 → 0.011 by u300, `approx_kl = clip_frac = 0` from there, reward
flat at 0.07 — cancelled at u575; a fixed point of the polar head's
noise, not a result about the 2:1 mix (re-run on another seed). The
schedule arm is healthy: u625 (160k) 13.0/15.4 steps, swept 0.48/0.45
and climbing after its u400 switch to 3:1. `one_k4_g` s43 holds the
screen: u2475 (634k) 11.3/12.9, 0.55/0.50.

### 7.12 End of the pre-maintenance wave (all arms TIMEOUT with checkpoints; 2026-09-15 16:40)

Last 4-eval windows, held-out deterministic:

| arm | last u | episodes | steps 0/10 | swept 0/10 |
|---|---|---|---|---|
| `one_k4_g` s43 (1:1) | 2800 | 717k | 11.2 / 12.8 | 0.54 / 0.53 (on the screen from u2000) |
| `one_k4_g_e75` s42 (3:1) | 3600 | 922k | 12.2 / 14.6 | 0.61 / 0.56 |
| `one_k4_g_e75` s43 (3:1) | 3050 | 781k | 12.3 / 14.2 | 0.59 / 0.56 |
| `one_k4_b32_g_e75` (3:1, 128/u) | 3475 | 445k | 12.9 / 15.5 | 0.54 / 0.52 |
| `one_k4_b16_g_e75` (3:1, 64/u) | 3900 | 250k | 13.4 / 15.6 | 0.52 / 0.50 |
| `one_k4_g_sched` (1:1 → 3:1 at u400) | 1300 | 333k | 12.4 / 14.5 (u1200) | 0.55 / 0.51 (u1200) |

Page v4 published (Part II final curves; overlays now include the schedule
and the 128-episode 3:1 arm). Queued for the 21:00 node release
(`queue_continues.sh`): `--continue_from` for `one_k4_g_sched`,
`one_k4_g` s43, `one_k4_g_e75` s43 and `one_k4_b32_g_e75` (12 h each,
same schedules), then a probe round 3 on their final checkpoints — the
schedule arm is the candidate for both halves in one model.

---

## 8. FIXED goals, few envs — make it generalize (2026-09-15)

Jack: *"when those are done, I'd like you to try to get three envs with no
goal refresh working. then if that works, try harder on one env with no
goal refresh. I'm not sure what this will involve. Maybe a smaller network?
Don't take my word for that. But the point is it has to learn to generalize
somehow, as the issue had been memorization."*

### 8.1 The failure to beat, and the levers

§7.10: with one (env, goal) pair the exploit half learns "position →
heading to the goal cell" off the wall code and path integration
(`follow_q` 0.05–0.2 on new arenas at every checkpoint; the redraw policy
0.9), perfect at home, coin-flip elsewhere. Two things make the map the
cheaper fit: position is readable (one barcode, `prev_disp` integration),
and `q` is the noisier target (0–10 distractor recalls). Levers, each a
regulariser against the map:

- **`h<N>`** — smaller trunk (Jack's guess). The barcode→position lookup
  is the expensive representation; atan2(`q`) is cheap.
- **`xod<p>`** — `--exploit_obs_dropout`: barcode dropout in EXPLOIT
  rollouts only, so the map's input is unreliable exactly when `q` is
  available; explore rollouts keep clean walls for sweeping. (Run-wide
  `obs_dropout` 0.5 killed a run and 0.3 did nothing for explore, §7.5 —
  this is a different placement of the same knob.)
- **`xhd<p>`** — `--exploit_heading_dropout`: the path-integration
  channels (`prev_action`, `prev_displacement`) dropped in exploit only.
- **`nd0`** — no exploit distractors: `q` is clean, so following it fits
  at least as well as the map. (Distractor robustness would then come
  from a curriculum, `N_TRAIN_DISTRACTORS_MAX_END`, once `q`-following is
  in place.)

Launcher: `VARIANT=fix3_<lever> run_nav_p2.sh` (3 envs × K=2 = 3+3 slots ×
32 = 192 episodes/update) and `fix1_<lever>` (1 env × K=2 = 1+1 × 64).
Verdict instrument: the standard held-out training eval plus the nav
probe's `follow_q` (§7.10's table) — the question is not only "does it
reach held-out goals" but "is it following `q` when it does".

### 8.2 Wave 1 (3 envs, fixed goals; 2026-09-15 16:50 → 09-16 morning): the smaller trunk generalizes

Arms: `fix3_base` (h 1024), `fix3_nd0`, `fix3_xod8`, `fix3_h128`; 3 envs ×
K=2 = 3+3 slots × 32 = 192 episodes/update; `fix3_base`/`nd0` 12 h on
ou_bcs (to ~u3870), `xod8`/`h128` 6 h on mit_normal_gpu (to u2625/u3725).
Held-out deterministic, 4-eval windows:

| arm | window | succ 0/10 | steps 0/10 | swept 0/10 |
|---|---|---|---|---|
| `fix3_base` | u3850 | 0.99/0.97 | 17.1 / 16.0 | 0.55/0.53 |
| `fix3_nd0` | u3875 | 0.99/0.95 | 21.3 / 18.0 | 0.56/0.55 |
| `fix3_xod8` | u2625 | **0.57/0.60** | 70 / 66 | 0.59/0.59 |
| **`fix3_h128`** | u3300 | **1.00/0.98** | **12.0 / 14.2** | 0.54/0.50 |
| `fix3_h128` | u3725 | 1.00/0.95 | 13.3 / 16.0 | 0.57/0.54 |

1. **Three arenas with fixed goals do not collapse the way one did** —
   the baseline holds 0.97–0.99 held-out success — but they navigate
   indirectly (17–25 steps against the 13/14 bar): a mixed policy, part
   map, part `q`.
2. **`h128` reaches the bar**: 12.0/14.2 steps at 1.00/0.98 with d0_base-
   level explore (0.54/0.50) at u3300 = 634k episodes. Jack's guess. The
   barcode→position lookup is the expensive representation and atan2(`q`)
   the cheap one; at h 128 the cheap one wins.
3. Clean `q` (`nd0`) did nothing for directness; exploit-only barcode
   dropout at 0.8 hurt exploit badly (trained under noise, evaluated
   clean) while giving the best explore (0.59). Milder xod (0.3/0.5) with
   h128 is the combination to test.

Probes: `follow_q` on held-out for h128 u3300/u3700, base u3600, nd0
u3600, xod8 u2600 (job 22824404). Wave 2 launched: `fix1_h128`,
`fix1_h64`, `fix1_h256` (ONE env, fixed goal, trunk sweep) and `fix3_h64`
(jobs 22824406–09, 12 h, ou_bcs).

Continuations (§7.12) all COMPLETED u4000: **`one_k4_g_e75` s43 u4000
11.5/13.6 steps, swept 0.60/0.56** — the 3:1 arm's exploit caught up
late while keeping d0_base explore; `one_k4_g` s43 u4000 11.4/12.4,
0.52/0.50; `sched` u4000 12.9/15.2, 0.55/0.52 (did not beat plain 3:1);
`b32_g_e75` u4000 (512k) 12.3/13.9, 0.53/0.51. Probe round 3 on these
(job 22824395).

### 8.3 Probes (09-16 08:45): three fixed goals already make a `q`-follower at d=0; the small trunk keeps it under distractors

**Round 3, redraw line (job 22824395; d0_base u725 tail 4 of 144):**

| checkpoint | episodes | × opt d=0/5/10 | explore d=0 swept/eff/tail | d=10 swept/eff/**tail** |
|---|---|---|---|---|
| d0_base u725 | 928k | 1.19 / 1.17 / 1.37 | 0.603 / 0.943 / 0 | 0.580 / 0.931 / **0.028 (4)** |
| **e75 s43 u4000** | 1.02M | 1.26 / 1.22 / **1.32** | **0.613 / 0.988 / 0** | 0.565 / 0.926 / 0.083 (12) |
| e75 s43 u3500 | 896k | 1.31 / 1.24 / 1.49 | 0.612 / 0.976 / 0 | 0.571 / 0.912 / 0.062 (9) |
| k4g s43 u3000 | 768k | **1.17 / 1.20 / 1.32** | 0.562 / 0.898 / 0 | 0.526 / 0.863 / 0.090 (13) |
| k4g s43 u4000 | 1.02M | 1.20 / 1.17 / 1.33 | 0.496 / 0.808 / 0.132 | 0.456 / 0.761 / 0.153 (22) |
| sched u4000 | 1.02M | 1.29 / 1.29 / 1.40 | 0.583 / 0.946 / 0.007 | 0.521 / 0.891 / 0.132 (19) |
| b32e75 u2500 | 320k | 1.29 / 1.24 / 1.35 | 0.599 / 0.935 / 0 | 0.546 / 0.890 / 0.104 (15) |
| b32e75 u4000 | 512k | 1.22 / 1.18 / 1.38 | 0.550 / 0.885 / 0.007 | 0.473 / 0.807 / 0.153 (22) |

`e75` s43 u4000 is the closest single one-arena checkpoint: exploit
success 1.000 everywhere, d=10 directness better than d0_base, d=0/5
+0.05–0.07 (past the 0.03 rule), explore efficiency ≥ d0_base at both
levels, d=10 tail 12 vs 4. The schedule arm did not beat plain 3:1;
`k4g` s43's explore DEGRADED from u3000 to u4000 (d=0 tail 0 → 0.132).
Delivered one-arena checkpoints stand as §7.11 point 3 plus
`..._one_k4_g_e75_s43_22763088/navigate_u4000.pt` as the best all-round.

**Three envs, fixed goals — `follow_q` on held-out (job 22824404):**

| checkpoint | d | success | steps | follow_q | fq t0/t1/t2/t6+ |
|---|---|---|---|---|---|
| **fix3_h128 u3300** | 0 | 1.000 | **12.8** | **0.83** | 0.79/0.89/0.86/0.81 |
| fix3_h128 u3300 | 10 | 0.995 | **14.9** | **0.65** | 0.80/0.87/0.85/0.54 |
| fix3_h128 u3700 | 0 / 10 | 1.0 / 0.99 | 15.9 / 18.9 | 0.68 / 0.52 | t6+ 0.59 / 0.42 |
| fix3_base u3600 | 0 | 1.000 | 13.5 | 0.83 | 0.90/0.88/0.92/0.78 |
| fix3_base u3600 | 10 | 0.990 | 19.6 | 0.58 | 0.83/0.84/0.88/0.48 |
| fix3_nd0 u3600 | 0 / 10 | 1.0 / 0.99 | 22.8 / 24.7 | 0.55 / 0.45 | t6+ 0.44 / 0.34 |
| fix3_xod8 u2600 | 0 / 10 | 0.63 / 0.63 | 71 / 57 | 0.11 / 0.11 | 0.65/0.49/0.39/0.09 |
| redraw one_k2_g u900 (§7.10) | 0 / 10 | 1.0 | 12 / 13 | 0.90 / 0.87 | t6+ 0.92 / 0.88 |

1. **Three (arena, goal) pairs are enough to make a `q`-follower at d=0**
   — `follow_q` 0.83 for BOTH trunk sizes, against 0.05–0.2 for one pair
   (§7.10). The one-arena collapse is a one-pair effect.
2. **The small trunk keeps following under distractors**: at d=10 h128
   0.65 / 14.9 steps vs h1024 0.58 / 19.6, and the difference is late in
   the episode (t6+ 0.54 vs 0.48) — the large trunk lets go of `q` when
   the recall is corrupted, the small one has less else to fall back on.
   h128's u3700 is worse than its u3300 (0.68/0.52): not monotone, pick
   by eval.
3. `nd0` (no exploit distractors) made a policy that cannot handle them
   (0.45 at d=10) and follows less even at d=0 (0.55, 22.8 steps).
   `xod8` destroyed `q`-following (0.11): training under 80% barcode
   dropout and evaluating clean is a distribution shift, not a
   regulariser, at this rate.

Wave 2 (one env, fixed goal, trunk sweep) running: `fix1_h128` u200
0.64/0.53 success.

### 8.4 Wave 2 digest (09-16 09:17): the h128 ONE-env fixed-goal arm rises where h1024 slid

| arm | u | episodes | succ 0/10 | steps 0/10 | swept |
|---|---|---|---|---|---|
| **fix1_h128** | 750 | 96k | **0.94/0.94** | 32 / 35 | 0.34 |
| fix1_h64 | 475 | 61k | 0.43/0.40 | 32 / 34 | 0.34 |
| fix1_h256 | 425 | 54k | 0.42/0.40 | 35 / 36 | 0.32 |
| fix3_h64 | 200 | 38k | 0.72/0.72 | 49 | 0.18 |

`fix1_h128` held-out success by window: 0.02, 0.40, 0.63, 0.71, 0.84,
0.92, 0.97, 0.94 (u100–750) — rising, where the h1024 one-env fixed-goal
arm (§7.3) peaked at 0.99 at u100 and slid to 0.5. Directness still 32–35
steps: whether it tightens toward `q`-following or plateaus as search is
the `follow_q` question at ~u1500. Launched on mit (6 h): `fix1_h128_xod3`
(+ mild exploit barcode dropout) and `fix1_h128` s43 (a second arena),
jobs 22825504/5.

### 8.5 ONE env, fixed goal, h128 — it works when the goal is interior (09-16 10:30; job 22826500)

`behavior_probe --mode nav`, held-out, 32 trials × 6 arenas:

| checkpoint | goal | d | success | steps | **follow_q** | fq t0/t1/t2/t6+ |
|---|---|---|---|---|---|---|
| **fix1_h128 s43 u500** (64k eps) | (11, 6) interior | 0 | 1.000 | **12.9** | **0.83** | 0.50/0.69/0.76/0.89 |
| fix1_h128 s43 u500 | | 10 | 0.995 | **14.4** | **0.72** | 0.44/0.61/0.69/0.74 |
| fix1_h128 s42 u800 | (17, 0) edge | 0 / 10 | 0.94 / 0.89 | 47 / 51 | 0.19 / 0.14 | t6+ 0.17 / 0.13 |
| fix1_h128 s42 u1200 | edge | 0 / 10 | 0.93 / 0.88 | 42 / 48 | 0.16 / 0.08 | 0.63/0.65/0.51/0.11 |
| fix1_h128_xod3 s42 u500 | edge | 0 / 10 | 0.96 / 0.92 | 36 / 32 | 0.33 / 0.29 | t6+ 0.28 / 0.25 |
| fix3_h64 s42 u400 | 3 edge goals | 0 / 10 | 1.0 / 0.99 | 21 / 22 | 0.57 / 0.45 | t6+ 0.54 / 0.39 |
| redraw one_k2_g u900 (§7.10) | — | 0 / 10 | 1.0 | 12 / 13 | 0.90 / 0.87 | t6+ 0.92 / 0.88 |

1. **One arena, one fixed goal, h 128, interior goal → a `q`-follower**:
   follow_q 0.83/0.72, 12.9/14.4 steps at 1.00/0.995 on six unseen arenas
   at u500 = 64k episodes. The h1024 fixed-goal policy was 0.05–0.2.
2. **A wall goal defeats it even at h 128.** s42's goal (17, 0): follow_q
   0.16, 93% held-out success by search (42–48 steps). The time profile is
   the tell — follows `q` for two steps (0.63/0.65) then lets go — the
   signature of "go to the bottom wall, slide along": a one-dimensional
   map that even a small trunk prefers. Mild exploit barcode dropout (0.3)
   moves it (0.16 → 0.33) but not enough. Note the seed-42 THREE-env set is
   three wall goals — (17, 0), (0, 13), (0, 15) — and h1024 follows `q`
   there (0.83, §8.3): three wall-maps cost more than one `q`.
3. The one-env h1024 arm on s43 (`one_k4` s43, §7.3) slid early (0.90 →
   0.74 by u125), so trunk size matters on the interior goal too;
   `fix1_base` s43 (22826508) is the clean control, `fix1_h128` s44/s45
   (22826509/10) the arena distribution. Running: `fix1_h64` (u1000 0.56
   success but 16.8 steps — direct when it works), `fix1_h256` (0.69, 48
   steps), `fix3_h64` (u500 1.00/0.96, 23.7 steps).

**10:05:** cancelled `fix1_h256` (u1000: 0.69 success, 48 steps — h256 is
not it). Queued for the wall-goal case: `fix1_h128_xod5` (22827721) and
`fix1_h64_xod3` (22827722), both s42. Pending: `fix1_base` s43 (h1024
control on the interior goal), `fix1_h128` s44/s45.

### 8.6 10:33 digest — one arena, fixed interior goal, h128 = d0_base exploit on held-out

| arm | u | episodes | succ 0/10 | steps 0/10 | swept |
|---|---|---|---|---|---|
| **fix1_h128 s43** (goal (11,6)) | 1525 | 195k | **1.00/0.99** | **11.4 / 12.4** | 0.49/0.46 ↑ |
| fix1_h128 s42 (goal (17,0)) | 1800 | 230k | 0.98/0.97 | 31 / 31 | 0.46/0.47 |
| fix1_h128_xod3 s42 | 1425 | 182k | 0.99/0.97 | 32 / 32 | 0.44/0.43 |
| fix1_h64 s42 | 1200 | 154k | 0.54/0.49 | 16 / 17 | 0.43/0.47 |
| fix3_h64 | 600 | 115k | 1.00/0.97 | 20 / 23 | 0.42/0.44 |

d0_base u725's own eval: 11.7/12.1. `fix1_h64` s42 is the odd one — half
the trials fail but the successes are direct (16 steps): the shape of a
`q`-follower with a broken gate; probe when it has more updates. Five arms
still queued on ou_bcs (backlog): `fix1_base` s43, `fix1_h128` s44/s45,
`fix1_h128_xod5`, `fix1_h64_xod3`.

**Page v5 (09-16 10:50):** Part III "fixed goals, few environments" —
overlays `results/nav_tri_probe/fix3_by_update.png`, `fix1_by_update.png`;
rows `fix3_h128`, `fix3_base`, `fix1_h128` s43 / s42; the `follow_q` table
(§7.10, §8.3, §8.5). Regenerate with `make_fix_plots.sh` + `splice_page.py`
(job tmp).

### 8.7 11:07 digest — interior-goal one-arena model beats d0_base's own directness; h64 works at 3 envs; a half-and-half follower on the wall goal

| arm | u | episodes | succ 0/10 | steps 0/10 | swept |
|---|---|---|---|---|---|
| **fix1_h128 s43** (interior) | 2275 | 291k | 1.00/0.99 | **11.1 / 11.7** | 0.52/0.49 |
| **fix3_h64** | 1000 | 192k | 1.00/1.00 | **12.7 / 14.3** | 0.54/0.49 |
| fix1_h64 s42 (wall) | 1875 | 240k | **0.50/0.48** | **12.9 / 13.9** | 0.43/0.46 |
| fix1_h128 s42 (wall) | 2500 | 320k | 0.99/0.97 | 31 / 33 | 0.50/0.50 |
| fix1_h128_xod3 s42 | 2000 | 256k | 1.00/0.98 | 27 / 30 | 0.43/0.43 |

d0_base u725's own eval is 11.7/12.1. `fix3_h64` reaches the bar at u1000
where `fix3_h128` needed u3300. `fix1_h64` on the wall goal: exactly half
the held-out trials succeed and those are direct — a follower that works
on some arenas and not others, or a broken gate; probe with a per-arena
breakdown (added to `behavior_probe` nav mode).
