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

*(filled as evals land; window means, `sample_eff_curve.py`)*

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
