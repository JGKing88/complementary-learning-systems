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

## 3. Wave S1 — pool size and passes over it

All arms are the `d0_base` recipe (w52 encoder, gain = β = 100, polar, κ cap
2.5, speed [0.5, 1.0], `interleave, empty_frac=0.5`, shuffle, ε 0.1/200,
goal 2.0, persistence 0.20, depth-1 Hopfield input) with seed 42. Only the
rows below differ. `target_kl` is new (`--target_kl`, k3 estimator, checked
after each minibatch step, stops the epoch loop); it exists so that 10 epochs
cannot run the policy off the data.

| arm | envs × batch | traj/update | epochs × mb | grad steps/u | traj per mb | ε-steps/u ceiling | samples vs d0_base |
|---|---|---|---|---|---|---|---|
| `d0_base` | 20 × 64 | 1280 | 4 × 4 | 16 | 320 | 256,000 | 1× |
| `se_b8` | 20 × 8 | 160 | 4 × 4 | 16 | 40 | 32,000 | **1/8** — the control: pool shrunk, optimizer untouched |
| `se_b8_e10` | 20 × 8 | 160 | 10 × 8, kl 0.02 | ≤ 80 | 20 | 32,000 | 1/8 |
| `se_b4_e10` | 20 × 4 | 80 | 10 × 8, kl 0.02 | ≤ 80 | 10 | 16,000 | **1/16** |
| `se_n10_b8_e10` | 10 × 8 | 80 | 10 × 8, kl 0.02 | ≤ 80 | 10 | 16,000 | 1/16, half the serial rollout calls — env diversity vs pool size |
| `se_b8_e10_h100` | 20 × 8, T = 100 | 160 | 10 × 8, kl 0.02 | ≤ 80 | 20 | 16,000 | 1/8 episodes, ~1/16 env-steps — explore rows always run the full T |
| `se_b8_e10_lr6` | 20 × 8 | 160 | 10 × 8, kl 0.02, lr 6e-4 | ≤ 80 | 20 | 32,000 | 1/8 — bigger step per update under the KL stop |

Schedules are `interleave:4000`; TIMEOUT at 24 h is the expected end and the
checkpoint series (every 25) is the result. Samples-to-target is read off the
first checkpoint that passes §1, never off the end of the run.

**Predictions on record, before any eval landed.**

1. `se_b8` (same optimizer, 1/8 the data) reaches the exploit lock (success
   pinned at 1.000) in *more* updates than d0_base's u125 but far fewer
   samples — the exploit gradient is not variance-limited at 160 trajectories
   when each holds a goal event. If it needs > 8× the updates the pool size is
   the constraint and the e10 arms are the only hope.
2. `se_b8_e10` beats `se_b8` on updates-to-lock by 2–4×, i.e. the extra passes
   are worth most of what they cost, and the KL stop fires on a minority of
   updates early and a majority late (policy sharpening).
3. The explore half is where the arms will separate: its slowest quantity is
   the collapsed tail, which d0_base only closed between u250 and u725. A
   small pool sees fewer of the rare high-‖q‖ goal-absent draws per update,
   so the tail may need as many *episodes* as d0_base regardless of PPO
   settings. That would cap the gain at ~2–4× and point wave S2 at the
   distractor distribution rather than the optimizer.
4. `h100` matches `se_b8_e10` on exploit and loses on explore at 200-step
   eval, because coverage over the second 100 steps is never rewarded.

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
