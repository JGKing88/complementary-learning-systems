# Explore-first training — experiment log

Plan: `docs/EXPLORE_FIRST_PLAN.md`. Started 2026-09-17. Branch `explore-first`.

The question, in one line: **given an agent that already explores, how many
exploit trajectories does it take to learn the water maze, and does the
exploring survive?** Phase 1 is the existing explore specialist
`p20_e_kcap u700`; phase 2 is the task regime with every explore reward off;
the arms differ only in how forgetting is prevented.

Conventions. "Trajectory" = one 200-step rollout of one batch row
(`cum_episodes` in the checkpoint), never a `collect_rollout` call; the two
differ by `batch_envs` = 64×. Every number is held-out unless marked
`(train)`. Updates are the run's own counter, u0 = the explorer before any
phase-2 step. Explore numbers are always given as a delta against the
explorer's u0 row.

## 0. Summary

| arm | job | status | u_criterion | traj_criterion | revisit sr / steps | follow_q | coverage d0 / d10 (Δ vs u0) | found_frac | CL |
|---|---|---|---|---|---|---|---|---|---|
| explorer u0 (`p20_e_kcap u700`) | — | not run | — | 0 | | | | | |
| E0 `xf_naive` | | not launched | | | | | | | |
| E1 `xf_ewc_lo` / `_hi` | | not launched | | | | | | | |
| E2 `xf_adp` | | not launched | | | | | | | |
| E3 `xf_kl_lo` / `_hi` | | not launched | | | | | | | |
| B0 `task1r_k4_h1024` | | not launched | | | | | | | |
| B1 `xf_scratch_nonov` | | not launched | | | | | | | |

`u_criterion` / `traj_criterion`: first eval at which held-out revisit
success ≥ 0.95 at ≤ 20 steps AND `follow_q` ≥ 0.80 (the task line's level;
`task1r_k4_h128` crossed it at u500 = 128k trajectories). `CL` = continual
protocol retention (5 envs × 100 iterations, stochastic, locked store).

## 1. Protocol (as planned; becomes "as run" when the first job lands)

Fork `p20_e_kcap u700` (hidden 1024, κ cap 2.5, `input_hopfield_raw`) into
`task:1000,visits=4,novelty=0,eps=0` on one arena with the goal redrawn per
visit sequence; batch 64 × 4 repeats = 256 trajectories per update; SE
optimizer (10 × 8 at 1e-4, target_kl 0.1); goal 2.0, wall −0.1, persistence
+0.2, time −0.05; U[0,10] outside-arena distractors; `EVAL_SCOPE=task` every
25 updates; checkpoints every 25. No `input_goal_in_memory`, ever. Nothing in
phase 2 pays for covering the arena.

## 2. Implementation log

- 2026-09-17 — worktree `explore-first` cut from `main` 77cf411 (the merged
  `nav-tri-metric` line plus the task-faithful line). Plan written. Read
  before writing it: wave 3 of `EXPERIMENTS_NAV_TRI.md` (arm A = the earlier
  explore-first, collapsed 0.367 → 0.068 at the end of its anneal; arm C =
  blocked, slid 0.351 → 0.223), `DUAL_TRAINING.md` §9 (the explorer has no
  ‖q‖ gate), the trainer's `task` stage (`novelty` / `eps` are stage knobs
  and reach `TaskRegime`; `RolloutBatch.explore_mask` is the per-step
  before-the-store mask), `hopfield_nav/continual/` (OnlineEWC with a Fisher
  estimator, LwF's `_frozen_copy` / `_masked_kl` for the polar
  distribution — written for the sequential RNN driver, not yet wired into
  `train_navigate`'s PPO loop), and the explorer checkpoints
  (`p20_e` u700 at `log_kappa_max 5.0`, `p20_e_kcap` u700 at 2.5, both
  hidden 1024 with d0_base's channels).

## 3. Wave 1

Not launched. Waiting on the plan's §9.
