# Explore-first training — experiment log

Plan: `docs/EXPLORE_FIRST_PLAN.md`. Started 2026-09-17. Branch `explore-first`.

The question, in one line: **given an agent that already explores, how many
exploit trajectories does it take to learn the water maze, and does the
exploring survive?** Phase 1 is an explore specialist retrained
under this line (`xf_explorer`, the `p20_e_kcap` recipe); phase 2 is the
task regime with every explore reward off;
the arms differ only in how forgetting is prevented.

Conventions. "Trajectory" = one 200-step rollout of one batch row
(`cum_episodes` in the checkpoint), never a `collect_rollout` call; the two
differ by `batch_envs` = 64×. Every number is held-out unless marked
`(train)`. Updates are the run's own counter, u0 = the explorer before any
phase-2 step. Explore numbers are always given as a delta against the
explorer's u0 row.

## 0. Summary

| arm | job | status | u_criterion | traj_criterion | revisit sr / steps (last) | cos_post (last) | swept d0 / d10 (last; band) | found_rate (last 8 mean) | exploit sr d0 / d10 | CL |
|---|---|---|---|---|---|---|---|---|---|---|
| phase 1 `xf_explorer` | 22889945 | **done** u700, 6.2 h, 896k traj | — | 0 | (no gate) | — | **0.538 / 0.538** (plateau u475–700: 0.556 / 0.551, min 0.528) | — | — | |
| **E0 `xf_naive`** *(central)* | 22918221 | queued 16:25 | | | | | | | | |
| **E0' `xf_naive_lr03`** *(central)* | 22918222 | queued | | | | | | | | |
| E1 `xf_ewc_1e3` / `_1e4` | 22918223 / 4 | queued | | | | | | | | |
| E3 `xf_kl_1` / `_10` | 22918226 / 7 | queued | | | | | | | | |
| B0 `task1r_k4_h1024` | 22891316 | **done** u1000, 3.3 h | **250** | **64,000** | 1.00 @ 15.6 | 0.91 | 0.396 / 0.385 (0.11–0.43; last-8 mean 0.325 / 0.298) | 0.51 | 1.00 / 0.99 | |
| B1 `xf_scratch_nonov` | 22891317 | **done** u1000, 3.3 h | **300** | **76,800** | 1.00 @ 21.5 | 0.86 | 0.284 / 0.244 (0.09–0.41; last-8 mean 0.282 / 0.253) | 0.39 | 1.00 / 1.00 | |

Series tables and figure: `$CLS_RESULTS/explore_first/wave1_phase1_controls_series.{md,png}`
(`analysis/explore_first/series.py`). Deterministic trainer evals; the
sampled probe pass on the final checkpoints comes after the forks land.

`u_criterion` / `traj_criterion`: first eval at which held-out revisit
success ≥ 0.95 at ≤ 20 steps AND `follow_q` ≥ 0.80 (the task line's level;
`task1r_k4_h128` crossed it at u500 = 128k trajectories). `CL` = continual
protocol retention (5 envs × 100 iterations, stochastic, locked store).

## 1. Protocol (as run)

**Phase 1** `xf_explorer` (job 22889945): `explore:700`, 20 arenas × 64
batch × 200 steps, hidden 1024 `rnn/relu`, polar head with state-dependent
spread, κ cap 2.5, `input_hopfield_raw`, multistep `"1"` (70 input dims),
encoder `w52_attract_fwhm/001_att0.5_seed=43` gain 100 / Hopfield β 100,
novelty 0.3 (remaining-scaled, cap 10) / wall −0.1 / persistence +0.2 /
time −0.05, `explore_goals_off`, ε 0.1→0 over 200 updates, U[0,10]
distractors, PPO 4 × 4 at 3e-4, `EVAL_SCOPE=expl` every 25, checkpoints
every 25. Seed 42. The `p20_e_kcap` recipe under today's defaults.

**Phase 2** (`xf_*`): fork the phase-1 checkpoint (`--load_checkpoint`,
Adam moments dropped) into `task:1000,visits=4,novelty=0,eps=0` on one
arena with the goal redrawn per visit sequence; batch 64 × 4 repeats = 256
trajectories per update (800 serial steps); SE optimizer (10 × 8 at 1e-4,
target_kl 0.1; `_lr03` at 3e-5); goal 2.0, wall −0.1, persistence +0.2,
time −0.05; U[0,10] outside-arena distractors; `EVAL_SCOPE=task` every 25
updates, plus a `[navigate_u0]` eval of the parent on the run's own held-out
envs before the first step; checkpoints every 25. No `input_goal_in_memory`,
ever. Nothing in phase 2 pays for covering the arena. Arms: none / lr 3e-5
/ `--ewc_lambda` 1e3, 1e4 / `--prior_kl_coef` 1, 10. Controls at the same
shape from scratch: `task1r_k4_h1024` (novelty on before the store) and
`xf_scratch_nonov` (novelty off).

## 2. Implementation log

- 2026-09-17 — worktree `explore-first` cut from `main` 77cf411 (the merged
  `nav-tri-metric` line plus the task-faithful line). Plan written. Read
  before writing it: wave 3 of `EXPERIMENTS_NAV_TRI.md` (arm A = the earlier
  explore-first, collapsed 0.367 → 0.068 at the end of its anneal; arm C =
  blocked, slid 0.351 → 0.223), `DUAL_TRAINING.md` §9 (the explorer has no
  ‖q‖ gate), the trainer's `task` stage (`novelty` / `eps` are stage knobs
  and reach `TaskRegime`; `RolloutBatch.explore_mask` is the per-step
  before-the-store mask), `hopfield_nav/continual/` (OnlineEWC with a Fisher
  estimator, LwF's `_frozen_copy` / `_masked_kl` — written for the
  sequential RNN driver's two-output agent and `kl_divergence`, which has no
  registration for the polar head), and the explorer checkpoints
  (`p20_e` u700 at `log_kappa_max 5.0`, `p20_e_kcap` u700 at 2.5, both
  hidden 1024 — and both with `input_hopfield_multistep [1, 2, 3]`, the
  pre-2026-09-06 layout: 74 input dims against today's 70).
- 2026-09-17 — Jack's answers: default task only; drop the adapter; retrain
  the explorer; goal reward as the baseline's; the naive fork is the central
  arm. Built (commit 39c6f5b): `training/prior.py` (`ExplorerPrior`: EWC
  with a once-estimated true Fisher on the first update's search steps, and
  the search-masked KL against a frozen copy), `polar_head.vonmises_kl` /
  `polar_kl` (analytic, Monte-Carlo checked), `ppo_update(prior=)`, the
  `xf_*` launcher family, `DRY_RUN=1`. Then: the u0 eval of a fork's parent
  in `train_navigate`, and the launcher's pool echo multiplied by
  `ENV_REPEATS`. 25 unit tests + 4 end-to-end smoke tests; 1,652 pass.
- 2026-09-17 ~09:10 — submitted `xf_explorer` 22889945 (ou_bcs_normal, 8 h),
  `task1r_k4_h1024` 22889972 and `xf_scratch_nonov` 22889973 (12 h). The
  `xf_*` forks wait on the explorer (~4 h).
- 2026-09-17 ~09:25 — the two 12 h controls had an estimated start of
  **2026-09-22** (`squeue --start`): ou_bcs_normal is saturated by the task
  line's ten 12 h jobs and a 12 h request cannot backfill. The h1024 task
  shape measured 11.5 s/update at 6 slots (`task3_k1_h1024`), so 4 slots is
  ~8 s/update: 1000 updates plus 40 evals at ~60 s is ~3 h. Cancelled and
  resubmitted at **5 h**: `task1r_k4_h1024` **22891316**, `xf_scratch_nonov`
  **22891317**. The phase-2 forks will go up at 5 h as well.

## 3. Wave 1

### 3.1 Phase 1 — the explorer (2026-09-17, job 22889945, done 16:22)

700 updates in 6.2 h (32 s/update on node3808), 896,000 trajectories.
Held-out deterministic `swept_coverage` @200: climbs to the plateau by
~u200 (0.53), and from u475 to u700 sits at **0.556 (d=0) / 0.551 (d=10)**,
min 0.528, max 0.578 — flat in distractors to the third decimal, as the
Aug explorer was. Before the plateau the deterministic series swung
between 0.29 and 0.60 from one checkpoint to the next (u250 0.42, u275 0.39,
u300 0.60, u350 0.29, u375 0.54), which is the κ-cap orbit effect on the
mean policy, not learning and unlearning; on the plateau the swing is gone.
No erosion, so **u700 is the fork checkpoint**, as the plan said. The
sampled re-score of the whole series (`run_xf_reeval.sh`, job 22918309,
`$CLS_RESULTS/explore_first/xf_explorer_reeval_stoch.log`) agrees and is
tighter: **sampled swept 0.59–0.61 from u475 to u700, u700 = 0.603 /
0.612** (d = 0 / 10), the plateau maximum. The deterministic dips are mostly
the mean-policy artefact — u350 reads 0.29 deterministic against 0.38
sampled, u250 0.42 against 0.49 — and before u175 the sampled series is
far above the deterministic one (u50: 0.40 vs 0.15). Sampled exploit-
regime success (goal pre-stored, no gate, stumbling onto it) is 0.6 at u700.

Against the Aug `p20_e_kcap`: its deterministic `mean_coverage` plateaued at
0.33–0.36 from u200; this one is at 0.34–0.36 from u200 — the same explorer,
now on the 70-dim input layout.

### 3.2 The two from-scratch controls (both done, 1000 updates, 3.3 h each)

Both learn exploit; neither learns to explore at the explorer's level.

- **B0 `task1r_k4_h1024`** (novelty on before the store, the task line's
  rule): crosses the exploit criterion at **u250 = 64,000 trajectories** —
  half the h128 baseline's u500/128k that the plan quoted, so the matched-
  width bar for the forks is u250, not u500. Held-out swept coverage never
  settles: 0.11–0.43 across the run, last-8 mean 0.325 / 0.298 (d=0/10),
  first-visit `found_rate` 0.2–0.65 eval to eval (last-8 mean 0.51). Exploit
  is stable from u250 on: revisits 1.00 at 13–16 steps, cos_post 0.86–0.93,
  exploit-regime success 1.00 / 0.98–1.00.
- **B1 `xf_scratch_nonov`** (novelty off everywhere — exploit-only reward
  from scratch): crosses at **u300 = 76,800 trajectories**. So **P6 is
  falsified**: with no explore reward at all, the untrained searcher still
  touches the goal on ~40% of first visits in 200 steps, and that is signal
  enough. Coverage 0.09–0.41, last-8 mean 0.282 / 0.253; `found_rate`
  last-8 mean 0.39; revisits 1.00 at 15–22 steps, cos_post 0.78–0.88.

What this sets up for the forks: the explorer starts at swept ≈ 0.55
against the controls' ≈ 0.30, and the question is whether that 0.25 gap is
still there after exploit is learned — and whether exploit arrives before
u250.

### 3.3 Phase 2 — submitted 16:25

`submit_xf_forks.sh` from `xf_explorer/navigate_u700.pt`: `xf_naive`
22918221, `xf_naive_lr03` 22918222, `xf_ewc_1e3` 22918223, `xf_ewc_1e4`
22918224, `xf_kl_1` 22918226, `xf_kl_10` 22918227 (ou_bcs_normal, 5 h).
