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

| arm | job | status | u_criterion | traj_criterion | last-8 revisit steps | last-8 cos_post | last-8 swept d0 / d10 (× u0) | last-8 found | exploit sr d0 / d10 (last) | probe, sampled (§3.5): swept d0 / d10; exploit d0 / d10 |
|---|---|---|---|---|---|---|---|---|---|---|
| phase 1 `xf_explorer` | 22889945 | done u700, 6.2 h, 896k traj (not charged) | — | 0 | 90 (no gate) | 0.04 | **0.539 / 0.544** = u0 row; sampled 0.603 / 0.612 | 0.57 | 0.45 / 0.41 | swept 0.60 / 0.60; exploit 0.46 / 0.35 |
| **E0 `xf_naive`** *(central)* | 22918221 | done u1000 | **925** | **236,800** | 15.9 | 0.76 | 0.419 / 0.383 (0.78 / 0.70) | 0.37 | 1.00 / 0.97 | swept 0.48 / 0.42; sr 1.00, path opt 0.71 / 0.69 |
| **E0' `xf_naive_lr03`** *(central)* | 22918222 | done u1000 | **750** | **192,000** | 14.9 | 0.86 | 0.403 / 0.387 (0.75 / 0.71) | 0.33 | 1.00 / 0.99 | swept 0.45 / 0.44; sr 1.00, path opt 0.72 / 0.68 |
| E1 `xf_ewc_1e3` | 22919093 | done u1000 | never (0.79 @ u900) | — | 20.4 | 0.70 | 0.486 / 0.477 (0.90 / 0.88) | 0.31 | 0.85 / 0.82 | swept 0.45 / 0.49; sr 1.00, path opt 0.58 / 0.58 |
| E1 `xf_ewc_1e4` | 22919094 | done u1000 | never | — | 20.0 | 0.64 | 0.391 / 0.365 (0.73 / 0.67) | 0.29 | 1.00 / 0.98 | swept 0.40 / 0.31; sr 1.00, path opt 0.60 / 0.58 |
| E3 `xf_kl_1` | 22918226 | done u1000 | never | — | 79.1 | 0.06 | 0.526 / 0.528 (0.98 / 0.97) | 0.50 | 0.41 / 0.43 | swept 0.60 / 0.60; exploit 0.45 / 0.26 |
| E3 `xf_kl_10` | 22918227 | done u1000 | never | — | 83.7 | 0.05 | 0.535 / 0.540 (0.99 / 0.99) | 0.55 | 0.47 / 0.42 | swept 0.61 / 0.60; exploit 0.48 / 0.27 |
| B0 `task1r_k4_h1024` (scratch, novelty on) | 22891316 | done u1000 | **250** | **64,000** | 14.1 | 0.92 | 0.325 / 0.298 | 0.51 | 1.00 / 0.99 | swept 0.48 / 0.47; sr 1.00, path opt 0.76 / 0.75 |
| B1 `xf_scratch_nonov` (scratch, novelty off) | 22891317 | done u1000 | **300** | **76,800** | 18.6 | 0.86 | 0.282 / 0.253 | 0.39 | 1.00 / 1.00 | swept 0.38 / 0.36; sr 1.00, path opt 0.60 / 0.58 |

**Wave 1 in one line: the explorer prior made exploit 3–3.7× *slower* to
learn, not faster, and the plain fork kept 71–80 % of the explorer's
coverage — level with a from-scratch run that is paid to explore, +0.06–0.10
over one that is not; no arm held both, and from-scratch-with-novelty
dominates every fork on the sampled metrics (§3.4, §3.5).**

Series tables and figure: `$CLS_RESULTS/explore_first/wave1_series.{md,png}`
(`analysis/explore_first/series.py`). Deterministic trainer evals, last-8 =
mean of the last eight evals (u825–u1000); the sampled probe pass on the
final checkpoints is §3.5.

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

The two EWC arms died at their first update — `estimate_fisher` put the
agent in `eval()` for its backward passes and cuDNN refuses the RNN backward
there ("cudnn RNN backward can only be called in training mode"; the CPU
tests never hit it). Fixed (the Fisher pass stays in train mode; the agent
has no dropout or batch-norm) and resubmitted as `xf_ewc_1e3` **22919093**
and `xf_ewc_1e4` **22919094** at ~17:20.

### 3.4 Wave 1 — results (2026-09-17, forks finished 20:00–22:10)

Series tables and the four-panel figure:
`$CLS_RESULTS/explore_first/wave1_series.{md,png}`. All numbers held-out,
deterministic trainer evals; "last-8" = mean over the run's last eight
evals (u825–u1000), which is the honest level given the deterministic
eval's ±0.1 wobble. The u0 row is the explorer scored on the forks' own
held-out envs: swept **0.539 / 0.544** (d = 0 / 10), first-visit
`found_rate` 0.57, revisits 0.51 at 90 steps, `cos_post` 0.04, exploit-
regime success 0.45 / 0.41 (stumbling; no gate).

| arm | crossed | trajectories | last-8 swept d0 / d10 (× u0) | min swept d0 | last-8 found | last-8 cos_post | last-8 revisit steps | exploit sr d0 / d10 (last) |
|---|---|---|---|---|---|---|---|---|
| **E0 `xf_naive`** *(central)* | **u925** | **236,800** | 0.419 / 0.383 (0.78 / 0.70) | 0.287 (u775) | 0.37 | 0.76 | 15.9 | 1.00 / 0.97 |
| **E0' `xf_naive_lr03`** *(central)* | **u750** | **192,000** | 0.403 / 0.387 (0.75 / 0.71) | 0.214 (u425) | 0.33 | 0.86 | 14.9 | 1.00 / 0.99 |
| E1 `xf_ewc_1e3` | never (0.79 at u900) | — | 0.486 / 0.477 (0.90 / 0.88) | 0.305 (u400) | 0.31 | 0.70 | 20.4 | 0.85 / 0.82 (u1000 dip; 1.00 / 0.99 at u975) |
| E1 `xf_ewc_1e4` | never | — | 0.391 / 0.365 (0.73 / 0.67) | 0.340 (u625) | 0.29 | 0.64 | 20.0 | 1.00 / 0.98 |
| E3 `xf_kl_1` | never | — | 0.526 / 0.528 (0.98 / 0.97) | 0.518 | 0.50 | 0.06 | 79.1 | 0.41 / 0.43 |
| E3 `xf_kl_10` | never | — | 0.535 / 0.540 (0.99 / 0.99) | 0.524 | 0.55 | 0.05 | 83.7 | 0.47 / 0.42 |
| B0 `task1r_k4_h1024` (scratch, novelty) | **u250** | **64,000** | 0.325 / 0.298 | 0.112 | 0.51 | 0.92 | 14.1 | 1.00 / 0.99 |
| B1 `xf_scratch_nonov` (scratch, no novelty) | **u300** | **76,800** | 0.282 / 0.253 | 0.089 | 0.39 | 0.86 | 18.6 | 1.00 / 1.00 |

**Part 1 — "exploit is cheap given explore" — is falsified, and in the
wrong direction.** The plain fork needed **3.7× the trajectories** the
from-scratch control needed to reach the same exploit level (236.8k vs
64k; the lr 3e-5 fork 3.0×). From scratch with no explore reward at all
(B1) crossed at 76.8k — P6 falsified too: the untrained searcher touches the
goal on ~40 % of first visits within 200 steps, which is signal enough. And
the explorer's first-visit advantage was small to begin with: `found_rate`
0.57 at u0 against 0.4–0.5 for the from-scratch runs after a few dozen
updates — the deterministic explorer sweeps the arena but does not stop at
goals any better than a random-ish walker does.

*Why slower, mechanistically.* The PPO diagnostics say it: the explorer
sits **at the κ cap** (κ 12.1 = e^2.5, σ 0.02–0.03, movement entropy −1.4
to −1.7) from u10 on, while the from-scratch policy at the same updates has
κ 7–10, σ 0.09–0.13, entropy +0.5 → −0.1. The explorer is a near-
deterministic sweeper: PPO has ~4× less angular noise with which to
discover that turning toward `q` pays after the store, and the confident
sweep is a local optimum it must first leave. The specialist's sharpness,
which is what makes it a good explorer, is what makes it a bad
initialisation for exploration in the RL sense. (The lower learning rate
crossing *earlier* — u750 vs u925 — is one seed and inside the
eval-to-eval wobble; not a finding.)

**Part 2 — "explore survives" — partially, as a slide, not a cliff.** E0's
swept coverage goes 0.54 → 0.49 → 0.47 → 0.34 (u75) and then wanders
between 0.29 and 0.49 for the rest of the run, last-8 mean **0.42 / 0.38 —
78 % / 70 % of u0**, minimum 0.29 (53 %). That is P2 confirmed (> 30 %
loss at the minimum; ~25 % at the level), and the shape is arm C's slide,
not arm A's cliff. It is also **not** the from-scratch level: E0 ends 0.10
above B0 (0.33 / 0.30) and 0.14 above B1 (0.28 / 0.25). So the prior is
neither kept nor erased — about half the gap to from-scratch survives. What
does not survive is the search competence the task actually needs:
first-visit `found_rate` falls 0.57 → 0.37, *below* B0's 0.51. The fork
keeps more of the sweep's shape than of its usefulness.

**The supporting arms map the trade-off, and it has no free point.**
- **KL (β = 1, 10):** holds the explorer to 97–99 % on coverage and learns
  **nothing** — `cos_post` 0.05–0.07, revisits 0.55–0.68 at 80 steps,
  exploit-regime success 0.41–0.47 — the u0 row, unchanged after 1000
  updates and 256k trajectories. `prior_kl` sat at 0.001–0.02 and PPO's
  per-step `approx_kl` at 0.001–0.005 (E0: 0.006–0.012): the search-step
  KL on the shared trunk held the whole policy to a few-times-smaller
  step, and q-following, which changes the function on inputs that differ
  from search inputs only by ‖q‖, never started. Both β are past the knee.
- **EWC (λ = 1e3, 1e4):** the mild end. Fisher mean 4.9e-5 / max 0.75 over
  3.4M parameters, penalty 0.01–0.06 — the PPO surrogate's own scale — and
  the policy still moves (drift 0.006–0.014 RMS). **λ = 1e3 is the closest
  any arm came to holding both**: last-8 swept 0.486 / 0.477 (90 % / 88 %
  of u0), revisits 1.00 at 20 steps, `cos_post` 0.70 last-8 and **0.79 at
  u900** (swept 0.51, found 0.57 at that eval) — one eval-point short of
  the criterion, and the series is still oscillating (0.59–0.79 over the
  last five evals) rather than settled. Its u1000 row dipped (exploit-
  regime success 0.85 / 0.82, revisits 0.97 at 24), so the checkpoint to
  carry forward would be u900, not u1000. λ = 1e4 kept 73 % at `cos_post`
  0.64 and never crossed. The Fisher's own noise is larger than the λ
  effect at these two values: 1e4 retained *less* coverage than 1e3.

**Ordering, on coverage kept:** KL (99 %) > EWC 1e3 (90 %) > E0 (78 %) ≈
E0' (75 %) ≈ EWC 1e4 (73 %) > from-scratch (—). **Ordering, on exploit
reached:** from-scratch (u250–300) > E0' (u750) > E0 (u925) > EWC 1e3
(0.79, not crossed) > EWC 1e4 (0.64) > KL (never). Nearly the same list
reversed, which is what "trade-off" means — with EWC 1e3 the one arm that
sits off the line: 90 % of the coverage at ~90 % of the exploit level.

**The original metrics, by regime (trainer, deterministic).** Exploit-
regime success is 1.00 / 0.97–1.00 for every arm that learned and
0.41–0.47 for the KL arms (the explorer's stumbling rate). Explore-regime
swept coverage is the column above. Path optimality and the sampled
headline row come from the probe pass (§3.5).

**Predictions, scored.** P1 falsified (no fork before u250; E0 at u925).
P2 confirmed (slide; −22 % level, −47 % minimum). P3 (lr) not supported —
the lower rate crossed earlier, one seed. P4 (KL holds and learns)
falsified — holds and does not learn, at both β. P5 (EWC monotone in λ)
not supported at these two values — 1e4 kept *less* coverage than 1e3 and
learned exploit slower, so the λ effect is inside the eval wobble here;
the two arms differ mostly in which local optimum the search fell into. P6 falsified (B1 crosses at 76.8k).

### 3.5 The original metrics, by regime — the probe pass (job 22933923)

`run_se_probe.sh` over every final checkpoint plus the explorer, one process,
`d0_base u725` appended by the script: **sampled** policies on held-out
place envs (6 envs × 24 explore trials / × 32 nav trials), the instrument
`DUAL_TRAINING` §9.8 was measured with. Files:
`$CLS_RESULTS/explore_first/se_xf_wave1_{d0,d10,nav}.json`, log
`$CLS_LOGS/se_probe_22933923.out`. The probe's `mean_start_dist` is
10.70 / 11.20 / 10.87 at d = 0 / 5 / 10 — the same constants as d0_base's
world — so path optimality is directly comparable, and the series tables
now carry it (`--start_dist 10.70 10.87`).

**Explore regime** (memory holds distractors only; swept @200; `swept_eff`
= swept ÷ billiard at the model's own realized speed; `frac<t` = share of
trials below half the billiard, the collapsed tail):

| | d=0 swept | swept_eff | tail | d=10 swept | swept_eff | tail | chase_t (d=10) |
|---|---|---|---|---|---|---|---|
| explorer u700 | **0.599** | 0.929 | 0.000 | **0.598** | 0.928 | 0.000 | — |
| **E0 `xf_naive`** | 0.482 (80 %) | 0.747 | 0.014 | 0.424 (71 %) | 0.670 | 0.104 | 0.13 |
| **E0' `xf_naive_lr03`** | 0.450 (75 %) | 0.724 | 0.028 | 0.436 (73 %) | 0.714 | 0.090 | 0.26 |
| EWC 1e3 | 0.450 (75 %) | 0.719 | 0.028 | **0.494 (83 %)** | 0.797 | 0.069 | 0.12 |
| EWC 1e4 | 0.402 (67 %) | 0.624 | 0.125 | 0.309 (52 %) | 0.498 | **0.514** | 0.09 |
| KL 1 / 10 | 0.604 / 0.607 | 0.940 / 0.944 | 0.000 | 0.604 / 0.604 | 0.937 | 0.000 | — |
| B0 scratch, novelty on | 0.477 | **0.826** | 0.035 | 0.469 | **0.828** | 0.069 | 0.28 |
| B1 scratch, no novelty | 0.383 | 0.723 | 0.076 | 0.357 | 0.696 | 0.160 | 0.23 |
| d0_base u725 | 0.610 | 0.954 | 0.000 | 0.586 | 0.932 | 0.028 | 0.43 |

**Exploit regime** (goal pre-stored by the oracle; success / mean steps /
per-episode path optimality / `follow_q`):

| | d=0 | d=5 | d=10 |
|---|---|---|---|
| explorer u700 | 0.46 / 56.7 / 0.35 / 0.08 | 0.43 / 47.7 / 0.31 / 0.08 | 0.47 / 50.2 / 0.27 / 0.09 |
| **E0 `xf_naive`** | 1.00 / 14.6 / **0.71** / 0.77 | 0.99 / 15.4 / 0.69 / 0.71 | 0.99 / 14.7 / 0.69 / 0.72 |
| **E0' `xf_naive_lr03`** | 1.00 / 13.4 / 0.72 / 0.86 | 0.99 / 17.6 / 0.70 / 0.71 | 0.99 / 14.8 / 0.68 / 0.78 |
| EWC 1e3 | 1.00 / 25.2 / 0.58 / 0.56 | 0.99 / 26.5 / 0.58 / 0.54 | 0.99 / 25.7 / 0.58 / 0.52 |
| EWC 1e4 | 1.00 / 17.9 / 0.60 / 0.69 | 1.00 / 20.6 / 0.58 / 0.65 | 0.99 / 19.0 / 0.58 / 0.66 |
| KL 1 / 10 | 0.45–0.48 / 58–81 / 0.26–0.27 / 0.07–0.08 | 0.46 / 47–56 / 0.32–0.39 | 0.46–0.50 / 69–81 / 0.25–0.29 |
| B0 scratch, novelty on | 1.00 / 12.7 / **0.76** / 0.92 | 0.99 / 13.0 / 0.76 / 0.86 | 0.99 / 12.9 / 0.75 / 0.87 |
| B1 scratch, no novelty | 1.00 / 16.7 / 0.60 / 0.83 | 1.00 / 18.0 / 0.58 / 0.81 | 1.00 / 17.2 / 0.58 / 0.83 |
| d0_base u725 | 1.00 / 11.7 / **0.82** / 0.92 | 0.99 / 13.3 / 0.80 / 0.84 | 0.99 / 12.0 / 0.80 / 0.82 |

**What the probe adds to §3.4.**

1. **The explorer is d0_base's equal on explore** — sampled swept 0.599 /
   0.598 against 0.610 / 0.586, `swept_eff` 0.93 against 0.95, tail 0.000
   at both distractor levels. Phase 1 produced the specialist it was meant
   to. The KL arms are that specialist, untouched, after 256k trajectories.
2. **The plain fork's retained coverage is not better than from-scratch-
   with-novelty.** Sampled, E0 keeps 0.48 / 0.42; B0, trained from scratch
   with the explore reward on before the store, reaches 0.48 / 0.47 — and
   does it at a lower speed, so its `swept_eff` is higher (0.83 vs 0.75 /
   0.67). The deterministic trainer eval had E0 0.10 above B0; the sampled
   probe puts them level at d=0 and B0 ahead at d=10. The like-for-like
   comparison — same objective, no explore reward anywhere — is E0 against
   B1: **0.48 vs 0.38 at d=0, 0.42 vs 0.36 at d=10**. That +0.06–0.10 is
   what the prior leaves behind after 1000 updates; it is a quarter of the
   0.22–0.24 it started with.
3. **At d=10 the fork's tail carries a corner-trap signature.** E0's
   `frac<t` is 0.104 with `chase_t` 0.13 above `chase_r` 0.07 (E0': 0.258
   vs 0.055): in a tenth of the trials the fork chases phantoms — the
   wave-3 D2 mechanism, mild, present. The explorer and the KL arms have
   none. EWC 1e3 has the least of any learning fork (0.069, chase 0.12)
   and is the best fork on d=10 coverage (0.494, 83 %).
4. **On the exploit metrics the fork is behind from-scratch at the same
   budget.** Path optimality at u1000: d0_base 0.82 > B0 0.76 > E0 0.71 ≈
   E0' 0.72 > EWC 0.58–0.60 ≈ B1 0.60; success 1.00 / 0.99 for every
   learning arm. `follow_q` orders the same way (0.92 > 0.77–0.86 > 0.56–
   0.69). So after 256k trajectories the prior has cost exploit quality as
   well as time: B0 navigates straighter than either fork.
5. **The trade-off, on the sampled measures, at u1000** (explore d=10 swept
   → exploit d=10 path optimality): KL 0.60 → 0.27 (the explorer); EWC 1e3
   0.49 → 0.58; E0 0.42 → 0.69; E0' 0.44 → 0.68; B0 0.47 → 0.75; d0_base
   0.59 → 0.80. **B0 dominates every fork** — more coverage than E0 and
   straighter paths — and d0_base dominates B0. Nothing in wave 1 is on
   the d0_base frontier; the interleaved recipe remains the only one that
   has both.

**The one-line verdict, on the metrics Jack asked for.** Given this
explorer, exploit is learned 3–3.7× slower than from scratch and ends
straighter-than-nothing but worse than from scratch (0.71 vs 0.76 path
optimality); the exploring that survives is 71–80 % of the explorer's
sweep, level with what a from-scratch run paid to explore reaches and
0.06–0.10 above one that is not; and no protection tried keeps the
explorer without also keeping it from learning. The sharpness that makes
the specialist a good explorer (κ at the cap) is what makes it a poor RL
initialisation, and that — not forgetting — is the first thing wave 2
should attack (plan §10.2).

## 4. Wave 2 — re-open the spread at the fork; sampled evals (submitted 2026-09-18 ~00:05)

Jack, on the wave-1 verdict: *"Ok do that. You can also try eval with
stochastic."*

**Built (commit f0a8287).** `--reset_kappa_head`: after `--load_checkpoint`,
`PolarHead.reset_spread(init_log_kappa)` puts the state-dependent log-κ
head back to zero weight + 1.85 bias (κ 6.4, the construction init; the
from-scratch run's early noise level), headings untouched.
`--no-eval_deterministic`: `do_eval` now passes `deterministic` to
`evaluate_navigation` and `evaluate_exploration`; **`evaluate_task` always
sampled** (it runs the training collector), so the `task=` rows in every
wave-1 log — found_rate, revisits, cos_post, i.e. the exploit criterion —
were sampled already, and only the `nav=` / `expl=` rows (exploit-regime
success and swept coverage) were deterministic. Sampled blocks are marked
`eval_policy=sampled` in the log. Launcher levers `_kreset`, `_ent02`
(move_ent_coef 0.02), `_sev`.

**Arms** (all fork `xf_explorer u700`, `task:1000,visits=4,novelty=0,eps=0`,
sampled `nav=`/`expl=` evals, ou_bcs_normal 5 h):

| arm | job | what it isolates |
|---|---|---|
| `xf_naive_kreset_sev` s42 | 22942303 | the reset alone — does a fork at κ 6.4 cross by u250? |
| `xf_naive_ent02_sev` s42 | 22942304 | entropy bonus alone (0.02 vs 0.005), no reset |
| `xf_naive_kreset_ent02_sev` s42 | 22942305 | both |
| `xf_naive_kreset_ewc_1e3_sev` s42 | 22942306 | the reset with wave 1's nearest-to-both protection |
| `xf_naive_sev` **s43** | 22942307 | the central arm's seed replicate, sampled evals |
| `xf_naive_kreset_sev` **s43** | 22942330 | the reset's seed replicate |

**Sampled re-score of wave 1** (`run_xf_reeval.sh`, `nav=` + `expl=`,
`--no-deterministic`, every 25 updates): jobs 22942335–42, one per wave-1
run, → `$CLS_RESULTS/explore_first/<arm>_reeval_stoch.log`. This gives the
wave-1 coverage series on the same footing as wave 2's.

**Pre-registered.** (1) If the reset alone crosses the criterion by u250
(the from-scratch time), §3.4's mechanism is confirmed and the prior's
problem was confidence; the coverage it keeps at the crossing is then the
number to compare with wave 1's 0.42 (sampled) — the drift argument (fewer
updates, less drift) predicts *more* kept. (2) If the reset does not speed
exploit, the slowness is in the trunk (the sweep as a local optimum), not
in the spread, and the entropy arm will not help either. (3) Sampled
coverage series for the wave-1 forks are predicted to sit 0.03–0.08 above
the deterministic ones (the probe: E0 0.48 sampled vs 0.42–0.46
deterministic at the end) with less wobble, and not to change the
ordering.

- 2026-09-18 ~01:10 — **the κ reset is transient.** In every reset arm κ is
  6.5 at u1, 9.4–10.8 at u10 and back at the cap (12.08) by u50; the
  entropy bonus at 0.02 does not hold it (`xf_naive_kreset_ent02`: 12.08 at
  u50, entropy −1.04). B0 from scratch sat at κ 7.4 through u50 and 8.3 at
  u100 — it *earned* its sharpness over ~200 updates; the fork's trunk
  already encodes the confident sweep and the κ head reads it straight
  back off the features. So the six arms above are, on this axis,
  replicates of E0 (still worth having: seeds 43 and sampled series).
  The plain lever that pins the spread is the cap itself: **`_kcap20`**
  (`LOG_KAPPA_MAX` 2.0 → κ ≤ 7.4, B0's own early level) and **`_kcap20a`**
  (2.0 → 2.5 over 300 updates, the `d1_kanneal` machinery). Submitted
  `xf_naive_kcap20_sev` **22946511** and `xf_naive_kcap20a_sev`
  **22946512** (commit for the levers: see git log).

### 4.1 Wave 1 re-scored sampled (jobs 22942335–42, done 2026-09-18 ~02:30)

`nav=` + `expl=` of every wave-1 run re-scored with the policy sampled,
every 25 updates, same held-out envs (`$CLS_RESULTS/explore_first/
<arm>_reeval_stoch.log`). Held-out swept @200 at d=0, last-8 means:

| arm | deterministic (trainer) | **sampled** | sampled min | sampled d=10 | sampled exploit-regime sr d0/d10 |
|---|---|---|---|---|---|
| explorer u700 | 0.538 | **0.603** | — | 0.612 | 0.62 / 0.69 |
| E0 `xf_naive` | 0.419 | **0.424** (70 %) | 0.317 | 0.414 | 1.00 / 0.99 |
| E0' `xf_naive_lr03` | 0.403 | 0.441 (73 %) | 0.241 | 0.425 | 1.00 / 0.99 |
| E1 `xf_ewc_1e3` | 0.486 | **0.504 (84 %)** | 0.331 | 0.495 | 0.99 / 0.99 |
| E1 `xf_ewc_1e4` | 0.391 | 0.413 (68 %) | 0.352 | 0.400 | 1.00 / 1.00 |
| E3 `xf_kl_1` / `_10` | 0.526 / 0.535 | 0.600 / 0.598 (99 %) | 0.589 | 0.601 | 0.52 / 0.52 |
| B0 scratch, novelty on | 0.325 | **0.457** | 0.113 | 0.433 | 1.00 / 0.99 |
| B1 scratch, no novelty | 0.282 | 0.338 | 0.109 | 0.338 | 1.00 / 1.00 |

Two corrections to the deterministic reading, both in the direction the
probe (§3.5) already pointed: **(1)** the from-scratch-with-novelty control
was understated by 0.13 — its mean policy orbits, its sampled policy
explores — so sampled, **B0 (0.46) is above the plain fork (0.42)**, not
0.10 below it; the fork's retained coverage is what a from-scratch run paid
to explore reaches anyway, and +0.09 over one that is not (B1 0.34).
**(2)** The forks themselves barely move under sampling (E0 0.419 → 0.424):
a fork's mean policy and sampled policy cover the same, the explorer's do
not (0.54 → 0.60) — which is itself a symptom of what the fork lost. The
KL arms are the sampled explorer to the third decimal. EWC 1e3 at 84 %
stays the best protected arm that learned anything. Ordering unchanged.
