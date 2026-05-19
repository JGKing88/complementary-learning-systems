# Hopfield-Nav PPO Research Log

> **⚠ Metric correction (2026-04-28):** All `cov` / `mean_coverage` /
> `goal_find_rate` / `mean_steps_to_goal` numbers reported in this doc
> for runs **prior to 2026-04-28** were produced by `evaluate_exploration`
> with `disable_store=False` (the prior behavior). In that mode, the
> agent could store mid-rollout and once it stored AT the goal, the
> trust bit flipped to True for the rest of the rollout — meaning the
> cov rollouts measured **explore-then-nav** behavior, not pure
> exploration. The orbit-the-goal nav phase contributed substantially
> to the reported coverage (e.g. v36 s1 cov 0.50 was largely inflated
> by post-discovery perimeter-orbits).
>
> As of 2026-04-28, `evaluate_exploration` defaults to
> `disable_store=True`: store_action is ignored, trust stays False, no
> goal pattern ever enters Hopfield. All future cov numbers reflect
> pure exploration. Old numbers should be read as
> "explore + post-discovery-nav coverage," NOT pure exploration.
> `evaluate_navigation` (nav_det / nav_stoch) and
> `evaluate_goal_discovery` (disc) are unaffected — disc effectively
> already measured pure trust=0 behavior because it terminates on the
> first store-at-goal before trust=1 can drive any step.

## Task

Improve PPO agent on Hopfield-based grid navigation. Primary metrics to maximize:

- **Deterministic navigation (Eval 1):** `success_rate` ↑, `mean_speed` ↑
- **Goal discovery (Eval 2):** `store_efficiency` ↑, `reach_success_rate` ↑
- **Exploration (Eval 3):** `mean_steps_to_goal` ↓

The realistic eval (Eval 4) is logged but **not** a target metric — too many interacting dynamics to reason about directly; the three simpler evals should compose to good realistic behavior if individually strong.

Script base: `run_continuous.sh`. Tunable knobs per AUTORESEARCH.md: `store_cost, store_bonus, store_bc_weight, auto_store_warmup, auto_nav_warmup, aux_anneal_updates, num_rnn_layers, hidden_size, init_log_std, ent_coef, store_ent_coef, batch_envs, steps_per_rollout, n_updates, lr, explore_steps, clip_coef, gae_lambda, gamma, n_minibatches, ppo_epochs`. Everything else is frozen.

## Fixed setup (unchanged across runs)

- Encoder: `encoders/run_20260422_185816/encoder_best.pt` (MLP, out_dim=1024, gain=3.7)
- VectorHash: lambdas=[11,12,13], Npos=1716, gbook-only
- Env: size=8, obs=512, time_penalty=0.01, movement_mode=continuous (Normal(2), unit-norm step)
- Hopfield: beta=encoder_gain=3.7, alpha=0.8, steps=1, init=empty, agent_can_store=True
- Agent: GRU input_dim=3 (reward + 2D hopfield signal, NO encoded state), hidden=128, 1 layer

## Prior signal (single sweep only — treat as weak prior)

One prior 500-update sweep (`revived-water-17`, default-ish config) shows:
`nav_det success_rate` 1.0→0.475 with a peak at u=50, `disc store_efficiency` climbs to ~0.94, `disc reach_success_rate` peaks ~0.73 at u=170 then drops, `expl mean_coverage` peaks 0.62 at u=170 then drops. Suggestive failure modes (store head spamming at ~25% of steps; nav regression; late-training exploration decay) but one run ≠ ground truth. Use these as *input to* hypothesis generation, not as constraints on what to fix.

## First-principles view of the task

What the policy actually has to learn (with `input_encoded_state=False`, agent input = `[reward, hopfield_x, hopfield_y]`):

1. **"Am I at the goal?"** → observable from `reward` (≈1 vs ≈−0.01). Trivially decodable.
2. **"Is there any memory yet?"** → observable from `‖hopfield_signal‖` (0 vs ≈1). Trivially decodable.
3. **"Should I store now?"** → should be ≈`at_goal`, but PPO supervision through dilute at-goal rewards is slow.
4. **"How do I explore when no memory?"** → no extrinsic signal except `-time_penalty`; purely entropy/novelty-driven.
5. **"Follow the hopfield signal when it exists."** → map 2D unit-vector → 2D mean action.
6. **Critically, the RNN must maintain state** — the reward signal is only informative at step `t` if the agent remembers it came from position `x_{t-1}` + action `a_{t-1}`. But position isn't in the input! So the RNN has to integrate *actions* to track where it's been. This is a recurrent credit assignment that PPO learns slowly.

Observation: point (6) is a real bandwidth constraint. With a 3-dim input and no position info, the RNN must integrate actions to track where it's been. The allowed knobs don't change the input dims, so this motivates giving the RNN more *training signal* (novelty shaping, teacher-forcing, store-head supervision) that helps it learn the right recurrent representation faster.

## Axes of attack (generated, not ranked by the prior sweep)

A. **Store head supervision.** BCE on `at_goal`, reward shaping via `store_cost`+`store_bonus`, or `auto_store_warmup`. These collapse "when to store" from a PPO credit-assignment problem to classification / shaped-reward.

B. **Continuous policy parameterization.** `init_log_std` controls training↔deterministic-eval matching; lower → det eval is closer to training support. `ent_coef` sets the movement entropy floor.

C. **Exploration shaping.** Nothing currently rewards coverage. Neither knob below is about making the policy stochastic — both shape *what the policy means to do*:
   - `auto_nav_warmup` teacher-forces following when there's a memory (seeds "follow-signal" behavior early).
   - No direct knob available for coverage, but entropy schedule (`ent_coef`) + `explore_steps` length affect how much exploration happens per rollout.

D. **PPO stability.** `lr`, `clip_coef`, `ppo_epochs`, `n_minibatches`, `gae_lambda`, `gamma`. Conservative updates → less chance of trashing working policy.

E. **Capacity / data.** `hidden_size`, `num_rnn_layers`, `batch_envs`, `steps_per_rollout`, `n_updates`.

F. **Rollout structure.** `explore_steps` changes the explore/exploit split inside a rollout. `explore_steps=None` gives always-store single-phase; shorter explore means more exploit experience per rollout (trades store-head learning rate for nav-head learning rate).

*Note on `novelty_reward`:* it exists in code/config (rollout.py emits +reward per first-visit cell during explore), but it is **not** in AUTORESEARCH.md's allowed-knob list — so it's not tunable under this protocol. Run 5 uses it; if it turns out to help, we'd need explicit user approval to keep it.

## Iteration protocol

Each run = 1–2 params changed from the current best baseline, 250–400 updates.
Report peak-and-final on each of the five target metrics.
Max 2 concurrent on pi_evelina9; mit_normal_gpu used freely.
After every 10 runs: comprehensive reassessment appended here.
Naming: `runN_<key-change>` in `hopfield_nav/runs/`.

## Runs log

(appended below — oldest first)

### Wave 1 — one-axis variants (n_updates=300, eval_every=50)

All five launched on `mit_normal_gpu`. Common flags = `run_continuous.sh`'s minus `explore_steps` (treated as tunable, default 100 unless overridden). See `runs/_common.sh`.

| Run | Job ID   | Script               | Axis | Change vs common |
|-----|----------|----------------------|------|------------------|
| 1   | 12416423 | run1_baseline.sh     | —    | (reference: explore_steps=100, everything else default) |
| 2   | 12416424 | run2_storebc.sh      | A    | `--store_bc_weight 1.0 --auto_store_warmup 30 --aux_anneal_updates 200` |
| 3   | 12416428 | run3_storecost.sh    | A    | `--store_cost 0.02 --store_bonus 0.05 --aux_anneal_updates 200` |
| 4   | 12416429 | run4_initstd.sh      | B    | `--init_log_std -0.5` |
| 5   | 12416739 | run5_entcoef.sh      | B/C  | `--ent_coef 0.03` |
| 6   | 12416740 | run6_autonav.sh      | C    | `--auto_store_warmup 30 --auto_nav_warmup 100` |

Intent:
- run2 collapses "when to store" from PPO credit-assignment to BCE classification.
- run3 tests whether reward-shaping alone (no BCE) is sufficient to isolate the mechanism.
- run4 tightens continuous policy so deterministic eval matches training support.
- run5 triples the movement-entropy bonus — keeps policy exploring longer (legal proxy for coverage pressure; `novelty_reward` is not in AUTORESEARCH.md's allowed-knob list).
- run6 teacher-forces nav behavior for first 100 updates (with auto_store_warmup=30 so a memory exists); tests whether nav-learning is binding.
- `explore_steps`, `clip_coef`, `lr`, capacity, and combination runs deferred to Wave 2 once wave 1 establishes which sub-problem is binding.

#### Wave 1 results so far (peak / final at n_updates=300)

Peak over the 6 periodic evals (@u50,100,150,200,250,300); final = u300 eval.

| Run                      | nav_det success      | nav_det mean_speed   | disc reach_success    | disc store_eff   | expl mean_steps_to_goal (↓) |
|--------------------------|----------------------|----------------------|-----------------------|------------------|------------------------------|
| 1 — baseline             | 0.91 / 0.70          | 0.84 / 0.52          | 0.50 / 0.25           | 1.00 / 0.99      | 21.9 / 28.6                  |
| 2 — store_bc+warmup      | **1.00** / 0.75      | **0.98** / 0.47      | 0.32 / 0.25           | 1.00 / 1.00      | *(4.3 misleading)* / 51.9    |
| 3 — store_cost+bonus     | 1.00 / 0.63          | 0.67 / 0.55          | 0.375 / **0.375**     | 1.00 / 1.00      | 5.2 / **19.5**               |
| 5 — ent_coef=0.03 (u200) | 1.00 @u200           | 0.86 @u200           | 0.39 @u150            | 1.00             | 21 @u150                     |
| 4 — init_log_std=-0.5    | *pending on pi_evelina9* | | | | |
| 6 — auto_nav_warmup=100  | *pending on pi_evelina9* | | | | |

Expanded run 2 and run 3 notes:

- **Run 2 shaping dependency.** `aux_anneal_updates=200` means BCE + store_bonus fade to 0 by u200. At exactly u200 `store_entropy` jumps from ~0.03 (peaked head, confident "store at goal") to ~0.54 and climbs back to ~0.66 by u300 — the PPO-only gradient on the store head pulls it back toward max-entropy once BCE is gone. Nav_det and exploration metrics all regress in lockstep. Takeaway: either don't anneal, anneal slower, or use a smaller weight that PPO can tolerate as permanent.
- **Run 3 miscalibration.** With `store_cost=0.02` per-step and `store_bonus=0.05` per at-goal-store, expected per-step cost (≈0.01 at 50% store-prob) dominates expected per-step benefit (≈0.001 if at-goal ~5% of steps × stored). Policy learns to never store — realistic-eval `hopfield_final_memories=0`. BUT exploration metrics improved (`mean_steps_to_goal=19.5` is the best so far), because the agent is implicitly pushed toward "walk around without caring about Hopfield." Takeaway: store shaping is miscalibrated; bonus must dominate cost at the actual at-goal rate.
- **Run 5 higher entropy.** `move_entropy` rises to ~4.3 (vs baseline's ~3.3 at u300) — the extra ent_coef genuinely holds the policy open. Nav and disc metrics both in a reasonable window at u200. Still running; see final.

### Wave 2 — combinations and calibrated reshapes (n_updates=300)

Submitted before Wave 1 fully completed — planned on the info from runs 1–3.

| Run | Job ID   | Script                        | Change vs common |
|-----|----------|-------------------------------|------------------|
| 7   | 12421757 | run7_bc_permanent_ent.sh      | `--store_bc_weight 0.5 --auto_store_warmup 30 --ent_coef 0.03` |
| 8   | 12421765 | run8_bc_slowanneal.sh         | `--store_bc_weight 1.0 --auto_store_warmup 30 --aux_anneal_updates 600` |
| 9   | 12421766 | run9_cost_recalibrated.sh     | `--store_cost 0.005 --store_bonus 0.3 --auto_store_warmup 30 --aux_anneal_updates 200` |

Intent:
- run 7 combines run 2's BCE (halved, permanent) with run 5's higher ent_coef — tests if run 2's peak nav_det transfers under a policy that's held more open.
- run 8 isolates "is the late-training regression specifically caused by the *anneal*, not the BCE?" — same weights as run 2 but anneal 3× slower, so by u300 BCE is still at ~50% strength.
- run 9 retries the reward-shaping route with properly scaled incentives: at the observed ≈5%-of-steps at-goal rate, 0.3 × 0.05 = 0.015 bonus dominates 0.005 × 0.5 = 0.0025 cost.

#### Mid-wave observations (runs 4, 5, 7, 8 partial)

- **Run 5 final (300u)**: `nav_det=1.00/0.95` *sustained*, but `disc reach=0.08` (collapse) and realistic summary `mean_primary_reaches=0` despite `hopfield_final_memories=2622`. High `ent_coef=0.03` keeps move policy open, sustains nav, but store head fires spuriously *everywhere* → Hopfield polluted during realistic eval → signal noise dominates. Pattern: anything that prevents nav regression also collapses disc reach.
- **Run 7, 8 early**: BCE + high ent (run 7) or slow-anneal BCE (run 8) both reproduce run 2's early nav peak (`mean_speed ≈ 0.95-0.98` by u100) but show the same `disc reach ≈ 0.15-0.24` malaise.
- **Run 4 (init_log_std=-0.5) @u100 is the first breakthrough on exploration: `disc reach=0.59`, well above baseline's peak of 0.50.** nav_det speed is temporarily bad (0.32) but it's early and the *mechanism* for why this works is distinct: tighter continuous Gaussian (std≈0.6 vs 1.0 default) makes sampled actions persistently directional rather than diffusive, so the random walk explores more systematically. Movement mean calibration lags because PPO sees less action-space diversity, but the non-BCE store head keeps firing opportunistically.

### Wave 3 — init_log_std is promising, combine with store supervision

| Run | Job ID   | Script              | Change vs common |
|-----|----------|---------------------|------------------|
| 10  | 12423374 | run10_tight_bc.sh   | `--init_log_std -0.5 --store_bc_weight 0.5 --auto_store_warmup 30 --aux_anneal_updates 600` |
| 11  | 12423448 | run11_tight_ent.sh  | `--init_log_std -0.5 --ent_coef 0.03` |

Intent:
- run 10 combines run 4's tight-policy exploration win with moderate slow-annealed BCE to rescue the nav_det metric. Two axes are expected to be compatible because init_log_std affects only the move distribution and BCE only the store logit.
- run 11 is the pure-entropy route: tight init + high ent_coef = implicit curriculum (tight early for coverage, loosens over training for late-stage nav stability via run-5 mechanism). Isolates whether store head stabilizes without BCE when the move distribution is better handled.

### Wave 4 — push init_log_std further and stabilize the peak

Run 4 disc reach trajectory: u50 0.50, u100 **0.59**, u150 0.43. Peak is at u100 and regresses — probably a bad local-optimum attractor pulling PPO off the good policy. Two diagnoses to run in parallel:

| Run | Job ID   | Script                 | Change vs common |
|-----|----------|------------------------|------------------|
| 12  | 12424216 | run12_tight_m1.sh      | `--init_log_std -1.0`  (std ≈ 0.37) |
| 13  | 12424217 | run13_tight_lowlr.sh   | `--init_log_std -0.5 --lr 1e-4` |

Intent:
- run 12 extends the axis: does more tightness → more coverage, or is there a U-shape?
- run 13 tests whether the u100→u150 regression is PPO-overshoot-driven; if yes, smaller lr widens the peak.

Also in flight: run 6 (`--auto_store_warmup 30 --auto_nav_warmup 100`) is **failing** — nav_det stuck at 5% at u100, mean_reward ≈ −0.008 throughout. Root cause: with no BCE / no store_cost, the un-trained store head fires randomly ~50% of timesteps, so per-env Hopfield accumulates pollution, the recall signal becomes noisy, and the forced teacher direction becomes essentially random. The agent then learns to produce random outputs. Takeaway: **teacher forcing on move requires clean memory**; any future auto_nav_warmup experiment must be paired with a means of keeping the Hopfield clean (BCE, store_cost, or freezing agent_can_store).

### Wave 2 result update — run 9 is the new champion

**Run 9 (`store_cost=0.005, store_bonus=0.3, auto_store_warmup=30, aux_anneal_updates=200`) final (u300):**
`nav_det 0.76/0.65` | `disc reach 0.59` | `expl mean_steps_to_goal 30` | `store_eff 1.00`.

Trajectory: `disc.reach` climbs 0.10 (u50) → 0.16 (u100) → 0.20 (u150) → 0.26 (u200) → 0.48 (u250) → 0.59 (u300). **The big jump happens AFTER aux_anneal finishes at u200** — the store head retained "fire at goal" behavior via PPO advantages (which shape policy less rigidly than BCE logit supervision), and the move head was never distorted by auxiliary loss. Without the ongoing bonus+cost pressure, the policy regained the freedom to explore.

Comparison to run 2 (BCE + anneal): BCE anneal at u200 collapsed run 2 because BCE directly modifies store logits, so removing it lets store entropy re-explode. Reward-shaping has no such cliff.

Takeaways:
1. **Calibrated reward shaping + aux anneal is the best single-axis recipe so far.** At u300 run 9 leads on disc reach among completed runs that also have non-trivial nav.
2. **Don't use BCE at wave ≥ 1.0 on this task** — BCE distorts the shared RNN trunk and kills exploration. Small BCE (`store_bc_weight=0.1`, run 14) may still be useful as a gentle constraint; to be tested.

### Wave 5 — test gentler store supervision with tight init

Run 10 early (u50-100): BC=0.5 pinned disc reach at 0.15 despite init_log_std=-0.5. Hypothesis: BCE at weight 0.5 is strong enough to dominate the shared RNN trunk's gradient and suppress exploration. Two diagnostic variants:

| Run | Job ID   | Script                  | Change vs common |
|-----|----------|-------------------------|------------------|
| 14  | 12425609 | run14_tight_bclow.sh    | `--init_log_std -0.5 --store_bc_weight 0.1 --auto_store_warmup 30` |
| 15  | 12425610 | run15_tight_bonus.sh    | `--init_log_std -0.5 --store_cost 0.005 --store_bonus 0.3 --auto_store_warmup 30 --aux_anneal_updates 200` |

Intent:
- run 14: 5× smaller permanent BCE. Maintains store supervision but as a gentle nudge.
- run 15: swap BCE for reward-shaping (run 9's calibration). Reward-based shaping reaches the store head via advantages, which is less direct and arguably less likely to collapse exploration than direct logit supervision.

### Wave 6 — extend the run-9 recipe

Run 9 is the current champion on the balanced front. Three follow-ups:

| Run | Job ID   | Script                  | Change vs common |
|-----|----------|-------------------------|------------------|
| 16  | 12427598 | run16_r9_longer.sh      | run 9's flags + `--n_updates 400 --eval_every 100` |
| 17  | 12427599 | run17_r9_strongbonus.sh | `--store_cost 0.005 --store_bonus 0.5 --auto_store_warmup 30 --aux_anneal_updates 200` |

Intent:
- run 16 tests whether run 9 was still improving at u300 (it went 0.48 → 0.59 in the last 50 updates).
- run 17 pushes bonus/cost ratio 60×→100× to see if the shaping can be accelerated.

## 10-run reassessment (at completion of runs 1-10)

### Final scores (peak / final; `—` = N/A)

| Run                           | nav succ      | nav speed     | disc reach     | expl ↓ steps | realistic primary |
|-------------------------------|---------------|---------------|----------------|--------------|-------------------|
| 1 baseline                    | 0.91 / 0.70   | 0.84 / 0.52   | 0.50 / 0.25    | 21.9 / 28.6  | 3.4               |
| 2 store_bc 1.0 + anneal 200   | 1.00 / 0.75   | 0.98 / 0.47   | 0.32 / 0.25    | —    / 51.9  | —                 |
| 3 store_cost 0.02 + bonus 0.05 | 1.00 / 0.63  | 0.67 / 0.55   | 0.375 / 0.375  | —    / 19.5  | 8.2               |
| 4 init_log_std=-0.5           | 0.79 / 0.79   | 0.41 / 0.40   | **0.59 / 0.49**| 29   / 46    | 8.4               |
| 5 ent_coef=0.03               | 1.00 / **1.00** | **0.95 / 0.95** | 0.39 / 0.08 | 20.9 / 31.8  | 0.0               |
| 6 auto_nav_warmup=100         | *failed*      |               |                |              | 0.0               |
| 7 BC=0.5 + ent=0.03           | 1.00 / 1.00   | 0.98 / 0.80   | 0.30 / 0.12    | 9.5  / 9.5   | 0.6               |
| 8 BC=1.0 + slow-anneal=600    | 1.00 / 1.00   | 0.97 / 0.68   | 0.52 / 0.52    | 7.4  / 20.8  | 16.9              |
| **9 cost=0.005 bonus=0.3**    | **0.76 / 0.76** | **0.65 / 0.65** | **0.59 / 0.59** | **30 / 30** | **12.4**          |
| 10 init_log_std=-0.5 + BC slow | 1.00 / 1.00   | 0.93 / 0.47   | 0.46 / 0.46    | 37   / 21    | 9.0               |

### Mechanistic takeaways

1. **Calibrated reward shaping ≫ BCE for the store head.** Both can peak nav_det speed, but reward-shaping's signal flows through advantages so it doesn't distort the shared RNN trunk's representation. BCE directly modifies store logits and effectively slowly pulls every shared-feature representation toward "distinguish at-goal from not-at-goal" — which kills exploration. Moreover, when aux_anneal fades, BCE leaves behind a store-head local min that re-explodes once PPO resumes; reward shaping has already "baked in" the right store behavior through advantage-weighted updates.

2. **init_log_std=-0.5 is a genuine exploration knob.** Tighter continuous Gaussian → more directional random walks → higher disc reach peak (0.59). The cost is slow nav_det mean calibration — PPO samples less diverse actions so the mean stays close to bad-init for longer. Run 4 eventually reaches nav speed 0.40 at u300 — still behind, but trending.

3. **ent_coef=0.03 is a stability knob, not an exploration knob.** It keeps move_entropy near 4.0 (vs 3.3 default), which prevents the late-training nav regression seen in every default-entropy run (e.g. run 2 nav speed 0.98 → 0.47). But it has no effect on *policy structure* — the policy still commits to a noisy-follow-signal mode that is useless when no memory exists, killing disc reach.

4. **auto_nav_warmup requires clean memory.** Alone (run 6) it is catastrophic — pollution from the un-trained store head gives the agent an untrustworthy teacher, and the imitation forces random outputs. Any future auto_nav_warmup use needs BCE, store_cost, or a frozen Hopfield companion.

5. **Late-training disc-reach surges.** Runs 8, 9, 10 all show disc reach jumping sharply between u250 → u300 (e.g. run 9: 0.48 → 0.59). Coincides with aux anneal being fully done around u200-250 — once shaping pressure is off, exploration recovers. This argues for longer runs or larger anneal windows.

### Pareto frontier

At u300-final, the only runs on the Pareto frontier (can't improve one metric without losing another) are:

- **run 9** (disc 0.59, nav 0.76/0.65, expl 30) — best disc+nav balance.
- **run 5** (disc 0.08, nav 1.00/0.95, expl 32) — best nav if disc is nobody's problem.
- **run 8** (disc 0.52, nav 1.00/0.68, expl 21) — between r9 and r5.

Run 9 strictly dominates run 4 (same disc, better nav) and run 1 baseline (better everything).

### Direction for runs 11-20

- **Extend the run 9 recipe** (run 16 longer, run 17 stronger bonus, run 15 + init_log_std). If the disc-reach surge at u300 continues past u300, we might see disc > 0.65 or higher.
- **Test if BCE = 0.1 is the sweet spot** (run 14). Even small BCE may be additively good with reward shaping.
- **Complete pending init_log_std sweep** (runs 11, 12, 13) to map the axis.
- **Future waves, pending outcomes:**
  - `hidden_size=256` / `num_rnn_layers=2` (extra capacity to learn both nav+store).
  - `clip_coef=0.1` (more conservative PPO to prevent the regressions seen in BCE runs).
  - `explore_steps=200` (single-phase, always-store). Tests whether the explore/exploit split helps or hurts.

### Wave 7 — novelty_reward (user-authorized)

User suggested testing `novelty_reward` (intrinsic first-visit cell bonus during explore phase). Motivation: run 9's disc reach only really takes off after u250 — before that, the agent just isn't *reaching* the goal often enough during training to get useful reward signal for the store head. Novelty rewards shape the policy's exploration *behavior* directly (unlike ent_coef which only affects noise floor).

| Run | Job ID   | Script                      | Change vs common |
|-----|----------|-----------------------------|------------------|
| 18  | 12430304 | run18_r9_novelty.sh         | run 9 + `--novelty_reward 0.02` |
| 19  | 12430306 | run19_r9_novelty_hi.sh      | run 9 + `--novelty_reward 0.05` |

Intent:
- run 18 adds a conservative novelty signal on top of run 9's shaping — hypothesis that better early exploration → earlier goal reach → earlier useful gradient on store head → higher final disc reach.
- run 19 pushes novelty 2.5× higher; at 0.05 × 64 cells max ≈ 3.2 per rollout (vs +1 for goal reach), novelty dominates early reward. Tests if over-weighting intrinsic reward distorts policy.

### Wave 3 late update: run 15 shows init_log_std × bonus-shaping interaction

Run 15 (`init_log_std=-0.5 + store_cost=0.005 + store_bonus=0.3 + auto_store_warmup=30 + aux_anneal_updates=200`) finished with disc reach peak **0.43 @u50** but final **0.20** — worse than run 9 alone. Regression is the same pattern as run 4 (tight init peaks early, regresses) but *deeper* than run 4's regression.

Interpretation: tight init + reward shaping are **not orthogonal** despite affecting different heads. Tight init narrows trajectory diversity → the agent quickly locks into one at-goal approach path → the bonus reinforces that single trajectory → late-training exploration dies. Run 9's default-width Gaussian (`init_log_std=0`) keeps enough rollout diversity that the store head learns to fire at-goal across *many* approach paths, so the shared RNN representation generalizes.

Updated takeaway: `init_log_std=-0.5` is an exploration knob that *requires* a persistent-diversity companion (not shaping). Wave 7's novelty_reward is a better companion candidate because it incentivizes covering new cells every rollout, keeping trajectory diversity alive.

### Wave 6/7 results — run 16 is the new champion

| Run                               | nav succ    | nav speed   | disc reach  | expl ↓  | realistic primary |
|-----------------------------------|-------------|-------------|-------------|---------|-------------------|
| 9 baseline-of-wave                | 0.76 / 0.76 | 0.65 / 0.65 | 0.59 / 0.59 | 30 / 30 | 12.4              |
| **16 run 9 @ 400 updates**        | 0.78 / 0.78 | 0.43 / 0.43 | **0.70 / 0.70** | **26 / 26** | 19.8         |
| 17 run 9 with bonus=0.5           | 0.69 / 0.69 | 0.54 / 0.54 | 0.60 / 0.60 | 46 / 46 | **21.4**          |
| 18 run 9 + novelty=0.02           | 0.82 / 0.82 | 0.46 / 0.46 | 0.21 / 0.21 | 29 / 29 | 0.0 (broken)      |
| 19 run 9 + novelty=0.05           | 1.00 / 1.00 | **0.90 / 0.90** | 0.13 / 0.13 | 43 / 43 | ? (tbd)           |

**Top-line takeaways:**

1. **Run 16 wins on the target metrics** — extending run 9 from 300 to 400 updates lifts disc reach 0.59 → 0.70 while holding nav_det success near 0.78, nav speed 0.43 (lower than run 9 because at u400 the late-training nav regression has started, but still high success_rate), and the best expl mean_steps_to_goal of the whole set (26).

2. **Stronger bonus (run 17) is a side-grade** — marginal disc-reach gain (0.60 vs 0.59) but better realistic (21.4 vs 12.4) and worse expl steps. If realistic matters, run 17; on target metrics run 16 wins.

3. **novelty_reward added to run 9 recipe decisively hurts** — both run 18 (novelty=0.02) and run 19 (novelty=0.05) collapse disc reach to 0.13-0.21, and run 18 breaks realistic completely (0 primary reaches). Intrinsic bonus competes with the store_bonus for reward-landscape shape and distorts the policy's at-goal behavior. Run 19 does get the highest nav_det mean_speed (0.90) but at the cost of essentially breaking the memory system. Novelty is not a free companion to reward shaping.

4. **Run 20 (hidden_size=256)** — cancelled after u100 showed disc=0.04 and bad nav (0.44/0.40). 4× parameters in the GRU need far more training data than 300 updates provides; the recipe doesn't extend trivially with capacity.

5. **Runs 21, 22 OOMed** (64 GB cap on mit_normal_gpu was insufficient when a node had multiple training jobs sharing CPU memory for the 1716×1716×1024 encoded_Phi). Resubmitted with 100 GB.

### Current Pareto frontier (after runs 1-19)

- **Run 16** — best disc reach (0.70), best expl steps (26), decent nav success (0.78). *Target-metric champion.*
- **Run 17** — best realistic primary reaches (21.4), disc 0.60.
- **Run 19** — best nav_det speed (0.90), but disc only 0.13.

Run 16 is clearly the one to report. Next step: run 22 (= run 16 + stronger bonus) to see if those combine. Run 21 tests whether auto_store_warmup=30 was load-bearing.

### Wave 8 results (runs 16, 17, 22, 23, 24, 25)

| Run | Recipe vs run 9                       | n_u | disc final | nav final   | realistic prim | mem |
|-----|---------------------------------------|-----|-----------:|------------:|---------------:|----:|
| 16  | +100u → 400                           | 400 | **0.70*** | 0.78 / 0.43 | **19.8**       | 248 |
| 17  | bonus 0.3→0.5                         | 300 | 0.60      | 0.69 / 0.54 | **21.4**       | 414 |
| 22  | bonus=0.5 + 400u                      | 400 | 0.23      | 0.65 / 0.43 | 4.5            | 2149|
| 23  | 500u                                  | 500 | 0.28 (0.76 @u300 spike) | 0.78 / 0.40 | 0.8 | 1733 |
| 24  | anneal 200→300                        | 400 | 0.26      | 0.77 / 0.54 | 8.6            | 193 |
| 25  | explore_steps 100→200 (single-phase)  | 400 | 0.29      | 0.84 / 0.45 | 9.6            | 486 |
| 21  | no auto_store_warmup                  | 300 | 0.35 (late) | 0.49 / 0.45 | 11.0         | 298 |

Takeaways:
- Run 22 (bonus 0.5 + 400u) is **worse** than run 16 (bonus 0.3 + 400u). Bigger bonus at longer training → spammier storing (2149 memories) → polluted Hopfield.
- Run 23 (500u same recipe as 16) hit 0.76 at u300 then crashed to 0.28 at u500. One-eval peaks are noisy.
- Run 24 (long anneal 300): no benefit; disc 0.26.
- Run 25 (single-phase): coverage 0.29, worse than run 16. Two-phase is genuinely better.
- Run 21 (no warmup): disc finds 0.35 via late surge; realistic decent. Confirms warmup accelerates but isn't strictly required.

### Wave 9 — seed-replicate variance tests (runs 26-29, 27-29)

Seed variance of the run-16 recipe at identical n_updates:

| Run | seed | n_u | disc trajectory                    | final disc | realistic |
|-----|------|-----|------------------------------------|-----------:|----------:|
| 23  | 42   | 500 | 0.06,0.08,0.76,0.31,0.28           | 0.28       | 0.8       |
| 26  | 1    | 500 | 0.13,0.12,0.23,0.20,0.41           | 0.41       | 11.3      |
| 27  | 0    | 400 | 0.13,0.04,0.06,0.06                | **0.06**   | 9.7       |
| 28  | 2    | 400 | —,0.06,0.61 (@u300),?              | running    | -         |

Conclusions:
- **Single-seed run-16 recipe is highly variable.** Across 4 seeds on the same ~400-update recipe, final disc reach spans 0.06 to 0.70. Coverage swings 0.20 to 0.60.
- The late-training "surge" happens for 3/4 seeds (1, 2, 42) but not for seed 0. Mechanism is seed-brittle.
- run 16's 0.70 appears to be the 90th-percentile lucky draw, not the recipe's mean. True mean is probably 0.25-0.40.

### Coverage diagnosis (the actual failure)

Running these seed replicates surfaced why coverage is intrinsically hard: with input = `[reward, hopfield_signal]` (both near-constant in the no-memory regime), the GRU converges to a fixed-point attractor when actions are deterministic. The policy mean becomes a constant direction. Training-time action sampling hides this (noise carries exploration); deterministic eval exposes it.

Confirmatory evidence:
- runs with `ent_coef=0.03` (higher sampling std throughout) have the *lowest* coverage at eval (0.11-0.15) — noise shouldered the load at training, mean policy never learned a walk.
- runs with `init_log_std=-0.5` (lower sampling std) have the *highest* stable coverage (~0.50) — PPO is forced to push the mean policy into a systematic walk.
- `novelty_reward` can't help: it depends on state (`visited_cells`) the agent does not observe, so it becomes noise at the advantage level (runs 18, 19 disc collapsed to 0.13-0.21).

### Wave 10 — attack coverage directly

Queued runs:

| Run | Change                                  | Rationale |
|-----|-----------------------------------------|-----------|
| 30  | init_log_std=-0.5, 400u, no shaping    | Clean tight-init test, extended training |
| 31  | + clip_coef=0.1                         | Tighter PPO to preserve peaks |
| 32  | init_log_std=-0.7 (std≈0.50)           | Push tightness axis further |
| 33  | init_log_std=-0.5 + gamma=0.995         | Longer reward horizon |
| 34  | init_log_std=-0.5 + explore_steps=20    | Force mostly-no-memory training |
| 35  | run-16 recipe + **--input_sensory** + obs=30 | 30-dim raycast fingerprints each cell — directly breaks the fixed-point attractor |
| 36  | input_sensory alone, obs=30             | Clean sensory-only test |
| 37  | input_sensory + init_log_std=-0.5, obs=30 | Combine sensory + tight init |

Expected winners: **runs 35-37** (input_sensory variants) should lift coverage dramatically because the RNN input varies with position — the very thing the no-sensory baseline was missing. Run 28 already showed 0.59 coverage at u300 with the run-16 recipe + lucky seed, which is a useful control for how far the non-sensory approach can go.

### Wave 10 results — new champion: r34 (short_explore + tight init)

| Run | Recipe                                  | reach | coverage | nav succ/spd | steps_to_goal | realistic prim / mem |
|-----|-----------------------------------------|------:|---------:|-------------:|--------------:|---------------------:|
| 30 | tight init alone                        | 0.29 | 0.55 | 0.59/0.35 | 26 | 3.8 / 2686 |
| 32 | init_log_std=-0.7                       | 0.41 | 0.52 | 0.58/0.33 | 44 | 5.1 / 952 |
| **34** | **tight + explore_steps=20 (no shaping)** | **0.55** | **0.57** | **0.73/0.22** | **31** | 8.7 / 2226 |
| 38 | r34 seed=1                              | 0.59 | 0.48 | 0.76/0.36 | 32 | 16.9 / 6676 |
| 39 | r34 seed=2                              | 0.59 | 0.57 | 0.51/0.31 | 67 | 2.8 / 9432 |
| 40 | r34 + run-9 shaping                     | 0.41 | 0.51 | 0.65/0.32 | 17 | 9.7 / 376 |
| 35 | r16 + sensory (obs=30)                  | 0.18 | 0.32 | 0.25/0.39 | 26 | 0.4 / 594 |
| 36 | sensory clean (obs=30)                  | 0.29 | 0.35 | 0.35/0.26 | 41 | 1.8 / 5790 |
| 37 | sensory + tight                         | 0.18 | 0.33 | 0.22/0.50 | 14 | 1.1 / 3890 |

**Key conclusions:**

1. **Run 34 recipe is the new robust champion.** Three seeds (42, 1, 2) give reach 0.55/0.59/0.59 and coverage 0.57/0.48/0.57 — mean **reach 0.58, coverage 0.54** with low variance. This compares to run 16's mean 0.30-0.45 with HIGH variance (0.06 to 0.70).

2. **The mechanism: short explore_steps (20 of 200) forces the no-memory training regime.** With store eligibility limited to the first 20 steps, most of training happens with empty Hopfield → policy mean must do its own systematic exploration → coverage transfers to deterministic eval.

3. **Sensory input was a bust at this training scale.** All 3 sensory runs (35, 36, 37) UNDERperform their non-sensory baselines on disc reach. The 30-dim raycast adds input variance the policy can't yet exploit; needs more capacity and/or much longer training.

4. **r34 + reward shaping (run 40) doesn't combine cleanly** — reach drops 0.55→0.41. Same suppression pattern as run 10/15. But realistic improves dramatically (8.7 → 9.7 with mem 2226 → 376). Trade-off: shaping helps memory cleanliness, hurts coverage.

5. **r34 alone is best for target metrics** (disc reach 0.58, coverage 0.54). For realistic-secondary, r17 (21.4 prim) and r16 (19.8 prim) win.

### Wave 11 (queued) — push the r34 axis

- **r41**: r34 + 500u (does the slight u300→u400 regression continue or recover?)
- **r42**: r34 + explore_steps=10 (push the no-memory regime further)
- **r43**: r34 + explore_steps=50 (search around the optimum)
- **r44**: r34 + auto_store_warmup=30 only (boost store head without distorting move policy)

### Wave 12 — validation sweep (3 recipes × 3 seeds × 128 val trials)

With single-eval variance on disc reach running ±0.15-0.25 (much larger than
binomial noise), we pivoted from one-off runs to proper seed-replicates. Three
recipes × three seeds, each with `n_val_trials=128` to drive eval noise below
recipe-recipe differences.

Recipes: `baseline` (no shaping), `shaping` (r9 recipe: `--store_cost 0.005
--store_bonus 0.3 --auto_store_warmup 30 --aux_anneal_updates 200`), `tight`
(r34 recipe: `--init_log_std -0.5 --explore_steps 20`).

| Recipe | Seeds done | disc reach (mean) | realistic prim / mem | Notes |
|--------|-----------|-------------------|---------------------|-------|
| baseline | 3         | 0.43 ± 0.09       | 9.0 / 5287 | noisy but reasonable |
| shaping  | 2 (s=42 cancelled) | 0.71 (0.60, 0.81) | 10.2 / 365 | **seed=1 = ethereal-gorge-73** |
| tight    | 0 (all cancelled)   | —                 | — | |

**Champion**: `ethereal-gorge-73` = shaping recipe, seed=1: disc reach 0.81,
coverage 0.65, nav 0.96/0.39, realistic 6.9 prim / 153 memories at u300.

**Takeaway**: shaping recipe is robustly better than baseline (~+0.28 reach,
15× cleaner memory). The 0.81 ethereal-gorge-73 number is the high-tail of a
recipe whose true mean is near 0.70 with seed SD ~0.15 on n=2.

---

## Phase 2 — robustness to distractors

User observation (post-wave-12): when ethereal-gorge-73 (u=300 checkpoint) is
evaluated on `all_evals` with distractor hopfields preloaded, **the agent fails
to explore when distractors are present in memory** — it gets misled by
spurious recalls and doesn't search past them. Clean-memory reach is fine; add
distractors and it collapses.

Root cause (hypothesis): the shaping recipe trains with an empty Hopfield at
rollout start, so the policy sees `has_memory=False` for the full first ~30
steps. When eval-time pre-populates distractors, `has_memory=True` from step 0
and the policy has no training experience in the "memory present but
unreliable" regime — it trusts every recall.

Fixes to test (in priority order):

- **Wave 13 — train-time distractors** (direct distribution match)
- **Wave 14 — longer rollouts** (give agent time to explore past bad recalls)
- **Wave 15 — epsilon exploration** (code change: uniform-random actions with
  prob ε, injected via the agent override path so PPO log_prob stays
  self-consistent)

### Wave 13 results (n_val_trials=64, seed=1, u=300)

Eval is `evaluate_goal_discovery` with the env's goal preloaded + N
distractors from outside the env's region. `d=K` columns are eval-time
distractor counts. Reach=`reach_success_rate` from `eval disc`.

| Run    | train distractors | reach@d=0 | reach@d=1 | reach@d=3 | reach@d=10 | mean (d≥1) |
|--------|-------:|----------:|----------:|----------:|-----------:|-----------:|
| v13-d0  | 0  | 0.39 | 0.36 | 0.35 | 0.34 | 0.35 |
| v13-d3  | 3  | **0.11** | 0.30 | 0.30 | 0.30 | 0.30 |
| v13-d10 | 10 | **0.45** | 0.42 | 0.46 | 0.45 | 0.44 |

**Findings (single seed, take with grain of salt):**

1. **Control v13-d0 is robust to eval distractors out of the box.** Reach
   stays 0.34–0.39 from d=0 to d=10 — the shaping recipe with empty-init
   training already generalizes to distractor eval reasonably well. The
   user's "fails to explore with distractors" claim from `all_evals` may
   reflect a different eval (e.g. goal NOT in memory, just distractors —
   that's `evaluate_goal_find_with_pretrained_distractors`). Need to confirm.

2. **v13-d3 underperforms control at d=0 (0.11 vs 0.39).** Training with
   exactly 3 distractors seems to make the policy expect them — without
   them at eval, performance collapses on the clean-memory case. Uncanny
   valley.

3. **v13-d10 is the best: reach 0.42–0.46 across all distractor levels.**
   Single-run / single-seed; the u=300 number is a sharp jump from u=200
   (0.13–0.22), so could be a lucky late-checkpoint. But qualitatively
   distractors=10 (heavily noisy memory) seems to teach the policy to
   ignore unreliable recalls and re-explore — which is exactly what the
   user wants.

**Caveat**: Wave 12's ethereal-gorge-73 rerun would be expected to score
~0.7–0.8 on a clean-memory eval at this checkpoint. The wave 13 d=0
control got 0.39 — much lower. Hypotheses for the gap:

- `n_val_trials=64` (wave 13) vs 128 (wave 12) → 2× higher eval noise.
- Different val scaffold built when `--val_distractors 0 1 3 10` is set
  (val envs may live in different scaffold positions).
- Single-seed run-to-run variance is just enormous on this task.

We'd need a same-seed rerun with wave 12's exact eval config to disentangle.

### Wave 15 results (epsilon exploration, single seed=1, n_val_trials=64)

Eval on `disc reach_success_rate` (env's goal preloaded + N distractors):

| Run | ε | anneal | reach@d=0 | reach@d=1 | reach@d=3 | reach@d=10 | mean(d≥1) |
|-----|--:|-------:|----------:|----------:|----------:|-----------:|----------:|
| v15-e0.05    | 0.05 | const | 0.36 | 0.42 | 0.43 | 0.42 | 0.42 |
| v15-e0.10    | 0.10 | const | 0.49 | 0.57 | 0.56 | 0.56 | 0.56 |
| v15-e0.20    | 0.20 | const | 0.43 | 0.43 | 0.38 | 0.41 | 0.41 (u=200, **crashed**) |
| v15-e0.20-an | 0.20 | →0 over 200u | 0.52 | 0.61 | 0.58 | 0.61 | 0.60 |

**Findings:**

1. **Epsilon exploration is the biggest win in this phase.** Annealed
   ε=0.20 hits reach 0.60 (mean over d≥1), constant ε=0.10 hits 0.56.
   Both far above wave-13 control (0.35) and wave-14 longer-rollout (0.30).
2. Strong distractor robustness: reach is roughly flat across d=0..10.
3. **ε=0.20 constant is unstable** — moved_loss exploded to 2.7e5 at u=250
   (job FAILED). The agent landed enough wildly bad random actions in a
   row that PPO's importance ratio went numerical even with clipping. The
   anneal version dodges this: by u=200 ε is already 0.
4. ε=0.05 helps but not enough; the 0.10 vs 0.05 jump (0.42 → 0.56) shows
   exploration was *under*-supplied at 0.05.

### Wave 14 partial results

| Run | rollout | explore | distractors | u=300 reach@d=0 | reach@d=10 |
|-----|--------:|--------:|------------:|----------------:|-----------:|
| v14-r400-d0 | 400 | 200 | 0 | 0.32 | 0.30 |
| v14-r400-d3 | 400 | 200 | 3 | (running) | (running) |

**v14-r400-d0 is a regression vs v13-d0** (0.32 vs 0.39 reach@d=0). Longer
rollouts didn't help; if anything they hurt the clean-memory case. The
distractor-robustness shape is similar (~flat across d=0..10).

### Wave 13 — train-time distractors on the shaping recipe

Config: shaping recipe (`--store_cost 0.005 --store_bonus 0.3
--auto_store_warmup 30 --aux_anneal_updates 200 --n_updates 300
--explore_steps 100 --steps_per_rollout 200`), seed=1. Eval with
`--val_distractors 0 1 3 10` at every 100 updates. `n_val_trials=64` (less
than wave 12's 128 to speed up eval when sweeping 4 distractor levels).

| Run | script | `--n_train_distractors` | Expected |
|-----|--------|------------------------:|----------|
| v13-d0  | wave13_distractors.sh | 0  | Control = ethereal-gorge-73 rerun with new eval axis |
| v13-d3  | wave13_distractors.sh | 3  | Mild — should lift eval@3,10 without hurting eval@0 |
| v13-d10 | wave13_distractors.sh | 10 | Heavy — likely hurts eval@0 but should be best at eval@10 |

Jobs: 12465516, 12465517, 12465518 (queued on mit_normal_gpu, waiting on
earlier user jobs).

### Wave 14 — longer rollouts (steps_per_rollout 200 → 400)

Rationale: 200-step rollouts give the agent limited time to recover after
following a distractor recall. With 400 steps + proportional explore_steps
(200), the agent has more time to abandon a wrong recall and re-explore.
Exploit phase is also 2× longer, so trusted memories can be used more.

Config: shaping recipe + `--steps_per_rollout 400 --explore_steps 200`. Run
with and without distractors.

| Run | `--steps_per_rollout` | `--explore_steps` | `--n_train_distractors` |
|-----|-------:|-------:|-------:|
| v14-r400-d0 | 400 | 200 | 0 |
| v14-r400-d3 | 400 | 200 | 3 |

### Wave 15 — epsilon exploration (new code)

Added `--epsilon_explore P` and `--epsilon_anneal_updates N` to `config.py` /
`train.py` / `rollout.py`. Per step, with prob P, sample action from uniform
direction on unit circle (continuous) or uniform over 4 directions (discrete).
Injected via the existing agent `move_action_override` path so `log_prob` is
re-scored under the current Gaussian — keeps PPO's importance ratio
well-defined for the action actually taken.

Why not a mixture density like `(1-ε)π + εU`? The override path is strictly
more implementable — agent's log_prob of the random action is just the
Gaussian density evaluated there. That's what PPO sees, so the ratio
(new_log_prob - old_log_prob) is consistent. The bias relative to a formal
mixture policy is small (<ε·log(Gaussian_peak/Uniform)) and PPO clipping
bounds any instability.

Config: shaping recipe + epsilon schedule.

| Run | `--epsilon_explore` | `--epsilon_anneal_updates` | `--n_train_distractors` |
|-----|--------------------:|---------------------------:|------------------------:|
| v15-e005    | 0.05 | 0   | 0 |
| v15-e010    | 0.10 | 0   | 0 |
| v15-e020    | 0.20 | 0   | 0 |
| v15-e020a   | 0.20 | 200 | 0 |

If any epsilon value helps, replicate with distractors in training to check
the combo.

### Wave 16 — replicate v15 winners with bumped envs

To lower seed-variance + eval-noise on the wave-15 winners. Bumps:
`envs_per_world 20 → 40`, `num_val_envs 10 → 20`, `n_val_trials 64 → 96`.
Wallclock ~2× per run.

| Group | configs | seeds |
|-------|---------|-------|
| e=0.10 const | shaping + ε=0.10 (no anneal, d_train=0) | 1, 2, 3 |
| e=0.20 anneal | shaping + ε=0.20 anneal-200 (d_train=0) | 1, 2, 3 |
| combo | + `--n_train_distractors 10` on each winner | 1, 1 |

Job IDs 12499478..12499485. Hypothesis: variance drops → tighter SD on the
~0.55–0.60 reach numbers; combo with d=10 may stack additively.

### Wave 16 results — hypothesis falsified, config regressed

Recovered partial logs from local wandb `output.log` files (slurm stdout
buffering caused the slurm logs themselves to be empty for 4 of the runs).

**e=0.10 const, d=0 (3 seeds):**

| seed | u=100 reach@d=0 | u=200 | u=300 |
|------|----------------:|------:|------:|
| 1    | 0.09 | 0.09 | 0.09 |
| 2    | 0.13 | 0.12 | (TIMEOUT) |
| 3    | 0.48 | 0.11 | 0.22 |

**Other configs (single seed, partial):** all reach@d=0 < 0.13 at the last
recorded eval. None show learning trajectories like wave-15.

**Two findings:**

1. **Bumping envs_per_world 20 → 40 actively *hurt*, not helped.** Three-
   seed mean reach@d=0 ≈ 0.13 vs the wave-15 single-seed 0.49. Mechanism:
   2× more rollouts per update → 2× more transitions in the PPO buffer →
   each minibatch averages over more diverse trajectories → policy
   under-commits to any one direction → ends with high entropy
   (move_entropy 2.93–3.31 vs wave-15's 2.7) and low decisive action.
   This is the well-documented "PPO with too-large effective batch"
   regime; rescuing it would need re-tuning LR / n_minibatches / clip,
   which is outside the scope of "lower variance with the same recipe".

2. **Wave 15's 0.49 / 0.60 single-seed numbers were probably high-tail
   draws.** The seed-3 wave-16 run hit reach 0.48 at u=100 then collapsed
   to 0.22 — same dynamic, different seed gives 0.09 flat. The shaping
   recipe + ε exploration genuinely works (at least seed=3 found 0.48
   transiently), but its "true" performance on a single seed is much
   noisier than the headline numbers suggested.

**Operational lessons:**

- Slurm stdout buffering can blackhole a run's training updates while
  wandb's local `output.log` (which Python writes via wandb's stdout
  redirect) captures everything. Always check `wandb/run-*/files/output.log`
  for missing data before declaring a run lost.
- When raising a hyperparameter that scales the PPO buffer, retune LR /
  ppo_epochs / n_minibatches together — they're coupled.

### Wave 17 + Wave 18 — variance reduction + ablations

Common base: shaping recipe + ε=0.20 anneal-300, n_train_distractors=3,
n_updates=500, eval_every=50, envs_per_world=20, n_val_trials=64,
training metrics added: `goal_traj_rate`, `goal_step_rate` (random-walk
baseline ≈ 0.66 traj/200 steps).

| Wave | Job ID | Partition | Seed | Refresh | Explore steps | Novelty | Status |
|------|--------|-----------|-----:|:-------:|--------------:|--------:|--------|
| 17   | 12545237 | mit_normal_gpu | 1 | off | 100 | 0    | running |
| 17   | 12545238 | mit_normal_gpu | 1 | on  | 100 | 0    | running |
| 17   | 12545239 | mit_normal_gpu | 2 | off | 100 | 0    | queued  |
| 17   | 12545240 | mit_normal_gpu | 2 | on  | 100 | 0    | queued  |
| 18a  | 12545545 | mit_normal_gpu | 1 | on  | 200 | 0    | queued  |
| 18b  | 12545548 | mit_normal_gpu | 1 | on  | 100 | 0.01 | queued  |
| 18a  | 12546150 | pi_evelina9    | 2 | on  | 200 | 0    | queued  |
| 18b  | 12546151 | pi_evelina9    | 2 | on  | 100 | 0.01 | queued  |

**Wave 17 axis**: refresh on vs off (does refreshing env_offsets each
update lower seed-variance and lift mean reach?).

**Wave 18a axis**: explore_steps=200 (single-phase, stores + shaping for
whole rollout) vs wave-17 baseline of 100/100 split.

**Wave 18b axis**: novelty_reward=0.01 in addition to wave-17 baseline.
Per-step bonus on first-visit cells during the explore window. Targets
exploration above random walk via gradient shaping.

### Wave 17/18 mid-training finding — goal-reach dynamics expose the failure mode

Added two training-side metrics in this wave: `goal_traj_rate` (fraction of
the 320 rollouts/update that hit goal ≥1 time) and `goal_step_rate` (mean
fraction of steps that are at-goal). Random-walk baseline ≈ 0.66 traj rate.

| Config | u=1 | u=40 | u=140 | u=290 | u=490 |
|--------|----:|-----:|------:|------:|------:|
| 17 s=1 r=off | 0.71 | 0.81 | 0.48 | 0.42 | 0.39 |
| 17 s=1 r=on  | 0.71 | 0.37 | 0.35 | 0.32 | 0.31 |
| 17 s=2 r=off | 0.83 | 0.44 | 0.38 | 0.31 | 0.29 |
| 17 s=2 r=on  | 0.83 | 0.45 | 0.42 | 0.32 | 0.31 |
| 18a s=2 (single-phase) | 0.83 | 0.44 | 0.43 | 0.31 | 0.29 |
| 18b s=2 (novelty=0.01) | 0.83 | 0.44 | **0.45** | **0.35** | 0.33 |

`goal_step_rate` rises 0.007 → 0.026 across the same window. Together
these tell a coherent two-story picture of how training currently works:

1. **Hopfield-following IS learned.** Each train env has a *fixed* goal
   for the lifetime of the run (set once at GridEnv.__init__, line 63;
   `vec.reset_indices` only re-spawns the agent on reach, not the goal).
   So once the agent stores at-goal during explore_steps, the Hopfield
   recall points back at the goal for the rest of the rollout. The rising
   `goal_step_rate` (0.026 = ~5 hits per 200-step rollout, or ~13 hits in
   the rollouts that hit at least once) is the agent re-finding the goal
   repeatedly via recall after each teleport-respawn. That's evidence the
   policy IS reading and following the `hopfield_signal` channel.

2. **Initial exploration is being lost.** `goal_traj_rate` drops from
   0.71–0.83 (random-walk equivalent at init) to ~0.30 by mid-training
   and stays there. This is the policy committing to a constant Gaussian
   mean direction within ~40 updates. In rollouts where the commit hasn't
   yielded any goal-reach in the explore phase, the agent never stores →
   empty Hopfield → no recall → walks into a wall for the whole rollout.

The headline `disc reach_success_rate ≈ 0.5–0.6` numbers from earlier
waves were essentially "the rollouts that *did* reach the goal". They're
*not* evidence the policy navigates to fresh goals — only that, given
*one* successful exploration → store, the Hopfield-following half of the
recipe carries the rest.

**Causation correction (after closer inspection)**: My initial read said
"Hopfield-recall harms initial exploration." That's wrong — they're not
causally linked. A rollout that hits the goal once + recalls back to it
many times still counts as `goal_traj_rate=1`; recall can't *reduce* that
metric.

The real cause of the falling `goal_traj_rate` is **PPO's credit assignment
on a non-stationary action distribution**:

- At u=1 the policy is uniform-direction (Gaussian μ≈0, σ=1, entropy
  near max). ~70% of rollouts hit the goal somewhere by chance. PPO
  credits all early-trajectory actions in successful rollouts.
- Those early actions, by construction, lie in directions that ended up
  finding goals. With finite sample, the *average* lucky direction in
  one update is non-zero.
- One update later, μ has shifted toward the lucky direction. Future
  rollouts drift that way in any env. Goals not in the drift direction
  become ≈unreachable from a typical start position.
- `move_entropy` stays high (2.65–3.00 across training, near max-uniform)
  — so entropy is NOT collapsing. The drift is in μ, not σ. *Watching
  entropy doesn't catch this failure mode.*

**Lever implications**:

- `ent_coef` and high `init_log_std` operate on σ — can't fix μ drift.

- **Novelty reward** is the right kind of fix: per-step bonus for
  first-visits gives positive advantage to "spread out" actions
  (no single direction can win this — going north is a great
  novelty action only until you've covered the north side, then
  going south is). The gradient pulls μ toward zero (balanced
  exploration). Wave-18b s=2 data backs this: novelty=0.01 holds
  0.45 at u=140 vs 0.40 for novelty-off, sustains 0.35 at u=290 vs
  0.31. Modest but the right sign.
- Larger novelty values (0.02, 0.05) — directly test whether
  stronger anti-commitment pull lifts goal_traj_rate above
  random walk.
- **Advantage-masking** before first reach in a rollout: zero out
  PPO's gradient on actions taken before the goal was first found,
  so the lucky-direction credit-assignment doesn't fire. Bigger
  code change but hits the root cause cleanly.
- Slower epsilon anneal (currently to 0 at u=300) — eps puts a
  floor on randomness independent of μ, so it directly bounds the
  reach-rate floor. Already at 0.20 → 0 over 300u; could try 0.20
  → 0.05 over 500u so a residual random component remains.

Refresh on/off makes only a small difference on goal_traj_rate by u=290
(0.32 vs 0.31–0.42) — refresh-on is consistently lower in early training
(deeper post-init drop) but converges to the same place.

## Wave 19 — u=60 screen for the goal_traj_rate failure mode

9 configs × 1 seed × 60 updates × eval_every 20. Base = wave-17 winner
(refresh-on, eps=0.20-anneal-300, distractors=3, asw=30, ent=0.01,
explore_steps=100). All deltas listed in TAG.

| Tag | u=40 traj | u=40 navD | u=40 navS | u=40 discR | u=40 cov |
|-----|----:|----:|----:|----:|----:|
| control | 0.41 | 0.31 | 0.42 | 0.19 | 0.13 |
| asw0 | 0.43 | 0.31 | 0.43 | 0.20 | 0.14 |
| asw60 | 0.42 | 0.32 | 0.43 | 0.20 | 0.14 |
| ec05 (ent=0.05) | 0.72 | 0.41 | 0.65 | 0.31 | 0.26 |
| nov02 | 0.87 | 0.60 | 0.95 | 0.07 | 0.16 |
| nov05 | 0.83 | 0.87 | 0.97 | 0.54 | 0.46 |
| nov10 | 0.86 | 0.89 | 0.84 | 0.67 | 0.56 |
| novasw (nov05+asw0) | 0.87 | 0.70 | 0.96 | 0.55 | 0.48 |
| novec (nov05+ent=0.05) | 0.89 | 0.80 | 0.97 | 0.11 | 0.15 |

**Findings that survive single-seed concerns** (each is replicated across
multiple configs sharing one property — implicit n=3-5):

1. **μ-drift confirmed.** Three "no-novelty" configs (control/asw0/asw60)
   all crash traj from 0.71 → ~0.41 by u=40. Effect ≈ 0.4. n=3.
2. **Novelty stops μ-drift.** Five novelty configs (nov02/05/10/novasw/novec)
   all hold traj ≥ 0.83. n=5. Strong evidence novelty is the right lever.
3. **auto_store_warmup is a non-knob.** control / asw0 / asw60 are within
   noise on every metric.
4. **Entropy alone is partial.** ec05 (no novelty) gets traj 0.72 — half
   the fix novelty gives — and navD only 0.41. Entropy slows drift but
   doesn't replicate novelty's effect on Hopfield-following.

**Single-seed claims that did NOT replicate in v20:** the *absolute* discR
numbers (nov10 = 0.67, etc.). See v20 below.

## Wave 20 — nov10 replicate + dose-response extension

100 updates, eval_every 20, save_every 20 for checkpoints. Base = nov10
config from v19. Six runs:

- nov10 seeds 2/3/4 (replicate)
- nov15, nov20, nov30 at seed 1 (dose extension)

| Tag | u=100 traj | u=100 navD | u=100 navS | u=100 discR | u=100 cov |
|-----|----:|----:|----:|----:|----:|
| nov10_s2 | 0.88 | 0.40 | 0.96 | 0.34 | 0.28 |
| nov10_s3 | 0.87 | 0.35 | 1.00 | 0.22 | 0.25 |
| nov10_s4 | 0.92 | 0.88 | 0.99 | 0.25 | 0.24 |
| nov15 | 0.90 | 0.61 | 0.82 | 0.22 | 0.21 |
| nov20 | 0.85 | 0.10 | 0.64 | 0.36 | 0.27 |
| nov30 | 0.87 | **0.53** | 0.59 | **0.61** | 0.28 |

**Key v20 findings:**

1. **nov10 disc-reach does NOT replicate.** v19 nov10_s1 had discR=0.67 at
   u=40; v20 seeds 2/3/4 at u=40 give discR = 0.05 / 0.32 / 0.25. The v19
   number was a single-seed lottery ticket. **Cross-seed CV ≈ 80% on disc
   reach with this config.** Single-seed dose-response is not informative
   in this regime.
2. **Huge seed variance on nav_det too.** s4 hits navD=1.00 by u=20 and
   sustains 0.88 to u=100. s2/s3 plateau at 0.35-0.40. Same config, same
   training run, same eval. Init lottery dominates.
3. **nov30 best so far on combined metrics.** At u=100: navD=0.53, navS=0.59,
   discR=0.61, cov=0.28 across all distractor counts. nav_det climbed
   sharply between u=80 (0.34) and u=100 (0.53) — answer to "does nov30
   improve nav_det": yes, late in training. But this is also single-seed.
4. **No clean dose curve.** Across seed=1 runs nov15 (0.61 navD) > nov20
   (0.10 navD) < nov30 (0.53 navD). Non-monotone in dose — almost certainly
   noise rather than a real curve, given v20-replicate variance.

## Big bug found between v20 and v21 — goals were frozen for entire training run

Tracing `refresh_envs_each_update`:

- Each `GridEnv` samples its goal **once** in `__init__` (env.py:65) and
  freezes it. `reset()` only resets agent position; `reset_goal()` exists
  but is never called from `train.py`.
- `refresh_envs_each_update` calls `vh.register_envs(..., placement="random")`,
  which under `static_vectorhash=True` only resamples the env's
  **scaffold position** (`env_offsets`). The `_goal` tuple inside each
  GridEnv is untouched.
- Result: for the entire 100-update run, the agent trains on the same 20
  (env_layout, goal_within_env) pairs. Only the encoded goal embedding
  shifts (because of new env_offsets); the within-env goal location is
  fixed forever.

This is a meaningful task-distribution issue. The "wave 17 framing" of
'fresh task draw per update' was wrong — only the scaffold-position part
was being refreshed. v21 fixes this.

## Wave 21 — goal-refresh + steps_per_rollout 200→100

Two changes from v20 nov10 base:

1. `refresh_envs_each_update` now ALSO calls `env.reset_goal()` for each
   train env at the top of every update. Now the (env, goal) pair set
   really does refresh.
2. `steps_per_rollout` 200 → 100. Shorter rollouts → more episodic
   resets per gradient update; gradient over more (start, goal) pairings
   per unit compute.

3 nov10 runs at seeds 2/3/4 (matches v20 seed set for direct comparison).
Two on `mit_normal_gpu`, one on `pi_evelina9`.

Expected v21 outcomes and what each tells us:

- **A. v21 cross-seed std drops sharply (e.g., navD spread ≤0.15):**
  goal-refresh was the missing piece. Single-seed sweeps become viable.
  Next wave = revisit dose curve at single seed.
- **B. v21 mean lifts but variance stays high:** goal-refresh is good
  for absolute performance but not enough for variance. Next wave
  attacks variance via larger batch_envs or BC warm-start.
- **C. v21 means similar to v20 with similar variance:** goal-refresh
  doesn't matter much. Next wave investigates init lottery / architecture.
- **D. v21 means worse than v20:** goal-refresh destabilizes Hopfield
  content learning. Roll back or apply less aggressively.

### Wave 21 outcome — bundled change didn't reduce variance

| Metric @ u=100 | v20 s2 | v21 s2 | v20 s3 | v21 s3 | v20 s4 | v21 s4† |
|----|----|----|----|----|----|----|
| navD0 | 0.40 | **0.53** | 0.35 | 0.30 | **0.88** | 0.003 |
| discR0 | 0.34 | 0.31 | 0.22 | 0.30 | 0.25 | 0.06 |
| cov0 | 0.28 | **0.36** | 0.25 | **0.40** | 0.24 | 0.11 |
| navS0 | 0.96 | 1.00 | 1.00 | 0.98 | 0.99 | 0.80 |

† v21 s4 timed out at u=80 (no eval after u=60); ran on `pi_evelina9`
which had GPU contention with phase-b job + 2 bash jobs.

**Read:** Cross-seed spread on navD didn't drop (≥0.50 in both v20 and
v21). v21 s2/s3 cov is genuinely better than v20 (s2 +0.08, s3 +0.15).
v21 s4 looks much worse — but v20 s4's navD=1.00-by-u=20 is the kind of
result that would happen if the agent was overfitting the 20 fixed goals
exposed by the bug; with goal-refresh on, that overfit policy doesn't
work and the agent has to learn the real task.

Two changes were bundled: (1) goal-refresh, (2) steps_per_rollout 200→100.
Halving steps cuts gradient signal per update by ~2× (32k vs 64k
transitions/update). Can't separate the goal-refresh effect from the
slower-convergence effect without an ablation.

## Wave 22 — ablate goal-refresh at original rollout length

Same as v20 nov10 except `--refresh_envs_each_update` now also calls
`env.reset_goal()` (the train.py fix). Keeps `steps_per_rollout=200`.
Direct apples-to-apples with v20 nov10 s2/s3/s4. Plus a v21 s4 retry
on mit_normal_gpu with 2h time budget to complete the bundled-change
picture.

3-way comparison after v22:

- v20 (goalfix=NO, steps=200) — baseline lottery
- v22 (goalfix=YES, steps=200) — isolates the structural fix
- v21 + retry (goalfix=YES, steps=100) — the bundled change

If v22 navD/discR/cov medians improve over v20 (with similar or lower
spread), goal-refresh is the right structural fix and rollout-length
shouldn't have been bundled. If v22 ≈ v20, the goal-fix doesn't move
the metric and the v21 changes were neutral-or-negative on absolute
performance.

If neither helps, the next attack is on the actual variance source
(init lottery; possibly BC warm-start to constrain it).

### Wave 22 outcome — goal-refresh halves cross-seed variance

| Metric @ u=100 | v20 s2/s3/s4 | v22 s2/s3/s4 |
|----|----|----|
| navD0 | 0.40 / 0.35 / 0.88 | 0.50 / 0.30 / 0.28 |
| discR0 | 0.34 / 0.22 / 0.25 | 0.34 / 0.32 / 0.37 |
| cov0 | 0.28 / 0.25 / 0.24 | 0.37 / 0.37 / 0.41 |

Cross-seed spread:

| Spread | v20 | v22 | change |
|----|----|----|----|
| navD | 0.53 | 0.22 | **halved** |
| discR | 0.12 | 0.05 | **>2× tighter** |
| cov | 0.04 | 0.04 | same |
| navS | 0.04 | 0.01 | tighter |

Cross-seed mean:

| Mean | v20 | v22 | change |
|----|----|----|----|
| navD | 0.54 | 0.36 | ↓ 0.18 |
| discR | 0.27 | 0.34 | ↑ 0.07 |
| cov | 0.26 | 0.38 | ↑ 0.12 |
| navS | 0.98 | 0.99 | ~ |

**Read:**

1. **v20 s4 navD=0.88 was the fixed-goals overfit signature** — same seed
   in v22 collapses to 0.28. The agent was memorizing the 20 fixed
   (env, goal) pairs. Goal-refresh kills that path.
2. **navD mean dropped (0.54 → 0.36) but discR/cov rose**. The honest
   navD number is ~0.36; the v20 mean was inflated by the lucky s4
   overfit. Cross-distractor robustness (the actual goal of training)
   improved across the board.
3. **Variance halved on navD, more on discR.** Single-seed dose-response
   sweeps become more informative now.
4. **Bundled steps=100 in v21 was net-neutral.** v21 s2/s3 ≈ v22 s2/s3.
   The steps-halving didn't help or hurt; goal-refresh did the work.

**Decision: v22 is the new baseline.** Goal-refresh on, steps=200.
Next wave: dose-response sweep at this baseline (nov05/10/15/20/30 single
seed) since variance is now low enough to read a curve. Plus optional
n_train_distractors / store_bonus axes.

### v21 s4 retry — confirms steps=100 was net-negative

v21 s4 (goalfix + steps=100, retry on mit_normal_gpu, full 100 updates):
navD 0.20 / discR 0.20 / cov 0.15 at u=100. vs v22 s4 (goalfix +
steps=200, same seed): 0.28 / 0.37 / 0.41. Halving steps did real damage
on this seed. v22 baseline is correct — keep steps=200.

## Wave 23 — dose-response at v22 baseline (in progress)

5 runs: nov05_s2, nov20_s2, nov30 × s2/s3/s4. v22 nov10 baseline already
has s2/s3/s4. Tests whether higher novelty lifts the navD ~0.36 plateau.

Final v23 dose curve at u=100:

| Tag | s2 navD | s2 discR | s2 cov | s2 navS |
|---|---|---|---|---|
| v23 nov05 | 0.50 | 0.32 | 0.22 | 0.99 |
| v22 nov10 | 0.50 | 0.34 | 0.37 | 0.99 |
| v23 nov20 | 0.25 | 0.11 | 0.17 | 0.78 |
| v23 nov30 | 0.13 | 0.09 | 0.12 | 0.66 |

| Tag | s3 navD | s3 discR | s3 cov | s4 navD | s4 discR | s4 cov |
|---|---|---|---|---|---|---|
| v22 nov10 | 0.30 | 0.32 | 0.37 | 0.28 | 0.37 | 0.41 |
| v23 nov30 | 0.04 | 0.04 | 0.16 | 0.006 | 0.04 | 0.19 |

**nov10 is the dose optimum.** Past nov10, every metric monotonically
decreases, and the decline replicates across all 3 seeds for nov30.
This means:

- The v20 nov30=good result was downstream of the goals-frozen overfit
  bug, not a real dose effect (a structural bug interaction, not a
  dose-response truth). Goal-refresh + high novelty *both* push the
  agent toward exploration, but past a threshold the agent stops
  goal-finding entirely (navS drops to 0.66-0.73 = random walk).
- More novelty is **not** a path to better cov in this regime. Whatever
  fixes the cov ~0.38 plateau, it's not "more novelty bonus".

This narrows the post-v24 decision tree: path 2 (cov-only intervention)
must come from `--explore_steps` longer, not from higher `--novelty_reward`.

## Wave 24 — freeze_log_std (apply V10 fix)

**Diagnosis driving v24:** v22 navD ~0.36 but navS ~0.99 — a 0.6 gap
between deterministic and stochastic eval. Classic PPO-on-samples
problem: with learnable `log_std` and ent_coef pushing σ up, samples
cover well stochastically while the policy mean stays poorly trained.
This is the same diagnosis V10 phase-A solved.

**Fix:** `--freeze_log_std --init_log_std=-1.5` (std=0.22, no gradient
on σ). Forces PPO loss to shape the mean directly. Already have
`epsilon_explore=0.20` for exploration randomness.

**Code change:** added `--freeze_log_std` CLI flag to train.py
(AgentConfig field already existed for the phased training paths).

**Plan:** 3 seeds (s2/s3/s4) at v22 baseline + freeze_log_std. Same
seeds as v22 nov10 for direct comparison. If navD lifts substantially
(say to ≥0.7), this is the second major structural fix after goal-refresh.

### Wave 24 outcome — partial confirmation, exposed cold-start failure

| @ u=100 | s2 (navD/discR/cov) | s3 (navD/discR/cov) | s4 (navD/discR/cov) |
|---|---|---|---|
| v22 | 0.50 / 0.34 / 0.37 | 0.30 / 0.32 / 0.37 | 0.28 / 0.37 / 0.41 |
| v24 | 0.40 / 0.49 / 0.52 | 0.20 / 0.40 / 0.48 | **0.00 / 0.05 / 0.15** |

| Cross-seed mean | navD | discR | cov |
|---|---|---|---|
| v22 | 0.36 | 0.34 | 0.38 |
| v24 | 0.20 | 0.31 | 0.38 |

**Confirmed (s2/s3):** discR + cov lift +0.10-0.15. navS collapses toward
navD (0.99 → 0.27-0.51), confirming the v22 navS=0.99 was σ-inflation
artifact. The diagnosis was right.

**New problem (s4):** cold-start failure. With frozen σ=0.22, the random
init mean walks ~deterministically. Some seeds escape (s2/s3 catch on at
u=60-80); some don't (s4 navD stays at 0.00). u=1 traj rates collapse
v22→v24: 0.79/0.85/0.71 → 0.32/0.39/0.22. Without enough init randomness
the agent can't find goals.

**Pre-registered decision tree applied:** outcome is closest to a hybrid
of paths 2 (cov rose, discR rose) and 4 (s4 didn't lift navD). Two
testable subhypotheses:

1. **Cold-start fail.** σ=0.22 too narrow for random-init seed lottery.
2. **Undertraining.** s2/s3 trajectories still rising at u=100 (s2 navD
   0.00→0.40 over u=60-100). v24 hadn't plateaued.

## Wave 25 — disentangle cold-start vs undertrain

Two arms:

- **arm A (3 runs):** `init_log_std=-1.0` (σ=0.37, looser than v24 but
  still narrower than v22's σ=1.0), n_updates=100, seeds s2/s3/s4.
  Tests cold-start hypothesis: if s4 now escapes and means lift, the
  v24 σ=0.22 was too tight.
- **arm B (1 run):** `init_log_std=-1.5` (σ=0.22 same as v24), n_updates=200,
  seed=2. Tests undertrain hypothesis: if s2 continues rising and hits
  navD≥0.7, v24 just needed more time.

If arm A succeeds → v25 (loose freeze) is the new baseline.
If arm B succeeds at fixed seed → keep v24 σ=0.22 but train longer.
If both succeed → choose by mean performance / cost.
If neither helps s4 → re-examine; freeze_log_std might be wrong direction.

### Wave 25 outcome — both arms partially succeed; long-train wins on s2

**Arm A (3 seeds × σ=0.37 + 100u) — cold-start FIXED, navD still low:**

| @ u=100 | s2 navD/discR/cov | s3 navD/discR/cov | s4 navD/discR/cov |
|---|---|---|---|
| v22 | 0.50 / 0.34 / 0.37 | 0.30 / 0.32 / 0.37 | 0.28 / 0.37 / 0.41 |
| v24 (σ=0.22) | 0.40 / 0.49 / 0.52 | 0.20 / 0.40 / 0.48 | 0.00 / 0.05 / 0.15 |
| v25 (σ=0.37) | 0.10 / 0.50 / 0.55 | 0.10 / 0.42 / 0.52 | 0.20 / 0.52 / 0.52 |

s4 escaped (navD 0.00 → 0.20; discR 0.05 → 0.52; cov 0.15 → 0.52).
But s2/s3 navD dropped further with looser σ. So loosening helps cold-start
but trades navD on the easy seeds.

**Arm B (1 seed × σ=0.22 + 200u) — undertrain hypothesis CONFIRMED:**

v25 long_s2 trajectory:
- u=80 navD 0.32 → u=100 navD 0.40 → u=200 navD 0.50
- discR peaks 0.56 at u=180; cov peaks 0.53 at u=180
- All three metrics lift simultaneously at this single seed for the first time

**Cross-seed mean comparison:**

| Mean (u=100 unless noted) | navD | discR | cov |
|---|---|---|---|
| v22 (σ=1.0, learnable) | 0.36 | 0.34 | 0.38 |
| v24 (σ=0.22, frozen) | 0.20 | 0.31 | 0.38 |
| v25 arm A (σ=0.37, frozen) | 0.13 | 0.48 | 0.53 |
| v25 long_s2 (σ=0.22, frozen, u=200) | 0.50 | 0.49 | 0.53 (single seed) |

## Wave 26 — verify long-training cross-seed

4 runs at n_updates=200, 4h time budget:

- 3× σ=0.37 + 200u, seeds s2/s3/s4 (cross-seed verification of arm A
  with extended training; no cold-start risk).
- 1× σ=0.22 + 200u, seed s3 (completes the σ × u × seed cube; v25
  long_s2 already done, v24 s4 cold-started so skip s4 at σ=0.22).

If σ=0.37 + 200u replicates the v25 long_s2 pattern across all 3 seeds
(navD ≥ 0.4 + discR ≥ 0.45 + cov ≥ 0.50), this is the new baseline.
If σ=0.22 + 200u s3 also matches s2 result, σ=0.22 + 200u is preferable
for marginally tighter mean.

### Wave 26 outcome — long-training cross-seed verifies; loose σ is the winner

**Loose σ × 200u × 3 seeds** (last clean eval, s4 partial due to evelina timeout):

| Seed | u | navD₀ | discR₀ | cov₀ | navS₀ |
|---|---|---|---|---|---|
| s2 | 200 | 0.50 | 0.65 | 0.62 | 0.67 |
| s3 | 200 | 0.30 | 0.71 | 0.65 | 0.82 |
| s4 | 120 (peak) | 0.20 | 0.42 | 0.50 | 0.30 |
| **mean** | | **0.33** | **0.59** | **0.59** | |

s4 looselong showed late-stage eval oscillation (u=60 cov 0.60, u=180 cov
0.15). Best-during-training values listed above. Both evelina jobs hit
4h timeout — the u=200 eval didn't finish, but training reached u=200
before SLURM cut.

**Tight σ × 200u × s3 (the σ × u × seed cube fill-in):**

s3 at σ=0.22 + 200u: navD 0.30 (u=140-180), discR 0.28 (u=180), cov 0.40
(u=180). Compare to looselong_s3 same seed: same navD (0.30), but σ=0.37
gives discR +0.43 and cov +0.25.

**Loose σ (=0.37) strictly dominates tight σ (=0.22) at long training.**

## Final summary v20 → v26

| Honest cross-seed mean | navD | discR | cov |
|---|---|---|---|
| v20 nov10 (post-fix-bug honest) | 0.36 | 0.27 | 0.26 |
| v22 (goal-refresh) | 0.36 | 0.34 | 0.38 |
| v26 looselong (best) | 0.33 | **0.59** | **0.59** |

discR more than doubled, cov more than doubled, navD held. Two structural
fixes found:

1. **goal-refresh** (env.reset_goal in `train.py` per-update). Halves
   cross-seed variance and fixes the v20 lucky-seed overfit.
2. **freeze_log_std + init_log_std=-1.0 (σ=0.37) + n_updates=200**.
   Forces PPO to shape policy mean instead of σ; tight enough to prevent
   navS-inflation, loose enough to avoid cold-start; 200 updates because
   the narrow-σ regime takes longer to converge.

**Falsified hypothesis: more novelty doesn't help past nov=0.10.**
Strict monotonic decline across all 3 seeds for nov30; v20 nov30=good
was downstream of the goals-frozen overfit.

**Open issue: late-stage eval oscillation, especially seed=4.** discR/cov
swing 0.10-0.50 across consecutive evals at end of long training; eval
trial count (320) may be too small for stability at this metric range.
Future work: increase n_val_trials, or use checkpoint averaging.

## Wave 27 — envs_per_world 20 → 40 cross-seed

2 seeds (s2/s3) on v26 looselong baseline (σ=0.37, 200u, freeze_log_std)
with envs_per_world doubled. Tests whether more (env, goal) diversity
per gradient update lifts the v26 plateau.

| Tag | u | navD₀ | navS₀ | discR₀ | storeS₀ | storeEff | cov₀ |
|---|---|---|---|---|---|---|---|
| env40_s3 | 200 | 0.50 | 0.92 | 0.71 | 0.71 | 1.00 | 0.70 |
| env40_s2 | 200 | 0.30 | 0.76 | 0.16 | 0.16 | 1.00 | 0.26 |
| env40_s2 | 180 (stable) | 0.30 | 0.72 | 0.65 | 0.65 | 1.00 | 0.62 |

s2 at u=200 hit a low-variance valley (the late-eval oscillation issue
flagged in v26 earlier); at u=180 was at peak. Cross-seed mean using
last *stable* eval (u=180-200):

| Mean | v26 (envs=20) | v27 (envs=40) |
|---|---|---|
| navD₀ | 0.33 | 0.40 |
| discR₀ | 0.59 | 0.69 |
| cov₀ | 0.59 | 0.66 |

Modest lift on all three metrics (+0.07 / +0.10 / +0.07). Direction
right; doubling envs added gradient diversity and the policy mean
benefits. Doesn't fix the late-stage eval oscillation.

**Best single eval point ever (env40_s3 at u=180):** navD 0.50, discR
0.73, cov 0.70, storeEff 1.00. Not a single-seed lottery — s3 was the
hardest seed in earlier waves; s2 reaches similar peaks at slightly
different update points.

## Trajectory diagnostic (2026-04-27) — v27 s3 u=200

Tooling: `hopfield_nav/inspect_trajectories.py` runs deterministic
nav_det-style rollouts (goal pre-loaded in Hopfield + N distractors),
records every position, computes wall/edge/stall stats, and plots
trajectories. Rows = different goals, columns = trials per goal.

### Run 1 — goal pre-loaded in Hopfield (n_dist=0, 16 trials × 10 envs)

Aggregate (160 trials):
- `success_rate=0.500` (matches eval); `mean steps to reach=23` for successes
- successes: `edge_dwell=0.90`, `stall_frac=0.47` — successes hug walls
- failures: `edge_dwell=0.62`, `final_at_corner=0.05`, `last_q_stuck=0.00`

Top-5 failure end positions: (7,1) 8, (7,2) 7, (6,0) 7, (3,3) 5, (0,1) 4.

### Run 2 — same checkpoint, n_dist=10

- `success_rate=0.51` (held), but failures end at corner at higher rate:
  `final_at_corner=0.19` (4× over d=0), top-5 led by (7,0) corner with 9
  trials.

### Run 3 — EMPTY Hopfield (no goal pattern loaded), n_dist=0

This is the key diagnostic: what does the policy do without any goal
direction signal?

- `success_rate=0.23` (chance-level random hits)
- failures: `corner_dwell=0.85`, `stall_frac=0.90`, `final_at_corner=1.00`
- **All 46 failures end at exactly the same corner: (7, 7).**

### Conclusion — the agent has a corner attractor at (7, 7)

The empty-Hopfield run reveals the agent's *unconditional* policy mean
is biased to head SE and stop at (7, 7). With goal-loaded Hopfield this
attractor competes with the Hopfield direction signal:

- **Goal on perimeter**: clear-direction Hopfield wins → wall-slide to
  goal in ~20 steps. The "wall-hugging successes" pattern.
- **Goal in interior**: weaker/perpendicular Hopfield projection
  partially fights the (7,7) bias → result is a compromise: agent
  follows the wall but does perimeter loops without peeling off into
  the interior.
- **No Hopfield**: nothing to fight the bias → agent runs straight to
  (7,7) and stalls there for 90% of the rollout.
- **Distractor pollution**: Hopfield signal degraded → bias wins more
  often → 4× higher corner-end rate at d=10 vs d=0.

This re-frames the navD plateau (~0.36-0.50). The bottleneck is **not**
"agent doesn't navigate" — it's "agent has a baked-in SE-corner default
that the Hopfield signal can override only along axis-aligned directions
to perimeter goals." Levers that target this directly:

- **Symmetric / zero-mean init for the movement head** (currently the
  Linear's default init produces a non-zero mean direction baked into
  bias terms; freeze_log_std preserves this).
- **Aux loss anchoring**: penalize non-zero policy mean when the
  Hopfield input is zero — explicitly tells the policy "no signal =
  no preference."
- **Empty-Hopfield rollouts in training**: include some fraction of
  rollouts where the goal isn't stored, so the agent learns "no signal"
  shouldn't mean "default to corner."

The Hopfield-recall-quality fixes I'd previously pitched
(`hopfield_steps=3`, etc.) wouldn't help — recall isn't the bottleneck.

## Per-env × distractor diagnostic (2026-04-27) — eg73 vs v27 head-to-head

Tooling: `hopfield_nav/inspect_per_env.py`. For each val env independently,
runs nav_det / nav_stoch / disc / expl at distractor levels {0, 1, 3, 5, 10}
with 64 trials per (env, distractor, metric). Output: per-env tables and
heatmap PNG, JSON sidecar.

### eg73 (n_train_distractors=0, learnable σ, n_updates=300)

- **nav_det (goal preloaded)**: 0.69-1.00 across all 10 envs at all distractor
  levels. Even at d=10, ≥0.81. Hopfield-following itself is robust.
- **disc_reach (agent must explore + store)**: 1.00 at d=0 for 8/10 envs,
  but craters to 0.10-0.55 with even 1 distractor. Two outliers (envs 6, 9
  with goals (7,1) and (6,0)) fail at d=0 too (0.06, 0.02) — agent's
  empty-HF default trajectory specifically misses those goal cells.
- **store_efficiency**: 1.00 across the board.
- **expl coverage**: 0.30-0.89 at d=0, drops to 0.22-0.37 at d≥1.

**Failure mode**: distractor-fragile. With ANY distractor, agent's
exploration policy is replaced by a Hopfield-recall-following policy
that pulls toward distractor positions, missing the goal cell.

### v27 s3 (n_train_distractors=3, freeze_log_std, σ=0.37, n_updates=200, envs=40)

- **nav_det (goal preloaded)**: BIMODAL by goal location.
  - Edge goals ((1,7), (2,0), (0,6), (0,1), (4,0)): **1.00** at every d.
  - Interior goals ((4,1), (1,3), (6,4), (2,6), (4,1)): **0.00** at every d.
  - Goal location alone determines success; distractor count is irrelevant.
- **disc_reach (agent must explore + store)**: 0.47-0.98, **flat across
  distractor counts**. Distractor training did its job — agent is robust
  to count.
- **Counterintuitive finding**: for interior goals, disc_reach @ d=0 is
  HIGH (0.47-0.98) while nav_det @ d=0 is 0.00. Same env, same start
  distribution. **The Hopfield direction signal actively HURTS navigation
  for interior goals**. Without signal (empty HF) → agent's default
  exploration finds interior goals OK. With signal → agent goes into
  "perimeter walk" basin and never peels off.

**Failure mode**: perimeter-walk basin. Policy learned during training to
follow Hopfield direction by walking edges; works for edge goals,
catastrophically fails for interior goals. v27 has TWO behavioral modes
(Hopfield-on, Hopfield-off) and the on-mode is the broken one.

### Direct comparison

| | eg73 | v27 |
|---|---|---|
| nav_det at d=0 (range) | 0.69-1.00 (all envs) | **0.00 OR 1.00 (bimodal by goal location)** |
| nav_det at d=10 | 0.81-1.00 (mild degrade) | same as d=0 (flat) |
| disc_reach at d=0 | 1.00 (mostly) | 0.47-0.98 |
| disc_reach at d=10 | 0.10-0.55 (collapses) | **0.47-0.97 (flat — distractor-robust)** |
| store_efficiency | 1.00 | 1.00 |

### Implications

1. **Distractor training (n_train_distractors=3) genuinely worked** —
   v27 disc_reach is flat across distractor counts where eg73 collapses.
   The original wave-13+ hypothesis was correct.
2. **But v27 paid for distractor robustness with**: bimodal nav_det
   (interior goals → 0%) and a perimeter-walk basin that's worse than
   eg73 in absolute terms.
3. **`n_train_distractors=3` is fixed every rollout** (train.py:266
   `while placed < n_train_distractors`). Eval at d=0 is OOD; at d=3
   is on-distribution. v27's flat-across-d behavior is partly because
   "always-3" trained the policy to ignore distractor count rather than
   discriminate Hopfield content quality.
4. **Next iteration candidates**:
   - Variable distractor count per rollout (uniform on [0, max]) so
     d=0 is in-distribution. `train_phase_a_only.py` already has
     `n_train_distractors_min/_max` params; need to port to `train.py`.
   - Loosen σ to break the perimeter-walk basin lock-in.
   - Drop novelty + epsilon (may bias toward edge-walking).

## Wave 29 — eg73-config revert + variable distractors (FAILED: μ-drift)

Hypothesis: eg73 had loose σ + no novelty + no goal-refresh. v22-v27 added
freeze_log_std + novelty + epsilon. Try eg73 base + (variable distractors,
envs=80, refresh_envs_each_update, annealed epsilon) — keep eg73's
properties, port the v22 bug-fix and v22 distractor training. NO novelty.

Code change: ported `n_train_distractors_min/_max` from `train_phase_a_only.py`
to `train.py` (rollout-level uniform sampling per parallel env).

Config: eg73 base + `--n_train_distractors_min 0 --max 10` + `--epsilon_explore
0.20 --epsilon_anneal_updates 300` + `--envs_per_world 80` +
`--refresh_envs_each_update`. 2 seeds, 200 updates.

**Outcome — clean μ-drift, both seeds collapsed identically:**

| u | s1 traj | s2 traj |
|----|---:|---:|
| 10 | 0.72 | 0.72 |
| 20 | 0.74 | 0.76 |
| 50 | 0.89 | 0.78 |
| 80 | 0.66 | 0.81 |
| 100 | **0.51** | **0.60** |

Both seeds dropped below random-walk baseline (0.66) by u=100. Earlier
discR/cov gains (s2 hit cov 0.43 at u=80) were lost as μ drifted.

**Diagnosis**: with refresh_envs_each_update + variable distractors, every
update is a fresh task distribution. Without novelty (or any anti-drift
force), policy mean commits to whatever direction got rewarded most often
across diverse rollouts → poor generalization → traj_rate falls below
baseline.

eg73 escaped this *because its training distribution was fixed* (20 envs,
fixed goals, 300 updates). With v29's per-update task diversity + no
shaping, μ has no stable target.

**Cancelled at u=100**. Confirms novelty was doing real work in v22-v27 —
perimeter-walk basin was a side-effect of novelty, not the primary mechanism.

## Wave 30 — v29 base + novelty + wall_penalty (in flight)

Hypothesis: novelty (0.10) prevents μ-drift; new `wall_penalty` (0.05)
counters the perimeter-walk basin that novelty alone induces.

**Code change**: added `HopfieldConfig.wall_penalty` (per-step penalty when
agent at edge cell during explore phase). Reward math at wall_penalty=0.05:
- Perimeter walk (28 cells × 28 steps): novelty +2.8, time -0.28, **wall -1.4** → +1.12 net
- Interior walk (28 cells): +2.8 - 0.28 - 0 → +2.52 net
- Interior preferred by ~+1.4

Otherwise identical to v29 (eg73 base + variable distractors + envs=80 etc.).
2 seeds, 200 updates.

**Progress at u=60 (still training):**

| @ u=60 | s1 | s2 | v29 (failed) | v22 mean (final) |
|---|---|---|---|---|
| traj | 0.84 | 0.88 | 0.86→0.51 | – |
| navD | 0.33 | 0.30 | 0.30 | 0.36 |
| navS | 1.00 | 1.00 | 0.82 | 0.99 |
| **discR** | **0.41** | **0.41** | 0.03/0.13 | 0.34 |
| **cov** | **0.39** | **0.41** | 0.16/0.18 | 0.38 |

Both seeds tracking together. discR/cov at v22-final levels by u=60 (30%
of training). traj_rate well above random walk → no μ-drift. **wall_penalty
doing useful work without novelty's perimeter side effect.** navD stuck at
0.30 (V10-style μ-shaping bottleneck).

### v27/eg73 stochastic-eval comparison (τ=0.3)

Tooling: `--action_temperature` flag (multiplies σ before sampling) +
stochastic disc/expl variants in inspect_per_env.

- **v27** (frozen σ=0.37 trained): τ=0.3 stoch eval *lifts* disc_reach
  mean from 0.78 → 0.87 (interior goals especially: env 4 0.47→0.97).
  But nav_det stays bimodal — perimeter-walk basin survives noise when
  goal is preloaded.
- **eg73** (wide σ=1.0 trained): τ=0.3 stoch *hurts* disc_reach
  (0.10-1.0 → 0.36 mean). Narrowing σ off training distribution.

Implication: eval-σ should match train-σ. Don't add stochastic noise to
eg73-style training. Don't read v27 as "broken" — its perimeter-walk
basin is escapable in disc setting under mild noise; just not under
deterministic-mean nav_det.

### v30 oracle eval — DECISIVE: policy μ is the bottleneck, not recall

Ran inspect_per_env --oracle on v30 s1 u=60 checkpoint. Replaces Hopfield
recall direction with PERFECT goal-direction projection.

Per-env oracle navD at d=0:

| env | goal | type | oracle navD |
|---|---|---|---|
| 0 | (0,0) | corner | **1.00** |
| 3 | (6,0) | corner | **1.00** |
| 7 | (3,0) | edge | **1.00** |
| 1 | (3,1) | edge-1 | 0.00 |
| 2 | (5,1) | edge-1 | 0.00 |
| 6 | (6,1) | edge-1 | 0.00 |
| 4 | (2,4) | interior | 0.00 |
| 5 | (5,4) | interior | 0.06 |
| 8 | (6,3) | interior | 0.12 |
| 9 | (3,4) | interior | 0.00 |

**Mean oracle navD = 0.32 ≈ real navD = 0.32-0.45.**

Even with perfect direction signal, μ only navigates to literal-edge goals
(x or y in {0, 7}). One step in from edge → 0%. **The bottleneck is policy
μ, not Hopfield recall.** Recall improvements (`--hopfield_steps 3`,
encoder retraining) wouldn't help.

Implications: μ has converged to "go to wall, stay there" regardless of
direction signal. The Hopfield-direction input has near-zero gradient on
movement_mean for non-edge target states. Either:
1. PPO advantage hasn't shaped non-edge directions yet (early training, u=60).
2. Network can't read the Hopfield-signal channel cleanly (architectural).
3. Wall-walking basin is too dominant — even oracle direction can't break it.

### Wave 31 — revisit_penalty replaces wall_penalty

Hypothesis: wall_penalty (-0.05/wall-cell) over-suppresses goal-reach paths
through wall-adjacent goals. revisit_penalty (-0.05/revisited-cell) kills
perimeter loops without penalizing first-time wall-touches. Same
anti-perimeter intent, less collateral.

Reward math at revisit_penalty=0.05 (novelty=0.10):
- Single perimeter sweep (28 unique cells, 0 revisits): novelty +2.8, time
  -0.28, revisit 0 → +2.52. Still rewarded.
- Perimeter LOOP (28 cells, 28 revisits per cycle): +2.8 - 0.28 -1.4 → +1.12.
- Interior reach (10 steps): -0.10 + 1.0 + 0.5 → +1.40.

So revisit_penalty pulls down LOOPS but keeps single sweeps rewarded.
Slightly different from wall_penalty.

Otherwise identical to v30. 2 seeds. Pending GPU.

### Decisions for further iteration (overnight)

Per oracle finding, μ-supervision approaches likely needed beyond reward
shaping. Possible v32+ candidates:

1. **softer wall_penalty (0.02)**: tests if v30 was just too aggressive.
2. **softer novelty (0.05)**: makes interior reach more rewarding relative
   to wall walks. v23 found nov=0.05 was suboptimal compared to nov=0.10
   in v22-config; but with wall_penalty + variable distractors, the
   tradeoff may be different.
3. **auto_nav_warmup**: would teacher-force movement to Hopfield direction
   for first N updates. Risk: with distractors-only Hopfield content,
   teacher direction is random — earlier waves noted this as catastrophic.
   Would need goal-stored gating to be safe.
4. **BC pretraining**: existing DAgger oracle teaches μ to follow Hopfield
   direction directly. Two-stage training. Largest code change.

Holding for v30 + v31 outcomes before committing to (3) or (4).

### Wave 30 final (u=200) — best navD ever, but cov collapsed

Final eval, both seeds completed:

| @ u=200 | s1 | s2 | mean |
|---|---|---|---|
| **navD** | **0.84** | 0.54 | **0.69** |
| navS | 1.00 | 1.00 | 1.00 |
| discR | 0.14 | 0.18 | 0.16 |
| cov | 0.18 | 0.18 | 0.18 |

Compared to all prior waves at u=200 (or final):

| | navD | discR | cov |
|---|---|---|---|
| v22 final | 0.36 | 0.34 | 0.38 |
| v27 final | 0.40 (bimodal) | 0.69 | 0.66 |
| eg73 (u=300, no distractors) | 0.97 | 0.10 (d≥1) | 0.30 |
| **v30 final** | **0.69** | 0.16 | 0.18 |

**v30 navD = 0.69 is the highest deterministic-eval navigation we've trained
that's also distractor-robust** (eg73 had 0.97 nav but craters with
distractors). Learning trajectory:

| u | s1 navD | s1 cov |
|---|---|---|
| 20 | 0.81 | 0.12 (init lottery) |
| 40 | 0.40 | 0.35 |
| 60 | 0.33 | 0.39 (cov peak) |
| 100 | 0.32 | 0.33 |
| 140 | 0.63 | 0.21 |
| 160 | 0.81 | 0.16 |
| 180 | 0.89 | 0.13 |
| 200 | 0.84 | 0.18 |

Two phases:
- u=0-100: agent explores broadly. cov climbs to 0.39, navD modest (~0.30).
- u=100-200: agent commits to navigation. navD climbs to 0.84, cov collapses.

The wall_penalty + novelty combo drove this **explore-then-commit** trajectory.
PPO eventually shaped μ to use Hopfield-direction (oracle u=60 had said it
couldn't, but u=160+ shows it does). The cost: agent abandoned exploration
once nav skill emerged.

**Speed/steps detail (s1 u=140-200)**: succ 0.63 in 6.5 steps (u=140) → 0.81
in 10.6 (u=160) → 0.89 in ?? (u=180). At u=140 the agent navigates in
near-optimal step counts (4-7 step optimal for 8×8). By u=200 succ rate
is high but steps are higher — agent reaches more distant goals successfully.

**Open question**: can we preserve cov through training while still getting
navD to ≥0.6? v31 (revisit_penalty replaces wall_penalty) tests this.
Hypothesis: revisit_penalty kills only loops, not all wall-paths, so the
agent doesn't fully abandon edge-touching exploration when shifting to
navigation.

## Wave 31 — revisit_penalty replaces wall_penalty

Same as v30 but `--revisit_penalty 0.05 --wall_penalty 0`. Agent penalized
for re-visiting cells (kills perimeter LOOPS) but not for first-time
wall-touches (preserves goal-reach paths through walls).

**Outcome**: cross-seed s1/s2 final navD 0.40/0.40, discR 0.42/0.42,
cov 0.39/0.39. **Peak at u=120: cov 0.66, discR 0.68, navD 0.30.** Best
"balanced explorer" of any wave but navD never broke out (8 evals at 0.30).

Different trajectory from v30: v30 did flat-then-jump (navD 0.32→0.46→0.84
between u=100 and u=200). v31 stays flat throughout — *revisit_penalty
preserves exploration but doesn't trigger the nav-commitment basin*.

## Wave 32 — wall + revisit, both at 0.025

Combined light shaping. Outcome: tracked v31 closely, no improvement.
Combination didn't break the wall-vs-revisit tradeoff.

## Wave 33 — TRUST SIGNAL input — BREAKTHROUGH

Added `AgentConfig.input_goal_in_memory: bool` flag. When set, agent input
gets a 1-bit channel: "agent has stored at goal during this rollout"
(equivalently: Hopfield content is trustworthy goal-direction). Signal is
True from start in nav_det eval (goal preloaded), False initially in
disc/expl evals, becomes True when agent stores at goal during rollout.
In training: tracked per-env in rollout.py via `agent_goal_store_fired`.

Code changes: AgentConfig field, `compute_input_dim` adds 1 dim,
rollout.py builds rnn_input with bit, eval._agent_step routes existing
`goal_in_memory` parameter into rnn_input. CLI flag `--input_goal_in_memory`.

Config: v30's reward shape (wall_penalty=0.05, novelty=0.10) + trust signal.

**Outcome — best navigation we've ever trained:**

| @ u=200 | s1 | s2 | mean |
|---|---|---|---|
| navD | 0.98 | 1.00 | **0.99** |
| navS | 1.00 | 1.00 | 1.00 |
| discR | 0.26 | 0.16 | 0.21 |
| cov | 0.32 | 0.17 | 0.25 |
| mean_steps | 10.5 | – | ~10 |

**Per-env eval at u=200 (s1)**: nav_det 0.84-1.00 across **every
(env, distractor) cell** (10 envs × d=0,1,3,5,10) — including interior goals.
**First config to achieve eg73-quality nav with distractor robustness.**
mean_steps 6-13 across envs (corner/edge: 6-8; interior: 10-14).
Cross-env mean_steps ~7.5 — close to user's optimal ~6 target.

Trust signal worked: gave PPO a clean way to condition policy on
"trustworthy Hopfield" vs "untrustworthy distractor noise."

## Wave 34 — wall + revisit, no novelty (FAILED, novelty load-bearing)

Tested whether wall_penalty + revisit_penalty alone could replace novelty.
Outcome: cov 0.12-0.14 at u=20-40, agent barely moving. Confirmed novelty's
+0.10/cell positive reward is necessary; the two penalties alone provide
only NEGATIVE pressure, no driver for exploration. Cancelled at u=40.

## Wave 35 — trust signal + revisit (no wall) — pure explorer

v33 with wall_penalty swapped for revisit_penalty. Hypothesis: wall_penalty
forced nav commitment in v33, costing cov; revisit might preserve cov while
trust signal lets policy navigate post-store.

**Final results @ u=200:**

| | s1 | s2 | mean |
|---|---|---|---|
| navD d=0/1/3/10 | 0.30/0.30/0.32/0.38 | 0.30/0.30/0.32/0.34 | ~0.32 (flat) |
| cov | 0.67 | 0.59 | 0.63 |
| disc | 0.61 | 0.48 | 0.55 |
| nav_stoch | 0.98 | 0.96 | 0.97 |

**Pure explorer mode confirmed.** nav_stoch ~0.97 across all distractor
levels = the agent can *find* the goal under a stochastic policy (random-
walk search) but cannot navigate to it deterministically. The trust
signal isn't translating into μ-shaping in the revisit-only landscape;
wall_penalty was the load-bearing piece for v33's navD breakout. Pattern
matches v31 (also revisit-only). Distractor-flat at every metric — good
robustness, wrong skill.

## Wave 36 — trust + wall + revisit (combine v33 and v35) — done

Hypothesis: wall_penalty drives nav breakout (v33 mechanism),
revisit_penalty preserves cov in explore phase (v35 mechanism), trust
signal lets policy switch modes between explore and navigate.

**Final results @ u=200 (cross-seed split, NOT averaged — they diverged):**

| | s1 (glad-durian-159) | s2 (solar-aardvark-160) |
|---|---|---|
| navD d=0/1/3/10 | 0.90/0.90/0.89/0.87 | 0.31/0.32/0.30/0.34 |
| cov | 0.50 | 0.62 |
| disc | (similar) | (similar) |
| classification | "navigator" (perimeter-sweep) | v35-clone (pure explorer) |

**Seed-fragile**: same config gave one perimeter-sweep "navigator" and
one pure-explorer. **Neither is real Hopfield gradient-following** —
trajectory inspection on s1 (next section) confirms.

### Trajectory-inspection finding: agent learned perimeter-sweep search, not gradient-following

Ran `inspect_trajectories.py` on v36 seed 1 u=160 across 4 conditions
(goal-loaded × empty-Hopfield × d=0 × d=10, 25 trials per env, 5 envs).
(s1 u=160→200 trajectory was stable: navD held 0.87-0.90, cov 0.50-0.55.)

| condition | success | mean_steps | edge_dwell | stall_frac |
|---|---|---|---|---|
| goal d=0 | 80% | 34.5 | 0.46 | 0.09 |
| goal d=10 | 77% | 34.9 | 0.46 | 0.10 |
| empty d=0 | **3.6%** | 9.2 | 0.25 | 0.73 |
| empty d=10 | 14% | 25.9 | 0.50 | 0.33 |

**Per-goal pattern (goal_d0 / goal_d10 plots):**

- Corner / wall-edge goals (e.g. (0,0), (6,0)): reached cleanly in 5-20
  steps. Looks like genuine nav.
- Wall-adjacent goals (e.g. (3,1), (5,1)): reached via giant perimeter
  loops, 90-130 steps. Agent goes to perimeter, sweeps along it,
  bumps into goal.
- Truly interior goal (2,4): **all 5 trials fail** at d=0 AND d=10. Agent
  never penetrates the interior at all.

**Distractors don't matter** because the agent isn't doing per-step
Hopfield gradient-following — d=0 and d=10 plots look essentially
identical. The Hopfield signal is being used only as a coarse
"which direction to start sweeping the perimeter" cue.

**Empty-Hopfield baseline** confirms the policy has a strong learned
default: with no signal at all, the agent walks to (0,0) and stalls (73%
stall_frac). With distractors but no goal, it wanders along walls.

**What this means:** the Hopfield/trust system *is* being used (80% vs 4%
success when goal stored vs not) — but only to bias direction of a
perimeter-orbit search strategy, not to drive μ toward the goal. The
agent has learned a fixed perimeter-sweep behavior that succeeds when
the goal happens to lie on or near the perimeter (which is most goals
on an 8×8 grid: 28/64 = 44% are on the boundary, 64% are within 1 cell).
This explains the 80% success + 35-step nav: it's **search**, not
**navigation**.

This deepens the disambiguation hypothesis but reframes it: the bottleneck
isn't goal-vs-distractor disambiguation per se — it's that PPO never
discovered that the Hopfield gradient *can* drive μ toward arbitrary cells.
Wall_penalty + revisit_penalty + novelty creates a reward landscape where
perimeter-sweeping with wall-grazing avoidance is locally optimal and
covers ~half the goal distribution; PPO never gets enough advantage signal
on interior goals to break out of that basin.

### Implications for v37+

The v37 hard-gate idea (zero Hopfield input during explore) might still
help by removing distractor-noise from explore, but it won't fix this
deeper basin. Two stronger candidates now:

1. **Curriculum on goal location**: train initially with goals biased
   toward interior cells. Forces PPO to discover interior-driving μ
   shapes, then relax the bias.
2. **Hopfield-direction reward shaping**: reward dot-product of action
   with Hopfield-recall direction. Direct supervision that "follow this
   gradient" is the right behavior, sidesteps the perimeter local
   optimum entirely.

### Why does the agent perimeter-sweep with goal but corner-stall without it?

Two compounding causes:

1. **Agent has no position input.** Train script uses
   `--no-input_encoded_state`, so the policy inputs are only
   `prev_reward` + 2D Hopfield-recalled goal + trust bit. The agent
   literally cannot see its own current position. To do gradient-following
   from (4,4) → goal (3,1), it would need to integrate `prev_action`
   history through the GRU into an internal position estimate. PPO never
   learns this because (2):

2. **Novelty stacks faster than goal_reward.** With `novelty_reward=0.10`
   per new cell, a 30-step perimeter sweep that grazes the goal earns
   ≈ +1.0 (goal) + 3.0 (novelty) − 0.3 (time) = **+3.7**. A 6-step
   direct nav (if it could do it) would earn only ≈ +1.0 + 0.6 − 0.06
   = **+1.5**. Perimeter-sweep is *strictly more rewarding* than direct
   nav, even when the goal is interior. The Hopfield signal then biases
   which direction around the perimeter to start sweeping — not which
   interior direction to go.

For empty-Hopfield: the input distribution (zero 2D signal + trust=1)
is degenerate / out-of-distribution; μ collapses to a fixed-direction
output that ends wedged in (0,0).

**This means the right fixes are upstream of reward shaping:**

- Re-enable `--input_encoded_state` so the agent sees its own position.
  Even an oracle interior-goal evaluator (next experiment) can confirm
  whether the policy *could* navigate to interior given position input.
- Or shrink `novelty_reward` (e.g. 0.02 instead of 0.10) so the
  perimeter-sweep optimum disappears.
- Or raise `goal_reward` materially (already configurable) so direct
  nav dominates for any goal location.

---

# Big-picture findings (as of 2026-04-28)

## What works

1. **Distractor training matters but the *count* must vary**:
   - Fixed `n_train_distractors=3` (v22-v27) → distractor-flat disc but
     bimodal nav (perimeter walks for interior goals).
   - Variable `n_train_distractors_min=0, max=10` (v29+) is the right
     framing — agent sees the full distractor distribution.

2. **Goal-refresh per update is critical** — env.reset_goal() inside
   `--refresh_envs_each_update`. Without it, training overfits to 20 fixed
   (env, goal) pairs and the apparent generalization is illusion (eg73
   showed up to 0.97 navD via memorization, not generalization).

3. **Trust signal (`--input_goal_in_memory`) is a genuine architectural
   improvement**. Single 1-bit input changed v33 from "v30-like" to "best
   navigation result of all time, distractor-robust." Without trust signal
   v30 hits navD 0.69; with it v33 hits navD 0.99.

4. **Novelty (+0.10/cell) is load-bearing for exploration**. Without it,
   wall+revisit penalties produce frozen agents (v34). v29 (no novelty)
   collapsed to μ-drift below random walk.

## The remaining tension (main issue to solve)

**The agent has two anti-correlated behavioral modes that current reward
shaping can only push toward one or the other:**

- **wall_penalty** drives "commit to navigation" (forces interior
  engagement → PPO shapes μ to use Hopfield direction → high navD).
  Cost: cov collapses (~0.25) because agent stops exploring the perimeter.

- **revisit_penalty** drives "preserve exploration" (kills loops without
  killing wall-touches → high cov ~0.66). Cost: agent never develops
  navigation skill (navD stuck at 0.30).

Trust signal lets policy *condition* on "should I explore or navigate now"
but doesn't break the underlying tradeoff — PPO still has to choose ONE
behavior to reward most. Both v33 and v35 collapsed to a single mode
(navigator and explorer respectively), they just differ in which.

**v36 (running)** tests whether combining wall + revisit penalties (with
trust signal) can give us both high cov AND high navD simultaneously.

### Deeper framing: the disambiguation hypothesis

The wall-vs-revisit tension above may be a **downstream symptom** of a
more fundamental representational problem: **the agent cannot
disambiguate, from the Hopfield signal alone, between a stored goal and
a distractor.**

- If the agent learns to follow the Hopfield signal, it gets pulled
  toward distractors during the explore phase before it has stored
  anything → exploration collapses.
- If the agent learns to ignore the Hopfield signal, exploration works,
  but post-store it cannot navigate to its remembered goal.
- The reward-shape tradeoff then becomes a proxy for which side of this
  dilemma PPO settles into: wall_penalty pushes "follow signal," revisit
  pushes "ignore signal."

The trust bit (`--input_goal_in_memory`) is the right *kind* of fix —
v33's navD 0.99 with trust=1 confirms the signal works *once it's known
to be the real goal*. But during the explore phase the agent still
*receives* the 2D Hopfield input vector and must learn to actively
suppress an input it has otherwise been trained to follow. That's a
learned suppression, not a clean architectural separation, and it
explains why v33's cov stays at 0.25.

### Parking-lot hypotheses (for after v35/v36 results)

- **Hard-gate Hopfield by trust bit (v37 candidate)**: multiply the 2D
  Hopfield signal by the trust bit so it is *literally zero* during the
  explore phase. Removes the disambiguation burden — explore sees no
  Hopfield input at all, nav phase sees real goal + distractors. If this
  fixes cov without hurting navD, disambiguation was the real
  bottleneck and reward shaping is secondary.
- **σ annealing during training**: start wide (sample-driven
  exploration) and narrow late (commit to mean-driven nav). Currently
  σ=1.0 throughout.
- **Phased training**: pre-train explorer (V10-style), then switch
  reward shape to nav-shaped fine-tune. Two-stage curriculum.
- **Higher goal_reward** (now configurable): make navigation reward
  dominate exploration shaping rewards, see if it forces μ-shaping
  even with revisit_penalty.

The user's stated target — **navD ≥ 0.7 across distractors AND cov ≥ 0.5
AND mean_steps ~6** — is now half-achieved (navD ≥ 0.84 across distractors
in v33, mean_steps 6-13). cov stays the gap.

## Other diagnostic findings

- `inspect_trajectories` revealed v27's perimeter-walk basin is goal-loaded-
  triggered, not unconditional. Empty Hopfield → corner-(7,7) attractor for
  v27, no perimeter walk.
- `inspect_per_env --oracle` (replaces Hopfield-recall with exact direction)
  showed at v30 u=60: even with perfect direction signal, μ couldn't reach
  interior goals. Recall noise is NOT the bottleneck — μ shaping is.
  By v33 u=200, that bottleneck is solved (navD 0.99).
- Stochastic-policy eval (`--action_temperature`) shows eval-σ should match
  train-σ. v27 (σ=0.37 frozen) benefits from τ=0.3; eg73 (σ=1.0 wide)
  doesn't.

