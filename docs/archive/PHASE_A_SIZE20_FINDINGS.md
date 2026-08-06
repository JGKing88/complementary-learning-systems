> **Archived.** Moved out of `hopfield_nav/` by phase 6 of the 2026-08
> refactor. Not maintained; describes what was believed and tried at the time,
> which in places is no longer true of the code. Start from `docs/archive/README.md`
> for what replaced it.

# Phase A 20×20 — findings log

Single living doc per `PHASE_A_SIZE20.md` step 9. Append-only. Newest entries at top.

Prior best (pre-log): `sunny-dew-92` u180 cov 0.533 / sr 0.92, u140 cov 0.515 / sr 0.96 — chained-resume of v15-style recipe.

---

## Checkpoint summary (2026-05-13)

### Campaign goals
Phase A on 20×20 continuous-action grid. Baseline to beat: **v27 u880 = 0.963 sr / 22.9 ms** (fresh-eval, avg over distractor counts 0/5/10).

### Levers tested
| Family | Variants | Idea |
|---|---|---|
| **Action magnitude** | v25 (norm), v28 (max=1.5), v31 (max=1.0), v32 (clamp [0.5,1.25]) | Cap step size to prevent overshoot near goal |
| **Goal radius** | v33 (rad=1.0 alone), v34 (rad+clamp stack) | Wider goal ball relaxes precision requirement |
| **Distractor curriculum** | v26, v30 (ramp 10→20) | Force harder distractor handling |
| **Entropy / log_std** | v29 (anneal looser), **v36** (anneal tighter, active) | Less exploration noise late = more precision |
| **Time penalty** | **v35** (5× default, active) | Stronger gradient toward shorter trajectories |

### Current frontier (fresh-eval)
- **v35 u380 (gentle-terrain-124)**: 0.994 / 22.9 / min **0.991** — new sr champion, ties v27 on ms
- **v35 u440 (gentle-terrain-124)**: 0.988 / 21.3 / min 0.975 — best sr/ms balance
- **v33 u540 (pretty-totem-110)**: 0.982 / **20.3** / min 0.972 — best ms
- **v34 u360 (driven-snowflake-111)**: 0.995 / 26.3 / min 0.988 — high-sr / high-ms point
- v27 u880 (lively-surf-104): 0.963 / 22.9 / min 0.953 — historic baseline

### Open hypotheses being tested now
- **v35** (time_penalty=0.05): **confirmed strongest new family** — u380 fresh = 0.994/22.9 (best sr ever), u440 fresh = 0.988/21.3 (best balance). Strict win over v27 on sr (+3.1pp) at matched ms.
- **v36** (log_std anneal -1.8→-2.2 over u300-u600): late-training noise squeeze for fine approach precision. Never produced data — first attempt died in cluster maintenance before u20. Still queued.

### Key learning from goal_r=1.0 re-eval
Models trained at radius=0.5 generalize *better* at lenient eval (rad=1.0) than models trained at rad=1.0 — they learn tighter approach paths. v33's win came from being *trained* at rad=1.0 + more updates, not just from the radius itself.

### To run when cluster is back (queued ideas, all with goal_radius=1.0)
- **v36** = v33 base + log_std anneal -1.8 → -2.2 over u300-u600. Tests whether late-training noise squeeze (σ 0.165 → 0.111) breaks the ms ceiling. Lever never produced data — first attempt died in cluster maintenance before u20. Recipe already added to sweep script.
- **v37** = v35 (time_penalty=0.05) + v36 (log_std anneal -1.8→-2.2) stack. Stacks the two best levers — v35 family is current sr champion (u380 = 0.994/22.9), v36's anneal lever untested. If they compose, target is sub-20 ms with sr ≥0.99. Recipe already added to sweep script.
- **v38 idea**: v35 base + max_action_norm=1.25 (no min floor). Stronger time_penalty pushes policy to step bigger; cap prevents overshoot. Complements anneal — anneal tightens noise, max cap tightens deterministic mean.
- **v39 idea**: v35 base + larger envs_per_world=160 (vs 80). Better gradient estimates per update, may stabilize the sr regression seen in v35 u400 (0.945 dip).
- **Re-eval pass for v36/v37 once they hit u380-u540**: the v35 fresh-eval peak was at u380-u440. Don't wait past u600 to start fresh-evaluating.

### Latest fresh-eval leaderboard
| ckpt | avg srD | msD | min srD | wins |
|---|---|---|---|---|
| v27 u880 (lively-surf-104) | 0.963 | 22.9 | 0.953 | historic baseline |
| v33 u540 (pretty-totem-110) | 0.982 | **20.3** | 0.972 | best ms |
| v35 u380 (gentle-terrain-124) | **0.994** | 22.9 | **0.991** | best sr / best min sr |
| v35 u440 (gentle-terrain-124) | 0.988 | 21.3 | 0.975 | best balance |
| v34 u360 (driven-snowflake-111) | 0.995 | 26.3 | 0.988 | high-sr / high-ms point |

---

## Deep-analysis diagnosis (2026-05-08)

After v24 u380 hit the avg_msD 22.7 floor (training-time eval; fresh-eval gives 28.2), a thorough trajectory + metric analysis identified the actual rate-limits:

**Headline conclusions:**

1. **ms is bottlenecked by action-magnitude overshoot, NOT RNN capacity.** v24's empirical action `std` drifts 0.166 → 0.359 from u1 → u520 — entirely from policy mean magnitude growing (log_std frozen at -1.8 throughout). Policy is learning to step BIGGER than 1.0 in raw mode (continuous_scale=1.0, no normalization). Confirmed by v25: with action magnitude pinned to 1.0 by `continuous_normalize`, ms d=0 plateaus at ~28 vs v24's ~18 — the magnitude growth is most of what drives ms below 28. But pushed too far → bang-bang oscillation near goal cell. mean_speed climbs 0.65 → 0.76 between u380→u520 while ms barely moves: bigger steps, more wasted moves.

2. **value_loss is RISING during the plateau** (v24: u380 → u460 → u500 = 2.02 → 3.92 → 3.68). The critic is losing fit, not gaining. Policy outruns it in expanding action-distribution space.

3. **d=10 sr failures are deterministic local minima, NOT Hopfield capacity.** Across all best ckpts, **stochastic eval rescues 2-3pt at d=10** (v24 u380: srD=0.947, srS=0.969). If it were Hopfield capacity-saturation, sampling noise wouldn't help. The cue is correct — the policy gets stuck between two distractor "pulls."

4. **`move_ent_coef` has zero gradient effect when log_std is frozen** — `entropy(N(μ, σ))` depends only on σ. The v21/v24 vs v16/v20 advantage that I attributed to ent_coef must be either single-seed noise, store-entropy interactions, or numerical Adam-sensitivity. Not a "real" entropy bonus on the move policy.

5. **Store head is doing nothing.** store_entropy stays pinned at log(2) ≈ 0.69 throughout. All discriminative work is in the move policy.

**This motivates v28-v30** (below): A targets the action-overshoot bottleneck, B targets the d=10 stochastic-rescue gap with proper noise, C tests whether v26's flat curriculum result was due to recipe (v20 base) rather than the curriculum lever itself.

---

## v18d39_size20_v28 — Exp A: v24 + max_action_norm 1.5

- **Run:** slurm job `13602970`. Started 2026-05-08. v24 base (raw + h=1024 + ent_coef=0.005) plus new `--max_action_norm 1.5` flag (env clips L2(action) to ≤1.5 before applying scale).
- **Hypothesis:** v24's empirical action `std` drifted to 0.359 by u520 → policy means stepping ~2.0 → bang-bang oscillation near goal. Soft cap at 1.5 preserves direction, kills overshoot.
- **Predicted:** avg_msD ↓ from 28.2 to 16-22; d=10 srD unchanged.
- **Result (in progress, u600):** u160 was a transient peak. Plateau u200-450 at ms 33-37, then late-training breakthrough u500+ → **u600 avg ms 20.6 / min_sr 0.959**. Pareto-incomparable with v27 u840 (ms 19.0 / min_sr 0.953) — v28 slightly slower but more reliable. Cap=1.5 thesis vindicated, just needed more training time than predicted.
- **Required code:** added `EnvConfig.max_action_norm` (config.py), env clipping in both `ContinuousGridEnv.step` (env.py) and `ContinuousVecEnv.step` (vec_env.py), CLI flag in train_phase_a_only.py.

---

## v27 update (lively-surf-104, v25 + ent_coef=0.005)

- **Result (in progress, u540):** d=0 sr 1.000 / ms **21.1**! d=10 0.978 / ms 22.8 → **avg srD 0.977 / avg ms 23.8 / min_sr 0.953**. **Likely surpasses v24 u380 (fresh-eval msD 28.2)** — pending fresh eval. Confirms ent_coef + continuous_normalize gets to a better ms floor than either alone.

---

## v18d39_size20_v29 — Exp B: v24 + scheduled log_std anneal -1.8 → -1.4 over u300-u500

- **Run:** slurm job `13602971`. Started 2026-05-08. v24 base + new `--log_std_anneal_*` flags (programmatic interpolation of `agent.movement_log_std` from -1.8 to -1.4 across u300→u500). Stays at -1.4 thereafter.
- **Hypothesis:** d=10 srS-srD gap of ~3pt across all best ckpts means stochastic noise unsticks deterministic local minima at distractor confusions. Wider σ in late training gives policy gradient access to those neighborhoods.
- **Predicted:** d=10 srD ↑ from 0.95 to 0.97; ms unchanged or +1.
- **Result:** in progress.
- **Required code:** CLI flags + interpolation block in `run_phase_a_sweep` loop.

---

## v18d39_size20_v30 — Exp C: v24 + distractor curriculum 10→20 over 1500u

- **Run:** slurm job `13602972`. Started 2026-05-08. v24 base + curriculum flags from v26.
- **Hypothesis:** v26 (curriculum on v20 base) was flat on d=10 sr (~0.95 throughout u100-u440). Test whether v24's recipe (h=1024 + ent_coef=0.005) lets the curriculum actually push d=10.
- **Predicted:** avg srD up to 0.99 by u600 IF recipe matters; flat IF curriculum truly doesn't help.
- **Result (in progress, u400):** Big jump u380→u400: avg ms 32.9 → **24.4**. Per-d: d=0 1.0/22.2, d=5 0.96/26.5, d=10 0.975/24.6. avg srD 0.978 / min_sr 0.959. **Now competitive with v27 u600 (avg ms 23.3)** and beats v24 u380 fresh-eval (28.2). Confirms curriculum needs the v24 recipe to bite.

---

## v18d39_size20_v16 — single 3000u long-run from scratch (canonical recipe)

- **Run:** wandb name `legendary-fleet-93` (run id `vd6tpidq`), ckpt `phase_a_u<TBD>.pt`, slurm job `13374374`, node `node3905`. Started 2026-05-05 18:22 UTC. Agent input_dim=22.
- **Diff vs prior best:** Single-shot fresh training instead of chained-resume; `--phase_a_updates 3000`, `--ppo_clip_coef 0.15` (vs v15's 0.05), `--explore_goals_off` added; `--time=7-00:00:00`. Otherwise identical to v15 recipe (size 20, 400-step rollouts, multistep recall 1/2/3, `--input_hopfield_raw`, freeze log_std −1.8, ε 0.4 anneal 200u, novelty 0.3, wall 0.1, persistence 0.05, 80 envs/world, 0–10 distractors).
- **Result (CANCELLED u280):** cov stable 0.27–0.29. sr d=0: peaked 0.944 at u200 → 0.93 → 0.89 → **0.78** at u260. sr d=10: peaked 0.953 → 0.89 → 0.89 → **0.72**. Net −23pt on sr d=10 over 60u post-anneal.
- **Best ckpt:** `phase_a_u200.pt` — sr d=0 0.944, sr d=10 0.953, cov avg 0.293. Best Phase-A 20×20 sr to date on a fresh single-shot run.
- **Verdict:** Cancelled at u280 — peak ckpt already saved, continued runs were just confirming ongoing degradation. GPU reallocated to v21 (ent_coef test).

---

## Headline findings

1. **`move_ent_coef=0.005` is the more important lever than `hidden_size`.** v21 (h=512 + ent_coef=0.005) at u380 hit avg_sr **0.987** — highest of any ckpt across all runs, beating v20 (h=1024, no ent_coef) at every u. The entropy floor seems to be doing more work than the extra capacity. v20 still has lowest cst (33.0 at u360) due to faster bootstrap, but v21 is catching up.

2. **Best ckpts (sr+ms, min_sr ≥ 0.95):**
   - **`glamorous-valley-101/phase_a_u380.pt`** (v24, h=1024+ent_coef=0.005): avg_sr 0.977 / **avg_ms 22.7** / min_sr 0.953. **New ms leader.**
   - **`unique-field-102/phase_a_u100.pt`** (v25, action norm): avg_sr 0.984 / avg_ms 27.3 / **min_sr 0.975**. Highest min_sr; bootstrap champion (5h training).
   - **`zany-capybara-98/phase_a_u380.pt`** (v21): **avg_sr 0.987** / avg_ms 35.6. Highest avg_sr.

3. **`hidden_size=1024 + move_ent_coef=0.005` (v24)** late-training pulled ahead of all baselines. v24 u380 cst floor (avg_ms 22.7) is below v20's plateau (24.8). The combo of capacity + entropy floor reaches further than either alone with extended training.

4. **`--continuous_normalize` accelerates bootstrap dramatically** but doesn't lower the late-training ms floor below v24's (v25 u440: avg_ms 31.1 vs v24 u380: 22.7). Useful for cheap exploration of recipe variants in fewer updates.

3. **`hidden_size=1024` accelerates bootstrap, but isn't required for top quality.** v20 reaches sr=1.0 by u100 vs v21 reaches it by u280. With enough training (u380), the smaller model + ent_coef matches or exceeds the larger model.

4. **Post-ε-anneal at u200, raw signal decisively beats normalized.** v16 sr d=10 = 0.953 vs v17's 0.772 at the matched eval. v17 cancelled at u210.

5. **`freeze_log_std` is a no-op when `move_ent_coef=0`** (v18 negative result). PPO doesn't pressure log_std without entropy bonus, so the freeze flag is belt-and-suspenders.

6. **`time_penalty=0.05` (v23) produced higher cov but slower follow** than v20. Useful for cov-prioritized workloads but doesn't lower cst. Post-ε-anneal cov 0.41 at u200 (vs v20's 0.38) at sr=1.0.

---

## Reporting note (2026-05-07)

Switched headline metric from `cst = (ms × n_succ + max_steps × n_fail) / n_total` to **paired (avg_sr, avg_ms) reporting** — clearer when sr stays above 0.95 (since cst becomes ms-dominated anyway) and avoids the conflation between "agent is slow" and "agent fails sometimes".

---

## v18d39_size20_v26 — distractor curriculum (max 10→20 over 1500u)

- **Run:** wandb name `zany-wood-103`, slurm job `13491684`. Started 2026-05-07 09:50 UTC. Replaces cancelled v21.
- **Diff vs v20:** added `--n_train_distractors_max_end 20 --n_train_emp_distractors_max_end 20 --distractor_curriculum_updates 1500`. Required adding curriculum CLI flags + linear ramp inside the update loop.
- **Why:** d=10's 4-5% miss rate dominates avg cst. Targeting d=10 sr → 0.99 should drop avg ms toward ~25.
- **Result (in progress, u130):** u80: 0.11/0.88/74 → u100: 0.18/0.981/51 → u120: **0.24/0.980/56.5** (cov/avg_sr/avg_ms). 2nd high-quality eval. Tracking similar to v20 baseline (curriculum still at max≈11).
- **Verdict:** Healthy. Curriculum effect not visible yet (max barely above 10). Real signal at u500+.

---

## v18d39_size20_v25 — env-side action normalization (--continuous_normalize)

- **Run:** wandb name `unique-field-102`, slurm job `13491171`. Started 2026-05-07 09:35 UTC. Replaces cancelled v20.
- **Diff vs v20:** added `--continuous_normalize` (new CLI flag). Env unit-normalizes the action vector before applying.
- **Result (in progress, u120):** u80: 0.30/0.946/30 → u100: 0.34/0.984/27.3 → u120: **0.40/0.952/31.2** (cov/avg_sr/avg_ms). cov pushed past 0.40; sr regressed to 0.95 (cov-vs-sr tradeoff at this stage). u100 still v25's best ckpt.
- **Verdict:** u100 is **Pareto-best on robustness** (min_sr 0.975 vs v20 u360's 0.959). ε=0 at u200 is the next critical event.

---

## v18d39_size20_v24 — best-of-both: ent_coef=0.005 + hidden_size=1024

- **Run:** wandb name `glamorous-valley-101`, slurm job `13484221`. Started 2026-05-07 04:36 UTC. Replaces cancelled v23.
- **Diff vs v20:** add `--move_ent_coef 0.005`. Diff vs v21: `--hidden_size 512` → `--hidden_size 1024`. Tests whether the two best levers (capacity + entropy floor) combine for a new absolute best.
- **Result (in progress, u210):** u160: 0.39/0.958/55 → u180: 0.42/0.906/48 → u200: **0.46/0.902/39** (cov/avg_sr/avg_ms). post-anneal: ms dropped to 39 (best v24 ms ever), cov pushed to 0.46, but sr only 0.90 (min_sr 0.88).
- **Verdict:** v24 post-anneal: high cov, fast follow, but sr below quality threshold. Not Pareto-best. v25 (action norm) still leads.

---

## v18d39_size20_v23 — direct cst lever: --time_penalty 0.05 (5× default)

- **Run:** wandb name `sleek-aardvark-100`, slurm job `13447956`. Started 2026-05-06 16:18 UTC. Replaces cancelled v22.
- **Diff vs v20:** added `--time_penalty 0.05` (was hardcoded default 0.01). Required adding `--time_penalty` CLI flag in `train_phase_a_only.py`. Targets cst directly: every step now costs 5× more, while goal_reward stays 5.0.
- **Result (CANCELLED u240):** u200: 0.407/1.000/44.8 → u220: 0.461/1.000/49.5 → u240: **0.458/0.975/45.5** (cov/sr d=0/ms). cov stable ~0.46. cst hovering 58–64 post-anneal. avg cst u240 = 63.7.
- **Verdict:** Time_penalty 0.05 lever produces high-cov + moderate-cst — different shape than v20, not better on cst. Cancelled to free GPU for v24 (best-of-both). v23 u200 ckpt remains a viable cov-prioritized choice (cov 0.41 / sr 1.0 / cst 57.5).

---

## v18d39_size20_v22 — explore rollouts get goal reward (--no-explore_goals_off)

- **Run:** wandb name `hardy-waterfall-99`, slurm job `13424825`. Started 2026-05-06 09:27 UTC. Replaces cancelled v19.
- **Diff vs v20:** `--explore_goals_off` → `--no-explore_goals_off`. Explore rollouts now get +goal_reward (5.0) on goal-cell hit and teleport on success, like nav rollouts. Hypothesis: unifying the reward structure may help generalization; risk is goal_reward (5.0) drowning out novelty (0.3) in explore signal.
- **Result (CANCELLED u120):** u80: 0.130/0.787 → u100: 0.139/0.838 (cov/sr d=0). Matched v16 baseline sr at u100 but slower bootstrap than v20. No instability or breakage; just slower learning under the goal+novelty competition.
- **Verdict:** Cancelled at u120 — captured the signal (explore_goals_on slows learning vs the canonical recipe but doesn't break it). GPU reallocated to v23 (cst-targeted time_penalty test). If v20 plateaus, worth revisiting v22 with rebalanced novelty/goal_reward.

---

## v18d39_size20_v21 — soft entropy floor: move_ent_coef=0.005 on v16 recipe

- **Run:** wandb name `zany-capybara-98`, slurm job `13423537`. Started 2026-05-06 08:14 UTC. Replaces cancelled v16.
- **Diff vs v16:** `--move_ent_coef 0` → `--move_ent_coef 0.005`. Tests whether a small entropy bonus prevents the post-anneal sr collapse that hit both v16 and v17. Frozen log_std prevents the V6-era log_std-blowup failure mode.
- **Result (CANCELLED u500):** trajectory plateau-ish u380–u500 — sr d=0 stayed ≥0.997, ms drifted 31.5 → 32.4. cov pushed to 0.50.
- **Verdict:** Plateau. **u380 ckpt remains v21's best** (avg sr 0.987, avg ms ~36). Cancelled at u500 — h=512 ceiling reached. GPU reallocated to v26 (distractor curriculum).

---

## v18d39_size20_v20 — capacity bump on the winning recipe (raw + hidden_size 1024)

- **Run:** wandb name `glamorous-field-97`, slurm job `13414706`. Started 2026-05-06 05:43 UTC. Replaces cancelled v17.
- **Diff vs v16:** `--hidden_size 512` → `--hidden_size 1024`. Tests if capacity helps the post-anneal-winner raw recipe push past v16's eventual plateau.
- **Result (CANCELLED u560):** u540: 0.547/0.972/21.6 (cov 0.55 — best ever, ms d=0 21.6 — new low). avg cst u540 = 47.0 (d=5/d=10 sr at 0.92). Plateau confirmed: cst oscillating 33–47 since u340.
- **Verdict:** Plateau. **u360 ckpt remains v20's best** (cst 33.0). Cancelled at u560 — marginal value of more training is low; GPU reallocated to v25 (action normalization test).

---

## v18d39_size20_v19 — capacity bump: hidden_size 1024 on v17 recipe (open question #3)

- **Run:** wandb name `eager-water-96`, slurm job `13407578`, node `node3905`. Started 2026-05-06 02:43 UTC. Agent input_dim=22 (multistep dims dominate; same as v17). Replaces cancelled v18.
- **Diff vs v17:** `--hidden_size 512` → `--hidden_size 1024`. Tests whether more capacity helps the norm-signal recipe push past v17's eventual plateau.
- **Result (CANCELLED u120):** cov u20=0.033 → u100=0.167. sr d=0: 0.028 → u80=0.86 → u100=0.84. Pre-anneal trajectory ~1.5× faster than v17.
- **Verdict:** Cancelled at u120 to free GPU for v22 (explore_goals_on test). Lower marginal info than v22 since raw beats norm post-anneal in v17 already; v20 will independently test the h=1024 stabilization question on the winning recipe.

---

## v18d39_size20_v18 — ablation: --no-freeze_log_std (open question #2)

- **Run:** wandb name `zesty-mountain-95` (run id `nu4w7xjj`), slurm job `13390223` (CANCELLED at u100), node `node3905`. Started 2026-05-05 20:48 UTC.
- **Diff vs v16:** flip `--freeze_log_std` → `--no-freeze_log_std`. Init_log_std stays at −1.8.
- **Result (CANCELLED u100):** v18 was byte-identical to v16 through u100 — every eval row, every train log line, even the empirical `std` value matched v16's exactly. Conclusion: with `move_ent_coef=0`, PPO doesn't pressure log_std in this regime, so `--freeze_log_std` is effectively a no-op. v18 was producing zero new information.
- **Verdict:** discarded as redundant. Useful negative result: freezing log_std doesn't matter when entropy bonus is off — it's belt-and-suspenders. To meaningfully test unfrozen log_std, would need to also re-enable `--move_ent_coef > 0`. GPU hours reallocated to v19.

---

## v18d39_size20_v17 — ablation: --no-input_hopfield_raw (open question #1)

- **Run:** wandb name `carbonite-astromech-94` (run id `ahddjo23`), slurm job `13375478`, node `node3905`. Started 2026-05-05 18:29 UTC. Agent input_dim=22 (same as v16, since multistep dims dominate).
- **Diff vs v16:** flip `--input_hopfield_raw` → `--no-input_hopfield_raw`. Single-knob test of whether normalized direction signal is better than raw projected q given multistep recall (1/2/3) already provides recall-trajectory magnitude info.
- **Result (CANCELLED u210):** cov 0.146→0.185→0.261→0.272→0.324→0.324 (u200). sr d=0: 0.73→0.70→0.93→0.86→0.87→0.83. sr d=10: 0.72→0.68→0.86→0.84→0.84→0.77. Post-ε-anneal sr collapsed to 0.77 d=10 (vs v16's 0.95). std climbing 0.18→0.20 (policy losing concentration), move_entropy growing −0.49→−0.39.
- **Verdict:** discarded — pre-anneal v17 looked better only because ε noise was helping it; once ε=0 its weaker direction signal can't lock onto goals. Useful negative result: with multistep recall channels, the *normalization* of the main signal still matters when noise is removed. GPU hours reallocated to v20 (raw + hidden=1024).
