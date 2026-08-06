> **Archived.** Moved out of `hopfield_nav/` by phase 6 of the 2026-08
> refactor. Not maintained; describes what was believed and tried at the time,
> which in places is no longer true of the code. Start from `docs/archive/README.md`
> for what replaced it.

# Phase A training — full reference

Companion to `PHASE_A_SIZE20.md` (recipe + workflow) and `PHASE_A_SIZE20_FINDINGS.md` (experiment log). This doc explains **what every CLI flag actually does** in code. Behavior here is verified against `train_phase_a_only.py`, `rollout.py`, `config.py`, `env.py`, `vec_env.py`, `agent.py`, `ppo.py`, and `train_phased.py` as of `merge-workspace`.

---

## 1. What Phase A is

Phase A trains a single RNN policy to do **two behaviors** from a single shared trunk, *without* training the store action:

1. **Follow** — when a goal pattern is in the Hopfield, the policy's movement should track the Hopfield-derived direction signal toward the goal.
2. **Explore** — when the Hopfield is empty or contains only distractors, the policy should cover the grid.

The store head stays frozen for the entire phase (`set_phase_freeze(agent, freeze_move=False, freeze_store=True, freeze_value=False, freeze_rnn=False)` — `train_phase_a_only.py:108`). A later phase ("Phase B") trains the store head on the frozen Phase A trunk; in the current workflow that step is usually skipped in favor of oracle-store evaluation.

### The two-regime rollout

Every PPO update collects rollouts from **two regimes in parallel**, both stepped by the same agent in the same env-shaped batches:

- **Pre-stored ("follow") rollouts** — Hopfield is pre-loaded with the goal pattern (plus optionally N distractor patterns). `goals_active=True` so the agent gets `+goal_reward` and teleports on goal-reach. **Reward shaping is OFF** here (novelty=0). The follow gradient stays clean: the only positive signal is reaching the goal.
- **Empty ("explore") rollouts** — Hopfield is empty (or contains only distractor patterns, no goal). `goals_active` may be on or off (`--explore_goals_off`). Reward shaping is ON: novelty / revisit / wall / persistence terms drive the coverage gradient.

The split per update is controlled by `--interleave_empty_fraction` (and the anneal flags). Both regimes are part of the same `rollouts` list passed to `ppo_update`, so the policy update mixes gradients from both.

### Phase A is one stream of PPO updates

There is no separate "follow phase" then "explore phase" inside Phase A — each PPO update sees a mix. The `run_phase_a_sweep` loop (`train_phase_a_only.py:167-393`) does, per update:

1. Decide `n_pre_now` and `n_emp_now` (regime counts) based on warmup state and current interleave fraction.
2. For each (world, env), build the Hopfield (pre-stored vs. empty, with or without distractors), set `env.goals_active`, decide `epsilon_now`, optionally refresh the goal.
3. Run a `RolloutCollector.collect_rollout` for that env. The collector handles the reward shaping, ε-greedy override, auto-nav teacher forcing (off in Phase A), Hopfield signal computation, and per-step input assembly.
4. Pool all rollouts → call `ppo_update` once.
5. Optionally log, eval, save checkpoint.

### Optional warmup prefix

`--warmup_explore_only_updates N` runs N updates of **100% empty rollouts** before the interleaved loop. During warmup `n_pre_now=0, n_emp_now=envs_per_world` (`train_phase_a_only.py:220-224`). Use to bootstrap coverage before turning on follow gradient.

---

## 2. Flag reference

Defaults and types come from `train_phase_a_only.py:506-690`. Where a flag overrides a dataclass field, the dataclass is named.

### 2.1 Required / encoder

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--encoder_checkpoint` | required | str | Path to a `.pt` file produced by `encoder_training`. The encoder's `out_dim` becomes `embed_dim`; `encoder_gain` (loaded with it) becomes the default `cfg.hopfield.beta` if not overridden. |
| `--encoder_gain` | None | float | Overrides the encoder's saved gain. Affects (a) Hopfield `beta` (if `beta` is None at startup, defaults to `encoder_gain`), (b) the `validate_config` check between the encoder and VectorHash lambdas. Leave as None unless you know what you're doing. |
| `--fwhm_ratio` | 0.25 | float | Spatial smoothing radius for `gbook` lookup, in units of grid module periods. Set at scaffold build time and used by VectorHash. Smaller = sharper; larger = smoother. |
| `--lambdas` | `[11, 12, 13]` | list[int] | Grid-cell module periods. Three coprime values is the canonical choice; `Npos = prod(lambdas) = 1716` for the default. Larger product = larger scaffold = more capacity for distinct env patches. |
| `--Np` | 400 | int | Place-cell count. Sets `VectorHashConfig.Np`. Only used when `static_vectorhash=False` (no pbook is built when static). |
| `--static-vectorhash` / `--no-static-vectorhash` | `True` | flag | When True, build `gbook + sbook + encoded_Phi` only — skip `pbook`, `Wgp`, `Wsp`, `Wps`, and the scaffold self-test. Faster startup, smaller memory. Phase A on 20×20 always uses static. |

### 2.2 Environment

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--size` | 8 | int | Grid side length. Each training env is a `size×size` open grid. Effective state space scales as `size²`. |
| `--observation_size` | 12 | int | Length of the per-env sensory codebook vector (each cell has a fixed `observation_size`-dim random vector). Larger = more discriminable cells, more sensory input dims (only if `--input_sensory`). |
| `--movement_mode` | `continuous` | str | `discrete`: 4 cardinal actions, `Categorical(4)` policy. `continuous`: 2-D `Normal(μ, σ)` policy producing `(dx, dy)` floats. The active workflow uses `continuous`. |
| `--hopfield_mode` | `continuous` | str | `discrete`: Hopfield direction signal is one-hot (4-dim) via `classify_direction_batch`. `continuous`: 2-D unit vector (or raw `q` if `--input_hopfield_raw`). |
| `--goals_active` / `--no-goals_active` | `True` | flag | Sets `EnvConfig.goals_active`. **Note:** Phase A overrides this per-env at rollout time: pre-stored envs always have `goals_active=True`, empty envs have `goals_active = not explore_goals_off`. So this CLI flag effectively only sets the env's *initial* state — `--explore_goals_off` is what you want for "pure explore" behavior. Eval envs always run with `goals_active=True` (`train_phase_a_only.py:436-439`). |
| `--goal_reward` | 1.0 | float | Reward at goal cell when `goals_active`. Larger values make follow gradients dominate; can saturate the value head and destabilize PPO if too high (`--ppo_clip_coef 0.15` often paired). |
| `--goal_radius` | 0.5 | float | L2 ball around goal that counts as "at goal" (`env.at_goal`). 0.5 reproduces snap-equality on integer-snapped positions. 1.0 includes 4-connected neighbors. Larger = easier to register goal-hit; trains looser approach paths. |
| `--time_penalty` | None | float | Overrides `EnvConfig.time_penalty` (default 0.01). Per-step `-time_penalty` reward whenever not at goal. Larger values pressure policy toward shorter trajectories. v23/v35 found 0.05 (5× default) was a strong cst-lever for size 20. |
| `--continuous_normalize` / `--no-continuous_normalize` | None | flag | When True, env unit-normalizes the action vector before scaling by `continuous_scale`. Step magnitude is then fixed at `continuous_scale` (=1.0). Decouples step size from policy μ magnitude. Cancels effects of `max_action_norm` / `min_action_norm`. v25 (`unique-field-102`) tested this on size 20. |
| `--max_action_norm` | None | float | Soft L2 cap on `(dx, dy)` actions in env (continuous mode, `continuous_normalize=False`). Action with `‖a‖ > max` is scaled down preserving direction (`vec_env.py:298-302`, `env.py:369-370`). Targets the v24-era late-training overshoot from policy-mean magnitude growth. Mean over time can climb past 1.0 with a 0-init mean head and frozen log_std; capping at 1.5 (v28) or 1.0 (v31) preserves direction. |
| `--min_action_norm` | None | float | Soft L2 floor on actions. When `0 < ‖a‖ < min`, scale up preserving direction. Combined with max, action L2 is clamped to `[min, max]`. Forces a minimum step magnitude so near-zero mean predictions don't waste steps. Only applies when `continuous_normalize=False`. |

### 2.3 RNN inputs

Each enabled channel adds dims to the input vector (`agent.compute_input_dim` — `agent.py:11-30`). `current_reward` (1) is always present. Order in the concatenated vector matches the order below.

| Flag | Default | Adds dims | What it does |
|---|---|---|---|
| `--input_prev_reward` / `--no-input_prev_reward` | True | 1 | Pass last step's reward as an explicit channel. Reset to 0 on rollout start and after teleport. RNN anchor for "did I just hit goal". |
| `--input_encoded_state` / `--no-input_encoded_state` | False | `embed_dim` (= 1024 typical) | Pass the encoded grid-state directly. Off in canonical recipe — sensory codebook + Hopfield signal already carry the needed info, and `embed_dim` swamps the RNN input layer. |
| `--input_hopfield_signal` / `--no-input_hopfield_signal` | True | 4 (discrete) or 2 (continuous) | The Hopfield-derived direction. Continuous: either unit `q/‖q‖` (default) or raw `q` if `--input_hopfield_raw`. Discrete: 4-d one-hot via `classify_direction_batch`. **Off → no Hopfield signal at all** — explore-only policy. |
| `--input_hopfield_raw` / `--no-input_hopfield_raw` | True | (same 2 as signal) | Continuous mode only. When True, the `input_hopfield_signal` channel carries raw `q` (`rollout.py:368-372`) — magnitude encodes "memory present" and "signal confidence." When False, `q` is L2-normalized. The doc's open question 1 — V17 (norm) regressed sr d=10 by 0.18 vs V16 (raw) at matched eval. |
| `--input_hopfield_multistep` | `[]` | 2 × len(list) | Project Hopfield recall at each of these iteration counts (e.g. `1 2 3` → 6 extra dims). Implemented via `recall_batch_trajectory` and `_compute_multistep_q` (`rollout.py:710-763`). Recall *trajectory* carries "is this attractor real" info that converged recall hides. Continuous mode only. **Load-bearing** for the size-20 recipe per `PHASE_A_SIZE20.md`. |
| `--input_sensory` / `--no-input_sensory` | True | `observation_size` (=12 default) | Pass the env's per-cell sensory codebook vector. Without sensory the agent corner-traps in explore mode (V18d27/V18d38/V18d41 confirmed). **Load-bearing.** |
| `--input_prev_action` / `--no-input_prev_action` | True | 4 (discrete) or 2 (continuous) | Pass last step's action. Reset to 0 on rollout start and after teleport. Canonical size-20 recipe disables this (`--no-input_prev_action`). |
| `--input_goal_in_memory` / `--no-input_goal_in_memory` | False | 1 | A 1-bit flag indicating "the agent has stored a goal pattern this rollout" (pre-stored rollouts: True from t=0; empty rollouts: flips True when agent stores at goal). **The "bit" referenced in PHASE_A_SIZE20.md.** Trivially lets policy distinguish follow vs explore — we deliberately leave this OFF and force the policy to infer from natural inputs (recall structure, sensory, prev_reward). |

### 2.4 Agent architecture / policy

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--hidden_size` | 128 | int | GRU hidden width. Canonical size-20 v15 uses 512; v20/v24 used 1024 (faster bootstrap, higher capacity ceiling, no late-training advantage shown in FINDINGS). |
| `--num_rnn_layers` | 1 | int | GRU layer count. Always 1 in active workflow. |
| `--init_log_std` | -0.5 | float | Initial `movement_log_std`. Translates to σ via `exp(log_std)`. `-0.5 → σ≈0.61`; `-1.5 → σ≈0.22`; `-1.8 → σ≈0.165`; `-2.0 → σ≈0.135`. Smaller σ → more deterministic policy. Init negative so deterministic eval (`action = μ`) is meaningful; zero-init lets PPO entropy bonus inflate σ to 2-3 per dim, making the policy reliant on sample noise. |
| `--freeze_log_std` / `--no-freeze_log_std` | False | flag | When True, `movement_log_std` is held at `init_log_std` with `requires_grad=False` (`agent.py:62-63`). Pins variance so PPO loss directly pressures the policy mean instead of the policy "hiding" the mean inside large σ. V18 found this is a no-op when `move_ent_coef=0` (PPO doesn't push log_std without an entropy bonus). |

### 2.5 Phase A core scheduling

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--phase_a_updates` | 600 | int | Number of *interleaved* updates after any warmup. Total updates = `warmup_explore_only_updates + phase_a_updates`. |
| `--phase_a_lr` | 3e-4 | float | Adam LR for the trunk + move + value heads. Store head is frozen so its params don't enter the optimizer. |
| `--phase_a_novelty_reward` | 0.1 | float | Per-first-visit-cell bonus on **empty-regime envs only**. Sets `cfg.hopfield.novelty_reward` per rollout (`train_phase_a_only.py:267, 297`). 0.3 was sweet spot on size 20; 0.5 destabilized (v18d39_size20_v14). 0 disables. |
| `--warmup_explore_only_updates` | 0 | int | Pure-explore prefix: first N updates are 100% empty rollouts. After warmup the loop switches to interleaved. Useful to bootstrap coverage before any follow gradient. Total budget = warmup + phase_a_updates. |
| `--novelty_anneal` / `--no-novelty_anneal` | False | flag | If True, linearly anneal `phase_a_novelty_reward` from `base` → 0 over the full `n_updates_total` (warmup + phase_a_updates). Scale = `max(0, 1 - (u-1)/n_total)` (`train_phase_a_only.py:184-188`). Canonical recipe leaves this OFF. |

### 2.6 Interleave fraction & curriculum

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--interleave_empty_fraction` | 0.5 | float | Fraction of `envs_per_world` that runs the empty regime each update. `1.0` = pure explore (after warmup ends), `0.0` = pure follow, `0.5` = half-half. Computed as `n_emp = round(envs_per_world * frac)`, `n_pre = envs_per_world - n_emp`. |
| `--interleave_empty_target` | None | float | If set, anneal `interleave_empty_fraction` → this value linearly over `interleave_anneal_updates` updates, then hold. Lets you start mostly-explore and ramp to half-half (or vice versa). None = no anneal. |
| `--interleave_anneal_updates` | 0 | int | Anneal window for the interleave fraction. 0 = no anneal (use initial value forever). |

### 2.7 Goal handling

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--randomize_goal_per_rollout` / `--no-randomize_goal_per_rollout` | False | flag | Call `env.reset_goal()` at the start of each rollout for **empty-regime envs only** (`train_phase_a_only.py:308-309`). Breaks "in env X go to position Y" memorization shortcut. Skipped for pre-stored envs (re-rolling the goal would invalidate the pre-loaded Hopfield pattern). |
| `--explore_goals_off` / `--no-explore_goals_off` | False | flag | When True, set `env.goals_active = False` for empty-regime envs (`train_phase_a_only.py:300`). Empty envs get no +goal_reward and no teleport on goal-reach. Forces explore mode to be rewarded *purely* by novelty / revisit / wall / persistence / time-penalty. Pre-stored envs always keep `goals_active=True`. Eval envs always use `goals_active=True`. |

### 2.8 Exploration override (ε-greedy)

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--epsilon_explore` | 0.0 | float | Per-step probability of replacing the policy's sampled action with a uniform-random direction (`rollout.py:249-264`). Continuous: random angle on the unit circle. Discrete: uniform over 4 actions. **Applied only to empty-regime envs** — pre-stored envs need clean follow signal. The agent re-scores log_prob under its current policy on the overridden action so PPO's importance ratio remains well-defined; PPO masks these steps out of move_loss via `policy_action_mask` (`rollout.py:421-422`, `ppo.py`). |
| `--epsilon_anneal_updates` | 0 | int | Linearly anneal `epsilon_explore` from base → 0 over this many updates (`_compute_epsilon`, `train_phase_a_only.py:41-51`). 0 = constant. V18d39_size20_v8 found sustained ε (`epsilon_anneal_updates=750`) hurt sr (0.68 vs 0.89) — too much random action during follow. |

### 2.9 Reward shaping (empty-regime only, "in_explore")

All four below apply only during the `in_explore` phase of each rollout (`cfg.explore_steps is None` → always in_explore, the default). Each fires independently — earlier code coupled them all under `need_visited`, masking `wall_penalty` and `revisit_penalty` when `novelty=0`; that's fixed.

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--revisit_penalty` | 0.0 | float | Per-step `-revisit_penalty` when post-step cell was already visited this rollout (`rollout.py:498-501`). Zeroed for teleport rows (`moved` mask). Densifies coverage gradient — novelty alone goes silent on revisits; this keeps signal alive late in a rollout. v21 used 0.05. |
| `--wall_penalty` | 0.0 | float | Per-step `-wall_penalty` when post-step `(x, y)` is on a grid edge (`x in {0, size-1}` or `y in {0, size-1}`) (`rollout.py:507-515`). Counters perimeter-walk basin learned when novelty rewards walking along edges. Canonical size-20 recipe uses 0.1. |
| `--persistence_bonus` | 0.0 | float | Per-step `+persistence_bonus × cos(action_t, action_{t-1})` (`rollout.py:516-534`). Continuous: cos sim of raw 2-D vectors (normalized). Discrete: one-hot dot product (1 iff same direction else 0). Stateless dense alternative to revisit_penalty. Canonical recipe uses 0.05. |
| `--novelty_scale_remaining` / `--no-novelty_scale_remaining` | False | flag | When True, novelty bonus per new cell is `base × min(cap, total_cells / n_remaining_unvisited)` (`rollout.py:479-493`). Late-game (rare) cells pay more, keeping coverage gradient alive as it saturates. |
| `--novelty_scale_cap` | 10.0 | float | Upper bound on the remaining-scale multiplier. Prevents value-head instability from rare-cell jackpots. Only used when `novelty_scale_remaining=True`. |

**Reward computation order per step in empty-regime:**

1. `current_reward = goal_reward if at_goal & goals_active else -time_penalty` (computed pre-step from current position, `rollout.py:191-198`).
2. Agent acts; env steps; `vec.step_batch` returns the reward at the post-step position with the same logic.
3. `store_cost`, `store_bonus` (Phase A: 0 since store frozen and `effective_bonus=0`).
4. Novelty / revisit shaping at post-step position, masked by `moved = ~at_goal_pre`.
5. Wall, persistence shaping (also masked by `moved`).

### 2.10 Distractors (training-time)

Distractors are extra Hopfield patterns sampled from grid cells *outside the env's region* in the global VectorHash scaffold, then shuffled into the Hopfield. Matches eval-time distractor setup so training/eval distributions align.

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--n_train_distractors_min` | 0 | int | Min distractors per **pre-stored** rollout. Per-rollout count `N ~ Uniform[min, max+1)` (`train_phase_a_only.py:246-249`). |
| `--n_train_distractors_max` | 0 | int | Max distractors per pre-stored rollout. **Setting `max > 0` enables variable-count distractors.** When `max=0` (default), pre-stored rollouts use the cached `pre_pools` Hopfield with only the goal pattern. |
| `--n_train_emp_distractors_min` | 0 | int | Min distractors per **empty** rollout (no goal pattern). Trains the explore policy to ignore non-goal recall signals — so eval-time distractors don't trigger spurious follow behavior in explore mode. |
| `--n_train_emp_distractors_max` | 0 | int | Max for empty rollout. `max > 0` enables; when 0, empty rollouts use cached `emp_pools` (empty Hopfield). |
| `--n_train_distractors_max_end` | None | int | Curriculum end value for `n_train_distractors_max`. Linear ramp `max → max_end` over `distractor_curriculum_updates` updates (`train_phase_a_only.py:192-201`). None = no curriculum. |
| `--n_train_emp_distractors_max_end` | None | int | Curriculum end for `n_train_emp_distractors_max`. |
| `--distractor_curriculum_updates` | 0 | int | Window for the distractor max ramp. 0 = no curriculum. v18d39_size20_v26 / v30 use this. |

Distractor sampling rejects cells inside the env's region (`vec_env`-relative bounds, `train_phase_a_only.py:257-258`).

### 2.11 log_std anneal (programmatic)

Lets you slowly *change* the policy's σ from one value to another during training, independent of (or in conjunction with) PPO's gradient updates. Implemented as a direct write to `agent.movement_log_std.data` each update (`train_phase_a_only.py:170-181`).

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--log_std_anneal_start_update` | 0 | int | Update at which to start interpolating `log_std` toward target. 0 disables the anneal (since `end > start` is required). |
| `--log_std_anneal_end_update` | 0 | int | Update at which interpolation completes (log_std reaches target). Must be > start. |
| `--log_std_anneal_target` | None | float | Target log_std. e.g. -1.4 → σ≈0.247. None = no anneal. |

Interaction with `--freeze_log_std`:
- With `--no-freeze_log_std`: anneal sets the new value each update; PPO can then move it further via the entropy bonus (if `ent_coef > 0`).
- With `--freeze_log_std`: anneal sets it; `requires_grad=False` keeps PPO from touching it. The log_std stays at the annealed value once `end_update` is reached.

### 2.12 PPO overrides

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--move_ent_coef` | None | float | Override `PPOConfig.ent_coef` (default 0.01 — but most variants set this to 0). Entropy bonus weight on the move policy. **Has zero gradient effect on policy mean when log_std is frozen** (entropy of `N(μ, σ)` depends only on σ; frozen → no pressure). When `log_std` is unfrozen, `move_ent_coef > 0` *inflates* σ. v21/v24 used 0.005. |
| `--ppo_clip_coef` | None | float | Override `PPOConfig.clip_coef` (default 0.2). PPO ratio clip bound — limits per-update policy change. Lower (0.10-0.15) helps stability when `goal_reward > 1` inflates value targets (canonical size-20 recipe uses 0.15). |

The rest of PPO (`gamma=0.99`, `gae_lambda=0.95`, `vf_coef=0.5`, `store_ent_coef=0.05`, `max_grad_norm=1.0`, `ppo_epochs=4`, `n_minibatches=4`) is not CLI-exposed — change the dataclass defaults in `config.py` if you need.

### 2.13 Rollout / batch sizing

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--batch_envs` | 16 | int | Parallel trajectories per env per rollout. The vectorized env runs B trajectories from random starts; PPO pools all across envs/worlds. Set to `cfg.batch_envs`. |
| `--steps_per_rollout` | 400 | int | T (timesteps per rollout). Each env produces a (B, T) tensor of obs/actions/rewards. Larger T → longer credit-assignment horizon, more env steps per update, slower. |
| `--num_worlds` | 1 | int | Independent VectorHash scaffolds (with their own envs) per update. Always 1 in current recipe — multi-world was used in earlier phased pipeline. |
| `--envs_per_world` | 20 | int | Envs per world (`world["envs"]` length). Sets the regime split (`n_pre`, `n_emp`). Canonical size-20 uses 80. |

Total env-rollouts per update = `num_worlds × envs_per_world`. Total transitions per update = `that × batch_envs × steps_per_rollout`. With defaults: `1 × 20 × 16 × 400 = 128,000` transitions/update; canonical size-20 (80 envs/world, 16 batch_envs, 400 steps): `1 × 80 × 16 × 400 = 512,000` transitions/update.

### 2.14 Evaluation

`do_eval` (`train_phased.py:355-394`) runs three eval families against each distractor count in `val_distractors`:

- `evaluate_navigation` (deterministic, goal pre-stored) → `success_rate`, `mean_steps`
- `evaluate_goal_discovery` (empty Hopfield, must find then return) → `disc_*`
- `evaluate_exploration` (empty Hopfield, coverage focus) → `mean_coverage`

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--num_val_envs` | 10 | int | Distinct val envs (each with its own codebook/goal). Eval world built once at startup with `goals_active=True` regardless of train-time setting. |
| `--n_val_trials` | 32 | int | Parallel trials per env per metric per distractor count. |
| `--val_distractors` | `[0]` | list[int] | Distractor counts to evaluate at. `0 5 10` is canonical for size-20 leaderboard. |
| `--union_cov_trials` | 0 | int | If >0, also runs `evaluate_union_coverage` (multi-rollout cov: across N attempts, what fraction of grid does the agent visit at least once). 10 is a good default. 0 = skip. |
| `--eval_every` | 50 | int | Eval cadence in updates. Each eval also saves a checkpoint at `<save_dir>/phase_a_u<u>.pt`. Set to 10-25 for short runs, 50+ for long. |

### 2.15 General

| Flag | Default | Type | What it does |
|---|---|---|---|
| `--seed` | 0 | int | Sets `torch.manual_seed`, `np.random.seed`, and the per-world RNG. Distractor RNG = `cfg.seed + 7919` (`train_phase_a_only.py:128-129`). |
| `--device` | `cuda` | str | Falls back to `cpu` if CUDA unavailable. |
| `--save_dir` | None | str | Output dir for checkpoints. None = resolved to `checkpoint/phase_a_only_<wandb-run-name-or-timestamp>` after wandb init. |
| `--use_wandb` | False | flag | Enable wandb logging. |
| `--wandb_project` | `hopfield-nav-phase-a-sweep` | str | wandb project name. |
| `--load_checkpoint` | None | str | Path to a Phase A/B `.pt` to resume from. Loads `agent_state_dict` into the freshly built agent. World/encoder/optimizer state NOT loaded — only network weights. Resume with the same architecture flags as the original. Size-curriculum resumes (size 14 ckpt → size 20 run, v18d39_size20_v10) NaN-crashed. |

---

## 3. Code paths worth knowing

These are the "where to look" pointers for non-obvious behavior:

### Rollout collector regime branching

In `RolloutCollector.collect_rollout` (`rollout.py:48-639`), three things change per-env based on whether it's pre-stored or empty:

- **Hopfield setup** — caller-controlled. Phase A passes the right Hopfield object per env from `pre_pools[w_idx][local_idx]` or `emp_pools[w_idx][local_idx]` (or freshly constructs one with distractors).
- **`novelty_reward`** — caller flips `cfg.hopfield.novelty_reward = 0.0` for pre-stored (`train_phase_a_only.py:267`) and `= current_novelty` for empty (`:297`). The collector reads `cfg.hopfield.novelty_reward` at the top of the loop to decide `novelty_on`, so the flip per-env *while sharing a config* is load-bearing.
- **`env.goals_active`** — caller sets it per env (`:269` for pre-stored True, `:300` for empty `not explore_goals_off`). The collector reads `vec.goals_active` per step for the reward computation.

After the rollouts loop the caller resets `cfg.hopfield.novelty_reward = 0.0` (`:329`) to leave config in clean state for the next update.

### What gets saved to `agent_goal_store_fired`

`rollout.py:105` initializes a per-env bool to `goal_in_memory_init` (caller-passed, True for pre-stored, False for empty). The flag is OR'd True when `agent_store & at_goal_mask` fires during the in-explore phase (`rollout.py:439-440`). This drives:

1. The BC teacher's "trust the Hopfield" gate (BC mode only).
2. The `input_goal_in_memory` input bit when that flag is enabled.

In Phase A the store head is **frozen**, so `agent_store` is whatever the (untrained) head's Bernoulli sample is — effectively random until trained. The bit therefore flips True at random times during empty rollouts when `--input_goal_in_memory` is on. This is one reason the canonical recipe leaves the bit off.

### Frozen store head implication for replays / Hopfield writes

In Phase A:
- Pre-stored rollouts use a **shared Hopfield per env** (the pre-loaded one or a fresh one with goal + distractors). `shared_hopfield=True` → per-env stores via `hopfields[b].input_memory` are **skipped** (`rollout.py:439`).
- Empty rollouts use **per-trajectory Hopfields** (`emp_pools` is a list-per-env; each env in the regime gets its own fresh Hopfield list). `shared_hopfield=False` → stores fire if the (frozen, random) store head outputs >0.5.

Net effect: the policy sometimes writes garbage memories during empty rollouts, but those affect only that rollout (Hopfields are fresh next update via `make_hops` / fresh-build).

### auto_nav / auto_store warmup

`train_phase_a_only.py:153-154` zeros out both warmups at the start of Phase A: `cfg.hopfield.auto_nav_warmup = 0`, `cfg.hopfield.auto_store_warmup = 0`. These flags only matter in the earlier `train_phased.py` pipeline. They're restored at end of Phase A.

### What "the bit" means

`--input_goal_in_memory` (the "bit") is OFF in the active recipe. With it on, the policy gets a clean 1-bit "follow vs explore" cue. The whole point of bitless training is to force the policy to infer this from natural inputs — multistep recall structure being the main one. Re-read PHASE_A_SIZE20.md §"Why it's hard, and the fix" for the reasoning.

---

## 4. Sanity-check matrix

If you're getting weird results, these are the canonical "is your config sane" combinations:

| Symptom | Check |
|---|---|
| Coverage caps near 0.15-0.30 | `--input_sensory` on? `--input_hopfield_multistep 1 2 3`? wall+persistence pair? (raw novelty alone caps low on size 20) |
| sr d=10 << sr d=0 | Train-time distractors `n_train_distractors_max > 0`? Otherwise eval distractors are out-of-distribution |
| sr drops post-ε-anneal | Try `--epsilon_anneal_updates` shorter (≤200); sustained ε corrupts follow signal |
| ms (mean_steps) > 30 at d=0 | Action overshoot? Try `--max_action_norm 1.5` or `--continuous_normalize` |
| corner-trap in trajectories | `--input_sensory --no-input_sensory` mismatch? wall_penalty too low? |
| `ent_coef` doesn't seem to help | `--freeze_log_std` on? Entropy on `N(μ, σ)` has zero gradient when σ is frozen |
| NaN at resume | Are you changing `--size` from the loaded ckpt? Size-shift on resume is unstable |
| Hopfield signal is garbage | Hopfield β set? If `cfg.hopfield.beta=None`, training defaults it to `encoder_gain` at startup (`train_phase_a_only.py:429-430`). Should not be None at first PPO update. |

---

## 5. What this doc does NOT cover

- `PhasedConfig`, `PhasedConfigV2`, `PhasedConfigV3` phases B and C — Phase A only consumes `PhasedConfigV3.phase_a_updates / phase_a_lr / phase_a_novelty_reward`. The B/C fields are inert in this entry point.
- The `train_phased.py` 4-phase orchestrator (`run_phase1` / `run_phase2` etc.) — kept as a module for the `setup_world` / `make_hops` / `set_phase_freeze` / `do_eval` helpers. Don't invoke the 4-phase pipeline from Phase-A workflow.
- BC / DAgger training. See `EXPERIMENTS_BC.md` and `bc.py` docstring. BC is a separate `training_mode="bc"` flow that uses `RolloutCollector` with `collect_teacher=True` — same collector, different downstream loss.
- RNN-baseline. See `RNN_BASELINE.md` and `train_rnn.py`.
- The encoder. See `encoder_training/`.
