# Codebase Map

Written 2026-08-05 from the code on `main` (`959322f`). Everything here was
checked against the source, not against the older markdown files in
`hopfield_nav/`. Where the code and an existing doc/script disagree, the code
wins and the disagreement is listed in [Known drift](#known-drift-and-landmines).

Companion docs:
- [TRAINING_AND_EVAL_REFERENCE.md](TRAINING_AND_EVAL_REFERENCE.md) — how every
  script runs and what each flag actually does.
- [REFACTOR_ASSESSMENT.md](REFACTOR_ASSESSMENT.md) — migrate-vs-refactor verdict.

---

## 1. What this repository is

A model of goal navigation in which spatial memory is a **Hopfield network over
learned embeddings of grid-cell codes**, and the controller is a small recurrent
policy that decides both *where to move* and *when to write a memory*.

The scientific chain, in the order the code executes it:

```
lambdas = [11,12,13]
   │
   │  cls.vectorhash.assoc_utils_np_2D.gen_gbook_2d
   ▼
gbook  (Ng, Npos, Npos)          Ng = Σλ², Npos = Πλ  — one-hot per module
   │
   │  encoder_training.utils.smooth_gbook(fwhm_ratio)     Gaussian bump per module
   ▼
Phi    (Ng, Npos, Npos)          smoothed grid codes
   │
   │  encoder_training.models.GridEncoder / GridEncoderCNN
   │  trained by encoder_training.train with a contrastive
   │  "near → cos 1, far → cos 0" objective
   ▼
encoded_Phi (Npos, Npos, D)      unit-sphere embedding of every global position
   │                             (VectorHash.precompute_encoded_phi)
   │
   ├─ store:  Hopfield.input_memory(encoded_Phi[goal])   W += z zᵀ / D
   │
   └─ recall: x̂ = tanh(β · W x) ... normalize            (1 step by default)
        │
        │  Gram–Schmidt local frame at the agent's cell:
        │     d_N = Φ[x, y+1] − Φ[x, y],  d_E = Φ[x+1, y] − Φ[x, y]
        │     q   = W_gs · (x̂ − x)        → 2-D (East, North) displacement
        ▼
     policy input → NavAgent (GRU) → {move, store, value}
        │
        └─ trained by PPO (hopfield_nav.ppo) or DAgger BC (hopfield_nav.bc)
```

The **encoder is the object of study** in `encoder_training/`; the **policy on
top of a frozen encoder** is the object of study in `hopfield_nav/`. The encoder
is always loaded frozen (`hopfield_nav/encoder.py:50` `requires_grad_(False)`).

---

## 2. Top-level layout

| Path | Kind | Status | Notes |
|---|---|---|---|
| `cls/` | package | mostly legacy, narrowly live | Original VectorHash research library. Live code uses only a handful of functions (§5). |
| `encoder_training/` | package | live | Encoder architecture, contrastive training, nav-eval, sweeps. |
| `hopfield_nav/` | package | live | Envs, scaffold, Hopfield, policy, PPO/BC, evals, figure pipelines. |
| `tests/` | tests | legacy | Only exercises `cls.*` (`GridWMEnv`, `cls.utils.GridUtils.VectorHash`, `cls.encoder`, `cls.hopfield`). Nothing here covers `hopfield_nav` or `encoder_training`. |
| `hopfield_nav/tests/` | tests | live | `test_at_goal.py`, `test_audit.py`, `test_phase_decoding_v2.py`. No runner script and no `conftest.py`/pytest config. Verified 2026-08-05: `~/.conda/envs/cls/bin/python -m pytest hopfield_nav/tests -q` from the repo root gives **135 passed, 1 failed** (see §8 item 4). |
| `notebooks/` | exploration | legacy | Nov 2025 – Apr 2026 exploratory notebooks + two large scripts (`train_dist_encoder.py` 1573 lines, `testing_dist_encoder.py` 1501 lines) that predate `encoder_training/`. |
| `docs/` | docs | live | `coordinate_conventions.md` + these three new files. |
| `run.sh`, `sweep_cosine_width.py`, `launch_jupyter.sh` | scripts | legacy | Root sbatch runner currently points at `sweep_cosine_width.py` (a `cls`-only cosine-width sweep). |
| `checkpoint/`, `checkpoints/`, `checkpoint_rnn/`, `encoders/`, `wandb/`, `plots/`, `images/`, `npos_sweep/`, `displacement_plots/`, `action_classifiers/`, `smoke_pd2/`, `smoke_seq/` | outputs | untracked | ~75 GB total (§6). |

Package installability: `pyproject.toml` declares only `cls` as a package
(`include = ["cls*"]`). `hopfield_nav` and `encoder_training` are **not**
installed — they work because every entry point is run as `python -m ...` from
the repo root, so the CWD is on `sys.path`. This is why every sbatch script
contains `cd /home/jackking/cls`.

---

## 3. `hopfield_nav/` — the agent

### 3.1 Core simulation and model modules

| Module | Lines | Responsibility |
|---|---:|---|
| `config.py` | 368 | All dataclasses: `EnvConfig`, `VectorHashConfig`, `HopfieldConfig`, `AgentConfig`, `PPOConfig`, `BCConfig`, `TrainConfig`, `RNNAgentConfig`, `RNNBCConfig`, `RNNTrainConfig`, and three phase schedules `PhasedConfig` / `PhasedConfigV2` / `PhasedConfigV3`. `validate_train_config()` performs exactly one cross-check (`agent_can_store=False` + `auto_store_warmup>0` → `ValueError`). |
| `env.py` | 429 | `GridEnv` (discrete) and `ContinuousGridEnv` (float positions, snapped for lookup); the canonical `at_goal(env)` predicate (L2 ball of radius `goal_radius` on the *raw* position); the per-cell foveal sensory codebook (120° cone, heading fixed North, one ±1 code per ray from the wall segment it hits); `make_env()` factory. |
| `vec_env.py` | 341 | `VecEnv` / `ContinuousVecEnv`: B parallel episodes sharing one env's codebook and goal. Defines the at-goal step semantics used everywhere (§3.3). |
| `vectorhash.py` | 472 | `VectorHash`: builds `gbook` (+ optional place/sensory layers), places envs as non-overlapping patches in the `Npos × Npos` scaffold, precomputes `encoded_Phi`, and provides `get_encoded_state`, `gram_schmidt_projection`, `project_displacement`, `get_goal_encodings`. |
| `hopfield.py` | 208 | `Hopfield` (Hebbian store, iterative tanh recall) plus batched free functions `recall_per_env_batch` / `recall_per_env_batch_trajectory` for per-env weight matrices. |
| `encoder.py` | 78 | Loads a frozen `encoder_training` checkpoint; tolerates three historical save formats; resolves the effective gain. `validate_config()` only checks that encoder `lambdas` match VectorHash `lambdas`. |
| `agent.py` | 192 | `NavAgent` = GRU trunk + movement head (Categorical(4) or Normal(2)) + Bernoulli store head + value head. `compute_input_dim()` is the single source of truth for the input layout. |
| `rollout.py` | 799 | `RolloutCollector.collect_rollout()` — the heart of training: env stepping, embedding lookup, Hopfield recall, reward shaping, ε-exploration, teacher forcing, BC label generation, GAE bootstrap. |
| `ppo.py` | 302 | `RolloutBatch` dataclass, `compute_gae`, `ppo_update` (clipped surrogate + value + entropies + auxiliary store BCE). |
| `bc.py` | 204 | DAgger novelty oracles and `bc_update` (CE on teacher move + weighted BCE on store). |
| `oracle_bfs.py` | 78 | Shortest-path oracle for the open grid (greedy Manhattan with random tie-break), used by the RNN baseline. |
| `utils.py` | 72 | Gram–Schmidt basis, direction classification/one-hot, re-export of `smooth_g`/`smooth_gbook` from `encoder_training.utils`. |

### 3.2 Training entry points (five of them)

| Entry point | Flags | Algorithm | Save dir |
|---|---:|---|---|
| `hopfield_nav.train` | 77 | Single-phase PPO **or** DAgger BC (`--training_mode`) over `num_worlds × envs_per_world` envs. | `checkpoint/<run>/hopfield_nav_update{N}.pt` |
| `hopfield_nav.train_phased` | 39 | Four sequential phases (store pretrain → follow → explore → compose). Also the **shared helper module** (`setup_world`, `make_hops`, `set_phase_freeze`, `do_eval`) imported by the two entry points below. | `checkpoint/phased_<run>/phased_final.pt` |
| `hopfield_nav.train_phase_a_only` | 73 | The **active** harness. Phase A only: interleaved pre-stored ("follow") and empty ("explore") rollouts with novelty/revisit/wall/persistence shaping, ε-greedy, distractor curricula, log-std annealing. | `checkpoint/phase_a_only_<run>/phase_a_u{N}.pt` |
| `hopfield_nav.train_phase_b_only` | 11 | Loads a Phase-A checkpoint, freezes everything but the store head, trains it with detached-trunk BCE. | `checkpoint/phase_b_only_<run>/phase_b_u{N}.pt` |
| `hopfield_nav.train_rnn` | 37 | No-memory control baseline: GRU policy on raw sensory, BC against the BFS oracle, in `sequential` / `mixed` / `finetune` mode. | `checkpoint_rnn/<run>/final.pt` |

Supporting modules for the baseline: `agent_rnn.py`, `rollout_rnn.py`,
`bc_rnn.py`, `eval_rnn.py`. They deliberately share nothing with the Hopfield
stack except `env.py`/`vec_env.py`.

### 3.3 The one semantic everything depends on: the at-goal step

`VecEnv.step_batch` / `ContinuousVecEnv.step_batch` check `at_goal` on the
**pre-action** position. If the agent is standing on the goal:

1. reward = `goal_reward` (not `-time_penalty`),
2. the movement action is **ignored**,
3. the env teleports to a new random non-goal cell,
4. the caller zeroes that row's RNN hidden state and `prev_*` channels.

So the agent gets exactly one observable timestep at the goal — which is the
step on which it can fire `store` and have `embeddings[b]` be the goal
embedding. Every evaluator in `eval.py` reproduces this by treating "pre-step
`at_goal` is True" as the success/reach event. `goals_active=False` disables
the whole branch (no reward, no teleport) — the pure-explore regime.

### 3.4 Evaluation stack

`eval.py` (1162 lines) is single-env, single-trial, `@torch.no_grad`, built on
one shared step function `agent_step()`. Seven evaluators:

| Function | Question it answers | Hopfield contents |
|---|---|---|
| `evaluate_navigation` | Can the policy follow a recall signal to the goal? | goal + N distractors, preloaded |
| `evaluate_goal_discovery` | On reaching the goal, does it fire `store`? | distractors only, agent writes |
| `evaluate_exploration` | How much of the grid does one rollout cover? | distractors only, stores disabled by default |
| `evaluate_union_coverage` | How *diverse* are N rollouts (union of visited cells)? | distractors only, stores off |
| `evaluate_realistic` | Interference: one persistent Hopfield across envs, retest prior envs with storing disabled | accumulates over the whole eval |
| `evaluate_repeat` | Same as realistic's primary phase, but a fresh Hopfield per trial | per-trial |
| `evaluate_sequential_episodes` | Paper-style continual protocol: block *i* introduces env *i*, each iteration runs one mini-episode in every env ≤ *i* | persistent, revisits frozen |

Drivers: `eval_all.py` (29 flags — the general one, JSON + plots),
`eval_checkpoints.py` (sweeps run × update over a hardcoded `checkpoints/`
root), `eval_distractors.py` (one ckpt × distractor sweep), `eval_rnn.py`
(standalone, for the RNN baseline).

Two oracle switches let you cut the loop at different points
(`eval.agent_step`): `hopfield_oracle` replaces the *recall* with the true
goal-minus-current displacement (same projection), and `action_oracle` replaces
the *movement* with a greedy step toward the goal. Both are gated on
"goal is in memory".

### 3.5 Analysis / figure pipelines

**`final_plotting/`** — the continual-learning figure (Hopfield agent vs RNN
baseline on the same axes):

```
prep_scaffold.py   build encoded_Phi once → content-addressed cache dir
        │                (hash of lambdas, fwhm_ratio, Npos, encoder path)
        ▼
agenthash.py       frozen policy, persistent Hopfield, sequential blocks
   or   baseline.py  RNN baseline (BC-trains as it goes)
        │            each writes one history JSON per iteration seed
        ▼
merge_histories.py  N single-iter histories → one multi-iter history
        ▼
plotting.py        <prefix>_forgetting.{png,pdf}, <prefix>_steps_to_goal.{png,pdf}
```

Both `agenthash.py` and `baseline.py` emit the **same history JSON schema**, so
one plotter renders both. `run_agenthash.sh` / `run_baseline.sh` are the sbatch
drivers (variables at the top of the file); `just_plot.sh` re-renders from an
existing history.

**`phase_decoding_v2/`** — representational analysis of the trained controller's
GRU state:

- Exp 1 (`exp1.py`): collect explore-vs-exploit trials over `num_arenas` arenas,
  then compute **parallelism score** (cosine between the exploit−explore
  centroid vectors of train vs test arenas) and **decodability** (balanced
  accuracy of an L2 logistic regression fit on train arenas, tested on held-out
  arenas) across four split families (LOO, random 80/20, quadrant 1v3, quadrant
  3v1). Bar plot + `metrics.json`.
- Exp 2 (`exp2.py`): PCA of MLP hidden activations — per-trial explore/exploit
  scatter, and a two-episode trajectory PCA where the goal is oracle-stored
  mid-trajectory and the agent teleports *without* resetting `h_rnn` or the
  Hopfield.
- `--random_agent` gives a same-architecture, untrained-weights control.

**`figures/`** — five standalone matplotlib schematic generators (encoder,
decoder, Hopfield energy landscape, memory storage, store mechanism). Pure
illustration; they do not read checkpoints.

**`visualize_trajectories.py`** — a checkpoints × trials grid of actual
trajectories, in `combined` / `explore_only` / `exploit_only` mode.

**`viz_sensory.py`** — visualizes the foveal sensory codebook.

---

## 4. `encoder_training/` — the encoder

| Module | Responsibility |
|---|---|
| `config.py` | `EncoderModelConfig`, `LossConfig`, `PatchConfig`, `NavEvalConfig`, `TrainConfig`. |
| `data.py` | Full-grid code generation (`build_full_grid`), non-overlapping patch sampling, patch extraction, mixed vs single-env batch iterators. |
| `models.py` | `GridEncoder` (MLP) and `GridEncoderCNN`; both end with `tanh(gain · z)` then L2-normalize, so outputs live on the unit sphere. `create_encoder()` factory. |
| `losses.py` | `mse_attract_repel` (the "binary method": near pairs → cos 1, far pairs → cos 0), plus `cka_loss` and `uniformity_loss` for ablation. |
| `train.py` | Training loop with per-epoch gain annealing (`gain_start → gain_end`), uniformity ramp, periodic nav-eval, `encoder_best.pt` / `encoder_final.pt`. |
| `evaluate.py` | Thin wrapper that delegates to `cls.eval.nav_eval`. |
| `evaluate_nav.py` | Standalone nav eval of a saved encoder checkpoint (16 flags, optional JSON line for sweeps). |
| `sweep.py` | Cartesian-product SLURM sweep driver — **edit `BASE`/`GRID`/`EVAL`/`SLURM` dicts in the file**, then `python -m encoder_training.sweep [name]`. |
| `plot_sweep.py` | Aggregates `meta.json` + `result.json` per run → bar chart, CSV, 1-D curve or 2-D heatmap. |
| `save_untrained_encoder.py` | Writes a randomly-initialized encoder in the same checkpoint format (the untrained-encoder control). |
| `trajectory.py`, `viz.py` | Exploratory single-trajectory rollouts and similarity/unique-radius plots. |
| `experiments/encoder_scaffold.py` | Replaces VectorHash's random-projection place layer with a tap into the trained encoder (`p = encoder(g)` at hidden layer *k*), pseudo-inverse `Wgp`/`Wsp`/`Wps`, and compares grid recovery + observation bit-error against the canonical scaffold under observation noise. |
| `experiments/capacity_scaling.py` | Sweeps the number of stored patterns for encoder-tap vs random-projection scaffolds; plots accuracy vs flip probability. |

The nav-eval that gates `encoder_best.pt` is **not** the agent eval — it is a
policy-free simulation (`cls/nav.py:simulate_trajectory`) in which the agent
always steps along the projected Hopfield recall direction. An encoder is
"good" if that open-loop chase reaches the goal from a lattice of starts.

---

## 5. `cls/` — legacy library, narrow live surface

| Module | Lines | Still used? |
|---|---:|---|
| `cls/vectorhash/assoc_utils_np.py` | 957 | **Yes** — `nonlin`, `train_pbook`, `train_gcpc`, `pseudotrain_Wsp`, `pseudotrain_Wps` (imported by `hopfield_nav/vectorhash.py`, `encoder_training/experiments/encoder_scaffold.py`). |
| `cls/vectorhash/assoc_utils_np_2D.py` | 227 | **Yes** — `gen_gbook_2d` (imported by `hopfield_nav/vectorhash.py`, `encoder_training/data.py`, `encoder_scaffold.py`). |
| `cls/eval/nav_eval.py` | 409 | **Yes** — the encoder nav-eval (`encoder_training/evaluate.py`). |
| `cls/nav.py` | 137 | **Yes** — projection + trajectory primitives used by `nav_eval.py` and `encoder_training/trajectory.py`. |
| `cls/hopfield.py` | 182 | **Yes** (encoder side only). Note: its `recall()` returns `(x, cos_sims)`, whereas `hopfield_nav/hopfield.py`'s returns a bare tensor — two incompatible Hopfield classes coexist. |
| `cls/vectorhash/{seq,sensory,sensgrid,sens_pcrec,sens_sparseproj,senstranspose,theory,capacity,data}_utils.py`, `assoc_utils.py` | ~3200 | No live importer outside `cls` itself and the root `tests/`. |
| `cls/utils/GridUtils.py` | 767 | Only `cls/envs/environments.py` and root `tests/`. |
| `cls/envs/environments.py` | 978 | Only root `tests/`. (`WMEnv` is the package's public API in `cls/__init__.py`.) |
| `cls/models.py`, `cls/encoder.py`, `cls/types.py` | 543 | `cls/encoder.py` + `cls/hopfield.py` used by root `tests/`; `models.py` has no live importer. |

So: about 1,700 of ~5,900 lines in `cls/` are on the live path, and they are
almost all pure functions (`gen_gbook_2d` and the pseudo-inverse trainers) plus
the encoder nav-eval.

---

## 6. Outputs and disk

| Directory | Size | Written by |
|---|---:|---|
| `encoders/` | 21 GB (870 run dirs) | `encoder_training.train` (`--save_dir`) |
| `hopfield_nav/final_plotting/` | 21 GB | histories, scaffold cache, figures |
| `hopfield_nav/phase_decoding_v2/results/` | 17 GB | per-arena `.npz` trial dumps |
| `checkpoint/` | 8.5 GB (309 run dirs) | `train.py`, `train_phase_a_only.py`, `train_phase_b_only.py`, `train_phased.py` |
| `wandb/` | 6.6 GB (817 runs) | wandb local cache |
| `hopfield_nav/phase_decoding` | 1.1 GB | the **v1** phase-decoding pipeline — present on disk, not in git |
| `checkpoints/` (plural) | 63 MB | older `train.py` runs; `eval_checkpoints.py` still hardcodes this root |
| `checkpoint_rnn/` | 13 MB | `train_rnn.py` |
| others (`plots/`, `images/`, `npos_sweep/`, `smoke_*`, `action_classifiers/`, `displacement_plots/`) | ~50 MB | ad-hoc |

`.gitignore` excludes all of it (plus `*.png`, `*.pdf`, `*.json`, `*.log`,
`*.out`) — which is why figure outputs and result JSONs never enter git, and
also why `hopfield_nav/phase_decoding/` (v1) exists on disk with no git history.

Run naming: with `--use_wandb` the save directory is the wandb run name
(`driven-snowflake-111`, `phase_a_only_lively-surf-104`, …); without it, a
timestamp. The wandb run name is therefore the primary key linking a
checkpoint, a wandb run, and a slurm log.

---

## 7. Experiment families (what the four live tracks are)

**A. Encoder training + sweeps** — `encoder_training/`.
Question: what embedding geometry makes Hopfield recall + local projection point
in the right direction? Knobs that define the experiment: `lambdas`,
`out_dim`, patch layout (`nenv`/`npos`/`npos_list`), the near-radius
(`per_env_radius_frac` or fixed `radius`), `attract_lambda`/`repel_weight`, and
gain annealing. Metric: nav-eval accuracy on **val** patches (envs placed
outside every training patch). Driven by `scripts/submit_train.sh` for single
runs and `sweep.py` for grids.

**B. Phase-A PPO explore/follow** — `train_phase_a_only.py` +
`run_phase_a_sweep_evelina.sh`.
Question: can one policy both explore a novel arena and follow a Hopfield recall
signal to a stored goal? The sbatch script is a 785-line `case` statement with
**101 named variants** (`v6a`, `v6b`, `v7`, `baseline`, `v8`…`v18d42`,
`v18d39_size20_v28`…): each variant is one `EXTRA` string of flags appended
after a fixed base invocation, so later flags override earlier ones. `VARIANT`
and `SEED` are environment variables. This file *is* the experiment registry —
there is no separate config store.

**C. Paper figure pipelines** — `final_plotting/` and `phase_decoding_v2/`.
Question 1: does the Hopfield agent avoid catastrophic forgetting where the RNN
baseline does not (continual sequential protocol)? Question 2: is there an
abstract, arena-general "explore vs exploit" axis in the controller's hidden
state (parallelism + decodability + PCA)?

**D. BC/DAgger and RNN baselines** — `train.py --training_mode bc`,
`train_rnn.py`, `bc.py`, `bc_rnn.py`, `eval_rnn.py`, `pretrain_baseline_rnn.sh`.
Question: how much of the behavior is reachable by supervised imitation of an
oracle, and how does a memoryless GRU do on the same continual protocol?

---

## 8. Known drift and landmines

Each of these was verified against the current source.

1. **Three sbatch scripts are broken by a flag rename.** `--gbook-only` was
   renamed to `--static-vectorhash`, but the shim was only added for *checkpoint
   config dicts* (`_coerce_legacy_cfg`), not for the CLI. argparse exits 2 on
   the unknown flag, so these fail immediately:
   - [run_eval_all.sh:134](../hopfield_nav/run_eval_all.sh#L134) (`GBOOK_ONLY=1`
     by default) → `eval_all.py`, whose parser has only `--static-vectorhash`
     ([eval_all.py:850](../hopfield_nav/eval_all.py#L850));
   - [run_continuous.sh:31](../hopfield_nav/run_continuous.sh#L31) → `train.py`
     ([train.py:480](../hopfield_nav/train.py#L480));
   - [run_new_sweep.sh:29](../hopfield_nav/run_new_sweep.sh#L29) → `train.py`.
2. **`--goal_radius` does nothing in `train.py`'s training envs.**
   `setup_train_world` constructs `GridEnv(...)` directly and passes only
   `size, speed, observation_size, seed, time_penalty`
   ([train.py:55-59](../hopfield_nav/train.py#L55)), so `goals_active`,
   `goal_reward` and `goal_radius` fall back to the `GridEnv` defaults. The eval
   world *does* honor them (it goes through `make_env(cfg.env, ...)`). The same
   applies to `train_rnn.py`'s `build_envs()`, which passes `goals_active` but
   not `goal_radius`. `train_phased.py` / `train_phase_a_only.py` /
   `train_phase_b_only.py` use `setup_world` → `make_env` and are unaffected.
3. **`GridEnv.clone()` cannot run.** It calls `self._random_position_with_rng`
   ([env.py:313](../hopfield_nav/env.py#L313)), which is a `@staticmethod` that
   unconditionally raises `NotImplementedError`
   ([env.py:324-331](../hopfield_nav/env.py#L324)). No live caller exists —
   `train.py:253` clones a *Hopfield*, not an env — so this is dead code, not an
   active bug. (`_random_position` is also defined twice in the class; the
   second definition wins.)
4. **One failing test, and the reason is a latent fragility.**
   `hopfield_nav/tests/test_audit.py::TestBCStoreCap::test_cap_active` fails
   with `TypeError: expected Tensor as element 0 in argument 0, but got
   NoneType` at [bc.py:127](../hopfield_nav/bc.py#L127). `bc_update`
   unconditionally does `torch.cat([r.trust_hop_mask for r in rollouts])`, but
   `RolloutBatch.trust_hop_mask` defaults to `None`
   ([ppo.py:41](../hopfield_nav/ppo.py#L41)). The live path is safe —
   `RolloutCollector` always populates it when `training_mode == "bc"` — but any
   hand-built batch, or a PPO-mode rollout handed to `bc_update`, crashes. The
   test predates the `trust_hop_mask` / `nav_weight` feature and was never
   updated. Fix is one line (`torch.ones_like(mm)` when the field is `None`),
   plus the test.
   Also: **`phase_decoding_v2/README.md` points at `hopfield_nav/run_tests.sh`**,
   which does not exist, and there is no `conftest.py` or pytest config, so the
   suite must be run as `python -m pytest` from the repo root.
5. **`eval_checkpoints.py` and `eval_distractors.py` hardcode absolute
   paths** to `/home/jackking/cls/checkpoints/...` and a 2026-04-09 encoder.
   `eval_checkpoints.py`'s default `--runs` list (`r2_r2a_low_ent`, …) refers to
   runs from that era.
6. **The eval-world reconstruction is not bit-identical to training.**
   `build_eval_world` replays the val-env seed sequence
   (`RandomState(cfg.seed)`, burning `envs_per_world × num_worlds` draws first),
   so codebooks and goals match — but `eval_checkpoint` calls
   `np.random.seed(0)` first and `VectorHash.register_envs("spread")` draws its
   jitter from the *global* `np.random`, so the scaffold offsets generally
   differ from the training run's. Offsets are arbitrary, so this is not a
   correctness problem; it does mean "same checkpoint, same flags" can differ
   from the number logged during training.
7. **Three near-identical copies of the checkpoint-loading helpers** exist:
   `_coerce_legacy_cfg` + `make_cfg_from_checkpoint` + `build_eval_world` in
   `eval_all.py`, `eval_checkpoints.py` and `eval_distractors.py`.
   `train_phase_b_only.py` imports `make_cfg_from_checkpoint` from
   `eval_checkpoints`, while `final_plotting/agenthash.py` and
   `phase_decoding_v2/rollout.py` import their copy from `eval_all`.
8. **Unused config surface.** `PhasedConfigV2` is defined in `config.py` and
   imported nowhere. `HopfieldConfig.n_train_distractors` (the fixed-count
   variant) is read only by `train.py`; the Phase-A harness uses the
   `_min`/`_max` pair instead. `HopfieldConfig.init_mode` is *read* in exactly
   one place (`train.py:69`, to decide whether to build a pre-stored template);
   `train_phased.py` sets and restores it around each phase, but nothing
   downstream consumes it — the actual Hopfield contents come from
   `make_hops(role)`.
9. **Two `load_encoder` functions** with different return signatures:
    `hopfield_nav/encoder.py:load_encoder` → `(encoder, cfg, gain)` and
    `encoder_training/train.py:load_encoder` → `(encoder, ckpt_dict)`.
10. **SLURM partitions are inconsistent**: `pi_evelina9` in most scripts,
    `mit_normal_gpu` in `run_continuous.sh` and `pretrain_baseline_rnn.sh`,
    `mit_normal` in `just_plot.sh`, `pi_fiete` in `encoder_training/sweep.py`.
11. **A wandb API key is committed in plaintext** in every sbatch script
    (`export WANDB_API_KEY=...`). Worth rotating and moving to `~/.netrc` or a
    non-tracked env file.
