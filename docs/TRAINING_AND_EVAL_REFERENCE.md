# Training and Evaluation Reference

> **Renamed 2026-08-06.** `train_phase_a_only` is now **`train_navigate`** and
> `train_phase_b_only` is now **`train_store`**. Run directories, checkpoint
> filenames and eval tags follow: `agent_ckpts/navigate_<run>/navigate_u{N}.pt`
> and `navigate_final.pt`, `agent_ckpts/store_<run>/store_u{N}.pt` and
> `store_final.pt`. `train_store` keeps `--phase_b_updates` / `--phase_b_lr`.
> The ~150 pre-rename run directories are untouched and still readable;
> `RUN_KINDS` keeps their prefixes so `backfill_manifests` can parse them.
>
> **`train_navigate` took a schedule, 2026-08-06.** Its `--phase_a_*` and
> `--interleave_*` flags are **gone**, replaced by `--schedule` (§4.2).
> `--goals_active` went too, and that one was never doing anything: every
> rollout assigns `env.goals_active` from its regime, so a run-wide value was
> overwritten before the first step. `--explore_goals_off` is the live knob.
> The
> 101 variants in `run_phase_a_sweep_evelina.sh` will not run as written; that
> script is now a record of a completed sweep, not a launcher. Use
> `run_navigate.sh` / `run_explore.sh` / `run_exploit.sh` instead.


Written 2026-08-05 from the code on `main` (`959322f`). Every flag table below
was read off the `argparse` blocks in the source, and every "what it does" entry
was traced to the line that consumes the value. Nothing here is inherited from
the older markdown files (now under `docs/archive/`).

> **Updated 2026-08-06 for the phase-6 layout.** Module paths were rewritten
> mechanically; see the banner in `CODEBASE_MAP.md` for the full mapping. Flag
> names and semantics are unchanged, and every entry point is still invoked the
> same way — except the analysis pipelines, which moved out of `hopfield_nav`:
> `python -m hopfield_nav.final_plotting.X` is now `python -m analysis.continual.X`,
> and `hopfield_nav.phase_decoding_v2.X` is now `analysis.phase_decoding.X`.

Contents:
1. [How anything gets run](#1-how-anything-gets-run)
2. [The pipeline, end to end](#2-the-pipeline-end-to-end)
3. [Stage 1 — encoder training](#3-stage-1--encoder-training)
4. [Stage 2 — agent training](#4-stage-2--agent-training)
5. [Stage 3 — evaluation](#5-stage-3--evaluation)
6. [Stage 4 — figure and analysis pipelines](#6-stage-4--figure-and-analysis-pipelines)
7. [The knobs that actually matter](#7-the-knobs-that-actually-matter)
8. [Metric definitions](#8-metric-definitions)
9. [Checkpoint formats and compatibility](#9-checkpoint-formats-and-compatibility)

---

## 1. How anything gets run

Every entry point is a module, run from the **repository root**:

```bash
cd /home/jackking/cls && python -m hopfield_nav.train_navigate --encoder_checkpoint ...
```

**Updated 2026-08-06:** run *outputs* no longer depend on the CWD. All five
trainers resolve their default `save_dir` through `cls_paths.run_dir()`, so
`CLS_RUNS` relocates agent checkpoints along with everything else, and starting
a job outside the repo root no longer creates a stray `./checkpoint/`. The `cd`
is still required for *imports*.

The root must be the CWD because `hopfield_nav` and `encoder_training` are not
installed packages — `pyproject.toml` installs only `cls`. All sbatch scripts
therefore contain `cd /home/jackking/cls` (note: `/home/jackking/cls` and
`/orcd/home/002/jackking/cls` are the same tree; `encoder_training/scripts/*`
use the `/orcd/...` form).

Common preamble in every sbatch script:

```bash
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
export WANDB_API_KEY=<literal key in the file>
unset CUDA_VISIBLE_DEVICES
```

### Launchers

| Script | Launches | Partition / time | How you configure it |
|---|---|---|---|
| `hopfield_nav/run_navigate.sh` | `hopfield_nav.train_navigate` | `pi_evelina9`, 3 d, 1 GPU, 100 G | `SCHEDULE='…' SEED=<n> sbatch ...` — explore → interleave → exploit by default |
| `hopfield_nav/run_explore.sh` | `hopfield_nav.train_navigate` | `pi_evelina9`, 1 d, 1 GPU, 100 G | same env vars; defaults to `explore:600` |
| `hopfield_nav/run_exploit.sh` | `hopfield_nav.train_navigate` | `pi_evelina9`, 1 d, 1 GPU, 100 G | same env vars; defaults to `exploit:300`. `LOAD_CKPT=…` to continue a run |
| `hopfield_nav/navigate_job.sh` | *(sourced, not submitted)* | — | the shared body of the three above; every env var is documented in its header |
| `hopfield_nav/run_phase_a_sweep_evelina.sh` | `hopfield_nav.train_navigate` | `pi_evelina9`, 7 d, 1 GPU, 100 G | `VARIANT=<name> SEED=<n> sbatch ...` — 101 named variants in a `case` block. **No longer runs**: passes the removed `--phase_a_*` / `--interleave_*` flags. Kept as the record of the sweep |
| `hopfield_nav/run_continuous.sh` | `hopfield_nav.train` | `mit_normal_gpu`, 2 h | edit flags in file — **currently broken**, passes `--gbook-only` |
| `hopfield_nav/run_new_sweep.sh` | `hopfield_nav.train` ×4 | `pi_evelina9`, 2 d | edit `COMMON` in file — same `--gbook-only` breakage |
| `hopfield_nav/pretrain_baseline_rnn.sh` | `hopfield_nav.train_rnn --mode mixed` | `mit_normal_gpu`, 2 h | edit flags in file |
| `hopfield_nav/run_eval_all.sh` | `hopfield_nav.eval_all` per ckpt | `pi_evelina9`, 12 h | `CKPTS` array + env vars — **currently broken**, passes `--gbook-only` |
| `analysis/run_trajectories.sh` | `analysis.trajectories` | `pi_evelina9`, 30 m | edit variables at top |
| `analysis/continual/run_agenthash.sh` | `prep_scaffold` → N× `agenthash` → `merge_histories` → `plotting` | `pi_evelina9`, 6 h, 32 CPU | edit variables at top |
| `analysis/continual/run_baseline.sh` | N× `baseline` → `merge_histories` → `plotting` | `pi_evelina9`, 12 h, 32 CPU | edit variables at top |
| `analysis/continual/just_plot.sh` | `plotting` only | `mit_normal`, 2 h | edit variables |
| `analysis/phase_decoding/run_exp1.sh` | `phase_decoding_v2.exp1` | `pi_evelina9`, 1 d | env vars: `CKPT`, `NUM_ARENAS`, `N_STARTS`, `MAX_STEPS`, `N_DIST_MIN/MAX`, `N_RAND`, `TEST_FRAC`, `SEED`, `POLICY_FLAG`, `OUT` |
| `analysis/phase_decoding/run_exp2.sh` | `phase_decoding_v2.exp2` | `pi_evelina9`, 2 h | env vars incl. `TRIALS_DIR` to reuse exp1 trials |
| `encoder_training/scripts/submit_train.sh` | `encoder_training.train` | `pi_evelina9`, 4 h, A100 | variables at top; empty string ⇒ fall back to `train.py` default |
| `encoder_training/scripts/submit_eval.sh` | `encoder_training.evaluate_nav` | `pi_evelina9`, 30 m, A100 | variables at top |
| `encoder_training/sweep.py` | N× (train → evaluate_nav) | `pi_fiete`, 12 h | edit `BASE`/`GRID`/`EVAL`/`SLURM` dicts, then run the module (it calls `sbatch` itself) |
| `analysis/scaffold_experiments/run_exp.sh` | `capacity_scaling` ×2 | `pi_evelina9`, 1.5 h | fixed |
| `run.sh` (repo root) | `encoder_training.sweep_cosine_width` | `pi_evelina9`, 2 d, CPU | legacy `cls`-only sweep |

### Logs and run identity

- slurm stdout → `hopfield_nav/logs/slurm_*_%j.out`, `encoder_training/scripts/logs/`,
  `encoder_training/logs/<run_name>.log`.
- With `--use_wandb`, the **wandb run name** becomes the checkpoint directory
  name (`$CLS_RUNS/agent_ckpts/navigate_<name>/`). Without it, a `YYYYmmdd_HHMMSS`
  timestamp. That name is the only key linking wandb ↔ checkpoint ↔ slurm log.

---

## 2. The pipeline, end to end

```
 (1) encoder_training.train                 → encoders/<run>/encoder_best.pt
        │  contrastive embedding of grid codes; nav-eval gates "best"
        ▼
 (2) hopfield_nav.train_navigate        → $CLS_RUNS/agent_ckpts/navigate_<run>/navigate_u{N}.pt
     (or .train / .train_phased / .train_store / .train_rnn)
        │  frozen encoder + VectorHash scaffold + Hopfield + recurrent policy, PPO or BC
        ▼
 (3) hopfield_nav.eval_all                  → JSON + PNG per checkpoint
     analysis.trajectories    → trajectory grids
        ▼
 (4) final_plotting.{prep_scaffold,agenthash,baseline,merge_histories,plotting}
     phase_decoding_v2.{exp1,exp2,plot}     → paper figures
```

Stage 2 **requires** a stage-1 checkpoint (`--encoder_checkpoint`) and never
updates it. Stages 3–4 read a stage-2 checkpoint and rebuild the encoder from
the path saved inside it (overridable).

---

## 3. Stage 1 — encoder training

### 3.1 `python -m encoder_training.train`

What it does, in order (`encoder_training/train.py:79-220`):

1. Seeds torch/numpy with `cfg.seed`; creates `save_dir/run_name/`
   (`run_name` empty ⇒ `enc_<timestamp>`).
2. `build_full_grid(lambdas, fwhm_ratio)` → `Phi_full` of shape
   `(Σλ², Πλ, Πλ)`: `gen_gbook_2d` one-hots, then a wrapped Gaussian bump per
   module with `σ = λ·fwhm_ratio / 2√(2 ln 2)`.
3. `sample_nonoverlapping_patches` → `nenv` square patches of side `npos`
   (or one patch per entry of `npos_list`). These are the *training
   environments*: rejection-sampled, non-overlapping.
4. `extract_patches` flattens them into `Phi_flat (N, Σλ²)`, `coords`, `env_ids`
   on the device. `--shuffle` permutes `Phi_flat` across positions (ablation:
   destroys the position↔code correspondence).
5. Per-env "near" radius = `per_env_radius_frac × patch_size` when
   `per_env_radius_frac > 0`, else the scalar `--radius`; radius ≤ 0 means
   "every same-env pair is near".
6. Training loop over `epochs`:
   - `gain = linspace(gain_start, gain_end, epochs)[ep-1]` — annealed **every**
     epoch and passed into `encoder(x, gain)`, so the output nonlinearity is
     `tanh(gain·z)` with a growing gain.
   - uniformity weight ramps `0 → uniformity_lambda` over
     `uniformity_anneal_epochs`, then holds.
   - batches: `--single_env_batch` ⇒ exactly one batch per env per epoch (so
     "far" pairs are always same-env, distant positions); otherwise random
     batches over all points.
   - `z = encoder(Phi_flat[idx], gain)`; `K = clamp(z zᵀ, -1, 1)`;
     `loss = attract_lambda·mean((K[near]−1)²) + repel_weight·mean(K[far]²)`
     (or `1 − CKA` when `--loss_mode cka`), `+ unif_λ·uniformity_loss(z)`.
   - AdamW(`lr`, `weight_decay=1e-4`), grad-clip 1.0 (neither is exposed on the CLI).
7. Every `eval_every` epochs: encode the **whole** `Πλ × Πλ` grid and run the
   nav eval on **val** placements; if accuracy improves, write `encoder_best.pt`.
8. After the last epoch, write `encoder_final.pt`.

| Flag | Default | Effect |
|---|---|---|
| `--encoder_type` | `mlp` | `mlp` → `GridEncoder`, `cnn` → `GridEncoderCNN`. |
| `--lambdas` | `11 12 13` | Grid module periods. Sets `in_dim = Σλ²` and grid side `Npos = Πλ` (11·12·13 = 1716 ⇒ a 1716² grid). **Must match the VectorHash lambdas used later** or `hopfield_nav.encoder.validate_config` raises. |
| `--out_dim` | `256` | Embedding dim `D`; becomes the Hopfield size (`D×D` weight matrix per env) downstream. |
| `--hidden_dim` | `1024` | MLP width (also the CNN head width). |
| `--num_hidden_layers` | `4` | MLP depth. |
| `--hidden_channels` | `128` | CNN only. |
| `--num_conv_layers` | `3` | CNN only. |
| `--kernel_size` | `5` | CNN only. |
| `--nenv` | `25` | Number of training patches (ignored if `--npos_list`). |
| `--npos` | `100` | Side length of each patch (ignored if `--npos_list`). |
| `--npos_list` | `None` | Comma-separated sizes, e.g. `40,60,80`; overrides `--nenv/--npos`, one patch per entry. |
| `--per_env_radius_frac` | `0.0` (CLI) / `0.1` (dataclass) | Near-radius as a fraction of patch size. >0 takes precedence over `--radius`. |
| `--radius` | `10.0` | Fixed near-radius in cells; used only when `per_env_radius_frac ≤ 0`. |
| `--single_env_batch` | off | One batch per env per epoch. With it on there are no cross-env pairs at all. |
| `--loss_mode` | `mse_contrastive` | `mse_contrastive` (attract/repel) or `cka`. |
| `--attract_lambda` | `2.0` | Weight on near-pair MSE (also the CKA weight in `cka` mode). |
| `--repel_weight` | `5.0` | Weight on far-pair MSE. `mse_contrastive` only. |
| `--uniformity_lambda` | `0.0` | End value of the uniformity regularizer. |
| `--uniformity_anneal_epochs` | `25` | Epochs to ramp uniformity from 0. |
| `--epochs` | `600` | Also the length of the gain-annealing schedule. |
| `--lr` | `2.48e-4` | AdamW lr. |
| `--batch_size` | `4096` | Clamped to the total number of points at config build time. |
| `--seed` | `42` | torch + numpy + nav-eval RNG. |
| `--device` | `cuda` | Falls back to CPU if CUDA is unavailable. |
| `--fwhm_ratio` | `0.25` | Gaussian smoothing width of grid codes, as a fraction of λ. **Must match the value used in stage 2**, since stage 2 re-smooths with its own `--fwhm_ratio` before encoding. |
| `--gain_start` | `1.0` | Gain at epoch 1. |
| `--gain_end` | `5.0` | Gain at the final epoch; also stored as `model_config.gain` and as the checkpoint's `gain`, which becomes the default Hopfield `beta` in stage 2. |
| `--shuffle` | off | Ablation: permute grid codes across positions. |
| `--nav_env_size` | `20` | Side of each nav-eval environment. |
| `--nav_n_train` | `5` | Train-split eval envs *per Hopfield*. |
| `--nav_n_val` | `5` | Val-split eval envs *per Hopfield*. |
| `--nav_num_hopfields` | `20` | Number of separate Hopfields; total placements = `num_hopfields × n_{train,val}_envs`. Each Hopfield stores the goals of its own chunk, so this controls memory load per network. |
| `--nav_n_starts` | `100` | Target starts per env; actual starts are a `⌈√n⌉²` lattice minus the goal cell. |
| `--save_dir` | `/home/jackking/cls/encoders` | Parent of the run dir. |
| `--run_name` | `""` | Run dir name; empty ⇒ `enc_<timestamp>`. |
| `--eval_every` | `50` | Epochs between nav evals; `0` disables (then `encoder_best.pt` is never written). |

Not exposed on the CLI but present in `TrainConfig`: `weight_decay` (1e-4),
`grad_clip` (1.0), `nonlinearity`/`output_nonlinearity` (`gelu`/`tanh`),
`loss.centered`, and all `NavEvalConfig` fields except the five `nav_*` above
(`platform_radius=1.0`, `max_steps_mult=3`, `scale=1.0`, `normalize=True`,
`recompute_interval=1`, `hopfield_alpha=0.8`).

Artifacts: `encoders/<run_name>/encoder_best.pt`, `encoder_final.pt`. Contents:
`state_dict`, `model_config`, `train_config`, `y0s/x0s/sizes` (the training
patches), `gain`, and for `best` also `val_nav_acc` + `epoch`.

### 3.2 `python -m encoder_training.evaluate_nav`

Standalone nav eval of a saved encoder. Rebuilds the grid from the checkpoint's
`lambdas` and `train_config.fwhm_ratio`, encodes it, then runs `split="val"`
(and `split="train"` with `--train_eval`).

| Flag | Default | Effect |
|---|---|---|
| `--ckpt` | required | `encoder_{best,final}.pt`. |
| `--device` | `cuda` | |
| `--train_eval` | off | Also evaluate on placements *inside* the training patches. |
| `--env_size`, `--n_train_envs`, `--n_val_envs`, `--num_hopfields`, `--n_starts_per_env`, `--platform_radius`, `--max_steps_mult`, `--scale`, `--normalize` (0/1), `--recompute_interval`, `--hopfield_alpha` | from ckpt's `train_config.nav_eval`, else `NavEvalConfig` defaults | Override individual nav-eval settings. |
| `--seed` | `42` | Val RNG; train split uses `seed+1`. |
| `--json` | off | Print one `JSON: {...}` line — this is what `sweep.py` greps into `result.json`. |

### 3.3 `python -m encoder_training.sweep [name]`

Reads the module-level `BASE`, `GRID`, `EVAL`, `SLURM` dicts (edit the file),
validates that every `GRID` key exists in `BASE`, then submits one sbatch job per
Cartesian-product point. Each job trains, then evaluates with `--json`, greps
the JSON line into `result.json`, and writes `meta.json`. Output tree:
`encoder_training/sweeps/<name>/<NNN_key=value>/`. Plot with
`python -m analysis.encoder_sweep <sweep_dir>` (bar chart + CSV; a 1-D
curve for one swept key, a heatmap for two).

---

## 4. Stage 2 — agent training

All four Hopfield entry points share the same startup sequence:

1. `load_encoder(cfg.encoder_checkpoint, device, cfg.encoder_gain)` → frozen
   encoder, its `EncoderModelConfig`, and the effective gain (explicit override
   → checkpoint `gain` → `model_config.gain`).
2. `validate_config` — raises unless encoder `lambdas == vectorhash.lambdas`.
3. `cfg.hopfield.beta = encoder_gain` **if `--hopfield_beta` was not given**.
   This couples recall sharpness to the encoder's output gain.
4. Build worlds: envs + `VectorHash` + `build_scaffold()` +
   `register_envs(placement="spread")` + `precompute_encoded_phi()`. The last
   step encodes all `Npos²` positions — with `lambdas 11 12 13` that is
   1716² ≈ 2.9 M positions × `out_dim` floats, which is why these jobs ask for
   64–100 GB. `--Np`/`--Npos` and `--static-vectorhash` are the main levers on
   this cost.
5. Build `NavAgent` with `input_dim = compute_input_dim(cfg.agent, embed_dim,
   observation_size)`.

### 4.1 `python -m hopfield_nav.train` — single-phase PPO or BC

Per update (`train.py:198-407`):

1. `aux_scale = max(0, 1 − (u−1)/aux_anneal_updates)` (1.0 if disabled) —
   multiplies `store_bonus` and `store_bc_weight`.
2. `epsilon_now = epsilon_explore × max(0, 1 − (u−1)/epsilon_anneal_updates)`.
3. If `--refresh_envs_each_update`: every train env draws a new goal and the
   scaffold re-places all envs (`placement="random"`). Requires
   `--static-vectorhash`, else `RuntimeError`. **Superseded** by the per-trait
   refresh below, and refused alongside it: random re-placement over the whole
   scaffold can drop a train env on the validation region, and it never updates
   `split.train`, so `world.json` would describe envs the run had stopped using.
4. For each (world, env): build the Hopfield set —
   - `agent_can_store=True` → a list of `batch_envs` Hopfields (clones of the
     pre-stored template if `--hopfield_init pre_stored`, else empty). Only in
     this case does the rollout write memories at all.
   - `agent_can_store=False` → a *single shared* Hopfield; the rollout skips all
     writes.
   Then optionally preload distractor patterns sampled from outside the env's
   own region (`n_train_distractors`, or `Uniform[min, max]` when
   `n_train_distractors_max > 0`).
5. `collect_rollout(...)` for `steps_per_rollout` steps × `batch_envs` parallel
   episodes (details in §4.5).
6. One pooled `ppo_update` (or `bc_update`) over every rollout collected this
   update.
7. Logging: `mean_reward`, `goal_traj_rate` (fraction of trajectories that ever
   sat on the goal), `goal_step_rate` (at-goal steps / total steps).
8. Every `eval_every`: `eval_max = max(200, 5·size²)`, then `nav_det`,
   `nav_stoch`, `discovery`, `exploration`, and `union` (10 trials,
   `eval_max/2` steps).
9. Every `save_every`: `$CLS_RUNS/agent_ckpts/<run>/hopfield_nav_update{u}.pt`,
   carrying `world_spec`. A refreshing run rewrites `world.json` first, so the
   checkpoint names the file as it stands.
10. After training, `evaluate_realistic` if `realistic_steps_per_env > 0`.

<details>
<summary><b>All 77 flags</b></summary>

**Encoder / scaffold**

| Flag | Default | Effect |
|---|---|---|
| `--encoder_checkpoint` | required | Path to a stage-1 `.pt`. |
| `--encoder_gain` | `None` | Override the checkpoint's gain (also becomes Hopfield β unless `--hopfield_beta`). |
| `--fwhm_ratio` | `0.25` | Smoothing applied to `gbook` before encoding. |
| `--lambdas` | `11 12` | VectorHash modules; must equal the encoder's. |
| `--Np` | `1600` | Place-cell count. Ignored under `--static-vectorhash`. |
| `--Npos` | `None` | Override scaffold side length (default `Πλ`). Main memory lever. |
| `--static-vectorhash` / `--no-static-vectorhash` | `False` | Build only `gbook` + `encoded_Phi`; skip `pbook`, `Wgp`, `Wsp/Wps` and the ≥95 %-grid-recovery self-test. Required by `--refresh_envs_each_update`. |

**Environment**

| Flag | Default | Effect |
|---|---|---|
| `--size` | `8` | Grid side. |
| `--observation_size` | `60` | Number of foveal rays = sensory vector width. |
| `--time_penalty` | `0.01` | Per-step reward `−time_penalty` when not at goal. |
| `--movement_mode` | `discrete` | `discrete` → Categorical(4) + `VecEnv`; `continuous` → Normal(2) + `ContinuousVecEnv`. Sets both `EnvConfig` and `AgentConfig`. |
| `--goal_radius` | `0.5` | At-goal L2 threshold. **Only affects eval envs** — training envs in this script don't receive it (see CODEBASE_MAP §8 item 2). |

**Hopfield / memory**

| Flag | Default | Effect |
|---|---|---|
| `--hopfield_beta` | `None` | Recall temperature; `None` ⇒ encoder gain. |
| `--hopfield_alpha` | `1.0` | Recall mixing: `x ← (1−α)x + α·tanh(βWx)`. |
| `--hopfield_steps` | `1` | Recall iterations per query. |
| `--hopfield_init` | `empty` | `pre_stored` builds a template Hopfield holding *all* train-env goals, cloned per trajectory. |
| `--agent_can_store` / `--no-agent_can_store` | `True` | False ⇒ one shared Hopfield and **no writes at all** during rollouts. |
| `--store_cost` | `0.0` | Reward penalty per store action (explore phase only). |
| `--store_bonus` | `0.0` | Reward bonus for storing while at goal (scaled by `aux_scale`). |
| `--store_bc_weight` | `0.0` | Weight of the auxiliary BCE(store_logits, at_goal) term in PPO (scaled by `aux_scale`). |
| `--auto_store_warmup` | `0` | For the first N updates, force a store whenever at goal. Requires `agent_can_store=True` (`validate_train_config` raises otherwise). |
| `--auto_nav_warmup` | `0` | For the first N updates, overwrite the movement action with the Hopfield-suggested direction for every env that has memory (teacher forcing; those steps are excluded from the PPO move loss). |
| `--aux_anneal_updates` | `0` | Linearly decay `store_bonus` + `store_bc_weight` to 0 over N updates. |
| `--n_train_distractors` | `0` | Fixed number of distractor patterns preloaded per trajectory Hopfield. |
| `--n_train_distractors_min` / `--n_train_distractors_max` | `0` / `0` | If max > 0, draw the count `~U[min, max]` per trajectory (overrides the fixed knob). |

**Reward shaping / exploration**

| Flag | Default | Effect |
|---|---|---|
| `--novelty_reward` | `0.0` | `+r` on first visit to a snapped cell (explore phase, per rollout). |
| `--revisit_penalty` | `0.0` | `−r` per step on an already-visited cell. |
| `--wall_penalty` | `0.0` | `−r` per step on a grid-edge cell. |
| `--epsilon_explore` | `0.0` | Per-step probability of replacing the sampled action with a uniform random direction; log-prob is re-scored, and the step is masked out of the PPO move loss. |
| `--epsilon_anneal_updates` | `0` | Linear decay of ε to 0 over N updates (0 = constant). |
| `--load_checkpoint` | — | Resume from any `hopfield_nav` checkpoint, including one written by `train_navigate`. **The checkpoint's config is the base**; only flags you actually type override it, so an architecture you do not restate is inherited rather than silently replaced by this script's defaults (they differ on 8 of 17 agent fields, `movement_mode` among them). `--save_dir` is never inherited. Writes `world_parent.json` alongside `world.json`. |
| `--refresh_envs_each_update` | off | Re-draw goals and re-place envs each update, randomly over the whole scaffold. Superseded by `--refresh_place` / `--refresh_goal`; refused alongside them. |
| `--env_generator` / `--no-` | off | Draw envs from declared domains instead of the historical placement path. Off keeps today's envs for a given `--seed`; on fixes the offset-reproducibility bug and enforces train/val separation. **`world.json` is written either way.** |
| `--place_region` | `anywhere` | `anywhere` or `rect:X0,Y0,W,H`. Declaring a rect is what makes a place-OOD val set possible later — its complement. |
| `--goal_region` | `any` | `any`, `ring:W`, `interior:W`, `quadrant:Q`. |
| `--wall_seeds` | `0,10000000` | `LO,HI` range training draws wall seeds from. |
| `--place_margin` | derived | Edge-to-edge train/val clearance, in cells. Default reads the scaffold's own cosine-vs-distance curve. |
| `--goal_val_frac` | `0.2` | Share of goal cells reserved for validation. |
| `--refresh_place` / `--refresh_wall` / `--refresh_goal` / `--refresh_size` | — | Re-draw that trait across the train set every N updates, from the declared domain and clear of the fixed val envs. Requires `--env_generator`. Every draw is recorded into `split.used`. |
| `--explore_steps` | `None` | Two-phase rollout: stores + shaping are active only for the first N steps of each rollout. |

Note: `persistence_bonus`, `novelty_scale_remaining` and `novelty_scale_cap`
exist in `HopfieldConfig` but are **not** exposed here (only in
`train_navigate`).

**Agent architecture / inputs**

| Flag | Default | Effect |
|---|---|---|
| `--hidden_size` | `128` | Trunk width. |
| `--num_rnn_layers` | `1` | Trunk depth (dropout only applies with >1 layer, and `dropout` is not on the CLI). |
| `--rnn_cell` | `gru` | Recurrent cell: `gru` (historical default) or `rnn` (vanilla Elman, ungated). |
| `--rnn_nonlinearity` | `tanh` | Activation for `--rnn_cell rnn`. `tanh`/`relu` run on cuDNN; `softplus` is a Python recurrence and gives a strictly positive, unbounded state (idles at a `+0.693`/unit DC offset, so the heads see a shifted feature distribution — see `policy/recurrent.py`). Combining a non-`tanh` value with `--rnn_cell gru` is an error, not a silent no-op. |
| `--hopfield_mode` | `discrete` | `discrete` ⇒ signal is a 4-way one-hot of the projected direction; `continuous` ⇒ the 2-D unit vector. |
| `--input_encoded_state` / `--no-` | `True` | Feed the `D`-dim embedding of the current cell. |
| `--input_hopfield_signal` / `--no-` | `True` | Feed the recall direction (4-d or 2-d). Turning this off makes `--hopfield-oracle` a no-op at eval (eval_all warns). |
| `--input_prev_action` / `--no-` | `False` | Feed the previous action (one-hot or raw 2-d); zeroed after a teleport. |
| `--input_prev_reward` / `--no-` | `False` | Feed the previous step's reward; zeroed after a teleport. |
| `--input_hopfield_raw` / `--no-` | `False` | Continuous mode: feed the **unnormalized** `q` so its magnitude carries recall confidence. |
| `--input_goal_in_memory` / `--no-` | `False` | Feed a 1-bit "the agent has stored at the goal this rollout" flag. |
| `--input_sensory` / `--no-` | `False` | Feed the raw `observation_size` foveal vector. |
| `--init_log_std` | `0.0` | Continuous policy initial log σ. |
| `--freeze_log_std` | off | Pin log σ (no gradient). |

`--input_hopfield_multistep` is **not** available here (only in
`train_navigate`), even though `AgentConfig` supports it.

**PPO / BC**

| Flag | Default | Effect |
|---|---|---|
| `--lr` | `3e-4` | Adam lr (PPO mode). |
| `--ent_coef` | `0.01` | Movement entropy bonus. |
| `--store_ent_coef` | `0.05` | Store entropy bonus (masked by the explore phase). |
| `--training_mode` | `ppo` | `bc` switches to DAgger: rollouts record oracle labels, `bc_update` replaces `ppo_update`, and the optimizer lr comes from `--bc_lr`. |
| `--bc_lr` | `3e-4` | lr in BC mode. |
| `--bc_store_weight` | `1.0` | Weight on store BCE vs. movement CE. |
| `--bc_move_ent_coef` | `0.0` | Entropy bonus in BC mode. |
| `--bc_supervise_explore` / `--no-` | `True` | False ⇒ only supervise steps where the Hopfield direction is trusted (post-store-at-goal). |
| `--bc_nav_weight` | `1.0` | Per-step weight on trusted-Hopfield move labels (>1 counteracts dilution by abundant novelty labels). |
| `--bc_n_minibatches` | `4` | |
| `--bc_epochs` | `1` | Gradient epochs per rollout buffer. |
| `--bc_novelty_fallback` | `random` | Novelty-oracle behavior when every neighbor is visited: `random` or `stay`. |
| `--bc_bce_pos_weight_cap` | `0.0` | Cap on the store-BCE `pos_weight` (0 = uncapped `n_neg/n_pos`). |

Not exposed (PPO defaults from `PPOConfig`): `gamma=0.99`, `gae_lambda=0.95`,
`clip_coef=0.2`, `vf_coef=0.5`, `max_grad_norm=1.0`, `ppo_epochs=4`,
`n_minibatches=4`, `bce_detach_trunk=False`, `bce_pos_weight_cap=0.0`.

**Scale / schedule / bookkeeping**

| Flag | Default | Effect |
|---|---|---|
| `--num_worlds` | `1` | Independent scaffolds. Each costs a full `encoded_Phi`. |
| `--envs_per_world` | `4` | Envs per scaffold; rollouts per update = `num_worlds × envs_per_world`. |
| `--batch_envs` | `16` | Parallel trajectories per rollout (`B`). |
| `--steps_per_rollout` | `64` | Steps per trajectory (`T`). |
| `--n_updates` | `1000` | Total updates. |
| `--num_val_envs` | `2` | Envs in the dedicated eval world. |
| `--n_val_trials` | `32` | Trials per (val env, distractor count). |
| `--val_distractors` | `0` | Distractor counts swept at every eval, e.g. `0 5 10`. |
| `--realistic_steps_per_env` | `1000` | End-of-training realistic eval length; 0 skips. |
| `--hopfield_oracle` | off | Eval-only: replace recall with the true goal-minus-current displacement. |
| `--action_oracle` | off | Eval-only: replace movement with a greedy step toward the goal. |
| `--eval_every` | `50` | Updates between evals (0 disables). |
| `--ckpt_every` | `None` | Updates between checkpoints. `None` = follow `--eval_every`. Before 2026-08-06 the save sat inside the eval branch, so a large `--eval_every` also thinned the checkpoint series `analysis.trajectories` draws its rows from. |
| `--save_every` | `100` | Updates between checkpoints. |
| `--save_dir` | `None` | Defaults to `$CLS_RUNS/agent_ckpts/<wandb name or timestamp>`. |
| `--load_checkpoint` | `None` | Resume/fine-tune; also restores the optimizer state but forces the CLI lr onto every param group. |
| `--seed` | `0` | torch + numpy + env seeds. |
| `--device` | `cuda` | |
| `--use_wandb` | off | |
| `--wandb_project` | `hopfield-nav` | |

</details>

### 4.2 `python -m hopfield_nav.train_navigate` — the active harness

**Explore and exploit are regimes, not phases.** An exploit ("follow",
"pre_stored") env starts with the goal already in its Hopfield; an explore
("empty") env does not. Every update collects rollouts from both and pools them
into **one PPO step**, so what the schedule controls is the *fraction of envs in
the explore regime* on each update.

Structure (`train_navigate.py`, with the regimes in `training/explore.py` and
`training/exploit.py` and the grammar in `training/stages.py`):

- Store head frozen for the entire run (`set_phase_freeze(..., freeze_store=True)`);
  `auto_nav_warmup`, `auto_store_warmup`, `store_bc_weight` are forced to 0 and
  `bce_detach_trunk` to False. So **nothing store-related trains here** — that
  is `train_store`'s job.
- Total updates = the sum of the schedule's stages.
- Each update splits `envs_per_world` into `n_emp = round(n_envs · frac)`
  explore envs and the rest exploit, `frac` coming from the current stage.
  **The first `n_pre` envs are the exploit regime**, the remainder explore.
- Exploit env: Hopfield gets the goal encoding plus `U[min, max]` distractors
  drawn from outside the env's patch, stored in shuffled order (so the goal is
  not always first). `novelty_reward = 0`, `goals_active = True`,
  `goal_in_memory_init = True`, ε = 0.
- Explore env: Hopfield gets `U[min, max]` distractors and **no goal**.
  `novelty_reward = current_novelty`, `goals_active = not explore_goals_off`,
  optional fresh goal per rollout, ε-exploration applied here only.
- One Adam optimizer spans the whole schedule — stages are segments of a single
  training trajectory, so its moments carry across a stage boundary. A stage's
  `lr=` retunes the existing param group in place.
- Run-wide anneals, all keyed off the **global** update counter, not a
  stage-local one: novelty anneal (linear to 0 over the whole budget),
  ε anneal, distractor curriculum (max counts ramp to `*_max_end` over
  `distractor_curriculum_updates`), and a programmatic log-σ anneal that writes
  `agent.movement_log_std` directly between `log_std_anneal_start_update` and
  `log_std_anneal_end_update`.
- Every `eval_every`: `do_eval` (nav-det + discovery + exploration) with
  `max_steps = steps_per_rollout`. Every `ckpt_every` (default: follow
  `eval_every`): a checkpoint `navigate_u{u}.pt`.

#### The schedule

```
--schedule "explore:200,novelty=0.3 ; interleave:800,empty_frac=1.0->0.5,anneal=50 ; exploit:100,lr=1e-4"
```

Stages separated by `;`; each is `<kind>:<updates>` plus optional
`,key=value` overrides. Whitespace is ignored.

| Kind | Meaning |
|---|---|
| `explore` | `empty_frac` pinned at 1.0 — every env starts without the goal in memory. |
| `exploit` | `empty_frac` pinned at 0.0 — every env starts with it. |
| `interleave` | A mix. Takes `empty_frac`; defaults to 0.5. |

| Key | Meaning |
|---|---|
| `lr` | Adam lr for this stage. |
| `empty_frac` | A number, or `start->end` to anneal linearly across the stage. `interleave` only — the other two kinds already imply theirs, and saying it there is an error. |
| `anneal` | Updates from the **stage's** start to reach the end fraction, then hold. Defaults to the whole stage. Must not exceed it. |
| `novelty`, `eps` | Novelty reward / ε for this stage. |
| `dist_min`, `dist_max` | Distractor counts, exploit regime. |
| `emp_dist_min`, `emp_dist_max` | Distractor counts, explore regime. |

A stage value is an **absolute override**: `explore:200,novelty=0.3` means
novelty is exactly 0.3 for those updates even if `--novelty_anneal` has scaled
the run-wide default down. So a schedule reads without also reading the flags.

Pure exploration is `--schedule "explore:600"`; pure following is
`--schedule "exploit:300"`.

**Translating the old flags.** `--warmup_explore_only_updates W
--interleave_empty_fraction A --interleave_empty_target B
--interleave_anneal_updates K --phase_a_updates N` becomes:

```
--schedule "explore:W ; interleave:N,empty_frac=A->B,anneal=K"
```

with `--phase_a_lr` → `--lr` and `--phase_a_novelty_reward` →
`--novelty_reward`. One deliberate difference: the old anneal clock was global,
so with `W > 0` it was already `W/K` of the way through when the first
interleaved update ran. The stage-local clock starts at the top of the stage.
Everything else is bit-identical — `hopfield_nav/tests/test_schedule.py` pins
both the equivalence and the exception.

<details>
<summary><b>All 70 flags</b></summary>

Flags shared with `train.py` (`--encoder_checkpoint`, `--encoder_gain`,
`--fwhm_ratio`, `--size`, `--observation_size`, `--movement_mode`,
`--hopfield_mode`, `--lambdas`, `--Np`, `--static-vectorhash`, `--hidden_size`,
`--num_rnn_layers`, `--init_log_std`, `--freeze_log_std`, `--batch_envs`,
`--steps_per_rollout`, `--num_worlds`, `--envs_per_world`, `--num_val_envs`,
`--n_val_trials`, `--val_distractors`, `--seed`, `--device`, `--eval_every`,
`--save_dir`, `--use_wandb`, `--wandb_project`, `--load_checkpoint`, the six
`--input_*` toggles, `--goal_radius`, `--epsilon_explore`,
`--epsilon_anneal_updates`, `--revisit_penalty`, `--wall_penalty`) behave as in
§4.1 except for the different defaults noted below.

| Flag | Default | Effect |
|---|---|---|
| `--schedule` | *required* | The stage list. See above. Required unless `--load_checkpoint` carries one. |
| `--lr` | `3e-4` | Adam lr; a stage's `lr=` overrides it. |
| `--novelty_reward` | `0.1` | Novelty reward applied **only to explore-regime envs**. |
| `--novelty_anneal` / `--no-` | `False` | Linear decay of novelty to 0 over the full budget. A stage's `novelty=` ignores it. |
| `--load_checkpoint` | `None` | Start from this `.pt`. Its config becomes the **base** for the run: every setting is inherited except the flags actually on the command line, so a child reproduces its parent's recipe without re-listing it. `--save_dir` is never inherited. |
| `--explore_goals_off` / `--no-` | `False` | Explore-regime envs emit no goal reward and never teleport on goal-reach; exploit-regime envs are unaffected. Forces explore to be paid purely by novelty / revisit / wall / time. Eval envs are always built with `goals_active=True` regardless. |
| `--move_ent_coef` | `None` | Overrides `PPOConfig.ent_coef`. |
| `--ppo_clip_coef` | `None` | Overrides `PPOConfig.clip_coef` (0.2). |
| `--persistence_bonus` | `0.0` | `+bonus · cos(aₜ, aₜ₋₁)` per step — stateless straightness reward. |
| `--novelty_scale_remaining` / `--no-` | `False` | Scale novelty by `total_cells / n_unvisited` so late cells pay more. |
| `--novelty_scale_cap` | `10.0` | Cap on that multiplier. |
| `--n_train_distractors_min` / `--n_train_distractors_max` | `0` / `0` | Distractors in **pre-stored** (follow) Hopfields, `~U[min, max]` per rollout. max=0 disables (falls back to the cached goal-only pool). |
| `--n_train_emp_distractors_min` / `--n_train_emp_distractors_max` | `0` / `0` | Distractors in **empty** (explore) Hopfields, no goal pattern. Teaches the explore policy to ignore non-goal recalls. |
| `--n_train_distractors_max_end`, `--n_train_emp_distractors_max_end` | `None` | Curriculum targets for the two max counts. |
| `--distractor_curriculum_updates` | `0` | Updates over which the max counts ramp. |
| `--log_std_anneal_start_update`, `--log_std_anneal_end_update`, `--log_std_anneal_target` | `0`, `0`, `None` | Programmatically interpolate `movement_log_std` from its init to the target across that window. Works even with `--freeze_log_std` (the value is written, then stays frozen at the new value). |
| `--goal_reward` | `1.0` | Reward at the goal cell. Raising it strengthens the follow gradient relative to novelty. |
| `--time_penalty` | `None` | Override `EnvConfig.time_penalty` (0.01). |
| `--continuous_normalize` / `--no-` | `None` | Unit-normalize the action before applying `continuous_scale` (fixed step length). |
| `--max_action_norm`, `--min_action_norm` | `None` | Clamp/boost action L2 (only when `continuous_normalize` is False). |
| `--input_hopfield_multistep` | `[]` | e.g. `1 2 3` — project the recall at each of those iteration counts and pass each as an extra 2-D input (continuous mode only). |
| `--union_cov_trials` | `0` | If >0, `do_eval` also runs union-coverage with this many rollouts per env. |

**The env generator and per-trait refresh.** An environment is four independent
traits — wall pattern, env-local goal cell, scaffold placement, size — and the
generator declares a domain for each, then draws train and validation together so
separation is a property of the draw rather than something checked afterwards.
Full rationale in `docs/EVAL_SPLITS_DESIGN.md`; status and per-phase outcomes in
`docs/ENV_GENERATOR_STATUS.md`.

| Flag | Default | Effect |
|---|---|---|
| `--env_generator` / `--no-` | `False` | Draw envs from declared domains instead of the historical placement path. Off keeps today's envs for a given `--seed`; on fixes the offset-reproducibility bug (§1.4) and enforces train/val separation. `world.json` is written either way. |
| `--place_region` | `anywhere` | Where train envs may sit: `anywhere` or `rect:X0,Y0,W,H`. Declaring a rect is what makes a place-OOD val set possible later — it is the complement. |
| `--goal_region` | `any` | Which env-local cells may hold a goal: `any`, `ring:W`, `interior:W`, `quadrant:Q`. Declaring a region is what makes a goal-OOD val set possible. |
| `--wall_seeds` | `0,10000000` | `LO,HI` range training draws wall seeds from. |
| `--place_margin` | `None` | Edge-to-edge clearance between **every** pair of envs — train↔train, train↔val, val↔val. Measured on the torus, max over axes. `None` derives it from the scaffold's own cosine-vs-distance curve (≈80 at `lambdas=11,12,13`, `fwhm=0.25`). |
| `--goal_val_frac` | `0.2` | Share of goal cells reserved for validation. Only binds when `--refresh_goal` is set. |
| `--refresh_place` | `None` | Re-draw train placements every N updates, clear of the fixed val envs by the margin. |
| `--refresh_wall` | `None` | Re-draw train wall seeds every N updates, excluding every seed the run has already used. Rebuilds the envs — 56 ms at `observation_size` 12, 129 ms at 60, for 80 envs. |
| `--refresh_goal` | `None` | Re-draw train goals every N updates from the train share of `--goal_region`. Replaces the old `--randomize_goal_per_rollout`, which drew uniformly over the arena and so could land on a cell reserved for validation. Setting it also caps the train goal cells at `1 - --goal_val_frac` of the region up front, without which the arena is exhausted in a few updates. |
| `--refresh_size` | `None` | Re-draw the train env size every N updates. Needs more than one declared size, which nothing produces yet, so this currently errors at startup. |

All four `--refresh_*` require `--env_generator` — the legacy path declares no
domains to re-draw from — and each is a cadence in updates, firing when
`update % N == 0`. **Only the train set refreshes**: `base_val` is drawn once and
held, because a validation set that moved under the model would make every
in-training curve unreadable. Every refreshed value is folded into the union
recorded in `world.json`, which is what a later held-out val set excludes
against; the file is rewritten on the `--ckpt_every` cadence.

**A refreshing run gets a startup preflight.** Refreshed values are a pure
function of `(seed, tick)` and the declared domains — nothing training does
enters — so it replays the exact ticks (≈1.4 s for 300 updates, building no
envs) and reports what a post-hoc held-out eval will still be able to ask for,
into stdout and `world.json`'s `diagnostics.preflight`. Two outcomes, and only
one is fatal:

- **A shrinking ceiling is recorded and the run proceeds.** The largest held-out
  place set a run can support falls as the used-offset union grows — at the
  working config with `--refresh_place 1`, from ~187 envs to ~10. That limits a
  later `--split place=held_out --num_val_envs N`, and nothing else; a run that
  only ever uses `--split recorded` is unaffected.
- **A domain that runs dry is refused at startup.** If a trait's domain empties
  mid-run — a narrow `--wall_seeds` against a fast `--refresh_wall`, say — the
  refresh *raises*, hours in, at a tick already decided. The preflight names the
  update and the run does not start.

Different defaults vs `train.py`: `observation_size` 12, `movement_mode`/
`hopfield_mode` `continuous`, `input_sensory`/`input_prev_action`/
`input_prev_reward`/`input_hopfield_raw` **on**, `input_encoded_state` **off**,
`init_log_std` −0.5, `lambdas` `11 12 13`, `Np` 400, `static_vectorhash`
**True**, `steps_per_rollout` 400, `envs_per_world` 20, `num_val_envs` 10.

</details>

**Launching.** Three sbatch scripts wrap this entry point: `run_explore.sh`
(`explore:600`), `run_exploit.sh` (`exploit:300`), and `run_navigate.sh`
(explore → interleave → exploit). All three are env-var driven:

```bash
SCHEDULE='explore:400 ; exploit:200' SEED=7 sbatch hopfield_nav/run_navigate.sh
LOAD_CKPT=$CLS_RUNS/agent_ckpts/navigate_<run>/navigate_final.pt sbatch hopfield_nav/run_exploit.sh
```

Their shared body, `hopfield_nav/navigate_job.sh`, is a **pass-through, not a
policy**: every flag above has an environment variable named after it in upper
case, and *an unset variable is not passed at all*. One rule covers both cases —
on a fresh run an unpassed flag falls back to the trainer's own argparse
default, and under `LOAD_CKPT` it is inherited from the parent. Booleans take
`1`/`0` and become `--flag` / `--no-flag`; lists take a space-separated string
(`LAMBDAS="11 12 13"`). Only `SCHEDULE`, `ENCODER` and `DEVICE` are always
passed. (`--union_cov_trials` is the one flag with no variable — it is
deprecated and ignored.)

Each launcher assigns only the knobs that run means to set, as
`X=${X:-value}` so the environment still wins, and leaves the rest to the
trainer's defaults. Any of the 70 can be added to a launcher — the shared body
has a variable for every one. Note that anything a launcher assigns overrides
a `LOAD_CKPT` parent too, so on a resume set only what you mean to change.

**Sweep mechanics (historical).** `run_phase_a_sweep_evelina.sh` picks `EXTRA`
from a `case $VARIANT` block and appends it to a fixed base command, so a
variant's flags *override* the base by appearing later on the command line. It
passes the removed `--phase_a_*` / `--interleave_*` flags, so **it no longer
runs**; it is kept as the record of what each of the 101 variants was. To read
what a past run was, `grep -A5 "  <variant>)" hopfield_nav/run_phase_a_sweep_evelina.sh`.

### 4.3 `python -m hopfield_nav.train_phased` — the four-phase pipeline

| Phase | Updates flag | Hopfield role | Frozen | Loss |
|---|---|---|---|---|
| 1 store pretrain | `--phase1_updates` (20) | per-env empty, force-store at goal | move, value, RNN | BCE(store_logits from **detached** features, at_goal) × `phase1_bce_weight`, `pos_weight = n_neg/n_pos` |
| 2 follow pretrain | `--phase2_updates` (100) | `pre_stored_shared` (goal preloaded, no writes) | store (if `phase2_freeze_store`) | PPO; `auto_nav_warmup = --phase2_auto_nav_warmup` teacher-forces the first N updates |
| 3 explore pretrain | `--phase3_updates` (200) | `empty_shared` (no writes) | store | PPO; log σ is first overwritten with `--phase3_init_log_std` |
| 4 compose | `--phase4_updates` (300) | `empty_per_env` (agent writes) | nothing | PPO with `store_bc_weight = --phase4_bce_weight`, `bce_detach_trunk = True`, lr `--phase4_lr` |

`do_eval` runs at every `--eval_every` inside phases 2–4 and once after each
phase boundary. Only one artifact is written: `phased_final.pt` at the very end
(no intermediate checkpoints).

Remaining flags (39 total) mirror §4.1: `--encoder_checkpoint`, `--encoder_gain`,
`--fwhm_ratio`, `--size`, `--observation_size` (12), `--movement_mode`
(continuous), `--goal_radius`, `--hopfield_mode` (continuous), the six
`--input_*` toggles (`prev_reward`/`prev_action`/`hopfield_raw`/`sensory` on,
`encoded_state` off, `hopfield_signal` on), `--lr`, `--batch_envs`,
`--steps_per_rollout` (64), `--num_worlds`, `--envs_per_world` (4),
`--num_val_envs` (2), `--n_val_trials`, `--val_distractors`, `--seed`,
`--device`, `--eval_every`, `--save_dir`, `--use_wandb`, `--wandb_project`
(`hopfield-nav-phased`), `--lambdas` (11 12 13), `--Np` (1600),
`--static-vectorhash` (default **True**).

### 4.4 `python -m hopfield_nav.train_store` — store-head pretrain

Loads a Phase-A checkpoint, reconstructs its config, then forces
`goals_active=True`, `store_bc_weight=1.0`, `bce_detach_trunk=True`,
`bce_pos_weight_cap=--bce_pos_weight_cap`, `ent_coef=0`. Freezes move, value and
RNN; the optimizer sees only the store head. Rollouts use `empty_shared`
Hopfields, and `ppo_update` runs — but with everything else frozen, the only
live gradients are the store surrogate and the store BCE.

**Two worlds are recorded.** Phase B is a continuation of the *agent*, not of
the world: it draws its own envs from its own `--seed`, which is usually not the
parent's, so its eval numbers are **not on the same axes** as Phase A's. Rather
than leave that to be discovered by whoever plots the two curves together, the
run directory says it outright:

| File | What |
|---|---|
| `world.json` | Phase B's own world, as for every other trainer. This is what `--split` and every eval driver resolve. |
| `world_parent.json` | A **verbatim** copy of Phase A's `world.json` — byte for byte, so its own `spec_hash` still verifies, and so this directory answers "what did the parent train on" after the parent has been moved or garbage-collected. |

Since 2026-08-12 this is **every** trainer's behaviour, not `train_store`'s
alone: any run with a `--load_checkpoint` parent records both worlds.

`world.json`'s `diagnostics.parent` block records the parent's hash and how much
of its world this run reuses: `val_envs_identical` (the one that answers "can
these curves go on the same axes"), shared wall seeds, shared goal cells, shared
train offsets, and `min_place_gap_vs_parent` — 0 would mean the two train sets
overlap on the scaffold. Checkpoints carry both summaries as `world_spec` and
`parent_world_spec`. A parent written before world recording gets a printed note
and no `parent` block, rather than a silent absence.

The refresh cadence inherited from the parent's config is now **applied** rather
than silently dropped; a cadence with no generator behind it is refused at
startup, as elsewhere.

| Flag | Default | Effect |
|---|---|---|
| `--load_checkpoint` | required | Phase-A `.pt` to start from. |
| `--encoder_checkpoint` | required | Encoder path (not taken from the ckpt). |
| `--phase_b_updates` | `50` | |
| `--phase_b_lr` | `3e-4` | Adam lr on the store head. |
| `--bce_pos_weight_cap` | `5.0` | Cap on `n_neg/n_pos`; 0 = uncapped. Uncapped values (~19) previously drove high off-goal firing. |
| `--steps_per_rollout` | `None` | Override the checkpoint's value. |
| `--eval_every` | `5` | Evaluation cadence. |
| `--ckpt_every` | `None` | Checkpoint cadence (`store_u{u}.pt`). `None` = follow `--eval_every`, which is what it did unconditionally before 2026-08-06. |
| `--seed` / `--device` / `--use_wandb` / `--wandb_project` | `42` / `cuda` / off / `hopfield-nav-phase-b` | `--seed` draws Phase B's own world; it is usually not the parent's seed. |
| `--save_dir` | `<CLS_RUNS>/agent_ckpts/store_<run>` | Not inherited from the parent, which would have Phase B overwrite its own parent. |

### 4.5 What one rollout actually does

`RolloutCollector.collect_rollout` (`rollout.py:48-639`), per step *t*:

1. `positions = vec.positions()`, `at_goal_mask = at_goal(vec)`.
2. `current_reward = goal_reward if at_goal (and goals_active) else −time_penalty`
   — this is always input channel 0.
3. `embeddings = encoded_Phi[positions + env_offset]`.
4. Hopfield recall for every trajectory whose Hopfield is non-empty; project
   through the cached Gram–Schmidt basis (recomputed when
   `t % recompute_interval == 0` or the trajectory just teleported) →
   `q (B,2)`; signal = one-hot direction (discrete) or `q/‖q‖` (continuous).
5. Action overrides: ε-exploration (random direction with probability
   `epsilon_now`) and/or auto-nav teacher forcing; ε wins where both apply.
   Overridden steps get `policy_action_mask = 0` and are dropped from the PPO
   move loss. In BC mode overrides are suppressed and teacher **labels** are
   recorded instead.
6. RNN input is concatenated in this fixed order:
   `[current_reward, (prev_reward), (encoded_state), (hopfield signal or raw q),
   (multistep q…), (prev_action), (sensory), (goal_in_memory bit)]`.
7. `agent.get_action_and_value(...)`.
8. Stores are applied **before** the env step, only when the trajectory owns its
   own Hopfield and the step is in the explore phase:
   `effective_store = agent_store | (auto_store_active & at_goal)`.
9. `vec.step_batch(actions)` → base reward; then explore-phase shaping:
   `−store_cost·store`, `+store_bonus·(store & at_goal)`,
   `+novelty·first_visit` (optionally scaled by `total/remaining`, capped),
   `−revisit_penalty·revisit`, `−wall_penalty·at_edge`,
   `+persistence_bonus·cos(aₜ, aₜ₋₁)`. All shaping terms are multiplied by
   `~at_goal_mask`, so a teleporting trajectory earns none of them.
10. Teleported trajectories: hidden state, `prev_reward`, `prev_action` zeroed;
    Gram–Schmidt cache invalidated.
11. After the loop, the value head is evaluated once more at the final state
    (with a real Hopfield signal) to give GAE its bootstrap.

`ppo_update` then pools every rollout, computes GAE **per rollout**, normalizes
advantages over the whole pool, and runs `ppo_epochs × n_minibatches` gradient
steps with minibatches drawn over whole trajectories (preserving the RNN's time
axis). The store surrogate and store entropy are masked by the explore phase;
the move surrogate is masked by `policy_action_mask`.

### 4.6 `python -m hopfield_nav.train_rnn` — the no-memory baseline

Each env is a separate `GridEnv` with its own codebook and goal; the agent sees
only sensory (plus optional `prev_action`/`prev_reward`/`grid_state`) and is
trained by BC against the BFS oracle. In a rollout, once a trajectory sits on
the goal it is marked *done* and frozen (no teleport, no further steps) and all
its later steps are masked out of the loss.

| Flag | Default | Effect |
|---|---|---|
| `--mode` | `sequential` | `sequential`: train env 0…N−1 in order, evaluating envs `0..i` after **every** update (this is the forgetting curve). `mixed`: pool rollouts from all envs each update (pretraining). `finetune`: load a checkpoint, then run sequential. |
| `--n_envs` | `4` | Number of envs. |
| `--updates_per_env` | `100` | sequential/finetune. |
| `--n_updates` | `1000` | mixed only. |
| `--size`, `--observation_size`, `--time_penalty`, `--movement_mode` | `8`, `60`, `0.01`, `continuous` | Env config. `--goal_radius` (0.5) is accepted but **not passed to the envs**. |
| `--hidden_size`, `--num_rnn_layers`, `--dropout` | `128`, `1`, `0.0` | Trunk. |
| `--rnn_cell`, `--rnn_nonlinearity` | `gru`, `tanh` | Recurrent cell; see the `train_navigate` table. Restored from the checkpoint in finetune mode, since the cell changes parameter shapes. |
| `--init_log_std`, `--freeze_log_std` | `0.0`, off | Continuous head. |
| `--input_prev_action`, `--input_prev_reward`, `--input_grid_state` | off | Extra input channels. `input_grid_state` appends the smoothed-gbook column at the agent's **global** scaffold position (requires building a VectorHash; adds `Σλ²` dims). |
| `--fwhm_ratio`, `--lambdas` | `0.25`, `11 12` | Only used when `--input_grid_state`. Auto-restored from the checkpoint in finetune mode. |
| `--lr`, `--move_ent_coef`, `--epochs`, `--n_minibatches`, `--max_grad_norm` | `1e-3`, `0.0`, `4`, `4`, `1.0` | BC update. |
| `--only_train_on_reached` | off | Drop trajectories that never reached the goal; skip the update if none did. |
| `--batch_envs`, `--steps_per_rollout` | `16`, `64` | Rollout shape. |
| `--eval_every`, `--n_eval_trials`, `--eval_max_steps` | `25`, `32`, `64` | Console-log cadence and eval budget. |
| `--seed`, `--device`, `--save_dir`, `--load_checkpoint` | `0`, `cuda`, `checkpoint_rnn/<run>`, `None` | |
| `--env_generator` / `--no-` | `False` | Draw envs from declared domains and record them, as `train_navigate` does, instead of the historical placement path. Builds a scaffold for placement **whether or not the agent observes one** — under `--no-input_grid_state` the RNN never sees where its envs sit, but the offsets are still part of the world's identity and an agent-hash run on the same `world.json` does see them. The same config draws the same envs at the same offsets either way. Needs an explicit `--place_margin`: deriving one reads the scaffold's cosine-vs-distance curve, which needs an encoder this stack does not have. |
| `--place_region`, `--goal_region`, `--wall_seeds`, `--place_margin`, `--goal_val_frac`, `--n_val_envs` | `anywhere`, `any`, `0,10000000`, `None`, `0.2`, `2` | Same meanings as the `train_navigate` table above. |

Both this and `analysis/continual/baseline.py` write a **`world.json`** beside
their output, on both paths — the same file and reader `train_navigate` uses.
That is what lets a baseline run and an agent-hash run be handed one record,
rather than being matched through the draw-order convention documented at
`agenthash.py:325-333`. Placement previously drew from global `np.random` here
too, so which offsets a baseline used was unrecoverable afterwards.
| `--use_wandb`, `--wandb_project` | off, `hopfield-nav-rnn` | |
| `--plot_smooth_window` | `1` | Rolling mean for the two auto-generated plots. |

Artifacts: `final.pt` (weights, optimizer, cfg, full history, env goals),
`forgetting.png`, `steps_to_goal.png`.

---

## 5. Stage 3 — evaluation

### 5.1 `python -m hopfield_nav.eval_all --ckpt <path>`

Reconstructs the training config from the checkpoint, rebuilds the eval world,
loads the agent, and runs whichever evaluators are enabled.

**The eval world comes from the run's `world.json` when it has one** (found from
`--ckpt`). Until 2026-08 nothing passed the record through, so every evaluation
replayed the training seed stream — which recovers wall codes and goals but
*not* offsets, because placement drew from global `np.random`. Runs written
before Phase 3 have no record and still take that path, with a warning; their
offsets are a fresh draw, not the ones training evaluated against.

| Flag | Default | Effect |
|---|---|---|
| `--ckpt` | required | Any `hopfield_nav` agent checkpoint. |
| `--split` | `recorded` | Which validation envs to evaluate on. **Repeatable** — each extra one reuses the same scaffold, so several combinations cost one build. `recorded` is the run's own `base_val`, read from `world.json` exactly as trained against. Otherwise `trait=level` pairs over `place`/`wall`/`goal`, levels `same` \| `held_out` \| `ood`; unnamed traits default to `held_out`. Needs a `world.json` for anything but `recorded`. |
| `--val_seed` | `0` | Seed for minting split env sets; changing it draws a different set at the same levels. |
| `--val_size` | ckpt | Mint the validation envs at this arena size — the size-OOD axis. **Repeatable**, and crossed with `--split`, so one run gives a size column in the same table. `recorded` is paired with its own size and left out of the cross product; passing `--val_size` when every `--split` is `recorded` is an error rather than a no-op. Equal to the trained size it is a no-op and the results are byte-identical to the same `--split` without it; differing, it appends `,size=N` to the split key. Wall novelty across sizes is seed disjointness only — the Hamming margin is reported as `null` with a reason, because wall codes of two sizes are different-shaped draws. |
| `--scaled-budget` / `--no-` | on | At a `--val_size` other than the trained one, also run exploration at a `size²`-scaled `max_steps` and navigation at a `size`-scaled one, under `exploration_scaled` / `nav_det_scaled`. A bigger arena has more cells, so coverage at a fixed budget must fall whether or not anything generalizes; reporting one number would confound capability with budget. Doubles those two evaluators' cost. |
| `--encoder` | ckpt's path | Override the encoder. |
| `--device` | `cuda` | |
| `--tag` | basename of ckpt | Label in printed output and JSON. |
| `--Npos` | ckpt / `Πλ` | Override scaffold side. Changing it changes the embedding geometry, so results are not comparable across values. |
| `--num-val-envs` | ckpt | Number of eval envs. |
| `--goal_radius` | ckpt (0.5 for old ckpts) | At-goal threshold for all evals. |
| `--static-vectorhash` / `--no-` | ckpt | Skip pbook/Wgp/Wsp build. |
| `--hopfield-oracle` / `--no-` | ckpt | Replace recall with the exact goal displacement (same projection). Warns if `input_hopfield_signal=False`, in which case it does nothing. |
| `--action-oracle` / `--no-` | ckpt | Replace movement with a greedy step toward the goal; store head untouched. |
| `--num_trials` | `32` | Trials per (env, distractor count) for nav/discovery/exploration. |
| `--max_steps` | `200` | Step cap for those three. |
| `--n_distractors` | `0` | List, e.g. `0 5 10`. |
| `--no-nav-stoch` | off | Skip the sampled-action navigation eval. |
| `--realistic-steps` | `1000` | Steps per env in the realistic eval. |
| `--realistic-seed-offset` | `1000` | Added to `cfg.seed` for its RNG. |
| `--skip-realistic` | off | Skip it entirely. |
| `--repeat-trials` | `0` | >0 enables the repeat eval. |
| `--repeat-steps` | `200` | Steps per repeat trial. |
| `--repeat-seed-offset` | `2000` | |
| `--seq-iters-per-block` | `0` | >0 enables the sequential continual eval. |
| `--seq-max-steps` | `32` | Mini-episode cap. |
| `--seq-ma-window` | `20` | Moving-average window on the sequential plot only. |
| `--seq-seed-offset` | `3000` | |
| `--lock-store-after-goal` | off | Realistic + sequential: suppress further stores in an env once its goal is stored. |
| `--oracle-store-at-goal` | off | Sequential only: bypass the store head — store fires exactly when at goal. Use for Phase-A-only checkpoints, whose store head is untrained. |
| `--show-stores` / `--no-show-stores` | on | Store markers on the sequential plot. |
| `--output-json` | `None` | Write the full result dict. |
| `--plot-path` | `None` | Base path; always writes `*_scaffold_layout.png`, plus drift/interval/repeat/sequential PNGs for whichever evals ran. |

### 5.2 Other eval drivers

- ~~`python -m hopfield_nav.eval_distractors`~~ and
  ~~`python -m hopfield_nav.eval_checkpoints`~~ — **deleted in phase 6**. Both
  were thinner CLIs over the same evaluators `eval_all` drives, with hardcoded
  April-2026 paths. For the distractor sweep use
  `eval_all --n_distractors 0 1 3 5 10`; for a run × update table, loop
  `eval_all --ckpt <dir>/<name>_update{N}.pt --output-json` and compare the
  JSON. `eval_checkpoints`' pass gate (`nav_stoch ≥ 0.7`, `store_eff ≥ 0.7`,
  `coverage ≥ 0.45`) has no replacement and was not carried over.
- `python -m analysis.trajectories --checkpoint_dir <dir>` —
  rows = checkpoints (files matching `*_u{N}.pt` or `*_update{N}.pt`; files
  without an update number are skipped), cols = trials, with a fixed
  (env, start, distractor) scenario per column. `--mode combined|explore_only|
  exploit_only`, `--trials 6`, `--explore_steps 200`, `--nav_steps 100`,
  `--n_distractors 0`, `--force_store` (disables the natural store head and
  forces a store at the first at-goal step), `--updates` (comma list),
  `--seed 42`, `--goal_radius`, `--out`, `--device`. Writes PNG **and** PDF.

### 5.3 What the evaluators do

All of them: fresh Hopfield per trial, `RandomState(seed)` per distractor level
(default seed 42), loops env → trial → step, and use the single-env
`agent_step`.

| Evaluator | Hopfield preload | Termination | Reported |
|---|---|---|---|
| `evaluate_navigation` | goal + N distractors, shuffled | first at-goal | `success_rate`, `mean_speed`, `mean_steps`, totals |
| `evaluate_goal_discovery` | distractors only | store fires while at goal | `store_success_rate`, `reach_success_rate`, `store_efficiency`, `mean_steps_to_store`, `mean_steps_all` |
| `evaluate_exploration` | distractors only | never (full `max_steps`) | `mean_coverage`, `goal_find_rate`, `mean_steps_to_goal` |
| `evaluate_union_coverage` | distractors only | never | `mean_union_coverage`, `mean_union_per_rollout` |
| `evaluate_realistic` | persistent across the whole eval | fixed step budget per phase | per-env `n_reaches`/`intervals`, drift curves, `interference_drop` |
| `evaluate_repeat` | fresh per trial | fixed budget | per-trial `n_reaches`, `mean_reaches` |
| `evaluate_sequential_episodes` | persistent | mini-episode ends at goal or `max_steps` | per-env `(iter, success, stored_at_goal, stored_off_goal)`, `mean_primary_success`, `mean_final_revisit_success`, `interference_drop` |

`deterministic=True` means movement = policy mean/argmax and store = `p > 0.5`.
`evaluate_navigation` is the only one called both ways by the training loop
(`nav_det` and `nav_stoch`).

---

## 6. Stage 4 — figure and analysis pipelines

### 6.1 Continual-learning figure (`final_plotting/`)

```bash
sbatch analysis/continual/run_agenthash.sh   # edit RUN_NAME / CKPT / knobs at top
sbatch analysis/continual/run_baseline.sh
bash   analysis/continual/just_plot.sh       # re-render only
```

`run_agenthash.sh` does: `prep_scaffold` once (content-addressed cache keyed by
lambdas + fwhm_ratio + Npos + encoder path) → `NUM_FULL_ITERS` parallel
`agenthash` subruns with seeds `SEED, SEED+1, …` → `merge_histories` →
`plotting`.

`agenthash.py` runs the sequential continual protocol with a **frozen** policy
(only the Hopfield changes), recording `reached`, `steps_to_goal` and
`path_to_goal` per mini-episode:

| Flag | Default | Effect |
|---|---|---|
| `--ckpt`, `--out`, `--run_name`, `--encoder_override`, `--device` | — | ckpt is a phased/Phase-A `.pt`; `--out` is the history JSON path. |
| `--n_envs` | ckpt's `num_val_envs` | Number of envs in the protocol. |
| `--iters_per_block` | `800` | Outer iterations per block. |
| `--max_steps` | `100` | Mini-episode cap. |
| `--seed` | `3000` | Protocol RNG. |
| `--env_seed` | `None` | Use a baseline-compatible env draw order so iter *k* matches `baseline.py`'s iter *k*. |
| `--num_full_iters` | `1` | Repeat the whole protocol with seeds `seed..seed+N−1` for mean ± 1σ bands. |
| `--static_vectorhash`, `--scaffold_cache`, `--mmap` | off | Load `encoded_Phi` from the prep cache; `--mmap` shares OS page-cache pages across parallel subruns. |
| `--stochastic_policy` | off (greedy) | Sample actions instead of argmax. |
| `--lock_store_after_goal` | off | One store per env, ever. |
| `--oracle_store_at_goal` | off | Force a store when at goal. |
| `--oracle_lock_store_not_at_goal` | off | Suppress all off-goal stores. (Setting both reproduces the old combined oracle.) |
| `--train_store`, `--train_store_lr`, `--train_store_updates_per_rollout` | off, `3e-4`, `1` | Online BCE training of the store head during the protocol, trunk frozen. |
| `--goal_radius` | ckpt | Override. |

`baseline.py` runs the same protocol shape for the RNN baseline, but it **BC-trains
as it goes** (that is the point of the comparison: the baseline must relearn each
env). Its 29 flags mirror `train_rnn.py` plus `--out`, `--iters_per_block`,
`--num_full_iters`, `--load_checkpoint` (finetune mode).

`plotting.py --history <json> --out_prefix <prefix> [--smooth N] [--show_std 0|1]`
writes `<prefix>_forgetting.{png,pdf}` and `<prefix>_steps_to_goal.{png,pdf}`.
`merge_histories.py --inputs a.json b.json ... --out merged.json` requires the
inputs to agree on `model_class`, `n_envs`, `env_size`, `iters_per_block`.

### 6.2 Phase decoding (`phase_decoding_v2/`)

```bash
CKPT=/path/to/ckpt.pt sbatch analysis/phase_decoding/run_exp1.sh
CKPT=/path/to/ckpt.pt TRIALS_DIR=/path/to/exp1/trials sbatch analysis/phase_decoding/run_exp2.sh
```

**Exp 1** (`exp1.py`, 19 flags): collect `n_starts` explore trials and
`n_starts` exploit trials per arena over `num_arenas` arenas (distractor count
`~U[n_dist_min, n_dist_max]`, cap `max_steps`), recording the recurrent hidden state
at every step; then for four split families compute per-fold **parallelism**
(cosine between the exploit−explore centroid difference on train arenas and on
test arenas) and **decodability** (balanced accuracy of L2 logistic regression
trained on train arenas, tested on held-out arenas). Notable flags:
`--trials_dir` (reuse an existing collection), `--skip_loo` (LOO dominates
runtime), `--subsample_train N` (bounds LR memory — a 100-arena × 100-start LOO
pool is ~1.8 M rows × 512 dims), `--stochastic`/`--deterministic` (default
stochastic), `--random_agent` + `--random_init_seed` (same architecture,
untrained weights — the control).

**Exp 2** (`exp2.py`, 22 flags): (1) PCA scatter of MLP hidden activations for
explore vs exploit trials; (2) trajectory PCA over two-episode trajectories
where the goal is **oracle-stored** at the switch point and the agent is
teleported *without* resetting `h_rnn` or the Hopfield. MLP knobs:
`--mlp_hidden 64`, `--mlp_epochs 30`, `--mlp_lr 1e-3`, `--mlp_batch 256`;
trajectory knobs: `--n_traj_per_arena 4`, `--max_steps_ep1/ep2 200`.

`plot.py --metrics a/metrics.json b/metrics.json --labels A B --out fig.png`
overlays runs.

---

## 7. The knobs that actually matter

If you are re-running or re-tuning, these are the ones that change outcomes
rather than costs.

**Encoder (stage 1)**
1. `--per_env_radius_frac` — the definition of "near". Everything downstream
   (how far a Hopfield basin reaches, how steep the projected gradient is) is
   set here.
2. `--gain_end` — the tanh sharpness at the end of training, and by default
   the Hopfield β at agent-training time. One number controls both.
3. `--out_dim` — Hopfield capacity and per-env `D×D` memory cost.
4. `--single_env_batch` — with it on, "far" pairs are always within-env, so the
   loss never sees cross-env pairs.
5. `--lambdas` — must be identical in stage 2; also fixes the scaffold size `Πλ`.

**Agent (stage 2)**
6. `--schedule` — the explore/exploit mixture over time. Its predecessor
   (`interleave_empty_fraction` + target/anneal) was the single most-varied
   knob across the 101 sweep variants.
7. `init_log_std` + `freeze_log_std` (+ the anneal window) — with a learnable
   σ and an entropy bonus, σ inflates and the policy navigates by sampling
   noise, which shows up as a large nav-stochastic vs nav-deterministic gap.
8. `epsilon_explore` / `epsilon_anneal_updates` — exploration floor independent
   of σ. Applied only to empty-regime envs in Phase A.
9. The shaping quartet `novelty_reward`, `revisit_penalty`, `wall_penalty`,
   `persistence_bonus` — these define what "explore" means to the agent, and
   they interact (novelty alone rewards perimeter orbits, which is what
   `wall_penalty` exists to counter).
10. `n_train_{,emp_}distractors_{min,max}` — whether training memory
    distributions match eval. Without them the policy sees a clean single-memory
    Hopfield in training and a crowded one at eval.
11. `goal_reward` vs `time_penalty` vs shaping magnitudes — the relative scale
    of these decides which gradient dominates the shared trunk.
12. `steps_per_rollout` — also the eval `max_steps` in Phase A's `do_eval`, so
    changing it changes both the training horizon and the eval budget.
13. `envs_per_world` × `batch_envs` — the effective batch: rollouts per update
    is `num_worlds × envs_per_world`, each of `batch_envs × steps_per_rollout`
    steps.

**Eval**
14. `n_distractors` — the honest measure of memory discrimination; results at 0
    distractors are much easier than at 5–10.
15. `--oracle-store-at-goal` — mandatory for Phase-A-only checkpoints, whose
    store head never trained; without it the sequential eval measures a random
    store head.
16. `deterministic` vs stochastic — `nav_det` vs `nav_stoch` differ by design.
17. `--hopfield-oracle` / `--action-oracle` — the two ablations that separate
    readout error from policy error.

**Cost, not outcome**: `Npos`/`--static-vectorhash` (memory), `num_worlds`
(scaffold copies), `n_val_trials`, `union_cov_trials`, `realistic_steps`,
`seq_iters_per_block`.

---

## 8. Metric definitions

Traced to source; use these when comparing numbers across docs.

- **`success_rate`** (navigation): fraction of trials where `at_goal(env)` was
  true after some step within `max_steps`.
- **`mean_speed`**: mean over successful trials of `start_dist / steps_taken`,
  where `start_dist` is Euclidean in continuous mode and Manhattan in discrete
  mode. Not comparable across movement modes.
- **`mean_steps`**: mean steps over successful trials only.
- **`reach_success_rate`** (discovery): fraction of trials that reached the goal
  at all. **`store_success_rate`**: fraction where a store fired on a step whose
  pre-step position was the goal. **`store_efficiency`** = store ÷ reach.
- **`mean_coverage`** (exploration): unique snapped cells visited ÷ `size²`, per
  rollout, averaged over trials.
- **`mean_union_coverage`**: union of cells across `num_trials` independent
  rollouts ÷ `size²`. **`mean_union_per_rollout`** divides that by `num_trials`,
  so its range is `[0, 1/num_trials]` — a *diversity* measure, not a coverage
  one.
- **`n_reaches`** (realistic/repeat): number of times the agent sat on the goal
  during the phase; **`intervals`** are the step gaps between consecutive
  reaches. **`interference_drop`** = mean over envs of
  `(primary − final_retest) / primary`.
- **Sequential `success`**: 1 if the agent sat on the goal within `max_steps` of
  a mini-episode. **`mean_primary_success`** averages over each env's own block;
  **`mean_final_revisit_success`** averages over the last block's revisits.
- **Encoder nav `accuracy`**: fraction of lattice starts whose open-loop
  Hopfield-chase ends within `platform_radius` of the goal, averaged over envs.
  **`mean_dir_acc_*`**: fraction of steps that decreased distance to goal.
- **RNN baseline `nav_det`**: fraction of parallel trials whose pre-step position
  ever equalled the goal within `eval_max_steps`.

---

## 9. Checkpoint formats and compatibility

| Producer | Keys |
|---|---|
| `encoder_training.train` | `state_dict`, `model_config`, `train_config`, `y0s`, `x0s`, `sizes`, `gain`, (+`val_nav_acc`, `epoch` for best) |
| `hopfield_nav.train` | `agent_state_dict`, `optimizer_state_dict`, `config` (=`asdict(TrainConfig)`), `update` |
| `train_navigate` | `agent_state_dict`, `config`, `update` (per `ckpt_every`) / no `update` (final). `config` carries `schedule`, and is snapshotted before the loop's temporary overrides so it records the run's settings rather than the scratch values the loop parks in `cfg` between rollouts — `--load_checkpoint` reads it back as the base config. |
| `train_store` | `agent_state_dict`, `config`, `update` |
| `train_phased` | `agent_state_dict`, `config`, `phased_config` |
| `train_rnn` | `agent_state_dict`, `optimizer_state_dict`, `cfg`, `history`, `env_goals` |

Every run directory also carries a `run.json` manifest (`run_manifest.py`):
kind, name, status, git sha/branch/dirty, argv, host, slurm job id, wandb
block, `parent` (the `--load_checkpoint` it resumed from), `encoder`
(path + sha256 + out_dim + lambdas + gain), the full config, and the checkpoint
list with update numbers. It is an **index, not the source of truth** — the
config embedded in each `.pt` remains authoritative, and a missing or corrupt
manifest degrades to the pre-manifest behavior. `scripts/backfill_manifests.py`
wrote one for each of the ~350 pre-existing run directories (marked
`provenance: "backfilled"`, with argv/git/wandb null rather than guessed);
`scripts/gc_runs.py` classifies the tree from them.

Compatibility shims that exist in the code:

- `hopfield_nav/encoder_io.py:load_encoder` accepts three encoder-checkpoint
  layouts: top-level `model_config`, `config["model_params"]`, or `config` as
  the model dict; and resolves gain from override → `ckpt["gain"]` →
  `model_config.gain`.
- `coerce_legacy_cfg` (one copy, in `evaluation/checkpoint_io.py` since phase 6;
  previously duplicated across the three eval drivers) renames
  `val_envs_per_world → num_val_envs` and
  `vectorhash.gbook_only → vectorhash.static_vectorhash` when loading old agent
  checkpoints. **This shim covers checkpoints only** — the CLI flag was renamed
  separately, and phase 0 restored `--gbook-only` as a deprecated alias.
  `hopfield_nav/tests/test_checkpoint_io.py` pins both renames.
- `train_rnn.restore_arch_from_ckpt` overrides CLI architecture flags with the
  values saved in the checkpoint (printing a NOTE for each), so finetune runs
  can't accidentally change parameter shapes.

Fields added to `TrainConfig` after a checkpoint was written simply take their
dataclass defaults on load (`cfg_from_checkpoint` only sets keys present in
the saved dict), so e.g. `goal_radius` becomes 0.5 for pre-`goal_radius`
checkpoints.
