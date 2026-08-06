> **Archived.** Moved out of `hopfield_nav/` by phase 6 of the 2026-08
> refactor. Not maintained; describes what was believed and tried at the time,
> which in places is no longer true of the code. Start from `docs/archive/README.md`
> for what replaced it.

# RNN Baseline (`*_rnn.py`)

The vanilla-RNN baseline is a **memory-free control** for the Hopfield-nav
project: a GRU policy with a single movement head, trained by behavior cloning
(DAgger) against a shortest-path oracle. No encoder, no VectorHash, no
Hopfield, no store/value heads — just sensory-in, action-out. It is the
reference point that lets us claim Hopfield-nav's continual-learning behavior
isn't just RNN dynamics.

This doc covers (a) the files exclusive to the baseline, (b) the configs, and
(c) the shared modules that overlap with the main (Hopfield) training stack.

## Files exclusive to the baseline

All paths are under `hopfield_nav/`.

| File | Role |
|---|---|
| `agent_rnn.py` | `RNNAgent` (GRU + 1 move head) and `compute_rnn_input_dim`. |
| `bc_rnn.py` | `bc_rnn_update`: minibatched CE on teacher move action, masked. |
| `rollout_rnn.py` | `RNNRolloutBatch` and `collect_rollout_rnn` (DAgger collector). |
| `eval_rnn.py` | `evaluate_nav_one_env` / `evaluate_nav_all` (per-env nav success). |
| `train_rnn.py` | Entry point: `sequential` / `mixed` / `finetune` modes + plots. |
| `regen_plots.py` | Re-render forgetting / steps-to-goal plots from a saved `final.pt`. |

There are no dedicated `run_*_rnn.sh` scripts — invoke `python -m
hopfield_nav.train_rnn` directly (see Quickstart).

## Configs (`config.py`)

Three dataclasses, fully independent from the Hopfield-side `AgentConfig` /
`BCConfig` / `TrainConfig`:

- **`RNNAgentConfig`** — `hidden_size`, `num_rnn_layers`, `dropout`,
  `movement_mode` (`discrete` / `continuous`), `init_log_std`,
  `freeze_log_std`, and the optional auxiliary input channels
  `input_prev_action` / `input_prev_reward`. The sensory codebook vector is
  always on; the goal is never an input.
- **`RNNBCConfig`** — `lr`, `move_ent_coef`, `epochs`, `n_minibatches`,
  `max_grad_norm`. Move-only loss; no `store_bce_weight`.
- **`RNNTrainConfig`** — top-level. Holds `EnvConfig`, `RNNAgentConfig`,
  `RNNBCConfig`, plus `mode`, `n_envs`, `updates_per_env` (sequential),
  `n_updates` (mixed), `batch_envs`, `steps_per_rollout`, eval cadence/trials,
  W&B fields, and `plot_smooth_window`.

The baseline reuses **`EnvConfig`** verbatim — same grid size, observation
size, time penalty, movement mode, etc. Everything else (`AgentConfig`,
`PPOConfig`, `BCConfig`, `HopfieldConfig`, `VectorHashConfig`, `PhasedConfig*`)
is unused.

## Shared modules (overlap with Hopfield training)

The baseline is deliberately thin and reuses these without modification:

| Module | What's reused | Used by |
|---|---|---|
| `env.py` | `GridEnv`, `CARDINAL_ACTIONS` | `train_rnn`, `rollout_rnn`, `eval_rnn` |
| `vec_env.py` | `VecEnv`, `ContinuousVecEnv` | `train_rnn`, `rollout_rnn`, `eval_rnn` |
| `oracle_bfs.py` | `bfs_action_batch_discrete`, `bfs_action_batch_continuous` | `rollout_rnn` |
| `config.py` | `EnvConfig` | `train_rnn` |

What is **not** shared (and never imported by any `*_rnn.py` file):
`agent.py`, `bc.py`, `eval.py`, `eval_all.py`, `rollout.py`, `ppo.py`,
`encoder.py`, `hopfield.py`, `vectorhash.py`, `train.py`, `train_phased*.py`.

The baseline's BC rollout/update mirrors the trajectory-level minibatched
shape of `bc.py:bc_update`, but without the store-head BCE term and without
any Hopfield/VectorHash plumbing — it only consumes `obs`, the teacher move
label, and a mask.

## Per-step pipeline

`collect_rollout_rnn` (DAgger) does this on each of `steps_per_rollout` ticks
across `B = batch_envs` parallel envs:

1. Read `sensory = vec.obs_batch()` and `positions = vec.positions()`.
2. Compute teacher action via `oracle_bfs` (greedy Manhattan, ties broken
   uniformly; `bfs_action_batch_continuous` returns a unit (dx, dy)).
3. Build the agent input with `_build_rnn_input(sensory, prev_action_oh,
   prev_reward, cfg)` — concat optional channels then `(B, 1, D)`-shape.
4. `agent.act(x, h)` samples a student action; advance hidden state `h`.
5. Mask = 0 for any env whose pre-step position is the goal or that already
   finished its episode (so at-goal supervision never enters the loss).
6. Step the env with the student action (`teacher_force=True` swaps in the
   oracle action — used only for an oracle-sanity smoke test).
7. Newly at-goal envs are **frozen** for the rest of the rollout (no
   teleport, no further transitions). Each per-env trajectory is a single
   navigation episode.

The collector returns an `RNNRolloutBatch` of `(B, T, …)` tensors. `bc_rnn_update`
concatenates a list of these, runs `cfg.epochs` passes of minibatched
trajectory-level CE (Categorical for discrete, Normal for continuous, with the
log-prob summed over the 2 axes), masked by `move_label_mask`, with optional
entropy bonus and grad-norm clipping.

`evaluate_nav_one_env` runs `n_trials` parallel trials from random starts
under deterministic acting; success = pre-step position equals goal at any
point within `max_steps`. On goal-reach the per-trial RNN hidden state is
**zeroed** so a stale memory from a finished trial doesn't leak into the
remaining (unused) ticks. Reports `nav_det`, `mean_steps_to_goal`,
`mean_episode_return`.

## Training modes (`train_rnn.train`)

Selected by `--mode`:

- **`sequential`** — continual-learning protocol. Train env 0 for
  `updates_per_env` updates, then env 1, …, then env `n_envs-1`. After every
  update, eval `nav_det` on every env trained so far; the per-update history
  is written to `history["trace"]`, with `history["blocks"]` recording
  `(start, end, env_idx)` ranges. This is what produces the "forgetting
  curve" plot.
- **`mixed`** — pretraining. Pool one rollout per env each update for
  `n_updates` updates; eval all envs at `eval_every` cadence. No forgetting
  signal.
- **`finetune`** — load `--load_checkpoint`, then run `sequential`. The
  optimizer state is **not** restored (fresh Adam moments), unlike `mixed`
  which keeps optimizer momentum if present in the checkpoint.

Each `GridEnv` is built from an independent seed, so it has its own sensory
codebook **and** its own goal — i.e. envs differ in both observation
distribution and target. The agent only ever sees the codebook output; the
goal is never input. Only the BFS teacher knows the goal.

After training, `train_rnn` saves `final.pt` (agent state, optimizer state,
config, full history, and the goal of each env) plus
`forgetting.png` / `steps_to_goal.png` (sequential / finetune modes only).

## Quickstart

Sequential continual baseline (4 envs, 100 updates each):

```bash
python -m hopfield_nav.train_rnn \
    --mode sequential --n_envs 4 --updates_per_env 100 \
    --size 8 --observation_size 60 --movement_mode discrete \
    --hidden_size 128 --lr 1e-3 --epochs 4 --n_minibatches 4 \
    --batch_envs 16 --steps_per_rollout 64 \
    --n_eval_trials 32 --eval_max_steps 64 --eval_every 25 \
    --seed 0 --device cuda
```

Mixed pretraining → sequential finetune (the `b1` recipe used for prior
runs):

```bash
# 1) pretrain on a pool of envs
python -m hopfield_nav.train_rnn --mode mixed --n_envs 4 --n_updates 1000 \
    --save_dir checkpoint_rnn/pretrain_b1

can also use hopfield_nav/pretrain_baseline_rnn.sh

# 2) finetune sequentially on a fresh set of envs from a different seed
python -m hopfield_nav.train_rnn --mode finetune --seed 7 \
    --n_envs 4 --updates_per_env 100 \
    --load_checkpoint checkpoint_rnn/pretrain_b1/final.pt \
    --save_dir checkpoint_rnn/finetune_b1
```

Re-render forgetting plots from a saved run with smoothing:

```bash
python -m hopfield_nav.regen_plots checkpoint_rnn/finetune_b1 --smooth 5
```
