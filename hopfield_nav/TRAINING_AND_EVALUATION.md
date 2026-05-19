# Training and Evaluation Guide

## Overview

The system trains an RNN agent to navigate grid environments using a Hopfield associative memory network. The agent receives a directional signal from the Hopfield network (which stores encoded goal positions) and learns to follow it to reach goals. The agent can also learn to store new patterns in the Hopfield network.

There are two packages:

1. **encoder_training** — trains a grid cell encoder that maps grid one-hot codes to embeddings on the unit sphere. Run this first to produce an encoder checkpoint.
2. **hopfield_nav** — trains the navigation agent using a pretrained encoder. The encoder is frozen during navigation training.

---

## Step 1: Encoder Training

The encoder learns a distance-preserving mapping from grid cell activations to a compact embedding space. Nearby grid positions should have similar embeddings; far positions should have dissimilar embeddings.

### What it trains

A neural network (MLP or CNN) that takes a flattened grid one-hot vector and outputs an L2-normalized embedding on the unit sphere.

### How it trains

- Generate grid codes for all positions in the Npos x Npos grid
- Optionally smooth grid codes with per-module wrapped Gaussians (controlled by `fwhm_ratio`)
- For each batch: encode positions, compute cosine similarity kernel between embeddings, compare to RBF target kernel from spatial coordinates using kernel alignment (CKA) loss
- Optional regularizers: uniformity (spread embeddings evenly on sphere), coplanarity (straight paths map to great circles)
- Gain annealing: the output nonlinearity scaling ramps from `gain_start` to `gain_end` over training

### Running it

```bash
python -m encoder_training.train \
    --encoder_type mlp \
    --lambdas 11 12 13 \
    --out_dim 256 \
    --hidden_dim 1024 \
    --epochs 300 \
    --batch_size 4096 \
    --loss_mode cka \
    --save_dir encoders
```

### Output

Checkpoints in `encoders/<run_name>/encoder_final.pt` containing the model state dict and config. These are loaded by `hopfield_nav`.

---

## Step 2: Navigation Training

### What it trains

An RNN (GRU) policy with three output heads:

- **Movement**: which direction to move (discrete: N/E/S/W categorical; continuous: (dx, dy) Gaussian)
- **Store**: whether to store the current position's encoding in the Hopfield network (binary)
- **Value**: estimated future return (for PPO)

### How it trains

**Setup:**

1. Load the frozen pretrained encoder
2. For each "world": create train + val GridEnvs, build a VectorHash scaffold, precompute `encoded_Phi` (the encoder applied to every grid position)
3. If `init_mode=pre_stored`: create a template Hopfield with goal patterns pre-stored

**Per training update:**

1. For each world, create fresh Hopfield instances (one per parallel episode if `agent_can_store=True`)
2. For each training env in the world, collect a rollout of `steps_per_rollout` steps across `batch_envs` parallel episodes

**Per rollout step (the core loop):**

1. Get the agent's current snapped grid position
2. Compute the **current reward** from this position (1.0 if at goal, -time_penalty otherwise). This is fed to the agent *before* it acts, so the agent knows when it's at the goal.
3. Look up the position's embedding from `encoded_Phi`
4. If Hopfield has stored memories: recall from the embedding, project the displacement through a Gram-Schmidt local coordinate frame to get a 2D direction signal. If no memories: zero signal.
  - Discrete mode: classify the 2D signal to a 4-way one-hot (N/E/S/W)
  - Continuous mode: normalize to unit direction vector
5. Build RNN input: `[current_reward, embedding (optional), hopfield_signal (optional)]`
6. Agent forward pass → movement action, store action, value estimate
7. If store action fires: store the current embedding in this episode's Hopfield
8. Step the environment with the movement action
9. If goal reached: auto-teleport to random position, reset RNN hidden state (Hopfield persists)

**After collecting rollouts:**

- Run PPO update: compute GAE advantages (no terminal states within rollout, only truncation bootstrap), clipped policy loss for both movement and store actions, value loss, entropy bonuses
- Movement and store share the same advantages — storing is rewarded indirectly through its effect on future navigation success

**Between environments within a world:** RNN hidden state resets, Hopfield persists (memories from env 1 are available in env 2).

**Between worlds:** everything resets.

### Key training configs


| Config                      | Effect                                                                                                                                                                                                                                                                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `init_mode=pre_stored`      | Goal patterns loaded into Hopfield at start. Agent gets directional signal from step 1.                                                                                                                                                                                                                                             |
| `init_mode=empty`           | Hopfield starts blank. Agent must explore and learn to store useful patterns.                                                                                                                                                                                                                                                       |
| `agent_can_store=True`      | Agent's store action modifies Hopfield. Each parallel episode gets its own Hopfield copy.                                                                                                                                                                                                                                           |
| `agent_can_store=False`     | Hopfield is read-only. Shared across batch.                                                                                                                                                                                                                                                                                         |
| `input_encoded_state=False` | Don't pass the 256-dim embedding to the RNN. Only pass reward + hopfield signal. Simpler, trains faster.                                                                                                                                                                                                                            |
| `hopfield_mode=discrete`    | Agent sees a 4-way one-hot direction. Movement is categorical.                                                                                                                                                                                                                                                                      |
| `hopfield_mode=continuous`  | Agent sees a normalized 2D direction vector. Movement is Gaussian.                                                                                                                                                                                                                                                                  |
| `store_cost=<float>`        | Reward penalty per fired store action (metabolic cost). Pushes the store head toward firing only when it pays off.                                                                                                                                                                                                                  |
| `store_bonus=<float>`       | Reward bonus added when the agent fires store while at the goal. Shapes the store head toward the desired "store-at-goal" behavior.                                                                                                                                                                                                 |
| `store_bc_weight=<float>`   | Auxiliary BCE loss on the store head with `at_goal` as the label (pos-weighted for class imbalance). Direct supervision for when to store, independent of PPO rewards.                                                                                                                                                              |
| `auto_store_warmup=N`       | For the first N updates, force a store on every at-goal step regardless of the agent's store action. Seeds the Hopfield with goal patterns early so the rest of the system has a signal to learn from.                                                                                                                              |
| `auto_nav_warmup=N`         | For the first N updates, override the agent's movement with the Hopfield-suggested direction in any env that has a stored memory (teacher forcing). PPO re-scores the log-prob of the forced action, so the policy is trained to imitate the Hopfield-following behavior while the advantage shaping still filters bad suggestions. |
| `aux_anneal_updates=N`      | Linearly decay `store_bonus` and `store_bc_weight` from full → 0 over N updates. Tests whether the learned store behavior persists after the shaping signal fades.                                                                                                                                                                  |
| `novelty_reward=<float>`    | +reward on first visit to a snapped cell during the explore phase (tracked per rollout). Encourages exploration beyond the policy's entropy bonus.                                                                                                                                                                                  |


### Running it

```bash
python -m hopfield_nav.train \
    --encoder_checkpoint encoders/confused-sweep-160/encoder_final.pt \
    --encoder_gain 3.0 \
    --size 8 \
    --lambdas 11 12 13 \
    --Npos 200 \
    --hopfield_init pre_stored \
    --no-agent_can_store \
    --hopfield_mode discrete \
    --no-input_encoded_state \
    --n_updates 500 \
    --device cpu
```

---

## Evaluation

Four evaluation methods, all **independent of the training setup** — they can be run on any trained agent regardless of how it was trained (pre_stored vs empty, discrete vs continuous, etc.). Each uses the shared `_agent_step` helper which runs the full pipeline: position → reward → embedding → Hopfield recall → Gram-Schmidt → agent forward → step.

Evals 1–3 (navigation, goal discovery, exploration) run periodically during training (`--eval_every`) with a **fresh Hopfield per trial** so memory state doesn't leak between trials or envs. They use the dedicated eval world (built once at startup, decoupled from training worlds) with val envs spread across the scaffold on a jittered lattice.

Eval 4 (realistic) runs **only at end of training** with a **single persistent Hopfield** accumulating memories across all envs.

**`hopfield_oracle` (disambiguation mode):** When enabled (`--hopfield_oracle` in `train` / `eval_all`, or `--hopfield-oracle` in `eval_all.py`), the Hopfield *readout* is **not** used whenever the “goal in memory” conditions are met. Instead, the 2D signal is the same projection as in training, but the displacement in embedding space is **goal embedding − current embedding** (a straight “toward the goal” cue in the local Gram—Schmidt frame). That isolates *policy* mistakes from *associative recall* mistakes. **`input_hopfield_signal` must be true** in the agent config (and the checkpoint must be trained with a hopfield input slot); otherwise the policy still receives an all-zero hopfield vector and the oracle is ineffective — `eval_all` logs `hopfield_oracle_effective` in the JSON and prints a warning. `goal_in_memory` is always true for **Eval 1 (navigation)**, where the goal is preloaded. For **Evals 2, 3, 4** it flips to true only after a successful store-while-at-goal for that env; **realistic** keeps a per-env flag across the shared Hopfield (retest phases inherit whether that env’s goal was ever written). Disable with `--no-hopfield-oracle` in `eval_all` to force off regardless of the checkpoint.

### Eval 1: Navigation (`evaluate_navigation`)

**Question:** Can the agent follow a pre-stored Hopfield signal to reach the goal?

**Setup:**

- Build a shared Hopfield with the val env goals pre-stored (agent store disabled)
- For each val env, run `num_trials` independent trials

**Per trial:**

- Random start position, fresh RNN hidden state
- Up to `max_steps` steps with **deterministic** actions (argmax for discrete, mean for continuous)
- Success = snapped position equals goal
- Trial ends immediately on goal reach (no teleportation)

**Metrics:**

- `success_rate` — fraction of trials that reached the goal
- `mean_speed` — average of (Manhattan distance / steps taken) for successful trials. 1.0 = optimal (no wasted steps). >1.0 possible for continuous movement taking diagonal shortcuts.
- `mean_steps` — average steps to reach goal for successful trials
- `total_trials`, `total_successes` — raw counts

**What it tells you:** Whether the trained policy can interpret the Hopfield directional signal and navigate efficiently. This is the basic competence test. If a correct goal encoding is in the Hopfield, can the agent follow it?

### Eval 2: Goal Discovery (`evaluate_goal_discovery`)

**Question:** When the agent encounters the goal through exploration, does it store the goal encoding in the Hopfield network?

**Setup:**

- The test env's goal is NOT in the Hopfield
- The Hopfield is pre-loaded with N "distractor" goal patterns from other regions of the grid (not overlapping with the test env). Sweep N = 0, 1, 3, 5, 10.
- Agent uses **deterministic** policy actions (argmax / mean, store if prob > 0.5) with store enabled; set `deterministic=False` in code to sample instead

**Per trial:**

- Random start, fresh RNN, fresh Hopfield (with distractors only)
- Up to `max_steps` steps
- The agent walks around freely — no teleportation on goal reach
- If the agent is at the goal AND fires store → success, trial ends
- If the agent is at the goal but doesn't fire store → nothing special happens, agent keeps walking. It may revisit the goal later and get another chance to store.
- If the agent fires store while NOT at the goal → the non-goal embedding is stored in the Hopfield (pollution), but it's not counted as success
- `reached_goal` is set to True if the agent was ever at the goal position during the trial

**Metrics (per distractor count):**

- `store_success_rate` — fraction of trials where the agent stored while at the goal
- `reach_success_rate` — fraction of trials where the agent reached the goal at least once
- `store_efficiency` — `store_success_rate / reach_success_rate`. Probability of storing *given* that the agent found the goal. Separates "can explore" from "knows when to store".
- `mean_steps_to_store` — average steps until successful store (only over successful trials)
- `mean_steps_all` — average steps across all trials (max_steps for failures)

**What it tells you:** Whether the agent has learned to recognize the goal (via the reward=+1.0 signal) and respond by storing. The distractor sweep tests whether having other goals in the Hopfield helps (attracts the agent toward goal-like positions) or hurts (confuses the recall signal).

### Eval 3: Exploration Efficiency (`evaluate_exploration`)

**Question:** How well does the agent explore the grid when it doesn't have a goal stored?

**Setup:**

- Same as goal discovery: test env's goal NOT in Hopfield, N distractor goals from other regions
- Agent uses **deterministic** policy actions (same as Eval 2; optional stochastic via `deterministic=False` in code)

**Per trial:**

- Random start, fresh RNN, fresh Hopfield (with distractors only)
- Runs for the **full** `max_steps` — no early termination, even if goal is found
- Track all unique snapped grid positions visited
- Track whether/when the agent first reaches the goal
- Agent can store or not (it's a flag, default false)

**Metrics (per distractor count):**

- `mean_coverage` — average fraction of grid cells visited (1.0 = visited every cell)
- `goal_find_rate` — fraction of trials where the agent reached the goal at least once
- `mean_steps_to_goal` — average steps to first goal reach (only over trials that found it)

**What it tells you:** Whether the agent explores efficiently (spreads out to cover the grid) vs getting stuck or circling. The distractor sweep tests whether Hopfield memories from other envs bias the agent's exploration — do they attract it to certain areas, or does it ignore them?

### Relationship between Eval 2 and Eval 3

Eval 2 and Eval 3 test complementary aspects of the same scenario (agent in a new env without its goal stored):

- **Eval 3** measures how well the agent *finds* the goal through exploration
- **Eval 2** measures how well the agent *stores* the goal once found
- `**store_efficiency`** (from Eval 2) bridges the two: it's the conditional probability of storing given finding

A good agent should have:

- High coverage in Eval 3 (explores broadly)
- High reach_success_rate in Eval 2 (exploration leads to goal)
- High store_efficiency in Eval 2 (recognizes goal and stores it)

### Eval 4: Realistic / Catastrophic Interference (`evaluate_realistic`)

**Question:** When a single Hopfield memory accumulates goals from many envs visited one-after-another, does the agent still navigate to earlier envs' goals, or do later memories overwrite them?

Only runs at the **end of training** (not during periodic eval). Logged to wandb under the `realistic/` namespace (separate from `train/` and `eval/`).

**Setup:**

- Reuses the dedicated eval world (same `num_val_envs` spread across the scaffold)
- Creates **one** empty Hopfield that persists for the entire eval — it is **never reset** between envs or between phases
- Agent weights frozen, `deterministic=True` (argmax/mean actions)

**Protocol:**
For each val env `i` in order `0..N-1`:

1. **Primary phase**: random start, fresh RNN, run `realistic_steps_per_env` steps (default 1000) with the agent's own store head driving Hopfield writes. On every goal-reach: teleport to a random non-goal cell and reset RNN. No early termination; agent can accumulate multiple reaches per phase.
2. **Retest phase** (for each prior env `j < i`): same mechanics as primary but with **storing disabled** — the store head is ignored, Hopfield is frozen during retests. Fresh random start + fresh RNN per retest.

This runs quadratically in `N`: primary is `N · B` steps, retests total `N(N-1)/2 · B` steps.

**Metrics per phase:**

- `n_reaches` — total goal reaches in this phase
- `intervals` — list of step-counts between each start/teleport and the next reach (trailing partial intervals with no reach are discarded)
- `mean_interval` — mean of `intervals`

**Aggregated output:**

- `primary[i]` — primary-phase metrics for env `i`
- `retest[(i, j)]` — metrics for retesting env `j` after visiting env `i` (only `j < i`)
- `drift[j]` — list of `(gap, metrics)` for env `j`: gap=0 is primary, gap=k>0 is the retest that ran after visiting env `j+k`. Used to plot reach-count decay vs "envs visited since this one."
- `summary`:
  - `mean_primary_reaches` — average `n_reaches` across envs in their primary phase
  - `mean_final_retest_reaches` — average `n_reaches` at the final retest (i.e. after all envs have been visited), over envs `0..N-2`
  - `interference_drop` — mean fractional drop `(primary - final_retest) / primary`. Main catastrophic-interference scalar.
  - `hopfield_final_memories` — total patterns stored in the Hopfield at the end

**Key flags:**


| Flag                                       | Effect                                                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `realistic_steps_per_env=N`                | Steps per env per phase (both primary and retest). Default 1000. Set 0 to skip.                  |
| `realistic_seed_offset` (in `eval_all.py`) | Added to `cfg.seed` for the per-eval RNG. Default 1000. Vary it to measure across-seed variance. |


**What it tells you:** Whether the trained agent's memory system scales beyond one env — the hardest version of the task. High `mean_primary_reaches` with a low `interference_drop` means the agent learns a good policy AND its Hopfield memory survives contamination from later envs. Low primary reaches mean the policy itself is bad; low retest with high primary means the memory is being overwritten.

### Eval 5: Repeat (`evaluate_repeat`)

**Question:** Ignoring interference, how reliably does the agent **cold-start** in an env — find the goal, store it, then exploit the store to return — when Hopfield and RNN are fully reset between trials?

Runs **only on demand** from `eval_all.py` (set `--repeat-trials > 0`). Not run during training.

**Setup:** reuses the dedicated eval world. For each val env, runs `n_trials` independent trials with a **fresh Hopfield** and **fresh RNN** each trial. Agent weights frozen, `deterministic=True`.

**Per trial:**

- Random non-goal start, fresh RNN, fresh empty Hopfield
- Run `steps_per_env` steps with the agent's own store head enabled. On every goal-reach: teleport + reset RNN + mark that env's `goal_in_memory=True`. Interval data recorded exactly like the primary phase of the realistic eval (including an open ring on reaches where a store fired, and a cut-off square if the trial ends mid-trajectory).

**Output:** per-env list of trial dicts (`intervals`, `stored_at_reach`, `tail_steps`, `start`, `trial_idx`) and a per-trial intervals plot (`*_repeat_intervals.png`). One scalar summary `mean_reaches = mean over (env, trial) of n_reaches`.

**What it tells you:** the policy's cold-start competence, independent of catastrophic interference. Use it together with the realistic eval: a big gap between repeat-eval `mean_reaches` and realistic-eval `mean_primary_reaches` points at interference; a low repeat-eval score points at policy weakness or insufficient training.

### Eval 6: Sequential continual (`evaluate_sequential_episodes`)

**Question:** Under a continual-learning protocol — introduce envs one at a time, then revisit **every** previously introduced env on **every** iteration — can the agent keep solving old envs as new ones are added to the shared Hopfield?

Runs **only on demand** from `eval_all.py` (set `--seq-iters-per-block > 0`). Not run during training.

**Setup:** reuses the dedicated eval world with `N = cfg.num_val_envs` envs. A **single persistent Hopfield** is built once and never reset. Agent weights frozen, `deterministic=True`.

**Protocol:** for each env `i = 0..N-1` (block `i`):

- For each outer iteration `k = 0..iters_per_block-1`:
  - For **every already-introduced env** `j ≤ i`:
    - random non-goal start, fresh RNN, run up to `max_steps` steps;
    - `j == i` (primary) — agent's store head may write into the shared Hopfield;
    - `j < i` (revisit) — stores disabled, Hopfield frozen for that mini-episode;
    - record **one 0/1 bit**: `reached = True` if the agent touched the goal during the episode.

This produces, for each env `j`, a sequence `env_iters[j] = [(outer_iter, 0/1), ...]` covering the outer iterations in which env `j` is active (its primary block + all later blocks as a revisit). In total there are `∑_{i=0..N-1} (i+1) * iters_per_block` mini-episodes, `≤ max_steps` steps each.

**Plot (`*_sequential.png`):** one colored line per env, y = trailing moving-average of the 0/1 success bit over `ma_window` iterations, x = outer iteration. Dashed verticals mark block boundaries; the env introduced in each block is labeled below the x-axis. This reproduces the figure style from the reference paper.

**Metrics (in `summary`):**

- `per_env_primary_success[j]` — mean success during env `j`'s primary block
- `per_env_final_revisit_success[j]` — mean success of env `j` in the last block (revisit after all envs introduced), `NaN` for `j == N-1`
- `mean_primary_success`, `mean_final_revisit_success`
- `interference_drop = mean_primary_success - mean_final_revisit_success`
- `hopfield_final_memories`
- `stored_at_goal_count[j]` — times the agent fired store while at env `j`'s goal during its primary block

**Key flags:**


| Flag                    | Effect                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------- |
| `--seq-iters-per-block` | Iterations per env block. Default 0 (skip). `SEQ_ITERS_PER_BLOCK` in the shell.       |
| `--seq-max-steps`       | Max steps per mini-episode. Default 32. `SEQ_MAX_STEPS`.                              |
| `--seq-ma-window`       | Moving-average window for the plot (plot-only; does not affect stored data). Def. 20. |
| `--seq-seed-offset`     | Added to `cfg.seed` for the sequential-eval RNG. Default 3000.                        |


**What it tells you:** same question as the realistic eval (does the Hopfield survive adding new envs?) but with the paper's episodic success rate metric instead of teleport-based reach counts, and with **every** previous env retested at **every** outer iteration instead of once per new env. That gives a dense moving-average line per env that cleanly shows when, and by how much, each env's performance degrades as new memories are added.

### Running eval on a batch of checkpoints

`hopfield_nav/run_eval_all.sh` + `hopfield_nav/eval_all.py` evaluate a list of checkpoints on all four eval types. Fill in checkpoint paths in `CKPTS=(...)` and submit. Per-checkpoint JSON results and drift plots are written to `eval_results/<timestamp>/`. Override eval parameters via env vars, e.g. `NUM_TRIALS=16 REALISTIC_STEPS=2000 sbatch hopfield_nav/run_eval_all.sh`.

---

## SLURM Submission

Example scripts are in `hopfield_nav/run_*.sh`. Key settings:

```bash
#SBATCH --partition=pi_fiete        # or pi_evelina9, mit_normal_gpu
#SBATCH --mem=16G                    # 16G sufficient for Npos=200
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00           # 2 hours for 500 updates
```

For GPU: uncomment `--gres=gpu:1` and set `--device cuda`. For CPU-only: comment out gres and set `--device cpu`.

Memory requirements scale with Npos^2. With Npos=200 and Np=1600: ~2GB for scaffold. With Npos=1716 (full product of [11,12,13]): ~40GB — will OOM on most allocations. Always set `--Npos` to a manageable value (50-200).