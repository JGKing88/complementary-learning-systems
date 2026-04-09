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

| Config | Effect |
|---|---|
| `init_mode=pre_stored` | Goal patterns loaded into Hopfield at start. Agent gets directional signal from step 1. |
| `init_mode=empty` | Hopfield starts blank. Agent must explore and learn to store useful patterns. |
| `agent_can_store=True` | Agent's store action modifies Hopfield. Each parallel episode gets its own Hopfield copy. |
| `agent_can_store=False` | Hopfield is read-only. Shared across batch. |
| `input_encoded_state=False` | Don't pass the 256-dim embedding to the RNN. Only pass reward + hopfield signal. Simpler, trains faster. |
| `hopfield_mode=discrete` | Agent sees a 4-way one-hot direction. Movement is categorical. |
| `hopfield_mode=continuous` | Agent sees a normalized 2D direction vector. Movement is Gaussian. |

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

All three evaluation methods are **independent of the training setup**. They can be run on any trained agent regardless of how it was trained (pre_stored vs empty, discrete vs continuous, etc.). Each uses the shared `_agent_step` helper which runs the full pipeline: position → reward → embedding → Hopfield recall → Gram-Schmidt → agent forward → step.

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
- Agent uses **stochastic** actions (not deterministic) with store enabled

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
- Agent uses **stochastic** actions with store enabled

**Per trial:**
- Random start, fresh RNN, fresh Hopfield (with distractors only)
- Runs for the **full** `max_steps` — no early termination, even if goal is found
- Track all unique snapped grid positions visited
- Track whether/when the agent first reaches the goal
- Agent can store if it wants (store actions modify the Hopfield)

**Metrics (per distractor count):**
- `mean_coverage` — average fraction of grid cells visited (1.0 = visited every cell)
- `goal_find_rate` — fraction of trials where the agent reached the goal at least once
- `mean_steps_to_goal` — average steps to first goal reach (only over trials that found it)

**What it tells you:** Whether the agent explores efficiently (spreads out to cover the grid) vs getting stuck or circling. The distractor sweep tests whether Hopfield memories from other envs bias the agent's exploration — do they attract it to certain areas, or does it ignore them?

### Relationship between Eval 2 and Eval 3

Eval 2 and Eval 3 test complementary aspects of the same scenario (agent in a new env without its goal stored):

- **Eval 3** measures how well the agent *finds* the goal through exploration
- **Eval 2** measures how well the agent *stores* the goal once found
- **`store_efficiency`** (from Eval 2) bridges the two: it's the conditional probability of storing given finding

A good agent should have:
- High coverage in Eval 3 (explores broadly)
- High reach_success_rate in Eval 2 (exploration leads to goal)
- High store_efficiency in Eval 2 (recognizes goal and stores it)

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
