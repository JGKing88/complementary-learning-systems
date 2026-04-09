# Hopfield Navigation Experiments

## Encoder

All experiments use encoder `confused-sweep-160`:
- Type: MLP, out_dim=256, lambdas=[11,12,13]
- Gain: 3.0, fwhm_ratio=0.25
- Checkpoint: `encoders/confused-sweep-160/encoder_final.pt`

---

## Experiment Log

### Exp 1: Pre-stored baseline (discrete)

**Goal:** Verify agent can follow pre-stored Hopfield signal to goals.

| Param | Value |
|---|---|
| movement_mode | discrete |
| hopfield_init | pre_stored |
| agent_can_store | False |
| input_encoded_state | False |
| steps_per_rollout | 32 |
| n_updates | 500 |
| lr | 3e-4 |
| seed | 42 |

**Result:** 100% success rate by update 100. Mean speed ~0.85-0.95. Stable throughout training.

**Takeaway:** Discrete navigation with pre-stored Hopfield signal works perfectly. The 5-dim input (1 reward + 4 one-hot direction) is sufficient.

---

### Exp 2: Pre-stored baseline (continuous)

**Goal:** Same as Exp 1 but with continuous movement.

| Param | Value |
|---|---|
| movement_mode | continuous |
| hopfield_init | pre_stored |
| agent_can_store | False |
| input_encoded_state | False |
| steps_per_rollout | 32 |
| n_updates | 500 |
| lr | 3e-4 |
| seed | 42 |

**Result:** 100% success rate by update 100. Continuous movement works after normalizing Hopfield signal to unit direction before agent input.

**Takeaway:** Continuous mode matches discrete. Signal normalization is critical — without it, variable magnitude confuses the policy.

**Checkpoint:** `checkpoints/hopfield_nav_update500.pt` (continuous)

---

### Exp 3/4: Curriculum — learn to store (discrete + continuous)

**Goal:** Can a nav-competent agent learn when to fire the store action?

**Approach:** Load pre-stored checkpoint, fine-tune with empty Hopfield + agent_can_store.

| Param | Value |
|---|---|
| hopfield_init | empty |
| agent_can_store | True |
| input_encoded_state | False |
| steps_per_rollout | 128 |
| n_updates | 1000 |
| lr | 1e-4 |
| store_cost | 0.02 (bug: CLI default overrode intended 0.0 for no-penalty run) |

**SLURM:** 11558462 (intended no-penalty), 11558745 (intended penalty) — both ran with store_cost=0.02 due to CLI default bug.

**Result (discrete, Phase 2):**
- Run A: 100% → drops to 40% by update 250, fluctuates 20-65%
- Run B: 100% → holds 100% for 200 updates → degrades to 9-30%
- store_entropy stays ~0.5-0.6 (near random) — agent never learns when to store

**Result (continuous, Phase 3):**
- Both identical (same checkpoint, same effective config)
- 93% → 100% → collapses to 12-50% by update 350

**Takeaway:** 0.02 store cost is not enough. Random stores corrupt Hopfield. Agent can't learn selective store timing from PPO alone.

---

### Exp 5: High store cost sweep (continuous)

**Goal:** Does a higher metabolic penalty prevent Hopfield corruption?

| store_cost | SLURM |
|---|---|
| 0.1, 0.2, 0.5 | 11561310 |

All load continuous pre-stored checkpoint, same params as Exp 3/4 otherwise.

**Results:**

| Cost | Early peak | Late performance | store_entropy |
|---|---|---|---|
| 0.1 | 100% (update 200-250) | Collapses to 3-6% | drops to ~0 |
| 0.2 | 100% (update 200) | Degrades to 12-37% | drops to ~0 |
| **0.5** | **90.6% (update 350)** | **Holds 50-87%** | **drops to 0.0004** |

**Takeaway:** Higher cost prevents collapse but agent learns to **never store** instead. store_entropy → 0 means the agent suppresses all stores. The 50-70% success is just random exploration, not Hopfield-guided. The penalty shifts behavior from "store everywhere → corrupt" to "store nowhere → wander." Neither solves the credit assignment problem.

---

### Exp 6: Two-phase rollout, store_cost=0.1 (continuous)

**Goal:** Separate explore and exploit within one rollout to improve credit assignment for store action.

**Approach:** First `explore_steps` of rollout: agent can store, Hopfield is live. Remaining steps: Hopfield frozen, no more stores. GAE runs over the full trajectory so exploit rewards propagate back to explore store decisions.

| Param | Value |
|---|---|
| load_checkpoint | checkpoints/hopfield_nav_update500.pt |
| movement_mode | continuous |
| hopfield_init | empty |
| agent_can_store | True |
| store_cost | 0.1 |
| steps_per_rollout | 128 |
| explore_steps | 64, 32 (sweep) |
| n_updates | 1000 |
| lr | 1e-4 |

**Script:** `hopfield_nav/run_two_phase.sh`
**SLURM:** 11563871

**Results:**

explore=64, cost=0.1:

| Update | Success | store_entropy |
|---|---|---|
| 100 | 87.5% | 0.33 |
| 150-250 | **100%** | 0.20-0.24 |
| 300-400 | 84-90% | 0.15-0.17 |
| 500 | 75% | 0.21 |
| 700+ | 37-50% | 0.04 → 0.003 |

explore=32, cost=0.1:

| Update | Success | store_entropy |
|---|---|---|
| 200-250 | **100%** | ~0.20 |
| 300-450 | 62-87% | dropping |
| 600+ | 6-21% | → 0 (collapsed) |

**Takeaway:** Two-phase is the best structure so far — holds 100% for several checkpoints. But store_cost=0.1 still drives store_entropy to zero, eventually killing performance. explore=64 > explore=32 (more time to find goal). Next: try with zero or very low store cost, since the two-phase structure already limits corruption.

---

### Exp 7: Two-phase rollout, lower store cost (continuous)

**Goal:** Does removing/reducing the store cost let the two-phase agent maintain storing behavior longer?

| Config | explore | cost | SLURM |
|---|---|---|---|
| 7a | 64 | 0.0 | 11565908 |
| 7b | 64 | 0.05 | 11565908 |
| 7c | 32 | 0.0 | 11565908 |

**Script:** `hopfield_nav/run_two_phase_v2.sh`

**Results:**

| Config | 100% window | Late perf | store_entropy |
|---|---|---|---|
| 7a (e64, c=0.0) | updates 50-200 | 40-50% | **0.69 throughout** (random) |
| **7b (e64, c=0.05)** | **updates 150-200** | **28-44%** | **0.66→0.45** (moderate) |
| 7c (e32, c=0.0) | updates 50-150, 250-300 | 22-37% | **0.69 throughout** (random) |

**Takeaway:** Zero cost keeps store_entropy alive (~0.69) but agent stores randomly — still corrupts Hopfield. Light penalty (0.05) finds a middle ground where entropy doesn't collapse to zero but also doesn't stay fully random. However, **no configuration learns selective storing.** The agent either stores everywhere (no cost) or nowhere (high cost), never converging to "store only at goal."

### Cross-experiment summary: the store learning problem

| Approach | Best result | Failure mode |
|---|---|---|
| Single-phase, low cost | 100% briefly, collapses | Random stores corrupt Hopfield |
| Single-phase, high cost | 50-87% stable | Agent learns to never store |
| Two-phase, cost=0.1 | 100% for 150 updates | Cost still kills storing eventually |
| Two-phase, cost=0.0 | 100% for 200 updates | Random stores corrupt Hopfield |
| Two-phase, cost=0.05 | 100% for 100 updates | Moderate — best balance, still degrades |

**Core problem:** PPO's advantage signal is too diffuse to learn the narrow "store at goal" behavior. The store action's benefit (better Hopfield signal → better future navigation) is separated from the store decision by many timesteps even within the two-phase rollout. The agent never discovers the correlation between "store when reward=+1.0" and "better exploit performance."

---

## Notes

### What we learned about input design
- Passing the 256-dim embedding to the RNN drowns out the 4/2-dim Hopfield signal. `--no-input_encoded_state` gives much cleaner learning.
- Hopfield signal must be normalized to unit direction (continuous mode) before agent input.
- Current reward (not previous) is critical — agent needs to know it's at the goal *before* deciding to store.

### What we learned about the store action
- With untrained store policy, random stores destroy Hopfield attractors (nav success drops from 82.5% to 57.1%).
- Store action shares PPO advantages with movement — no separate reward signal needed in principle.
- The credit assignment problem is severe: PPO can't learn selective store timing in single-phase.
  - Low cost (0.02): agent stores randomly, corrupts Hopfield → nav degrades
  - High cost (0.5): agent learns to never store → no Hopfield signal → random wandering
- **Two-phase rollout helps significantly:** explore/exploit split gives the best results so far (100% for several checkpoints). But store cost still drives store_entropy → 0 eventually.
- Key pattern across all experiments: the agent either stores too much (corruption) or learns to never store (penalty avoidance). Finding the "store only at goal" middle ground is the core challenge.
- Two-phase rollout is the current approach: structurally separate explore (store allowed) from exploit (Hopfield frozen), letting exploit returns credit explore stores through GAE.

### Potential next approaches if two-phase doesn't work
1. Supervised store signal — train store head with cross-entropy on "at goal" label
2. Auxiliary reward — small bonus for storing at goal
3. Auto-store during curriculum, let agent take over store later
