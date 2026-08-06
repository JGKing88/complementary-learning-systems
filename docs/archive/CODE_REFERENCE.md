> **Archived.** Moved out of `hopfield_nav/` by phase 6 of the 2026-08
> refactor. Not maintained; describes what was believed and tried at the time,
> which in places is no longer true of the code. Start from `docs/archive/README.md`
> for what replaced it. **Superseded by `docs/CODEBASE_MAP.md`** -- the module paths and duplicated helpers described below were changed by the refactor.

# Code Reference

## Package Structure

```
hopfield_nav/
  config.py        — All dataclass configs
  hopfield.py      — Hopfield associative memory network
  encoder.py       — Pretrained encoder loader + config validation
  vectorhash.py    — Grid/place/sensory scaffold + encoded_Phi + Gram-Schmidt
  env.py           — GridEnv (discrete) + ContinuousGridEnv
  vec_env.py       — VecEnv (discrete batched) + ContinuousVecEnv (float batched)
  agent.py         — NavAgent: GRU policy with movement/store/value heads
  rollout.py       — RolloutCollector: per-step pipeline orchestration
  ppo.py           — GAE + PPO clipped update
  train.py         — Main training loop + CLI
  eval.py          — Three evaluation methods
  utils.py         — Gram-Schmidt, direction classification, smoothing

encoder_training/
  config.py        — Encoder + training configs
  models.py        — GridEncoder (MLP) + GridEncoderCNN
  losses.py        — CKA, uniformity, local-attract-far-repel, coplanarity
  data.py          — Grid code generation, patch sampling, RBF kernels
  train.py         — Encoder training loop + CLI
  evaluate.py      — Encoder quality metrics
  utils.py         — Grid smoothing (onehot-to-Gaussian)
```

## hopfield_nav

### config.py

Six nested dataclasses:

| Dataclass | Key fields |
|---|---|
| `EnvConfig` | `size`, `observation_size`, `time_penalty`, `movement_mode` ("discrete"/"continuous"), `continuous_scale` |
| `VectorHashConfig` | `lambdas` (grid module periods), `Np` (place cells), `Npos` (grid size override) |
| `HopfieldConfig` | `beta`, `alpha`, `steps`, `init_mode` ("empty"/"pre_stored"), `agent_can_store` |
| `AgentConfig` | `hidden_size`, `num_rnn_layers`, `input_encoded_state`, `input_hopfield_signal`, `hopfield_mode`, `movement_mode` |
| `PPOConfig` | `lr`, `gamma`, `gae_lambda`, `clip_coef`, `vf_coef`, `ent_coef`, `store_ent_coef`, `ppo_epochs` |
| `TrainConfig` | Nests all above + `encoder_checkpoint`, `num_worlds`, `envs_per_world`, `batch_envs`, `steps_per_rollout`, `n_updates` |

### hopfield.py — `Hopfield`

| Method | Description |
|---|---|
| `input_memory(z)` | Store pattern via Hebbian learning: `W += scale * z z^T` |
| `recall(x0, steps, beta, alpha)` | Iterative recall: `x = (1-a)x + a*tanh(b*W@x)`, L2-normalized each step |
| `recall_batch(x0_batch, ...)` | Batched recall (only when W is shared/read-only across batch) |
| `clone()` | Deep copy with independent W matrix |
| `reset()` | Zero W and memory count |
| `energy(x)` | Hopfield energy: `-0.5 x^T W x` |

### encoder.py

- `load_encoder(path, device, gain_override)` — Load pretrained encoder from checkpoint. Supports old (nested `model_params`) and new (flat) formats. Returns frozen model + config + gain.
- `validate_config(encoder_cfg, vectorhash_lambdas, ...)` — Fail-fast check that encoder lambdas match VectorHash.

### vectorhash.py — `VectorHash`

Lifecycle: `__init__` → `build_scaffold()` → `register_envs(envs)` → `precompute_encoded_phi(encoder)`

| Method | Description |
|---|---|
| `build_scaffold()` | Generate gbook, pbook, train Wpg/Wgp weight matrices |
| `register_envs(envs)` | Place envs at non-overlapping grid offsets, train Wsp/Wps, validate scaffold |
| `recall(obs)` / `recall_batch(obs_batch)` | Sensory → place → grid recall chain (no second threshold) |
| `precompute_encoded_phi(encoder, fwhm_ratio, device)` | Encode all grid positions → `encoded_Phi[gx, gy]` array |
| `get_encoded_state(positions, env_offset)` | Look up encoded_Phi for local positions |
| `gram_schmidt_projection(positions, env_offset)` | Compute local 2D basis from encoded_Phi neighbors → `(B, 2, embed_dim)` |
| `project_displacement(current, recalled, W)` | Project `(recalled - current)` through W → `(B, 2)` |
| `get_goal_encodings(envs)` | Get encoded goal patterns for a list of environments |

**`env_offsets`**: list of `(C_X, C_Y)` tuples mapping each registered env to its position in the Npos x Npos grid. Indices correspond to the order passed to `register_envs(all_envs)`.

### env.py

- **`GridEnv`** — Discrete grid. Integer positions, cardinal actions `(N/E/S/W)`, binary codebook observations. Goal stays fixed; `reset()` teleports to random start.
- **`ContinuousGridEnv(GridEnv)`** — Float positions, `(dx, dy)` actions, snaps to grid for obs lookup.

### vec_env.py

- **`VecEnv`** — Batched discrete env. `_pos: (B, 2) int32`. `step_batch(action_indices)` auto-teleports on goal reach.
- **`ContinuousVecEnv`** — Batched continuous env. `_pos_f: (B, 2) float64` (source of truth), `_pos: (B, 2) int32` (snapped, updated via `_update_snapped()`). `positions()` returns `_pos`. Goal check uses snapped position.

### agent.py — `NavAgent(nn.Module)`

Three-headed GRU policy:
- **Movement head**: `Categorical(4)` for discrete, `Normal(2)` for continuous
- **Store head**: `Bernoulli(1)` — binary store action
- **Value head**: scalar estimate

`compute_input_dim(cfg, embed_dim)` computes RNN input size from config flags.

`forward(x, h)` returns `(move_dist, store_dist, values, h_next)` — distribution objects.

`get_action_and_value(x, h, deterministic)` — single-step inference returning dict with actions, log_probs, value, h_next.

### rollout.py — `RolloutCollector`

`collect_rollout(env, agent, hopfields, h_rnn, env_offset)` → `RolloutBatch`

Per-step pipeline:
1. `vec.positions()` → snapped positions
2. Compute **current reward** from position (1.0 at goal, -time_penalty otherwise)
3. `vectorhash.get_encoded_state(pos, offset)` → embeddings
4. Hopfield recall → Gram-Schmidt project → normalize → hopfield signal
5. Build RNN input: `[current_reward, embedding?, hopfield_signal?]`
6. Agent forward → move/store actions
7. Execute store (if `agent_can_store` and store_action > 0.5)
8. `vec.step_batch(actions)` → rewards, goal_reached
9. Reset `h_rnn` for teleported batch elements

Creates `VecEnv` or `ContinuousVecEnv` based on `movement_mode`.

### ppo.py

- `RolloutBatch` — dataclass holding `obs, move_actions, store_actions, move_log_probs, store_log_probs, values, rewards, bootstrap_value`.
- `compute_gae(rewards, values, bootstrap_value, gamma, lam)` — GAE with truncation bootstrap only (no terminal states within rollout).
- `ppo_update(agent, rollouts, cfg, optimizer, aux_scale)` — PPO clipped update on a POOLED list of rollouts. Concatenates rollouts along the trajectory axis, computes per-rollout GAE, normalizes advantages globally, then runs `ppo_epochs × n_minibatches` gradient steps with shuffled trajectory-level minibatches. Store head and store entropy are masked by `explore_mask`.

### train.py

- `setup_world(...)` — Create train+val envs, build VectorHash, precompute encoded_Phi, create template Hopfield if pre_stored. Returns env lists + global index mappings.
- `train(cfg)` — Main loop: load encoder → setup worlds → for each update: collect rollouts per world → PPO update → eval → save.

### eval.py

Shared helper `_agent_step(...)` implements one step of the full pipeline (reward from current position, embedding lookup, Hopfield recall, Gram-Schmidt, agent forward, step). Used by all three eval methods.

`_snap_position(pos_f, grid_size)` — canonical snapping function.

`_at_goal(pos, goal)` — snapped position == goal check.

`_sample_distractor_goals(vectorhash, test_env_offset, env_size, n, rng)` — sample encoded patterns from outside the test env's grid region.

Three evaluation methods described in TRAINING_AND_EVALUATION.md.

### utils.py

- `gram_schmidt_2d_batch(d_forward, d_right)` → `(B, 2, D)` orthonormal basis. Row 0 = East, Row 1 = North.
- `classify_direction_batch(q)` → `(B,)` int: 0=N, 1=E, 2=S, 3=W from 2D vectors.
- `direction_to_onehot(idx)` → `(B, 4)` one-hot.
- `smooth_g`, `smooth_gbook` — re-exported from `encoder_training.utils`.

---

## encoder_training

### models.py

- **`GridEncoder`** — MLP: Linear → act → ... → Linear → tanh(gain*z) → L2-normalize. Input `(B, sum(l^2))`, output `(B, out_dim)` on unit sphere.
- **`GridEncoderCNN`** — Reshapes flat grid codes to 2D per-module channels → Conv stack → AdaptiveAvgPool → MLP → tanh(gain*z) → L2-normalize.
- `create_encoder(cfg)` — factory function.

### losses.py

| Function | Description |
|---|---|
| `kernel_alignment_loss(K_pred, K_tgt)` | CKA: `1 - alignment`. Double-centered by default. |
| `weighted_kernel_alignment_loss(...)` | CKA with up-weighted top-K neighbors. |
| `local_attract_far_repel_loss(...)` | Top-K local pairs: CKA. Far pairs: push cosine → -1. |
| `uniformity_loss(z, t)` | Spread embeddings on sphere: `log mean exp(-t ||z_i - z_j||^2)`. |
| `coplanarity_loss_sphere(z_triples)` | Encourage great-circle straightness for position triples. |

### data.py

- `build_grid_data(lambdas, fwhm_ratio, device)` → `(Phi, Xcoords, Npos)`. Generates smoothed grid codes and coordinate tensors.
- `rbf_kernel_batch(Xcoords, idx, tau)` → `(B, B)` RBF kernel.
- `estimate_tau_median(Xcoords)` — median pairwise distance for RBF scale.
- `sample_random_patches(H, W, ...)` — random rectangular patches as index lists.
- `build_grid_triples(H, W, stride)` — consecutive position triples for coplanarity loss.

### train.py

- `train_epoch(...)` — one pass over all grid positions. Supports three loss modes: "cka", "mod_only", "local_far".
- `train(cfg)` — full training: build data → create encoder → epoch loop with gain/uniformity annealing → eval → save.

### evaluate.py

- `eval_encoder(...)` → dict with `align_loss`, `pearson_sim`, `spearman_sim`, `triplet_acc`, `nn_consistency`.
