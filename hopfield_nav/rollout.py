"""RolloutCollector: orchestrates env + vectorhash + encoder + hopfield per step.

Per timestep:
  1. raw_obs from VecEnv
  2. VectorHash recall → g_hot → decode to position
  3. Look up encoded_Phi[position] → embedding
  4. Hopfield recall (per-batch) → signal
  5. Build RNN input → agent forward → actions
  6. If store: hopfield.input_memory(embedding)
  7. VecEnv.step_batch(move_action)
"""
from __future__ import annotations

import numpy as np
import torch

from .config import TrainConfig
from .hopfield import Hopfield, recall_per_env_batch
from .vectorhash import VectorHash
from .vec_env import VecEnv, ContinuousVecEnv
from .env import GridEnv
from .agent import NavAgent
from .ppo import RolloutBatch
from .utils import classify_direction_batch, direction_to_onehot


class RolloutCollector:
    """Collects batched rollouts for one world."""

    def __init__(
        self,
        vectorhash: VectorHash,
        cfg: TrainConfig,
        embed_dim: int,
        device: torch.device,
    ) -> None:
        self.vectorhash = vectorhash
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.device = device
        self.B = cfg.batch_envs
        self.T = cfg.steps_per_rollout

    def collect_rollout(
        self,
        env: GridEnv,
        agent: NavAgent,
        hopfields: list[Hopfield] | Hopfield,
        h_rnn: torch.Tensor | None = None,
        env_offset: tuple[int, int] = (0, 0),
        update_idx: int = 0,
        aux_scale: float = 1.0,
    ) -> RolloutBatch:
        """Collect one rollout of T steps across B parallel episodes in one env.

        Args:
            env: The base environment (VecEnv created internally).
            agent: Policy network (eval mode).
            hopfields: List of B Hopfield instances, or single shared instance.
            h_rnn: Initial RNN hidden state or None.
            env_offset: (C_X, C_Y) global offset for this env in the VectorHash grid.
            update_idx: Current training update (1-indexed). Used for auto_store_warmup gating.
            aux_scale: Scalar in [0, 1] applied to store_bonus this rollout (for linear annealing).

        Returns:
            RolloutBatch with all tensors on self.device.
        """
        cfg = self.cfg
        B, T = self.B, self.T
        shared_hopfield = not isinstance(hopfields, list)
        auto_store_active = (
            not shared_hopfield
            and cfg.hopfield.auto_store_warmup > 0
            and update_idx <= cfg.hopfield.auto_store_warmup
        )
        auto_nav_active = (
            cfg.hopfield.auto_nav_warmup > 0
            and update_idx <= cfg.hopfield.auto_nav_warmup
        )
        effective_bonus = cfg.hopfield.store_bonus * aux_scale

        if cfg.agent.movement_mode == "continuous":
            vec = ContinuousVecEnv(env, batch_size=B, scale=cfg.env.continuous_scale)
        else:
            vec = VecEnv(env, batch_size=B)
        vec.reset_all()

        # Determine input dimensions for signal
        signal_dim = 4 if cfg.agent.hopfield_mode == "discrete" else 2

        # Buffers
        all_obs = torch.zeros(B, T, agent.rnn.input_size, device=self.device)
        if cfg.agent.movement_mode == "discrete":
            all_move_actions = torch.zeros(B, T, dtype=torch.long, device=self.device)
        else:
            all_move_actions = torch.zeros(B, T, 2, device=self.device)
        all_store_actions = torch.zeros(B, T, device=self.device)
        all_move_lp = torch.zeros(B, T, device=self.device)
        all_store_lp = torch.zeros(B, T, device=self.device)
        all_values = torch.zeros(B, T, device=self.device)
        all_rewards = torch.zeros(B, T, device=self.device)
        all_goal_reached = torch.zeros(B, T, device=self.device)  # 1 if agent is at goal this step
        all_explore_mask = torch.zeros(B, T, device=self.device)

        # Novelty bonus: track per-rollout visited cells (B, size, size) bool.
        # +novelty_reward on first visit during the explore phase; revisits get 0.
        novelty_on = cfg.hopfield.novelty_reward > 0
        if novelty_on:
            visited_cells = np.zeros((B, cfg.env.size, cfg.env.size), dtype=bool)
            # Mark initial positions as visited so first step doesn't award novelty for starting cell
            init_pos = vec.positions()
            visited_cells[np.arange(B), init_pos[:, 0], init_pos[:, 1]] = True

        cached_W: np.ndarray | None = None
        steps_since_recompute = np.zeros(B, dtype=np.int32)

        agent.eval()
        with torch.no_grad():
            for t in range(T):
                # 1. Positions → embeddings
                positions = vec.positions()  # (B, 2) int

                # 2. Current reward from current position (before acting)
                goal_arr = np.array(vec._goal)
                at_goal = (positions[:, 0] == goal_arr[0]) & (positions[:, 1] == goal_arr[1])
                current_reward = np.where(at_goal, 1.0, -cfg.env.time_penalty).astype(np.float32)
                current_reward_t = torch.from_numpy(current_reward).to(self.device).unsqueeze(1)  # (B, 1)

                # 3. Look up embeddings from encoded_Phi
                embeddings_np = self.vectorhash.get_encoded_state(positions, env_offset)
                embeddings = torch.from_numpy(embeddings_np).float().to(self.device)

                # 4. Hopfield signal (and raw projected displacement q for teacher forcing)
                hopfield_signal = torch.zeros(B, signal_dim, device=self.device)
                q_full = np.zeros((B, 2), dtype=np.float32)
                memory_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

                if shared_hopfield:
                    hop = hopfields
                    if hop.num_memories > 0:
                        recalled = hop.recall_batch(
                            embeddings, steps=cfg.hopfield.steps,
                            beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
                        )
                        sig_np, q_full = self._compute_signal(
                            embeddings_np, recalled.cpu().numpy(),
                            positions, env_offset, cached_W, steps_since_recompute,
                        )
                        hopfield_signal = torch.from_numpy(sig_np).float().to(self.device)
                        memory_mask[:] = True
                else:
                    # Batched per-env recall: stack W matrices for envs that have memories
                    # and run one bmm instead of B python-dispatched recall calls.
                    has_memory = [b for b in range(B) if hopfields[b].num_memories > 0]
                    if has_memory:
                        idx = torch.as_tensor(has_memory, device=self.device, dtype=torch.long)
                        W_stack = torch.stack([hopfields[b].W for b in has_memory], dim=0)  # (M, D, D)
                        x0_stack = embeddings.index_select(0, idx)                          # (M, D)
                        recalled_stack = recall_per_env_batch(
                            x0_stack, W_stack,
                            steps=cfg.hopfield.steps,
                            beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
                        )
                        recalled_np_full = np.zeros((B, self.embed_dim), dtype=np.float32)
                        recalled_np_full[has_memory] = recalled_stack.cpu().numpy()
                        sig, q_full = self._compute_signal(
                            embeddings_np, recalled_np_full,
                            positions, env_offset, cached_W, steps_since_recompute,
                        )
                        sig_t = torch.from_numpy(sig).float().to(self.device)
                        memory_mask[idx] = True
                        hopfield_signal = torch.where(memory_mask.unsqueeze(-1), sig_t, hopfield_signal)

                steps_since_recompute += 1

                # 4b. Teacher action (auto-nav warmup): override move_action for any env
                # that has a stored memory with the Hopfield-suggested direction.
                # Re-scoring log_prob under the current policy is handled inside the agent.
                move_action_override = None
                move_override_mask = None
                if auto_nav_active and memory_mask.any():
                    if cfg.agent.movement_mode == "discrete":
                        teacher_idx = classify_direction_batch(q_full)  # (B,) int
                        move_action_override = torch.from_numpy(
                            teacher_idx.astype(np.int64)
                        ).to(self.device).unsqueeze(1)  # (B, 1)
                    else:
                        mag = np.linalg.norm(q_full, axis=-1, keepdims=True).clip(1e-8)
                        teacher_vec = (q_full / mag).astype(np.float32)  # unit vector
                        move_action_override = torch.from_numpy(
                            teacher_vec
                        ).to(self.device).unsqueeze(1)  # (B, 1, 2)
                    move_override_mask = memory_mask

                # 5. Build RNN input (current_reward, not prev_reward)
                parts = [current_reward_t]
                if cfg.agent.input_encoded_state:
                    parts.append(embeddings)
                if cfg.agent.input_hopfield_signal:
                    parts.append(hopfield_signal)

                rnn_input = torch.cat(parts, dim=-1).unsqueeze(1)  # (B, 1, input_dim)

                # 6. Agent forward
                result = agent.get_action_and_value(
                    rnn_input, h_rnn,
                    move_action_override=move_action_override,
                    move_override_mask=move_override_mask,
                )
                h_rnn = result["h_next"]

                # Store in buffers
                all_obs[:, t] = rnn_input.squeeze(1)
                all_move_actions[:, t] = result["move_action"]
                all_store_actions[:, t] = result["store_action"]
                all_move_lp[:, t] = result["move_log_prob"]
                all_store_lp[:, t] = result["store_log_prob"]
                all_values[:, t] = result["value"]
                all_goal_reached[:, t] = torch.from_numpy(
                    at_goal.astype(np.float32)).to(self.device)

                in_explore = cfg.explore_steps is None or t < cfg.explore_steps
                all_explore_mask[:, t] = float(in_explore)
                agent_store = (result["store_action"] > 0.5).cpu().numpy()

                # 7. Apply stores BEFORE the env step. Under the new vec_env semantics,
                #    the agent observably sits at the goal for one step before teleport,
                #    so `at_goal` here is True precisely when the agent is at goal and
                #    embeddings[b] is the goal embedding — so auto_store and agent_store
                #    can both just store embeddings[b] directly.
                effective_store = agent_store | (auto_store_active & at_goal)
                if not shared_hopfield and in_explore:
                    for b in range(B):
                        if effective_store[b]:
                            hopfields[b].input_memory(embeddings[b])

                # 8. Step environment — teleports envs that were at goal this step.
                if cfg.agent.movement_mode == "discrete":
                    actions = result["move_action"].cpu().numpy().astype(int)
                else:
                    actions = result["move_action"].cpu().numpy()

                rewards, goal_reached, _ = vec.step_batch(actions)

                # Store cost: metabolic penalty on the agent's own store action.
                if cfg.hopfield.store_cost > 0 and in_explore:
                    rewards -= cfg.hopfield.store_cost * agent_store.astype(np.float32)

                # Store bonus: agent fired store on the at-goal step.
                if effective_bonus > 0 and in_explore:
                    rewards += effective_bonus * (agent_store & at_goal).astype(np.float32)

                # Novelty bonus: reward first-visit to a snapped cell during explore.
                if novelty_on and in_explore:
                    new_pos = vec.positions()  # (B, 2) post-step snapped ints
                    xs, ys = new_pos[:, 0], new_pos[:, 1]
                    not_visited = ~visited_cells[np.arange(B), xs, ys]
                    rewards += cfg.hopfield.novelty_reward * not_visited.astype(np.float32)
                    visited_cells[np.arange(B), xs, ys] = True

                all_rewards[:, t] = torch.from_numpy(rewards).to(self.device)

                # Reset state for teleported episodes
                if goal_reached.any():
                    steps_since_recompute[goal_reached] = 0
                    if h_rnn is not None:
                        reached_idx = torch.from_numpy(
                            np.where(goal_reached)[0]).to(self.device)
                        h_rnn[:, reached_idx, :] = 0.0

            # Bootstrap value at truncation
            pos_final = vec.positions()
            goal_arr = np.array(vec._goal)
            at_goal_final = (pos_final[:, 0] == goal_arr[0]) & (pos_final[:, 1] == goal_arr[1])
            boot_reward = np.where(at_goal_final, 1.0, -cfg.env.time_penalty).astype(np.float32)
            boot_reward_t = torch.from_numpy(boot_reward).to(self.device).unsqueeze(1)

            parts_final = [boot_reward_t]
            if cfg.agent.input_encoded_state:
                emb_final = torch.from_numpy(
                    self.vectorhash.get_encoded_state(pos_final, env_offset)
                ).float().to(self.device)
                parts_final.append(emb_final)
            if cfg.agent.input_hopfield_signal:
                parts_final.append(torch.zeros(B, signal_dim, device=self.device))

            final_input = torch.cat(parts_final, dim=-1).unsqueeze(1)
            _, _, bootstrap_val, _ = agent(final_input, h_rnn)
            bootstrap_value = bootstrap_val.squeeze(1)

        return RolloutBatch(
            obs=all_obs,
            move_actions=all_move_actions,
            store_actions=all_store_actions,
            move_log_probs=all_move_lp,
            store_log_probs=all_store_lp,
            values=all_values,
            rewards=all_rewards,
            bootstrap_value=bootstrap_value,
            goal_reached=all_goal_reached,
            explore_mask=all_explore_mask,
        )

    def _compute_signal(
        self,
        embeddings_np: np.ndarray,
        recalled_np: np.ndarray,
        positions: np.ndarray,
        env_offset: tuple[int, int],
        cached_W: np.ndarray | None,
        steps_since_recompute: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Hopfield signal from embeddings and recalled patterns.

        Returns (signal, q):
            signal: (B, signal_dim) — 4 for discrete, 2 for continuous.
            q:      (B, 2)          — raw projected displacement in (East, North)
                                       coordinates, used for teacher forcing.
        """
        recompute_mask = (steps_since_recompute >= self.cfg.recompute_interval)
        W = self.vectorhash.gram_schmidt_projection(
            positions, env_offset,
            cached_W=cached_W, recompute_mask=recompute_mask,
        )
        q = self.vectorhash.project_displacement(embeddings_np, recalled_np, W)  # (B, 2)
        q = q.astype(np.float32, copy=False)

        if self.cfg.agent.hopfield_mode == "discrete":
            idx = classify_direction_batch(q)
            signal = direction_to_onehot(idx)
        else:
            mag = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
            signal = (q / mag).astype(np.float32)
        return signal, q
