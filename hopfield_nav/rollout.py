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
from .hopfield import Hopfield
from .vectorhash import VectorHash
from .vec_env import VecEnv, ContinuousVecEnv
from .env import GridEnv, CARDINAL_ACTIONS
from .agent import NavAgent
from .ppo import RolloutBatch
from .utils import gram_schmidt_2d_batch, classify_direction_batch, direction_to_onehot


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
    ) -> RolloutBatch:
        """Collect one rollout of T steps across B parallel episodes in one env.

        Args:
            env: The base environment (VecEnv created internally).
            agent: Policy network (eval mode).
            hopfields: List of B Hopfield instances, or single shared instance.
            h_rnn: Initial RNN hidden state or None.
            env_offset: (C_X, C_Y) global offset for this env in the VectorHash grid.

        Returns:
            RolloutBatch with all tensors on self.device.
        """
        cfg = self.cfg
        B, T = self.B, self.T
        shared_hopfield = not isinstance(hopfields, list)

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

                # 4. Hopfield signal
                hopfield_signal = torch.zeros(B, signal_dim, device=self.device)

                if shared_hopfield:
                    hop = hopfields
                    if hop.num_memories > 0:
                        recalled = hop.recall_batch(
                            embeddings, steps=cfg.hopfield.steps,
                            beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
                        )
                        hopfield_signal = self._compute_signal(
                            embeddings_np, recalled.cpu().numpy(),
                            positions, env_offset, cached_W, steps_since_recompute,
                        )
                        hopfield_signal = torch.from_numpy(hopfield_signal).float().to(self.device)
                else:
                    recalled_list = []
                    has_memory = []
                    for b in range(B):
                        if hopfields[b].num_memories > 0:
                            r = hopfields[b].recall(
                                embeddings[b], steps=cfg.hopfield.steps,
                                beta=cfg.hopfield.beta, alpha=cfg.hopfield.alpha,
                            )
                            recalled_list.append(r.cpu().numpy())
                            has_memory.append(b)
                        else:
                            recalled_list.append(np.zeros(self.embed_dim, dtype=np.float32))

                    if has_memory:
                        recalled_np = np.stack(recalled_list)
                        sig = self._compute_signal(
                            embeddings_np, recalled_np,
                            positions, env_offset, cached_W, steps_since_recompute,
                        )
                        for b in has_memory:
                            hopfield_signal[b] = torch.from_numpy(sig[b]).float()

                steps_since_recompute += 1

                # 5. Build RNN input (current_reward, not prev_reward)
                parts = [current_reward_t]
                if cfg.agent.input_encoded_state:
                    parts.append(embeddings)
                if cfg.agent.input_hopfield_signal:
                    parts.append(hopfield_signal)

                rnn_input = torch.cat(parts, dim=-1).unsqueeze(1)  # (B, 1, input_dim)

                # 6. Agent forward
                result = agent.get_action_and_value(rnn_input, h_rnn)
                h_rnn = result["h_next"]

                # Store in buffers
                all_obs[:, t] = rnn_input.squeeze(1)
                all_move_actions[:, t] = result["move_action"]
                all_store_actions[:, t] = result["store_action"]
                all_move_lp[:, t] = result["move_log_prob"]
                all_store_lp[:, t] = result["store_log_prob"]
                all_values[:, t] = result["value"]

                # 7. Store action (frozen after explore_steps in two-phase mode)
                in_explore = cfg.explore_steps is None or t < cfg.explore_steps
                if not shared_hopfield and in_explore:
                    for b in range(B):
                        if result["store_action"][b].item() > 0.5:
                            hopfields[b].input_memory(embeddings[b])

                # 8. Step environment
                if cfg.agent.movement_mode == "discrete":
                    actions = result["move_action"].cpu().numpy().astype(int)
                else:
                    actions = result["move_action"].cpu().numpy()

                rewards, goal_reached, _ = vec.step_batch(actions)

                # Store cost: metabolic penalty for firing store (only during explore)
                if cfg.hopfield.store_cost > 0 and in_explore:
                    store_mask = (result["store_action"] > 0.5).cpu().numpy().astype(np.float32)
                    rewards -= cfg.hopfield.store_cost * store_mask

                # Store bonus: reward for storing while at goal
                if cfg.hopfield.store_bonus > 0 and in_explore:
                    store_at_goal = ((result["store_action"] > 0.5).cpu().numpy() & at_goal).astype(np.float32)
                    rewards += cfg.hopfield.store_bonus * store_at_goal

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
        )

    def _compute_signal(
        self,
        embeddings_np: np.ndarray,
        recalled_np: np.ndarray,
        positions: np.ndarray,
        env_offset: tuple[int, int],
        cached_W: np.ndarray | None,
        steps_since_recompute: np.ndarray,
    ) -> np.ndarray:
        """Compute Hopfield signal from embeddings and recalled patterns.

        Returns (B, signal_dim) numpy array — 4 for discrete, 2 for continuous.
        """
        recompute_mask = (steps_since_recompute >= self.cfg.recompute_interval)
        W = self.vectorhash.gram_schmidt_projection(
            positions, env_offset,
            cached_W=cached_W, recompute_mask=recompute_mask,
        )
        q = self.vectorhash.project_displacement(embeddings_np, recalled_np, W)  # (B, 2)

        if self.cfg.agent.hopfield_mode == "discrete":
            idx = classify_direction_batch(q)
            return direction_to_onehot(idx)
        else:
            mag = np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
            return (q / mag).astype(np.float32)
