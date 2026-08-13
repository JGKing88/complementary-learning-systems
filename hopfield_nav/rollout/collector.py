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

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..policy import channels
from . import signal
from ..world import episode
from ..config import TrainConfig
from hopfield import Hopfield, recall_per_env_batch, recall_per_env_batch_trajectory
from ..world.scaffold import VectorHash
from ..world.vec_env import VecEnv, ContinuousVecEnv
from ..world.env import GridEnv, at_goal
from ..policy.agent import NavAgent
from .types import RolloutBatch
from ..utils import classify_direction_batch, direction_to_onehot
from .oracles import novelty_action_batch_discrete, novelty_action_batch_continuous


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
        *,
        allow_store: bool,
        h_rnn: torch.Tensor | None = None,
        env_offset: tuple[int, int] = (0, 0),
        update_idx: int = 0,
        aux_scale: float = 1.0,
        epsilon_now: float = 0.0,
        goal_in_memory_init: bool = False,
    ) -> RolloutBatch:
        """Collect one rollout of T steps across B parallel episodes in one env.

        Args:
            env: The base environment (VecEnv created internally).
            agent: Policy network (eval mode).
            hopfields: List of B Hopfield instances, or single shared instance.
            allow_store: May the agent's store action write to memory? Required,
                with no default, so every caller states its intent. Until 2026-08
                this was inferred from `isinstance(hopfields, list)`, which made
                the type of the object silently carry a policy decision -- and
                made `train_navigate`'s store head inert for reasons no single
                file recorded.
            h_rnn: Initial RNN hidden state or None.
            env_offset: (C_X, C_Y) global offset for this env in the VectorHash grid.
            update_idx: Current training update (1-indexed). Used for auto_store_warmup gating.
            aux_scale: Scalar in [0, 1] applied to store_bonus this rollout (for linear annealing).

        Returns:
            RolloutBatch with all tensors on self.device.
        """
        cfg = self.cfg
        B, T = self.B, self.T
        # `shared_hopfield` keeps only its structural job: whether to index
        # `hopfields[b]` or use one object for the whole batch. Whether a write
        # is *permitted* is `allow_store`, declared by the caller.
        shared_hopfield = not isinstance(hopfields, list)
        if allow_store and shared_hopfield:
            raise ValueError(
                "allow_store=True needs one Hopfield per trajectory: with a "
                "single shared instance all B trajectories would write into the "
                "same memory and contaminate each other. Pass a list of B."
            )
        auto_store_active = (
            allow_store                     # auto-store forces a *write*
            and cfg.hopfield.auto_store_warmup > 0
            and update_idx <= cfg.hopfield.auto_store_warmup
        )
        auto_nav_active = (
            cfg.hopfield.auto_nav_warmup > 0
            and update_idx <= cfg.hopfield.auto_nav_warmup
        )
        effective_bonus = cfg.hopfield.store_bonus * aux_scale

        # DAgger BC mode: record teacher labels, but let the student act on
        # its own sample. When collect_teacher is True, auto_nav teacher-
        # forcing is suppressed so the student visits its own distribution.
        collect_teacher = (cfg.training_mode == "bc")
        # The teacher only trusts the Hopfield-direction reading when a
        # **goal** pattern is known to be in memory. We track per env whether
        # a store fired *while at goal* this rollout — that's the only way
        # the agent can plant a trustworthy memory. Store-firings at non-goal
        # cells (common early in training) would populate the Hopfield with
        # nonsense and make a naive `has_memory` gate steer the student toward
        # junk. With pre-populated distractors, `has_memory` is True from step
        # zero but no goal-memory exists until the agent stores at goal.
        # When `goal_in_memory_init=True` (caller-declared: pre_stored
        # rollouts where the goal pattern was loaded into the Hopfield
        # before the rollout started), initialize all B trajectories' bit
        # to True. This matches eval-time semantics for evaluate_navigation
        # (which hardcodes goal_in_memory=True since the eval pre-loads goal).
        agent_goal_store_fired = np.full(B, goal_in_memory_init, dtype=bool)

        if cfg.agent.movement_mode == "continuous":
            vec = ContinuousVecEnv(
                env, batch_size=B,
                scale=cfg.env.continuous_scale,
                normalize=cfg.env.continuous_normalize,
                max_action_norm=cfg.env.max_action_norm,
                min_action_norm=cfg.env.min_action_norm,
            )
        else:
            vec = VecEnv(env, batch_size=B)
        vec.reset_all()

        # The at-goal contract this rollout runs under, stated rather than
        # inherited from whichever env class was constructed. When goals_active
        # is False the env reports no at-goal rows at all, so the contract has
        # nothing to act on and behavior is unchanged.
        goal_contract = episode.contract_for("training_rollout")

        # Determine input dimensions for signal
        signal_dim = channels.signal_width(cfg.agent)
        prev_action_dim = channels.prev_action_width(cfg.agent)
        # The policy input layout, resolved once: both assembly sites below
        # (per-step and truncation bootstrap) build against this same list.
        input_specs = channels.channel_specs(
            cfg.agent, self.embed_dim, cfg.env.observation_size,
        )

        # Enrichment buffers: prev_reward and prev_action feed step t's input
        # based on step t-1's outcomes. Initialized to zero at rollout start;
        # reset to zero for envs that just teleported (post-goal-reach).
        prev_reward_t = torch.zeros(B, 1, device=self.device)
        prev_action_t = torch.zeros(B, prev_action_dim, device=self.device)

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
        # 1 where the executed action came from the policy sample, 0 where it
        # was replaced by an off-policy override (ε-explore or auto-nav teacher
        # forcing). PPO move_loss masks these out — those steps were env
        # exploration, not policy gradient signal, and including them blows
        # up the importance ratio under narrow / unfrozen std.
        all_policy_action_mask = torch.ones(B, T, device=self.device)

        # Novelty bonus: track per-rollout visited cells (B, size, size) bool.
        # +novelty_reward on first visit during the explore phase; revisits get 0.
        # BC mode also needs this buffer for the novelty-action teacher.
        novelty_on = cfg.hopfield.novelty_reward > 0
        revisit_on = cfg.hopfield.revisit_penalty > 0
        wall_on = cfg.hopfield.wall_penalty > 0
        persistence_on = cfg.hopfield.persistence_bonus > 0
        # visited_cells is needed for novelty + revisit shaping (and BC novelty
        # teacher). Wall + persistence shaping only need positional / action
        # info, not visited_cells.
        need_visited = novelty_on or revisit_on or collect_teacher
        if need_visited:
            visited_cells = np.zeros((B, cfg.env.size, cfg.env.size), dtype=bool)
            # Mark initial positions as visited so first step doesn't award novelty for starting cell
            init_pos = vec.positions()
            visited_cells[np.arange(B), init_pos[:, 0], init_pos[:, 1]] = True

        # BC teacher-label buffers (populated only in collect_teacher mode).
        if collect_teacher:
            if cfg.agent.movement_mode == "discrete":
                teacher_move = torch.zeros(B, T, dtype=torch.long, device=self.device)
            else:
                teacher_move = torch.zeros(B, T, 2, device=self.device)
            teacher_store = torch.zeros(B, T, device=self.device)
            move_label_mask = torch.zeros(B, T, device=self.device)
            store_label_mask = torch.zeros(B, T, device=self.device)
            trust_hop_mask = torch.zeros(B, T, device=self.device)

        # Caching for the Gram-Schmidt basis. We invalidate the cache for an
        # env on (a) the very first step (cached_W=None forces full recompute),
        # (b) every recompute_interval steps, and (c) the step right after a
        # teleport (the cached W was computed at a now-stale position).
        cached_W: np.ndarray | None = None
        invalidated_envs = np.zeros(B, dtype=bool)

        agent.eval()
        with torch.no_grad():
            for t in range(T):
                # 1. Positions → embeddings
                positions = vec.positions()  # (B, 2) int

                # 2. Current reward from current position (before acting)
                at_goal_mask = at_goal(vec)
                if vec.goals_active:
                    current_reward = np.where(
                        at_goal_mask, cfg.env.goal_reward, -cfg.env.time_penalty
                    ).astype(np.float32)
                else:
                    current_reward = np.full(at_goal_mask.shape, -cfg.env.time_penalty, dtype=np.float32)
                current_reward_t = torch.from_numpy(current_reward).to(self.device).unsqueeze(1)  # (B, 1)

                # 3. Look up embeddings from encoded_Phi
                embeddings_np = self.vectorhash.get_encoded_state(positions, env_offset)
                embeddings = torch.from_numpy(embeddings_np).float().to(self.device)

                # 3b. Raw sensory observations from env codebook (if agent uses them)
                if cfg.agent.input_sensory:
                    sensory_np = vec.obs_batch()
                    sensory = torch.from_numpy(sensory_np).float().to(self.device)
                else:
                    sensory = None

                # 4. Hopfield signal (and raw projected displacement q for teacher forcing).
                # Build per-env recompute mask: full recompute on first step, every
                # recompute_interval steps, or for any env that just teleported.
                if cached_W is None or (t % cfg.recompute_interval == 0):
                    recompute_mask = np.ones(B, dtype=bool)
                else:
                    recompute_mask = invalidated_envs.copy()
                invalidated_envs[:] = False

                hopfield_signal, q_full, memory_mask, new_W = self._hopfield_signal_at(
                    embeddings_np, embeddings, positions, env_offset,
                    hopfields, shared_hopfield, signal_dim,
                    cached_W=cached_W, recompute_mask=recompute_mask,
                )
                multistep_q = self._compute_multistep_q(
                    embeddings_np, embeddings, hopfields, shared_hopfield,
                    cached_W if new_W is None else new_W,
                    cfg.agent.input_hopfield_multistep,
                )
                if new_W is not None:
                    cached_W = new_W

                # 4b. Teacher action (auto-nav warmup): override move_action for any env
                # that has a stored memory with the Hopfield-suggested direction.
                # Re-scoring log_prob under the current policy is handled inside the agent.
                # DAgger BC mode records labels instead of overriding actions — the
                # student must visit its own state distribution, so the override is
                # suppressed. Labels are attached further down.
                # Compose epsilon-explore + auto-nav overrides. Both write into
                # the agent's move_action_override slot; if naively assigned in
                # sequence, the second one clobbers the first. We resolve with
                # epsilon precedence on its mask, auto_nav filling memory_mask
                # elsewhere. Both branches re-score log_prob inside
                # get_action_and_value so PPO's ratio stays well-defined.
                move_action_override = None
                move_override_mask = None
                eps_action: torch.Tensor | None = None
                eps_mask: torch.Tensor | None = None
                if epsilon_now > 0 and not collect_teacher:
                    eps_roll = torch.rand(B, device=self.device)
                    candidate_mask = eps_roll < epsilon_now
                    if candidate_mask.any():
                        if cfg.agent.movement_mode == "continuous":
                            theta = torch.rand(B, device=self.device) * (2 * math.pi)
                            rand_vec = torch.stack(
                                [torch.cos(theta), torch.sin(theta)], dim=-1,
                            )
                            eps_action = rand_vec.unsqueeze(1)  # (B, 1, 2)
                        else:
                            rand_idx = torch.randint(
                                0, 4, (B,), device=self.device, dtype=torch.long,
                            )
                            eps_action = rand_idx.unsqueeze(1)  # (B, 1)
                        eps_mask = candidate_mask

                nav_action: torch.Tensor | None = None
                nav_mask: torch.Tensor | None = None
                if auto_nav_active and memory_mask.any() and not collect_teacher:
                    if cfg.agent.movement_mode == "discrete":
                        teacher_idx = classify_direction_batch(q_full)
                        nav_action = torch.from_numpy(
                            teacher_idx.astype(np.int64)
                        ).to(self.device).unsqueeze(1)  # (B, 1)
                    else:
                        mag = np.linalg.norm(q_full, axis=-1, keepdims=True).clip(1e-8)
                        teacher_vec = (q_full / mag).astype(np.float32)
                        nav_action = torch.from_numpy(
                            teacher_vec
                        ).to(self.device).unsqueeze(1)  # (B, 1, 2)
                    nav_mask = memory_mask

                if eps_action is not None and nav_action is not None:
                    # Both fire: epsilon wins on its mask, nav fills the rest of memory_mask.
                    if cfg.agent.movement_mode == "continuous":
                        move_action_override = torch.where(
                            eps_mask.view(-1, 1, 1), eps_action, nav_action,
                        )
                    else:
                        move_action_override = torch.where(
                            eps_mask.view(-1, 1), eps_action, nav_action,
                        )
                    move_override_mask = eps_mask | nav_mask
                elif eps_action is not None:
                    move_action_override = eps_action
                    move_override_mask = eps_mask
                elif nav_action is not None:
                    move_action_override = nav_action
                    move_override_mask = nav_mask

                # 4c. BC teacher-label computation. Movement label:
                #   has_memory=True  -> Hopfield-derived direction (student
                #                       learns to reproduce its own input).
                #   has_memory=False -> novelty action (first-visit neighbor).
                # Store label: 1 at goal else 0.
                # Masks: skip movement label at goal (env about to teleport);
                # optionally skip pre-memory nav labels when supervise_explore=False.
                if collect_teacher:
                    has_memory_np = memory_mask.cpu().numpy()
                    # Trust Hopfield-as-goal-direction only when (a) memory is
                    # populated AND (b) the agent has stored while *at goal*
                    # this rollout. Mid-training the agent can also fire store
                    # at non-goal cells — treating such memories as reliable
                    # would teach the student to follow junk recalls. With
                    # distractors pre-populated, has_memory is True from step 0
                    # but no goal pattern is present until the agent stores at
                    # goal — this gate handles both cases uniformly.
                    trust_hop = has_memory_np & agent_goal_store_fired
                    if cfg.agent.movement_mode == "discrete":
                        hop_idx = classify_direction_batch(q_full)                 # (B,) int
                        nov_idx = novelty_action_batch_discrete(
                            positions, visited_cells, cfg.env.size, vec._rng,
                            fallback=cfg.bc.novelty_fallback,
                        )
                        tm_t = np.where(trust_hop, hop_idx, nov_idx).astype(np.int64)
                        teacher_move[:, t] = torch.from_numpy(tm_t).to(self.device)
                    else:
                        mag = np.linalg.norm(q_full, axis=-1, keepdims=True).clip(1e-8)
                        hop_vec = (q_full / mag).astype(np.float32)                # (B, 2)
                        pos_cont = (
                            vec.positions_continuous()
                            if hasattr(vec, "positions_continuous")
                            else positions.astype(np.float32)
                        )
                        nov_vec = novelty_action_batch_continuous(
                            pos_cont, visited_cells, cfg.env.size, vec._rng,
                        )
                        tm_t = np.where(trust_hop[:, None], hop_vec, nov_vec).astype(np.float32)
                        teacher_move[:, t] = torch.from_numpy(tm_t).to(self.device)

                    teacher_store[:, t] = torch.from_numpy(
                        at_goal_mask.astype(np.float32)
                    ).to(self.device)

                    move_valid = ~at_goal_mask
                    if not cfg.bc.supervise_explore:
                        # `supervise_explore=False` means only supervise trusted
                        # post-memory nav — gate on trust_hop, not raw has_memory.
                        move_valid = move_valid & trust_hop
                    move_label_mask[:, t] = torch.from_numpy(
                        move_valid.astype(np.float32)
                    ).to(self.device)
                    # trust_hop_mask: 1 wherever the teacher's move label is a
                    # Hopfield-direction (post-store-at-goal) label AND the step
                    # is non-at-goal (i.e., move_valid AND trust_hop). Used to
                    # upweight nav labels in the BC loss.
                    trust_hop_mask[:, t] = torch.from_numpy(
                        ((~at_goal_mask) & trust_hop).astype(np.float32)
                    ).to(self.device)
                    store_label_mask[:, t] = 1.0

                # 5. Build RNN input. Two extra enrichment channels (gated by
                # config flags) carry information the RNN would otherwise have
                # to reconstruct from hidden state alone: prev_reward (did my
                # last step land on goal?) and prev_action. Raw hopfield
                # feeds the unnormalized q so magnitude encodes memory presence
                # + signal confidence — the agent reads "is there a memory" off
                # ‖q‖ instead of a ground-truth flag.
                if (cfg.agent.input_hopfield_raw
                        and cfg.agent.hopfield_mode == "continuous"):
                    sig_for_rnn = torch.from_numpy(q_full).float().to(self.device)
                else:
                    sig_for_rnn = hopfield_signal

                values = {
                    "current_reward": current_reward_t,
                    "prev_reward": prev_reward_t,
                    "encoded_state": embeddings,
                    "hopfield_signal": sig_for_rnn,
                    "prev_action": prev_action_t,
                    "goal_in_memory": torch.from_numpy(
                        agent_goal_store_fired.astype(np.float32)
                    ).to(self.device).unsqueeze(-1),
                }
                if cfg.agent.input_sensory:
                    values["sensory"] = sensory
                for s, q_s in multistep_q.items():
                    values[channels.multistep_name(s)] = (
                        torch.from_numpy(q_s).float().to(self.device))

                rnn_input = channels.build_policy_input(
                    input_specs, values, batch_size=B,
                ).unsqueeze(1)  # (B, 1, input_dim)

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
                    at_goal_mask.astype(np.float32)).to(self.device)

                in_explore = cfg.explore_steps is None or t < cfg.explore_steps
                all_explore_mask[:, t] = float(in_explore)
                # move_loss may only see steps whose action actually moved the
                # agent. Two kinds do not:
                #   - ε / auto-nav overrides, which replaced the policy sample;
                #   - at-goal steps, where the contract discards the move.
                # The second is the subtler one. That action is inert, so the
                # advantage is identical whichever action was sampled: the
                # expected gradient is zero and the sample gradient is pure
                # variance -- injected at precisely the highest-|advantage|
                # steps in the buffer, since those are the ones that collect
                # goal_reward. These steps stay in the store loss (store IS
                # causal at the goal) and in the value loss (the return is
                # real); only the movement surrogate drops them.
                move_ignored = (
                    at_goal_mask & vec.goals_active & goal_contract.move_ignored
                )
                policy_chose = torch.from_numpy(~move_ignored).to(self.device)
                if move_override_mask is not None:
                    policy_chose &= ~move_override_mask
                all_policy_action_mask[:, t] = policy_chose.float()
                agent_store = (result["store_action"] > 0.5).cpu().numpy()

                # 7. Apply stores BEFORE the env step. Under the new vec_env semantics,
                #    the agent observably sits at the goal for one step before teleport,
                #    so `at_goal_mask` here is True precisely when the agent is at goal and
                #    embeddings[b] is the goal embedding — so auto_store and agent_store
                #    can both just store embeddings[b] directly.
                effective_store = agent_store | (auto_store_active & at_goal_mask)
                # Update agent_goal_store_fired for the BC teacher's trust gate
                # (BC mode) AND for the agent's input bit (PPO + BC modes when
                # cfg.agent.input_goal_in_memory is True). Only count stores
                # that fire *at goal* — those are the only events that plant a
                # trustworthy goal pattern in memory. Gated by in_explore so the
                # bit cannot flip True during the exploit phase, when the
                # input_memory() call below is suppressed (otherwise the bit
                # would claim "goal in memory" without actually storing it).
                if allow_store and in_explore:
                    agent_goal_store_fired |= (effective_store & at_goal_mask)
                    # With goal_radius > 0.5 the at-goal ball can extend onto
                    # neighbouring cells, so `embeddings[b]` is not necessarily
                    # the goal cell's pattern. cfg.env.allow_offcell_store
                    # decides which one a store writes; the default (True)
                    # returns `embeddings` unchanged.
                    if cfg.env.allow_offcell_store:
                        store_patterns = embeddings
                    else:
                        store_patterns = torch.from_numpy(
                            self.vectorhash.get_store_patterns(
                                positions, env_offset,
                                at_goal_mask=at_goal_mask,
                                goal=tuple(vec._goal),
                                allow_offcell=False,
                            )
                        ).float().to(self.device)
                    for b in range(B):
                        if effective_store[b]:
                            hopfields[b].input_memory(store_patterns[b])

                # 8. Step environment — teleports envs that were at goal this step.
                if cfg.agent.movement_mode == "discrete":
                    actions = result["move_action"].cpu().numpy().astype(int)
                else:
                    actions = result["move_action"].cpu().numpy()

                rewards, goal_reached, _ = vec.step_batch(
                    actions, contract=goal_contract)

                # Store cost: metabolic penalty on the agent's own store action.
                if cfg.hopfield.store_cost > 0 and in_explore:
                    rewards -= cfg.hopfield.store_cost * agent_store.astype(np.float32)

                # Store bonus: agent fired store on the at-goal step.
                if effective_bonus > 0 and in_explore:
                    rewards += effective_bonus * (agent_store & at_goal_mask).astype(np.float32)

                # Novelty bonus: reward first-visit to a snapped cell during explore.
                # In BC mode we still track visits to drive the novelty teacher,
                # but only the PPO-mode reward path fires when novelty_on.
                # Gated by ~at_goal_mask so the post-step position used for shaping
                # is the actual move outcome, not a teleport target (envs that
                # were at-goal pre-step were teleported by step_batch).
                # Explore-phase reward shaping. Each shaping component fires
                # independently of the others — earlier code coupled them all
                # under `need_visited`, which silently disabled wall_penalty
                # and revisit_penalty when novelty=0.
                if in_explore and (need_visited or wall_on or persistence_on):
                    new_pos = vec.positions()  # (B, 2) post-step snapped ints
                    xs, ys = new_pos[:, 0], new_pos[:, 1]
                    moved = (~at_goal_mask).astype(np.float32)  # zero out teleport rows

                    if need_visited:
                        not_visited = ~visited_cells[np.arange(B), xs, ys]
                        if novelty_on:
                            if cfg.hopfield.novelty_scale_remaining:
                                # Per-env scale: each new visit pays
                                # base * (total / n_remaining), capped.
                                # n_remaining computed pre-update.
                                total_cells = float(cfg.env.size * cfg.env.size)
                                n_visited_pre = visited_cells.sum(axis=(1, 2)).astype(np.float32)
                                n_remaining = np.maximum(1.0, total_cells - n_visited_pre)
                                scale = np.minimum(
                                    cfg.hopfield.novelty_scale_cap,
                                    total_cells / n_remaining,
                                ).astype(np.float32)
                                rewards += (cfg.hopfield.novelty_reward
                                            * not_visited.astype(np.float32)
                                            * scale
                                            * moved)
                            else:
                                rewards += (cfg.hopfield.novelty_reward
                                            * not_visited.astype(np.float32)
                                            * moved)
                        if revisit_on:
                            rewards -= (cfg.hopfield.revisit_penalty
                                        * (~not_visited).astype(np.float32)
                                        * moved)
                        # Mark post-step positions as visited (including
                        # teleport targets) so a future deliberate visit
                        # doesn't re-claim novelty for that cell.
                        visited_cells[np.arange(B), xs, ys] = True

                    if wall_on:
                        size = cfg.env.size
                        at_edge = (
                            (xs == 0) | (xs == size - 1)
                            | (ys == 0) | (ys == size - 1)
                        )
                        rewards -= (cfg.hopfield.wall_penalty
                                    * at_edge.astype(np.float32)
                                    * moved)
                    if persistence_on:
                        # cos(action_t, action_{t-1}). prev_action_t was
                        # set at end of previous iteration (or zeros at
                        # rollout start / post-teleport). Continuous: raw
                        # 2D vectors, normalize. Discrete: one-hot codes,
                        # dot product == 1 if same direction else 0.
                        if cfg.agent.movement_mode == "discrete":
                            a_t = F.one_hot(
                                result["move_action"].long(), num_classes=4
                            ).float()
                            cos_sim = (a_t * prev_action_t).sum(-1)
                        else:
                            a_t = result["move_action"].float()
                            a_norm = a_t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                            p_norm = prev_action_t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                            cos_sim = ((a_t / a_norm) * (prev_action_t / p_norm)).sum(-1)
                        rewards += (cfg.hopfield.persistence_bonus
                                    * cos_sim.cpu().numpy().astype(np.float32)
                                    * moved)

                all_rewards[:, t] = torch.from_numpy(rewards).to(self.device)

                # Advance enrichment buffers for next step. prev_reward is the
                # reward the agent *just observed* at its decision position
                # this step. prev_action is what the agent actually committed
                # (post teacher-forcing override, post sampling).
                prev_reward_t = current_reward_t
                if cfg.agent.movement_mode == "discrete":
                    prev_action_t = F.one_hot(
                        result["move_action"].long(), num_classes=4
                    ).float()
                else:
                    prev_action_t = result["move_action"].float()

                # Reset state for teleported episodes
                if goal_reached.any():
                    # Cached Gram-Schmidt basis is stale at the new (teleported)
                    # position; force-recompute next step for these envs.
                    invalidated_envs |= goal_reached
                    reached_idx = torch.from_numpy(
                        np.where(goal_reached)[0]).to(self.device)
                    if h_rnn is not None:
                        h_rnn[:, reached_idx, :] = 0.0
                    # Enrichment buffers also reset: the post-teleport agent
                    # has no valid "previous step" in its new episode segment.
                    prev_reward_t[reached_idx] = 0.0
                    prev_action_t[reached_idx] = 0.0

            # Bootstrap value at truncation
            pos_final = vec.positions()
            at_goal_final = at_goal(vec)
            if vec.goals_active:
                boot_reward = np.where(
                    at_goal_final, cfg.env.goal_reward, -cfg.env.time_penalty
                ).astype(np.float32)
            else:
                boot_reward = np.full(at_goal_final.shape, -cfg.env.time_penalty, dtype=np.float32)
            boot_reward_t = torch.from_numpy(boot_reward).to(self.device).unsqueeze(1)

            # Real Hopfield signal at the bootstrap state (was zeros). Without
            # this the value head sees "no memory" at horizon regardless of the
            # actual Hopfield content, biasing the truncation bootstrap.
            emb_final_np = self.vectorhash.get_encoded_state(pos_final, env_offset)
            emb_final = torch.from_numpy(emb_final_np).float().to(self.device)
            sig_final, q_final, _, W_final = self._hopfield_signal_at(
                emb_final_np, emb_final, pos_final, env_offset,
                hopfields, shared_hopfield, signal_dim,
                cached_W=None, recompute_mask=None,
            )
            multistep_q_final = self._compute_multistep_q(
                emb_final_np, emb_final, hopfields, shared_hopfield,
                W_final, cfg.agent.input_hopfield_multistep,
            )
            if cfg.agent.input_hopfield_raw and cfg.agent.hopfield_mode == "continuous":
                sig_for_rnn_final = torch.from_numpy(q_final).float().to(self.device)
            else:
                sig_for_rnn_final = sig_final

            values_final = {
                "current_reward": boot_reward_t,
                "prev_reward": prev_reward_t,
                "encoded_state": emb_final,
                "hopfield_signal": sig_for_rnn_final,
                "prev_action": prev_action_t,
                "goal_in_memory": torch.from_numpy(
                    agent_goal_store_fired.astype(np.float32)
                ).to(self.device).unsqueeze(-1),
            }
            if cfg.agent.input_sensory:
                values_final["sensory"] = torch.from_numpy(
                    vec.obs_batch()).float().to(self.device)
            for s, q_s in multistep_q_final.items():
                values_final[channels.multistep_name(s)] = (
                    torch.from_numpy(q_s).float().to(self.device))

            final_input = channels.build_policy_input(
                input_specs, values_final, batch_size=B,
            ).unsqueeze(1)
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
            policy_action_mask=all_policy_action_mask,
            trust_hop_mask=trust_hop_mask if collect_teacher else None,
            teacher_move_action=teacher_move if collect_teacher else None,
            teacher_store_action=teacher_store if collect_teacher else None,
            move_label_mask=move_label_mask if collect_teacher else None,
            store_label_mask=store_label_mask if collect_teacher else None,
        )

    # The three helpers below moved to signal.py so that eval.agent_step runs
    # the same code instead of its own B=1 reimplementation. They stay as thin
    # methods because the call sites read better with the collector's state
    # (cfg, vectorhash, device, embed_dim) supplied implicitly.

    def _hopfield_signal_at(
        self,
        embeddings_np: np.ndarray,
        embeddings: torch.Tensor,
        positions: np.ndarray,
        env_offset: tuple[int, int],
        hopfields,
        shared_hopfield: bool,
        signal_dim: int,
        cached_W: np.ndarray | None = None,
        recompute_mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray | None]:
        """Recall + Gram-Schmidt projection for a batch of states.

        ``signal_dim`` is accepted for call-site symmetry and derived from the
        config inside; see :func:`signal.hopfield_signal_at`.
        """
        return signal.hopfield_signal_at(
            self.vectorhash, self.cfg, embeddings_np, embeddings, positions,
            env_offset, hopfields, shared_hopfield, self.device,
            self.embed_dim, cached_W=cached_W, recompute_mask=recompute_mask,
        )

    def _compute_multistep_q(
        self,
        embeddings_np: np.ndarray,
        embeddings: torch.Tensor,
        hopfields,
        shared_hopfield: bool,
        cached_W: np.ndarray | None,
        multistep_steps: list[int],
    ) -> dict[int, np.ndarray]:
        """See :func:`signal.multistep_q`."""
        return signal.multistep_q(
            self.vectorhash, self.cfg, embeddings_np, embeddings, hopfields,
            shared_hopfield, cached_W, multistep_steps, self.embed_dim,
            self.device,
        )

    def _compute_signal(
        self,
        embeddings_np: np.ndarray,
        recalled_np: np.ndarray,
        positions: np.ndarray,
        env_offset: tuple[int, int],
        cached_W: np.ndarray | None,
        recompute_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See :func:`signal.project_to_signal`."""
        return signal.project_to_signal(
            self.vectorhash, embeddings_np, recalled_np, positions, env_offset,
            self.cfg.agent, cached_W, recompute_mask,
        )

