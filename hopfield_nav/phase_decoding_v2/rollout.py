"""Rollout primitives for phase_decoding_v2.

`RolloutEngine` wraps ckpt loading + encoder + val-env construction and exposes a
single ``rollout`` primitive used by both the trial collector and the trajectory
collector. Handles deterministic vs stochastic via a constructor arg passed
through to ``hopfield_nav.eval.agent_step``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from hopfield_nav.agent import NavAgent, compute_input_dim
from hopfield_nav.encoder import load_encoder
from hopfield_nav.env import at_goal
from hopfield_nav.eval import agent_step, random_start, _sample_distractor_goals
from hopfield_nav.eval_all import (
    build_eval_world,
    make_cfg_from_checkpoint,
    scaffold_layout_dict,
)
from hopfield_nav.hopfield import Hopfield


@dataclass
class EnvBundle:
    """Resolved val-world for a single ckpt: envs + offsets + goal metadata.

    quadrants[i] partitions arenas by their goal's quadrant in [0,3]:
        q = 2 * (r >= half) + (c >= half)   where half = env_size / 2
    """
    envs: list
    offsets: list[tuple[int, int]]
    goals_local: list[tuple[int, int]]
    quadrants: list[int]
    env_size: int
    vh: Any
    cfg: Any
    agent: NavAgent
    device: torch.device
    val_idxs: list[int]
    encoder_path: str
    ckpt_path: str

    def arena_ids(self) -> list[int]:
        return list(range(len(self.envs)))

    def scaffold(self) -> dict:
        return scaffold_layout_dict(self.cfg, self.vh, self.envs, self.val_idxs)


def _quadrant(goal_local: tuple[int, int], env_size: int) -> int:
    half = env_size / 2.0
    r, c = int(goal_local[0]), int(goal_local[1])
    return 2 * int(r >= half) + int(c >= half)


class RolloutEngine:
    """Owns the ckpt + encoder + agent. Exposes ``rollout`` and ``build_bundle``."""

    def __init__(
        self,
        ckpt_path: str,
        encoder_path: str | None = None,
        device: str | torch.device | None = None,
        num_arenas: int = 100,
        random_agent: bool = False,
        random_init_seed: int = 0,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.num_arenas = int(num_arenas)
        self.random_agent = bool(random_agent)
        self.random_init_seed = int(random_init_seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"[rollout] loading ckpt {ckpt_path} on {self.device}"
              + (" (RANDOM-AGENT control: cfg only, weights skipped)"
                 if self.random_agent else ""),
              flush=True)
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = make_cfg_from_checkpoint(ck["config"])
        cfg.num_val_envs = self.num_arenas
        self.cfg = cfg
        print(f"[rollout] cfg ok: env_size={cfg.env.size} hidden_size="
              f"{cfg.agent.hidden_size} num_val_envs={cfg.num_val_envs}",
              flush=True)

        enc_path = encoder_path or cfg.encoder_checkpoint
        print(f"[rollout] loading encoder {enc_path}", flush=True)
        encoder, enc_cfg, enc_gain = load_encoder(enc_path, str(self.device))
        embed_dim = enc_cfg.out_dim
        if cfg.hopfield.beta is None:
            cfg.hopfield.beta = float(enc_gain)
        self.encoder_path = enc_path
        self.embed_dim = embed_dim

        print(f"[rollout] building val world (precomputing encoded_Phi over "
              "Npos² positions — slow on CPU, seconds on GPU)", flush=True)
        val_envs, vh, val_idxs = build_eval_world(cfg, encoder, str(self.device))
        input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
        # Seed BEFORE NavAgent construction so the random init is reproducible.
        torch.manual_seed(self.random_init_seed)
        agent = NavAgent(cfg.agent, input_dim).to(self.device)
        if self.random_agent:
            print(f"[rollout] RANDOM-AGENT: using freshly-initialized weights "
                  f"(torch.manual_seed={self.random_init_seed}); ckpt "
                  "agent_state_dict NOT loaded", flush=True)
        else:
            agent.load_state_dict(ck["agent_state_dict"])
        agent.eval()
        print(f"[rollout] val world built: {len(val_envs)} envs, "
              f"agent ready ({sum(p.numel() for p in agent.parameters())} "
              "params)", flush=True)

        self.val_envs = val_envs
        self.vh = vh
        self.val_idxs = val_idxs
        self.agent = agent

    def build_bundle(self) -> EnvBundle:
        env_size = int(self.cfg.env.size)
        offsets: list[tuple[int, int]] = []
        goals_local: list[tuple[int, int]] = []
        quadrants: list[int] = []
        for i, env in enumerate(self.val_envs):
            off = self.vh.env_offsets[self.val_idxs[i]]
            g = env.goal_location
            offsets.append((int(off[0]), int(off[1])))
            goals_local.append((int(g[0]), int(g[1])))
            quadrants.append(_quadrant(g, env_size))
        return EnvBundle(
            envs=self.val_envs,
            offsets=offsets,
            goals_local=goals_local,
            quadrants=quadrants,
            env_size=env_size,
            vh=self.vh,
            cfg=self.cfg,
            agent=self.agent,
            device=self.device,
            val_idxs=self.val_idxs,
            encoder_path=self.encoder_path,
            ckpt_path=self.ckpt_path,
        )

    def make_hopfield(self) -> Hopfield:
        return Hopfield(self.embed_dim, beta=self.cfg.hopfield.beta,
                        device=str(self.device))

    def seed_distractors(
        self,
        hopfield: Hopfield,
        env_offset: tuple[int, int],
        n_distractors: int,
        rng: np.random.RandomState,
    ) -> None:
        for pat in _sample_distractor_goals(
            self.vh, env_offset, self.cfg.env.size, n_distractors, rng,
        ):
            hopfield.input_memory(torch.from_numpy(pat).float())

    @torch.no_grad()
    def rollout(
        self,
        env,
        env_offset: tuple[int, int],
        *,
        hopfield: Hopfield,
        h_rnn: torch.Tensor | None,
        prev_reward: torch.Tensor | None,
        prev_action: torch.Tensor | None,
        goal_in_memory_flag: bool,
        max_steps: int,
        deterministic: bool,
        record_positions: bool = False,
        stop_on_goal: bool = True,
    ) -> dict:
        """Run up to ``max_steps`` agent steps. Records h_t each step.

        If ``stop_on_goal`` is True, terminates the step *after* the agent
        reaches the goal (so the goal-step itself is recorded). Hopfield is
        modified in-place when the agent fires its store head.
        """
        cfg = self.cfg
        device = self.device
        agent = self.agent
        goal = env.goal_location

        h_buf: list[np.ndarray] = []
        store_buf: list[float] = []
        pos_buf: list[tuple[int, int]] = []
        reached = False
        steps_taken = 0

        for t in range(max_steps):
            if record_positions:
                pos_buf.append(tuple(int(c) for c in env.current_location))
            out = agent_step(
                agent, env, env_offset, self.vh, hopfield,
                h_rnn, cfg, device,
                deterministic=deterministic,
                goal_local=goal,
                goal_in_memory=goal_in_memory_flag,
                prev_reward=prev_reward, prev_action=prev_action,
            )
            h_buf.append(out["h_rnn"][-1, 0].cpu().numpy().astype(np.float32))
            store_buf.append(float(out["store_action"]))

            if out["store_action"] > 0.5:
                hopfield.input_memory(out["embedding"][0])

            h_rnn = out["h_rnn"]
            prev_reward = out["next_prev_reward"]
            prev_action = out["next_prev_action"]

            if stop_on_goal and at_goal(env):
                reached = True
                steps_taken = t + 1
                break
        else:
            steps_taken = max_steps

        if not h_buf:
            h_arr = np.zeros((0, cfg.agent.hidden_size), dtype=np.float32)
        else:
            h_arr = np.stack(h_buf, axis=0)
        return {
            "h": h_arr,
            "store": np.asarray(store_buf, dtype=np.float32),
            "positions": (np.asarray(pos_buf, dtype=np.int32)
                          if record_positions else None),
            "reached": reached,
            "steps_taken": int(steps_taken),
            "h_rnn": h_rnn,
            "prev_reward": prev_reward,
            "prev_action": prev_action,
        }
