from __future__ import annotations

"""Train an RNN policy to imitate the environment's greedy best action.

This script generates trajectories in the `WMEnv`, labels each
time step with the environment's `best_action_to_goal` (mapped to one of four
cardinal actions), and optimizes a simple GRU policy using cross-entropy.

Key choices:
- Observation features are compact (heading one-hot, positions, and summary
  statistics of the environment-provided observation code) to keep the example
  focused on wiring rather than representation.
- WMEnv observations are fixed binary codes per (position, heading), not wall/FOV slices.
- Labels consist of the 4 cardinal actions [N, E, S, W]. For non-cardinal actions
  like diagonals, we map to the dominant-axis cardinal action.
- Variable-length episodes are padded in the batch and ignored via ignore_index.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
import time

from cls import WMEnv
from cls.models import Agent, GRU
from cls.utils.GridUtils import VectorHash
from cls.envs.environments import GridWMEnv, GridWMVecEnv, WMVecEnv
from cls.encoder import GridEncoder

CARDINAL_ACTIONS: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

@dataclass
class TrainConfig:
    size: int
    speed: int
    seed: int
    num_envs: int
    num_val_envs: int
    num_train_worlds: int
    hidden_size: int
    num_model_layers: int
    batch_episodes: int
    steps_per_episode: int
    n_epochs: int
    val_batch_episodes: int
    plot_every: int
    val_epochs: int
    lr: float
    device: str
    use_wandb: bool
    wandb_project: str
    time_penalty: float
    input_size: int
    input_addendum: Optional[str]
    input_type: str
    model_class: str
    encoder_dim: Optional[int]
    num_encoder_layers: int
    num_actions: int
    dropout: float
    train_method: str
    ppo_clip: float
    ppo_vf_coef: float
    ppo_ent_coef: float
    ppo_epochs: int
    max_envs_per_epoch: int
    use_preconv_codebook: bool
    ppo_input_reward: bool
    lambdas: list
    Npos: int | None
    encoder_gain: float | None
    hopfield_gain: float | None
    hopfield_alpha: float
    expert_actions: bool

@torch.no_grad()
def _policy_action(model: Agent, obs: np.ndarray, h: torch.Tensor | None, device: str, epsilon: float = 0.0) -> tuple[int, torch.Tensor | None]:
    x = torch.from_numpy(obs).to(device).float().view(1, 1, -1)
    logits, values, h_next = model(x, h)
    if epsilon > 0.0 and np.random.rand() < epsilon:
        action_idx = int(np.random.randint(0, len(CARDINAL_ACTIONS)))
    else:
        action_idx = int(torch.argmax(logits[0, 0]).item())
    return action_idx, h_next


def generate_episode(
    env: WMEnv | GridWMEnv,
    model: Agent,
    device: str,
    max_steps: int,
    randomize_best: bool = False,
    epsilon: float = 0.0,
    input_addendum: str | None = None,
    ppo_input_reward: bool = False,
):
    """Roll out an episode using the MODEL's actions; label each step with the env's best action.

    Returns stacked observations (T, F) and integer labels (T,) where labels are
    env.best_action_to_goal at each position.
    """
    obs_list: List[np.ndarray] = []
    label_list: List[int] = []

    env.reset()
    h: torch.Tensor | None = None
    last_reward: float = 0.0
    for _ in range(max_steps):

        # Build features and label
        obs = env.obs()
        if input_addendum == "goal":
            obs = np.concatenate([obs, env.obs_at_goal()])
        elif input_addendum == "diff":
            obs = env.obs_at_goal() - env.obs()
        elif input_addendum == "next_best":
            obs = np.concatenate([obs, env.obs_at_next_best_step()])
        elif input_addendum == "hopfield":
            recalled = env.vectorhash.hopfield_recall(obs)
            obs = np.concatenate([obs, recalled])
        if ppo_input_reward:
            obs = np.concatenate([obs, np.asarray([last_reward], dtype=np.float32)])

        best_vec = env.best_action_to_goal(randomize=randomize_best)
        # Use WMEnv's heading index mapping to maintain consistency
        best_idx = CARDINAL_ACTIONS.index(best_vec) #int(env._heading_index(best_vec))

        obs_list.append(obs)
        label_list.append(best_idx)

        # Step with policy action
        action_idx, h = _policy_action(model, obs, h, device, epsilon)
        vec = CARDINAL_ACTIONS[action_idx]
        cur, goal, obs, reward = env.step(vec)
        last_reward = float(reward)
        if cur == env.goal_location:
            break

    return np.stack(obs_list, axis=0), np.array(label_list, dtype=np.int64)


@torch.no_grad()
def generate_episodes_vectorized(
    env: WMEnv | GridWMEnv,
    model: Agent,
    device: str,
    max_steps: int,
    batch_episodes: int,
    randomize_best: bool = False,
    epsilon: float = 0.0,
    input_addendum: str | None = None,
    ppo_input_reward: bool = False,
    action_selection: str = "greedy",  # "greedy" or "sample"
    profile: bool = False,
    prof: dict | None = None,
    use_preconv_codebook: bool = False,
    expert_actions: bool = False,  # If True, step with best action instead of model's prediction
):
    """Run multiple independent clones of the same env in lockstep and return episodes.

    Each clone shares the codebook (and vectorhash, if present) via `clone()`
    but has independent start/heading state.
    """
    # Build true vectorized env from base env (shared codebook/goal semantics)
    print("building vectorized env")
    B = max(1, int(batch_episodes))
    if isinstance(env, GridWMEnv):
        # use_preconv_codebook will be controlled at call-site
        vec = GridWMVecEnv(env, batch_size=B, use_preconv_codebook=use_preconv_codebook)
    else:
        vec = WMVecEnv(env, batch_size=B)
    vec.reset_all()

    done = [False] * B
    hs: list[torch.Tensor | None] = [None] * B
    buffers_obs: list[list[np.ndarray]] = [[] for _ in range(B)]
    buffers_lbl: list[list[int]] = [[] for _ in range(B)]
    buffers_actions: list[list[int]] = [[] for _ in range(B)]
    buffers_rewards: list[list[float]] = [[] for _ in range(B)]
    buffers_dones: list[list[bool]] = [[] for _ in range(B)]
    buffers_values: list[list[float]] = [[] for _ in range(B)]
    buffers_logprobs: list[list[float]] = [[] for _ in range(B)]

    print("rolling out episodes")
    for step in range(max_steps):
        active_idx = [i for i, d in enumerate(done) if not d]
        if not active_idx:
            break
        t0 = time.perf_counter() if profile else 0.0
        obs_batch = vec.obs_batch(active_idx, input_addendum)
        if ppo_input_reward:
            # Append previous step rewards; at step 0 use zeros
            prev_rewards = np.zeros((len(active_idx), 1), dtype=np.float32)
            if step > 0:
                prev_rewards = np.asarray([buffers_rewards[i][-1] if buffers_rewards[i] else 0.0 for i in active_idx], dtype=np.float32).reshape(-1, 1)
            obs_batch = np.concatenate([obs_batch, prev_rewards], axis=-1)
        if profile and prof is not None:
            prof['obs_ms'] = prof.get('obs_ms', 0.0) + (time.perf_counter() - t0) * 1000.0
        # Best action vectors -> indices 0..3
        t1 = time.perf_counter() if profile else 0.0
        best_vecs = vec.best_action_to_goal_batch(active_idx, randomize_best)
        if profile and prof is not None:
            prof['label_ms'] = prof.get('label_ms', 0.0) + (time.perf_counter() - t1) * 1000.0
        best_idx_batch = [CARDINAL_ACTIONS.index(v) for v in best_vecs]

        x = torch.from_numpy(obs_batch).to(device).float().view(len(active_idx), 1, -1)
        # Prepare hidden state with correct shape (num_layers, batch, hidden)
        need_init_h = any(hs[i] is None for i in active_idx)
        h_in = None if need_init_h else torch.stack([hs[i] for i in active_idx], dim=1)
        if h_in is None and step > 0 and model.is_recurrent:
            raise ValueError("Hidden state went to None")

        t2 = time.perf_counter() if profile else 0.0
        logits, values, h_out = model(x, h_in)
        if profile and prof is not None:
            prof['forward_ms'] = prof.get('forward_ms', 0.0) + (time.perf_counter() - t2) * 1000.0

        # Greedy actions with optional epsilon-greedy
        t3 = time.perf_counter() if profile else 0.0
        if action_selection == "sample":
            probs = torch.softmax(logits[:, 0], dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            actions_tensor = dist.sample()
            pred_indices = actions_tensor.tolist()
            step_logprobs = dist.log_prob(actions_tensor).tolist()
        else:
            pred_indices = torch.argmax(logits[:, 0], dim=-1).tolist()
            # compute logprobs of greedy actions for logging/ppo storage
            probs = torch.softmax(logits[:, 0], dim=-1)
            step_logprobs = torch.log(torch.gather(probs, 1, torch.tensor(pred_indices, device=probs.device).view(-1,1))).squeeze(1).tolist()
        # Optional epsilon-greedy on top
        if epsilon > 0.0 and action_selection == "greedy":
            for j in range(len(pred_indices)):
                if np.random.rand() < epsilon:
                    pred_indices[j] = int(np.random.randint(0, len(CARDINAL_ACTIONS)))
        if profile and prof is not None:
            prof['action_ms'] = prof.get('action_ms', 0.0) + (time.perf_counter() - t3) * 1000.0

        for j, i in enumerate(active_idx):
            buffers_obs[i].append(obs_batch[j])
            buffers_lbl[i].append(best_idx_batch[j])
            buffers_actions[i].append(int(pred_indices[j]))
            buffers_values[i].append(float(values[j, 0].item()))
            buffers_logprobs[i].append(float(step_logprobs[j]))
        # Step all active envs at once using vectorized env
        # Use best actions if expert_actions, otherwise use model's predictions
        step_indices = best_idx_batch if expert_actions else pred_indices
        action_vectors = [CARDINAL_ACTIONS[a] for a in step_indices]
        t4 = time.perf_counter() if profile else 0.0
        curs, goals, rewards, dones = vec.step_batch(active_idx, action_vectors)
        if profile and prof is not None:
            prof['step_ms'] = prof.get('step_ms', 0.0) + (time.perf_counter() - t4) * 1000.0
        for j, i in enumerate(active_idx):
            buffers_rewards[i].append(float(rewards[j]))
            done_flag = bool(dones[j])
            buffers_dones[i].append(done_flag)
            hs[i] = h_out[:, j, :] if h_out is not None else None
            if done_flag:
                done[i] = True
        if profile and prof is not None:
            prof['steps'] = prof.get('steps', 0) + 1

    episodes = []
    for i in range(B):
        if not buffers_obs[i]:
            continue
        episodes.append({
            "obs": np.stack(buffers_obs[i], axis=0),
            "labels": np.array(buffers_lbl[i], dtype=np.int64),
            "actions": np.array(buffers_actions[i], dtype=np.int64),
            "rewards": np.array(buffers_rewards[i], dtype=np.float32),
            "dones": np.array(buffers_dones[i], dtype=np.bool_),
            "values": np.array(buffers_values[i], dtype=np.float32),
            "log_probs": np.array(buffers_logprobs[i], dtype=np.float32),
        })
    return episodes

def collate_supervised(episodes: List[dict] | List[Tuple[np.ndarray, np.ndarray]]):
    """Pad a list of (T, F) observation arrays and (T,) label arrays into a batch.

    Padded labels use -100 (PyTorch CrossEntropy ignore_index) so they do not
    contribute to the loss.
    """
    # episodes: list of (T, F), (T,)
    # Support both dict episodes and tuples
    if isinstance(episodes[0], dict):
        max_T = max(ep["obs"].shape[0] for ep in episodes)
        F = episodes[0]["obs"].shape[1]
    else:
        max_T = max(obs.shape[0] for obs, _ in episodes)
        F = episodes[0][0].shape[1]
    B = len(episodes)
    obs_batch = np.zeros((B, max_T, F), dtype=np.float32)
    tgt_batch = np.full((B, max_T), -100, dtype=np.int64)  # ignore index
    lengths = []
    for i, ep in enumerate(episodes):
        if isinstance(ep, dict):
            obs = ep["obs"]
            tgt = ep["labels"]
        else:
            obs, tgt = ep
        T = obs.shape[0]
        obs_batch[i, :T] = obs
        tgt_batch[i, :T] = tgt
        lengths.append(T)
    return (
        torch.from_numpy(obs_batch),
        torch.from_numpy(tgt_batch),
        torch.tensor(lengths, dtype=torch.int64),
    )


def collate_rollouts(episodes: List[dict]):
    """Pad rollout fields for PPO into batched tensors.

    Returns padded tensors and lengths along with a mask (B, T) where True indicates valid timesteps.
    """
    max_T = max(ep["obs"].shape[0] for ep in episodes)
    B = len(episodes)
    F = episodes[0]["obs"].shape[1]
    obs_batch = np.zeros((B, max_T, F), dtype=np.float32)
    actions = np.zeros((B, max_T), dtype=np.int64)
    rewards = np.zeros((B, max_T), dtype=np.float32)
    dones = np.zeros((B, max_T), dtype=np.bool_)
    values = np.zeros((B, max_T), dtype=np.float32)
    log_probs = np.zeros((B, max_T), dtype=np.float32)
    mask = np.zeros((B, max_T), dtype=np.bool_)
    lengths = []
    for i, ep in enumerate(episodes):
        T = ep["obs"].shape[0]
        obs_batch[i, :T] = ep["obs"]
        actions[i, :T] = ep["actions"]
        rewards[i, :T] = ep["rewards"]
        dones[i, :T] = ep["dones"]
        values[i, :T] = ep["values"]
        log_probs[i, :T] = ep["log_probs"]
        mask[i, :T] = True
        lengths.append(T)
    return (
        torch.from_numpy(obs_batch),
        torch.from_numpy(actions),
        torch.from_numpy(rewards),
        torch.from_numpy(dones),
        torch.from_numpy(values),
        torch.from_numpy(log_probs),
        torch.from_numpy(mask),
        torch.tensor(lengths, dtype=torch.int64),
    )


def compute_gae(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float):
    """Compute Generalized Advantage Estimation over padded batch.

    rewards, dones, values, mask: (B, T)
    Returns advantages and returns (B, T)
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    for b in range(B):
        last_adv = 0.0
        last_value = 0.0
        for t in reversed(range(T)):
            if not mask[b, t]:
                continue
            not_done = 1.0 - dones[b, t].float()
            next_value = values[b, t+1] if (t+1 < T and mask[b, t+1]) else (0.0 if dones[b, t] else values[b, t])
            delta = rewards[b, t] + gamma * next_value * not_done - values[b, t]
            last_adv = delta + gamma * lam * not_done * last_adv
            advantages[b, t] = last_adv
            returns[b, t] = advantages[b, t] + values[b, t]
    return advantages, returns


def ppo_loss(logits: torch.Tensor, values: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor, clip_coef: float, vf_coef: float, ent_coef: float):
    """Compute PPO clipped objective with value and entropy bonuses over masked sequences."""
    B, T, A = logits.shape
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    ratio = torch.exp(new_log_probs - old_log_probs)
    # Masked means
    mask_f = mask.float()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    policy_loss = -torch.sum(torch.min(unclipped, clipped) * mask_f) / (mask_f.sum() + 1e-8)
    value_loss = torch.sum((returns - values) ** 2 * mask_f) / (mask_f.sum() + 1e-8)
    entropy_loss = torch.sum(entropy * mask_f) / (mask_f.sum() + 1e-8)
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
    return total, policy_loss.detach(), value_loss.detach(), entropy_loss.detach()


def plot_grid_path(positions: List[Tuple[int, int]], goal: Tuple[int, int], size: int, title: str) -> None:
    """Save a 2D grid plot of the path taken."""
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    plt.figure(figsize=(5, 5))
    for i in range(size + 1):
        plt.plot([0, size - 1], [i - 0.5, i - 0.5], color="#eeeeee", linewidth=1)
        plt.plot([i - 0.5, i - 0.5], [0, size - 1], color="#eeeeee", linewidth=1)

    plt.plot(xs, ys, marker="o", color="tab:blue")
    plt.scatter([xs[0]], [ys[0]], color="green", s=80, label="start")
    plt.scatter([goal[0]], [goal[1]], color="red", s=80, label="goal")
    plt.xlim(-0.5, size - 0.5)
    plt.ylim(-0.5, size - 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(range(size))
    plt.yticks(range(size))
    plt.grid(False)
    plt.legend(loc="best")
    plt.title(title)


def plot_compare_paths(
    policy_positions: List[Tuple[int, int]],
    best_positions: List[Tuple[int, int]],
    goal: Tuple[int, int],
    size: int,
    title: str,
) -> None:
    """Overlay policy path and best path on the grid."""
    pxs = [p[0] for p in policy_positions]
    pys = [p[1] for p in policy_positions]
    bxs = [p[0] for p in best_positions]
    bys = [p[1] for p in best_positions]

    plt.figure(figsize=(5, 5))
    for i in range(size + 1):
        plt.plot([0, size - 1], [i - 0.5, i - 0.5], color="#eeeeee", linewidth=1)
        plt.plot([i - 0.5, i - 0.5], [0, size - 1], color="#eeeeee", linewidth=1)

    plt.plot(pxs, pys, marker="o", color="tab:blue", label="policy")
    plt.plot(bxs, bys, marker="x", color="tab:orange", label="best")
    plt.scatter([pxs[0]], [pys[0]], color="green", s=80, label="start")
    plt.scatter([goal[0]], [goal[1]], color="red", s=80, label="goal")
    plt.xlim(-0.5, size - 0.5)
    plt.ylim(-0.5, size - 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(range(size))
    plt.yticks(range(size))
    plt.grid(False)
    plt.legend(loc="best")
    plt.title(title)


def rollout_policy_episode(
    model: Agent, env: WMEnv | GridWMEnv, steps: int, device: str, do_reset: bool = True, input_addendum: str | None = None
 ) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """Roll out a single episode using the policy's greedy actions."""
    positions: List[Tuple[int, int]] = []
    if do_reset:
        cur, goal, obs, reward = env.reset()
    else:
        cur, goal = env.current_location, env.goal_location
    positions.append(cur)

    h = None
    for _ in range(steps):
        if input_addendum == "goal":
            obs = np.concatenate([env.obs(), env.obs_at_goal()])
        elif input_addendum == "diff":
            obs = env.obs_at_goal() - env.obs()
        elif input_addendum == "next_best":
            obs = np.concatenate([obs, env.obs_at_next_best_step()])
        else:
            obs = env.obs()

        x = torch.from_numpy(obs).to(device).float().view(1, 1, -1)
        with torch.no_grad():
            logits, h = model(x, h)
            idx = int(torch.argmax(logits[0, 0]).item())
        vec = CARDINAL_ACTIONS[idx]
        cur, goal, obs, reward = env.step(vec)
        positions.append(cur)
        if cur == goal:
            break

    return positions, goal


def rollout_best_episode(env: WMEnv | GridWMEnv, steps: int, do_reset: bool = True) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """Roll out a single episode using the environment's best_action_to_goal."""
    positions: List[Tuple[int, int]] = []
    if do_reset:
        cur, goal, obs, reward = env.reset()
    else:
        cur, goal = env.current_location, env.goal_location
    positions.append(cur)

    for _ in range(steps):
        vec = env.best_action_to_goal(randomize=False)
        cur, goal, obs, reward= env.step(vec)
        positions.append(cur)
        if cur == goal:
            break

    return positions, goal


def validate(
    model: Agent,
    env_pool: List[WMEnv | GridWMEnv],
    steps_per_episode: int,
    val_batch_episodes: int,
    device: str,
    epoch: int,
    out_dir: str,
    save_plots: bool,
    label: str = "val",
    input_addendum: str | None = None,
    use_preconv_codebook: bool = False,
    expert_actions: bool = False,
):
    """Validate current policy using batched rollouts per environment.

    Uses `generate_episodes_vectorized` to roll out `val_batch_episodes` episodes
    in parallel for each environment, then computes token-level accuracy against
    best-action labels. Success is approximated as finishing before max steps.
    """
    os.makedirs(out_dir, exist_ok=True)
    accs = []
    success_counts = []
    for env_idx, env in enumerate(env_pool):
        with torch.no_grad():
            episodes = generate_episodes_vectorized(
                env=env,
                model=model,
                device=device,
                max_steps=steps_per_episode,
                batch_episodes=val_batch_episodes,
                randomize_best=False,
                epsilon=0.0,
                input_addendum=input_addendum,
                use_preconv_codebook=use_preconv_codebook,
                expert_actions=expert_actions,
            )

            if not episodes:
                accs.append(0.0)
                success_counts.append(0)
                continue

            obs_batch, tgt_batch, lengths = collate_supervised(episodes)
            obs_batch = obs_batch.to(device).float()
            tgt_batch = tgt_batch.to(device)

            logits, values, _ = model(obs_batch)
            pred = logits.argmax(dim=-1)
            mask = tgt_batch != -100
            correct = (pred[mask] == tgt_batch[mask]).float().sum().item()
            total = mask.float().sum().item()
            accs.append(float(correct / max(1.0, total)))

            # Success if episode terminated before max steps
            successes = int((lengths < steps_per_episode).sum().item())
            success_counts.append(successes)

        # Optional plotting could be adapted for a single sample if desired

    mean_acc = float(np.mean(accs)) if accs else 0.0
    mean_success = float(np.sum(success_counts)) / float(len(env_pool) * max(1, val_batch_episodes))
    per_env_success = ", ".join(f"{s}/{val_batch_episodes}" for s in success_counts)
    print(
        f"epoch {epoch:04d} | {label} acc mean {mean_acc:.3f} | {label} success mean {mean_success:.3f} | "
        f"acc per-env: " + ", ".join(f"{a:.3f}" for a in accs) +
        f" | success per-env: {per_env_success}"
    )
    return mean_acc, mean_success


def train(
    env_pool: List[WMEnv | GridWMEnv],
    pos_env_pool: List[WMEnv | GridWMEnv],
    new_env_pool: List[WMEnv | GridWMEnv],
    cfg: TrainConfig,
):
    """Main training loop.

    Generates a new batch of episodes each update (on-policy imitation data),
    pads and stacks them, and optimizes the GRU policy with cross entropy.
    Prints running loss and token-level accuracy every 50 updates.
    """

    model = Agent(input_size=cfg.input_size, hidden_size=cfg.hidden_size, num_model_layers=cfg.num_model_layers, model_class=cfg.model_class, encoder_dim=cfg.encoder_dim, num_encoder_layers=cfg.num_encoder_layers, num_actions=cfg.num_actions, dropout=cfg.dropout)
    model.to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    out_dir_base = os.path.join(os.getcwd(), "runs")
    out_dir_train = os.path.join(out_dir_base, "train")
    out_dir_pos_validation = os.path.join(out_dir_base, "pos_validation")
    out_dir_goal_validation = os.path.join(out_dir_base, "goal_validation")
    # Optional Weights & Biases tracking
    if cfg.use_wandb:
        wandb_cfg = dict(
            size=cfg.size,
            speed=cfg.speed,
            seed=cfg.seed,
            num_envs=cfg.num_envs,
            num_val_envs=cfg.num_val_envs,
            num_train_worlds=cfg.num_train_worlds,
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_model_layers=cfg.num_model_layers,
            batch_episodes=cfg.batch_episodes,
            steps_per_episode=cfg.steps_per_episode,
            n_epochs=cfg.n_epochs,
            val_batch_episodes=cfg.val_batch_episodes,
            plot_every=cfg.plot_every,
            val_epochs=cfg.val_epochs,
            lr=cfg.lr,
            device=cfg.device,
            input_addendum=cfg.input_addendum,
            input_type=cfg.input_type,
            model_class=cfg.model_class,
            encoder_dim=cfg.encoder_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            train_method=cfg.train_method,
            ppo_clip=cfg.ppo_clip,
            ppo_vf_coef=cfg.ppo_vf_coef,
            ppo_ent_coef=cfg.ppo_ent_coef,
            ppo_epochs=cfg.ppo_epochs,
            max_envs_per_epoch=cfg.max_envs_per_epoch,
            lambdas=cfg.lambdas,
            encoder_gain=cfg.encoder_gain,
            hopfield_gain=cfg.hopfield_gain,
            hopfield_alpha=cfg.hopfield_alpha,
        )
        wandb.init(project=cfg.wandb_project, config=wandb_cfg)
        
    print("Starting training")
    model.train()
    for epoch in range(1, cfg.n_epochs + 1):
        # For each episode in the batch, pick a random environment from the pool
        episodes = []
        # Collect episodes
        model.eval()
        with torch.no_grad():
            if len(env_pool) >= cfg.max_envs_per_epoch:
                env_idxs = np.random.choice(len(env_pool), cfg.max_envs_per_epoch, replace=False)
            else:
                env_idxs = range(len(env_pool))
            prof = {}
            for env_idx in env_idxs:
                env = env_pool[env_idx]
                episodes.extend(
                    generate_episodes_vectorized(
                        env,
                        model,
                        cfg.device,
                        cfg.steps_per_episode,
                        batch_episodes=cfg.batch_episodes,
                        input_addendum=cfg.input_addendum,
                        ppo_input_reward=cfg.ppo_input_reward,
                        action_selection=("sample" if cfg.train_method == "ppo" else "greedy"),
                        profile=True,
                        prof=prof,
                        use_preconv_codebook=cfg.use_preconv_codebook,
                        expert_actions=cfg.expert_actions,
                    )
                )
        model.train()
        if prof:
            total = sum(prof.get(k, 0.0) for k in ("obs_ms","label_ms","forward_ms","action_ms","step_ms"))
            print(
                f"rollout steps={prof.get('steps',0)} | obs {prof.get('obs_ms',0.0):.1f}ms | label {prof.get('label_ms',0.0):.1f}ms | forward {prof.get('forward_ms',0.0):.1f}ms | action {prof.get('action_ms',0.0):.1f}ms | step {prof.get('step_ms',0.0):.1f}ms | total {total:.1f}ms"
            )

        if cfg.train_method == "supervised":
            obs, tgt, lengths = collate_supervised(episodes)
            obs = obs.to(cfg.device)
            tgt = tgt.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            logits, values, _ = model(obs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = tgt != -100
                acc = (pred[mask] == tgt[mask]).float().mean().item()
                
                # Diagnostic: label distribution and per-class accuracy (every 100 epochs)
                if epoch % 100 == 1:
                    valid_tgt = tgt[mask].cpu().numpy()
                    valid_pred = pred[mask].cpu().numpy()
                    valid_obs = obs[mask].cpu().numpy()
                    
                    label_counts = np.bincount(valid_tgt, minlength=4)
                    per_class_correct = np.zeros(4)
                    per_class_total = np.zeros(4)
                    for c in range(4):
                        class_mask = valid_tgt == c
                        per_class_total[c] = class_mask.sum()
                        per_class_correct[c] = (valid_pred[class_mask] == c).sum()
                    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
                    print(f"  Label dist: N={label_counts[0]}, E={label_counts[1]}, S={label_counts[2]}, W={label_counts[3]}")
                    print(f"  Per-class acc: N={per_class_acc[0]:.3f}, E={per_class_acc[1]:.3f}, S={per_class_acc[2]:.3f}, W={per_class_acc[3]:.3f}")
                    
                    # Check prediction distribution
                    pred_counts = np.bincount(valid_pred, minlength=4)
                    print(f"  Pred dist: N={pred_counts[0]}, E={pred_counts[1]}, S={pred_counts[2]}, W={pred_counts[3]}")
                    
                    # Check for ambiguous cases (current == next_best) if using next_best addendum
                    if cfg.input_addendum == "next_best" and valid_obs.shape[1] % 2 == 0:
                        half = valid_obs.shape[1] // 2
                        current_obs = valid_obs[:, :half]
                        next_best_obs = valid_obs[:, half:]
                        same_obs = np.allclose(current_obs, next_best_obs, atol=1e-5)
                        n_same = np.sum(np.all(np.abs(current_obs - next_best_obs) < 1e-5, axis=1))
                        print(f"  Ambiguous (current==next_best): {n_same}/{len(valid_obs)} ({100*n_same/len(valid_obs):.1f}%)")
                        
                        # Accuracy on non-ambiguous vs ambiguous
                        if n_same > 0:
                            is_same = np.all(np.abs(current_obs - next_best_obs) < 1e-5, axis=1)
                            acc_same = (valid_pred[is_same] == valid_tgt[is_same]).mean() if is_same.sum() > 0 else 0
                            acc_diff = (valid_pred[~is_same] == valid_tgt[~is_same]).mean() if (~is_same).sum() > 0 else 0
                            print(f"  Acc on ambiguous: {acc_same:.3f}, Acc on clear: {acc_diff:.3f}")
                    
            print(f"epoch {epoch:04d} | loss {loss.item():.4f} | train acc {acc:.3f}")
            if cfg.use_wandb:
                wandb.log({"train/loss": float(loss.item()), "train/acc": float(acc)}, step=epoch)
        elif cfg.train_method == "ppo":
            obs, actions, rewards, dones, old_values, old_log_probs, mask, lengths = collate_rollouts(episodes)
            obs = obs.to(cfg.device).float()
            actions = actions.to(cfg.device)
            rewards = rewards.to(cfg.device)
            dones = dones.to(cfg.device)
            old_values = old_values.to(cfg.device)
            old_log_probs = old_log_probs.to(cfg.device)
            mask = mask.to(cfg.device)

            with torch.no_grad():
                _, values, _ = model(obs)
            advantages, returns = compute_gae(rewards, dones, values, mask, gamma=0.99, lam=0.95)
            # Normalize advantages
            adv_mean = advantages[mask].mean()
            adv_std = advantages[mask].std(unbiased=False) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # PPO epochs (single minibatch over full batch for simplicity)
            for _ in range(cfg.ppo_epochs):
                opt.zero_grad(set_to_none=True)
                logits, values, _ = model(obs)
                total_loss, pol_loss, val_loss, ent = ppo_loss(
                    logits=logits,
                    values=values,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    returns=returns,
                    mask=mask,
                    clip_coef=cfg.ppo_clip,
                    vf_coef=cfg.ppo_vf_coef,
                    ent_coef=cfg.ppo_ent_coef,
                )
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            if cfg.use_wandb:
                wandb.log({
                    "train/ppo_total_loss": float(total_loss.item()),
                    "train/ppo_policy_loss": float(pol_loss.item()),
                    "train/ppo_value_loss": float(val_loss.item()),
                    "train/ppo_entropy": float(ent.item()),
                }, step=epoch)
        else:
            raise ValueError("train_method must be 'supervised' or 'ppo'")

        # Validation based on val_epochs cadence
        if cfg.val_epochs <= 0:
            do_val = False
        else:
            do_val = (epoch % cfg.val_epochs == 0)
        if do_val:
            model.eval()
            save_plots = (cfg.plot_every > 0 and (epoch % cfg.plot_every == 0))
            # Progress on the training env pool (was "val")
            train_acc, train_success = validate(
                model, env_pool, cfg.steps_per_episode, cfg.val_batch_episodes, cfg.device, epoch, out_dir_train, save_plots, label="train", input_addendum=cfg.input_addendum, use_preconv_codebook=cfg.use_preconv_codebook, expert_actions=cfg.expert_actions
            )

            # Position validation: same goals as train, new env instances
            pos_acc, pos_success = validate(
                model, pos_env_pool, cfg.steps_per_episode, cfg.val_batch_episodes, cfg.device, epoch, out_dir_pos_validation, save_plots, label="pos_validation", input_addendum=cfg.input_addendum, use_preconv_codebook=cfg.use_preconv_codebook, expert_actions=cfg.expert_actions
            )

            # Goal validation (was "newval"): different goals and env instances
            goal_acc, goal_success = validate(
                model, new_env_pool, cfg.steps_per_episode, cfg.val_batch_episodes, cfg.device, epoch, out_dir_goal_validation, save_plots, label="goal_validation", input_addendum=cfg.input_addendum, use_preconv_codebook=cfg.use_preconv_codebook, expert_actions=cfg.expert_actions
            )
            model.train()
            if cfg.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/acc": float(train_acc),
                        "train/success": float(train_success),
                        "pos_validation/acc": float(pos_acc),
                        "pos_validation/success": float(pos_success),
                        "goal_validation/acc": float(goal_acc),
                        "goal_validation/success": float(goal_success),
                    },
                    step=epoch,
                )

    if cfg.use_wandb:
        wandb.finish()

    return model


def main():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_val_envs", type=int, default=4)
    parser.add_argument("--num_train_worlds", type=int, default=1,
                        help="Number of independent training worlds, each with its own GridEnv, VectorHash, and Hopfield")

    # Model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_model_layers", type=int, default=1)
    parser.add_argument("--model_class", type=str, default="GRU")
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--num_encoder_layers", type=int, default=0)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--max_envs_per_epoch", type=int, default=8)
    parser.add_argument("--batch_episodes", type=int, default=16)
    parser.add_argument("--steps_per_episode", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--val_batch_episodes", type=int, default=4)
    parser.add_argument("--val_epochs", type=int, default=1)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="cls")
    parser.add_argument("--time_penalty", type=float, default=0.01)
    parser.add_argument("--observation_size", type=int, default=512)
    parser.add_argument("--ppo_input_reward", action="store_true", default=True)
    parser.add_argument("--input_addendum", type=str, choices=["goal", "diff", "none", "next_best", "hopfield"], default="none")
    parser.add_argument("--train_method", type=str, choices=["supervised", "ppo"], default="supervised")
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--ppo_vf_coef", type=float, default=0.5)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)

    # Vectorhash
    parser.add_argument("--vectorhash", action="store_true", default=False)
    parser.add_argument("--Np", type=int, default=1600)
    parser.add_argument("--lambdas", type=int, nargs="+", default=[11,12])
    parser.add_argument("--Npos", type=int, default=None)
    parser.add_argument("--input_type", type=str, default="g_idx") #g_idx, g_hot, s, p, or encoded_g
    parser.add_argument("--use_preconv_codebook", action="store_true", default=False)
    parser.add_argument("--fwhm_ratio", type=float, default=0.0,
                        help="FWHM ratio for smoothing g_hot (0 = no smoothing, e.g. 0.25 = FWHM is 1/4 of lambda)")
    
    # GridEncoder (for input_type="encoded_g")
    parser.add_argument("--encoder_weights", type=str, default=None, 
                        help="Filename of encoder weights in encoders/ directory (e.g. 'my_encoder.pt')")
    parser.add_argument("--encoder_gain", type=float, default=None,
                        help="Override gain from loaded encoder (default: use encoder's saved gain)")
    
    # Hopfield (for input_addendum="hopfield")
    parser.add_argument("--hopfield_gain", type=float, default=None,
                        help="Hopfield gain/beta (default: use encoder's gain if using encoder, else 2.0)")
    parser.add_argument("--hopfield_alpha", type=float, default=1.0,
                        help="Hopfield recall mixing coefficient (1.0 = full update)")
    parser.add_argument("--hopfield_steps", type=int, default=1)
    parser.add_argument("--expert_actions", action="store_true", default=False,
                        help="Use best action for stepping during rollouts (behavioral cloning from expert)")
    parser.add_argument("--use_headings", action="store_true", default=False,
                        help="Use heading-dependent observations (default: heading-invariant)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = parser.parse_args()

    if args.input_addendum not in ("goal", "diff", "none", "next_best", "hopfield"):
        raise ValueError("input_addendum must be one of: goal, diff, none, next_best, hopfield")

    # Initialize GridEncoder if using encoded_g
    grid_encoder = None
    encoder_gain = None  # Will be set from loaded encoder or args
    if args.input_type == "encoded_g" and args.vectorhash:
        
        g_hot_dim = sum(l**2 for l in args.lambdas)
        
        if args.encoder_weights is None:
            raise ValueError("--encoder_weights required when using input_type='encoded_g'")
        
        # Load encoder config and weights from file
        encoder_path = os.path.join(os.getcwd(), "encoders", args.encoder_weights)
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")
        
        checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
        
        # Load config from checkpoint
        config = checkpoint.get("config", {})
        hidden = config.get("hidden")
        out_dim = config.get("out_dim")
        nonlinearity = config.get("nonlinearity", "gelu")
        output_nonlinearity = config.get("output_nonlinearity", "tanh")
        
        if hidden is None or out_dim is None:
            raise ValueError(f"Encoder checkpoint missing 'hidden' or 'out_dim' in config: {encoder_path}")
        
        # Get gain: use args override if provided, else use checkpoint's value
        loaded_gain = config.get("gain")
        # Also check sweep_params for sigmoid_scale_end (common in sweep-trained encoders)
        if loaded_gain is None:
            sweep_params = checkpoint.get("sweep_params", {})
            loaded_gain = sweep_params.get("sigmoid_scale_end", 1.0)
        
        encoder_gain = args.encoder_gain if args.encoder_gain is not None else loaded_gain
        
        grid_encoder = GridEncoder(
            in_dim=g_hot_dim,
            hidden=hidden,
            out_dim=out_dim,
            nonlinearity=nonlinearity,
            output_nonlinearity=output_nonlinearity,
            gain=encoder_gain,
        )
        
        # Load weights
        if "state_dict" in checkpoint:
            grid_encoder.load_state_dict(checkpoint["state_dict"])
        else:
            grid_encoder.load_state_dict(checkpoint)
        
        print(f"Loaded GridEncoder from {encoder_path}: {g_hot_dim} -> {hidden} -> {out_dim}, gain={encoder_gain}")
        
        grid_encoder.to(device)
        grid_encoder.eval()

    # Helper: safe env factory that passes only supported kwargs
    def _make_env(env_cls, *, size: int, speed: int, seed: int, observation_size: int, time_penalty: float, input_type: str | None, encoder=None, fwhm_ratio: float = 0.0, use_headings: bool = False):
        if env_cls is GridWMEnv:
            env = env_cls(size=size, speed=speed, seed=seed, time_penalty=time_penalty, observation_size=observation_size, input_type=input_type, encoder=encoder, fwhm_ratio=fwhm_ratio, use_headings=use_headings)
            if encoder is not None:
                env._encoder_device = torch.device(device)
            return env
        else:
            return WMEnv(size=size, speed=speed, seed=seed, time_penalty=time_penalty, observation_size=observation_size, use_headings=use_headings)

    # Initialize a pool of environments with fixed goals per env
    seed = args.seed
    size = args.size
    speed = args.speed
    num_envs = args.num_envs
    num_val_envs = args.num_val_envs
    num_train_worlds = args.num_train_worlds
    rng = np.random.RandomState(seed)
    env_pool: List[WMEnv | GridWMEnv] = []

    env_class = GridWMEnv if args.vectorhash else WMEnv

    all_envs = []
    
    # Determine hopfield_gain early: use args if provided, else use encoder's gain, else default to 2.0
    hopfield_gain = args.hopfield_gain
    if hopfield_gain is None:
        hopfield_gain = encoder_gain if encoder_gain is not None else 2.0

    print(f'initializing {num_train_worlds} train world(s) with {num_envs} envs each')
    for world_idx in range(num_train_worlds):
        print(f'  world {world_idx + 1}/{num_train_worlds}')
        world_envs: List[WMEnv | GridWMEnv] = []
        
        for i in range(num_envs):
            env_i = _make_env(
                env_class,
                size=size,
                speed=speed,
                seed=int(rng.randint(0, 10_000_000)),
                time_penalty=args.time_penalty,
                observation_size=args.observation_size,
                input_type=(args.input_type if args.vectorhash else None),
                encoder=grid_encoder,
                fwhm_ratio=args.fwhm_ratio,
                use_headings=args.use_headings,
            )
            world_envs.append(env_i)
        
        # Each world gets its own VectorHash (and Hopfield if enabled)
        if args.vectorhash:
            use_hopfield = (args.input_addendum == "hopfield")
            world_vectorhash = VectorHash(
                Np=args.Np,
                Npos=args.Npos,
                lambdas=args.lambdas, 
                size=size,
                use_hopfield=use_hopfield,
                hopfield_gain=hopfield_gain,
                hopfield_alpha=args.hopfield_alpha,
                hopfield_steps=args.hopfield_steps,
                use_headings=args.use_headings,
            )
            world_vectorhash.initiate_vectorhash(world_envs)
        
        env_pool.extend(world_envs)
        all_envs.extend(world_envs)

    # Position validation pool: new env instances but with same goals as training envs
    print('initializing pos val envs')
    pos_env_pool: List[WMEnv | GridWMEnv] = []
    for i in range(num_val_envs):
        env_i = _make_env(
            env_class,
            size=size,
            speed=speed,
            seed=int(rng.randint(0, 10_000_000)),
            time_penalty=args.time_penalty,
            observation_size=args.observation_size,
            input_type=(args.input_type if args.vectorhash else None),
            encoder=grid_encoder,
            fwhm_ratio=args.fwhm_ratio,
            use_headings=args.use_headings,
        )
        # copy goal from training envs (use modulo in case num_val_envs > total train envs)
        env_i._goal = env_pool[i % len(env_pool)]._goal
        pos_env_pool.append(env_i)
    
    if args.vectorhash:
        use_hopfield = (args.input_addendum == "hopfield")
        pos_vectorhash = VectorHash(
            Np=args.Np,
            Npos=args.Npos,
            lambdas=args.lambdas, 
            size=size,
            use_hopfield=use_hopfield,
            hopfield_gain=hopfield_gain,
            hopfield_alpha=args.hopfield_alpha,
            hopfield_steps=args.hopfield_steps,
            use_headings=args.use_headings,
        )
        pos_vectorhash.initiate_vectorhash(pos_env_pool)
    all_envs.extend(pos_env_pool)

    # New goal validation pool: new env instances with new goals
    print('initializing new goal val envs')
    new_env_pool: List[WMEnv | GridWMEnv] = []
    for i in range(num_val_envs):
        env_i = _make_env(
            env_class,
            size=size,
            speed=speed,
            seed=int(rng.randint(0, 10_000_000)),
            time_penalty=args.time_penalty,
            observation_size=args.observation_size,
            input_type=(args.input_type if args.vectorhash else None),
            encoder=grid_encoder,
            fwhm_ratio=args.fwhm_ratio,
            use_headings=args.use_headings,
        )
        new_env_pool.append(env_i)
    
    if args.vectorhash:
        use_hopfield = (args.input_addendum == "hopfield")
        new_vectorhash = VectorHash(
            Np=args.Np,
            Npos=args.Npos,
            lambdas=args.lambdas, 
            size=size,
            use_hopfield=use_hopfield,
            hopfield_gain=hopfield_gain,
            hopfield_alpha=args.hopfield_alpha,
            hopfield_steps=args.hopfield_steps,
            use_headings=args.use_headings,
        )
        new_vectorhash.initiate_vectorhash(new_env_pool)
    all_envs.extend(new_env_pool)

    # Determine model input size
    if args.vectorhash:
        input_size = env_pool[0].get_input_size()
    else:
        # Raw WMEnv observation size
        input_size = env_pool[0]._observation_size
    if args.input_addendum in ("goal", "next_best", "hopfield"):
        input_size = input_size * 2
    if args.ppo_input_reward:
        input_size = input_size + 1
    
    cfg = TrainConfig(
        size=size,
        speed=speed,
        seed=seed,
        num_envs=num_envs,
        num_val_envs=num_val_envs,
        num_train_worlds=num_train_worlds,
        hidden_size=args.hidden_size,
        num_model_layers=args.num_model_layers,
        batch_episodes=args.batch_episodes,
        steps_per_episode=args.steps_per_episode,
        n_epochs=args.n_epochs,
        val_batch_episodes=args.val_batch_episodes,
        plot_every=args.plot_every,
        val_epochs=args.val_epochs,
        lr=args.lr,
        device=device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        time_penalty=args.time_penalty,
        input_size=input_size,
        input_addendum=(None if args.input_addendum == "none" else args.input_addendum),
        input_type=args.input_type,
        model_class=args.model_class,
        encoder_dim=args.encoder_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_actions=args.num_actions,
        dropout=args.dropout,
        train_method=args.train_method,
        ppo_clip=args.ppo_clip,
        ppo_vf_coef=args.ppo_vf_coef,
        ppo_ent_coef=args.ppo_ent_coef,
        ppo_epochs=args.ppo_epochs,
        max_envs_per_epoch=args.max_envs_per_epoch,
        use_preconv_codebook=args.use_preconv_codebook,
        ppo_input_reward=args.ppo_input_reward,
        lambdas=args.lambdas,
        Npos=args.Npos,
        encoder_gain=encoder_gain,
        hopfield_gain=hopfield_gain,
        hopfield_alpha=args.hopfield_alpha,
        expert_actions=args.expert_actions,
    )

    train(
        env_pool=env_pool,
        pos_env_pool=pos_env_pool,
        new_env_pool=new_env_pool,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()


