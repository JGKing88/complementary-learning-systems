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
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import wandb

from cls import WMEnv
from cls.models import Agent, GRU
from cls.utils.GridUtils import VectorHash
from cls.envs.environments import GridWMEnv

CARDINAL_ACTIONS: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

@torch.no_grad()
def _policy_action(model: Agent, obs: np.ndarray, h: torch.Tensor | None, device: str, epsilon: float = 0.0) -> tuple[int, torch.Tensor | None]:
    x = torch.from_numpy(obs).to(device).float().view(1, 1, -1)
    logits, h_next = model(x, h)
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
):
    """Roll out an episode using the MODEL's actions; label each step with the env's best action.

    Returns stacked observations (T, F) and integer labels (T,) where labels are
    env.best_action_to_goal at each position.
    """
    obs_list: List[np.ndarray] = []
    label_list: List[int] = []

    env.reset()
    h: torch.Tensor | None = None
    for _ in range(max_steps):

        # Build features and label
        obs = env.obs()
        if input_addendum == "goal":
            obs = np.concatenate([obs, env.obs_at_goal()])
        elif input_addendum == "diff":
            obs = env.obs_at_goal() - env.obs()

        best_vec = env.best_action_to_goal(randomize=randomize_best)
        # Use WMEnv's heading index mapping to maintain consistency
        best_idx = CARDINAL_ACTIONS.index(best_vec) #int(env._heading_index(best_vec))

        obs_list.append(obs)
        label_list.append(best_idx)

        # Step with policy action
        action_idx, h = _policy_action(model, obs, h, device, epsilon)
        vec = CARDINAL_ACTIONS[action_idx]
        cur, _, _, _ = env.step(vec)
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
):
    """Run multiple independent clones of the same env in lockstep and return episodes.

    Each clone shares the codebook (and vectorhash, if present) via `clone()`
    but has independent start/heading state.
    """
    # Create independent clones so we don't mutate the original env
    sub_envs = [env.clone() for _ in range(max(1, int(batch_episodes)))]
    B = len(sub_envs)

    for e in sub_envs:
        e.reset()

    done = [False] * B
    hs: list[torch.Tensor | None] = [None] * B
    buffers_obs: list[list[np.ndarray]] = [[] for _ in range(B)]
    buffers_lbl: list[list[int]] = [[] for _ in range(B)]

    for _ in range(max_steps):
        active_idx = [i for i, d in enumerate(done) if not d]
        if not active_idx:
            break
        obs_batch = []
        best_idx_batch = []
        for i in active_idx:
            obs = sub_envs[i].obs()
            if input_addendum == "goal":
                obs = np.concatenate([obs, sub_envs[i].obs_at_goal()])
            elif input_addendum == "diff":
                obs = sub_envs[i].obs_at_goal() - sub_envs[i].obs()
            obs_batch.append(obs)
            best_vec = sub_envs[i].best_action_to_goal(randomize=randomize_best)
            best_idx_batch.append(CARDINAL_ACTIONS.index(best_vec))

        x = torch.from_numpy(np.stack(obs_batch)).to(device).float().view(len(active_idx), 1, -1)
        # Prepare hidden state with correct shape (num_layers, batch, hidden)
        need_init_h = any(hs[i] is None for i in active_idx)
        h_in = None if need_init_h else torch.stack([hs[i] for i in active_idx], dim=1)
        logits, h_out = model(x, h_in)

        # Greedy actions with optional epsilon-greedy
        pred_indices = torch.argmax(logits[:, 0], dim=-1).tolist()
        if epsilon > 0.0:
            for j in range(len(pred_indices)):
                if np.random.rand() < epsilon:
                    pred_indices[j] = int(np.random.randint(0, len(CARDINAL_ACTIONS)))

        for j, i in enumerate(active_idx):
            buffers_obs[i].append(obs_batch[j])
            buffers_lbl[i].append(best_idx_batch[j])
            vec = CARDINAL_ACTIONS[pred_indices[j]]
            cur, goal, _, _ = sub_envs[i].step(vec)
            hs[i] = h_out[:, j, :] if h_out is not None else None
            if cur == goal:
                done[i] = True

    episodes = [
        (np.stack(buffers_obs[i], axis=0), np.array(buffers_lbl[i], dtype=np.int64))
        for i in range(B)
        if buffers_obs[i]
    ]
    return episodes

def collate_batch(episodes: List[Tuple[np.ndarray, np.ndarray]]):
    """Pad a list of (T, F) observation arrays and (T,) label arrays into a batch.

    Padded labels use -100 (PyTorch CrossEntropy ignore_index) so they do not
    contribute to the loss.
    """
    # episodes: list of (T, F), (T,)
    max_T = max(obs.shape[0] for obs, _ in episodes)
    F = episodes[0][0].shape[1]
    B = len(episodes)
    obs_batch = np.zeros((B, max_T, F), dtype=np.float32)
    tgt_batch = np.full((B, max_T), -100, dtype=np.int64)  # ignore index
    lengths = []
    for i, (obs, tgt) in enumerate(episodes):
        T = obs.shape[0]
        obs_batch[i, :T] = obs
        tgt_batch[i, :T] = tgt
        lengths.append(T)
    return (
        torch.from_numpy(obs_batch),
        torch.from_numpy(tgt_batch),
        torch.tensor(lengths, dtype=torch.int64),
    )


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
        else:
            obs = env.obs()

        x = torch.from_numpy(obs).to(device).float().view(1, 1, -1)
        with torch.no_grad():
            logits, h = model(x, h)
            idx = int(torch.argmax(logits[0, 0]).item())
        vec = CARDINAL_ACTIONS[idx]
        cur, goal, _, _ = env.step(vec)
        positions.append(cur)
        if cur == goal:
            break

    return positions, goal


def rollout_best_episode(env: WMEnv | GridWMEnv, steps: int, do_reset: bool = True) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """Roll out a single episode using the environment's best_action_to_goal."""
    positions: List[Tuple[int, int]] = []
    if do_reset:
        cur, goal, _, _ = env.reset()
    else:
        cur, goal = env.current_location, env.goal_location
    positions.append(cur)

    for _ in range(steps):
        vec = env.best_action_to_goal(randomize=False)
        cur, goal, _, _ = env.step(vec)
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
            )

            if not episodes:
                accs.append(0.0)
                success_counts.append(0)
                continue

            obs_batch, tgt_batch, lengths = collate_batch(episodes)
            obs_batch = obs_batch.to(device).float()
            tgt_batch = tgt_batch.to(device)

            logits, _ = model(obs_batch)
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
    size: int = 8,
    speed: int = 1,
    seed: int = 0,
    num_envs: int = 16,
    hidden_size: int = 128,
    num_layers: int = 1,
    batch_episodes: int = 32,
    steps_per_episode: int = 32,
    n_epochs: int = 100,
    val_batch_episodes: int = 4,
    plot_every: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    use_wandb: bool = False,
    wandb_project: str = "cls",
    time_penalty: float = 0.01,
    input_size: int = 128,
    input_addendum: str | None = None,
    input_type: str = "g_idx",
    model_class: str = "GRU",
    encoder_dim: int | None = None,
    num_actions: int = 4,
    dropout: float = 0.0,
):
    """Main training loop.

    Generates a new batch of episodes each update (on-policy imitation data),
    pads and stacks them, and optimizes the GRU policy with cross entropy.
    Prints running loss and token-level accuracy every 50 updates.
    """

    model = Agent(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, model_class=model_class, encoder_dim=encoder_dim, num_actions=num_actions, dropout=dropout)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    out_dir_base = os.path.join(os.getcwd(), "runs")
    out_dir_train = os.path.join(out_dir_base, "train")
    out_dir_pos_validation = os.path.join(out_dir_base, "pos_validation")
    out_dir_goal_validation = os.path.join(out_dir_base, "goal_validation")
    # Optional Weights & Biases tracking
    if use_wandb:
        cfg = dict(
            size=size,
            speed=speed,
            seed=seed,
            num_envs=num_envs,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_episodes=batch_episodes,
            steps_per_episode=steps_per_episode,
            n_epochs=n_epochs,
            val_batch_episodes=val_batch_episodes,
            plot_every=plot_every,
            lr=lr,
            device=device,
            input_addendum=input_addendum,
            input_type=input_type,
            model_class=model_class,
            encoder_dim=encoder_dim,
        )
        wandb.init(project=wandb_project, config=cfg)

    model.train()
    for epoch in range(1, n_epochs + 1):
        # For each episode in the batch, pick a random environment from the pool
        episodes = []
        # Use model-generated actions to create training data (no grad during rollout)
        model.eval()
        with torch.no_grad():
            for env_idx in range(len(env_pool)):
                env = env_pool[env_idx]
                episodes.extend(
                    generate_episodes_vectorized(
                        env,
                        model,
                        device,
                        steps_per_episode,
                        batch_episodes=batch_episodes,
                        input_addendum=input_addendum,
                    )
                )
        model.train()
        obs, tgt, lengths = collate_batch(episodes)
        obs = obs.to(device)
        tgt = tgt.to(device)

        opt.zero_grad(set_to_none=True)
        # logits shape: (B, T, A)
        logits, _ = model(obs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = tgt != -100
            acc = (pred[mask] == tgt[mask]).float().mean().item()
        print(f"epoch {epoch:04d} | loss {loss.item():.4f} | train acc {acc:.3f}")
        if use_wandb:
            wandb.log({"train/loss": float(loss.item()), "train/acc": float(acc)}, step=epoch)

        # Validation each epoch with overlay plots
        model.eval()
        save_plots = (plot_every > 0 and (epoch % plot_every == 0))
        # Progress on the training env pool (was "val")
        train_acc, train_success = validate(
            model, env_pool, steps_per_episode, val_batch_episodes, device, epoch, out_dir_train, save_plots, label="train", input_addendum=input_addendum
        )

        # Position validation: same goals as train, new env instances
        pos_acc, pos_success = validate(
            model, pos_env_pool, steps_per_episode, val_batch_episodes, device, epoch, out_dir_pos_validation, save_plots, label="pos_validation", input_addendum=input_addendum
        )

        # Goal validation (was "newval"): different goals and env instances
        goal_acc, goal_success = validate(
            model, new_env_pool, steps_per_episode, val_batch_episodes, device, epoch, out_dir_goal_validation, save_plots, label="goal_validation", input_addendum=input_addendum
        )
        model.train()
        if use_wandb:
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

    if use_wandb:
        wandb.finish()

    return model


def main():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=16)

    # Model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--model_class", type=str, default="GRU")
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_episodes", type=int, default=16)
    parser.add_argument("--steps_per_episode", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--val_batch_episodes", type=int, default=4)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="cls")
    parser.add_argument("--time_penalty", type=float, default=0.01)
    parser.add_argument("--observation_size", type=int, default=256)
    parser.add_argument("--input_addendum", type=str, choices=["goal", "diff", "none"], default="none")

    # Vectorhash
    parser.add_argument("--vectorhash", action="store_true", default=False)
    parser.add_argument("--Np", type=int, default=1600)
    parser.add_argument("--lambdas", type=int, default=[4,5,7])
    parser.add_argument("--input_type", type=str, default="g_idx") #g_idx, g_hot, s, or p

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = parser.parse_args()

    if args.input_addendum not in ("goal", "diff", "none"):
        raise ValueError("input_addendum must be one of: goal, diff, none")

    # Initialize a pool of environments with fixed goals per env
    seed = args.seed
    size = args.size
    speed = args.speed
    num_envs = args.num_envs
    rng = np.random.RandomState(seed)
    env_pool: List[WMEnv | GridWMEnv] = []

    if args.vectorhash:
        env_class = GridWMEnv
    else:
        env_class = WMEnv

    all_envs = []

    for i in range(num_envs):
        env_i = env_class(size=size, speed=speed, seed=int(rng.randint(0, 10_000_000)), time_penalty=args.time_penalty, observation_size=args.observation_size, input_type=args.input_type)
        env_pool.append(env_i)
    all_envs.extend(env_pool)

    if args.vectorhash:
        main_vectorhash = VectorHash(Np=args.Np, lambdas=args.lambdas, size=size)
        main_vectorhash.initiate_vectorhash(all_envs)

    # Position validation pool: new env instances but with same goals as training envs
    pos_env_pool: List[WMEnv | GridWMEnv] = []
    for i in range(num_envs):
        env_i = env_class(size=size, speed=speed, seed=int(rng.randint(0, 10_000_000)), time_penalty=args.time_penalty, observation_size=args.observation_size, input_type=args.input_type)
        # copy goal from training envs
        env_i._goal = env_pool[i]._goal
        pos_env_pool.append(env_i)
    all_envs.extend(pos_env_pool)

    if args.vectorhash:
        pos_vectorhash = VectorHash(Np=args.Np, lambdas=args.lambdas, size=size)
        pos_vectorhash.initiate_vectorhash(all_envs)

    # New goal validation pool: new env instances with new goals

    new_env_pool: List[WMEnv | GridWMEnv] = []
    for i in range(num_envs):
        env_i = env_class(size=size, speed=speed, seed=int(rng.randint(0, 10_000_000)), time_penalty=args.time_penalty, observation_size=args.observation_size, input_type=args.input_type)
        new_env_pool.append(env_i)
    all_envs.extend(new_env_pool)
    
    if args.vectorhash:
        new_vectorhash = VectorHash(Np=args.Np, lambdas=args.lambdas, size=size)
        new_vectorhash.initiate_vectorhash(all_envs)
    
    # Not sure if we should hav one vectorhash for all envs
    # if args.vectorhash:
    #     vectorhash = VectorHash(Np=args.Np, lambdas=args.lambdas, size=size)
    #     vectorhash.initiate_vectorhash(all_envs)

    input_size = env_pool[0].get_input_size()
    if args.input_addendum == "goal":
        input_size = input_size * 2
    
    train(
        env_pool=env_pool,
        pos_env_pool=pos_env_pool,
        new_env_pool=new_env_pool,
        size=size,
        speed=speed,
        seed=seed,
        num_envs=num_envs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_episodes=args.batch_episodes,
        steps_per_episode=args.steps_per_episode,
        n_epochs=args.n_epochs,
        val_batch_episodes=args.val_batch_episodes,
        plot_every=args.plot_every,
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
        num_actions=args.num_actions,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()


