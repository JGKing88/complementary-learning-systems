from __future__ import annotations

"""Train an MLP to classify the action given (start_state, next_state).

We build a dataset by sampling random grid positions, headings, and actions
from a single `WMEnv`. For each sample we compute:
  x = concat(obs(start_pos, start_heading), obs(next_pos, next_heading))
  y = action index in {0:N, 1:E, 2:S, 3:W}

Then we train an MLP classifier (via the existing Agent with MLP backbone)
to predict the action index from x.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from cls.envs.environments import WMEnv, GridWMEnv
from cls.models import Agent
from cls.utils.GridUtils import VectorHash


CARDINAL_ACTIONS: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W


def build_dataset(
    env: WMEnv | GridWMEnv,
    num_samples: int,
    rng: np.random.RandomState,
    allowed_positions: list[tuple[int, int]] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y).

    X: (num_samples, F_concat) float32 where F_concat = 2 * observation_size
    y: (num_samples,) int64 action indices 0..3
    """
    size = env.size
    # Feature size depends on env type
    if isinstance(env, GridWMEnv):
        F = env.get_input_size()
    else:
        F = env._observation_size  # raw WMEnv observation size
    X = np.zeros((num_samples, 2 * F), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.int64)

    # Pre-generate random positions, headings, actions
    if allowed_positions is None:
        xs = rng.randint(0, size, size=num_samples)
        ys = rng.randint(0, size, size=num_samples)
        pos_idx = None
        pos_pool = None
    else:
        pos_pool = np.asarray(allowed_positions, dtype=int)
        if pos_pool.ndim != 2 or pos_pool.shape[1] != 2:
            raise ValueError("allowed_positions must have shape (K, 2)")
        pos_idx = rng.randint(0, pos_pool.shape[0], size=num_samples)
        xs = None
        ys = None
    heading_idx = rng.randint(0, 4, size=num_samples)
    action_idx = rng.randint(0, 4, size=num_samples)

    for i in range(num_samples):
        if pos_idx is None:
            pos = (int(xs[i]), int(ys[i]))
        else:
            px, py = pos_pool[int(pos_idx[i])]
            pos = (int(px), int(py))
        heading = CARDINAL_ACTIONS[int(heading_idx[i])]
        a_idx = int(action_idx[i])
        a_vec = CARDINAL_ACTIONS[a_idx]

        # next state by simulating movement (respecting boundaries)
        next_pos = env._simulate_move(pos, a_vec, env.speed)
        moved = (next_pos != pos)
        next_heading = a_vec if moved else heading

        # observations (do not mutate env; use codebook via helper)
        if isinstance(env, GridWMEnv):
            start_raw = env._code_for(pos, heading)
            next_raw = env._code_for(next_pos, next_heading)
            start_obs = env.convert_obs(start_raw)
            next_obs = env.convert_obs(next_raw)
        else:
            start_obs = env._code_for(pos, heading)
            next_obs = env._code_for(next_pos, next_heading)

        X[i] = np.concatenate([start_obs, next_obs]).astype(np.float32)
        y[i] = a_idx

    return X, y


def train_classifier(
    size: int,
    speed: int,
    observation_size: int,
    seed: int,
    num_samples: int,
    val_fraction: float,
    hidden_size: int,
    num_model_layers: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device: str,
    use_grid: bool,
    input_type: str | None,
    Np: int | None,
    lambdas: list[int] | None,
    val_samples_newenv: int | None = None,
    use_wandb: bool = False,
    wandb_project: str = "cls_action_classifier",
):
    all_envs: List[WMEnv | GridWMEnv] = []
    rng = np.random.RandomState(seed)
    if use_grid:
        env = GridWMEnv(size=size, speed=speed, seed=seed, observation_size=observation_size, input_type=input_type or "g_idx")
        # Attach vectorhash so convert_obs works
        vh = VectorHash(Np= int(Np if Np is not None else 1600), lambdas=(lambdas if lambdas is not None else [11,12]), size=size)
        all_envs.append(env)
    else:
        env = WMEnv(size=size, speed=speed, seed=seed, observation_size=observation_size)
    
    # Build new-env validation set (same VectorHash object for grid)
    if use_grid:
        env_new = GridWMEnv(size=size, speed=speed, seed=seed + 1, observation_size=observation_size, input_type=input_type or "g_idx")
        all_envs.append(env_new)
        vh.initiate_vectorhash(all_envs)
    else:
        env_new = WMEnv(size=size, speed=speed, seed=seed + 1, observation_size=observation_size)        

    # Choose disjoint start positions for train and val within the same env
    all_positions = np.array([(x, y) for x in range(size) for y in range(size)], dtype=int)
    rng.shuffle(all_positions)
    n_val_samples = int(val_fraction * num_samples)
    n_tr_samples = max(0, num_samples - n_val_samples)
    # Allocate a fraction of positions to validation to enforce disjoint starts
    n_val_positions = max(1, int(val_fraction * all_positions.shape[0]))
    val_positions = all_positions[:n_val_positions]
    train_positions = all_positions[n_val_positions:]

    # Build train/val datasets using the disjoint position sets
    X_tr, y_tr = build_dataset(env, num_samples=n_tr_samples, rng=rng, allowed_positions=train_positions)
    X_val, y_val = build_dataset(env, num_samples=n_val_samples, rng=rng, allowed_positions=val_positions)

    if val_samples_newenv is None:
        val_samples_newenv = n_val_samples

    X_val_new, y_val_new = build_dataset(env_new, num_samples=val_samples_newenv, rng=rng)

    # Model: use Agent with MLP backbone; sequence length 1
    print(X_tr.shape)
    print(X_val.shape)
    input_size = X_tr.shape[1]
    model = Agent(
        input_size=input_size,
        hidden_size=hidden_size,
        num_model_layers=num_model_layers,
        num_actions=4,
        dropout=0.0,
        model_class="MLP",
        encoder_dim=None,
        num_encoder_layers=0,
    )
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if use_wandb:
        cfg = dict(
            size=size,
            speed=speed,
            observation_size=observation_size,
            seed=seed,
            num_samples=num_samples,
            val_fraction=val_fraction,
            hidden_size=hidden_size,
            num_model_layers=num_model_layers,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            vectorhash=use_grid,
            Np= int(Np) if Np is not None else None,
            lambdas=lambdas,
            input_type=input_type,
        )
        wandb.init(project=wandb_project, config=cfg)

    def run_epoch(Xb: np.ndarray, yb: np.ndarray, train: bool) -> tuple[float, float]:
        model.train(mode=train)
        total_loss = 0.0
        total_correct = 0
        total = 0
        # simple batching
        for start in range(0, Xb.shape[0], batch_size):
            xb = Xb[start : start + batch_size]
            ybt = yb[start : start + batch_size]
            x = torch.from_numpy(xb).to(device).float().view(xb.shape[0], 1, -1)
            t = torch.from_numpy(ybt).to(device).long()
            if train:
                opt.zero_grad(set_to_none=True)
            logits, values, _ = model(x, None)
            loss = criterion(logits[:, 0, :], t)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            with torch.no_grad():
                pred = logits[:, 0, :].argmax(dim=-1)
                total_correct += int((pred == t).sum().item())
                total += int(t.numel())
                total_loss += float(loss.item()) * int(t.numel())
        avg_loss = total_loss / max(1, total)
        acc = total_correct / max(1, total)
        return avg_loss, acc

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = run_epoch(X_tr, y_tr, train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(X_val, y_val, train=False)
            val_new_loss, val_new_acc = run_epoch(X_val_new, y_val_new, train=False)
        print(
            f"epoch {epoch:04d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f} | val(newenv) {val_new_loss:.4f}/{val_new_acc:.3f}"
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": float(tr_loss),
                "train/acc": float(tr_acc),
                "val/loss": float(val_loss),
                "val/acc": float(val_acc),
                "val_new/loss": float(val_new_loss),
                "val_new/acc": float(val_new_acc),
            }, step=epoch)

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--observation_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_model_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Grid / Vectorhash options
    parser.add_argument("--vectorhash", action="store_true", default=False)
    parser.add_argument("--Np", type=int, default=1600)
    parser.add_argument("--lambdas", type=int, nargs="+", default=[11,12])
    parser.add_argument("--input_type", type=str, default="g_idx")  # g_idx, g_hot, s, p
    parser.add_argument("--val_samples_newenv", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="cls_action_classifier")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()

    train_classifier(
        size=args.size,
        speed=args.speed,
        observation_size=args.observation_size,
        seed=args.seed,
        num_samples=args.num_samples,
        val_fraction=args.val_fraction,
        hidden_size=args.hidden_size,
        num_model_layers=args.num_model_layers,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        use_grid=args.vectorhash,
        input_type=(args.input_type if args.vectorhash else None),
        Np=(args.Np if args.vectorhash else None),
        lambdas=(args.lambdas if args.vectorhash else None),
        val_samples_newenv=args.val_samples_newenv,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()


