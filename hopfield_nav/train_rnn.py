"""Train the vanilla-RNN BC baseline (no Hopfield, no goal storing).

Three modes:
  - sequential : continual learning. Train one env at a time; after each env
    finishes, eval nav_det on every env trained so far (forgetting curve).
  - mixed      : pretraining. Pool rollouts from all envs every update.
  - finetune   : load a checkpoint, then run sequential.

Each env is a distinct ``GridEnv`` instance with its own codebook and goal
(seeded independently). The agent observes only sensory; the goal is never
exposed in the input. The teacher (BFS) sees the goal; the student does not.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

from .agent_rnn import RNNAgent, compute_rnn_input_dim
from .bc_rnn import bc_rnn_update
from .config import EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig, VectorHashConfig
from .env import GridEnv, warn_if_offcell_stores
from .eval_rnn import evaluate_nav_all
from .final_plotting.plotting import (  # noqa: F401  (re-export for regen_plots.py)
    save_forgetting_plot, save_steps_to_goal_plot,
)
from .rollout_rnn import collect_rollout_rnn
from .utils import smooth_gbook
from .vec_env import ContinuousVecEnv, VecEnv
from .vectorhash import VectorHash


def restore_arch_from_ckpt(cfg: RNNTrainConfig, ckpt: dict) -> None:
    """Auto-restore architecture-affecting fields from a saved ckpt's cfg dict.

    Mutates ``cfg`` in place. Prints a NOTE for each field where the CLI value
    is being overridden. Restores fields whose values affect either the agent's
    parameter shapes (movement_mode, hidden_size, num_rnn_layers, input_prev_*,
    input_grid_state) or the gbook lookup encoding (lambdas, fwhm_ratio).
    Fields the ckpt doesn't have (legacy ckpts) are left as-is.
    """
    saved = ckpt.get("cfg", {}) or {}
    saved_agent = saved.get("agent", {}) or {}

    def _restore(obj, attr: str, saved_value, label: str) -> None:
        if saved_value is None:
            return
        cur = getattr(obj, attr)
        if cur != saved_value:
            print(f"  NOTE: --{label} {cur!r} ignored; using ckpt's {saved_value!r}")
        setattr(obj, attr, saved_value)

    _restore(cfg.agent, "movement_mode",     saved_agent.get("movement_mode"),     "movement_mode")
    _restore(cfg.agent, "hidden_size",       saved_agent.get("hidden_size"),       "hidden_size")
    _restore(cfg.agent, "num_rnn_layers",    saved_agent.get("num_rnn_layers"),    "num_rnn_layers")
    _restore(cfg.agent, "dropout",           saved_agent.get("dropout"),           "dropout")
    _restore(cfg.agent, "input_prev_action", saved_agent.get("input_prev_action"), "input_prev_action")
    _restore(cfg.agent, "input_prev_reward", saved_agent.get("input_prev_reward"), "input_prev_reward")
    _restore(cfg.agent, "input_grid_state",  saved_agent.get("input_grid_state"),  "input_grid_state")
    # Env-side movement_mode must mirror agent-side (VecEnv vs ContinuousVecEnv).
    cfg.env.movement_mode = cfg.agent.movement_mode
    _restore(cfg, "lambdas",    saved.get("lambdas"),    "lambdas")
    _restore(cfg, "fwhm_ratio", saved.get("fwhm_ratio"), "fwhm_ratio")


def build_envs(cfg: RNNTrainConfig, rng: np.random.RandomState) -> list[GridEnv]:
    """One GridEnv per seed; each gets its own codebook + goal."""
    envs: list[GridEnv] = []
    for _ in range(cfg.n_envs):
        seed = int(rng.randint(0, 10_000_000))
        # Continuous-mode env factory left as a future extension; sequential
        # currently uses the discrete VecEnv path.
        envs.append(GridEnv(
            size=cfg.env.size,
            speed=cfg.env.speed,
            observation_size=cfg.env.observation_size,
            seed=seed,
            time_penalty=cfg.env.time_penalty,
            goals_active=cfg.env.goals_active,
            # goal_reward and goal_radius were previously left at the GridEnv
            # defaults, so VecEnv (which reads them off the base env) ignored
            # the configured values.
            goal_reward=cfg.env.goal_reward,
            goal_radius=cfg.env.goal_radius,
        ))
    return envs


def _make_vec(env: GridEnv, batch: int, movement_mode: str,
              continuous_scale: float,
              continuous_normalize: bool = False) -> VecEnv | ContinuousVecEnv:
    if movement_mode == "continuous":
        vec = ContinuousVecEnv(
            env, batch_size=batch, scale=continuous_scale,
            normalize=continuous_normalize,
        )
    else:
        vec = VecEnv(env, batch_size=batch)
    vec.reset_all()
    return vec


def train_sequential(
    cfg: RNNTrainConfig,
    agent: RNNAgent,
    optimizer: torch.optim.Optimizer,
    envs: list[GridEnv],
    device: torch.device,
    wandb_run=None,
    sgb: np.ndarray | None = None,
    env_offsets: list[tuple[int, int]] | None = None,
) -> dict:
    """Train envs sequentially. After every update, eval nav_det on EVERY env
    (not just trained-so-far) so the forgetting plot has complete coverage.

    Returns a dict with:
      - "trace": list of (global_step, training_env_idx, {env_idx: nav_det})
      - "blocks": list of (start_step_inclusive, end_step_inclusive, env_idx)
    """
    movement_mode = cfg.agent.movement_mode
    n_envs = len(envs)
    trace: list[tuple[int, int, dict[int, float]]] = []
    blocks: list[tuple[int, int, int]] = []
    global_step = 0

    for i, env in enumerate(envs):
        print(f"\n=== Sequential env {i}/{n_envs - 1}  goal={env.goal_location} ===")
        block_start = global_step + 1
        vec = _make_vec(env, cfg.batch_envs, movement_mode, cfg.env.continuous_scale,
                       continuous_normalize=cfg.env.continuous_normalize)
        env_offset_i = env_offsets[i] if env_offsets is not None else None

        for upd in range(1, cfg.updates_per_env + 1):
            vec.reset_all()
            rollout = collect_rollout_rnn(
                vec, agent, cfg.agent, cfg.steps_per_rollout, device,
                deterministic=False, teacher_force=False,
                sgb=sgb, env_offset=env_offset_i,
            )
            losses = bc_rnn_update(agent, [rollout], cfg.bc, optimizer, movement_mode)
            global_step += 1

            # Per-update eval on every env that has started training (envs
            # 0..i). Untrained envs are excluded — both to save compute and
            # so the plot doesn't show pre-training noise lines.
            metrics = evaluate_nav_all(
                envs[: i + 1], agent, cfg.n_eval_trials, cfg.eval_max_steps,
                device, deterministic=True,
                continuous_scale=cfg.env.continuous_scale,
                continuous_normalize=cfg.env.continuous_normalize,
                sgb=sgb,
                env_offsets=env_offsets[: i + 1] if env_offsets is not None else None,
            )
            trace.append((global_step, i, metrics))
            per_env_navdet = {j: m["nav_det"] for j, m in metrics.items()}

            if upd == 1 or upd % cfg.eval_every == 0 or upd == cfg.updates_per_env:
                rollout_goal_rate = float(rollout.goal_reached.sum().item()) / max(
                    1, cfg.batch_envs * cfg.steps_per_rollout
                )
                summary = "  ".join(
                    f"e{j}={per_env_navdet[j]:.2f}"
                    for j in sorted(per_env_navdet.keys())
                )
                print(
                    f"  env={i} upd={upd}/{cfg.updates_per_env}  "
                    f"loss={losses['move_loss']:.3f}  "
                    f"ent={losses['move_entropy']:.2f}  "
                    f"goal_rate={rollout_goal_rate:.2f}  |  {summary}"
                )
            if wandb_run is not None:
                log = {
                    f"train/move_loss": losses["move_loss"],
                    f"train/move_entropy": losses["move_entropy"],
                    f"train/training_env": i,
                    "global_step": global_step,
                }
                for j, nd in per_env_navdet.items():
                    log[f"eval/env_{j}/nav_det"] = nd
                wandb_run.log(log)

        blocks.append((block_start, global_step, i))

    return {"trace": trace, "blocks": blocks}


def train_mixed(
    cfg: RNNTrainConfig,
    agent: RNNAgent,
    optimizer: torch.optim.Optimizer,
    envs: list[GridEnv],
    device: torch.device,
    wandb_run=None,
    sgb: np.ndarray | None = None,
    env_offsets: list[tuple[int, int]] | None = None,
) -> list[dict[int, dict[str, float]]]:
    """Pool rollouts from all envs every update (pretraining scaffolding)."""
    movement_mode = cfg.agent.movement_mode
    vecs = [
        _make_vec(env, cfg.batch_envs, movement_mode, cfg.env.continuous_scale,
                       continuous_normalize=cfg.env.continuous_normalize)
        for env in envs
    ]
    history: list[dict[int, dict[str, float]]] = []
    for upd in range(1, cfg.n_updates + 1):
        for vec in vecs:
            vec.reset_all()
        rollouts = [
            collect_rollout_rnn(
                vec, agent, cfg.agent, cfg.steps_per_rollout, device,
                deterministic=False, teacher_force=False,
                sgb=sgb,
                env_offset=env_offsets[k] if env_offsets is not None else None,
            )
            for k, vec in enumerate(vecs)
        ]
        losses = bc_rnn_update(agent, rollouts, cfg.bc, optimizer, movement_mode)
        if upd == 1 or upd % cfg.eval_every == 0 or upd == cfg.n_updates:
            total_goal = sum(float(r.goal_reached.sum().item()) for r in rollouts)
            denom = max(1, len(rollouts) * cfg.batch_envs * cfg.steps_per_rollout)
            print(
                f"  mixed upd={upd}/{cfg.n_updates}  "
                f"move_loss={losses['move_loss']:.4f}  "
                f"ent={losses['move_entropy']:.3f}  "
                f"goal_rate={total_goal / denom:.3f}"
            )
            if wandb_run is not None:
                wandb_run.log({
                    "train/mixed/move_loss": losses["move_loss"],
                    "train/mixed/move_entropy": losses["move_entropy"],
                    "train/mixed/goal_rate": total_goal / denom,
                    "global_step": upd,
                })
            metrics = evaluate_nav_all(
                envs, agent, cfg.n_eval_trials, cfg.eval_max_steps, device,
                deterministic=True, continuous_scale=cfg.env.continuous_scale,
                continuous_normalize=cfg.env.continuous_normalize,
                sgb=sgb, env_offsets=env_offsets,
            )
            history.append(metrics)
            for j, m in metrics.items():
                print(
                    f"    eval env_{j}: nav_det={m['nav_det']:.3f}  "
                    f"steps_to_goal={m['mean_steps_to_goal']:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        f"eval/mixed/env_{j}/nav_det": m["nav_det"],
                        f"eval/mixed/env_{j}/mean_steps_to_goal":
                            m["mean_steps_to_goal"],
                        "global_step": upd,
                    })
    return history


def train(cfg: RNNTrainConfig) -> None:
    warn_if_offcell_stores(cfg.env, where="train_rnn")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    envs = build_envs(cfg, rng)
    print(f"Built {len(envs)} envs (size={cfg.env.size}, "
          f"obs_dim={cfg.env.observation_size})")
    for i, env in enumerate(envs):
        print(f"  env {i}: goal={env.goal_location}")

    # Pre-load checkpoint (if any) so we can auto-restore architecture-
    # affecting fields BEFORE building VectorHash + agent (input_dim depends
    # on Ng = sum lambdas^2; head shape depends on movement_mode; etc.).
    ckpt = None
    if cfg.load_checkpoint:
        print(f"Loading checkpoint from {cfg.load_checkpoint}")
        ckpt = torch.load(cfg.load_checkpoint, map_location=device, weights_only=False)
        restore_arch_from_ckpt(cfg, ckpt)

    sgb = None
    env_offsets: list[tuple[int, int]] | None = None
    gbook_dim = 0
    if cfg.agent.input_grid_state:
        vh_cfg = VectorHashConfig(lambdas=list(cfg.lambdas), static_vectorhash=True)
        vh = VectorHash(vh_cfg, size=cfg.env.size)
        vh.build_scaffold()
        vh.register_envs(envs, placement="spread")
        sgb = smooth_gbook(vh.gbook, vh.lambdas, cfg.fwhm_ratio)
        env_offsets = list(vh.env_offsets)
        gbook_dim = int(vh.Ng)
        print(f"grid_state on  Ng={gbook_dim}  Npos={vh.Npos}  "
              f"lambdas={vh.lambdas}  fwhm_ratio={cfg.fwhm_ratio}")

    input_dim = compute_rnn_input_dim(cfg.agent, cfg.env.observation_size, gbook_dim)
    print(f"RNN input_dim={input_dim} (sensory={cfg.env.observation_size}, "
          f"+prev_action={cfg.agent.input_prev_action}, "
          f"+prev_reward={cfg.agent.input_prev_reward}, "
          f"+grid_state={cfg.agent.input_grid_state})")
    agent = RNNAgent(cfg.agent, input_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr)

    if ckpt is not None:
        agent.load_state_dict(ckpt["agent_state_dict"])
        if "optimizer_state_dict" in ckpt and cfg.mode != "finetune":
            # Preserve optimizer momentum unless we're finetuning (fresh moments).
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    wandb_run = None
    if cfg.use_wandb:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    if cfg.save_dir is None:
        sub = (wandb_run.name if wandb_run is not None and wandb_run.name
               else datetime.now().strftime("%Y%m%d_%H%M%S"))
        cfg.save_dir = os.path.join("checkpoint_rnn", sub)
    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"save_dir={cfg.save_dir}")

    if cfg.mode in ("sequential", "finetune"):
        history = train_sequential(
            cfg, agent, optimizer, envs, device, wandb_run,
            sgb=sgb, env_offsets=env_offsets,
        )
        nd_path = os.path.join(cfg.save_dir, "forgetting.png")
        steps_path = os.path.join(cfg.save_dir, "steps_to_goal.png")
        sw = getattr(cfg, "plot_smooth_window", 1)
        save_forgetting_plot(history, len(envs), nd_path, smooth_window=sw)
        save_steps_to_goal_plot(history, len(envs), steps_path, smooth_window=sw)
        print(f"Saved {nd_path}")
        print(f"Saved {steps_path}")
    elif cfg.mode == "mixed":
        history = train_mixed(
            cfg, agent, optimizer, envs, device, wandb_run,
            sgb=sgb, env_offsets=env_offsets,
        )
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")

    ckpt_path = os.path.join(cfg.save_dir, "final.pt")
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": asdict(cfg),
        "history": history,
        "env_goals": [env.goal_location for env in envs],
    }, ckpt_path)
    print(f"Saved {ckpt_path}")

    if wandb_run is not None:
        import wandb
        wandb.finish()


def main() -> None:
    p = argparse.ArgumentParser(description="Vanilla-RNN BC continual-learning baseline")
    # Mode + structure
    p.add_argument("--mode", choices=["sequential", "mixed", "finetune"],
                   default="sequential")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--updates_per_env", type=int, default=100)
    p.add_argument("--n_updates", type=int, default=1000,
                   help="Updates for --mode mixed; ignored for sequential/finetune")
    # Env
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--time_penalty", type=float, default=0.01)
    p.add_argument("--movement_mode", choices=["discrete", "continuous"], default="continuous")
    p.add_argument("--goal_radius", type=float, default=0.5,
                   help="Euclidean radius around goal that counts as 'at goal'. "
                        "Default 0.5 reproduces snap-equality on integer-snapped "
                        "positions; larger values fuzz the goal region.")
    p.add_argument("--allow_offcell_store",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Whether a store fired while at goal may write a cell other than the goal's. Only reachable at goal_radius > 0.5, where at_goal tests the float position but embeddings are read at the snapped cell. Default False: the goal cell's embedding is stored instead, so the pattern written is the one navigation will later recall. Pass --allow_offcell_store for the pre-2026-08 behavior.")
    # Agent
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_rnn_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--init_log_std", type=float, default=0.0)
    p.add_argument("--freeze_log_std", action="store_true")
    p.add_argument("--input_prev_action", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--input_prev_reward", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--input_grid_state", action=argparse.BooleanOptionalAction, default=False,
                   help="Append smoothed-gbook column at the agent's GLOBAL "
                        "(local + env_offset) position to the RNN input.")
    p.add_argument("--fwhm_ratio", type=float, default=0.25,
                   help="Spatial smoothing for gbook lookup. Auto-restored from "
                        "checkpoint in finetune mode.")
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12],
                   help="VectorHash module periods. Determines Ng (=sum lambdas^2) "
                        "and Npos (=prod lambdas). Auto-restored from checkpoint "
                        "in finetune mode.")
    # BC
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--move_ent_coef", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--n_minibatches", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--only_train_on_reached", action="store_true",
                   help="Per BC update, drop trajectories whose rollout never "
                        "reached the goal. If no trajectory reached, the update "
                        "is skipped entirely.")
    # Rollout / eval
    p.add_argument("--batch_envs", type=int, default=16)
    p.add_argument("--steps_per_rollout", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--n_eval_trials", type=int, default=32)
    p.add_argument("--eval_max_steps", type=int, default=64)
    # Bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--load_checkpoint", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="hopfield-nav-rnn")
    p.add_argument("--plot_smooth_window", type=int, default=1,
                   help="Rolling-mean window for the forgetting/steps_to_goal plots (1 = no smoothing).")

    args = p.parse_args()
    cfg = RNNTrainConfig(
        env=EnvConfig(
            size=args.size, observation_size=args.observation_size,
            time_penalty=args.time_penalty, movement_mode=args.movement_mode,
            goal_radius=args.goal_radius,
            allow_offcell_store=args.allow_offcell_store,
        ),
        agent=RNNAgentConfig(
            hidden_size=args.hidden_size, num_rnn_layers=args.num_rnn_layers,
            dropout=args.dropout, movement_mode=args.movement_mode,
            init_log_std=args.init_log_std, freeze_log_std=args.freeze_log_std,
            input_prev_action=args.input_prev_action,
            input_prev_reward=args.input_prev_reward,
            input_grid_state=args.input_grid_state,
        ),
        bc=RNNBCConfig(
            lr=args.lr, move_ent_coef=args.move_ent_coef,
            epochs=args.epochs, n_minibatches=args.n_minibatches,
            max_grad_norm=args.max_grad_norm,
            only_train_on_reached=args.only_train_on_reached,
        ),
        mode=args.mode,
        n_envs=args.n_envs,
        updates_per_env=args.updates_per_env,
        n_updates=args.n_updates,
        batch_envs=args.batch_envs,
        steps_per_rollout=args.steps_per_rollout,
        eval_every=args.eval_every,
        n_eval_trials=args.n_eval_trials,
        eval_max_steps=args.eval_max_steps,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        load_checkpoint=args.load_checkpoint,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        plot_smooth_window=args.plot_smooth_window,
        fwhm_ratio=args.fwhm_ratio,
        lambdas=list(args.lambdas),
    )
    train(cfg)


if __name__ == "__main__":
    main()
