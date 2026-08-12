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

import numpy as np
import torch

from cls_paths import run_dir, run_name
import run_manifest
from .policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from .policy.recurrent import add_recurrent_args
from .updates.bc_rnn import bc_rnn_update
from .config import EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig, VectorHashConfig
from .world.env import GridEnv, warn_if_offcell_stores
from .evaluation.rnn import evaluate_nav_all
from .rollout.rnn import collect_rollout_rnn
from .training.rnn_sequential import UpdateResult, run_sequential_blocks
from .training.rnn_setup import (
    build_envs_from_config, restore_arch_from_ckpt, rnn_world,
    write_rnn_world_spec)
from .utils import smooth_gbook
from .world.vec_env import ContinuousVecEnv, VecEnv, make_vec
from .world.scaffold import VectorHash, place_envs


def _make_vec(env: GridEnv, batch: int, movement_mode: str,
              continuous_scale: float,
              continuous_normalize: bool = False) -> VecEnv | ContinuousVecEnv:
    """Deprecated alias for vec_env.make_vec; kept for existing importers."""
    return make_vec(env, batch, movement_mode, continuous_scale,
                    continuous_normalize, reset=True)


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
    n_envs = len(envs)
    trace: list[tuple[int, int, dict[int, float]]] = []

    def _announce(i: int, env: GridEnv) -> None:
        print(f"\n=== Sequential env {i}/{n_envs - 1}  goal={env.goal_location} ===")

    def _record(u: UpdateResult) -> None:
        trace.append((u.global_step, u.block, u.metrics))
        per_env_navdet = {j: m["nav_det"] for j, m in u.metrics.items()}

        if (u.update == 1 or u.update % cfg.eval_every == 0
                or u.update == cfg.updates_per_env):
            rollout_goal_rate = float(u.rollout.goal_reached.sum().item()) / max(
                1, cfg.batch_envs * cfg.steps_per_rollout
            )
            summary = "  ".join(
                f"e{j}={per_env_navdet[j]:.2f}"
                for j in sorted(per_env_navdet.keys())
            )
            print(
                f"  env={u.block} upd={u.update}/{cfg.updates_per_env}  "
                f"loss={u.losses['move_loss']:.3f}  "
                f"ent={u.losses['move_entropy']:.2f}  "
                f"goal_rate={rollout_goal_rate:.2f}  |  {summary}"
            )
        if wandb_run is not None:
            log = {
                "train/move_loss": u.losses["move_loss"],
                "train/move_entropy": u.losses["move_entropy"],
                "train/training_env": u.block,
                "global_step": u.global_step,
            }
            for j, nd in per_env_navdet.items():
                log[f"eval/env_{j}/nav_det"] = nd
            wandb_run.log(log)

    blocks = run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=optimizer, envs=envs, device=device,
        n_eval_trials=cfg.n_eval_trials, sgb=sgb, env_offsets=env_offsets,
        on_update=_record, on_block_start=_announce,
    )

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

    # Pre-load checkpoint (if any) so we can auto-restore architecture-
    # affecting fields BEFORE building VectorHash + agent (input_dim depends
    # on Ng = sum lambdas^2; head shape depends on movement_mode; etc.).
    # The world is built after this too, because `input_grid_state` and
    # `lambdas` decide whether there is a scaffold to place in at all. The
    # restore consumes no randomness, so the legacy draw is unchanged by the
    # reorder.
    ckpt = None
    if cfg.load_checkpoint:
        print(f"Loading checkpoint from {cfg.load_checkpoint}")
        ckpt = torch.load(cfg.load_checkpoint, map_location=device, weights_only=False)
        restore_arch_from_ckpt(cfg, ckpt)

    envs, env_offsets, world_split, vh, world_kind = rnn_world(
        cfg, rng, want_offsets=bool(cfg.agent.input_grid_state))
    print(f"Built {len(envs)} envs (size={cfg.env.size}, "
          f"obs_dim={cfg.env.observation_size}, world={world_kind})")
    for i, env in enumerate(envs):
        print(f"  env {i}: goal={env.goal_location}")

    sgb = None
    gbook_dim = 0
    if cfg.agent.input_grid_state:
        sgb = smooth_gbook(vh.gbook, vh.lambdas, cfg.fwhm_ratio)
        gbook_dim = int(vh.Ng)
        print(f"grid_state on  Ng={gbook_dim}  Npos={vh.Npos}  "
              f"lambdas={vh.lambdas}  fwhm_ratio={cfg.fwhm_ratio}")
        for i, off in enumerate(env_offsets):
            print(f"  env {i}: offset={off}")

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
        sub = run_name(wandb_run.name if wandb_run is not None else None)
        cfg.save_dir = str(run_dir("rnn", sub))
    else:
        sub = os.path.basename(str(cfg.save_dir).rstrip("/"))
    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"save_dir={cfg.save_dir}")
    # Written on both paths, same file and same reader as train_navigate: a run
    # has to be able to say which envs it used.
    write_rnn_world_spec(cfg, world_split, vh, generator=world_kind,
                         save_dir=cfg.save_dir)

    # The RNN baseline has no encoder: it reads sensory input straight from the
    # env codebook, which is the point of the comparison.
    run_manifest.begin(
        cfg.save_dir, kind="rnn", name=sub, config=asdict(cfg),
        parent=cfg.load_checkpoint, wandb_run=wandb_run,
    )

    if cfg.mode in ("sequential", "finetune"):
        history = train_sequential(
            cfg, agent, optimizer, envs, device, wandb_run,
            sgb=sgb, env_offsets=env_offsets,
        )
        # Imported here, not at module scope: `analysis` is the layer above
        # training, and a top-level import would make matplotlib and the whole
        # figure stack a dependency of every training run. See
        # tests/test_layering.py.
        from analysis.continual.plotting import (
            save_forgetting_plot, save_steps_to_goal_plot,
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
    run_manifest.record_checkpoint(cfg.save_dir, "final.pt")
    run_manifest.finish(cfg.save_dir)
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
    # The declared-domain surface, matching train_navigate so a baseline and an
    # agent-hash run can be handed the same world.json.
    p.add_argument("--env_generator", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Draw envs from declared domains and record them, "
                        "instead of the historical placement path. Needs "
                        "--input_grid_state and an explicit --place_margin.")
    p.add_argument("--place_region", type=str, default="anywhere",
                   help="'anywhere' or 'rect:X0,Y0,W,H'.")
    p.add_argument("--goal_region", type=str, default="any",
                   help="'any', 'ring:W', 'interior:W' or 'quadrant:Q'.")
    p.add_argument("--wall_seeds", type=str, default="0,10000000",
                   help="'LO,HI' range wall seeds are drawn from.")
    p.add_argument("--place_margin", type=int, default=None,
                   help="Edge-to-edge clearance between every pair of envs. "
                        "Required with --env_generator: deriving one needs an "
                        "encoder and this stack has none.")
    p.add_argument("--goal_val_frac", type=float, default=0.2,
                   help="Share of goal cells reserved for validation.")
    p.add_argument("--n_val_envs", type=int, default=2,
                   help="Held-out envs recorded in world.json alongside the "
                        "train set.")
    # Agent
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_rnn_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    add_recurrent_args(p)
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
            rnn_cell=args.rnn_cell, rnn_nonlinearity=args.rnn_nonlinearity,
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
        env_generator=args.env_generator,
        place_region=args.place_region,
        goal_region=args.goal_region,
        wall_seeds=args.wall_seeds,
        place_margin=args.place_margin,
        goal_val_frac=args.goal_val_frac,
        n_val_envs=args.n_val_envs,
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
