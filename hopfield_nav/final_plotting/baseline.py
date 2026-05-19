"""Generate a continual-learning history JSON from the RNN baseline.

Two modes (chosen automatically):
  - ``--load_checkpoint`` set         → finetune (load weights, fresh Adam moments)
  - otherwise                         → sequential from scratch

For every BC update, evaluates a single deterministic trial per env (n_trials=1)
and records 0/1 reached + step-count in the history. Smoothing happens at plot
time, not here.

Output JSON shape: see ``hopfield_nav/final_plotting/plotting.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from ..agent_rnn import RNNAgent, compute_rnn_input_dim
from ..bc_rnn import bc_rnn_update
from ..config import EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig, VectorHashConfig
from ..env import GridEnv
from ..eval_rnn import evaluate_nav_all
from ..rollout_rnn import collect_rollout_rnn
from ..train_rnn import _make_vec, build_envs, restore_arch_from_ckpt
from ..utils import smooth_gbook
from ..vectorhash import VectorHash


def _to_emit_metrics(m: dict) -> dict:
    """`evaluate_nav_one_env` (with n_trials=1) → standardized history fields.

    nav_det ∈ {0.0, 1.0}                → reached  ∈ {0, 1}
    mean_steps_to_goal ∈ int | NaN      → steps_to_goal ∈ int | None
    mean_path_to_goal ∈ float | NaN     → path_to_goal ∈ float | None
    """
    reached = int(round(float(m["nav_det"])))
    sg = float(m["mean_steps_to_goal"])
    steps_to_goal = None if math.isnan(sg) else int(round(sg))
    pg = float(m["mean_path_to_goal"])
    path_to_goal = None if math.isnan(pg) else float(pg)
    return {
        "reached": reached,
        "steps_to_goal": steps_to_goal,
        "path_to_goal": path_to_goal,
    }


def _merge_iter_traces(
    iter_traces: list[tuple[list, list]],
) -> tuple[list, list]:
    """Merge N (trace, blocks) tuples into one combined (trace, blocks).

    For each (step, env_idx, metric_key), the merged value is a length-N list
    aligned to the iter axis. Inner-dict keys are unioned across iters; missing
    iters contribute None at the corresponding list position. Block structure
    is deterministic from cfg, so iter 0's blocks are reused. If
    num_full_iters == 1 the metrics are written as scalars (legacy schema).
    """
    if not iter_traces:
        return [], []
    n_iters = len(iter_traces)
    first_trace, first_blocks = iter_traces[0]
    n_steps = len(first_trace)
    for k, (tr, _) in enumerate(iter_traces):
        if len(tr) != n_steps:
            raise RuntimeError(
                f"iter {k} has {len(tr)} steps, expected {n_steps} (block "
                "structure must be deterministic across iters)"
            )

    combined: list[tuple[int, int, dict[int, dict]]] = []
    for s_idx in range(n_steps):
        step = first_trace[s_idx][0]
        train_env = first_trace[s_idx][1]
        envs_seen: set[int] = set()
        keys_seen: set[str] = set()
        for tr, _ in iter_traces:
            for env_j, m in tr[s_idx][2].items():
                envs_seen.add(env_j)
                if m is not None:
                    keys_seen.update(m.keys())
        inner: dict[int, dict] = {}
        for env_j in envs_seen:
            inner_metrics: dict[str, list | object] = {}
            for key in keys_seen:
                values = []
                for tr, _ in iter_traces:
                    m = tr[s_idx][2].get(env_j)
                    values.append(m.get(key) if m is not None else None)
                inner_metrics[key] = values[0] if n_iters == 1 else values
            inner[env_j] = inner_metrics
        combined.append((step, train_env, inner))
    return combined, first_blocks


def run_sequential(
    cfg: RNNTrainConfig,
    agent: RNNAgent,
    optimizer: torch.optim.Optimizer,
    envs: list[GridEnv],
    device: torch.device,
    sgb: np.ndarray | None = None,
    env_offsets: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int, dict[int, dict]]], list[tuple[int, int, int]]]:
    """One block per env. Per update: collect rollout, BC update, single-trial
    eval on every env trained so far (untrained envs are NOT evaluated — they'd
    just inject pre-training noise into the curve).

    Returns (trace, blocks). `blocks` end is inclusive.
    """
    movement_mode = cfg.agent.movement_mode
    trace: list[tuple[int, int, dict[int, dict]]] = []
    blocks: list[tuple[int, int, int]] = []
    global_step = 0

    for i, env in enumerate(envs):
        block_start = global_step + 1
        vec = _make_vec(
            env, cfg.batch_envs, movement_mode,
            cfg.env.continuous_scale,
            continuous_normalize=cfg.env.continuous_normalize,
        )
        env_offset_i = env_offsets[i] if env_offsets is not None else None
        for upd in range(cfg.updates_per_env):
            vec.reset_all()
            rollout = collect_rollout_rnn(
                vec, agent, cfg.agent, cfg.steps_per_rollout, device,
                deterministic=False, teacher_force=False,
                sgb=sgb, env_offset=env_offset_i,
            )
            bc_rnn_update(agent, [rollout], cfg.bc, optimizer, movement_mode)
            global_step += 1

            metrics = evaluate_nav_all(
                envs[: i + 1], agent,
                n_trials=1,                              # single trial → raw 0/1
                max_steps=cfg.eval_max_steps,
                device=device, deterministic=True,
                continuous_scale=cfg.env.continuous_scale,
                continuous_normalize=cfg.env.continuous_normalize,
                sgb=sgb,
                env_offsets=env_offsets[: i + 1] if env_offsets is not None else None,
            )
            inner = {j: _to_emit_metrics(m) for j, m in metrics.items()}
            trace.append((global_step, i, inner))

            if upd == 0 or (upd + 1) % 25 == 0 or upd + 1 == cfg.updates_per_env:
                summary = "  ".join(
                    f"e{j}={inner[j]['reached']}" for j in sorted(inner.keys())
                )
                print(f"  env={i} upd={upd + 1}/{cfg.updates_per_env}  {summary}")

        blocks.append((block_start, global_step, i))

    return trace, blocks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # I/O
    p.add_argument("--out", required=True,
                   help="Output history JSON path.")
    p.add_argument("--run_name", default=None,
                   help="Recorded in metadata.run_name. Default: derived from --out.")
    p.add_argument("--load_checkpoint", default=None,
                   help="If set: finetune mode. Loads agent_state_dict from this path.")
    # Protocol structure (shared flag names with agenthash.py)
    p.add_argument("--n_envs", type=int, default=4,
                   help="Number of distinct GridEnvs in the protocol.")
    p.add_argument("--iters_per_block", type=int, default=100,
                   help="BC updates per env-block.")
    p.add_argument("--max_steps", type=int, default=64,
                   help="Per-trial step cap during eval.")
    p.add_argument("--seed", type=int, default=0,
                   help="Master seed (base seed; iter k uses seed + k).")
    p.add_argument("--num_full_iters", type=int, default=1,
                   help="Run the entire protocol N times with seeds [seed, "
                        "seed+1, ..., seed+N-1]; the plotter aggregates as "
                        "mean ± 1σ across iters. N=1 reproduces the single-run "
                        "scalar history.")
    p.add_argument("--device", default="cuda",
                   help="Torch device.")
    # Env
    p.add_argument("--size", type=int, default=8,
                   help="Grid size.")
    p.add_argument("--observation_size", type=int, default=60,
                   help="Sensory codebook dim.")
    p.add_argument("--movement_mode", choices=["discrete", "continuous"],
                   default="continuous")
    p.add_argument("--goal_radius", type=float, default=0.5,
                   help="Euclidean radius around goal that counts as 'at goal'. "
                        "Default 0.5 reproduces snap-equality on integer-snapped "
                        "positions; larger values fuzz the goal region.")
    # Agent
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_rnn_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="GRU dropout (only effective with num_rnn_layers > 1).")
    p.add_argument("--input_prev_action", action="store_true")
    p.add_argument("--input_prev_reward", action="store_true")
    p.add_argument("--input_grid_state", action="store_true",
                   help="Append the smoothed-gbook column at the agent's GLOBAL "
                        "position to the RNN input. Requires VectorHash + smoothing.")
    p.add_argument("--fwhm_ratio", type=float, default=0.25,
                   help="Spatial smoothing parameter for smooth_gbook. "
                        "Auto-restored from checkpoint in finetune mode.")
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12],
                   help="VectorHash module periods. Determines Ng (=sum lambdas^2) "
                        "and Npos (=prod lambdas). Auto-restored from checkpoint "
                        "in finetune mode.")
    # BC
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--move_ent_coef", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=4,
                   help="BC epochs per update.")
    p.add_argument("--n_minibatches", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--only_train_on_reached", action="store_true",
                   help="Per BC update, drop trajectories whose rollout never "
                        "reached the goal. If no trajectory reached, the update "
                        "is skipped entirely.")
    # Rollout
    p.add_argument("--batch_envs", type=int, default=16,
                   help="Parallel rollouts per env per update.")
    p.add_argument("--steps_per_rollout", type=int, default=None)
    args = p.parse_args()

    # If steps_per_rollout is not set, set it to max_steps
    if args.steps_per_rollout is None:
        args.steps_per_rollout = args.max_steps

    cfg = RNNTrainConfig(
        env=EnvConfig(
            size=args.size, observation_size=args.observation_size,
            movement_mode=args.movement_mode,
            goal_radius=args.goal_radius,
        ),
        agent=RNNAgentConfig(
            hidden_size=args.hidden_size, num_rnn_layers=args.num_rnn_layers,
            dropout=args.dropout, movement_mode=args.movement_mode,
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
        n_envs=args.n_envs,
        updates_per_env=args.iters_per_block,
        batch_envs=args.batch_envs,
        steps_per_rollout=args.steps_per_rollout,
        n_eval_trials=1,
        eval_max_steps=args.max_steps,
        eval_every=1,
        seed=args.seed,
        device=args.device,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Pre-load checkpoint (if any) and auto-restore architecture-affecting
    # fields BEFORE building envs / VectorHash / agent.
    ckpt = None
    if args.load_checkpoint:
        print(f"[baseline] loading {args.load_checkpoint}  (fresh Adam moments)")
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        restore_arch_from_ckpt(cfg, ckpt)

    base_seed = args.seed
    n_iters = max(1, int(args.num_full_iters))
    iter_traces: list[tuple[list, list]] = []
    iter_env_goals: list[list[list[int]]] = []
    iter_env_offsets: list[list[list[int]] | None] = []
    last_vh_lambdas: list[int] = []
    last_vh_Npos = 0
    last_gbook_dim = 0

    for k in range(n_iters):
        seed_k = base_seed + k
        cfg.seed = seed_k
        torch.manual_seed(seed_k)
        np.random.seed(seed_k)
        rng = np.random.RandomState(seed_k)

        envs = build_envs(cfg, rng)
        if n_iters > 1:
            print(f"\n=== iter {k + 1}/{n_iters}  seed={seed_k} ===")
        print(f"[baseline] built {len(envs)} envs  size={cfg.env.size}  "
              f"obs_dim={cfg.env.observation_size}")
        for i, env in enumerate(envs):
            print(f"  env {i}: goal={env.goal_location}")

        sgb = None
        env_offsets: list[tuple[int, int]] | None = None
        gbook_dim = 0
        vh_lambdas: list[int] = []
        vh_Npos = 0
        if cfg.agent.input_grid_state:
            vh_cfg = VectorHashConfig(lambdas=list(cfg.lambdas), static_vectorhash=True)
            vh = VectorHash(vh_cfg, size=cfg.env.size)
            vh.build_scaffold()
            vh.register_envs(envs, placement="spread")
            sgb = smooth_gbook(vh.gbook, vh.lambdas, cfg.fwhm_ratio)
            env_offsets = list(vh.env_offsets)
            gbook_dim = int(vh.Ng)
            vh_lambdas = list(vh.lambdas)
            vh_Npos = int(vh.Npos)
            print(f"[baseline] grid_state on  Ng={gbook_dim}  Npos={vh_Npos}  "
                  f"lambdas={vh_lambdas}  fwhm_ratio={cfg.fwhm_ratio}")
            for i, off in enumerate(env_offsets):
                print(f"  env {i}: offset={off}")

        input_dim = compute_rnn_input_dim(cfg.agent, cfg.env.observation_size, gbook_dim)
        if k == 0:
            print(f"[baseline] RNN input_dim={input_dim}")
        agent = RNNAgent(cfg.agent, input_dim).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr)
        if ckpt is not None:
            agent.load_state_dict(ckpt["agent_state_dict"])

        mode = "finetune" if ckpt is not None else "sequential"
        if k == 0:
            print(f"[baseline] mode={mode}  iters_per_block={cfg.updates_per_env}  "
                  f"max_steps={cfg.eval_max_steps}  num_full_iters={n_iters}")

        trace, blocks = run_sequential(
            cfg, agent, optimizer, envs, device,
            sgb=sgb, env_offsets=env_offsets,
        )
        iter_traces.append((trace, blocks))
        iter_env_goals.append([list(env.goal_location) for env in envs])
        iter_env_offsets.append(
            [list(o) for o in env_offsets] if env_offsets is not None else None
        )
        last_vh_lambdas = vh_lambdas
        last_vh_Npos = vh_Npos
        last_gbook_dim = gbook_dim

    trace, blocks = _merge_iter_traces(iter_traces)
    # Local rebinds for metadata block below.
    vh_lambdas = last_vh_lambdas
    vh_Npos = last_vh_Npos
    env_goals_per_iter = iter_env_goals
    env_offsets_per_iter = iter_env_offsets

    run_name = args.run_name or os.path.splitext(os.path.basename(args.out))[0]
    history = {
        "metadata": {
            "model_class": "baseline",
            "run_name": run_name,
            "n_envs": cfg.n_envs,
            "env_size": cfg.env.size,
            "iters_per_block": cfg.updates_per_env,
            "max_steps": cfg.eval_max_steps,
            "x_axis_label": "update",
            "raw_metric_is_binary": True,
            "ckpt_path": args.load_checkpoint,
            "num_full_iters": n_iters,
            "extra": {
                "mode": mode,
                "base_seed": base_seed,
                "movement_mode": cfg.agent.movement_mode,
                "hidden_size": cfg.agent.hidden_size,
                "num_rnn_layers": cfg.agent.num_rnn_layers,
                "dropout": cfg.agent.dropout,
                "input_prev_action": cfg.agent.input_prev_action,
                "input_prev_reward": cfg.agent.input_prev_reward,
                "input_grid_state": cfg.agent.input_grid_state,
                "fwhm_ratio": cfg.fwhm_ratio if cfg.agent.input_grid_state else None,
                "vh_lambdas": vh_lambdas if cfg.agent.input_grid_state else None,
                "vh_Npos": vh_Npos if cfg.agent.input_grid_state else None,
                "vh_Ng": last_gbook_dim if cfg.agent.input_grid_state else None,
                "env_offsets_per_iter": env_offsets_per_iter,
                "env_goals_per_iter": env_goals_per_iter,
                "lr": cfg.bc.lr,
                "move_ent_coef": cfg.bc.move_ent_coef,
                "epochs": cfg.bc.epochs,
                "n_minibatches": cfg.bc.n_minibatches,
                "max_grad_norm": cfg.bc.max_grad_norm,
                "batch_envs": cfg.batch_envs,
                "steps_per_rollout": cfg.steps_per_rollout,
                "observation_size": cfg.env.observation_size,
                "seed": cfg.seed,
            },
        },
        "trace": [
            [s, t, {str(k): v for k, v in inner.items()}]
            for s, t, inner in trace
        ],
        "blocks": [list(b) for b in blocks],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[baseline] wrote {args.out}  "
          f"trace={len(trace)} steps  blocks={len(blocks)}")


if __name__ == "__main__":
    main()
