"""Generate a continual-learning history JSON from the RNN baseline.

Two modes (chosen automatically):
  - ``--load_checkpoint`` set         → finetune (load weights, fresh Adam moments)
  - otherwise                         → sequential from scratch

For every BC update, evaluates a single deterministic trial per env (n_trials=1)
and records 0/1 reached + step-count in the history. Smoothing happens at plot
time, not here.

Output JSON shape: see ``analysis/continual/plotting.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from hopfield_nav.continual.base import (
    CONTINUAL_METHODS, build_method, parse_method_args)
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.policy.hypernet import HNET_BASES, HyperRNNAgent
from hopfield_nav.policy.isolate import MultiHeadRNNAgent, XdGRNNAgent, warm_start
from hopfield_nav.policy.recurrent import add_recurrent_args
from hopfield_nav.config import EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig, VectorHashConfig
from hopfield_nav.world.env import GridEnv
from hopfield_nav.training.rnn_setup import (
    build_envs_from_config, restore_arch_from_ckpt, rnn_world,
    write_rnn_world_spec)
from hopfield_nav.training.rnn_sequential import UpdateResult, run_sequential_blocks
from hopfield_nav.utils import smooth_gbook
from hopfield_nav.world.scaffold import VectorHash, place_envs


def _to_emit_metrics(m: dict) -> dict:
    """`evaluate_nav_one_env` (with n_trials=1) → standardized history fields.

    nav_det ∈ {0.0, 1.0}                → reached  ∈ {0, 1}
    mean_steps_to_goal ∈ int | NaN      → steps_to_goal ∈ int | None
    mean_path_to_goal ∈ float | NaN     → path_to_goal ∈ float | None
    mean_optimal_to_goal ∈ float | NaN  → optimal_to_goal ∈ float | None
    mean_optimal_all ∈ float            → optimal_all ∈ float

    `optimal_to_goal` is the shortest attainable path for this trial and is
    what makes the other two comparable between arms: on its own `path_to_goal`
    is conditioned on success, so an arm that solves only the near goals is
    scored on nearer trials than one that solves the far ones too. Recorded
    from wave 4 on; every history written before that has neither field, and
    readers must treat them as absent rather than zero.
    """
    reached = int(round(float(m["nav_det"])))
    sg = float(m["mean_steps_to_goal"])
    steps_to_goal = None if math.isnan(sg) else int(round(sg))
    pg = float(m["mean_path_to_goal"])
    path_to_goal = None if math.isnan(pg) else float(pg)
    og = float(m["mean_optimal_to_goal"])
    optimal_to_goal = None if math.isnan(og) else float(og)
    return {
        "reached": reached,
        "steps_to_goal": steps_to_goal,
        "path_to_goal": path_to_goal,
        "optimal_to_goal": optimal_to_goal,
        "optimal_all": float(m["mean_optimal_all"]),
    }


def merge_iter_traces(
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


#: Parameter-name prefixes that constitute the movement head. Everything else
#: is "the trunk" for --freeze_trunk's purposes. Kept as a named constant
#: because a silent mismatch here would freeze the head instead and the run
#: would still look plausible -- it would simply learn nothing.
HEAD_PREFIXES = ("movement_head", "movement_mean", "movement_log_std")


def freeze_trunk_params(agent) -> tuple[int, int]:
    """Hold everything but the movement head. Returns (n_frozen, n_trainable).

    Plan section 3.2 P4. Must be called *after* any checkpoint load, so what is
    frozen is the pretrained trunk; freezing before would pin it at
    initialisation, which measures something else entirely.

    An agent may name its head something else -- the multi-head policy keeps a
    `ModuleList` of them -- so it can say so with a `head_prefixes` attribute.
    Without one, `HEAD_PREFIXES` applies, and an agent whose head matches
    neither trips the check at the bottom rather than training nothing.
    """
    prefixes = tuple(getattr(agent, "head_prefixes", HEAD_PREFIXES))
    n_frozen = 0
    for name, prm in agent.named_parameters():
        if not name.startswith(prefixes):
            prm.requires_grad_(False)
            n_frozen += prm.numel()
    n_trainable = sum(prm.numel() for prm in agent.parameters()
                      if prm.requires_grad)
    if n_trainable == 0:
        raise RuntimeError(
            f"freeze_trunk left nothing trainable; head prefixes {prefixes} "
            f"matched no parameter of {type(agent).__name__}")
    return n_frozen, n_trainable


#: Policy architectures the protocol can be run with. `rnn` is the baseline
#: every recorded history to date used; the rest are the isolation family
#: (plan section 4.3), and all three of them need the task id.
ARCHITECTURES: tuple[str, ...] = ("rnn", "hnet", "multihead", "xdg")


def build_arch_agent(args, cfg, input_dim: int, seed: int):
    """The policy `--arch` asks for.

    Kept as one function with one switch because the alternative -- four
    branches spread through `main` -- is how an architecture comes to miss a
    config field that the others get. Everything architecture-specific is here;
    everything shared is in `cfg.agent`.
    """
    if args.arch == "rnn":
        return RNNAgent(cfg.agent, input_dim)
    if args.arch == "hnet":
        return HyperRNNAgent(
            cfg.agent, input_dim, cfg.n_envs,
            emb_dim=args.hnet_emb_dim, chunk_dim=args.hnet_chunk_dim,
            hyper_hidden=tuple(args.hnet_hidden), base=args.hnet_base,
            init_out_scale=args.hnet_init_out_scale)
    if args.arch == "multihead":
        return MultiHeadRNNAgent(cfg.agent, input_dim, cfg.n_envs)
    if args.arch == "xdg":
        return XdGRNNAgent(cfg.agent, input_dim, cfg.n_envs,
                           gating=args.xdg_gating, seed=seed)
    raise ValueError(f"unknown --arch {args.arch!r}; known: {list(ARCHITECTURES)}")


def run_sequential(
    cfg: RNNTrainConfig,
    agent: RNNAgent,
    optimizer: torch.optim.Optimizer,
    envs: list[GridEnv],
    device: torch.device,
    sgb: np.ndarray | None = None,
    env_offsets: list[tuple[int, int]] | None = None,
    method=None,
    reset_optimizer_each_block: bool = False,
) -> tuple[list[tuple[int, int, dict[int, dict]]], list[tuple[int, int, int]]]:
    """One block per env. Per update: collect rollout, BC update, single-trial
    eval on every env trained so far (untrained envs are NOT evaluated — they'd
    just inject pre-training noise into the curve).

    ``method`` is a `hopfield_nav.continual.ContinualMethod`; None means naive
    sequential SGD, which is the floor the suite is measured against.

    Returns (trace, blocks). `blocks` end is inclusive.
    """
    trace: list[tuple[int, int, dict[int, dict]]] = []

    def _record(u: UpdateResult) -> None:
        inner = {j: _to_emit_metrics(m) for j, m in u.metrics.items()}
        trace.append((u.global_step, u.block, inner))
        if (u.update == 1 or u.update % 25 == 0
                or u.update == cfg.updates_per_env):
            summary = "  ".join(
                f"e{j}={inner[j]['reached']}" for j in sorted(inner.keys())
            )
            print(f"  env={u.block} upd={u.update}/{cfg.updates_per_env}  {summary}")

    blocks = run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=optimizer, envs=envs, device=device,
        # A single trial per env per update, so each point is a raw 0/1 rather
        # than an average -- the figure smooths it afterwards.
        n_eval_trials=1,
        sgb=sgb, env_offsets=env_offsets, on_update=_record, method=method,
        reset_optimizer_each_block=reset_optimizer_each_block,
    )

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
    # The declared-domain surface, matching train_navigate so a baseline and an
    # agent-hash run can be handed the same world.json.
    p.add_argument("--env_generator", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Draw envs from declared domains and record them, "
                        "instead of the historical placement path. Builds a "
                        "scaffold for placement whether or not the agent "
                        "observes one, so the offsets are recorded either way "
                        "and an agent-hash run can be pointed at the same "
                        "world.json. Needs an explicit --place_margin.")
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
    p.add_argument("--arch", default="rnn", choices=list(ARCHITECTURES),
                   help="Policy architecture. 'rnn' is the single shared "
                        "network every history to date used. The other three "
                        "are the parameter-isolation family (plan section "
                        "4.3) and all of them are given an oracle task id at "
                        "training and evaluation time, which is a real "
                        "advantage and is recorded as one.")
    p.add_argument("--hnet_base", default="learned", choices=list(HNET_BASES),
                   help="--arch hnet: what the generated weights are added to. "
                        "'learned' warm-starts a free base vector from the "
                        "checkpoint; 'frozen' pins it there forever so only "
                        "the task-conditioned part can move; 'none' is the "
                        "from-scratch von Oswald form, and the only one whose "
                        "parameter count matches the baseline policy.")
    p.add_argument("--hnet_emb_dim", type=int, default=32,
                   help="--arch hnet: width of the task and chunk embeddings.")
    p.add_argument("--hnet_chunk_dim", type=int, default=512,
                   help="--arch hnet: generated weights per chunk. 0 builds "
                        "the unchunked hypernetwork, whose output layer is "
                        "then hidden x 74k -- millions of parameters to "
                        "generate thousands.")
    p.add_argument("--hnet_hidden", type=int, nargs="+", default=[100, 100],
                   help="--arch hnet: hidden widths of the generator MLP.")
    p.add_argument("--hnet_init_out_scale", type=float, default=0.01,
                   help="--arch hnet: how much the generator's output layer is "
                        "shrunk at init. Small means every task starts at the "
                        "warm-started base, i.e. at the pretrained policy the "
                        "controls start from. Ignored for --hnet_base none.")
    p.add_argument("--xdg_gating", type=float, default=0.8,
                   help="--arch xdg: fraction of hidden units held OFF for "
                        "each task. 0.8 is the paper's value; the warm start "
                        "makes it expensive here, since the checkpoint was "
                        "trained with every unit available.")
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_rnn_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Inter-layer trunk dropout (only effective with "
                        "num_rnn_layers > 1).")
    add_recurrent_args(p)
    # Continuous-mode exploration scale. These exist on `train_rnn` but were
    # never wired through here, so every continual run to date used the
    # RNNAgentConfig default of 0.0 -- sigma = 1.0 against a unit-magnitude
    # action, and learnable. The DAgger student was exploring with noise the
    # size of the action itself and the run script had no way to say otherwise.
    p.add_argument("--init_log_std", type=float, default=0.0,
                   help="Continuous policy: initial log sigma. Ignored in "
                        "discrete mode. Auto-restored from ckpt in finetune.")
    p.add_argument("--freeze_log_std", action="store_true",
                   help="Hold log sigma at --init_log_std instead of learning "
                        "it. Continuous mode only.")
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
    p.add_argument("--reset_optimizer_each_block", action="store_true",
                   help="Clear Adam's moment estimates at every task boundary. "
                        "Off by default (what every recorded history did). "
                        "Adam's second moments otherwise carry across "
                        "boundaries, so the first steps in env i are scaled by "
                        "statistics from env i-1 -- plan section 3.1 W2.")
    p.add_argument("--eval_deterministic",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Evaluate on the policy mean (default, and what every "
                        "recorded history used) or by sampling. A forgotten "
                        "environment is one the network is uncertain about, "
                        "and a Gaussian head fitted to an uncertain target "
                        "puts its mean near zero -- so if forgetting shows up "
                        "as uncertainty rather than confident error, the "
                        "default understates retention.")
    p.add_argument("--only_train_on_reached", action="store_true",
                   help="Per BC update, drop trajectories whose rollout never "
                        "reached the goal. If no trajectory reached, the update "
                        "is skipped entirely.")
    # Rollout
    p.add_argument("--batch_envs", type=int, default=16,
                   help="Parallel rollouts per env per update.")
    p.add_argument("--steps_per_rollout", type=int, default=None)
    # Continual-learning method. See docs/CONTINUAL_CONTROLS_PLAN.md section 4
    # and hopfield_nav/continual/. "none" is naive sequential SGD -- the floor,
    # and what every recorded history to date used.
    p.add_argument("--freeze_trunk", action="store_true",
                   help="Adapt only the movement head; hold the recurrent trunk "
                        "at whatever the checkpoint gave it. Plan section 3.2 "
                        "P4, and the load-bearing half of OML's mechanism "
                        "(section 5.1) without the meta-learning: if confining "
                        "plasticity to a small head is what buys retention, "
                        "that is worth knowing before building a meta-learner. "
                        "Composes with any --method.")
    p.add_argument("--world_spec", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Write world.json beside --out. On by default. Turn it "
                        "off for sweeps: many seeds writing into one directory "
                        "leave a single world.json describing whichever "
                        "finished last, which is worse than none at all.")
    p.add_argument("--method", default="none", choices=list(CONTINUAL_METHODS),
                   help="Continual-learning method applied to the BC update.")
    p.add_argument("--method_args", default=None,
                   help="Comma-separated key=value pairs for the method, e.g. "
                        "'buffer_size=inf,replay_batches=1' or "
                        "'lam=1e3,gamma=1.0'. Values are coerced "
                        "int -> float -> bool -> str; 'inf' is accepted.")
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
            rnn_cell=args.rnn_cell, rnn_nonlinearity=args.rnn_nonlinearity,
            init_log_std=args.init_log_std,
            freeze_log_std=args.freeze_log_std,
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
        eval_deterministic=args.eval_deterministic,
        eval_max_steps=args.max_steps,
        eval_every=1,
        seed=args.seed,
        device=args.device,
        env_generator=args.env_generator,
        place_region=args.place_region,
        goal_region=args.goal_region,
        wall_seeds=args.wall_seeds,
        place_margin=args.place_margin,
        goal_val_frac=args.goal_val_frac,
        n_val_envs=args.n_val_envs,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Pre-load checkpoint (if any) and auto-restore architecture-affecting
    # fields BEFORE building envs / VectorHash / agent.
    ckpt = None
    if args.load_checkpoint:
        print(f"[baseline] loading {args.load_checkpoint}  (fresh Adam moments)")
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        restore_arch_from_ckpt(cfg, ckpt)

    method_kwargs = parse_method_args(args.method_args)
    last_method_desc: dict = {}
    last_arch_detail: dict = {}

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

        envs, env_offsets, world_split, vh, world_kind = rnn_world(cfg, rng)
        if n_iters > 1:
            print(f"\n=== iter {k + 1}/{n_iters}  seed={seed_k} ===")
        print(f"[baseline] built {len(envs)} envs  size={cfg.env.size}  "
              f"obs_dim={cfg.env.observation_size}  world={world_kind}")
        for i, env in enumerate(envs):
            print(f"  env {i}: goal={env.goal_location}")

        sgb = None
        gbook_dim = 0
        vh_lambdas: list[int] = []
        vh_Npos = 0
        if cfg.agent.input_grid_state:
            sgb = smooth_gbook(vh.gbook, vh.lambdas, cfg.fwhm_ratio)
            gbook_dim = int(vh.Ng)
            vh_lambdas = list(vh.lambdas)
            vh_Npos = int(vh.Npos)
            print(f"[baseline] grid_state on  Ng={gbook_dim}  Npos={vh_Npos}  "
                  f"lambdas={vh_lambdas}  fwhm_ratio={cfg.fwhm_ratio}")
            for i, off in enumerate(env_offsets):
                print(f"  env {i}: offset={off}")
        # Only the first iteration's world is recorded: `--num_full_iters` re-runs
        # the whole protocol at seed+k, so there is no single world to describe,
        # and writing k of them under one name would describe none of them.
        #
        # The same argument applies *across processes*, which is why this can be
        # switched off. A sweep writes many runs at different seeds into one
        # directory; a single `world.json` there describes whichever finished
        # last and none of the others, so it is worse than absent.
        if k == 0 and args.world_spec:
            write_rnn_world_spec(cfg, world_split, vh, generator=world_kind,
                                 save_dir=os.path.dirname(os.path.abspath(args.out)))

        input_dim = compute_rnn_input_dim(cfg.agent, cfg.env.observation_size, gbook_dim)
        if k == 0:
            print(f"[baseline] RNN input_dim={input_dim}")
        agent = build_arch_agent(args, cfg, input_dim, seed_k).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.bc.lr)
        if ckpt is not None:
            # `warm_start` routes to whatever this architecture needs: a
            # straight load for the baseline, the head fanned out across tasks
            # for the multi-head policy, the base vector for the hypernetwork.
            warm_start(agent, ckpt["agent_state_dict"])
        if k == 0 and args.arch != "rnn":
            print(f"[baseline] arch={args.arch}  "
                  f"{agent.describe() if hasattr(agent, 'describe') else {}}")

        if args.freeze_trunk:
            # After the load, so what is frozen is the *pretrained* trunk.
            # Freezing before would pin it at initialisation, which measures
            # something else entirely.
            n_frozen, trainable = freeze_trunk_params(agent)
            if k == 0:
                print(f"[baseline] freeze_trunk: {n_frozen} params held, "
                      f"{trainable} adapt")
            # Rebuild the optimizer over the surviving parameters only, so
            # Adam is not carrying state for tensors that can never move.
            optimizer = torch.optim.Adam(
                [prm for prm in agent.parameters() if prm.requires_grad],
                lr=cfg.bc.lr)

        mode = "finetune" if ckpt is not None else "sequential"
        if k == 0:
            print(f"[baseline] mode={mode}  iters_per_block={cfg.updates_per_env}  "
                  f"max_steps={cfg.eval_max_steps}  num_full_iters={n_iters}")

        # Rebuilt per iteration: a replay buffer or a Fisher must never leak
        # across seeds, or iteration k would start with iteration k-1's memory
        # and the seed-to-seed variance would be silently understated.
        method = build_method(args.method, seed=seed_k, **method_kwargs)
        if k == 0:
            print(f"[baseline] method={args.method}  {method.describe()}")

        trace, blocks = run_sequential(
            cfg, agent, optimizer, envs, device,
            sgb=sgb, env_offsets=env_offsets, method=method,
            reset_optimizer_each_block=args.reset_optimizer_each_block,
        )
        last_method_desc = method.describe()
        # Taken *after* training, not at construction. The parameter counts are
        # the same either way, but the hypernetwork also reports how
        # task-dependent its generated weights ended up -- and that number is
        # only worth anything once the run has happened. A generator that never
        # learned to condition on its task embedding produces an entirely
        # ordinary-looking run whose low retention says nothing about the
        # method, so the check belongs in every history rather than in a
        # separate investigation.
        # The plain policy has no `describe` of its own, but it still needs a
        # parameter count: it is the row every isolation arm's parameter cost
        # is read against, and a blank there makes the comparison unreadable.
        last_arch_detail = (
            agent.describe() if hasattr(agent, "describe") else {
                "arch": "rnn",
                "trainable_params": sum(prm.numel() for prm in agent.parameters()
                                        if prm.requires_grad),
            })
        iter_traces.append((trace, blocks))
        iter_env_goals.append([list(env.goal_location) for env in envs])
        iter_env_offsets.append(
            [list(o) for o in env_offsets] if env_offsets is not None else None
        )
        last_vh_lambdas = vh_lambdas
        last_vh_Npos = vh_Npos
        last_gbook_dim = gbook_dim

    trace, blocks = merge_iter_traces(iter_traces)
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
            # What continual-learning method produced this history, and what it
            # cost. `state_bytes` is one of the five axes of the cost frontier
            # (plan section 0.1), so it belongs beside the curve, not in a log.
            "method": args.method,
            "method_args": args.method_args,
            "method_detail": last_method_desc,
            # The architecture is a second axis alongside the method: a
            # hypernetwork with no regulariser and a plain RNN with one are
            # different runs, and a summary keyed on `method` alone would
            # average them together. `arch_detail` carries the parameter
            # counts, which is what puts an isolation arm on the frontier.
            "arch": args.arch,
            "arch_detail": last_arch_detail,
            "extra": {
                "mode": mode,
                "base_seed": base_seed,
                "movement_mode": cfg.agent.movement_mode,
                "hidden_size": cfg.agent.hidden_size,
                "num_rnn_layers": cfg.agent.num_rnn_layers,
                "dropout": cfg.agent.dropout,
                "rnn_cell": cfg.agent.rnn_cell,
                "rnn_nonlinearity": cfg.agent.rnn_nonlinearity,
                "init_log_std": cfg.agent.init_log_std,
                "freeze_log_std": cfg.agent.freeze_log_std,
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
                "eval_deterministic": args.eval_deterministic,
                "reset_optimizer_each_block": args.reset_optimizer_each_block,
                "freeze_trunk": args.freeze_trunk,
                "hnet_base": args.hnet_base if args.arch == "hnet" else None,
                "hnet_emb_dim": args.hnet_emb_dim if args.arch == "hnet" else None,
                "hnet_chunk_dim": (args.hnet_chunk_dim if args.arch == "hnet"
                                   else None),
                "hnet_hidden": (list(args.hnet_hidden) if args.arch == "hnet"
                                else None),
                "hnet_init_out_scale": (args.hnet_init_out_scale
                                        if args.arch == "hnet" else None),
                "xdg_gating": args.xdg_gating if args.arch == "xdg" else None,
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
