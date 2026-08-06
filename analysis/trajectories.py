"""Visualize trajectories at every training checkpoint of a run.

Three modes (one per invocation, selected by ``--mode``):

  combined      empty Hopfield (+ optional distractors), agent explores; on the
                first goal-reach it stores (natural store-head OR forced via
                ``--force_store``), then teleports to a fresh start and
                navigates back to the goal.

  explore_only  empty Hopfield, agent wanders for ``--explore_steps`` with
                ``goal_in_memory=False``. Goal is just a passive cell; no
                early-termination, no nav phase.

  exploit_only  goal pattern preloaded into Hopfield, agent navigates from a
                random start. Episode ends on goal-reach.

The combined panel is drawn as a single subplot per trial: explore segment in
orange, post-store nav segment in blue, store moment marked with a black
diamond on the goal cell, and the teleport jump rendered as a dashed gray
segment.

Output is a single grid figure (saved as both PNG and PDF): rows = checkpoints (sorted by update number),
cols = trials. Trial j uses a fixed (val_env, start, distractor sample) at
every row, so column j shows the SAME scenario at each checkpoint and you
can read changes off the row-axis as training progresses.

Encoder path is read from each checkpoint's saved ``cfg.encoder_checkpoint``
and is NOT a CLI flag.

Checkpoints come from the run's ``run.json`` manifest when it has one. For the
run directories written before manifests existed, the fallback matches any
basename ending in ``_u{N}.pt`` or ``_update{N}.pt`` (so both
``hopfield_nav_update200.pt`` and ``phase_a_u200.pt`` style runs work); files
like ``phase_a_only_final.pt`` that lack an update number are skipped either
way, since the figure's rows are indexed by update.

Usage:
    python -m analysis.trajectories \\
        --checkpoint_dir checkpoint/phase_a_only_hopeful-haze-46 \\
        --mode combined --trials 6 \\
        --explore_steps 200 --nav_steps 100 \\
        --force_store
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import torch

from cls_paths import REPO_ROOT, encoders_dir
import run_manifest
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.world.env import at_goal
from hopfield_nav.evaluation.metrics import agent_step, random_start
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, load_agent,
)
from hopfield import Hopfield


MODES = ("combined", "explore_only", "exploit_only")
PHASE_COLOR = {"explore": "#e08020", "nav": "#1f77b4"}
TELEPORT_COLOR = "#888888"

# Text sizing — kept central so labels / legend stay in sync.
ROW_LABEL_FONTSIZE = 24    # left-side "u{update}" row labels
LEGEND_FONTSIZE = 24


# ---------------------------------------------------------------------------
# Trial plans — fixed across checkpoints so column j is the same scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialPlan:
    env_local_idx: int
    start: tuple[int, int]
    nav_start: tuple[int, int]   # teleport target (combined mode only)
    distractor_seed: int


def make_trial_plans(n_trials: int, val_envs, base_seed: int) -> list[TrialPlan]:
    rng = np.random.RandomState(base_seed)
    plans = []
    for j in range(n_trials):
        env_idx = j % len(val_envs)
        env = val_envs[env_idx]
        goal = env.goal_location
        start = random_start(env.size, goal, rng)
        nav_start = random_start(env.size, goal, rng)
        distractor_seed = int(rng.randint(0, 2**31 - 1))
        plans.append(TrialPlan(env_idx, start, nav_start, distractor_seed))
    return plans


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def _capture_pos(env) -> tuple[tuple[int, int], np.ndarray]:
    """Return (snapped grid cell, raw continuous (x, y)).

    For ``ContinuousGridEnv`` we read ``_continuous_pos`` (the un-snapped
    float position). For ``GridEnv`` there is no continuous state, so the
    raw value is just the integer cell cast to float.
    """
    p = env.current_location
    if hasattr(env, "_continuous_pos"):
        cont = env._continuous_pos.copy().astype(np.float64)
    else:
        cont = np.array([float(p[0]), float(p[1])], dtype=np.float64)
    return tuple(p), cont


def _build_hopfield_with_distractors(
    vh, env_offset, env_size, n_distractors, seed, device, beta,
) -> Hopfield:
    rng = np.random.RandomState(seed)
    embed_dim = vh.encoded_Phi.shape[2]
    hopfield = Hopfield(embed_dim, beta=beta, device=str(device))
    for pat in sample_distractors(vh, env_offset, env_size, n_distractors, rng):
        hopfield.input_memory(torch.from_numpy(pat).float())
    return hopfield


def _store_goal_and_mark_timeout(
    hopfield: Hopfield,
    vh,
    env_offset,
    goal: tuple[int, int],
    positions: list[tuple[int, int]],
) -> tuple[int, bool]:
    """Force-store goal memory after explore timeout.

    Returns:
      - explore_end_idx: index to terminate the explore segment in plotting
      - timed_out_store: True, to signal an "X" marker at timeout location
    """
    goal_enc = goal_encoding(vh, env_offset, goal)
    hopfield.input_memory(torch.from_numpy(goal_enc).float())
    explore_end_idx = len(positions) - 1
    return explore_end_idx, True


@torch.no_grad()
def _rollout_combined(
    agent, env, env_offset, vh, cfg, device, plan,
    n_distractors, explore_steps, nav_steps, force_store,
) -> dict:
    """One discover-then-navigate cycle:
      Phase 1 explore  : empty + distractor Hopfield, goal_in_memory=False.
                         Run until at_g_pre AND (force_store OR store head fires),
                         or until ``explore_steps``.
      Phase 2 teleport : env.set_position(plan.nav_start); reset RNN/enrichment.
      Phase 3 nav      : Hopfield now contains goal pattern, goal_in_memory=True.
                         Run until at_g (post-step) or ``nav_steps``.
    """
    hopfield = _build_hopfield_with_distractors(
        vh, env_offset, cfg.env.size, n_distractors,
        plan.distractor_seed, device, cfg.hopfield.beta,
    )
    env.set_position(plan.start)
    goal = env.goal_location
    h_rnn = None
    prev_reward = None
    prev_action = None
    positions: list[tuple[int, int]] = []
    cont_positions: list[np.ndarray] = []

    def _append() -> None:
        p, c = _capture_pos(env)
        positions.append(p)
        cont_positions.append(c)

    _append()
    stored = False
    explore_end_idx: int | None = None
    timed_out_store = False
    reached_explore = False

    # Phase 1: explore.
    for _step in range(explore_steps):
        at_g_pre = at_goal(env)
        out = agent_step(
            agent, env, env_offset, vh, hopfield, h_rnn, cfg, device,
            deterministic=True, goal_local=goal, goal_in_memory=False,
            prev_reward=prev_reward, prev_action=prev_action,
        )
        h_rnn = out["h_rnn"]
        prev_reward = out["next_prev_reward"]
        prev_action = out["next_prev_action"]
        _append()

        if at_goal(env):
            reached_explore = True

        if at_g_pre:
            # When force_store is on, the agent's natural store head is
            # disabled — the only store that ever happens is this forced one.
            should_store = force_store or out["store_action"] > 0.5
            if should_store:
                # The store happened at the pre-step position (= goal cell).
                # The agent's just-applied move is post-store and irrelevant;
                # drop it so the explore segment terminates ON the goal.
                positions.pop()
                cont_positions.pop()
                hopfield.input_memory(out["store_embedding"][0])
                stored = True
                explore_end_idx = len(positions) - 1
                reached_explore = True
                break

    # If explore timed out without a store, force-store the goal and continue
    # with teleport + nav as if the goal had been reached/stored.
    if not stored:
        explore_end_idx, timed_out_store = _store_goal_and_mark_timeout(
            hopfield=hopfield,
            vh=vh,
            env_offset=env_offset,
            goal=goal,
            positions=positions,
        )
        stored = True

    nav_start_idx: int | None = None
    reached_nav = False
    if stored:
        # Phase 2: teleport. Reset RNN/enrichment for a clean nav segment
        # (matches evaluate_realistic semantics).
        env.set_position(plan.nav_start)
        _append()
        nav_start_idx = len(positions) - 1
        h_rnn = None
        prev_reward = None
        prev_action = None

        # Phase 3: navigate.
        for _step in range(nav_steps):
            out = agent_step(
                agent, env, env_offset, vh, hopfield, h_rnn, cfg, device,
                deterministic=True, goal_local=goal, goal_in_memory=True,
                prev_reward=prev_reward, prev_action=prev_action,
            )
            h_rnn = out["h_rnn"]
            prev_reward = out["next_prev_reward"]
            prev_action = out["next_prev_action"]
            _append()
            if at_goal(env):
                reached_nav = True
                break

    return {
        "mode": "combined",
        "positions": positions,
        "cont_positions": cont_positions,
        "goal": goal,
        "start": plan.start,
        "nav_start": plan.nav_start,
        "explore_end_idx": explore_end_idx,
        "nav_start_idx": nav_start_idx,
        "stored": stored,
        "timed_out_store": timed_out_store,
        "reached_explore": reached_explore,
        "reached_nav": reached_nav,
    }


@torch.no_grad()
def _rollout_explore(
    agent, env, env_offset, vh, cfg, device, plan, n_distractors, max_steps,
) -> dict:
    """Empty (+ distractor) Hopfield, goal_in_memory=False, run ``max_steps``
    without early termination — pure exploration."""
    hopfield = _build_hopfield_with_distractors(
        vh, env_offset, cfg.env.size, n_distractors,
        plan.distractor_seed, device, cfg.hopfield.beta,
    )
    env.set_position(plan.start)
    goal = env.goal_location
    h_rnn = None
    prev_reward = None
    prev_action = None
    positions: list[tuple[int, int]] = []
    cont_positions: list[np.ndarray] = []

    def _append() -> None:
        p, c = _capture_pos(env)
        positions.append(p)
        cont_positions.append(c)

    _append()
    found = False

    for _step in range(max_steps):
        out = agent_step(
            agent, env, env_offset, vh, hopfield, h_rnn, cfg, device,
            deterministic=True, goal_local=goal, goal_in_memory=False,
            prev_reward=prev_reward, prev_action=prev_action,
        )
        h_rnn = out["h_rnn"]
        prev_reward = out["next_prev_reward"]
        prev_action = out["next_prev_action"]
        _append()
        if at_goal(env):
            found = True

    coverage = len(set(positions)) / float(cfg.env.size * cfg.env.size)
    return {
        "mode": "explore_only",
        "positions": positions,
        "cont_positions": cont_positions,
        "goal": goal,
        "start": plan.start,
        "found_goal": found,
        "coverage": coverage,
    }


@torch.no_grad()
def _rollout_exploit(
    agent, env, env_offset, vh, cfg, device, plan, n_distractors, max_steps,
) -> dict:
    """Goal preloaded in Hopfield, navigate from random start to goal."""
    rng = np.random.RandomState(plan.distractor_seed)
    embed_dim = vh.encoded_Phi.shape[2]
    hopfield = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
    goal = env.goal_location
    goal_enc = goal_encoding(vh, env_offset, goal)
    patterns = [goal_enc] + sample_distractors(
        vh, env_offset, cfg.env.size, n_distractors, rng,
    )
    rng.shuffle(patterns)
    for pat in patterns:
        hopfield.input_memory(torch.from_numpy(pat).float())

    env.set_position(plan.start)
    h_rnn = None
    prev_reward = None
    prev_action = None
    positions: list[tuple[int, int]] = []
    cont_positions: list[np.ndarray] = []

    def _append() -> None:
        p, c = _capture_pos(env)
        positions.append(p)
        cont_positions.append(c)

    _append()
    reached = False

    for _step in range(max_steps):
        out = agent_step(
            agent, env, env_offset, vh, hopfield, h_rnn, cfg, device,
            deterministic=True, goal_local=goal, goal_in_memory=True,
            prev_reward=prev_reward, prev_action=prev_action,
        )
        h_rnn = out["h_rnn"]
        prev_reward = out["next_prev_reward"]
        prev_action = out["next_prev_action"]
        _append()
        if at_goal(env):
            reached = True
            break

    return {
        "mode": "exploit_only",
        "positions": positions,
        "cont_positions": cont_positions,
        "goal": goal,
        "start": plan.start,
        "reached": reached,
    }


def collect_trials(
    agent, val_envs, vh, val_idxs, cfg, device, plans,
    mode, n_distractors, explore_steps, nav_steps, force_store,
) -> list[dict]:
    trials = []
    for plan in plans:
        env = val_envs[plan.env_local_idx]
        env_offset = vh.env_offsets[val_idxs[plan.env_local_idx]]
        if mode == "combined":
            t = _rollout_combined(
                agent, env, env_offset, vh, cfg, device, plan,
                n_distractors, explore_steps, nav_steps, force_store,
            )
        elif mode == "explore_only":
            t = _rollout_explore(
                agent, env, env_offset, vh, cfg, device, plan,
                n_distractors, explore_steps,
            )
        elif mode == "exploit_only":
            t = _rollout_exploit(
                agent, env, env_offset, vh, cfg, device, plan,
                n_distractors, nav_steps,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        trials.append(t)
    return trials


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _set_grid_axes(ax, size: int) -> None:
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    # Keep gridlines (anchored to integer cell boundaries) but hide the
    # tick marks and tick labels entirely.
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0,
                   bottom=False, top=False, left=False, right=False)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")
    ax.invert_yaxis()


def _plot_segment(ax, positions: np.ndarray, color: str, zorder: int = 3) -> None:
    if len(positions) < 1:
        return
    if len(positions) >= 2:
        ax.plot(positions[:, 0], positions[:, 1], "-",
                color=color, linewidth=2.0, alpha=0.85, zorder=zorder)
    ax.scatter(positions[:, 0], positions[:, 1], s=18,
               color=color, alpha=0.75, zorder=zorder + 1)


def _draw_start_goal(ax, start, goal, goal_radius: float) -> None:
    ax.scatter([start[0]], [start[1]], marker="s", s=180,
               facecolors="none", edgecolors="green", linewidth=2.5, zorder=6)
    # Draw the goal as a hollow circle with a data-space radius of goal_radius.
    ax.add_patch(Circle((goal[0], goal[1]), radius=goal_radius, fill=False,
                        edgecolor="red", linewidth=2.5, zorder=5))


def _render_combined(ax, trial: dict, size: int,
                     pos_key: str = "positions",
                     goal_radius: float = 0.5) -> None:
    positions = np.array(trial[pos_key], dtype=float)
    goal = trial["goal"]
    start = trial["start"]
    expl_end = trial["explore_end_idx"]
    nav_start = trial["nav_start_idx"]

    _set_grid_axes(ax, size)

    if expl_end is None:
        _plot_segment(ax, positions, color=PHASE_COLOR["explore"])
    else:
        _plot_segment(ax, positions[: expl_end + 1], color=PHASE_COLOR["explore"])
        if trial.get("timed_out_store", False):
            # Explore exhausted max steps before reaching the goal.
            ax.scatter([positions[expl_end, 0]], [positions[expl_end, 1]],
                       marker="x", s=140, color="white", linewidths=5.0, zorder=8)
            ax.scatter([positions[expl_end, 0]], [positions[expl_end, 1]],
                       marker="x", s=140, color="red", linewidths=2.6, zorder=9)
        # Store marker at goal cell (temporarily disabled).
        # ax.scatter([goal[0]], [goal[1]], marker="D", s=70,
        #            color="black", linewidth=0, zorder=7)
        if nav_start is not None:
            # Teleport jump (dashed).
            ax.plot([positions[expl_end, 0], positions[nav_start, 0]],
                    [positions[expl_end, 1], positions[nav_start, 1]],
                    "--", color=TELEPORT_COLOR, alpha=0.55, linewidth=1.5,
                    zorder=2)
            _plot_segment(ax, positions[nav_start:], color=PHASE_COLOR["nav"])
            # Mark teleport target with a small open circle.
            ax.scatter([positions[nav_start, 0]], [positions[nav_start, 1]],
                       marker="o", s=140, facecolors="none",
                       edgecolors=TELEPORT_COLOR, linewidth=2, zorder=6)

    _draw_start_goal(ax, start, goal, goal_radius=goal_radius)


def _render_single(ax, trial: dict, size: int, color: str,
                   pos_key: str = "positions",
                   goal_radius: float = 0.5) -> None:
    positions = np.array(trial[pos_key], dtype=float)
    _set_grid_axes(ax, size)
    _plot_segment(ax, positions, color=color)
    _draw_start_goal(ax, trial["start"], trial["goal"], goal_radius=goal_radius)


def _legend_handles(mode: str) -> list:
    start_h = Line2D([], [], marker="s", color="green", lw=0,
                     mfc="none", mew=2.5, ms=14, label="start")
    goal_h = Line2D([], [], marker="o", color="red", lw=0,
                    mfc="none", mew=2.5, ms=14, label="goal")
    if mode == "combined":
        return [
            Line2D([], [], color=PHASE_COLOR["explore"], lw=3, label="explore"),
            Line2D([], [], color=PHASE_COLOR["nav"], lw=3, label="exploit"),
            Line2D([], [], color=TELEPORT_COLOR, lw=2, ls="--", label="teleport"),
            # Line2D([], [], marker="D", color="black", lw=0,
            #        mec="black", mew=0, ms=12, label="store"),
            Line2D([], [], marker="x", color="red", lw=0,
                   mew=2.5, ms=12, label="timeout"),
            Line2D([], [], marker="o", color=TELEPORT_COLOR, lw=0,
                   mfc="none", mew=2, ms=14, label="tp target"),
            start_h, goal_h,
        ]
    if mode == "explore_only":
        return [
            Line2D([], [], color=PHASE_COLOR["explore"], lw=3, label="explore"),
            start_h, goal_h,
        ]
    return [
        Line2D([], [], color=PHASE_COLOR["nav"], lw=3, label="exploit"),
        start_h, goal_h,
    ]


def plot_grid(
    ckpt_results: list[tuple[int, list[dict]]],
    mode: str,
    size: int,
    out_path: str,
    pos_key: str = "positions",
    goal_radius: float = 0.5,
) -> None:
    """Render the checkpoint × trial grid.

    ``pos_key`` selects which trajectory representation to draw:
      - ``"positions"``      : snapped grid cells (integer)
      - ``"cont_positions"`` : raw float positions (un-snapped) — only differs
                               from ``positions`` for ``ContinuousGridEnv``.
    """
    if not ckpt_results:
        raise ValueError("no checkpoints to plot")
    n_rows = len(ckpt_results)
    n_cols = max(len(t) for _, t in ckpt_results)
    panel = 3.4
    # Reserve a tall strip at the top of the figure for the legend so it has
    # real breathing room rather than fighting with row 0's panels.
    legend_strip = 0.7  # inches
    fig_w = panel * n_cols + 1.0
    fig_h = panel * n_rows + legend_strip
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False,
    )

    for r, (update, trials) in enumerate(ckpt_results):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(trials):
                ax.axis("off")
                continue
            trial = trials[c]
            if mode == "combined":
                _render_combined(ax, trial, size, pos_key=pos_key,
                                 goal_radius=goal_radius)
            elif mode == "explore_only":
                _render_single(ax, trial, size,
                               color=PHASE_COLOR["explore"], pos_key=pos_key,
                               goal_radius=goal_radius)
            else:  # exploit_only
                _render_single(ax, trial, size,
                               color=PHASE_COLOR["nav"], pos_key=pos_key,
                               goal_radius=goal_radius)
        axes[r, 0].set_ylabel(f"u{update}", fontsize=ROW_LABEL_FONTSIZE)

    # Lay out the panel grid first, leaving the top legend_strip empty.
    top_frac = 1.0 - (legend_strip / fig_h)
    fig.tight_layout(rect=[0, 0, 1, top_frac])

    # Center the legend horizontally inside that reserved strip.
    handles = _legend_handles(mode)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0 - legend_strip / (2.0 * fig_h)),
    )

    out_root, _ = os.path.splitext(out_path)
    plt.savefig(f"{out_root}.png", dpi=500, bbox_inches="tight")
    plt.savefig(f"{out_root}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """[(update, path)] for a run directory, sorted by update.

    Delegates to `run_manifest.checkpoints_in`, which reads the run's manifest
    when it has one and otherwise falls back to the regex over filenames that
    used to live here. The regex is not gone -- there are 2503 checkpoints
    across ten naming schemes in directories written before manifests existed,
    and they still have to be readable.

    What the manifest buys is that the *next* naming scheme costs nothing: the
    update number is recorded at write time instead of being recovered from a
    basename by a pattern that has to be extended each time.
    """
    return run_manifest.checkpoints_in(checkpoint_dir)


def _resolve_encoder_path(enc_path: str, checkpoint_dir: str) -> str:
    """Resolve cfg.encoder_checkpoint relative to common roots if not absolute.

    Checkpoints store the encoder path as saved at train time, which is usually
    relative to the repo root ("encoders/run_<ts>/encoder_best.pt"). Since the
    2026-08 storage migration the real directory lives under CLS_RUNS, with
    "encoders" left in the repo as a symlink -- so the repo-root candidates below
    still work, and the encoders_dir() candidate keeps working if that symlink is
    ever removed.
    """
    if os.path.isabs(enc_path) and os.path.exists(enc_path):
        return enc_path
    candidates = [
        enc_path,
        os.path.join(os.getcwd(), enc_path),
        os.path.join(str(REPO_ROOT), enc_path),
        os.path.join("/home/jackking/cls", enc_path),
        os.path.join("/orcd/home/002/jackking/cls", enc_path),
        os.path.join(checkpoint_dir, enc_path),
    ]
    # "encoders/<run>/<file>" -> "<CLS_RUNS>/encoders/<run>/<file>"
    parts = os.path.normpath(enc_path).split(os.sep)
    if parts and parts[0] == "encoders":
        candidates.append(str(encoders_dir().joinpath(*parts[1:])))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"could not locate encoder at {enc_path!r}; tried: {candidates}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True,
                   help="Directory containing hopfield_nav_update*.pt files.")
    p.add_argument("--mode", choices=MODES, default="combined")
    p.add_argument("--out", default=None,
                   help="Output path prefix/filename stem. Figures are always "
                        "saved as both .png and .pdf. Default stem: "
                        "<checkpoint_dir>/trajectories_<mode>")
    p.add_argument("--trials", type=int, default=6,
                   help="Number of trials (columns).")
    p.add_argument("--explore_steps", type=int, default=200,
                   help="Max steps for explore phase (combined or explore_only).")
    p.add_argument("--nav_steps", type=int, default=100,
                   help="Max steps for nav phase (combined or exploit_only).")
    p.add_argument("--n_distractors", type=int, default=0,
                   help="Distractor patterns preloaded into Hopfield.")
    p.add_argument("--force_store", action="store_true",
                   help="combined mode: force a store on the first goal-reach "
                        "(pre-step at goal). When set, the agent's natural "
                        "store head is fully disabled — the forced store is "
                        "the only one that ever fires.")
    p.add_argument("--updates", default=None,
                   help="Comma-separated update numbers to include. "
                        "Default: all hopfield_nav_update*.pt in the dir.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for trial-plan generation (env_idx, starts, "
                        "distractor sample). Fixed across checkpoints.")
    p.add_argument("--goal_radius", type=float, default=None,
                   help="Override EnvConfig.goal_radius from the saved checkpoint. "
                        "Default: use the value saved at training time (or 0.5 for "
                        "checkpoints saved before the field was added).")
    args = p.parse_args()

    device = torch.device(args.device if (
        args.device == "cpu" or torch.cuda.is_available()
    ) else "cpu")

    all_ckpts = discover_checkpoints(args.checkpoint_dir)
    if not all_ckpts:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir}")
    if args.updates:
        wanted = {int(x) for x in args.updates.split(",")}
        all_ckpts = [c for c in all_ckpts if c[0] in wanted]
        if not all_ckpts:
            raise SystemExit(f"No checkpoints matched --updates {args.updates}")

    # Build the eval world from the first checkpoint's cfg.
    first_update, first_path = all_ckpts[0]
    ck0 = torch.load(first_path, map_location=device, weights_only=False)
    cfg = cfg_from_checkpoint(ck0["config"])
    if args.goal_radius is not None:
        cfg.env.goal_radius = float(args.goal_radius)

    enc_path = getattr(cfg, "encoder_checkpoint", "") or ""
    if not enc_path:
        raise SystemExit(
            "Checkpoint cfg has no encoder_checkpoint; cannot proceed."
        )
    enc_path = _resolve_encoder_path(enc_path, args.checkpoint_dir)
    print(f"Encoder (from cfg): {enc_path}")

    encoder, enc_cfg, _ = load_encoder(enc_path, str(device))
    embed_dim = enc_cfg.out_dim

    torch.manual_seed(0)
    np.random.seed(0)
    val_envs, vh, val_idxs = build_eval_world(cfg, encoder, str(device))

    plans = make_trial_plans(args.trials, val_envs, args.seed)

    print(f"Mode: {args.mode}  trials: {args.trials}  "
          f"explore_steps: {args.explore_steps}  nav_steps: {args.nav_steps}  "
          f"n_distractors: {args.n_distractors}"
          f"{'  force_store' if args.force_store else ''}")
    print(f"Checkpoints: {[u for u, _ in all_ckpts]}")

    ckpt_results: list[tuple[int, list[dict]]] = []
    for update, path in all_ckpts:
        print(f"  loading update {update}: {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        agent = load_agent(cfg, ck["agent_state_dict"], embed_dim, device)
        trials = collect_trials(
            agent, val_envs, vh, val_idxs, cfg, device, plans,
            mode=args.mode, n_distractors=args.n_distractors,
            explore_steps=args.explore_steps, nav_steps=args.nav_steps,
            force_store=args.force_store,
        )
        ckpt_results.append((update, trials))

    out_path = args.out or os.path.join(
        args.checkpoint_dir, f"trajectories_{args.mode}.png"
    )
    base, ext = os.path.splitext(out_path)
    cont_path = f"{base}_continuous{ext}"

    plot_grid(ckpt_results, args.mode, cfg.env.size, out_path,
              pos_key="positions", goal_radius=float(cfg.env.goal_radius))
    print(f"Saved (snapped):    {out_path}")
    plot_grid(ckpt_results, args.mode, cfg.env.size, cont_path,
              pos_key="cont_positions", goal_radius=float(cfg.env.goal_radius))
    print(f"Saved (continuous): {cont_path}")


if __name__ == "__main__":
    main()
