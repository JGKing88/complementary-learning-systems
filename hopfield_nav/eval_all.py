"""Evaluate a single checkpoint on all evaluation types.

Used by `run_eval_all.sh` to run a batch of models through the full eval
stack: navigation det (+ optional stoch), goal discovery, exploration,
optional realistic (persistent Hopfield), optional repeat (fresh Hopfield
per trial), and optional sequential continual (paper-style moving-average
episodic success plot).

Usage:
    python -m hopfield_nav.eval_all --ckpt CHECKPOINT [flags]

All eval parameters are CLI flags so one script can drive whatever sweep
you want.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from hopfield_nav.config import (
    TrainConfig, EnvConfig, VectorHashConfig, HopfieldConfig,
    AgentConfig, PPOConfig,
)
from hopfield_nav.encoder import load_encoder
from hopfield_nav.env import make_env
from hopfield_nav.vectorhash import VectorHash
from hopfield_nav.agent import NavAgent, compute_input_dim
from hopfield_nav.eval import (
    evaluate_navigation, evaluate_goal_discovery, evaluate_exploration,
    evaluate_realistic, evaluate_repeat, evaluate_sequential_episodes,
)


def _coerce_legacy_cfg(cd: dict) -> dict:
    if "val_envs_per_world" in cd and "num_val_envs" not in cd:
        cd["num_val_envs"] = cd.pop("val_envs_per_world")
    vh = cd.get("vectorhash")
    if isinstance(vh, dict) and "gbook_only" in vh and "static_vectorhash" not in vh:
        vh["static_vectorhash"] = vh.pop("gbook_only")
    return cd


def make_cfg_from_checkpoint(ck_cfg_dict: dict) -> TrainConfig:
    cd = _coerce_legacy_cfg(dict(ck_cfg_dict))
    env = EnvConfig(**cd["env"])
    vh = VectorHashConfig(**cd["vectorhash"])
    hop = HopfieldConfig(**cd["hopfield"])
    ag = AgentConfig(**cd["agent"])
    ppo = PPOConfig(**cd["ppo"])
    cfg = TrainConfig(env=env, vectorhash=vh, hopfield=hop, agent=ag, ppo=ppo)
    for k, v in cd.items():
        if k in {"env", "vectorhash", "hopfield", "agent", "ppo"}:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def build_eval_world(cfg: TrainConfig, encoder, device: str):
    """Rebuild the training-time eval world: same seeding + scaffold."""
    rng = np.random.RandomState(cfg.seed)
    size = cfg.env.size
    # Skip train-env seeds to keep val-env seeds aligned with training.
    for _ in range(cfg.envs_per_world * cfg.num_worlds):
        rng.randint(0, 10_000_000)
    val_envs = [
        make_env(cfg.env, cfg.agent.movement_mode,
                 seed=int(rng.randint(0, 10_000_000)))
        for _ in range(cfg.num_val_envs)
    ]
    vh = VectorHash(cfg.vectorhash, size=size)
    vh.build_scaffold()
    vh.register_envs(val_envs, placement="spread")
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=device)
    val_idxs = list(range(cfg.num_val_envs))
    return val_envs, vh, val_idxs


def scaffold_layout_dict(
    cfg: TrainConfig,
    vh: VectorHash,
    val_envs: list,
    val_idxs: list[int],
) -> dict:
    """Serializable layout: Npos×Npos grid indices, env footprints, goals.

    ``cfg.vectorhash.Npos`` (when not None) is the checkpoint override; ``Npos``
    is the resolved size used by ``VectorHash`` (same as training when the
    checkpoint was saved with that config).
    """
    prod_lambdas = int(np.prod(cfg.vectorhash.lambdas))
    envs_out: list[dict] = []
    for i in range(len(val_envs)):
        off = vh.env_offsets[val_idxs[i]]
        g = val_envs[i].goal_location
        ox, oy = int(off[0]), int(off[1])
        gl0, gl1 = int(g[0]), int(g[1])
        envs_out.append({
            "idx": i,
            "offset": [ox, oy],
            "goal_local": [gl0, gl1],
            "goal_global": [gl0 + ox, gl1 + oy],
        })
    return {
        "Npos": int(vh.Npos),
        "Npos_config": cfg.vectorhash.Npos,
        "prod_lambdas": prod_lambdas,
        "lambdas": list(cfg.vectorhash.lambdas),
        "static_vectorhash": bool(cfg.vectorhash.static_vectorhash),
        "env_size": int(cfg.env.size),
        "placement": "spread",
        "envs": envs_out,
    }


@torch.no_grad()
def eval_checkpoint(
    ckpt_path: str,
    encoder_override: str | None,
    device: torch.device,
    npos: int | None,
    num_trials: int,
    max_steps: int,
    n_distractors: list[int],
    realistic_steps: int,
    realistic_seed_offset: int,
    run_realistic: bool,
    run_nav_stoch: bool,
    repeat_trials: int = 0,
    repeat_steps: int = 200,
    repeat_seed_offset: int = 2000,
    seq_iters_per_block: int = 0,
    seq_max_steps: int = 32,
    seq_ma_window: int = 20,
    seq_seed_offset: int = 3000,
    hopfield_oracle: bool | None = None,
    action_oracle: bool | None = None,
    static_vectorhash: bool | None = None,
    num_val_envs: int | None = None,
    lock_store_after_goal: bool = False,
    oracle_store_at_goal: bool = False,
    goal_radius: float | None = None,
) -> dict:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = make_cfg_from_checkpoint(ck["config"])
    if hopfield_oracle is not None:
        cfg.hopfield_oracle = bool(hopfield_oracle)
    if action_oracle is not None:
        cfg.action_oracle = bool(action_oracle)
    if static_vectorhash is not None:
        cfg.vectorhash.static_vectorhash = bool(static_vectorhash)
    if goal_radius is not None:
        cfg.env.goal_radius = float(goal_radius)
    if num_val_envs is not None:
        cfg.num_val_envs = int(num_val_envs)
    if bool(getattr(cfg, "hopfield_oracle", False)) and not bool(cfg.agent.input_hopfield_signal):
        print(
            "  WARNING: hopfield_oracle is on but the policy has "
            "input_hopfield_signal=False — the RNN was built without a hopfield "
            "input channel, so the oracle (and all hopfield) signal is all zeros. "
            "The model cannot use the disambiguation cue. Train with "
            "--input_hopfield_signal (default True) and re-checkpoint, or turn off "
            "--hopfield-oracle in eval.\n"
            "  (Eval runs normally; this is a config mismatch.)",
            flush=True,
        )
    npos_from_ck = cfg.vectorhash.Npos
    npos_effective: int
    if npos is not None:
        npos_effective = int(npos)
        cfg.vectorhash.Npos = npos_effective
    else:
        npos_effective = int(
            npos_from_ck
            if npos_from_ck is not None
            else int(np.prod(cfg.vectorhash.lambdas))
        )

    encoder_path = encoder_override or cfg.encoder_checkpoint
    encoder, enc_cfg, enc_gain = load_encoder(encoder_path, str(device))
    embed_dim = enc_cfg.out_dim
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(enc_gain)

    torch.manual_seed(0)
    np.random.seed(0)

    val_envs, vh, val_idxs = build_eval_world(cfg, encoder, str(device))
    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    agent = NavAgent(cfg.agent, input_dim).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()

    results: dict = {
        "ckpt_path": ckpt_path,
        "encoder_path": encoder_path,
        "Npos": npos_effective,
        "Npos_from_checkpoint": npos_from_ck,
        "Npos_override": npos,
        "hopfield_oracle": bool(getattr(cfg, "hopfield_oracle", False)),
        "input_hopfield_signal": bool(cfg.agent.input_hopfield_signal),
        "input_encoded_state": bool(cfg.agent.input_encoded_state),
        "input_sensory": bool(cfg.agent.input_sensory),
        "hopfield_oracle_effective": bool(getattr(cfg, "hopfield_oracle", False))
        and bool(cfg.agent.input_hopfield_signal),
        "action_oracle": bool(getattr(cfg, "action_oracle", False)),
        "static_vectorhash": bool(cfg.vectorhash.static_vectorhash),
        "num_val_envs": cfg.num_val_envs,
        "movement_mode": cfg.env.movement_mode,
        "lock_store_after_goal": bool(lock_store_after_goal),
        "scaffold_layout": scaffold_layout_dict(cfg, vh, val_envs, val_idxs),
    }

    print(f"  [nav_det]   num_trials={num_trials} max_steps={max_steps} "
          f"dist={n_distractors}", flush=True)
    nav_det = evaluate_navigation(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=num_trials, max_steps=max_steps,
        n_distractors_list=n_distractors, deterministic=True,
    )
    results["nav_det"] = nav_det

    if run_nav_stoch:
        print(f"  [nav_stoch] num_trials={num_trials} max_steps={max_steps} "
              f"dist={n_distractors}", flush=True)
        nav_stoch = evaluate_navigation(
            agent, val_envs, vh, val_idxs, cfg, device,
            num_trials=num_trials, max_steps=max_steps,
            n_distractors_list=n_distractors, deterministic=False,
        )
        results["nav_stoch"] = nav_stoch

    print(f"  [discovery] num_trials={num_trials} max_steps={max_steps} "
          f"dist={n_distractors}", flush=True)
    disc = evaluate_goal_discovery(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=num_trials, max_steps=max_steps,
        n_distractors_list=n_distractors,
    )
    results["discovery"] = disc

    print(f"  [explore]   num_trials={num_trials} max_steps={max_steps} "
          f"dist={n_distractors}", flush=True)
    expl = evaluate_exploration(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=num_trials, max_steps=max_steps,
        n_distractors_list=n_distractors,
    )
    results["exploration"] = expl

    if run_realistic and realistic_steps > 0:
        print(f"  [realistic] steps_per_env={realistic_steps} "
              f"n_envs={cfg.num_val_envs}", flush=True)
        real = evaluate_realistic(
            agent, val_envs, vh, val_idxs, cfg, device,
            steps_per_env=realistic_steps,
            seed=cfg.seed + realistic_seed_offset,
            deterministic=True,
            lock_store_after_goal=lock_store_after_goal,
        )
        # Collapse non-serializable parts (tuple keys, intervals lists are fine)
        retest_serializable = {
            f"visit_{vi}_retest_{rj}": m for (vi, rj), m in real["retest"].items()
        }
        drift_serializable = {
            str(j): [{"gap": g, **m} for g, m in curve]
            for j, curve in real["drift"].items()
        }
        results["realistic"] = {
            "primary": {str(k): v for k, v in real["primary"].items()},
            "retest": retest_serializable,
            "drift": drift_serializable,
            "summary": real["summary"],
        }

    if repeat_trials > 0:
        print(f"  [repeat]    n_trials={repeat_trials} steps_per_env={repeat_steps} "
              f"n_envs={cfg.num_val_envs}", flush=True)
        rep = evaluate_repeat(
            agent, val_envs, vh, val_idxs, cfg, device,
            n_trials=repeat_trials,
            steps_per_env=repeat_steps,
            seed=cfg.seed + repeat_seed_offset,
            deterministic=True,
        )
        results["repeat"] = {
            "trials": {str(k): v for k, v in rep["trials"].items()},
            "summary": rep["summary"],
        }

    if seq_iters_per_block > 0:
        print(f"  [sequential] iters_per_block={seq_iters_per_block} "
              f"max_steps={seq_max_steps} n_envs={cfg.num_val_envs}", flush=True)
        seq = evaluate_sequential_episodes(
            agent, val_envs, vh, val_idxs, cfg, device,
            iters_per_block=seq_iters_per_block,
            max_steps=seq_max_steps,
            seed=cfg.seed + seq_seed_offset,
            deterministic=True,
            lock_store_after_goal=lock_store_after_goal,
            oracle_store_at_goal=oracle_store_at_goal,
        )
        # Keep lists-of-tuples JSON-serializable.
        results["sequential"] = {
            "params": {**seq["params"], "ma_window": int(seq_ma_window)},
            "env_iters": {
                str(k): [[int(t) for t in pt] for pt in v]
                for k, v in seq["env_iters"].items()
            },
            "boundaries": [int(b) for b in seq["boundaries"]],
            "summary": seq["summary"],
        }

    return results


def _fmt_nav(res: dict, dist: int) -> str:
    m = res[dist]
    return (f"sr={m['success_rate']:.2f} steps={m['mean_steps']:.1f} "
            f"speed={m['mean_speed']:.3f}")


def _fmt_disc(res: dict, dist: int) -> str:
    m = res[dist]
    return (f"store_rate={m['store_success_rate']:.2f} "
            f"reach={m['reach_success_rate']:.2f} "
            f"eff={m['store_efficiency']:.2f}")


def _fmt_expl(res: dict, dist: int) -> str:
    m = res[dist]
    return (f"cov={m['mean_coverage']:.2f} "
            f"find={m['goal_find_rate']:.2f} "
            f"steps={m['mean_steps_to_goal']:.1f}")


def print_summary(tag: str, results: dict, n_distractors: list[int]) -> None:
    print(f"\n=== {tag} ===")
    print(f"  ckpt={results['ckpt_path']}")
    print(f"  mode={results['movement_mode']} num_val_envs={results['num_val_envs']}")
    for d in n_distractors:
        print(f"  [dist={d}]")
        print(f"    nav_det    : {_fmt_nav(results['nav_det'], d)}")
        if "nav_stoch" in results:
            print(f"    nav_stoch  : {_fmt_nav(results['nav_stoch'], d)}")
        print(f"    discovery  : {_fmt_disc(results['discovery'], d)}")
        print(f"    exploration: {_fmt_expl(results['exploration'], d)}")
    if "repeat" in results:
        rs = results["repeat"]["summary"]
        print(f"  repeat/summary: n_trials={rs['n_trials']} "
              f"steps_per_env={rs['steps_per_env']} "
              f"mean_reaches={rs['mean_reaches']:.2f}")
    if "sequential" in results:
        ss = results["sequential"]["summary"]
        sp = results["sequential"]["params"]
        print(f"  sequential/summary: iters_per_block={sp['iters_per_block']} "
              f"max_steps={sp['max_steps']} "
              f"mean_primary={ss['mean_primary_success']:.2f} "
              f"mean_final_revisit={ss['mean_final_revisit_success']:.2f} "
              f"drop={ss['interference_drop']:.3f} "
              f"hopfield_final={ss['hopfield_final_memories']}")
    if "realistic" in results:
        r = results["realistic"]
        s = r["summary"]
        print(f"  realistic/summary: mean_primary={s['mean_primary_reaches']:.2f} "
              f"mean_final_retest={s['mean_final_retest_reaches']:.2f} "
              f"drop={s['interference_drop']:.3f} "
              f"hopfield_final={s['hopfield_final_memories']}")
        # Per-env drift curve: for env j, primary at gap=0, retests at gap=1..
        N = len(r["primary"])
        print(f"  realistic/per_env  (gap=0 primary; gap=k retest after visiting env j+k)")
        header = "    env " + "".join(f"  gap{g:>2}" for g in range(N))
        print(header)
        for j in range(N):
            curve = r["drift"].get(str(j), [])
            by_gap = {d["gap"]: d for d in curve}
            cells = []
            for g in range(N):
                if g in by_gap:
                    cells.append(f"  {by_gap[g]['n_reaches']:>5d}")
                else:
                    cells.append("      .")
            print(f"    {j:>3}" + "".join(cells))


def save_drift_plot(results: dict, out_path: str) -> None:
    """Drift plot: x = envs visited so far (absolute index), y = n_reaches.
    One line per env j, starting at x=j (its primary) and extending rightward
    with its retest values up to x=N-1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = results["realistic"]
    N = len(r["primary"])
    fig, ax = plt.subplots(figsize=(max(6, N * 0.8), 5))
    for j in range(N):
        curve = r["drift"].get(str(j), [])
        pts = sorted([(d["gap"], d["n_reaches"]) for d in curve])
        xs = [j + g for g, _ in pts]
        ys = [n for _, n in pts]
        ax.plot(xs, ys, marker="o", label=f"env {j}")
    ax.set_xlabel("envs visited so far (0-indexed)")
    ax.set_ylabel("goal reaches in this env")
    ax.set_xticks(range(N))
    ax.set_title(f"Realistic eval drift — {results.get('tag', '')}\n"
                 f"hopfield_final={r['summary']['hopfield_final_memories']} "
                 f"interference_drop={r['summary']['interference_drop']:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_realistic_intervals_plot(results: dict, out_path: str) -> None:
    """Per-env subplots: each line is one phase (primary or retest after gap).

    Y = steps since last goal (or phase start) until this reach; X = reach
    index within that phase (1-based).  An open ring around a reach marks
    that a Hopfield store was written (same as ``store_fired`` in eval) on the
    same step as that goal reach. Retests have storing disabled, so no rings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = results["realistic"]
    N = len(r["primary"])
    fig, axs = plt.subplots(
        N, 1, figsize=(max(7, min(14, 4 + N * 0.5)), max(3.5, 2.4 * N)),
        squeeze=False,
        sharex=False,
    )
    cmap = plt.cm.tab10
    store_ring_legend_done = False
    for j, ax in enumerate(axs.flat):
        curve = r["drift"].get(str(j), [])
        ordered = sorted(curve, key=lambda d: d["gap"])
        for row, entry in enumerate(ordered):
            gap = int(entry["gap"])
            intervals = entry.get("intervals") or []
            stored_at = entry.get("stored_at_reach")
            if not stored_at or len(stored_at) != len(intervals):
                stored_at = [False] * len(intervals)
            tail = entry.get("tail_steps")
            color = cmap((row % 10) / 10.0)
            label = "primary" if gap == 0 else f"retest gap={gap}"

            if intervals:
                xs = np.arange(1, len(intervals) + 1)
                ax.plot(
                    xs, intervals, marker="o", markersize=3, linewidth=1.2,
                    color=color, label=label, zorder=2,
                )
                # Ring overlay: Hopfield write on the goal-reach step (see eval).
                sx = [xs[i] for i in range(len(intervals)) if stored_at[i]]
                sy = [intervals[i] for i in range(len(intervals)) if stored_at[i]]
                if sx:
                    leg = "store written at reach" if not store_ring_legend_done else "_nolegend_"
                    store_ring_legend_done = True
                    ax.plot(
                        sx, sy, linestyle="none",
                        marker="o", markersize=9, zorder=5,
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=1.5, label=leg,
                    )

            # Trailing cut-off segment (phase ended mid-trajectory, no reach):
            # drawn as an extra point at x = n_reaches + 1 with a different
            # marker + dashed connector to distinguish it from completed
            # intervals. Plotted even when intervals is empty.
            if tail is not None and tail > 0:
                tail_x = len(intervals) + 1
                if intervals:
                    ax.plot(
                        [len(intervals), tail_x],
                        [intervals[-1], tail],
                        linestyle="--", linewidth=1.0, color=color,
                    )
                tail_label = None if intervals else f"{label} (cut off)"
                ax.plot(
                    [tail_x], [tail], linestyle="none",
                    marker="s", markersize=5,
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=1.2, label=tail_label,
                )
        ax.set_ylabel("steps to goal")
        ax.set_title(f"env {j}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
    axs.flat[-1].set_xlabel("reach index (within phase)")
    fig.suptitle(
        f"Realistic eval — time-to-goal per reach — {results.get('tag', '')}\n"
        f"hopfield_final={r['summary']['hopfield_final_memories']} "
        f"interference_drop={r['summary']['interference_drop']:.3f}  |  "
        f"open ring = Hopfield store on that goal reach (primary; retest: store off)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _interval_plot_path(drift_plot_path: str) -> str:
    if drift_plot_path.endswith("_realistic_drift.png"):
        return drift_plot_path.replace("_realistic_drift.png", "_realistic_intervals.png")
    base, ext = os.path.splitext(drift_plot_path)
    return f"{base}_intervals{ext}"


def _repeat_plot_path(drift_plot_path: str) -> str:
    if drift_plot_path.endswith("_realistic_drift.png"):
        return drift_plot_path.replace("_realistic_drift.png", "_repeat_intervals.png")
    base, ext = os.path.splitext(drift_plot_path)
    return f"{base}_repeat_intervals{ext}"


def save_repeat_intervals_plot(results: dict, out_path: str) -> None:
    """Per-env subplots for the repeat eval: each line is one independent trial.

    Same marker conventions as :func:`save_realistic_intervals_plot`:
      - Y = steps since last goal reach (or trial start) until this reach.
      - X = reach index within the trial.
      - Open ring = Hopfield store fired on that goal-reach step.
      - Open square w/ dashed connector = trailing cut-off segment (no reach).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = results["repeat"]
    env_keys = sorted(r["trials"].keys(), key=lambda s: int(s))
    N = len(env_keys)
    if N == 0:
        return
    fig, axs = plt.subplots(
        N, 1, figsize=(max(7, min(14, 4 + N * 0.5)), max(3.5, 2.4 * N)),
        squeeze=False,
        sharex=False,
    )
    cmap = plt.cm.tab10
    store_ring_legend_done = False
    for ax_row, env_k in enumerate(env_keys):
        ax = axs.flat[ax_row]
        entries = r["trials"][env_k]
        for row, entry in enumerate(entries):
            intervals = entry.get("intervals") or []
            stored_at = entry.get("stored_at_reach")
            if not stored_at or len(stored_at) != len(intervals):
                stored_at = [False] * len(intervals)
            tail = entry.get("tail_steps")
            color = cmap((row % 10) / 10.0)
            start = entry.get("start", None)
            label = f"trial {entry.get('trial_idx', row)}"
            if start is not None:
                label += f" start=({start[0]},{start[1]})"

            if intervals:
                xs = np.arange(1, len(intervals) + 1)
                ax.plot(
                    xs, intervals, marker="o", markersize=3, linewidth=1.2,
                    color=color, label=label, zorder=2,
                )
                sx = [xs[i] for i in range(len(intervals)) if stored_at[i]]
                sy = [intervals[i] for i in range(len(intervals)) if stored_at[i]]
                if sx:
                    leg = "store written at reach" if not store_ring_legend_done else "_nolegend_"
                    store_ring_legend_done = True
                    ax.plot(
                        sx, sy, linestyle="none",
                        marker="o", markersize=9, zorder=5,
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=1.5, label=leg,
                    )

            if tail is not None and tail > 0:
                tail_x = len(intervals) + 1
                if intervals:
                    ax.plot(
                        [len(intervals), tail_x],
                        [intervals[-1], tail],
                        linestyle="--", linewidth=1.0, color=color,
                    )
                tail_label = None if intervals else f"{label} (cut off)"
                ax.plot(
                    [tail_x], [tail], linestyle="none",
                    marker="s", markersize=5,
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=1.2, label=tail_label,
                )
        ax.set_ylabel("steps to goal")
        ax.set_title(f"env {env_k}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
    axs.flat[-1].set_xlabel("reach index (within trial)")
    s = r["summary"]
    fig.suptitle(
        f"Repeat eval — {s['n_trials']} trials × {s['steps_per_env']} steps — "
        f"{results.get('tag', '')}\n"
        f"mean_reaches={s['mean_reaches']:.2f}  |  "
        f"fresh Hopfield + fresh RNN per trial  |  open ring = store at reach",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _sequential_plot_path(drift_plot_path: str) -> str:
    if drift_plot_path.endswith("_realistic_drift.png"):
        return drift_plot_path.replace("_realistic_drift.png", "_sequential.png")
    base, ext = os.path.splitext(drift_plot_path)
    return f"{base}_sequential{ext}"


def save_sequential_episodes_plot(
    results: dict, out_path: str, show_stores: bool = True,
) -> None:
    """Continual-learning plot in the forgetting-curve style.

    One smoothed success-rate line per env; block-shaded background colored
    by the active env in that block; "train env i" label centered above each
    block. ``show_stores=True`` overlays per-iter goal-store / off-goal-store
    markers above/below each line; ``False`` for a clean line plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "sequential" not in results:
        return
    s = results["sequential"]
    boundaries = [int(b) for b in s["boundaries"]]
    if not boundaries:
        return
    total_iters = boundaries[-1]
    ma_w = int(s["params"].get("ma_window", 20))
    env_iters = s["env_iters"]
    env_keys = sorted(env_iters.keys(), key=lambda k: int(k))
    N = len(env_keys)

    fig, ax = plt.subplots(figsize=(max(8, N * 1.6), 4.8))
    cmap = plt.cm.tab10

    # Block-shaded background regions (active env per block).
    prev = 0
    for j, b in enumerate(boundaries):
        ax.axvspan(prev, b, color=cmap(j % 10), alpha=0.10, zorder=0)
        prev = b

    for env_k in env_keys:
        pts = env_iters[env_k]
        if not pts:
            continue
        pts = sorted(pts, key=lambda xy: xy[0])
        xs = np.array([p[0] for p in pts], dtype=np.int64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        sag = np.array([p[2] if len(p) > 2 else 0 for p in pts], dtype=np.int64)
        sog = np.array([p[3] if len(p) > 3 else 0 for p in pts], dtype=np.int64)
        w = max(1, min(ma_w, len(ys)))
        kernel = np.ones(w) / w
        ma = np.convolve(ys, kernel, mode="full")[: len(ys)]
        if len(ys) >= 1:
            for i in range(min(w - 1, len(ys))):
                ma[i] = float(np.mean(ys[: i + 1]))
        color = cmap(int(env_k) % 10)
        ax.plot(xs, ma * 100.0, color=color, linewidth=1.6, label=f"env {env_k}")

        if show_stores:
            sag_xs = xs[sag == 1]
            if sag_xs.size > 0:
                sag_ys = ma[sag == 1] * 100.0 + 4.0
                ax.scatter(sag_xs, sag_ys, marker="o", s=14, color=color,
                           edgecolors="black", linewidths=0.4, zorder=5)
            sog_xs = xs[sog == 1]
            if sog_xs.size > 0:
                sog_ys = ma[sog == 1] * 100.0 - 4.0
                ax.scatter(sog_xs, sog_ys, marker="x", s=18, color=color,
                           linewidths=1.0, zorder=5)

    # Block labels at top, in active env's color (matches forgetting plot).
    prev = 0
    for j, b in enumerate(boundaries):
        center = (prev + b) / 2.0
        ax.text(
            center, 0.96, f"train env {j}",
            ha="center", va="top", fontsize=9, color=cmap(j % 10),
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
        prev = b

    ax.set_xlim(0, total_iters)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("sequential iteration")
    ax.set_ylabel("episode success rate (%)")
    s_sum = s["summary"]
    title = f"Sequential continual eval — {results.get('tag', '')}"
    if ma_w > 1:
        title = f"{title}  (smooth window={ma_w})"
    ax.set_title(
        f"{title}\n"
        f"mean_primary={s_sum['mean_primary_success']:.2f}  "
        f"mean_final_revisit={s_sum['mean_final_revisit_success']:.2f}  "
        f"drop={s_sum['interference_drop']:.3f}",
        fontsize=10, pad=10,
    )
    ax.grid(True, alpha=0.3)

    if show_stores:
        from matplotlib.lines import Line2D
        marker_handles = [
            Line2D([0], [0], marker="o", linestyle="", color="grey",
                   markerfacecolor="grey", markeredgecolor="black",
                   markersize=6, label="store @ goal"),
            Line2D([0], [0], marker="x", linestyle="", color="grey",
                   markersize=7, mew=1.5, label="store off goal"),
        ]
        leg1 = ax.legend(handles=marker_handles, loc="upper left", fontsize=8)
        ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=8, ncol=max(1, N // 5 + 1))
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _scaffold_layout_plot_path(drift_plot_path: str) -> str:
    if drift_plot_path.endswith("_realistic_drift.png"):
        return drift_plot_path.replace("_realistic_drift.png", "_scaffold_layout.png")
    base, ext = os.path.splitext(drift_plot_path)
    return f"{base}_scaffold_layout{ext}"


def save_scaffold_layout_plot(results: dict, out_path: str) -> None:
    """Πλ×Πλ theoretical period as background; Npos×Npos scaffold outline; envs + goals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    L = results["scaffold_layout"]
    prod = int(L["prod_lambdas"])
    Npos = int(L["Npos"])
    size = int(L["env_size"])
    extent = max(prod, Npos)
    fig, ax = plt.subplots(figsize=(7, 7))

    # 1) Full combinatorial period (theoretical torus before Npos truncation)
    ax.add_patch(
        Rectangle(
            (0, 0), prod, prod,
            facecolor="#d8e4f2", edgecolor="#3d5a80", linewidth=1.4,
            linestyle="-", zorder=0, label=f"Πλ={prod} (full period)",
        )
    )
    # 2) Actual VectorHash grid used at runtime (bottom-left [0,Npos)² subset of period)
    ax.add_patch(
        Rectangle(
            (0, 0), Npos, Npos,
            facecolor="none", edgecolor="#1b4332", linewidth=2.4,
            linestyle="-", zorder=2, label=f"Npos={Npos} (scaffold)",
        )
    )
    # Region where integer top-left offsets may land: [0, Npos-size] inclusive
    if Npos > size:
        ax.add_patch(
            Rectangle(
                (0, 0), Npos - size, Npos - size, facecolor="none",
                edgecolor="#555555", linewidth=1.2, linestyle="--", zorder=3,
                label="valid env origin (top-left) bounds",
            )
        )

    cmap = plt.cm.tab10
    for row, e in enumerate(L["envs"]):
        ox, oy = e["offset"]
        color = cmap((row % 10) / 10.0)
        ax.add_patch(
            Rectangle(
                (ox, oy), size, size, facecolor=color, edgecolor="black",
                linewidth=1.2, alpha=0.42, zorder=4,
            )
        )
        gx, gy = e["goal_global"]
        ax.plot(
            gx + 0.5, gy + 0.5, marker="*", markersize=11, color=color,
            markeredgecolor="black", markeredgewidth=0.4, zorder=5,
        )
        ax.plot([], [], color=color, marker="s", linestyle="None", markersize=8,
                label=f"env {e['idx']}")

    ax.set_aspect("equal")
    pad = 0.02 * extent
    ax.set_xlim(-pad, extent + pad)
    ax.set_ylim(-pad, extent + pad)
    ax.set_xlabel("global grid x (VectorHash index, origin-aligned)")
    ax.set_ylabel("global grid y (VectorHash index, origin-aligned)")
    ax.grid(True, alpha=0.2, zorder=1)

    nconf = L["Npos_config"]
    prod_l = L["prod_lambdas"]
    if nconf is not None:
        npos_note = f"Npos={Npos} (checkpoint override {nconf}; Πλ={prod_l})"
    else:
        npos_note = f"Npos={Npos} (config default Πλ={prod_l})"
    ax.set_title(
        f"VectorHash placement — {results.get('tag', '')}\n"
        f"{npos_note}  |  λ={L['lambdas']}  |  env size={size}  |  {L['placement']}",
        fontsize=10,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--encoder", default=None,
                   help="Override encoder path; default = checkpoint's saved path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default=None,
                   help="Label for the printed output (e.g. wandb run name)")
    p.add_argument(
        "--Npos", type=int, default=None,
        help="Override VectorHash scaffold side length; if omitted, use the "
             "checkpoint's vectorhash.Npos or prod(lambdas) when Npos is null",
    )
    p.add_argument(
        "--hopfield-oracle", dest="hopfield_oracle", default=None,
        action=argparse.BooleanOptionalAction,
        help="Use oracle Hopfield signal (goal embedding - current) in the "
             "local 2D frame when the goal is in memory, instead of recall. "
             "Omit to keep the checkpoint's hopfield_oracle setting.",
    )
    p.add_argument(
        # --gbook-only / --no-gbook-only are the pre-rename spelling, kept as a
        # deprecated alias so old sbatch scripts still parse (run_eval_all.sh
        # passed it until 2026-08). BooleanOptionalAction generates the --no- form.
        "--static-vectorhash", "--gbook-only",
        dest="static_vectorhash", default=None,
        action=argparse.BooleanOptionalAction,
        help="Override checkpoint VectorHash: static scaffold with gbook + position-indexed "
             "sbook (no pbook / assoc). Omit to use the value saved in the checkpoint "
             "(default False for old ckpts). (--gbook-only is a deprecated alias.)",
    )
    p.add_argument(
        "--action-oracle", dest="action_oracle", default=None,
        action=argparse.BooleanOptionalAction,
        help="Override movement with a greedy step toward goal when the goal is in memory "
             "(same gating as hopfield_oracle). Omit to keep the checkpoint setting.",
    )
    p.add_argument(
        "--num-val-envs", dest="num_val_envs", type=int, default=None,
        help="Override cfg.num_val_envs (number of val envs built for every eval). "
             "Omit to keep the checkpoint's saved value.",
    )
    p.add_argument(
        "--lock-store-after-goal", dest="lock_store_after_goal",
        action="store_true",
        help="In the realistic and sequential evals, suppress all further "
             "Hopfield store actions in an env once that env's goal has been "
             "stored. Has no effect on nav/discovery/exploration/repeat.",
    )
    p.add_argument(
        "--oracle-store-at-goal", dest="oracle_store_at_goal",
        action="store_true",
        help="In the sequential eval only: bypass the agent's store head — "
             "store fires automatically and only when at the env's goal. "
             "Off-goal stores suppressed. Useful for evaluating Phase-A-only "
             "ckpts whose store policy is untrained.",
    )
    p.add_argument(
        "--show-stores", dest="show_stores",
        action=argparse.BooleanOptionalAction, default=True,
        help="On the sequential plot, show goal-store / off-goal-store "
             "markers. Use --no-show-stores for a clean lines-only plot.",
    )

    # Nav/discovery/exploration eval params
    p.add_argument("--num_trials", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_distractors", type=int, nargs="+", default=[0])
    p.add_argument("--goal_radius", type=float, default=None,
                   help="Override EnvConfig.goal_radius from the saved checkpoint. "
                        "Default: use the value saved at training time (or 0.5 for "
                        "checkpoints saved before the field was added).")
    p.add_argument("--no-nav-stoch", action="store_true",
                   help="Skip the stochastic navigation eval (nav_det only)")

    # Realistic eval params
    p.add_argument("--realistic-steps", type=int, default=1000,
                   help="Steps per env in the realistic eval (0 = skip)")
    p.add_argument("--realistic-seed-offset", type=int, default=1000,
                   help="Added to cfg.seed for the realistic-eval RNG")
    p.add_argument("--skip-realistic", action="store_true")

    # Repeat eval params: N independent trials per env, fresh Hopfield each time,
    # primary phase only. Useful as a sanity check without interference.
    p.add_argument("--repeat-trials", type=int, default=0,
                   help="Trials per env in the repeat eval (0 = skip)")
    p.add_argument("--repeat-steps", type=int, default=200,
                   help="Steps per trial in the repeat eval")
    p.add_argument("--repeat-seed-offset", type=int, default=2000,
                   help="Added to cfg.seed for the repeat-eval RNG")

    # Sequential continual eval params: introduce envs one by one with
    # episodic success = binary "reached goal in ≤max_steps". Hopfield is
    # persistent; revisits have store disabled.
    p.add_argument("--seq-iters-per-block", type=int, default=0,
                   help="Iterations per env block in the sequential eval (0 = skip)")
    p.add_argument("--seq-max-steps", type=int, default=32,
                   help="Max steps per mini-episode in the sequential eval")
    p.add_argument("--seq-ma-window", type=int, default=20,
                   help="Moving-average window for the sequential plot")
    p.add_argument("--seq-seed-offset", type=int, default=3000,
                   help="Added to cfg.seed for the sequential-eval RNG")

    p.add_argument("--output-json", default=None,
                   help="Write full results dict as JSON to this path")
    p.add_argument("--plot-path", default=None,
                   help="Base path for PNGs: scaffold layout always; if realistic runs, also "
                        "drift + intervals (*_scaffold_layout.png, *_realistic_intervals.png "
                        "when path ends with _realistic_drift.png)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    tag = args.tag or os.path.basename(args.ckpt)

    print(f"\n>>> evaluating {tag}")
    print(f"    ckpt={args.ckpt}")
    print(f"    device={device}")
    if args.Npos is not None:
        print(f"    Npos override: {args.Npos}")
    if args.hopfield_oracle is not None:
        print(f"    hopfield_oracle: {args.hopfield_oracle} (CLI override)")
    if args.static_vectorhash is not None:
        print(f"    static_vectorhash: {args.static_vectorhash} (CLI override)")
    if args.action_oracle is not None:
        print(f"    action_oracle: {args.action_oracle} (CLI override)")
    if args.num_val_envs is not None:
        print(f"    num_val_envs: {args.num_val_envs} (CLI override)")
    if args.lock_store_after_goal:
        print("    lock_store_after_goal: True (realistic + sequential)")

    results = eval_checkpoint(
        ckpt_path=args.ckpt,
        encoder_override=args.encoder,
        device=device,
        npos=args.Npos,
        hopfield_oracle=args.hopfield_oracle,
        action_oracle=args.action_oracle,
        static_vectorhash=args.static_vectorhash,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        n_distractors=args.n_distractors,
        realistic_steps=args.realistic_steps,
        realistic_seed_offset=args.realistic_seed_offset,
        run_realistic=not args.skip_realistic,
        run_nav_stoch=not getattr(args, "no_nav_stoch"),
        repeat_trials=args.repeat_trials,
        repeat_steps=args.repeat_steps,
        repeat_seed_offset=args.repeat_seed_offset,
        seq_iters_per_block=args.seq_iters_per_block,
        seq_max_steps=args.seq_max_steps,
        seq_ma_window=args.seq_ma_window,
        seq_seed_offset=args.seq_seed_offset,
        num_val_envs=args.num_val_envs,
        lock_store_after_goal=args.lock_store_after_goal,
        oracle_store_at_goal=args.oracle_store_at_goal,
        goal_radius=args.goal_radius,
    )
    results["tag"] = tag

    print_summary(tag, results, args.n_distractors)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n  wrote {args.output_json}")

    if args.plot_path:
        lpath = _scaffold_layout_plot_path(args.plot_path)
        save_scaffold_layout_plot(results, lpath)
        print(f"  wrote {lpath}")
        if "realistic" in results:
            save_drift_plot(results, args.plot_path)
            print(f"  wrote {args.plot_path}")
            ipath = _interval_plot_path(args.plot_path)
            save_realistic_intervals_plot(results, ipath)
            print(f"  wrote {ipath}")
        if "repeat" in results:
            rpath = _repeat_plot_path(args.plot_path)
            save_repeat_intervals_plot(results, rpath)
            print(f"  wrote {rpath}")
        if "sequential" in results:
            spath = _sequential_plot_path(args.plot_path)
            save_sequential_episodes_plot(
                results, spath, show_stores=args.show_stores,
            )
            print(f"  wrote {spath}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
