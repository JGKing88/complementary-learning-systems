"""Explore vs exploit trial collector.

Per arena, ``n_starts`` trials per condition. Each trial:
  - sample n_distractors ~ Uniform[n_dist_min, n_dist_max]
  - sample random non-goal start
  - explore: hopfield seeded with distractors only (no goal)
  - exploit: hopfield seeded with distractors + the env's goal
  - rollout (stochastic or deterministic) until goal reached or max_steps
  - record h_t for every step

Stored on disk as a directory:
  <out>/per_arena/<idx>.npz   one npz per arena
  <out>/meta.json             collection config + summary
  <out>/scaffold.json         arena layout
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.rollout.distractors import goal_encoding

from .rollout import EnvBundle, RolloutEngine


@dataclass
class TrialData:
    """Per-arena collection: pooled across all trials of each condition."""
    arena_id: int
    goal_local: tuple[int, int]
    quadrant: int
    h_explore: np.ndarray          # (T_total, H) phase=0
    h_exploit: np.ndarray          # (T_total, H) phase=1
    trial_explore: np.ndarray      # (T_total,) trial id 0..n_starts-1
    trial_exploit: np.ndarray
    summaries_explore: list[dict]  # per-trial: {trial, n_distractors, reached, steps_taken}
    summaries_exploit: list[dict]


@dataclass
class TrialsDataset:
    per_arena: dict[int, TrialData]
    meta: dict = field(default_factory=dict)

    def arena_ids(self) -> list[int]:
        return sorted(self.per_arena.keys())

    def quadrant_map(self) -> dict[int, int]:
        return {a: td.quadrant for a, td in self.per_arena.items()}

    def pooled(self, arena_ids: list[int] | None = None,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten over arenas + both conditions. Returns (h, phase, arena_id)."""
        if arena_ids is None:
            arena_ids = self.arena_ids()
        h_chunks, phase_chunks, arena_chunks = [], [], []
        for a in arena_ids:
            td = self.per_arena[a]
            if td.h_explore.shape[0] > 0:
                h_chunks.append(td.h_explore)
                phase_chunks.append(np.zeros(td.h_explore.shape[0], dtype=np.int8))
                arena_chunks.append(np.full(td.h_explore.shape[0], a, dtype=np.int64))
            if td.h_exploit.shape[0] > 0:
                h_chunks.append(td.h_exploit)
                phase_chunks.append(np.ones(td.h_exploit.shape[0], dtype=np.int8))
                arena_chunks.append(np.full(td.h_exploit.shape[0], a, dtype=np.int64))
        if not h_chunks:
            H = self.meta.get("hidden_size", 0)
            return (np.zeros((0, H), dtype=np.float32),
                    np.zeros((0,), dtype=np.int8),
                    np.zeros((0,), dtype=np.int64))
        return (np.concatenate(h_chunks, axis=0),
                np.concatenate(phase_chunks, axis=0),
                np.concatenate(arena_chunks, axis=0))

    def centroids(self, arena_ids: list[int] | None = None,
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Group-level (centroid_explore, centroid_exploit) over arena_ids.

        Mean over the flattened (across arenas) timesteps. If a phase has zero
        rows, returns NaN-filled vector for that phase.
        """
        if arena_ids is None:
            arena_ids = self.arena_ids()
        explore_chunks, exploit_chunks = [], []
        for a in arena_ids:
            td = self.per_arena[a]
            if td.h_explore.shape[0] > 0:
                explore_chunks.append(td.h_explore)
            if td.h_exploit.shape[0] > 0:
                exploit_chunks.append(td.h_exploit)
        H = self.meta.get("hidden_size", None)
        if H is None and explore_chunks:
            H = explore_chunks[0].shape[1]
        elif H is None and exploit_chunks:
            H = exploit_chunks[0].shape[1]
        if not explore_chunks:
            c0 = np.full(H, np.nan, dtype=np.float32)
        else:
            c0 = np.concatenate(explore_chunks, axis=0).mean(axis=0).astype(np.float32)
        if not exploit_chunks:
            c1 = np.full(H, np.nan, dtype=np.float32)
        else:
            c1 = np.concatenate(exploit_chunks, axis=0).mean(axis=0).astype(np.float32)
        return c0, c1

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        (out / "per_arena").mkdir(parents=True, exist_ok=True)
        for a, td in self.per_arena.items():
            np.savez(
                out / "per_arena" / f"{a:04d}.npz",
                h_explore=td.h_explore,
                h_exploit=td.h_exploit,
                trial_explore=td.trial_explore,
                trial_exploit=td.trial_exploit,
                arena_id=np.int64(a),
                goal_local=np.asarray(td.goal_local, dtype=np.int64),
                quadrant=np.int64(td.quadrant),
            )
        # Per-arena summaries are JSON-serializable lists; bundle into meta.
        full_meta = dict(self.meta)
        full_meta["per_arena_summaries"] = {
            str(a): {"explore": td.summaries_explore,
                     "exploit": td.summaries_exploit,
                     "goal_local": list(td.goal_local),
                     "quadrant": int(td.quadrant)}
            for a, td in self.per_arena.items()
        }
        (out / "meta.json").write_text(json.dumps(full_meta, indent=2))

    @classmethod
    def load(cls, out_dir: str | Path) -> "TrialsDataset":
        out = Path(out_dir)
        meta = json.loads((out / "meta.json").read_text())
        per_arena_summaries = meta.pop("per_arena_summaries", {})
        per_arena: dict[int, TrialData] = {}
        for f in sorted((out / "per_arena").glob("*.npz")):
            z = np.load(f)
            a = int(z["arena_id"])
            ps = per_arena_summaries.get(str(a), {})
            per_arena[a] = TrialData(
                arena_id=a,
                goal_local=tuple(int(x) for x in z["goal_local"]),
                quadrant=int(z["quadrant"]),
                h_explore=np.asarray(z["h_explore"], dtype=np.float32),
                h_exploit=np.asarray(z["h_exploit"], dtype=np.float32),
                trial_explore=np.asarray(z["trial_explore"], dtype=np.int64),
                trial_exploit=np.asarray(z["trial_exploit"], dtype=np.int64),
                summaries_explore=ps.get("explore", []),
                summaries_exploit=ps.get("exploit", []),
            )
        return cls(per_arena=per_arena, meta=meta)


class ExploreExploitCollector:
    """For each arena, run n_starts trials per condition.

    The same engine instance is reused across arenas — it owns the agent and the
    val-world. ``deterministic`` flips between argmax (default-eval style) and
    sampled actions; we expose it as a flag rather than baking it in.
    """

    def __init__(self, engine: RolloutEngine) -> None:
        self.engine = engine

    def _one_trial(
        self,
        env,
        env_offset: tuple[int, int],
        condition: str,
        n_distractors: int,
        max_steps: int,
        deterministic: bool,
        rng: np.random.RandomState,
    ) -> dict:
        if condition not in {"explore", "exploit"}:
            raise ValueError(f"bad condition: {condition!r}")

        hopfield = self.engine.make_hopfield()
        goal = env.goal_location

        if condition == "exploit":
            goal_enc = goal_encoding(self.engine.vh, env_offset, goal)
            hopfield.input_memory(torch.from_numpy(goal_enc).float())
            goal_in_memory_flag = True
        else:
            goal_in_memory_flag = False

        self.engine.seed_distractors(hopfield, env_offset, n_distractors,
                                     rng, env_size=env.size)
        env.set_position(random_start(env.size, goal, rng))
        out = self.engine.rollout(
            env, env_offset,
            hopfield=hopfield, h_rnn=None,
            prev_reward=None, prev_action=None,
            goal_in_memory_flag=goal_in_memory_flag,
            max_steps=max_steps,
            deterministic=deterministic,
            record_positions=False,
            stop_on_goal=True,
        )
        return {
            "h": out["h"],
            "reached": out["reached"],
            "steps_taken": out["steps_taken"],
        }

    def collect(
        self,
        bundle: EnvBundle,
        *,
        n_starts: int,
        max_steps: int,
        n_dist_min: int,
        n_dist_max: int,
        deterministic: bool,
        seed: int = 0,
        progress_every: int = 1,
    ) -> TrialsDataset:
        rng = np.random.RandomState(seed)
        per_arena: dict[int, TrialData] = {}
        n_arenas = len(bundle.envs)
        total_trials_target = 2 * n_starts * n_arenas
        print(
            f"[collect_trials] starting: {n_arenas} arenas × 2 conditions × "
            f"{n_starts} starts = {total_trials_target} trials, "
            f"max_steps={max_steps}, deterministic={deterministic}",
            flush=True,
        )
        t_start = time.perf_counter()
        for arena_id, env in enumerate(bundle.envs):
            env_offset = bundle.offsets[arena_id]
            cond_h = {"explore": [], "exploit": []}
            cond_trial = {"explore": [], "exploit": []}
            cond_summary = {"explore": [], "exploit": []}
            for cond in ("explore", "exploit"):
                for trial in range(n_starts):
                    n_dist = int(rng.randint(n_dist_min, n_dist_max + 1))
                    res = self._one_trial(
                        env, env_offset, cond,
                        n_distractors=n_dist,
                        max_steps=max_steps,
                        deterministic=deterministic,
                        rng=rng,
                    )
                    n = int(res["h"].shape[0])
                    if n > 0:
                        cond_h[cond].append(res["h"])
                        cond_trial[cond].append(np.full(n, trial, dtype=np.int64))
                    cond_summary[cond].append({
                        "trial": trial,
                        "n_distractors": n_dist,
                        "reached": bool(res["reached"]),
                        "steps_taken": int(res["steps_taken"]),
                    })

            def _stack(chunks, default_h):
                if not chunks:
                    return default_h
                return np.concatenate(chunks, axis=0)

            H = bundle.cfg.agent.hidden_size
            empty_h = np.zeros((0, H), dtype=np.float32)
            empty_t = np.zeros((0,), dtype=np.int64)
            per_arena[arena_id] = TrialData(
                arena_id=arena_id,
                goal_local=bundle.goals_local[arena_id],
                quadrant=bundle.quadrants[arena_id],
                h_explore=_stack(cond_h["explore"], empty_h),
                h_exploit=_stack(cond_h["exploit"], empty_h),
                trial_explore=_stack(cond_trial["explore"], empty_t),
                trial_exploit=_stack(cond_trial["exploit"], empty_t),
                summaries_explore=cond_summary["explore"],
                summaries_exploit=cond_summary["exploit"],
            )

            if (arena_id + 1) % progress_every == 0 or arena_id + 1 == n_arenas:
                td = per_arena[arena_id]
                reach0 = sum(s["reached"] for s in cond_summary["explore"])
                reach1 = sum(s["reached"] for s in cond_summary["exploit"])
                elapsed = time.perf_counter() - t_start
                rate = (arena_id + 1) / max(elapsed, 1e-6)
                eta = (n_arenas - (arena_id + 1)) / max(rate, 1e-6)
                print(
                    f"[collect_trials] arena {arena_id + 1}/{n_arenas} "
                    f"q={td.quadrant} goal={td.goal_local} "
                    f"explore: {td.h_explore.shape[0]} steps "
                    f"({reach0}/{n_starts} reached) | "
                    f"exploit: {td.h_exploit.shape[0]} steps "
                    f"({reach1}/{n_starts} reached) | "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        total_steps = sum(td.h_explore.shape[0] + td.h_exploit.shape[0]
                          for td in per_arena.values())
        print(
            f"[collect_trials] done: {n_arenas} arenas, {total_steps} total steps "
            f"in {time.perf_counter() - t_start:.1f}s",
            flush=True,
        )
        meta = {
            "ckpt": bundle.ckpt_path,
            "encoder": bundle.encoder_path,
            "env_size": bundle.env_size,
            "hidden_size": int(bundle.cfg.agent.hidden_size),
            "movement_mode": bundle.cfg.agent.movement_mode,
            "num_arenas": len(bundle.envs),
            "n_starts": int(n_starts),
            "max_steps": int(max_steps),
            "n_distractors_range": [int(n_dist_min), int(n_dist_max)],
            "deterministic": bool(deterministic),
            "seed": int(seed),
        }
        return TrialsDataset(per_arena=per_arena, meta=meta)
