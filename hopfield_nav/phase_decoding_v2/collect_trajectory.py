"""Two-episode trajectory collector for Exp 2.

One trajectory =
    1. random non-goal start, hopfield seeded with N distractors only
    2. roll until goal (cap at max_steps_ep1); phase=0 every step
    3. ORACLE-STORE: write _goal_encoding(...) into the hopfield. Mark switch_t.
    4. teleport to a new random non-goal start. DO NOT reset h_rnn, prev_*, or
       hopfield.
    5. roll until goal (cap at max_steps_ep2) with goal_in_memory=True; phase=1
       every step.

If episode 1 doesn't reach the goal, the trajectory is dropped (no switch event
to label). Many trajectories may be collected per arena; saved as a flat list.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hopfield_nav.eval import _goal_encoding, random_start

from .rollout import EnvBundle, RolloutEngine


@dataclass
class TwoEpisodeTrajectory:
    arena_id: int
    traj_id: int
    quadrant: int
    h: np.ndarray            # (T, H)
    phase: np.ndarray        # (T,) int8
    switch_t: int            # index of first phase=1 step
    positions: np.ndarray    # (T, 2)
    ep1_steps: int
    ep2_steps: int
    ep2_reached: bool


@dataclass
class TrajectoryDataset:
    trajectories: list[TwoEpisodeTrajectory]
    meta: dict = field(default_factory=dict)

    def pooled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (h, phase, traj_id) flattened across all trajectories."""
        if not self.trajectories:
            H = self.meta.get("hidden_size", 0)
            return (np.zeros((0, H), dtype=np.float32),
                    np.zeros((0,), dtype=np.int8),
                    np.zeros((0,), dtype=np.int64))
        h_chunks, phase_chunks, tid_chunks = [], [], []
        for k, traj in enumerate(self.trajectories):
            h_chunks.append(traj.h)
            phase_chunks.append(traj.phase)
            tid_chunks.append(np.full(traj.h.shape[0], k, dtype=np.int64))
        return (np.concatenate(h_chunks, axis=0),
                np.concatenate(phase_chunks, axis=0),
                np.concatenate(tid_chunks, axis=0))

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        (out / "trajectories").mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta["n_trajectories"] = len(self.trajectories)
        meta["per_trajectory"] = []
        for k, traj in enumerate(self.trajectories):
            np.savez(
                out / "trajectories" / f"{k:04d}.npz",
                h=traj.h, phase=traj.phase, positions=traj.positions,
                switch_t=np.int64(traj.switch_t),
                arena_id=np.int64(traj.arena_id),
                traj_id=np.int64(traj.traj_id),
                quadrant=np.int64(traj.quadrant),
            )
            meta["per_trajectory"].append({
                "k": k,
                "arena_id": traj.arena_id,
                "traj_id": traj.traj_id,
                "quadrant": traj.quadrant,
                "T": int(traj.h.shape[0]),
                "switch_t": int(traj.switch_t),
                "ep1_steps": int(traj.ep1_steps),
                "ep2_steps": int(traj.ep2_steps),
                "ep2_reached": bool(traj.ep2_reached),
            })
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, out_dir: str | Path) -> "TrajectoryDataset":
        out = Path(out_dir)
        meta = json.loads((out / "meta.json").read_text())
        meta.pop("per_trajectory", None)
        meta.pop("n_trajectories", None)
        trajs: list[TwoEpisodeTrajectory] = []
        for f in sorted((out / "trajectories").glob("*.npz")):
            z = np.load(f)
            trajs.append(TwoEpisodeTrajectory(
                arena_id=int(z["arena_id"]),
                traj_id=int(z["traj_id"]),
                quadrant=int(z["quadrant"]),
                h=np.asarray(z["h"], dtype=np.float32),
                phase=np.asarray(z["phase"], dtype=np.int8),
                switch_t=int(z["switch_t"]),
                positions=np.asarray(z["positions"], dtype=np.int32),
                ep1_steps=0, ep2_steps=0, ep2_reached=False,  # not persisted; meta only
            ))
        return cls(trajectories=trajs, meta=meta)


class TwoEpisodeTrajectoryCollector:
    def __init__(self, engine: RolloutEngine) -> None:
        self.engine = engine

    def _one_trajectory(
        self,
        env,
        env_offset: tuple[int, int],
        n_distractors: int,
        max_steps_ep1: int,
        max_steps_ep2: int,
        deterministic: bool,
        rng: np.random.RandomState,
    ) -> dict | None:
        hopfield = self.engine.make_hopfield()
        self.engine.seed_distractors(hopfield, env_offset, n_distractors, rng)

        goal = env.goal_location
        env.set_position(random_start(env.size, goal, rng))

        ep1 = self.engine.rollout(
            env, env_offset,
            hopfield=hopfield, h_rnn=None,
            prev_reward=None, prev_action=None,
            goal_in_memory_flag=False,
            max_steps=max_steps_ep1,
            deterministic=deterministic,
            record_positions=True,
            stop_on_goal=True,
        )
        if not ep1["reached"]:
            return None

        # Oracle-store goal embedding into the (still-live) hopfield.
        goal_enc = _goal_encoding(self.engine.vh, env_offset, goal)
        hopfield.input_memory(torch.from_numpy(goal_enc).float())

        # Teleport without resetting h_rnn / prev_* / hopfield.
        env.set_position(random_start(env.size, goal, rng))

        ep2 = self.engine.rollout(
            env, env_offset,
            hopfield=hopfield,
            h_rnn=ep1["h_rnn"],
            prev_reward=ep1["prev_reward"],
            prev_action=ep1["prev_action"],
            goal_in_memory_flag=True,
            max_steps=max_steps_ep2,
            deterministic=deterministic,
            record_positions=True,
            stop_on_goal=True,
        )

        h = np.concatenate([ep1["h"], ep2["h"]], axis=0)
        positions = np.concatenate([ep1["positions"], ep2["positions"]], axis=0)
        switch_t = int(ep1["h"].shape[0])
        phase = np.concatenate([
            np.zeros(ep1["h"].shape[0], dtype=np.int8),
            np.ones(ep2["h"].shape[0], dtype=np.int8),
        ], axis=0)

        return {
            "h": h, "phase": phase, "positions": positions,
            "switch_t": switch_t,
            "ep1_steps": int(ep1["steps_taken"]),
            "ep2_steps": int(ep2["steps_taken"]),
            "ep2_reached": bool(ep2["reached"]),
        }

    def collect(
        self,
        bundle: EnvBundle,
        *,
        n_traj_per_arena: int,
        max_steps_ep1: int,
        max_steps_ep2: int,
        n_dist_min: int,
        n_dist_max: int,
        deterministic: bool,
        seed: int = 0,
        progress_every: int = 1,
    ) -> TrajectoryDataset:
        rng = np.random.RandomState(seed)
        trajs: list[TwoEpisodeTrajectory] = []
        n_dropped = 0
        n_arenas = len(bundle.envs)
        target = n_arenas * n_traj_per_arena
        print(
            f"[collect_trajectory] starting: {n_arenas} arenas × "
            f"{n_traj_per_arena} traj = {target} target trajectories, "
            f"max_steps_ep1={max_steps_ep1} max_steps_ep2={max_steps_ep2} "
            f"deterministic={deterministic}",
            flush=True,
        )
        t_start = time.perf_counter()
        for arena_id, env in enumerate(bundle.envs):
            env_offset = bundle.offsets[arena_id]
            arena_kept = 0
            arena_dropped = 0
            for j in range(n_traj_per_arena):
                n_dist = int(rng.randint(n_dist_min, n_dist_max + 1))
                res = self._one_trajectory(
                    env, env_offset,
                    n_distractors=n_dist,
                    max_steps_ep1=max_steps_ep1,
                    max_steps_ep2=max_steps_ep2,
                    deterministic=deterministic,
                    rng=rng,
                )
                if res is None:
                    n_dropped += 1
                    arena_dropped += 1
                    continue
                trajs.append(TwoEpisodeTrajectory(
                    arena_id=arena_id,
                    traj_id=j,
                    quadrant=bundle.quadrants[arena_id],
                    h=res["h"], phase=res["phase"], positions=res["positions"],
                    switch_t=res["switch_t"],
                    ep1_steps=res["ep1_steps"],
                    ep2_steps=res["ep2_steps"],
                    ep2_reached=res["ep2_reached"],
                ))
                arena_kept += 1

            if (arena_id + 1) % progress_every == 0 or arena_id + 1 == n_arenas:
                elapsed = time.perf_counter() - t_start
                rate = (arena_id + 1) / max(elapsed, 1e-6)
                eta = (n_arenas - (arena_id + 1)) / max(rate, 1e-6)
                print(
                    f"[collect_trajectory] arena {arena_id + 1}/{n_arenas} "
                    f"q={bundle.quadrants[arena_id]} "
                    f"kept={arena_kept}/{n_traj_per_arena} dropped={arena_dropped} | "
                    f"total kept={len(trajs)} dropped={n_dropped} | "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        print(
            f"[collect_trajectory] done: {len(trajs)} kept / {n_dropped} dropped "
            f"({target} target) in {time.perf_counter() - t_start:.1f}s",
            flush=True,
        )
        meta = {
            "ckpt": bundle.ckpt_path,
            "encoder": bundle.encoder_path,
            "env_size": bundle.env_size,
            "hidden_size": int(bundle.cfg.agent.hidden_size),
            "num_arenas": len(bundle.envs),
            "n_traj_per_arena": int(n_traj_per_arena),
            "max_steps_ep1": int(max_steps_ep1),
            "max_steps_ep2": int(max_steps_ep2),
            "n_distractors_range": [int(n_dist_min), int(n_dist_max)],
            "deterministic": bool(deterministic),
            "seed": int(seed),
            "n_dropped_no_ep1_goal": int(n_dropped),
        }
        return TrajectoryDataset(trajectories=trajs, meta=meta)
