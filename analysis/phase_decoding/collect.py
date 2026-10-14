"""Per-step collection for the explore/exploit abstraction analysis.

Two definitions of phase, one step loop.

Definition A, set by memory content. Each trial starts from a fresh RNN state
at a random non-goal cell. The Hopfield holds n distractors (explore, phase 0)
or the goal plus n distractors (exploit, phase 1). Nothing is ever written
during a trial: this is ``evaluate_navigation``'s contract, and d0_base's
exploit rollouts and task3r's revisit rollouts. The trial ends on the first
step after which the agent stands on the goal.

Definition B, the agent's own switch (the task-faithful contract,
``collector.py`` task_mode). The Hopfield starts with distractors only. On the
first step whose input is at the goal, the move is discarded, the oracle writes
the goal-cell pattern (``store`` trials) or writes nothing (``sham`` trials),
and the agent is teleported to a uniform non-goal cell with heading 0 and zero
displacement. The RNN state and previous reward/action are kept. Later arrivals
teleport the same way and never write. Phase 0 is every step before the first
arrival; phase 1 is every step after it.

The policy's own store head is never honoured: training ignores it in both
contracts, and honouring it is what voided the previous version of this
analysis (docs/PHASE_DECODING_AUDIT.md §1).
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hopfield_nav.evaluation.metrics import agent_step, random_start
from hopfield_nav.policy import channels
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world.env import at_goal

from .rollout import RolloutEngine

STEP_FIELDS = ("h", "x", "phase", "trial", "t", "t_seg", "pos", "dist", "speed",
               "wall", "qnorm", "at_goal_in", "n_mem", "n_dist")


@dataclass(frozen=True)
class CollectConfig:
    definition: str          # "A" or "B"
    n_trials: int            # per condition per arena
    max_steps: int           # A: per trial; B: cap on the search before the first arrival
    post_steps: int          # B: steps recorded after the first arrival; A: unused, must be 0
    n_dist_min: int
    n_dist_max: int
    deterministic: bool
    seed: int

    def __post_init__(self):
        assert self.definition in ("A", "B"), self.definition
        assert (self.post_steps == 0) == (self.definition == "A"), \
            "post_steps is a definition-B parameter"
        assert 0 <= self.n_dist_min <= self.n_dist_max


def _place(env, pos: tuple[int, int]) -> None:
    """Put the agent at ``pos`` the way the batched envs reset it: heading 0 and
    no previous displacement. ``set_position`` alone keeps both from the last
    trial."""
    env.set_position(pos)
    env._heading_rad = 0.0
    env._last_displacement = np.zeros(2)


def signal_slice(engine: RolloutEngine) -> slice:
    specs = channels.channel_specs(engine.cfg.agent, engine.embed_dim,
                                   engine.cfg.env.observation_size)
    off = 0
    for s in specs:
        if s.name == "hopfield_signal":
            return slice(off, off + s.width)
        off += s.width
    raise ValueError("policy has no hopfield_signal channel")


def run_trial(engine: RolloutEngine, env, env_offset, *, hopfield, mode: str,
              max_steps: int, post_steps: int, deterministic: bool,
              rng: np.random.RandomState, sig: slice, phase_init: int) -> dict:
    """One trial; returns per-step arrays and a summary.

    mode: "end" (definition A), "store" or "sham" (definition B).
    """
    assert mode in ("end", "store", "sham")
    goal = env.goal_location
    _place(env, random_start(env.size, goal, rng))
    h_rnn = prev_reward = prev_action = None
    rec = {k: [] for k in STEP_FIELDS if k not in ("trial", "n_dist")}
    phase, t_seg, switch_t, n_reach_post = phase_init, 0, None, 0
    reached = False
    t = 0
    while True:
        if mode == "end" and t >= max_steps:
            break
        if mode != "end":
            if switch_t is None and t >= max_steps:
                break                                   # search never found the goal
            if switch_t is not None and t >= switch_t + 1 + post_steps:
                break
        arrived = mode != "end" and at_goal(env)
        pos = np.asarray(env._continuous_pos if hasattr(env, "_continuous_pos")
                         else env.current_location, dtype=np.float64).copy()
        n_mem = int(hopfield.num_memories)
        out = agent_step(engine.agent, env, env_offset, engine.vh, hopfield,
                         h_rnn, engine.cfg, engine.device,
                         deterministic=deterministic, goal_local=goal,
                         goal_in_memory=bool(phase), prev_reward=prev_reward,
                         prev_action=prev_action)
        x = out["policy_input"].reshape(-1).cpu().numpy().astype(np.float32)
        disp = np.asarray(env._last_displacement, dtype=np.float64)
        act = out["next_prev_action"].reshape(-1).cpu().numpy().astype(np.float64)
        intended = np.linalg.norm(act) * float(engine.cfg.env.continuous_scale)

        rec["h"].append(out["h_rnn"][-1, 0].cpu().numpy().astype(np.float32))
        rec["x"].append(x)
        rec["phase"].append(phase)
        rec["t"].append(t)
        rec["t_seg"].append(t_seg)
        rec["pos"].append(pos)
        rec["dist"].append(float(np.hypot(pos[0] - goal[0], pos[1] - goal[1])))
        rec["speed"].append(float(np.linalg.norm(disp)) if not arrived else 0.0)
        rec["wall"].append(bool((not arrived) and np.linalg.norm(disp) < 0.99 * intended))
        rec["qnorm"].append(float(np.linalg.norm(x[sig])))
        rec["at_goal_in"].append(bool(arrived))
        rec["n_mem"].append(n_mem)

        h_rnn = out["h_rnn"]
        prev_reward = out["next_prev_reward"]
        prev_action = out["next_prev_action"]
        t += 1
        t_seg += 1

        if mode == "end":
            if at_goal(env):
                reached = True
                break
            continue
        if arrived:
            # The at-goal step's move is discarded: teleport overrides it.
            if switch_t is None:
                switch_t = t - 1
                if mode == "store":
                    hopfield.input_memory(out["store_embedding"][0])
                phase = 1
            else:
                n_reach_post += 1
            _place(env, random_start(env.size, goal, rng))
            t_seg = 0

    arrs = {k: np.asarray(v) for k, v in rec.items()}
    return {"steps": arrs, "reached": reached, "steps_taken": t,
            "switch_t": switch_t, "n_reach_post": n_reach_post}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def collect(engine: RolloutEngine, cc: CollectConfig, out_dir: Path) -> None:
    """Collect every arena of ``engine``'s val set; one npz per arena."""
    out_dir = Path(out_dir)
    (out_dir / "per_arena").mkdir(parents=True, exist_ok=True)
    bundle = engine.build_bundle()
    sig = signal_slice(engine)
    rng = np.random.RandomState(cc.seed)
    torch.manual_seed(cc.seed)
    conditions = (("explore", "end", 0), ("exploit", "end", 1)) if cc.definition == "A" \
        else (("store", "store", 0), ("sham", "sham", 0))
    summary = []
    t0 = time.perf_counter()
    trial_id = 0
    for a, env in enumerate(bundle.envs):
        off = bundle.offsets[a]
        goal = env.goal_location
        chunks = {k: [] for k in STEP_FIELDS}
        cond_col = []
        for cname, mode, phase_init in conditions:
            for _ in range(cc.n_trials):
                n_dist = int(rng.randint(cc.n_dist_min, cc.n_dist_max + 1))
                # As in evaluate_navigation: goal (exploit only) and
                # distractors, shuffled so storage order is random.
                patterns = list(sample_distractors(engine.vh, off, env.size, n_dist, rng))
                if cname == "exploit":
                    patterns.append(goal_encoding(engine.vh, off, goal))
                rng.shuffle(patterns)
                hop = engine.make_hopfield()
                for pat in patterns:
                    hop.input_memory(torch.from_numpy(pat).float())
                r = run_trial(engine, env, off, hopfield=hop, mode=mode,
                              max_steps=cc.max_steps, post_steps=cc.post_steps,
                              deterministic=cc.deterministic, rng=rng, sig=sig,
                              phase_init=phase_init)
                n = len(r["steps"]["t"])
                for k, v in r["steps"].items():
                    chunks[k].append(v)
                chunks["trial"].append(np.full(n, trial_id))
                chunks["n_dist"].append(np.full(n, n_dist))
                cond_col.append(np.full(n, cname))
                summary.append({"arena": a, "trial": trial_id, "condition": cname,
                                "n_dist": n_dist, "reached": r["reached"],
                                "steps_taken": r["steps_taken"], "switch_t": r["switch_t"],
                                "n_reach_post": r["n_reach_post"]})
                trial_id += 1
        np.savez(out_dir / "per_arena" / f"{a:04d}.npz",
                 condition=np.concatenate(cond_col), arena=np.int64(a),
                 goal=np.asarray(goal), offset=np.asarray(off),
                 **{k: np.concatenate(v) for k, v in chunks.items()})
        el = time.perf_counter() - t0
        print(f"[collect] arena {a + 1}/{len(bundle.envs)} {el:.0f}s", flush=True)

    specs = channels.channel_specs(engine.cfg.agent, engine.embed_dim,
                                   engine.cfg.env.observation_size)
    meta = {
        "ckpt": engine.ckpt_path, "encoder": engine.encoder_path,
        "random_agent": engine.random_agent, "random_init_seed": engine.random_init_seed,
        "split": engine.split, "val_seed": engine.val_seed,
        "n_arenas": len(bundle.envs), "env_size": bundle.env_size,
        "hidden_size": int(engine.cfg.agent.hidden_size),
        "channels": [[s.name, s.width] for s in specs],
        "collect_config": cc.__dict__, "git_sha": _git_sha(),
        "torch": torch.__version__, "numpy": np.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "trials.json").write_text(json.dumps(summary))
