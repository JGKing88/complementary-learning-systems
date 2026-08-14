"""Continuing an interrupted run, as against forking a new one from its weights.

``--load_checkpoint`` and ``--continue_from`` are different operations, and this
module exists so the two cannot quietly become one.

A note on the word, because this repo has used it both ways. The commits and
help text call ``--load_checkpoint`` "resume", and it is not one: it starts a new
run that happens to inherit weights and config. What follows calls that a
**fork**, reserves **continue** for picking one trajectory back up, and names the
new flag ``--continue_from`` rather than ``--resume`` so that no command line has
to be read twice to tell which is meant.

A **fork** takes a parent's weights and config as a starting point and expects to
be retuned on the way in: the sweeps that use it change reward shape, epsilon and
clip coefficient as they go, so Adam's second moments -- an estimate of the
gradient scale *of the objective that produced them* -- are stale by
construction. Dropping them is the correct behavior there, not an oversight, and
``--load_checkpoint`` goes on dropping them.

A **resume** continues one training trajectory across a wall-clock boundary. Then
everything the trajectory carries has to come back, or the second half is not a
continuation of the first:

* Adam's moments. A fresh optimizer needs order ``1/(1-beta2)`` updates to
  rebuild its second-moment estimate, and the transient is a burst of oversized
  steps landing on an already-converged policy.
* The global update counter. Five separate schedules in ``train_navigate`` key
  off it -- stage position, the log_std anneal, the novelty anneal, the
  distractor curriculum, the epsilon anneal -- so a resume that restarts at 1
  silently re-runs the beginning of every one of them.
* The torch and numpy RNG streams, which drive action sampling and the
  distractor draws.
* The env refresher's tick history, which is neither of the above and is handled
  by :meth:`Refresher.fast_forward`.

Only a resume needs any of this, and only the newest point is resumable, so it
lives in one rolling file rather than in every periodic checkpoint. Adam doubles
the parameter bytes -- a 12.9 MB navigate checkpoint would become ~39 MB, times
the dozens a run writes -- and the periodic checkpoints exist to be read by eval
and analysis, which never want optimizer state. ``resume_latest.pt`` is rewritten
on the checkpoint cadence and is the only file that grows.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

RESUME_FILE = "resume_latest.pt"

# Bumped when a field's *meaning* changes, not when one is added -- `load` names
# the mismatch, which is the only reason the file carries a version at all.
SCHEMA = 1

_REQUIRED = ("agent_state_dict", "optimizer_state_dict", "config", "update",
             "kind")


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

def rng_state() -> dict[str, Any]:
    """The streams a rollout draws from, as of now."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any] | None) -> None:
    """Put the streams back where the save found them.

    Tensors are forced to CPU because the caller loads the resume point with
    ``map_location=device``, which moves *every* tensor in it -- including these
    -- and ``set_rng_state`` wants a CPU ByteTensor.
    """
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    np.random.set_state(state["numpy"])

    cuda = state.get("cuda")
    if cuda is None or not torch.cuda.is_available():
        return
    if len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in cuda])
    else:
        # Restoring N states onto M devices would either raise or seed the wrong
        # device; say so instead. The run continues correctly, it just stops
        # being bit-identical to the uninterrupted one.
        print(f"  NOTE: resume point holds {len(cuda)} CUDA RNG states but this "
              f"host has {torch.cuda.device_count()} device(s); the CUDA stream "
              "starts fresh. Training is unaffected apart from losing "
              "bit-for-bit reproducibility against an uninterrupted run.",
              flush=True)


def env_rng_states(worlds, eval_world=None) -> dict[str, list]:
    """Every env's own RNG stream, in the order `_apply` slices them.

    `GridEnv` seeds a `RandomState` at construction and draws from it on each
    reset -- the agent's start cell, and the goal where an env re-rolls one. It
    therefore advances once per rollout and is as much a part of the training
    trajectory as the global streams are. A continuation rebuilds its envs from
    the world spec, which restarts every one of those streams at the spawn
    sequence it used on update 1.

    Not a theoretical concern: with the global torch and numpy streams, the
    optimizer moments, the weights, the config and the world spec all verified
    identical at the resume point, this alone moved the weights by 4e-3 on the
    first continued update.
    """
    return {
        "train": [e.rng.get_state() for w in worlds for e in w.envs],
        "eval": ([e.rng.get_state() for e in eval_world.envs]
                 if eval_world is not None else []),
    }


def restore_env_rng(state: dict[str, list] | None, worlds,
                    eval_world=None) -> None:
    """Put each env's stream back. Must run *after* any refresher fast-forward,
    which rebuilds env objects and would otherwise discard what was restored."""
    if not state:
        return

    def _apply(saved, envs, what):
        if not saved:
            return
        if len(saved) != len(envs):
            # Not fatal: training continues, it just stops being bit-identical.
            # Silence would be worse -- this is the one piece of resume state
            # whose loss is invisible in the logs.
            print(f"  NOTE: resume point holds {len(saved)} {what}-env RNG "
                  f"states but this run built {len(envs)}; those streams start "
                  "fresh and the continuation will not be bit-identical to an "
                  "uninterrupted run.", flush=True)
            return
        for env, s in zip(envs, saved):
            env.rng.set_state(s)

    _apply(state.get("train"), [e for w in worlds for e in w.envs], "train")
    if eval_world is not None:
        _apply(state.get("eval"), eval_world.envs, "eval")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save(save_dir: str, *, kind: str, agent, optimizer, update: int,
         config: dict[str, Any], world_spec: dict[str, Any] | None = None,
         extra: dict[str, Any] | None = None) -> str:
    """Write ``save_dir``'s rolling resume point, atomically.

    Atomically because this is the run's *only* resume point and SLURM's
    wall-clock kill arrives without warning: a torn write during the checkpoint
    cadence would destroy the thing the file exists to protect. ``os.replace``
    is atomic within a directory on POSIX, so a kill mid-write leaves the
    previous point intact and loses one cadence tick rather than the run.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload: dict[str, Any] = {
        "resume_schema": SCHEMA,
        "kind": kind,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "world_spec": world_spec,
        "update": int(update),
        "rng": rng_state(),
    }
    if extra:
        payload.update(extra)
    final = os.path.join(save_dir, RESUME_FILE)
    tmp = final + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    return final


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load(path: str, device, *, kind: str) -> dict[str, Any]:
    """Read a resume point, failing with the reason rather than a ``KeyError``."""
    if os.path.isdir(path):
        path = os.path.join(path, RESUME_FILE)
    if not os.path.exists(path):
        raise SystemExit(
            f"--continue_from: no resume point at {path}.\n"
            f"  A run writes {RESUME_FILE} on its checkpoint cadence, so one "
            "that died before its first checkpoint has none and has to start "
            "over.\n"
            "  Note that the periodic checkpoints (navigate_u*.pt, store_u*.pt) "
            "are fork points, not resume points: they deliberately carry no "
            "optimizer state. Fork them with --load_checkpoint.")

    ck = torch.load(path, map_location=device, weights_only=False)
    missing = [k for k in _REQUIRED if k not in ck]
    if missing:
        raise SystemExit(
            f"--continue_from: {path} is missing {', '.join(missing)}.\n"
            "  Resume points were introduced on 2026-08-14; anything written "
            "before that can be forked with --load_checkpoint but not resumed.")
    if ck.get("resume_schema") != SCHEMA:
        raise SystemExit(
            f"--continue_from: {path} is schema {ck.get('resume_schema')!r}, this "
            f"build reads {SCHEMA}. Fork it with --load_checkpoint instead.")
    if ck["kind"] != kind:
        raise SystemExit(
            f"--continue_from: {path} was written by `{ck['kind']}`, but this is "
            f"`{kind}`. Resume with the script that wrote it.")
    return ck


def restore_optimizer(optimizer, state: dict[str, Any], *, source: str) -> None:
    """Reload Adam's moments, naming a mismatch instead of tripping over it.

    The trainable set is config-dependent -- ``freeze_store`` and
    ``freeze_log_std`` each move it -- and Adam indexes its state by position
    within the param group rather than by name. Torch catches a *count* mismatch
    at load time. It does not catch a *shape* mismatch: that loads clean and
    raises inside the first ``.step()`` as a bare broadcast error naming no
    config field and no file. Both are caught here, where the message can say
    which knob did it and what to do instead.
    """
    mismatch = _first_shape_mismatch(optimizer, state)
    if mismatch is not None:
        pos, saved_shape, live_shape = mismatch
        raise SystemExit(
            f"--continue_from: {source} holds optimizer state whose parameter {pos} "
            f"has shape {saved_shape}, but this run's is {live_shape}.\n"
            "  The architecture moved since the save (hidden_size, "
            "movement_mode, an input channel). A resume continues one "
            "trajectory and cannot change the model -- fork with "
            "--load_checkpoint instead.")
    try:
        optimizer.load_state_dict(state)
    except (ValueError, KeyError) as exc:
        raise SystemExit(
            f"--continue_from: {source} holds optimizer state for a different number "
            f"of trainable parameters.\n  ({exc})\n"
            "  The freeze flags moved since the save; --freeze_store and "
            "--freeze_log_std each change which parameters Adam owns. A resume "
            "cannot change them -- fork with --load_checkpoint instead.") from exc


def _first_shape_mismatch(optimizer, state: dict[str, Any]):
    """``(position, saved_shape, live_shape)`` for the first param that moved."""
    live = [p for g in optimizer.param_groups for p in g["params"]]
    saved_idx = [i for g in state.get("param_groups", []) for i in g["params"]]
    if len(saved_idx) != len(live):
        return None                  # a count mismatch; load_state_dict says so
    saved = state.get("state", {})
    for pos, key in enumerate(saved_idx):
        entry = saved.get(key, saved.get(str(key)))
        if not entry:
            continue                 # a param that has not taken a step yet
        moment = entry.get("exp_avg")
        if moment is None:
            continue
        if tuple(moment.shape) != tuple(live[pos].shape):
            return pos, tuple(moment.shape), tuple(live[pos].shape)
    return None


# ---------------------------------------------------------------------------
# CLI guard
# ---------------------------------------------------------------------------

def reject_overrides(explicit, *, allowed, flag: str = "--continue_from") -> None:
    """A resume continues a run; it does not get to re-specify it.

    Silently ignoring a typed ``--goal_reward`` would be the same class of bug
    as the optimizer being dropped without a word: the run does something other
    than what the command line says, and nothing in the output admits it.
    """
    bad = sorted(set(explicit) - set(allowed))
    if not bad:
        return
    raise SystemExit(
        f"{flag} continues an existing run, so its config comes from the resume "
        f"point, not the command line.\n"
        f"  Remove: {', '.join('--' + b for b in bad)}\n"
        f"  Allowed alongside {flag}: "
        f"{', '.join('--' + a for a in sorted(allowed))}.\n"
        "  To change anything else, fork the run with --load_checkpoint.")
