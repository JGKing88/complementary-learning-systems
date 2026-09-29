"""Golden fingerprint of a tiny end-to-end ``task3r_k2_h128``-recipe run.

    python -m hopfield_nav.tests.golden_task_run           # write the golden
    python -m hopfield_nav.tests.golden_task_run --check   # compare, write nothing

The stub-scaffold goldens (``gen_golden.py``) never touch the task-faithful
path: ``TaskRegime`` memories with distractors, the oracle store under
``task_mode``, memory carried across ``visits``, redrawn goals, and
``evaluate_task``. This runs that path end to end through the real CLI on a
deliberately tiny world and records the final parameters of every saved
checkpoint plus the eval metrics logged along the way. ``test_golden_task_run``
asserts a fresh run reproduces them bit-for-bit.

Added before the memory-backend refactor (idea 1, docs/GRID_MLP_NAV_PLAN.md)
so the Hopfield backend can be shown unchanged by it. Regenerate only for an
intended behavior change, and say so in the commit message.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden" / "task_run.npz"

# The task3r_k2_h128 recipe (run_nav_p2.sh task block + globals), shrunk.
TASK_ARGS = [
    "--lambdas", "3", "4", "--Np", "40",
    "--size", "4", "--observation_size", "16",
    "--batch_envs", "2", "--steps_per_rollout", "8",
    "--schedule", "task:3,visits=2", "--env_repeats", "2",
    "--envs_per_world", "3", "--num_worlds", "1",
    "--redraw_goal_per_rollout",
    "--num_val_envs", "2", "--eval_every", "1", "--eval_scope", "task",
    "--n_train_distractors_min", "0", "--n_train_distractors_max", "3",
    "--val_distractors", "0", "2",
    "--input_hopfield_raw", "--input_hopfield_multistep", "1",
    "--input_sensory", "--input_prev_displacement",
    "--action_polar", "--state_dependent_std", "--log_kappa_max", "2.5",
    "--movement_mode", "continuous",
    "--max_action_norm", "1.0", "--min_action_norm", "0.5",
    "--goal_radius", "1.0", "--hidden_size", "16",
    "--device", "cpu", "--static-vectorhash", "--seed", "7",
]


def _run(args, env):
    proc = subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"`python -m {' '.join(args[:1])}` exited "
                           f"{proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return proc


def fingerprint(extra_args: list[str] | None = None) -> dict[str, np.ndarray]:
    """Run the tiny task job in a sandbox; return {name: array}."""
    with tempfile.TemporaryDirectory() as root:
        env = dict(os.environ, CLS_RUNS=root, WANDB_MODE="disabled",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        enc = Path(root) / "enc.pt"
        _run(["encoder_training.save_untrained_encoder", "--encoder-type", "mlp",
              "--out-dim", "8", "--hidden-dim", "32", "--num-hidden-layers", "2",
              "--gain", "5.0", "--lambdas", "3", "4", "--seed", "0",
              "--out", str(enc)], env)
        save = Path(root) / "run"
        proc = _run(["hopfield_nav.train_navigate", "--encoder_checkpoint", str(enc),
                     *TASK_ARGS, *(extra_args or []), "--save_dir", str(save)], env)
        out: dict[str, np.ndarray] = {}
        for ck_path in sorted(save.glob("*.pt")):
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            for k, v in ck["agent_state_dict"].items():
                out[f"{ck_path.stem}/{k}"] = v.detach().cpu().numpy()
        # The printed eval dicts (nav= / disc= / expl= / task=), in order.
        # eval_seconds is wall-clock and deliberately left out.
        evals = [ln.strip() for ln in proc.stdout.splitlines()
                 if any(f"] {k}=" in ln for k in ("nav", "disc", "expl", "task"))]
        if not evals:
            raise RuntimeError("no eval lines captured -- has the log format changed?")
        out["eval_lines"] = np.array(evals)
        for p in sorted(save.rglob("*.json")):
            out[f"json/{p.relative_to(save)}"] = np.array(p.read_text())
        return out


def compare(fresh: dict, golden: dict) -> list[str]:
    diffs = []
    for k in sorted(set(golden) | set(fresh)):
        if k.startswith("json/"):
            continue      # carries paths / timestamps; eval numbers come from the log
        if k not in fresh or k not in golden:
            diffs.append(f"{k}: present only in {'golden' if k in golden else 'fresh'}")
        elif golden[k].shape != fresh[k].shape or not np.array_equal(golden[k], fresh[k]):
            diffs.append(f"{k}: differs")
    return diffs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    fresh = fingerprint()
    if a.check:
        golden = dict(np.load(GOLDEN, allow_pickle=False))
        diffs = compare(fresh, golden)
        print("\n".join(diffs) if diffs else f"OK: {len(golden)} entries match")
        return 1 if diffs else 0
    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez_compressed(GOLDEN, **fresh)
    print(f"wrote {GOLDEN}: {len(fresh)} entries")
    for ln in fresh["eval_lines"]:
        print("  ", ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
