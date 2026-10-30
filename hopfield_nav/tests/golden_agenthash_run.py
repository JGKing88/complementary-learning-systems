"""Golden fingerprint of the continual protocol CLI (``analysis/continual/agenthash.py``).

    python -m hopfield_nav.tests.golden_agenthash_run           # write the golden
    python -m hopfield_nav.tests.golden_agenthash_run --check   # compare, write nothing

No other golden runs this CLI. The stub-scaffold goldens cover
``protocols.run_sequential_protocol`` through ``evaluate_sequential_episodes``,
but agenthash.py has its own outer loop, its own memory construction and its
own store logic. This trains the tiny task-recipe checkpoint that
``golden_task_run`` uses, runs the continual protocol on it twice
(deterministic and sampled), and records each run's whole history except the
paths it prints.

Added before stage 2 of GRID_MLP_NAV_PLAN (the sensory_kv backend in the
evaluators), so the Hopfield path can be shown unchanged by it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from hopfield_nav.tests.golden_task_run import REPO_ROOT, TASK_ARGS, _run, compare

GOLDEN = Path(__file__).resolve().parent / "golden" / "agenthash_run.npz"

PROTOCOL_ARGS = ["--device", "cpu", "--static_vectorhash", "--iters_per_block", "3",
                 "--max_steps", "20", "--seed", "0", "--oracle_store_at_goal",
                 "--oracle_lock_store_not_at_goal"]


def _strip(history: dict) -> dict:
    meta = history["metadata"]
    meta.pop("ckpt_path", None)
    meta.pop("run_name", None)
    meta.get("extra", {}).pop("encoder_path", None)
    return history


def fingerprint(train_extra: list[str] | None = None,
                protocol_extra: list[str] | None = None) -> dict[str, np.ndarray]:
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
        _run(["hopfield_nav.train_navigate", "--encoder_checkpoint", str(enc),
              *TASK_ARGS, *(train_extra or []), "--save_dir", str(save)], env)
        out = {}
        for name, extra in (("det", []), ("sampled", ["--stochastic_policy"])):
            hist = Path(root) / f"{name}.json"
            _run(["analysis.continual.agenthash", "--ckpt", str(save / "navigate_final.pt"),
                  "--out", str(hist), *PROTOCOL_ARGS, *extra, *(protocol_extra or [])], env)
            out[name] = np.array(json.dumps(_strip(json.load(open(hist))), sort_keys=True))
        return out


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
    for k, v in fresh.items():
        h = json.loads(str(v))
        print(f"wrote {k}: trace={len(h['trace'])} steps, blocks={h['blocks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
