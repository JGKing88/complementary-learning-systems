"""Merge N single-iter history JSONs into one multi-iter history JSON.

Each input is typically the output of ``baseline.py`` / ``agenthash.py`` invoked
with ``--num_full_iters 1`` (so its trace metrics are scalars). The merged
output's trace metrics become length-N lists, suitable for the mean ± 1σ plot
path in ``plotting.py``. Per-iter metadata fields (env_goals_per_iter,
env_offsets_per_iter, stored_at_goal_count_per_iter) are concatenated so the
final history records every iter's envs.

Sanity-checks: all inputs must share ``model_class``, ``n_envs``, ``env_size``,
``iters_per_block``. Mismatches abort with a clear error.
"""
from __future__ import annotations

import argparse
import json
import os

from .baseline import _merge_iter_traces


def _load_history(path: str) -> tuple[list, list, dict]:
    with open(path) as f:
        h = json.load(f)
    trace = []
    for entry in h["trace"]:
        step, train_env, inner_str = entry
        inner = {int(k): v for k, v in inner_str.items()}
        trace.append((int(step), int(train_env), inner))
    blocks = [(int(b[0]), int(b[1]), int(b[2])) for b in h["blocks"]]
    return trace, blocks, h.get("metadata", {})


def _check_compat(metadatas: list[dict]) -> None:
    """Abort if structural metadata fields differ across inputs."""
    keys = ("model_class", "n_envs", "env_size", "iters_per_block")
    base = {k: metadatas[0].get(k) for k in keys}
    for i, md in enumerate(metadatas[1:], start=1):
        diffs = {k: (base[k], md.get(k)) for k in keys if md.get(k) != base[k]}
        if diffs:
            raise RuntimeError(
                f"input {i} disagrees with input 0 on structural fields: {diffs}"
            )


def merge(in_paths: list[str], out_path: str, run_name: str | None) -> None:
    if not in_paths:
        raise RuntimeError("merge: no input histories provided")
    iter_traces: list[tuple[list, list]] = []
    metadatas: list[dict] = []
    for p in in_paths:
        trace, blocks, md = _load_history(p)
        iter_traces.append((trace, blocks))
        metadatas.append(md)
    _check_compat(metadatas)

    merged_trace, merged_blocks = _merge_iter_traces(iter_traces)

    n = len(metadatas)
    base_md = dict(metadatas[0])
    base_md["num_full_iters"] = n
    if run_name is not None:
        base_md["run_name"] = run_name

    # Concat per-iter lists in `extra`. Each input has these as length-1 lists
    # (when num_full_iters=1) or full lists (already-merged inputs).
    base_extra = dict(base_md.get("extra", {}))
    for key in (
        "env_goals_per_iter",
        "env_offsets_per_iter",
        "stored_at_goal_count_per_iter",
    ):
        combined = []
        any_present = False
        for md in metadatas:
            v = md.get("extra", {}).get(key)
            if v is None:
                continue
            any_present = True
            if isinstance(v, list):
                combined.extend(v)
            else:
                combined.append(v)
        if any_present:
            base_extra[key] = combined
    base_md["extra"] = base_extra

    out = {
        "metadata": base_md,
        "trace": [
            [s, t, {str(k): v for k, v in inner.items()}]
            for s, t, inner in merged_trace
        ],
        "blocks": [list(b) for b in merged_blocks],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[merge_histories] merged {n} histories → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Per-iter history JSONs to merge.")
    p.add_argument("--out", required=True,
                   help="Output path for the merged history JSON.")
    p.add_argument("--run_name", default=None,
                   help="Override the merged history's metadata.run_name. "
                        "Default: keep input 0's run_name.")
    args = p.parse_args()
    merge(args.inputs, args.out, args.run_name)


if __name__ == "__main__":
    main()
