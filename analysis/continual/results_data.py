"""Collect every number the results page shows into one JSON.

The page is generated from this rather than hand-transcribed, because
transcription is where numbers quietly stop matching the runs that produced
them. One command, one file, and the page's provenance is a path.

Emits:

    oracle        T0.3, the eval's own ceiling
    joint         T0.1, per (hidden, layers, lr), with the convergence slope
    scratch       T0.4, the from-scratch floor per arm
    recorded      the pre-existing histories the suite is measured against
    methods       every Wave-1 / Wave-2 arm, aggregated over seeds
    frontier      the cost axes, including the Hopfield agent's constants
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

from . import metrics as M
from . import wave0_summary as W0


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _sem(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


#: arm prefix -> (display name, family). Everything else in the directory is
#: ignored rather than silently folded into a table it does not belong in.
ARMS = {
    "A":      ("Naive SGD (tuned)",      "control"),
    "Abatch": ("Naive SGD, batch_envs=16", "control"),
    "A2":     ("Naive SGD, from scratch", "control"),
    "R":      ("No method (reference)",  "control"),
    "B":      ("Experience Replay",      "replay"),
    "C":      ("Online EWC",             "regularize"),
    "D":      ("CLEAR",                  "replay"),
    "E":      ("DER++",                  "replay"),
    "F":      ("Synaptic Intelligence",  "regularize"),
    "G":      ("LwF",                    "distill"),
}


def collect_methods(hist_dir: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(hist_dir, "*.json"))):
        base = os.path.splitext(os.path.basename(p))[0]
        prefix = base.split("_", 1)[0]
        if prefix not in ARMS:
            continue
        try:
            hist = json.load(open(p))
        except Exception:
            continue
        if not hist.get("blocks"):
            continue
        groups[re.sub(r"_s\d+$", "", base)].append(M.summarize(hist))

    out = []
    for label, rs in sorted(groups.items()):
        prefix = label.split("_", 1)[0]
        display, family = ARMS[prefix]
        row = {
            "config": label,
            "arm": prefix,
            "display": display,
            "family": family,
            "seeds": len(rs),
            "method": rs[0]["method"],
            "needs_task_boundaries": rs[0]["needs_task_boundaries"],
            "needs_task_id": rs[0]["needs_task_id"],
        }
        for k in ("retained", "current_env", "forgetting", "bwt",
                  "stability_gap", "episodes_to_criterion",
                  "criterion_censored_frac", "state_bytes"):
            row[k] = _mean([r[k] for r in rs])
            row[k + "_sem"] = _sem([r[k] for r in rs])
        out.append(row)
    return out


def collect_recorded(hist_dir: str) -> list[dict]:
    """The pre-existing histories: the Hopfield agent and the recorded RNN
    baselines. These are what the whole suite is measured against, so they
    belong in the same file rather than being remembered separately."""
    wanted = [
        ("agenthash_w_oracle.json", "Hopfield store (frozen policy)", "hopfield"),
        ("baseline_regular_200steps.json", "RNN, pretrain -> sequential", "recorded"),
        ("20x20_pretrained_10_full_iters.json", "RNN, pretrain -> finetune", "recorded"),
    ]
    out = []
    for fn, display, family in wanted:
        p = os.path.join(hist_dir, fn)
        if not os.path.exists(p):
            continue
        try:
            hist = json.load(open(p))
        except Exception:
            continue
        s = M.summarize(hist)
        s.update({"display": display, "family": family, "file": fn})
        p_mat = M.performance_matrix(hist)
        last = max(p_mat) if p_mat else None
        s["per_env"] = ({int(k): v for k, v in sorted(p_mat[last].items())}
                        if last is not None else {})
        out.append(s)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave0_dir", required=True)
    p.add_argument("--wave1_dir", required=True)
    p.add_argument("--recorded_dir", required=True)
    p.add_argument("--runs_root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    joint = W0.load_joint(args.runs_root)
    joint_rows = []
    for (hid, lay, lr), v in sorted(joint.items()):
        joint_rows.append({
            "hidden": hid, "layers": lay, "lr": lr, "seeds": v["seeds"],
            "final": _mean(v["final"]), "final_sem": _sem(v["final"]),
            "at200": _mean(v["at200"]), "end_slope": _mean(v["slope"]),
        })

    scratch = {}
    for arm in ("noprev", "prev"):
        r = W0.load_sequential(args.wave0_dir, arm)
        if r["seeds"]:
            pe = r["per_env"]
            n = max(pe) + 1
            scratch[arm] = {
                "seeds": r["seeds"],
                "per_env": {i: _mean(pe[i]) for i in sorted(pe)},
                "retained": _mean([_mean(pe[i]) for i in range(n - 1) if i in pe]),
                "current": _mean(pe.get(n - 1, [])),
            }

    data = {
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "oracle": W0.load_oracle(args.wave0_dir),
        "joint": joint_rows,
        "scratch": scratch,
        "recorded": collect_recorded(args.recorded_dir),
        "methods": collect_methods(args.wave1_dir),
        "hopfield_costs": {
            # Constants of the model, not measurements -- stated here so the
            # frontier figure has both ends of every axis in one place.
            "gradient_steps_per_env": 0,
            "episodes_per_env": 1,
            "stores_raw_data": False,
            "needs_task_id": False,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[results] wrote {args.out}")
    print(f"  oracle={data['oracle']}  joint_rows={len(joint_rows)}  "
          f"scratch_arms={len(scratch)}  recorded={len(data['recorded'])}  "
          f"method_configs={len(data['methods'])}")


if __name__ == "__main__":
    main()
