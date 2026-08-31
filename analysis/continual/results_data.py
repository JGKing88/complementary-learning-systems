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
    n20           the same, at a 20-env stream (the scaling panel)
    incontext     section 5.2 -- adaptation with zero weight updates
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
    "H":      ("Frozen trunk",           "isolate"),
    "I":      ("Experience Replay",      "replay"),
    # Wave 3. Every one of these is handed an oracle task id, which `summarize`
    # now reads off the architecture as well as the method -- see
    # `metrics.TASK_CONDITIONED_ARCHS`. They are upper bounds on their family,
    # not peers of the boundary-free arms above.
    "J":      ("Hypernetwork (HNET)",    "isolate"),
    "K":      ("HNET, frozen base",      "isolate"),
    "L":      ("HNET, from scratch",     "isolate"),
    # Distinct from A2, which is Wave 1's from-scratch *tuning* sweep. This one
    # is the single matched control the from-scratch hypernetwork is read
    # against, at the same configuration. Two frontier rows reading "Naive SGD,
    # from scratch" would look like a duplicate rather than two different runs.
    "L0":     ("Naive SGD, from scratch (matched)", "control"),
    "M":      ("Multi-head",             "isolate"),
    "N":      ("XdG",                    "isolate"),
    "N2":     ("XdG + SI",               "isolate"),
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
            # Wave 3 put a second axis beside the method: a hypernetwork with
            # no regulariser and a plain RNN with one are different runs, and a
            # table keyed on `method` alone would show them under the same name.
            "arch": rs[0].get("arch", "rnn"),
            "params": rs[0].get("params"),
        }
        for k in ("retained", "current_env", "forgetting", "bwt",
                  "stability_gap", "episodes_to_criterion",
                  "criterion_censored_frac", "state_bytes"):
            row[k] = _mean([r[k] for r in rs])
            row[k + "_sem"] = _sem([r[k] for r in rs])
        out.append(row)
    return out


def _read_json(path: str | None) -> dict | None:
    """A side measurement written by another tool, or None if it has not run.

    Missing rather than fatal on purpose: the page has to render before every
    auxiliary result exists, and a half-populated page is more useful during a
    wave than a crash.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ! unreadable {path}: {e}")
        return None


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


def collect_incontext(d: str) -> dict | None:
    """Section 5.2, aggregated over seeds.

    Reports both arms and the difference. The lifetime arm alone is not
    evidence -- only the gap over the episodic control is attributable to
    carrying state -- so the reader never gets one without the other.
    """
    files = sorted(glob.glob(os.path.join(d, "incontext_s*.json")))
    if not files:
        return None
    arms: dict[str, list] = defaultdict(list)
    for fp in files:
        try:
            j = json.load(open(fp))
        except Exception:
            continue
        for arm, v in (j.get("arms") or {}).items():
            arms[arm].append(v)
    if not arms:
        return None
    out: dict = {"seeds": len(files), "arms": {}}
    for arm, rs in arms.items():
        curves = [r["mean_curve"] for r in rs]
        n = min(len(c) for c in curves)
        mean_curve = [_mean([c[i] for c in curves]) for i in range(n)]
        out["arms"][arm] = {
            "mean_curve": mean_curve,
            "first_episode": _mean([r["first_episode"] for r in rs]),
            "last_episode": _mean([r["last_episode"] for r in rs]),
            "adaptation": _mean([r["adaptation"] for r in rs]),
            "adaptation_sem": _sem([r["adaptation"] for r in rs]),
            # The conditional test, which is the headline -- see
            # hopfield_nav/evaluation/incontext.py.
            "memory_lift": _mean([r.get("memory_lift") for r in rs]),
            "memory_lift_sem": _sem([r.get("memory_lift") for r in rs]),
            "p_next_given_hit": _mean([r.get("p_next_given_hit") for r in rs]),
            "p_next_given_miss": _mean([r.get("p_next_given_miss") for r in rs]),
        }
    if "lifetime" in out["arms"] and "episodic" in out["arms"]:
        out["attributable"] = (out["arms"]["lifetime"]["adaptation"]
                               - out["arms"]["episodic"]["adaptation"])
        lt, ep = out["arms"]["lifetime"], out["arms"]["episodic"]
        if lt.get("memory_lift") is not None and ep.get("memory_lift") is not None:
            out["attributable_lift"] = lt["memory_lift"] - ep["memory_lift"]
    #: What a scripted agent that provably remembers scores on this metric, from
    #: tests/test_memory_lift.py. Carried here so the page can state the null as
    #: a fraction of a detectable effect rather than as a bare number near zero.
    out["positive_control_lift"] = 0.559
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave0_dir", required=True)
    p.add_argument("--wave1_dir", required=True)
    p.add_argument("--recorded_dir", required=True)
    p.add_argument("--runs_root", required=True)
    p.add_argument("--n20_dir", default=None,
                   help="The 20-env scaling panel, if it has run.")
    p.add_argument("--incontext_dir", default=None,
                   help="Section 5.2 results, if they have run.")
    p.add_argument("--identifiability", default=None,
                   help="task_identifiability.json. Decides how the Wave 3 "
                        "arms are framed: they are all handed an oracle task "
                        "id, and whether that is a large advantage or a "
                        "formality is this number's job to say.")
    p.add_argument("--beta_calibration", default=None,
                   help="calibrate_beta output, if it was written to a file.")
    p.add_argument("--incontext_generalization", default=None,
                   help="incontext_generalization.json. Decides whether "
                        "section 5.2's result is interpretable at all: if the "
                        "pretrained policy memorised its pool, the held-out "
                        "evaluation is measuring a policy that cannot "
                        "navigate, and a flat curve means nothing.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    joint = W0.load_joint(args.runs_root)
    joint_rows = []
    for (hid, lay, lr, nupd), v in sorted(joint.items()):
        joint_rows.append({
            "hidden": hid, "layers": lay, "lr": lr, "n_updates": nupd,
            "seeds": v["seeds"],
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
        "n20": collect_methods(args.n20_dir) if args.n20_dir else [],
        "incontext": (collect_incontext(args.incontext_dir)
                      if args.incontext_dir else None),
        "identifiability": _read_json(args.identifiability),
        "incontext_generalization": _read_json(args.incontext_generalization),
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
    ic = data.get("incontext")
    print(f"  oracle={data['oracle']}  joint_rows={len(joint_rows)}  "
          f"scratch_arms={len(scratch)}  recorded={len(data['recorded'])}  "
          f"method_configs={len(data['methods'])}  n20={len(data['n20'])}  "
          f"incontext={'yes' if ic else 'no'}")


if __name__ == "__main__":
    main()
