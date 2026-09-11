"""Collect goal-pair runs into one table.

    python -m analysis.goal_pairs_results --prefix a1_

Reads every `<CLS_RUNS>/goal_pairs/<prefix>*/final_tables.json`, prints one
row per run with the six heldout cells (model and NN line) plus train
train×train, and writes the same as CSV beside the runs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

CELLS = ["trainxtrain", "trainxgoal_heldout", "trainxregion",
         "regionxtrain", "regionxgoal_heldout", "regionxregion"]
SHORT = {"trainxtrain": "tt", "trainxgoal_heldout": "tg", "trainxregion": "tr",
         "regionxtrain": "rt", "regionxgoal_heldout": "rg", "regionxregion": "rr"}


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    argv = d["argv"]
    fin = d["final"]
    row = {"run": os.path.basename(os.path.dirname(path)),
           "mode": argv["mode"], "mm": argv["movement_mode"][:4],
           "L": argv["num_layers"], "H": argv["hidden_size"],
           "act": argv["nonlinearity"], "lr": argv["lr"], "sched": argv["lr_schedule"],
           "wd": argv["weight_decay"], "seed": argv["seed"], "upd": argv["n_updates"],
           "params": d.get("params")}
    row["train_tt"] = fin["train"]["trainxtrain"]["model"]["metric"]
    row["same_tt"] = fin["same"]["trainxtrain"]["model"]["metric"]
    for c in CELLS:
        r = fin["heldout"].get(c)
        row[f"ho_{SHORT[c]}"] = None if r is None else r["model"]["metric"]
        row[f"nn_{SHORT[c]}"] = None if r is None else r["nn"]["metric"]
    # Best-by-history on heldout tt (the selection rule, plan sec 3.4).
    hist = d.get("history", [])
    if hist:
        key = "trainxtrain"
        mm = argv["movement_mode"]
        vals = [(h["tables"]["heldout"][key]["model"]["metric"], h["update"]) for h in hist
                if "heldout" in h["tables"] and key in h["tables"]["heldout"]]
        best = min(vals) if mm == "continuous" else max(vals)
        row["best_ho_tt"], row["best_upd"] = best
    return row


def fmt(v, mm):
    if v is None:
        return "   -  "
    return f"{v:6.2f}" if mm == "cont" else f"{v:6.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="a1_")
    p.add_argument("--root", default=os.path.join(os.environ.get("CLS_RUNS", "/orcd/pool/003/jackking/cls_runs"), "goal_pairs"))
    args = p.parse_args()
    paths = sorted(glob.glob(os.path.join(args.root, f"{args.prefix}*", "final_tables.json")))
    rows = [load(x) for x in paths]
    if not rows:
        print(f"no finished runs under {args.root}/{args.prefix}*")
        return
    rows.sort(key=lambda r: (r["mm"], r["L"], r["H"], r["act"], r["sched"], r["wd"], r["seed"]))
    hdr = (f"{'run':32s} {'mm':4s} {'L':>2s} {'H':>5s} {'act':4s} {'sch':6s} {'wd':>6s} s | "
           f"{'tr_tt':>6s} | {'ho_tt':>6s} {'ho_tg':>6s} {'ho_tr':>6s} {'ho_rt':>6s} {'ho_rg':>6s} {'ho_rr':>6s} | "
           f"{'nn_tt':>6s} {'nn_rr':>6s} | best@")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mm = r["mm"]
        print(f"{r['run'][:32]:32s} {mm:4s} {r['L']:2d} {r['H']:5d} {r['act']:4s} {r['sched']:6s} {r['wd']:6.0e} {r['seed']} | "
              f"{fmt(r['train_tt'], mm)} | "
              + " ".join(fmt(r[f"ho_{SHORT[c]}"], mm) for c in CELLS)
              + f" | {fmt(r['nn_tt'], mm)} {fmt(r['nn_rr'], mm)} | "
              + (f"{fmt(r.get('best_ho_tt'), mm)}@{r.get('best_upd')}" if 'best_upd' in r else ""))
    out = os.path.join(args.root, f"{args.prefix}results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
