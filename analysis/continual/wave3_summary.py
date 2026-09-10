"""Aggregate Wave 3 -- the parameter-isolation family -- against the suite.

Separate from `wave1_summary` for one reason that is not tidiness: every arm
here is handed an **oracle task id**, at training time and at evaluation time.
Waves 1 and 2 are not, and neither is the Hopfield store. Printing all of them
in one sorted table would rank a method that is told which environment it is in
against methods that have to work it out, which is not a comparison -- so this
table carries a `task id` column and the reading below says it again.

Five questions, in order:

  J   Does a task-conditioned hypernetwork retain, and is it the
      parameterisation or the regulariser that does it (beta=0 vs beta>0)?
  K   With the pretrained weights frozen, is there anything left to forget?
  L   Does any of it survive without the warm start?
  M   How much does perfect head isolation buy on its own?
  N   Does isolating *inside* the trunk beat isolating at the readout?
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


def _mean(xs):
    xs = [x for x in xs if x is not None
          and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _sem(xs):
    xs = [x for x in xs if x is not None
          and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def _load_group(d: str, pattern: str) -> dict[str, list[dict]]:
    """-> {config_label: [per-seed summary, with the arch fields attached]}"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, pattern))):
        try:
            hist = json.load(open(p))
        except Exception as e:
            print(f"  ! unreadable {os.path.basename(p)}: {e}")
            continue
        if not hist.get("blocks"):
            continue
        s = M.summarize(hist)
        meta = hist.get("metadata", {})
        detail = meta.get("arch_detail") or {}
        s["arch"] = meta.get("arch", "rnn")
        s["params"] = detail.get("trainable_params", float("nan"))
        s["needs_task_id"] = bool(
            (meta.get("method_detail") or {}).get("needs_task_id")
            or s["arch"] in ("hnet", "multihead", "xdg"))
        label = re.sub(r"_s\d+$", "",
                       os.path.splitext(os.path.basename(p))[0])
        groups[label].append(s)
    return dict(groups)


def _table(title: str, groups: dict[str, list[dict]],
           sort_key: str = "retained") -> list[tuple[str, dict]]:
    print(f"\n{title}")
    if not groups:
        print("  (nothing found)")
        return []
    hdr = (f"  {'config':<32} {'n':>3} {'retained':>16} {'current':>16} "
           f"{'forget':>8} {'params':>9} {'bytes':>10} {'task id':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for label, rs in groups.items():
        row = {k: _mean([r[k] for r in rs]) for k in
               ("retained", "current_env", "forgetting", "stability_gap",
                "episodes_to_criterion", "state_bytes", "params")}
        row["retained_sem"] = _sem([r["retained"] for r in rs])
        row["current_sem"] = _sem([r["current_env"] for r in rs])
        row["needs_task_id"] = any(r["needs_task_id"] for r in rs)
        row["n"] = len(rs)
        rows.append((label, row))
    rows.sort(key=lambda kv: (-kv[1][sort_key]
                              if not math.isnan(kv[1][sort_key]) else 0))
    for label, r in rows:
        print(f"  {label:<32} {r['n']:>3} "
              f"{r['retained']:>9.4f} +/-{r['retained_sem']:<5.4f} "
              f"{r['current_env']:>9.4f} +/-{r['current_sem']:<5.4f} "
              f"{r['forgetting']:>8.3f} {r['params']:>9.0f} "
              f"{r['state_bytes']/1e6:>8.2f}MB "
              f"{'YES' if r['needs_task_id'] else '-':>8}")
    return rows


#: Arms from earlier waves that Wave 3 has to be read against. Not re-run --
#: they are in the same directory, at the same configuration, on purpose.
REFERENCES = (
    ("R_*.json", "naive SGD, pretrained (the floor)"),
    ("I_erhi_rb32_*.json", "best replay in the suite (ER, ratio 32)"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    args = p.parse_args()
    d = args.dir

    print("=" * 118)
    print("WAVE 3 SUMMARY   parameter isolation   "
          "docs/CONTINUAL_CONTROLS_PLAN.md section 4.3")
    print("=" * 118)

    j = _table("J. Task-conditioned hypernetwork, learned base "
               "(b0 = no regulariser)", _load_group(d, "J_hnet_*.json"))
    k = _table("K. Hypernetwork with the pretrained base frozen",
               _load_group(d, "K_hnetfrz_*.json"))
    ll = _table("L. Hypernetwork from scratch, and its from-scratch control",
                {**_load_group(d, "L_hnetscratch_*.json"),
                 **_load_group(d, "L0_scratch_*.json")})
    m = _table("M. Multi-head, oracle task id", _load_group(d, "M_multihead_*.json"))
    n = _table("N. XdG (context-dependent gating), alone and with SI",
               {**_load_group(d, "N_xdg_*.json"),
                **_load_group(d, "N2_xdgsi_*.json")})

    refs = {}
    for pattern, desc in REFERENCES:
        g = _load_group(d, pattern)
        if g:
            label, rows = next(iter(g.items()))
            refs[desc] = {
                "retained": _mean([r["retained"] for r in rows]),
                "current": _mean([r["current_env"] for r in rows]),
                "label": label, "n": len(rows)}

    print("\n" + "-" * 118)
    print("READING")
    for desc, r in refs.items():
        print(f"  Reference -- {desc}: {r['label']} -> "
              f"retained {r['retained']:.4f}, current {r['current']:.4f} "
              f"(n={r['n']})")
    floor = next((r["retained"] for dsc, r in refs.items() if "floor" in dsc),
                 float("nan"))
    for name, rows in (("HNET (learned base)", j), ("HNET (frozen base)", k),
                       ("HNET / control from scratch", ll),
                       ("Multi-head", m), ("XdG", n)):
        if not rows:
            continue
        label, top = rows[0]
        print(f"  Best {name}: {label} -> retained {top['retained']:.4f} "
              f"(+/-{top['retained_sem']:.4f}), current {top['current_env']:.4f}, "
              f"{top['params']:.0f} params")
        if floor == floor:
            print(f"    vs the naive floor: {top['retained'] - floor:+.4f} retained")
    print("-" * 118)
    print("  Every arm above is given an ORACLE TASK ID at training and at")
    print("  evaluation time. Waves 1 and 2 are not, and the Hopfield store is")
    print("  not -- it acquires an env in 1 episode and 0 gradient steps with no")
    print("  task label and no boundary. These are upper bounds on their family.")
    print("-" * 118)
    print("  `current` is the plasticity check, not a footnote: a policy that")
    print("  stops learning retains perfectly and is worth nothing. Any arm whose")
    print("  current is far below the floor's is failing at the task, however")
    print("  well it scores on retention.")
    print("-" * 118)


if __name__ == "__main__":
    main()
