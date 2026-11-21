"""One table per probe set: navigation probe (K = 5, one recall step) beside field and kernel stats.

    python -m analysis.field_kernel.summary a [b ...]

Reads ``$RES/probe_<set>/t*/*.json`` (``analysis.hopfield_probe.run``) and
``$RES/measure/set_<set>.json`` (``analysis.field_kernel.measure``), joined by line order of
``analysis/field_kernel/sets/<set>.txt``.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from analysis.hopfield_probe.arm_summary import row

RES = "/orcd/pool/003/jackking/cls_runs/results/field_kernel"
HERE = os.path.dirname(os.path.abspath(__file__))
LAMBDA = 12.0


def load(set_name: str) -> list[dict]:
    lines = [l.strip() for l in open(f"{HERE}/sets/{set_name}.txt") if l.strip()]
    mfile = f"{RES}/measure/set_{set_name}.json"
    meas = json.load(open(mfile)) if os.path.exists(mfile) else []
    out = []
    for i, l in enumerate(lines):
        spec, label = l.split("|")[:2]
        d = {"set": set_name, "i": i, "spec": spec, "label": label}
        fs = [f for f in glob.glob(f"{RES}/probe_{set_name}/t{i}/*.json")
              if not f.endswith("manifest.json")]
        if fs:
            d["probe"] = row(json.load(open(fs[0])))
        if i < len(meas):
            d["meas"] = meas[i]
        out.append(d)
    return out


def fmt(v, p=2, w=7):
    return f"{'-':>{w}s}" if v is None else f"{v:{w}.{p}f}"


def table(rows: list[dict]) -> None:
    print(f"{'code':<26s}{'W_A':>6s}{'W/lam':>7s}{'f/act40':>8s}{'FW05':>7s}{'K(16)':>7s}"
          f"{'farmax':>7s} |{'err':>7s}{'exact':>7s}{'basin':>7s}{'reach':>7s}{'s15':>7s}"
          f"  dead@K 1/3/5/10/20")
    for d in rows:
        m, p = d.get("meas"), d.get("probe")
        s = f"{d['label']:<26s}"
        if m:
            kp = m["kernel"]["pearson"]
            s += (fmt(m["loop"]["W_A"], 1, 6) + fmt(m["loop"]["W_A"] / LAMBDA)
                  + fmt(m["arena"]["40"]["fields_per_active"], 2, 8) + fmt(kp["fw05"], 1)
                  + fmt(kp["at"]["16"]) + fmt(kp["far_max"], 3))
        else:
            s += " " * 42
        s += " |"
        if p:
            s += (fmt(p["err"]) + fmt(p["exact"], 3) + fmt(p["basin"], 1) + fmt(p["cont"], 3)
                  + fmt(p["s15"], 3) + "  "
                  + " ".join(fmt(p["dead"].get(k), 2, 4) for k in ("1", "3", "5", "10", "20")))
        print(s)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["a"]:
        print(f"\n=== set {name} ===")
        table(load(name))
