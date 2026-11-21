"""Markdown tables for Sec 20 (the grid wave network).

    python -m analysis.hopfield_probe.gwn_summary [OUT]

OUT defaults to ``$CLS_RESULTS/hopfield_probe/gwn``. Reads the step JSON
written by ``gwn_check`` / ``gwn_l1_learn`` and the probe runs
(``OUT/probe/t*/*.json``, K = 5 s = 1 via ``arm_summary.row``), with the ideal
r = 16 probe rows (``ideal_encoder/probe/t12``-``t15``) as the reference.
Missing pieces are skipped.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from analysis.hopfield_probe.arm_summary import row
from cls_paths import results_dir

REGIONS = ("whole", "corner", "centre", "opposite")


def _j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _k(m, s="far"):
    f = m.get(s) or m.get("unvisited")
    rh = f["r_half"]
    return (f"{f['C1']:.3f} / {rh:.1f} / {f['far_sd']:.3f} / {f['alias_max']:.3f}"
            if rh is not None else f"{f['C1']:.3f} / - / {f['far_sd']:.3f} / {f['alias_max']:.3f}")


def kernel_rows(title, d, keys, extra=None):
    print(f"\n#### {title}\n")
    hdr = "| condition | units | kept | strong | n_eff | far: C(1) / half-width / far sd / alias max |"
    cols = 6
    if extra:
        hdr = hdr[:-1] + " | " + " | ".join(e[0] for e in extra) + " |"
        cols += len(extra)
    print(hdr)
    print("|" + "---|" * cols)
    for k, lab in keys:
        r = d.get(k)
        if r is None:
            continue
        line = (f"| {lab} | {r.get('units', '')} | {r.get('kept', '')} | {r.get('strong', '')} | "
                f"{r.get('n_eff', 0):.0f} | {_k(r)} |")
        if extra:
            line = line[:-1] + " | " + " | ".join(e[1](r) for e in extra) + " |"
        print(line)


def main(out):
    s0 = _j(f"{out}/step0.json")
    if s0:
        print(f"ideal r = 16 (far): {_k(s0['ideal'])}")
        kernel_rows("Step 0: exact network", s0,
                    [("units_25k", "25k units"), ("units_100k", "100k"), ("units_400k", "400k"),
                     ("units_all", "all 1716² triples"), ("rule_cut", "400k, cut gain"),
                     ("rule_rational", "400k, rational gain"), ("rule_linear", "400k, linear gain"),
                     ("coverage_0.01", "400k, steps from 1% coverage"),
                     ("coverage_0.2", "400k, steps from 20% coverage")],
                    extra=[("slowness max abs error", lambda r: f"{r['slowness_err']['max_abs']:.1e}"
                            if "slowness_err" in r else "")])
    s1 = _j(f"{out}/step1.json")
    if s1:
        keys = [(f"{n}_fwhm{w}", f"{lab}, bump {w}") for w in ("0.25", "0.1")
                for n, lab in (("pca", "PCA"), ("move1k", "movement 1k steps"),
                               ("move10k", "movement 10k"), ("move100k", "movement 100k"),
                               ("online100k", "online rule 100k"))]
        kernel_rows("Step 1: learned L1 in the 400k network", s1, keys,
                    extra=[("purity mean", lambda r: f"{r['purity_mean']:.3f}"),
                           ("pairs > 0.99", lambda r: f"{r['frac_pure_0.99']:.2f}")])
    s2 = _j(f"{out}/step2.json")
    if s2:
        kernel_rows("Step 2: imperfect multiplication (400k)", s2,
                    [(f"coef_{s}", f"coefficient error sd {s}") for s in ("0.01", "0.05", "0.1", "0.3")]
                    + [(f"quarter_{p}", f"quarter squares, p = {p}") for p in ("1.5", "1.8", "2.2", "2.5")])
        print("\n#### Step 2: grid noise (gains learned on clean codes)\n")
        print("| noise | level | network, bump 0.25 | ideal, bump 0.25 | network, bump 0.10 | ideal, bump 0.10 |")
        print("|---|---|---|---|---|---|")
        for kind, levels in (("activity", ("0.001", "0.01", "0.05")), ("phase", ("0.05", "0.1", "0.2", "0.4"))):
            for lv in levels:
                cells = []
                for w in ("0.25", "0.1"):
                    for e in ("gwn", "ideal"):
                        r = s2.get(f"noise_{e}_fwhm{w}_{kind}_{lv}")
                        cells.append(f"{r['selfsim']:.3f} / {r['err_mean']:.1f} / {r['frac_gt5']:.2f}" if r else "-")
                print(f"| {kind} | {lv} | " + " | ".join([cells[0], cells[1], cells[2], cells[3]]) + " |")
        print("\ncells: self-similarity / mean decoded error (cells) / share of errors > 5 cells")
    s2b = _j(f"{out}/step2b.json")
    if s2b:
        print("\n#### Step 2b: gains learned under cell noise\n")
        print("| bump | noise | rule | strong | far: C(1) / half-width / far sd / alias max | self-sim / err / > 5 |")
        print("|---|---|---|---|---|---|")
        for k, r in s2b.items():
            w, n, rule = k.split("_")
            t = r["noise_test"]
            print(f"| {w[4:]} | {n[5:]} | {rule} | {r['strong']} | {_k(r)} | "
                  f"{t['selfsim']:.3f} / {t['err_mean']:.1f} / {t['frac_gt5']:.2f} |")
    s3 = _j(f"{out}/step3.json")
    s3b = _j(f"{out}/step3b.json")
    if s3 or s3b:
        print("\n#### Step 3: cost\n")
        print("| design | neurons | product branches | strong | far: C(1) / half-width / far sd / alias max |")
        print("|---|---|---|---|---|")
        if s0:
            for k, lab in (("units_400k", "random triples"), ("units_all", "all triples")):
                r = s0[k]
                print(f"| {lab} | {r['units']} | {r['units']} | {r['strong']} | {_k(r)} |")
        if s3:
            for k, r in s3.items():
                if k.startswith("staged"):
                    print(f"| staged, pair gain > {k[10:]} | {r['neurons']} | {r['product_synapses']} | "
                          f"{r['strong']} | {_k(r)} |")
        if s3b:
            print(f"| branch design | {s3b['neurons']} | {s3b['branch_products']} | {s3b['strong']} | "
                  f"{_k(s3b)} |  (distinct n {s3b['distinct_n']}, max |n| {s3b['max_abs_n']})")
    # probe
    rows = {}
    ideal_dir = os.path.join(os.path.dirname(out.rstrip("/")), "ideal_encoder", "probe")
    for t in ("t12", "t13", "t14", "t15"):
        for f in glob.glob(f"{ideal_dir}/{t}/*.json"):
            if not f.endswith("manifest.json"):
                r = json.load(open(f)); lab = r["header"]["label"]
                enc, _, reg = lab.partition(" | ")
                rows[("ideal r=16", reg or "whole")] = row(r)
    for f in glob.glob(f"{out}/probe/t*/*.json"):
        if f.endswith("manifest.json"):
            continue
        r = json.load(open(f)); lab = r["header"]["label"]
        enc, _, reg = lab.partition(" | ")
        try:
            rows[(enc, reg or "whole")] = row(r)
        except KeyError:
            pass
    if rows:
        print("\n#### Hopfield probe, K = 5, one step (Hebbian)\n")
        print("| encoder | region | abs err | acc45 | exact | basin | reach | dead at K = 5 / 10 / 20 |")
        print("|---|---|---|---|---|---|---|---|")
        encs = sorted({e for e, _ in rows}, key=lambda s: (s != "ideal r=16", s))
        for e in encs:
            for reg in REGIONS:
                d = rows.get((e, reg))
                if d is None:
                    continue
                dk = " / ".join(f"{d['dead'].get(k):.2f}" if d["dead"].get(k) is not None else "-"
                                for k in ("5", "10", "20"))
                print(f"| {e} | {reg} | {d['err']:.2f} | {d['acc']:.3f} | {d['exact']:.3f} | "
                      f"{d['basin']:.1f} | {d['cont']:.3f} | {dk} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(results_dir(), "hopfield_probe", "gwn"))
