"""Figures and summary tables from run_analysis.py outputs.

    python -m analysis.phase_decoding.figures --results RESULTS_DIR

Writes, into RESULTS_DIR:
  fig_A_<model>.pdf/.png   definition A: CCGP vs training arenas, parallelism
  fig_B_switch.pdf/.png    definition B: switch-aligned decoder output, store vs sham
  summary.md               every number the figures show, plus the controls
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Figures open in Illustrator: core Helvetica, no mathtext.
plt.rcParams.update({
    "pdf.use14corefonts": True, "font.family": "sans-serif",
    # The PDF names Helvetica; the PNG renders with URW's clone.
    "font.sans-serif": ["Helvetica", "Nimbus Sans"], "axes.unicode_minus": False,
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
# Categorical slots 1-3 of the reference palette, fixed order; grey for nulls.
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#3b3b38"
SERIES = [  # (label, run suffix, feature, colour, marker)
    ("RNN state h", "", "h", BLUE, "o"),
    ("policy input x", "", "x", ORANGE, "s"),
    ("h, random weights", "_random", "h", AQUA, "^"),
]
SELECTION = "matched"


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def fig_definition_a(res: dict, model: str, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.4),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (label, suf, feat, col, mk) in enumerate(SERIES):
        m = res[model + suf]["by_condition"]["all"][SELECTION][feat]
        ks = sorted(int(k) for k in m["ccgp"])
        mean = [m["ccgp"][str(k)]["mean"] for k in ks]
        sd = [m["ccgp"][str(k)]["sd"] for k in ks]
        q95 = [m["ccgp"][str(k)]["null_q95"] for k in ks]
        ax1.errorbar(ks, mean, yerr=sd, color=col, marker=mk, ms=4, lw=1.5,
                     capsize=0, label=label)
        ax1.plot(ks, q95, color=col, lw=0.8, ls=":")
        p = m["parallelism"]
        ax2.bar(i, p["ps"], width=0.6, color=col)
        ax2.plot([i - 0.3, i + 0.3], [p["null_q95"]] * 2, color=INK, lw=1)
    ax1.set_xscale("log")
    ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax1.set_xticks(ks, [str(k) for k in ks])
    ax1.axhline(0.5, color=GREY, lw=0.6)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xlabel("training arenas (held-out test arenas)")
    ax1.set_ylabel("balanced accuracy")
    ax1.set_title("cross-arena decoding of phase; dotted = geometric null 95%", fontsize=8)
    fig.legend(*ax1.get_legend_handles_labels(), frameon=False, fontsize=7,
               loc="lower center", ncol=3)
    ax2.set_xticks(range(3), ["h", "x", "h random"])
    ax2.axhline(0, color=GREY, lw=0.6)
    ax2.set_ylabel("parallelism score")
    ax2.set_title("mean pairwise cosine; line = shuffle 95%", fontsize=8)
    fig.suptitle(f"{model}: explore vs exploit set by memory content "
                 f"(matched on time and distance to goal)", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, out, f"fig_A_{model}")


def fig_definition_b(res: dict, out: Path) -> None:
    tr = res["t3_B"]["switch_traces"]
    rel = np.asarray(tr["rel"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=False)
    for ax, feat, title in zip(axes, ("h", "x"), ("decoder on RNN state h",
                                                  "decoder on policy input x")):
        for cond, col, ls in (("store", BLUE, "-"), ("sham", GREY, "--")):
            mu = np.asarray(tr[f"{feat}_{cond}_mean"], float)
            se = np.asarray(tr[f"{feat}_{cond}_sem"], float)
            ax.plot(rel, mu, color=col, lw=1.5, ls=ls,
                    label=f"{cond} (oracle write)" if cond == "store" else "sham (teleport, no write)")
            ax.fill_between(rel, mu - se, mu + se, color=col, alpha=0.2, lw=0)
        ax.axvline(0, color=INK, lw=0.6)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xlabel("steps from first arrival")
        ax.set_ylabel("decoder output (logit, >0 = post-store)")
        ax.set_title(title, fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7,
               loc="lower center", ncol=2)
    fig.suptitle("task3r: the agent's own search-to-navigate switch, held-out arenas "
                 "(mean +- s.e.m. over trials)", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, out, "fig_B_switch")


def _row(name, cond, sel, feat, m):
    f = m[feat]
    ks = sorted(f["ccgp"], key=int)
    cc = " ".join(f"{k}:{f['ccgp'][k]['mean']:.2f}/{f['ccgp'][k]['null_q95']:.2f}" for k in ks)
    p = f["parallelism"]
    return (f"| {name} | {cond} | {sel} | {feat} | {m['n_arenas']} | {m['rows_per_phase'][0]} | "
            f"{p['ps']:.2f} ({p['null_q95']:.2f}) | {cc} | "
            f"{f['within_arena']['mean']:.2f} | "
            f"{list(m['behaviour_only'].values())[0]['mean']:.2f} |")


def summary(res: dict, out: Path) -> None:
    lines = ["# Abstraction metrics summary", "",
             "PS: parallelism (shuffle 95%). CCGP k:mean/geometric-null-95%. "
             "Within: trial-held-out within-arena accuracy. Behav: phase decoded "
             "from behaviour alone (80% of arenas).", "",
             "| run | condition | selection | feature | arenas | rows/phase | PS | CCGP | within | behav |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        for cond, sels in r["by_condition"].items():
            for sel, m in sels.items():
                for feat in ("h", "x"):
                    lines.append(_row(name, cond, sel, feat, m))
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    out = Path(a.results)
    res = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json"))
           if p.stem != "config"}
    summary(res, out)
    for model in ("d0_A", "t3_A"):
        if model in res and model + "_random" in res:
            fig_definition_a(res, model, out)
    if "t3_B" in res:
        fig_definition_b(res, out)


if __name__ == "__main__":
    main()
