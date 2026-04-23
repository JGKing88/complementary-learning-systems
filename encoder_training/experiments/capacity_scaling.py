"""Capacity-scaling experiment: sweep Npatts for three hippocampal scaffolds.

Compares, as a function of the number of stored patterns:

- ``encoder`` scaffold tapped at ``hidden_1``  (best encoder tap we found)
- ``encoder`` scaffold tapped at ``hidden_4`` (= ``last_hidden``)
- ``random``  scaffold with matched ``Np`` (canonical VectorHash)

For each ``n_envs`` value, we place that many non-overlapping env patches
inside an ``Npos x Npos`` world, fully explore them, build each scaffold on
the resulting (sbook, pbook, gbook) triple, and evaluate grid-recovery
accuracy and observation bit-error under several obs-flip probabilities.

Saves:
    - PNG plot   : one subplot per pflip, three curves
    - JSON       : raw numbers (for reuse / replotting)

Does not modify any existing code; imports reusable pieces from
``encoder_scaffold``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

from encoder_training.experiments.encoder_scaffold import (
    ScaffoldCommon,
    EncoderScaffold,
    RandomProjectionScaffold,
    evaluate,
    _build_common,
    DEFAULT_CKPT,
)
from encoder_training.train import load_encoder


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _make_common(seed: int, n_envs: int, env_size: int, Npos: int, Ns: int,
                 lambdas: list[int]) -> ScaffoldCommon:
    """Thin wrapper so we can reuse `_build_common` with a fresh args namespace."""
    ns = types.SimpleNamespace(
        seed=seed, n_envs=n_envs, env_size=env_size, Npos=Npos,
        Ns=Ns, lambdas=lambdas,
    )
    return _build_common(ns, lambdas)


def _evaluate_all(
    com: ScaffoldCommon,
    encoder: torch.nn.Module | None,
    fwhm_ratio: float,
    device: str,
    Np_random: int,
    thresh: float,
    c: float,
    pflips: list[float],
    seed: int,
    wgp_rule: str = "pseudo",
) -> dict[str, dict[float, dict]]:
    """Build 3 scaffolds on `com` and evaluate each at every pflip.

    All three scaffolds use the same ``wgp_rule`` so encoder-vs-random
    comparisons hold the p->g training rule fixed.

    Returns: {scaffold_name: {pflip: metrics_dict}}
    """
    results: dict[str, dict[float, dict]] = {}

    # --- encoder hidden_1 ---
    print("    [build] encoder hidden_1")
    enc1 = EncoderScaffold(
        com, encoder, fwhm_ratio, device, tap="hidden_1", wgp_rule=wgp_rule)
    results["hidden_1"] = {}

    # --- encoder hidden_4 / last_hidden ---
    print("    [build] encoder hidden_4")
    enc4 = EncoderScaffold(
        com, encoder, fwhm_ratio, device, tap="hidden_4", wgp_rule=wgp_rule)
    results["hidden_4"] = {}

    # --- random projection ---
    print(f"    [build] random projection (Np={Np_random}, thresh={thresh}, c={c}, "
          f"wgp={wgp_rule})")
    rnd = RandomProjectionScaffold(
        com, Np=Np_random, thresh=thresh, c=c,
        rng=np.random.RandomState(seed),
        wgp_rule=wgp_rule,
    )
    results["random"] = {}

    # --- evaluate ---
    for pf in pflips:
        rng_seed = seed + 1000 + int(pf * 1000)
        rng1 = np.random.RandomState(rng_seed)
        rng4 = np.random.RandomState(rng_seed)
        rngR = np.random.RandomState(rng_seed)
        results["hidden_1"][pf] = evaluate(enc1, com, pflip=pf, rng=rng1)
        results["hidden_4"][pf] = evaluate(enc4, com, pflip=pf, rng=rng4)
        results["random"][pf]   = evaluate(rnd,  com, pflip=pf, rng=rngR)

    return results


# Known metrics: (y_label, fig_title, transform(raw -> percentage, higher=better))
_METRICS = {
    "g_accuracy": (
        "grid recall accuracy (%)",
        "Capacity scaling: grid-recovery accuracy vs stored patterns",
        lambda v: v * 100.0,
    ),
    "s_bit_err": (
        "observation recall accuracy (%)",
        "Capacity scaling: sensory reconstruction accuracy vs stored patterns",
        lambda v: (1.0 - v) * 100.0,
    ),
}


def _plot(
    n_envs_list: list[int],
    npatts_list: list[int],
    pflips: list[float],
    all_results: dict[int, dict[str, dict[float, dict]]],
    out_path: str,
    metric: str = "g_accuracy",
    title_suffix: str = "",
) -> None:
    """Make one subplot per pflip, each showing three curves over Npatts.

    For ``s_bit_err`` we plot ``(1 - bit_err) * 100`` as "observation recall
    accuracy (%)" so both panels share an identical y-scale where higher is
    better.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}; known: {list(_METRICS)}")
    ylabel, title, transform = _METRICS[metric]

    n_panels = len(pflips)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels, 4.0), sharey=True, squeeze=False,
    )
    axes = axes[0]

    styles = {
        "hidden_1": dict(color="C0", marker="o", label="encoder hidden_1"),
        "hidden_4": dict(color="C1", marker="s", label="encoder hidden_4 (last)"),
        "random":   dict(color="C2", marker="^", label="random projection"),
    }

    for ax, pf in zip(axes, pflips):
        for name, style in styles.items():
            ys = [transform(all_results[n][name][pf][metric]) for n in n_envs_list]
            ax.plot(npatts_list, ys, linewidth=2, markersize=6, **style)
        ax.set_xscale("log")
        ax.set_xlabel("Npatts (stored patterns)")
        ax.set_title(f"pflip = {pf:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-3, 103)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="lower left", framealpha=0.9)
    full_title = f"{title}{title_suffix}" if title_suffix else title
    fig.suptitle(full_title, fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--Npos", type=int, default=150,
                   help="World size.  Larger = can fit more non-overlapping envs.")
    p.add_argument("--env_size", type=int, default=20)
    p.add_argument("--Ns", type=int, default=1600)
    p.add_argument("--n_envs_list", type=str,
                   default="1,2,5,10,15,20,25",
                   help="Comma-separated n_envs values.  Must all fit in Npos^2 "
                        "with non-overlapping env_size patches.")
    p.add_argument("--Np", type=int, default=1024,
                   help="Np for the random projection (matched to encoder hidden dim).")
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--c", type=float, default=0.5)
    p.add_argument("--wgp_rule", type=str, default="pseudo",
                   choices=["hebbian", "pseudo"],
                   help="Wgp training rule used for ALL scaffolds in the "
                        "sweep (encoder hidden_1, encoder hidden_4, and "
                        "random). 'hebbian' matches the canonical VectorHash "
                        "setup; 'pseudo' uses G @ pinv(P). Default: pseudo.")
    p.add_argument("--pflips", type=str, default="0.0,0.05,0.1,0.2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--fwhm_ratio", type=float, default=None)
    p.add_argument("--out_dir", type=str,
                   default="/home/jackking/cls/encoder_training/experiments/plots")
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix for the output filenames.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_envs_list = [int(x) for x in args.n_envs_list.split(",") if x.strip()]
    pflips = [float(x) for x in args.pflips.split(",") if x.strip()]

    # Load encoder once.
    print(f"[1/3] Loading encoder: {args.ckpt}")
    encoder, ckpt = load_encoder(args.ckpt, device=device)
    encoder.eval()
    enc_lambdas = list(ckpt["model_config"]["lambdas"])
    train_cfg = ckpt.get("train_config") or {}
    fwhm_ratio = (args.fwhm_ratio if args.fwhm_ratio is not None
                  else float(train_cfg.get("fwhm_ratio", 0.25)))
    out_dim = int(ckpt["model_config"]["out_dim"])
    hidden_dim = int(ckpt["model_config"].get("hidden_dim", 1024))
    print(f"  lambdas={enc_lambdas}  out_dim={out_dim}  "
          f"hidden_dim={hidden_dim}  fwhm_ratio={fwhm_ratio}")
    print(f"  sweep n_envs = {n_envs_list}  (Npatts = n_envs * {args.env_size**2})")
    print(f"  pflips = {pflips}  Np_random = {args.Np}")

    # ----- sweep -------------------------------------------------------------
    all_results: dict[int, dict[str, dict[float, dict]]] = {}
    npatts_list: list[int] = []
    t_start = time.time()
    for i, n_envs in enumerate(n_envs_list):
        print(f"[2/3] ({i+1}/{len(n_envs_list)}) n_envs={n_envs} "
              f"Npatts={n_envs * args.env_size**2}")
        com = _make_common(
            seed=args.seed, n_envs=n_envs, env_size=args.env_size,
            Npos=args.Npos, Ns=args.Ns, lambdas=enc_lambdas,
        )
        npatts_list.append(int(com.locs.shape[0]))
        all_results[n_envs] = _evaluate_all(
            com=com, encoder=encoder, fwhm_ratio=fwhm_ratio, device=device,
            Np_random=args.Np, thresh=args.thresh, c=args.c,
            pflips=pflips, seed=args.seed, wgp_rule=args.wgp_rule,
        )
        # Short per-n_envs summary.
        for name in ("hidden_1", "hidden_4", "random"):
            row = all_results[n_envs][name]
            cells = "  ".join(
                f"pf={pf:.2f}:{row[pf]['g_accuracy']*100:5.1f}%" for pf in pflips
            )
            print(f"    {name:>9}  {cells}")
    print(f"  sweep done in {time.time()-t_start:.1f}s")

    # ----- save --------------------------------------------------------------
    print("[3/3] Saving outputs")
    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rule_tag = f"_{args.wgp_rule}"
    suffix = f"_{args.tag}" if args.tag else ""
    png_g_path = os.path.join(
        args.out_dir, f"capacity_scaling_gridacc_{ts}{rule_tag}{suffix}.png")
    png_s_path = os.path.join(
        args.out_dir, f"capacity_scaling_sensacc_{ts}{rule_tag}{suffix}.png")
    json_path = os.path.join(
        args.out_dir, f"capacity_scaling_{ts}{rule_tag}{suffix}.json")

    # JSON needs string keys for pflips.
    serialisable = {
        str(n): {
            name: {f"{pf:.3f}": m for pf, m in perpf.items()}
            for name, perpf in by_name.items()
        }
        for n, by_name in all_results.items()
    }
    json.dump(
        {
            "args": vars(args),
            "n_envs_list": n_envs_list,
            "npatts_list": npatts_list,
            "pflips": pflips,
            "fwhm_ratio": fwhm_ratio,
            "out_dim": out_dim,
            "hidden_dim": hidden_dim,
            "ckpt": args.ckpt,
            "results": serialisable,
        },
        open(json_path, "w"),
        indent=2,
    )
    print(f"  saved json:  {json_path}")

    title_suffix = f"  (wgp={args.wgp_rule})"
    _plot(
        n_envs_list=n_envs_list, npatts_list=npatts_list, pflips=pflips,
        all_results=all_results, out_path=png_g_path, metric="g_accuracy",
        title_suffix=title_suffix,
    )
    _plot(
        n_envs_list=n_envs_list, npatts_list=npatts_list, pflips=pflips,
        all_results=all_results, out_path=png_s_path, metric="s_bit_err",
        title_suffix=title_suffix,
    )
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
