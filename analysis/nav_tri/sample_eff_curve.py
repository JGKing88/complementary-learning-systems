"""Training-eval series against cumulative SAMPLES, for several runs at once.

docs/EXPERIMENTS_SAMPLE_EFF.md. The question there is "how few samples reach
d0_base u725's quality", so the x-axis has to be samples, not updates: the
arms differ by 8-16x in trajectories per update, and a table at matched update
index would flatter the fat pool.

Two sample counts, both read from the log:

    episodes    exact everywhere. envs x batch per update.
    env_steps   REALIZED transitions. Runs from 2026-09-13 print
                `[navigate_uN] samples={...}` beside every eval; older logs
                (d0_base) do not, and for those the count is RECONSTRUCTED:
                explore rows always run the full rollout (goals off), exploit
                rows end on arrival, so per update
                    n_expl x T  +  n_expt x min(T, steps_at_u)
                with steps_at_u linearly interpolated from the nav eval's own
                mean_steps at d=0 (a mean over successes; while success < 1 the
                failures ran to T, so this UNDER-counts early -- flagged as
                `~` in the table). d0_base's reconstruction lands at 104.6M
                by u725 against the 103.5M measured from its rollout diag.

Window means, not point values: the training eval swings 30+ points between
consecutive evals (feedback_eval_point_threshold), so every quality column is
the mean over the last `--window` evals ending at that row.

Usage:
    python -m analysis.nav_tri.sample_eff_curve \\
        --log d0_base=/path/nav_p2_22133273.out \\
              se_b8_e10=/path/nav_p2_<job>.out \\
        [--window 4] [--every 100] [--json out.json]

`--at_target` prints, per run, the first eval at which the window mean clears
every bar in --bar (default: d0_base u725's own training-eval window), with
the samples spent by then.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from .training_curve import parse_log


def parse_header(path: str) -> dict:
    """envs, batch, steps from the launcher banner; T falls back to 200."""
    with open(path, errors="replace") as fh:
        head = fh.read(20000)
    m = re.search(r"rollout\s*:\s*(\d+) envs x (\d+) batch x (\d+) steps", head)
    if not m:
        raise ValueError(f"{path}: no rollout banner")
    envs, batch, steps = (int(g) for g in m.groups())
    m2 = re.search(r"empty_frac=([0-9.]+)", head)
    emp = float(m2.group(1)) if m2 else 0.5
    return {"envs": envs, "batch": batch, "steps": steps, "empty_frac": emp}


def parse_samples(path: str) -> dict[int, dict]:
    """{update: {'episodes': int, 'env_steps': int}} from the samples= lines."""
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.match(r"\s*\[navigate_u(\d+)\] samples=\{'episodes': (\d+), "
                         r"'env_steps': (\d+)\}", line)
            if m:
                out[int(m.group(1))] = {"episodes": int(m.group(2)),
                                        "env_steps": int(m.group(3))}
    return out


def reconstruct_samples(log: dict, hdr: dict) -> dict[int, dict]:
    """Episodes exactly; env-steps from the eval's own mean_steps (see doc)."""
    per_update = hdr["envs"] * hdr["batch"]
    n_emp = int(round(hdr["envs"] * hdr["empty_frac"]))
    n_pre = hdr["envs"] - n_emp
    T = hdr["steps"]
    us = sorted(log)
    # steps_at_u: nav d=0 mean_steps, 0 -> T (no successes = every row ran out)
    pts = []
    for u in us:
        nav = log[u]["nav"]
        ms = _f(nav.get(0, nav[min(nav)])["mean_steps"])
        pts.append(T if ms <= 0 else min(T, ms))
    out, cum_steps = {}, 0
    prev_u, prev_ms = 0, T
    for u, ms in zip(us, pts):
        for uu in range(prev_u + 1, u + 1):
            # linear interpolation between eval points
            f = (uu - prev_u) / max(u - prev_u, 1)
            s = prev_ms + f * (ms - prev_ms)
            cum_steps += n_emp * hdr["batch"] * T + n_pre * hdr["batch"] * s
        out[u] = {"episodes": per_update * u, "env_steps": int(cum_steps),
                  "reconstructed": True}
        prev_u, prev_ms = u, ms
    return out


def load_run(path: str) -> dict:
    log = parse_log(path)
    hdr = parse_header(path)
    samples = parse_samples(path)
    recon = not samples
    if recon:
        samples = reconstruct_samples(log, hdr)
    return {"log": log, "hdr": hdr, "samples": samples, "reconstructed": recon}


def _f(x) -> float:
    return float("nan") if x is None else float(x)


METRICS = (("succ0", "nav", 0, "success_rate"),
           ("succ10", "nav", 10, "success_rate"),
           ("steps0", "nav", 0, "mean_steps"),
           ("steps10", "nav", 10, "mean_steps"),
           ("swept0", "expl", 0, "swept_coverage"),
           ("swept10", "expl", 10, "swept_coverage"))


def table(run: dict, window: int) -> list[dict]:
    us = sorted(u for u in run["log"] if u in run["samples"])
    rows = []
    for i, u in enumerate(us):
        win = us[max(0, i - window + 1): i + 1]
        row = {"u": u, "n_win": len(win)}
        row.update(run["samples"][u])
        for name, kind, nd, key in METRICS:
            vals = [_f(run["log"][w][kind][nd][key]) for w in win
                    if nd in run["log"][w][kind]]
            row[name] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)
    return rows


# d0_base u725's own training-eval window (last 4 evals, u650-u725), which is
# the same instrument the arms are scored on. The held-out probe is the real
# bar; this is the screen that says which checkpoints to probe.
DEFAULT_BAR = {"succ0": 0.99, "succ10": 0.99, "steps0": 13.0, "steps10": 14.0,
               "swept0": 0.55, "swept10": 0.50}
HIGHER = {"succ0", "succ10", "swept0", "swept10"}


def meets(row: dict, bar: dict) -> bool:
    for k, v in bar.items():
        x = row.get(k, float("nan"))
        if not np.isfinite(x):
            return False
        if k in HIGHER and x < v:
            return False
        if k not in HIGHER and x > v:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", nargs="+", required=True,
                   help="label=path pairs")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--every", type=int, default=100,
                   help="print rows at this update stride (plus the last)")
    p.add_argument("--json", default=None)
    p.add_argument("--bar", nargs="*", default=None,
                   help="key=value overrides of the target window, e.g. "
                        "steps10=14 swept0=0.55")
    args = p.parse_args()

    bar = dict(DEFAULT_BAR)
    for kv in (args.bar or []):
        k, v = kv.split("=")
        bar[k] = float(v)

    runs = {}
    for spec in args.log:
        label, path = spec.split("=", 1)
        runs[label] = load_run(path)

    out = {}
    for label, run in runs.items():
        rows = table(run, args.window)
        out[label] = {"hdr": run["hdr"], "reconstructed": run["reconstructed"],
                      "rows": rows}
        h = run["hdr"]
        per_u = h["envs"] * h["batch"]
        print(f"\n=== {label}: {h['envs']} envs x {h['batch']} batch x "
              f"{h['steps']} steps = {per_u} episodes/update"
              f"{'  [env_steps RECONSTRUCTED]' if run['reconstructed'] else ''}"
              f"  window={args.window} evals ===")
        print(f"  {'u':>5} {'episodes':>9} {'env_steps':>11} | "
              f"{'succ0':>6} {'succ10':>6} {'steps0':>7} {'steps10':>7} "
              f"{'swept0':>6} {'swept10':>7} | ok")
        hit = None
        for i, r in enumerate(rows):
            ok = meets(r, bar) and r["n_win"] >= args.window
            if ok and hit is None:
                hit = r
            if r["u"] % args.every == 0 or i == len(rows) - 1 or (ok and hit is r):
                tag = "~" if run["reconstructed"] else " "
                print(f"  {r['u']:>5} {r['episodes']:>9} {tag}{r['env_steps']:>10} | "
                      f"{r['succ0']:6.3f} {r['succ10']:6.3f} {r['steps0']:7.2f} "
                      f"{r['steps10']:7.2f} {r['swept0']:6.3f} {r['swept10']:7.3f} | "
                      f"{'YES' if ok else ''}{'  <- first' if hit is r else ''}")
        if hit is not None:
            print(f"  FIRST window clearing the bar: u{hit['u']} at "
                  f"{hit['episodes']:,} episodes / {hit['env_steps']:,} env-steps")
            out[label]["first_hit"] = hit
        else:
            print("  bar not yet cleared by any full window")
    print(f"\nbar (window means): {bar}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
