"""The continual-learning scalars, computed from the history JSON.

The plotting module draws per-env curves, which is the right primary figure but
is not something you can put in a table or rank methods by. These are the
standard scalars (plan section 7.1), plus two the standard set leaves out and
this experiment specifically needs.

Definitions, with `p[i][j]` = performance on env `j` at the end of block `i`:

    A_N          mean_j p[N-1][j]                       final average
    forgetting   mean_{j<N-1} ( max_i p[i][j] - p[N-1][j] )   peak minus final
    BWT          mean_{j<N-1} ( p[N-1][j] - p[j][j] )   backward transfer
    FT_j         (AUC_j - AUC_j^ref) / (1 - AUC_j^ref)   forward transfer

and the two extra ones:

    stability_gap    the *transient* dip on env j in the first updates after
                     the stream moves to env j+1. The per-update trace has
                     recorded this at full resolution all along and nobody has
                     ever plotted it; De Lange et al. 2023 show it survives in
                     methods whose *final* forgetting looks fine.
    episodes_to_criterion
                     updates into block j before env j is solved. At
                     `batch_envs=1` one update is one episode, so this is the
                     axis on which the Hopfield store's advantage is largest --
                     1 against ~200 -- and no current figure shows it.

**On reading `reached`.** The continual histories evaluate with `n_trials=1`,
so every entry is a raw 0/1 and a single point is nearly meaningless. Every
estimator here therefore aggregates over a window and over seeds, and
`episodes_to_criterion` smooths before thresholding. Per the eval-point rule in
the project's notes, do not read a direction out of a handful of points.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Iterable


# ---------------------------------------------------------------------------
# history access
# ---------------------------------------------------------------------------

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _values(entry, key: str) -> list[float]:
    """A trace entry's metric as a list, whatever schema it is in.

    `merge_histories` writes a list-per-iteration when `num_full_iters > 1` and
    a bare scalar when it is 1. Both appear in the recorded histories, so every
    reader has to cope with both.
    """
    v = entry.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return [float(x) for x in v if x is not None]
    return [float(v)]


def per_env_series(hist: dict, key: str = "reached") -> dict[int, list[tuple[int, float]]]:
    """-> {env_idx: [(global_step, mean_over_iters), ...]} sorted by step."""
    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for step, _train_env, inner in hist.get("trace", []):
        for k, entry in inner.items():
            vals = _values(entry, key)
            if vals:
                out[int(k)].append((int(step), sum(vals) / len(vals)))
    for k in out:
        out[k].sort()
    return dict(out)


def _mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def performance_matrix(
    hist: dict, key: str = "reached", tail_frac: float = 0.2,
) -> dict[int, dict[int, float]]:
    """`p[i][j]`: env `j`'s score over the last `tail_frac` of block `i`.

    A tail window rather than the final point, because at `n_trials=1` the
    final point is a single coin flip.
    """
    blocks = hist.get("blocks") or []
    series = per_env_series(hist, key)
    p: dict[int, dict[int, float]] = {}
    for lo_all, hi, block_env in blocks:
        lo = lo_all + int((1.0 - tail_frac) * (hi - lo_all))
        row: dict[int, float] = {}
        for j, pts in series.items():
            vals = [v for s, v in pts if lo <= s <= hi]
            if vals:
                row[j] = _mean(vals)
        p[int(block_env)] = row
    return p


# ---------------------------------------------------------------------------
# the scalars
# ---------------------------------------------------------------------------

def final_average(p: dict[int, dict[int, float]]) -> float:
    if not p:
        return float("nan")
    last = max(p)
    return _mean(p[last].values())


def retained_average(p: dict[int, dict[int, float]]) -> float:
    """A_N restricted to the envs the stream has already left.

    The headline number for this experiment. `final_average` includes the env
    currently being trained, which every method solves and which therefore
    dilutes exactly the difference the figure is about.
    """
    if not p:
        return float("nan")
    last = max(p)
    return _mean(v for j, v in p[last].items() if j != last)


def current_env_score(p: dict[int, dict[int, float]]) -> float:
    """Plasticity: how well the last env was learned. The sanity check that a
    method has not simply frozen the network to score well on retention."""
    if not p:
        return float("nan")
    last = max(p)
    return p[last].get(last, float("nan"))


def forgetting(p: dict[int, dict[int, float]]) -> float:
    """mean_j ( max_i p[i][j] - p[N-1][j] ) over envs the stream has left."""
    if not p:
        return float("nan")
    last = max(p)
    per = []
    for j in p[last]:
        if j == last:
            continue
        peak = max((row[j] for row in p.values() if j in row), default=float("nan"))
        if not math.isnan(peak):
            per.append(max(peak - p[last][j], 0.0))
    return _mean(per)


def backward_transfer(p: dict[int, dict[int, float]]) -> float:
    """mean_j ( p[N-1][j] - p[j][j] ). Negative is forgetting; positive means
    later envs actually improved earlier ones."""
    if not p:
        return float("nan")
    last = max(p)
    per = [p[last][j] - p[j][j]
           for j in p[last] if j != last and j in p and j in p[j]]
    return _mean(per)


def auc_per_env(hist: dict, key: str = "reached") -> dict[int, float]:
    """Mean score on env `j` across its own training block -- how fast it was
    learned, not just whether it ended up learned. The input to forward transfer."""
    blocks = hist.get("blocks") or []
    series = per_env_series(hist, key)
    out: dict[int, float] = {}
    for lo, hi, j in blocks:
        pts = [v for s, v in series.get(int(j), []) if lo <= s <= hi]
        if pts:
            out[int(j)] = _mean(pts)
    return out


def forward_transfer(hist: dict, reference: dict, key: str = "reached") -> float:
    """(AUC - AUC_ref) / (1 - AUC_ref), averaged over envs.

    `reference` is a from-scratch history (T0.4). This is the only metric that
    scores pretraining, and it is the one the existing figures never had -- so
    whether pretraining helps at all has never actually been measured.
    """
    a, b = auc_per_env(hist, key), auc_per_env(reference, key)
    per = []
    for j, av in a.items():
        bv = b.get(j)
        if bv is None or bv >= 1.0:
            continue
        per.append((av - bv) / (1.0 - bv))
    return _mean(per)


def stability_gap(
    hist: dict, key: str = "reached", window: int = 25,
) -> float:
    """Mean transient drop on env `j` just after the stream moves to `j+1`.

    Measured as (score over the last `window` updates of block j) minus (score
    over the first `window` updates of block j+1), floored at 0. A method can
    have clean final forgetting and still tear a hole here, which is the whole
    point of measuring it separately.
    """
    blocks = hist.get("blocks") or []
    series = per_env_series(hist, key)
    per = []
    for b in range(len(blocks) - 1):
        lo_a, hi_a, env_a = blocks[b]
        lo_b, hi_b, _ = blocks[b + 1]
        pts = series.get(int(env_a), [])
        before = [v for s, v in pts if hi_a - window < s <= hi_a]
        after = [v for s, v in pts if lo_b <= s < lo_b + window]
        if before and after:
            per.append(max(_mean(before) - _mean(after), 0.0))
    return _mean(per)


def episodes_to_criterion(
    hist: dict, key: str = "reached", threshold: float = 0.9,
    smooth: int = 25,
) -> float:
    """Mean updates into a block before that env is solved.

    At `batch_envs=1` an update is an episode, so this reads directly as
    "episodes of experience consumed" -- the Hopfield store's value is 1.
    Returns the block length for envs never solved, which is a floor on the
    true value and is flagged by `episodes_to_criterion_censored`.
    """
    got, _ = _criterion(hist, key, threshold, smooth)
    return _mean(got)


def episodes_to_criterion_censored(
    hist: dict, key: str = "reached", threshold: float = 0.9,
    smooth: int = 25,
) -> float:
    """Fraction of envs that never reached the criterion, so the mean above is
    read as the censored quantity it is rather than as a real average."""
    _, censored = _criterion(hist, key, threshold, smooth)
    return censored


def _criterion(hist, key, threshold, smooth) -> tuple[list[float], float]:
    blocks = hist.get("blocks") or []
    series = per_env_series(hist, key)
    got: list[float] = []
    n_censored = 0
    for lo, hi, j in blocks:
        pts = [(s, v) for s, v in series.get(int(j), []) if lo <= s <= hi]
        if not pts:
            continue
        vals = [v for _, v in pts]
        hit = None
        for i in range(len(vals)):
            w = vals[max(0, i - smooth + 1): i + 1]
            if len(w) >= min(smooth, len(vals)) and _mean(w) >= threshold:
                hit = pts[i][0] - lo + 1
                break
        if hit is None:
            n_censored += 1
            hit = hi - lo + 1
        got.append(float(hit))
    frac = (n_censored / len(got)) if got else float("nan")
    return got, frac


# ---------------------------------------------------------------------------

def summarize(hist: dict, reference: dict | None = None,
              key: str = "reached") -> dict:
    """Every scalar for one history, plus the cost axes off its metadata."""
    p = performance_matrix(hist, key)
    md = hist.get("metadata", {}) or {}
    detail = md.get("method_detail") or {}
    out = {
        "run_name": md.get("run_name"),
        "method": md.get("method", "none"),
        "method_args": md.get("method_args"),
        "n_envs": md.get("n_envs"),
        "seeds": md.get("num_full_iters", 1),
        # performance
        "retained": retained_average(p),
        "current_env": current_env_score(p),
        "final_average": final_average(p),
        "forgetting": forgetting(p),
        "bwt": backward_transfer(p),
        "stability_gap": stability_gap(hist, key),
        "episodes_to_criterion": episodes_to_criterion(hist, key),
        "criterion_censored_frac": episodes_to_criterion_censored(hist, key),
        # cost axes (plan section 0.1)
        "state_bytes": detail.get("state_bytes", 0),
        "needs_task_boundaries": detail.get("needs_task_boundaries", False),
        "needs_task_id": detail.get("needs_task_id", False),
    }
    if reference is not None:
        out["forward_transfer"] = forward_transfer(hist, reference, key)
    return out


__all__ = [
    "load", "per_env_series", "performance_matrix",
    "final_average", "retained_average", "current_env_score",
    "forgetting", "backward_transfer", "auc_per_env", "forward_transfer",
    "stability_gap", "episodes_to_criterion", "episodes_to_criterion_censored",
    "summarize",
]
