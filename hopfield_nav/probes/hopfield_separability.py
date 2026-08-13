"""Is "this recall came from my own env" readable off the policy's inputs?

The explore/exploit conflict in this project has one honest statement: the
policy must **follow** the Hopfield displacement when the recalled pattern is
its own env's goal, and **ignore** it when the memory holds only distractors
drawn from elsewhere in the scaffold. Every schedule, curriculum and shaping
knob is an attempt to teach that distinction. None of them can work if the
distinction is not present in the channels the policy actually reads.

So this measures the channels, not a policy. For a grid of agent positions in
real envs it builds the three memory contents a rollout can have --

    goal            the exploit regime at n_dist=0
    goal + K dist   the exploit regime at n_dist=K
    K dist          the explore regime, which is what must be ignored

-- and reports, per case, the two quantities the policy is handed:

    ||q_s||         magnitude of the projected displacement at Hopfield
                    iteration s (the `input_hopfield_raw` channel is q_1; the
                    `input_hopfield_multistep` channels are q_s for each s)
    cos(q_1, q_3)   how much the recall *moves* as it iterates

plus, for the goal cases, the angular error of q_1 against the true direction
to the goal, bucketed by distance -- which is what decides whether following
the channel is even the right policy.

The separability numbers at the end are the point. `auc_mag` is the
probability that a random goal-in-memory position has a larger ||q_1|| than a
random distractors-only one; 0.5 is chance and 1.0 is perfect. If it is near
chance the disambiguation is not in the input and no amount of PPO will find
it; if it is near 1 the input is fine and the failure is credit assignment.

    python -m hopfield_nav.diagnostics.hopfield_separability \
        --encoder <path> --size 20 --n_envs 4 --device cuda
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from ..config import AgentConfig, EnvConfig, TrainConfig, VectorHashConfig
from ..encoder_io import load_encoder
from ..rollout import signal as signal_ops
from ..rollout.distractors import goal_encoding, sample_distractors
from ..training.world_setup import build_field
from ..world.env import make_env
from ..world.world import build_world


def _angles(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[:, 1], v[:, 0])


def _cos_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1).clip(1e-12)
    nb = np.linalg.norm(b, axis=-1).clip(1e-12)
    return (a * b).sum(-1) / (na * nb)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random pos > random neg), by rank -- ties count a half.

    Mann-Whitney U over the two samples. Written out rather than pulled from
    scipy because this file is imported on nodes where scipy is not the
    dependency anyone checked.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(len(both), dtype=np.float64)
    ranks[order] = np.arange(1, len(both) + 1)
    # Average ranks within tie groups, so an all-ties comparison reads 0.5.
    sorted_vals = both[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _follow_stats(pos: np.ndarray, q: np.ndarray, size: int):
    """What happens to q after one step ALONG q -- the closed-loop cue.

    A recurrent policy does not have to classify the memory from a single
    frame. It can act and look: step along the recalled direction and see
    whether the recall behaves like a place that is really there. For a goal at
    distance r the displacement shrinks by about one cell and keeps pointing
    the same way; for a memory of somewhere outside this arena the projected
    field is just a rough function of position, so neither holds.

    Returns ``(follow_cos, follow_dmag, valid)`` per position: the cosine
    between q here and q one step on, the change in ||q||, and whether the step
    actually changed cell (a step that snaps back to the same cell says
    nothing, and would otherwise enter every statistic as a perfect score).
    """
    index = {(int(p[0]), int(p[1])): i for i, p in enumerate(pos)}
    n = len(pos)
    follow_cos = np.full(n, np.nan)
    follow_dmag = np.full(n, np.nan)
    mag = np.linalg.norm(q, axis=1)
    for i in range(n):
        if mag[i] < 1e-9:
            continue
        u = q[i] / mag[i]
        tgt = np.clip(np.rint(pos[i] + u).astype(int), 0, size - 1)
        j = index.get((int(tgt[0]), int(tgt[1])))
        if j is None or j == i:
            continue
        follow_cos[i] = _cos_between(q[i:i + 1], q[j:j + 1])[0]
        follow_dmag[i] = mag[j] - mag[i]
    return follow_cos, follow_dmag, ~np.isnan(follow_cos)


def probe(cfg: TrainConfig, encoder, steps: list[int], n_envs: int,
          n_dist: int, device: torch.device, seed: int = 0) -> dict:
    """Sweep every cell of `n_envs` envs under each memory content."""
    field = build_field(cfg, encoder)
    # `build_world`'s default rng is the np.random module, so env PLACEMENT is
    # global state. Two probe runs that differ only in a post-hoc statistic
    # otherwise land their envs on different scaffold patches and report
    # different magnitudes -- which is how the first pass of this file read a
    # goal/distractor magnitude ratio of 5.2 and the second, 2.0.
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    envs = [make_env(cfg.env, cfg.agent.movement_mode,
                     seed=int(rng.randint(0, 10_000_000)))
            for _ in range(n_envs)]
    world = build_world(field, envs, placement="spread", size=cfg.env.size)
    embed_dim = field.encoded_Phi.shape[2]

    size = cfg.env.size
    cells = np.array([(x, y) for x in range(size) for y in range(size)],
                     dtype=np.int32)

    rows: list[dict] = []
    for env_idx, (env, offset) in enumerate(zip(world.envs, world.offsets)):
        goal = tuple(int(c) for c in env.goal_location)
        goal_enc = goal_encoding(field, offset, goal)
        dists = sample_distractors(field, offset, env.size, n_dist, rng)

        contents = {
            "goal": [goal_enc],
            f"goal+{n_dist}dist": [goal_enc] + list(dists),
            f"{n_dist}dist": list(dists),
        }
        # A single distractor is the cleanest version of the explore signal:
        # one attractor, so recall converges to it and q is a coherent, wrong
        # direction rather than a blend. If anything drives chase behaviour it
        # is this case, not the K-pattern blend.
        if n_dist > 1:
            contents["1dist"] = [dists[0]]

        # Positions the agent can occupy. The goal cell itself is dropped: the
        # displacement to yourself is zero and would put a degenerate row in
        # every magnitude statistic.
        mask = ~((cells[:, 0] == goal[0]) & (cells[:, 1] == goal[1]))
        pos = cells[mask]
        emb_np = field.get_encoded_state(pos, offset)
        emb = torch.from_numpy(emb_np).float().to(device)
        true_vec = (pos - np.array(goal, dtype=np.int32)).astype(np.float64)
        # `gram_schmidt_2d_batch` stacks [East, North] -- East from the +x
        # neighbour difference, North from +y -- so q is (dx, dy) in the SAME
        # component order as the action the env applies to `_pos_f`. The
        # direction the agent should move is goal - pos, unswapped.
        want = -true_vec
        r = np.linalg.norm(true_vec, axis=1)

        for name, patterns in contents.items():
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            for pat in patterns:
                hop.input_memory(torch.from_numpy(pat).float())

            _sig, q1, _mask_t, W = signal_ops.hopfield_signal_at(
                field, cfg, emb_np, emb, pos, offset, hop, True, device,
                embed_dim,
            )
            msq = signal_ops.multistep_q(
                field, cfg, emb_np, emb, hop, True, W, steps, embed_dim, device,
            )
            q_last = msq[steps[-1]]
            err = np.degrees(np.abs(np.arctan2(
                np.sin(_angles(q1) - _angles(want)),
                np.cos(_angles(q1) - _angles(want)))))
            f_cos, f_dmag, f_ok = _follow_stats(pos, q1, size)
            for i in range(len(pos)):
                rows.append({
                    "case": name,
                    "env": int(env_idx),
                    "r": float(r[i]),
                    "mag1": float(np.linalg.norm(q1[i])),
                    "mag_last": float(np.linalg.norm(q_last[i])),
                    "cos_1_last": float(_cos_between(q1[i:i + 1],
                                                     q_last[i:i + 1])[0]),
                    "ang_err": float(err[i]),
                    "follow_cos": float(f_cos[i]) if f_ok[i] else None,
                    "follow_dmag": float(f_dmag[i]) if f_ok[i] else None,
                })
    return {"rows": rows, "steps": steps, "n_dist": n_dist,
            "n_envs": n_envs, "size": size}


def summarize(out: dict) -> dict:
    rows = out["rows"]
    n_dist = out["n_dist"]
    cases = sorted({r["case"] for r in rows})
    by_case = {c: [r for r in rows if r["case"] == c] for c in cases}

    def col(rs, k):
        return np.array([np.nan if r[k] is None else r[k] for r in rs],
                        dtype=np.float64)

    def fin(rs, k):
        """`col` with the undefined rows dropped -- see `_follow_stats`."""
        v = col(rs, k)
        return v[np.isfinite(v)]

    summary: dict = {"per_case": {}, "by_distance": {}, "separability": {}}
    for c in cases:
        rs = by_case[c]
        m1, ml = col(rs, "mag1"), col(rs, "mag_last")
        summary["per_case"][c] = {
            "n": len(rs),
            "mag1_median": float(np.median(m1)),
            "mag1_p10": float(np.percentile(m1, 10)),
            "mag1_p90": float(np.percentile(m1, 90)),
            "mag_last_median": float(np.median(ml)),
            "cos_1_last_median": float(np.median(col(rs, "cos_1_last"))),
            "cos_1_last_p10": float(np.percentile(col(rs, "cos_1_last"), 10)),
            "ang_err_median": float(np.median(col(rs, "ang_err"))),
            "frac_ang_err_lt_30": float((col(rs, "ang_err") < 30).mean()),
            # The closed-loop pair: step along q, then look again.
            "follow_cos_median": float(np.median(fin(rs, "follow_cos"))),
            "follow_dmag_median": float(np.median(fin(rs, "follow_dmag"))),
            "frac_follow_closes": float(
                (fin(rs, "follow_dmag") < 0).mean()),
        }

    # Following the channel is only the right policy where the direction is
    # right, and that is a function of how far the goal is.
    edges = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 30)]
    for c in cases:
        rs = by_case[c]
        r = col(rs, "r")
        buckets = {}
        for lo, hi in edges:
            sel = (r >= lo) & (r < hi)
            if not sel.any():
                continue
            buckets[f"{lo}-{hi}"] = {
                "n": int(sel.sum()),
                "mag1_median": float(np.median(col(rs, "mag1")[sel])),
                "ang_err_median": float(np.median(col(rs, "ang_err")[sel])),
                "frac_ang_err_lt_30": float(
                    (col(rs, "ang_err")[sel] < 30).mean()),
            }
        summary["by_distance"][c] = buckets

    # The realistic comparison is `goal+Kdist` against `Kdist`: an exploit
    # rollout and an explore rollout differ by whether the goal is among the
    # patterns, not by whether distractors are. `goal` vs `Kdist` is the
    # n_dist=0 idealization and is reported only to show how much of any cue
    # comes from the distractors rather than from the goal.
    neg_name = f"{n_dist}dist"
    if neg_name in by_case:
        neg = by_case[neg_name]
        for pos_name in ("goal", f"goal+{n_dist}dist"):
            if pos_name not in by_case:
                continue
            p = by_case[pos_name]
            entry = {
                "auc_mag1": _auc(col(p, "mag1"), col(neg, "mag1")),
                "auc_cos_1_last": _auc(col(p, "cos_1_last"),
                                       col(neg, "cos_1_last")),
                # A real target CLOSES as you step toward it, so the
                # discriminating score is -d||q||: more positive = more like a
                # goal. Both samples are negated, and the positive class goes
                # first, or the number comes out as its own complement.
                "auc_follow_dmag": _auc(-fin(p, "follow_dmag"),
                                        -fin(neg, "follow_dmag")),
                "auc_follow_cos": _auc(fin(p, "follow_cos"),
                                       fin(neg, "follow_cos")),
                "mag1_ratio_median": float(
                    np.median(col(p, "mag1")) / max(
                        float(np.median(col(neg, "mag1"))), 1e-12)),
                # The closed-loop cue as the one bit a policy would actually
                # act on -- "did stepping along q bring me closer?" -- rather
                # than as a ranking over a continuous score.
                "closes_pos": float((fin(p, "follow_dmag") < 0).mean()),
                "closes_neg": float((fin(neg, "follow_dmag") < 0).mean()),
            }
            # Per-env spread. Env PLACEMENT moves these a lot -- a distractor
            # drawn uniformly from a 1716x1716 scaffold lands at a distance
            # that depends on where the arena sits -- so a pooled AUC hides
            # whether the cue is reliable or merely reliable on average.
            envs = sorted({r["env"] for r in rows})
            per_env = []
            for e in envs:
                pe = [r for r in p if r["env"] == e]
                ne = [r for r in neg if r["env"] == e]
                per_env.append(_auc(col(pe, "mag1"), col(ne, "mag1")))
            entry["auc_mag1_per_env"] = [round(a, 3) for a in per_env]
            entry["auc_mag1_worst_env"] = float(np.min(per_env))
            summary["separability"][f"{pos_name}_vs_{neg_name}"] = entry
    return summary


def build_cfg(args) -> TrainConfig:
    cfg = TrainConfig()
    cfg.env = EnvConfig(
        size=args.size, observation_size=args.observation_size,
        movement_mode="continuous", goal_radius=args.goal_radius,
        continuous_normalize=False, wall_resolution=args.wall_resolution,
    )
    cfg.vectorhash = VectorHashConfig(
        lambdas=list(args.lambdas), Np=args.Np, static_vectorhash=True)
    cfg.agent = AgentConfig(
        hopfield_mode="continuous", movement_mode="continuous",
        input_hopfield_raw=True, input_hopfield_multistep=list(args.steps))
    cfg.fwhm_ratio = args.fwhm_ratio
    cfg.device = args.device
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--wall_resolution", type=int, default=4)
    p.add_argument("--goal_radius", type=float, default=1.0)
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--Np", type=int, default=400)
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--n_dist", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args)
    encoder, _enc_cfg, gain = load_encoder(args.encoder, str(device), None)
    cfg.encoder_gain = gain
    cfg.hopfield.beta = float(gain)
    print(f"encoder gain (= Hopfield beta) = {gain}", flush=True)

    out = probe(cfg, encoder, list(args.steps), args.n_envs, args.n_dist,
                device, seed=args.seed)
    summary = summarize(out)
    print(json.dumps(summary, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary,
                       "config": {k: v for k, v in vars(args).items()}}, f,
                      indent=2)
        print(f"\nwrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
