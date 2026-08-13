"""Why does navigation take N steps, and why does it ever fail?

`mean_steps` is one number covering two very different failures, and
`success_rate` a third. This decomposes them on the exact evaluation path.

The signal is not the constraint: with the goal in memory `q` points at it to a
4-degree median error at every distance (`hopfield_separability`). So the
distance from the start, ~10.4 cells on average here, should take about
`dist / (stride * alignment)` steps -- and if the measured count is much worse,
one of those two factors says which.

    stride      how far the policy actually moves per step
    align_goal  cos between the step it takes and the true goal direction
    align_q     cos between the step it takes and the recall direction it was
                handed. Low `align_q` with high `align_goal` would mean the
                policy is navigating by something other than the recall;
                low `align_q` AND low `align_goal` means it is not following.
    predicted   dist / (stride * align_goal), the steps a policy with these
                statistics would need. `measured / predicted` near 1 means the
                count is fully explained by speed and aim, and the policy is
                simply slow rather than lost.

For failures it records the **closest approach**. A trial that spent the budget
circling one cell outside `goal_radius` is a precision problem at the very end;
one that never came within five cells is a following problem. Those look
identical in `success_rate`.

    python -m hopfield_nav.probes.nav_motion --ckpt <navigate_uN.pt> --n_dist 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from ..config import TrainConfig
from ..encoder_io import load_encoder
from ..evaluation.batched import batched_navigation_trials
from ..evaluation.checkpoint_io import cfg_from_checkpoint
from ..policy.agent import NavAgent, compute_input_dim
from ..rollout.distractors import goal_encoding, sample_distractors
from ..training.world_setup import build_field, replay_env_seeds
from ..world.env import make_env
from ..world.world import build_world


class NavRecorder:
    """Per-step stride and alignment, plus each trial's closest approach."""

    def __init__(self, goal: tuple[int, int], n: int) -> None:
        self.goal = np.asarray(goal, dtype=float)
        self.stride: list[np.ndarray] = []
        self.align_goal: list[np.ndarray] = []
        self.align_q: list[np.ndarray] = []
        self.closest = np.full(n, np.inf)
        self.start_dist = np.full(n, np.nan)
        # Per-trial sums, so the same statistics can be re-read over only the
        # trials that succeeded. A failure burns the whole step budget while a
        # success spends ~20 steps, so pooling every live step lets a handful
        # of failures supply a third of the sample and drag both means down.
        self.trial_steps = np.zeros(n)
        self.trial_stride = np.zeros(n)
        self.trial_align_goal = np.zeros(n)
        self.trial_align_q = np.zeros(n)

    def __call__(self, step, pos_before, actions, q, active, pos_after) -> None:
        if step == 0:
            self.start_dist = np.linalg.norm(pos_before - self.goal, axis=1)
        d_before = np.linalg.norm(pos_before - self.goal, axis=1)
        d_after = np.linalg.norm(pos_after - self.goal, axis=1)
        # Only live rows: a frozen trial is not stepped, so counting it would
        # dilute every statistic with zeros.
        self.closest = np.minimum(
            self.closest, np.where(active, np.minimum(d_before, d_after),
                                   np.inf))
        if not active.any():
            return
        moved = (pos_after - pos_before)[active]
        want = (self.goal - pos_before)[active]
        a = actions[active]
        stride = np.linalg.norm(moved, axis=1)
        ag = _cos(moved, want)
        aq = _cos(a, q[active])
        self.stride.append(stride)
        self.align_goal.append(ag)
        self.align_q.append(aq)
        rows = np.flatnonzero(active)
        self.trial_steps[rows] += 1.0
        self.trial_stride[rows] += stride
        self.trial_align_goal[rows] += ag
        self.trial_align_q[rows] += aq

    def summary(self) -> dict:
        return {
            "stride_mean": float(np.concatenate(self.stride).mean()),
            "align_goal_mean": float(np.concatenate(self.align_goal).mean()),
            "align_q_mean": float(np.concatenate(self.align_q).mean()),
        }


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1).clip(1e-12)
    nb = np.linalg.norm(b, axis=1).clip(1e-12)
    return (a * b).sum(1) / (na * nb)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--n_dist", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="val", choices=("val", "train", "fresh"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cpu"
                          or torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: TrainConfig = cfg_from_checkpoint(ck["config"])
    cfg.device = str(device)
    if "egocentric_heading" not in ck["config"].get("env", {}):
        cfg.env.egocentric_heading = False
    encoder, _e, gain = load_encoder(cfg.encoder_checkpoint, str(device),
                                     cfg.encoder_gain)
    cfg.encoder_gain = gain
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)

    field = build_field(cfg, encoder)
    np.random.seed(args.seed)
    env_seeds = replay_env_seeds(cfg, args.split, args.n_envs, args.seed)
    envs = [make_env(cfg.env, cfg.agent.movement_mode, seed=s)
            for s in env_seeds]
    world = build_world(field, envs, placement="spread", size=cfg.env.size)
    embed_dim = field.encoded_Phi.shape[2]

    agent = NavAgent(cfg.agent, compute_input_dim(
        cfg.agent, embed_dim, cfg.env.observation_size)).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()

    rng = np.random.RandomState(args.seed)
    steps_all, closest_fail, start_all, stats = [], [], [], []
    per_trial = {k: [] for k in ("steps", "stride", "align_goal", "align_q")}
    for env, offset in zip(world.envs, world.offsets):
        goal = tuple(int(c) for c in env.goal_location)
        hops, starts = [], []
        for _ in range(args.trials):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            pats = [goal_encoding(field, offset, goal)]
            pats += sample_distractors(field, offset, env.size, args.n_dist, rng)
            rng.shuffle(pats)
            for pat in pats:
                hop.input_memory(torch.from_numpy(pat).float())
            hops.append(hop)
            while True:
                s = (int(rng.randint(0, env.size)), int(rng.randint(0, env.size)))
                if s != goal:
                    break
            starts.append(s)

        rec = NavRecorder(goal, args.trials)
        steps = batched_navigation_trials(
            agent=agent, env=env, env_offset=offset, vectorhash=field,
            hopfields=hops, cfg=cfg, device=device, starts=starts, goal=goal,
            max_steps=args.max_steps, deterministic=True, on_step=rec)
        steps_all.extend(steps)
        start_all.extend(rec.start_dist.tolist())
        stats.append(rec.summary())
        per_trial["steps"].append(rec.trial_steps)
        per_trial["stride"].append(rec.trial_stride)
        per_trial["align_goal"].append(rec.trial_align_goal)
        per_trial["align_q"].append(rec.trial_align_q)
        closest_fail.extend(float(c) for s, c in zip(steps, rec.closest)
                            if s < 0)

    steps_arr = np.asarray(steps_all, dtype=float)
    ok = steps_arr > 0
    stride = float(np.mean([s["stride_mean"] for s in stats]))
    align_goal = float(np.mean([s["align_goal_mean"] for s in stats]))
    align_q = float(np.mean([s["align_q_mean"] for s in stats]))
    mean_start = float(np.mean(start_all))
    predicted = mean_start / max(stride * align_goal, 1e-9)
    measured = float(steps_arr[ok].mean()) if ok.any() else float("nan")

    # The same three statistics over the successful trials only, against the
    # start distance of those same trials -- the comparison `mean_steps` is
    # actually made of, since `mean_steps` averages successes alone.
    tsteps = np.concatenate(per_trial["steps"])
    start_arr = np.asarray(start_all, dtype=float)
    succ = {}
    if ok.any() and tsteps[ok].sum() > 0:
        denom = float(tsteps[ok].sum())
        s_stride = float(np.concatenate(per_trial["stride"])[ok].sum()) / denom
        s_ag = float(np.concatenate(per_trial["align_goal"])[ok].sum()) / denom
        s_aq = float(np.concatenate(per_trial["align_q"])[ok].sum()) / denom
        s_start = float(start_arr[ok].mean())
        s_pred = s_start / max(s_stride * s_ag, 1e-9)
        succ = {
            "succ_stride": s_stride, "succ_align_goal": s_ag,
            "succ_align_q": s_aq, "succ_mean_start_dist": s_start,
            "succ_predicted_steps": s_pred,
            "succ_measured_over_predicted": measured / s_pred if s_pred else None,
        }

    out = {
        "ckpt": args.ckpt, "n_dist": args.n_dist,
        "success_rate": float(ok.mean()),
        "mean_steps": measured,
        "mean_start_dist": mean_start,
        "stride": stride, "align_goal": align_goal, "align_q": align_q,
        "predicted_steps": predicted,
        "measured_over_predicted": measured / predicted if predicted else None,
        "n_fail": int((~ok).sum()),
        "fail_closest_median": (float(np.median(closest_fail))
                                if closest_fail else None),
        "fail_frac_within_goal_radius_x2": (
            float(np.mean(np.asarray(closest_fail)
                          <= 2 * cfg.env.goal_radius))
            if closest_fail else None),
        "goal_radius": float(cfg.env.goal_radius),
        **succ,
    }
    print(json.dumps(out, indent=2))
    print(f"\nsuccess {out['success_rate']:.3f}  steps {measured:.1f}  "
          f"(predicted {predicted:.1f} from stride {stride:.2f} x "
          f"align {align_goal:.2f} over {mean_start:.1f} cells)")
    print(f"  align to q = {align_q:.2f}")
    if succ:
        print(f"  successes only: predicted {succ['succ_predicted_steps']:.1f} "
              f"from stride {succ['succ_stride']:.2f} x align "
              f"{succ['succ_align_goal']:.2f} over "
              f"{succ['succ_mean_start_dist']:.1f} cells, "
              f"measured/predicted "
              f"{succ['succ_measured_over_predicted']:.2f}, "
              f"align to q = {succ['succ_align_q']:.2f}")
    if closest_fail:
        print(f"  {out['n_fail']} failures, closest approach median "
              f"{out['fail_closest_median']:.2f} cells "
              f"(goal_radius {cfg.env.goal_radius}); "
              f"{out['fail_frac_within_goal_radius_x2']:.0%} got within 2x the "
              f"radius and still did not finish")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
