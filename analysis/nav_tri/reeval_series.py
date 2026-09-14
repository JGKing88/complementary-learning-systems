"""Re-score a run's checkpoint series with the training-time evaluators.

The trainer's `[navigate_uN] nav={...}` / `expl={...}` lines are DETERMINISTIC
(`do_eval` passes `deterministic=True` to both evaluators). This walks a run
directory's `navigate_u*.pt` files, rebuilds the run's own recorded validation
envs from `world.json` (no RNG replay -- exactly what the run was scored
against), scores each checkpoint with the same two evaluators at the same
6 envs x 16 trials x {0, 10} distractors x 200 steps, and writes a log in the
trainer's own line format -- so `training_curve.py` and `sample_eff_curve.py`
read it as they read a training log. The one knob is `--no-deterministic`:
actions sampled from the policy, which is the convention for explore
(EXPERIMENTS_NAV_P2 section 23; memory: "ALWAYS evaluate uncertain policies
sampled").

    python -m analysis.nav_tri.reeval_series \\
        --run_dir $CLS_CKPTS/navigate_navp2_se_b8_lr1_s42_22701298 \\
        --every 50 --no-deterministic \\
        --out $CLS_RESULTS/nav_tri_probe/reeval_se_b8_lr1_stoch.log
    python -m analysis.nav_tri.training_curve --log <that .log> --out_prefix ...

`samples=` lines are carried from each checkpoint's `cum_episodes` /
`cum_env_steps` when it has them, so the sample-axis reader works too.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_world_from_spec, load_agent, world_spec_for)
from hopfield_nav.evaluation.metrics import (
    evaluate_exploration, evaluate_navigation)


def _update_of(path: str) -> int:
    m = re.search(r"navigate_u(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out", required=True, help="log file to write (trainer format)")
    p.add_argument("--every", type=int, default=25,
                   help="score checkpoints whose update is a multiple of this")
    p.add_argument("--min_update", type=int, default=0)
    p.add_argument("--max_update", type=int, default=None)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True, help="policy mean (the trainer's eval) or sampled")
    p.add_argument("--trials", type=int, default=None,
                   help="trials per env (default: the run's n_val_trials)")
    p.add_argument("--n_distractors", type=int, nargs="+", default=None,
                   help="default: the run's val_n_distractors_list")
    p.add_argument("--max_steps", type=int, default=None,
                   help="default: the run's eval_max_steps (or 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "navigate_u*.pt")),
                   key=_update_of)
    ckpts = [c for c in ckpts
             if _update_of(c) % args.every == 0
             and _update_of(c) >= args.min_update
             and (args.max_update is None or _update_of(c) <= args.max_update)]
    if not ckpts:
        raise SystemExit(f"no matching checkpoints in {args.run_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(first["config"])
    cfg.device = str(device)
    spec = world_spec_for(args.run_dir)
    if spec is None:
        raise SystemExit(f"{args.run_dir} has no world.json; the recorded val "
                         "set cannot be rebuilt.")
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
                                          cfg.encoder_gain)
    cfg.encoder_gain = gain
    envs, field, offsets = eval_world_from_spec(spec, cfg, encoder, str(device))
    agent = load_agent(cfg, first["agent_state_dict"], enc_cfg.out_dim, device)

    nt = args.trials if args.trials is not None else int(cfg.n_val_trials)
    dist = (args.n_distractors if args.n_distractors is not None
            else list(cfg.val_n_distractors_list))
    max_steps = (args.max_steps if args.max_steps is not None
                 else int(cfg.eval_max_steps or 200))
    mode = "deterministic" if args.deterministic else "SAMPLED"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        # A banner in the launcher's shape, so sample_eff_curve.parse_header
        # can read the pool geometry off this file too.
        fh.write(f"=== reeval_series run={os.path.basename(args.run_dir)} "
                 f"mode={mode} ===\n")
        fh.write(f"    rollout    : {cfg.envs_per_world} envs x "
                 f"{cfg.batch_envs} batch x {cfg.steps_per_rollout} steps\n")
        fh.write(f"    schedule   : {cfg.schedule}\n")
        fh.write(f"    eval       : {len(envs)} envs x {nt} trials, "
                 f"n_dist={dist}, max_steps={max_steps}, {mode}, "
                 f"split=recorded\n")
        fh.flush()
        print(f"{len(ckpts)} checkpoints, {len(envs)} envs x {nt} trials, "
              f"n_dist={dist}, {mode}", flush=True)
        for c in ckpts:
            u = _update_of(c)
            ck = torch.load(c, map_location="cpu", weights_only=False)
            agent.load_state_dict(ck["agent_state_dict"])
            agent.eval()
            torch.manual_seed(args.seed + u)
            np.random.seed(args.seed + u)
            t0 = time.time()
            nav = evaluate_navigation(
                agent, envs, field, offsets, cfg, device, num_trials=nt,
                max_steps=max_steps, n_distractors_list=dist,
                deterministic=args.deterministic, seed=args.seed)
            expl = evaluate_exploration(
                agent, envs, field, offsets, cfg, device, num_trials=nt,
                max_steps=max_steps, n_distractors_list=dist,
                deterministic=args.deterministic, seed=args.seed)
            if "cum_episodes" in ck:
                fh.write(f"  [navigate_u{u}] samples={{'episodes': "
                         f"{ck['cum_episodes']}, 'env_steps': "
                         f"{ck['cum_env_steps']}}}\n")
            fh.write(f"  [navigate_u{u}] nav={nav}\n")
            fh.write(f"  [navigate_u{u}] expl={expl}\n")
            fh.write(f"  [navigate_u{u}] eval_seconds={time.time() - t0:.1f} "
                     f"mode={mode}\n")
            fh.flush()
            print(f"  u{u}: succ {nav[dist[0]]['success_rate']:.3f}/"
                  f"{nav[dist[-1]]['success_rate']:.3f} steps "
                  f"{nav[dist[0]]['mean_steps']:.1f}/{nav[dist[-1]]['mean_steps']:.1f} "
                  f"swept {expl[dist[0]]['swept_coverage']:.3f}/"
                  f"{expl[dist[-1]]['swept_coverage']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
