"""How far does one gradient step move this policy? -- the PPO optimizer probe.

docs/EXPERIMENTS_SAMPLE_EFF.md. The first se_* smoke reported approx_kl 0.58
and clip_frac 0.43 averaged over the two gradient steps update 1 took before
the KL stop fired -- i.e. after ONE Adam step ~86% of the scored transitions
were already outside the +/-0.15 clip region. If that holds past the init
policy it says d0_base's 16 steps per update were mostly clipped away after
the first, and it decides what target_kl / lr / minibatch size the
sample-efficiency arms can run.

This builds ONE update's pool exactly as `run_navigate` does (same regimes,
same distractor draws, same collector), from a checkpoint's own config with
the pool shape overridden, and then runs `ppo_update` from that pool under
several optimizer settings, each from a fresh copy of the agent and a fresh
Adam, printing the KL and clip fraction after every gradient step.

    python -m analysis.nav_tri.ppo_step_probe \\
        --ckpt <run>/navigate_u725.pt --fresh \\
        --envs 20 --batch 8 --steps 200 \\
        --grid "4x4:3e-4" "10x8:3e-4" "10x8:1e-4" "10x8:3e-4:kl0.02"

`--fresh` probes a freshly initialised agent (what the first updates of a new
run see) in addition to the checkpoint's weights. Grid entries are
EPOCHSxMINIBATCHES:LR[:klTARGET].
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder, validate_config
from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint
from hopfield_nav.policy.action_head import action_bounds_from
from hopfield_nav.policy.agent import NavAgent, compute_input_dim
from hopfield_nav.rollout.collector import RolloutCollector
from hopfield_nav.training.exploit import ExploitRegime
from hopfield_nav.training.explore import ExploreRegime
from hopfield_nav.training.stages import Knobs
from hopfield_nav.training.world_setup import (
    build_field, set_phase_freeze, setup_run_world)
from hopfield_nav.updates.ppo import ppo_update
import run_manifest


def collect_pool(cfg, agent, worlds, embed_dim, device, dist_rng, *,
                 empty_frac: float, eps: float, update: int = 1):
    """One update's rollouts, the way run_navigate collects them."""
    exploit_regime = ExploitRegime(cfg, embed_dim, device, dist_rng,
                                   use_distractors=True)
    explore_regime = ExploreRegime(cfg, embed_dim, device, dist_rng,
                                   goals_off=cfg.explore_goals_off,
                                   use_distractors=True,
                                   ends_on_goal=cfg.explore_ends_on_goal)
    knobs = Knobs(lr=cfg.ppo.lr, empty_frac=empty_frac,
                  novelty=cfg.hopfield.novelty_reward, eps=eps,
                  dist_min=cfg.n_train_distractors_min,
                  dist_max=cfg.n_train_distractors_max,
                  emp_dist_min=cfg.n_train_emp_distractors_min,
                  emp_dist_max=cfg.n_train_emp_distractors_max)
    n_envs = cfg.envs_per_world
    n_emp = int(round(n_envs * empty_frac))
    is_pre = np.zeros(n_envs, dtype=bool)
    is_pre[np.random.permutation(n_envs)[:n_envs - n_emp]] = True
    base_nov = cfg.hopfield.novelty_reward
    rollouts = []
    agent.eval()
    for w_idx, world in enumerate(worlds):
        collector = RolloutCollector(world.field, cfg, embed_dim, device)
        for local_idx, env in enumerate(world.envs):
            env_offset = world.offsets[local_idx]
            regime = exploit_regime if is_pre[local_idx] else explore_regime
            spec = regime.spec(w_idx, world, local_idx, env, env_offset, knobs)
            cfg.hopfield.novelty_reward = spec.novelty_reward
            env.goals_active = spec.goals_active
            rollouts.append(collector.collect_rollout(
                env, agent, spec.hop, allow_store=spec.allow_store,
                h_rnn=None, env_offset=env_offset, update_idx=update,
                aux_scale=1.0, epsilon_now=spec.epsilon,
                goal_in_memory_init=spec.goal_in_memory_init,
                ends_on_goal=spec.ends_on_goal))
    cfg.hopfield.novelty_reward = base_nov
    return rollouts, is_pre


def parse_grid(entry: str):
    parts = entry.split(":")
    ep, mb = (int(x) for x in parts[0].split("x"))
    lr = float(parts[1]) if len(parts) > 1 else 3e-4
    kl = None
    for extra in parts[2:]:
        if extra.startswith("kl"):
            kl = float(extra[2:])
    return {"epochs": ep, "mb": mb, "lr": lr, "target_kl": kl}


def run_grid(agent0, rollouts, cfg, grid, device, label, opt_state=None):
    out = {}
    for g in grid:
        agent = copy.deepcopy(agent0)
        set_phase_freeze(agent, freeze_move=False,
                         freeze_store=cfg.freeze_store,
                         freeze_value=False, freeze_rnn=False)
        agent.train()
        ppo = copy.deepcopy(cfg.ppo)
        ppo.ppo_epochs, ppo.n_minibatches = g["epochs"], g["mb"]
        ppo.lr, ppo.target_kl = g["lr"], g["target_kl"]
        opt = torch.optim.Adam([p for p in agent.parameters()
                                if p.requires_grad], lr=ppo.lr)
        if opt_state is not None:
            # Steady-state Adam: a fresh optimizer's first step moves EVERY
            # parameter by exactly lr (m/sqrt(v) = +/-1), which is the
            # first-update regime and not what a run in progress sees.
            opt.load_state_dict(copy.deepcopy(opt_state))
            for grp in opt.param_groups:
                grp["lr"] = ppo.lr
        trace: list = []
        torch.manual_seed(0)
        stats = ppo_update(agent, rollouts, ppo, opt, trace=trace)
        name = (f"{g['epochs']}x{g['mb']} lr={g['lr']:g}"
                + (f" kl={g['target_kl']}" if g["target_kl"] else ""))
        print(f"\n--- {label} :: {name} :: {stats['grad_steps']:.0f} steps, "
              f"{stats['epochs_run']:.0f} epochs, mean kl {stats['approx_kl']:.4f} "
              f"clip {stats['clip_frac']:.3f} ---")
        # per-epoch summary + the first few steps verbatim
        by_epoch: dict[int, list] = {}
        for t in trace:
            by_epoch.setdefault(t["epoch"], []).append(t)
        for e, ts in by_epoch.items():
            kls = [t["approx_kl"] for t in ts]
            cfs = [t["clip_frac"] for t in ts]
            print(f"  epoch {e}: kl mean {np.mean(kls):.4f} max {np.max(kls):.4f}"
                  f" | clip mean {np.mean(cfs):.3f} max {np.max(cfs):.3f}"
                  f" | grad_norm {np.mean([t['grad_norm'] for t in ts]):.2f}")
        print("  first steps: " + "  ".join(
            f"s{t['step']}(kl {t['approx_kl']:.3f}, clip {t['clip_frac']:.2f})"
            for t in trace[:6]))
        out[name] = {"stats": stats, "trace": trace}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--no_ckpt_weights", action="store_true",
                   help="skip the checkpoint-weights condition")
    p.add_argument("--envs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--empty_frac", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--grid", nargs="+",
                   default=["4x4:3e-4", "10x8:3e-4", "10x8:3e-4:kl0.02"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    p.add_argument("--optimizer_from", default=None,
                   help="A resume point (resume/latest.pt) whose Adam moments "
                        "are loaded for the checkpoint-weights condition, so "
                        "the probe reflects a run in progress rather than a "
                        "first update.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.envs_per_world = args.envs
    cfg.batch_envs = args.batch
    cfg.steps_per_rollout = args.steps
    cfg.use_wandb = False
    cfg.device = str(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
                                          cfg.encoder_gain)
    embed_dim = enc_cfg.out_dim
    validate_config(enc_cfg, cfg.vectorhash.lambdas, gain, cfg.fwhm_ratio)
    cfg.encoder_gain = gain
    field = build_field(cfg, encoder)
    ident = run_manifest.encoder_identity(cfg.encoder_checkpoint, enc_cfg, gain)
    rw = setup_run_world(cfg, encoder, embed_dim, rng, field, cadence=None,
                         n_updates=1, encoder_ident=ident,
                         where="ppo_step_probe")
    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    grid = [parse_grid(g) for g in args.grid]
    dist_rng = np.random.RandomState(cfg.seed + 7919)
    results = {}

    opt_state = None
    if args.optimizer_from:
        rs = torch.load(args.optimizer_from, map_location=device,
                        weights_only=False)
        opt_state = rs["optimizer_state_dict"]
        print(f"Adam moments from {args.optimizer_from} (u{rs.get('update')})")
    conditions = []
    if not args.no_ckpt_weights:
        conditions.append(("ckpt", ck["agent_state_dict"]))
        if opt_state is not None:
            conditions.append(("ckpt+adam", ck["agent_state_dict"]))
    if args.fresh:
        conditions.append(("fresh", None))
    for label, sd in conditions:
        torch.manual_seed(args.seed)
        agent = NavAgent(cfg.agent, input_dim,
                         action_bounds=action_bounds_from(cfg.env)).to(device)
        if sd is not None:
            agent.load_state_dict(sd)
        np.random.seed(args.seed)
        rollouts, is_pre = collect_pool(cfg, agent, rw.worlds, embed_dim,
                                        device, dist_rng,
                                        empty_frac=args.empty_frac,
                                        eps=args.eps)
        n_traj = sum(r.rewards.shape[0] for r in rollouts)
        n_steps = sum(int(r.alive_mask.sum()) if r.alive_mask is not None
                      else r.rewards.numel() for r in rollouts)
        print(f"\n===== {label}: pool {n_traj} trajectories, {n_steps} "
              f"realized env-steps, {int(is_pre.sum())} exploit envs =====")
        results[label] = run_grid(
            agent, rollouts, cfg, grid, device, label,
            opt_state=opt_state if label == "ckpt+adam" else None)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
