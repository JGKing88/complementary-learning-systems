"""Experiment B: a recurrent network on lifetimes of goal-conditioned episodes (plan §4).

    python -m hopfield_nav.train_goal_lifetimes --mode grid --arm full ...

The same world, holdouts and static table as Experiment A (`build_env_sets`,
`evaluate_pairs`), with one difference: the data are **rollouts**. Each
training env runs `batch_envs` parallel lifetimes; a lifetime is
`resample_envs_every` chunks of `steps_per_rollout` steps with the hidden
state carried across chunks, and on every goal-reach a row gets a fresh
start AND a fresh goal (a goal pool on the vec env, plan §4.3) with its
hidden state kept. DAgger against the same teacher A uses.

Three arms, the factorial that makes a B win attributable (§4.2):

    full   GRU + prev_action      the hypothesis
    rec    GRU, no prev_action    recurrence alone
    dist   MLP, no prev_action    the rollout data distribution, no memory

Readouts (§4.4): 1, A's static table at h = 0 through `RNNAgentAsPairModel`,
on every env set; 2, direction quality by (episode, step) over lifetimes on
the held-out sets (`evaluation/lifetime.py`).

Its own composer rather than a mode of `train_rnn.py`: that file's mixed
mode redraws envs through the legacy builder, which cannot place them in a
declared region -- and the corner world is the point of B1x.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from cls_paths import run_dir, run_name
import run_manifest
from .config import EnvConfig, RNNBCConfig, RNNTrainConfig
from .evaluation.goal_pairs import RNNAgentAsPairModel, format_table
from .evaluation.lifetime import aggregate_lifetimes, evaluate_lifetime_direction
from .policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from .rollout.rnn import collect_rollout_rnn
from .training.goal_pairs_setup import (
    ARMS, MODES, agent_cfg_for_mode, build_env_sets, eval_all, jsonable)
from .training.rnn_setup import write_rnn_world_spec
from .updates.bc_rnn import bc_rnn_update
from .world.vec_env import make_vec



def main() -> None:
    p = argparse.ArgumentParser(description="Experiment B: recurrent lifetimes")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p.add_argument("--movement_mode", choices=["discrete", "continuous"], default="continuous")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--nonlinearity", choices=["tanh", "relu"], default="relu")
    # World (same flags as A)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--n_val_envs", type=int, default=16)
    p.add_argument("--n_same_envs", type=int, default=8)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=120)
    p.add_argument("--wall_resolution", type=int, default=1)
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--place_margin", type=int, default=20)
    p.add_argument("--goal_val_frac", type=float, default=0.2)
    p.add_argument("--region_val_frac", type=float, default=0.1)
    p.add_argument("--wall_seeds", type=str, default="0,10000000")
    p.add_argument("--place_region", type=str, default="anywhere")
    p.add_argument("--n_ood_place", type=int, default=0)
    # Rollouts
    p.add_argument("--batch_envs", type=int, default=64, help="parallel lifetimes per env")
    p.add_argument("--envs_per_update", type=int, default=8)
    p.add_argument("--steps_per_rollout", type=int, default=64)
    p.add_argument("--resample_envs_every", type=int, default=32,
                   help="chunks per lifetime; 32 x 64 = 2048 steps")
    p.add_argument("--episode_max_steps", type=int, default=60)
    p.add_argument("--init_log_std", type=float, default=-1.0)
    # BC
    p.add_argument("--n_updates", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--n_minibatches", type=int, default=4)
    # Eval
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_pairs", type=int, default=2048)
    p.add_argument("--lifetime_every", type=int, default=500)
    p.add_argument("--n_lifetimes", type=int, default=64)
    p.add_argument("--n_eval_episodes", type=int, default=20)
    p.add_argument("--n_lifetime_envs", type=int, default=8, help="held-out envs per lifetime eval")
    p.add_argument("--ckpt_every", type=int, default=500)
    # Run
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb_project", default="train_goal_lifetimes")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    arm = ARMS[args.arm]
    acfg = agent_cfg_for_mode(args.mode, args.movement_mode,
                              hidden_size=args.hidden_size, num_rnn_layers=args.num_layers,
                              rnn_nonlinearity=args.nonlinearity if arm["rnn_cell"] == "mlp" else "tanh",
                              init_log_std=args.init_log_std, **arm)
    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode=args.movement_mode, wall_resolution=args.wall_resolution,
                      continuous_normalize=True, goals_active=True),
        agent=acfg, bc=RNNBCConfig(lr=args.lr, epochs=args.epochs, n_minibatches=args.n_minibatches),
        mode="lifetimes", n_envs=args.n_envs, n_val_envs=args.n_val_envs,
        n_updates=args.n_updates, batch_envs=args.batch_envs,
        steps_per_rollout=args.steps_per_rollout, eval_every=args.eval_every,
        carry_across_episodes=True, resample_envs_every=args.resample_envs_every,
        episode_max_steps=args.episode_max_steps, seed=args.seed, device=args.device,
        use_wandb=args.use_wandb, wandb_project=args.wandb_project,
        fwhm_ratio=args.fwhm_ratio, lambdas=list(args.lambdas), env_generator=True,
        place_margin=args.place_margin, goal_val_frac=args.goal_val_frac,
        region_val_frac=args.region_val_frac, wall_seeds=args.wall_seeds,
        place_region=args.place_region, resample_goal_on_reach=True,
    )
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    built = build_env_sets(cfg, rng, n_same=args.n_same_envs, n_ood_place=args.n_ood_place,
                           keep_field=True)
    train, heldout, same, split, vh, sgb = built[:6]
    heldout_out = built[6] if len(built) > 6 else None
    cells = split.cell_sets()
    sets = [train, heldout, same] + ([heldout_out] if heldout_out else [])
    print(f"world: {len(train)} train / {len(heldout)} {heldout.name} / {len(same)} same"
          + (f" / {len(heldout_out)} heldout_out" if heldout_out else "")
          + f"; place={args.place_region}; cells={cells.summary()}; {time.time()-t0:.1f}s")

    D = compute_rnn_input_dim(acfg, args.observation_size, vh.Ng)
    agent = RNNAgent(acfg, D).to(device)
    n_params = sum(q.numel() for q in agent.parameters())
    print(f"agent: mode={args.mode} arm={args.arm} cell={acfg.rnn_cell} prev_action={acfg.input_prev_action} "
          f"D={D} hidden={args.hidden_size} layers={args.num_layers} params={n_params:,}")
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project,
                               config={**asdict(cfg), "tag": args.tag, "arm": args.arm,
                                       "input_dim": D, "params": n_params})
    if args.save_dir is None:
        sub = run_name(wandb_run.name if wandb_run is not None else None)
        if args.tag:
            sub = f"{args.tag}_{sub}"
        args.save_dir = str(run_dir("goal_lifetimes", sub))
    else:
        sub = os.path.basename(args.save_dir.rstrip("/"))
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"save_dir={args.save_dir}")
    write_rnn_world_spec(cfg, split, vh, generator="declared", save_dir=args.save_dir)
    run_manifest.begin(args.save_dir, kind="goal_lifetimes", name=sub,
                       config={**asdict(cfg), "argv": vars(args)}, parent=None, wandb_run=wandb_run)

    # One vec per training env, each carrying `batch_envs` lifetimes and a
    # goal pool. `envs_per_update` of them are rolled out per update, round
    # robin, and each keeps its own carried state.
    vecs, carry_h, chunks_done = [], [], []
    for env in train.envs:
        v = make_vec(env, args.batch_envs, args.movement_mode, 1.0, True, reset=False)
        v.set_goal_pool(cells.goal_train)
        v.reset_all()
        vecs.append(v); carry_h.append(None); chunks_done.append(0)

    def lifetime_eval(u):
        out = {}
        for es in sets:
            if es.name == "train":
                continue
            res = []
            for i, (env, off) in enumerate(zip(es.envs[:args.n_lifetime_envs], es.offsets[:args.n_lifetime_envs])):
                res.append(evaluate_lifetime_direction(
                    env, agent, cells=cells, n_lifetimes=args.n_lifetimes,
                    n_episodes=args.n_eval_episodes, max_steps=args.episode_max_steps,
                    device=device, sgb=sgb, env_offset=off, deterministic=False,
                    seed=u * 1000 + i))
            out[es.name] = aggregate_lifetimes(res)
        return out

    def fmt_curve(vals, k=(0, 1, 2, 4, 9, 19)):
        return " ".join(f"e{i}={vals[i]:.1f}" for i in k if i < len(vals) and vals[i] == vals[i])

    history = []
    pair_model = RNNAgentAsPairModel(agent)
    t_train = time.time()
    order = np.arange(len(vecs))
    for u in range(1, args.n_updates + 1):
        pick = order[((u - 1) * args.envs_per_update) % len(vecs):][:args.envs_per_update]
        if len(pick) < args.envs_per_update:
            pick = np.concatenate([pick, order[:args.envs_per_update - len(pick)]])
        rollouts = []
        for k in pick:
            v = vecs[k]
            if chunks_done[k] >= args.resample_envs_every:
                # Lifetime boundary: same env, fresh starts/goals, state dropped.
                v.reset_all(); carry_h[k] = None; chunks_done[k] = 0
            # A fresh lifetime starts from explicit zeros rather than None:
            # `bc_rnn_update` refuses to mix chunks that carry a state with
            # chunks that do not, and round-robin over envs at different
            # points in their lifetimes produces exactly that mix. Zeros are
            # what None means, made concatenable.
            h0 = carry_h[k]
            if h0 is None:
                h0 = torch.zeros(acfg.num_rnn_layers, args.batch_envs, acfg.hidden_size,
                                 device=device)
            r = collect_rollout_rnn(
                v, agent, acfg, args.steps_per_rollout, device, deterministic=False,
                sgb=sgb, env_offset=train.offsets[k], carry_across_episodes=True,
                initial_h=h0, episode_max_steps=args.episode_max_steps)
            carry_h[k] = r.final_h; chunks_done[k] += 1
            rollouts.append(r)
        agent.train()
        losses = bc_rnn_update(agent, rollouts, cfg.bc, opt, args.movement_mode)
        agent.eval()

        if u == 1 or u % args.eval_every == 0 or u == args.n_updates:
            eps = float(torch.cat([r.episodes_completed for r in rollouts]).float().mean())
            goal_rate = float(np.mean([r.goal_reached.mean().item() for r in rollouts]))
            tables = eval_all(pair_model, sets, acfg, cells, args.movement_mode, device,
                              n_per_quadrant=args.eval_pairs, seed=u)
            line = (f"u={u:5d} loss={losses['move_loss']:.4f} eps/chunk={eps:.2f} "
                    f"goal_rate={goal_rate:.3f} | R1 train tt={tables['train'][('train','train')]['model']['metric']:.1f}")
            for es in sets[1:]:
                line += f" {es.name} tt={tables[es.name][('train','train')]['model']['metric']:.1f}"
            rec = {"update": u, "loss": losses["move_loss"], "eps_per_chunk": eps,
                   "goal_rate": goal_rate, "tables": jsonable(tables)}
            if u % args.lifetime_every == 0 or u == args.n_updates:
                lt = lifetime_eval(u)
                rec["lifetime"] = lt
                for name, agg in lt.items():
                    line += f" | R2 {name}: {fmt_curve(agg['by_episode'])}"
            history.append(rec)
            print(line + f" | {time.time()-t_train:.0f}s", flush=True)
            if wandb_run is not None:
                log = {"update": u, "train/loss": losses["move_loss"], "train/eps_per_chunk": eps,
                       "train/goal_rate": goal_rate}
                for es, agg in tables.items():
                    for (s_, g_), row in agg.items():
                        log[f"r1/{es}/{s_}x{g_}"] = row["model"]["metric"]
                if "lifetime" in rec:
                    for name, agg in rec["lifetime"].items():
                        for i, v in enumerate(agg["by_episode"]):
                            if v == v:
                                log[f"r2/{name}/ep{i}"] = v
                wandb_run.log(log)
        if u % args.ckpt_every == 0:
            path = os.path.join(args.save_dir, f"life_u{u}.pt")
            torch.save({"agent_state_dict": agent.state_dict(), "cfg": asdict(cfg),
                        "argv": vars(args), "input_dim": D, "update": u}, path)
            run_manifest.record_checkpoint(args.save_dir, os.path.basename(path), update=u)

    final = eval_all(pair_model, sets, acfg, cells, args.movement_mode, device,
                     n_per_quadrant=None, seed=args.n_updates)
    lt = lifetime_eval(args.n_updates)
    print("\n=== FINAL readout 1 (enumerated, h=0)")
    for es in sets:
        print(format_table(final[es.name], args.movement_mode, title=es.name)); print()
    print("=== FINAL readout 2 (by episode in lifetime, sampled)")
    for name, agg in lt.items():
        print(f"{name:12s} " + " ".join(f"{v:5.1f}" for v in agg["by_episode"]))
        print(f"{'':12s} ep0_step0={agg['ep0_step0']:.1f}   R1 tt={final[name][('train','train')]['model']['metric']:.1f}")
    with open(os.path.join(args.save_dir, "final_tables.json"), "w") as f:
        json.dump({"final": jsonable(final), "lifetime": lt, "history": history,
                   "argv": vars(args), "cells": cells.summary(), "input_dim": D,
                   "params": n_params}, f, indent=1)
    path = os.path.join(args.save_dir, "life_final.pt")
    torch.save({"agent_state_dict": agent.state_dict(), "cfg": asdict(cfg),
                "argv": vars(args), "input_dim": D, "update": args.n_updates}, path)
    run_manifest.record_checkpoint(args.save_dir, "life_final.pt", update=args.n_updates)
    run_manifest.finish(args.save_dir)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
