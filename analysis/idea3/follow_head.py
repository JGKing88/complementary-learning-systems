"""Would idea 3 return to the goal if it followed its own state? (§10.13)

    python -m analysis.idea3.follow_head --ckpt <navigate_uN.pt> --out <json>

The stacked GRU trained with the displacement head carries a rough direction
to the goal on held-out arenas (the trained head: R^2 0.48, 45 degrees), but
its moves ignore it. This runs ``evaluate_task``'s protocol (two visits per
arena, state carried, sampled) twice on the held-out arenas, with goals
redrawn per group as in ``goal_probe``:

- ``policy``: the agent as trained;
- ``head``: once a row has touched its goal, every move goes at full length
  along the agent's own displacement head (read from the same step's
  features), nothing else changed.

Only the agent's own state steers, so ``head`` is the ceiling of what the
policy could do with the direction it already holds. If ``head`` returns fast,
the remaining failure is that RL did not teach the policy to use it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from analysis.idea3.goal_probe import Recorder  # noqa: F401  (same loader path)
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint, load_agent
from hopfield_nav.rollout import task_stats
from hopfield_nav.rollout.collector import RolloutCollector
from hopfield_nav.world.generate import build_envs
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import WorldSpec


class HeadSteer(torch.nn.Module):
    """The agent; after a row's first goal touch (current reward > 0 in the
    input), its moves follow the agent's own displacement head."""

    def __init__(self, agent, active: bool, speed: float):
        super().__init__()
        self.agent = agent
        self.active = active
        self.speed = float(speed)
        self.found = None
        self.n_steered = 0

    @property
    def rnn(self):
        return self.agent.rnn

    def reset(self):
        self.found = None

    def forward(self, *a, **k):
        return self.agent(*a, **k)

    def get_action_and_value(self, x, h=None, **k):
        if not self.active:
            return self.agent.get_action_and_value(x, h, **k)
        if self.found is None:
            self.found = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        # Steered from the step AFTER the touch: the touch step's move is
        # discarded by the at-goal contract anyway.
        mask = self.found.clone()
        self.found |= x[:, 0, 0] > 0
        with torch.no_grad():
            _, _, _, _, feats = self.agent(x, h, return_features=True)
            d = self.agent.goal_disp_head(feats)                       # (B, 1, 2)
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.speed
        self.n_steered += int(mask.sum())
        return self.agent.get_action_and_value(
            x, h, move_action_override=d, move_override_mask=mask)


def run(agent, cfg, vh, envs, offsets, trials, max_steps, n_groups, seed,
        device, active):
    ecfg = copy.deepcopy(cfg)
    ecfg.batch_envs = int(trials)
    ecfg.steps_per_rollout = int(max_steps)
    ecfg.obs_dropout = 0.0
    ecfg.hopfield.heading_dropout = 0.0
    ecfg.hopfield.novelty_reward = max(float(cfg.hopfield.novelty_reward), 0.3)
    collector = RolloutCollector(vh, ecfg, vh.encoded_Phi.shape[2], device)
    w = HeadSteer(agent, active, cfg.env.max_action_norm).to(device)
    w.eval()
    rng = np.random.RandomState(seed + 7919)
    torch.manual_seed(seed)
    np.random.seed(seed)
    first, later = [], []
    for _g in range(n_groups):
        for j, env in enumerate(envs):
            env.set_goal(tuple(int(v) for v in rng.randint(0, env.size, 2)))
            env.goals_active = True
            w.reset()
            h, fired = None, None
            for v in range(2):
                r = collector.collect_rollout(
                    env, w, [None] * trials, allow_store=True, h_rnn=h,
                    env_offset=offsets[j], task_mode=True, store_fired_init=fired)
                (first if v == 0 else later).append(r.task)
                h, fired = r.h_final, r.store_fired_final
    n_cells = float(envs[0].size ** 2)
    f = task_stats.merge(first, n_cells=n_cells)
    l = task_stats.merge(later, n_cells=n_cells)
    keys_f = ("found_frac", "steps_first", "reaches_post", "steps_per_reach")
    keys_l = ("revisit_found_frac", "revisit_steps_first", "revisit_steps_per_reach")
    out = {k: float(f.get(k, np.nan)) for k in keys_f}
    out.update({k: float(l.get(k, np.nan)) for k in keys_l})
    out["steered_steps"] = w.n_steered
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--world", default=None)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_groups", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    world = a.world or os.path.join(os.path.dirname(a.ckpt), "world.json")
    specs = WorldSpec.read(world).split.base_val
    offsets = [tuple(s.offset) for s in specs]
    encoder, enc_cfg, enc_gain = load_encoder(cfg.encoder_checkpoint, str(device))
    torch.manual_seed(0)
    np.random.seed(0)
    vh = VectorHash(cfg.vectorhash)
    vh.build_scaffold()
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=str(device))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(enc_gain)
    agent = load_agent(cfg, ck["agent_state_dict"], enc_cfg.out_dim, device)
    if getattr(agent, "goal_disp_head", None) is None:
        raise SystemExit("this checkpoint has no displacement head")
    envs = build_envs(specs, cfg.env, cfg.agent.movement_mode)
    res = {"ckpt": a.ckpt}
    for name, active in (("policy", False), ("head", True)):
        res[name] = run(agent, cfg, vh, envs, offsets, a.trials, a.max_steps,
                        a.n_groups, a.seed, device, active)
        print(name, json.dumps(res[name]), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
