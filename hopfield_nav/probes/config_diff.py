"""What did e8 do that got 0.545, above the 400-step memoryless ceiling of 0.496?

e8 is the only policy in this project measured above what a memoryless walk can
do, so its recipe is the one existence proof that the RNN can carry visit
history here. Print the knobs this line varies, next to W6's (the best current
explorer) so the differences are the only thing on screen.
"""
import json
import os

A = "/orcd/pool/003/jackking/cls_runs/agent_ckpts"
RUNS = {
    "e8 (0.545)": f"{A}/navigate_explore_min_e8_20110835/run.json",
    "e4L": f"{A}/navigate_explore_min_e4L_20123174/run.json",
    "W6 (0.388)": f"{A}/navigate_ee_W6_20360577/run.json",
}

FLAT = ["steps_per_rollout", "envs_per_world", "batch_envs", "num_worlds",
        "schedule", "goal_reward", "time_penalty", "seed", "hidden_size"]
NESTED = {
    "hopfield": ["novelty_reward", "novelty_anneal", "novelty_scale_remaining",
                 "revisit_penalty", "wall_penalty", "persistence_bonus",
                 "epsilon_explore", "epsilon_anneal_updates", "goal_reward",
                 "time_penalty"],
    "agent": ["init_log_std", "freeze_log_std", "hidden_size", "rnn_cell",
              "rnn_nonlinearity", "input_hopfield_raw", "egocentric_heading"],
    "ppo": ["lr", "clip_coef", "move_ent_coef", "gamma", "gae_lambda"],
    "env": ["size", "observation_size", "wall_resolution", "goal_radius"],
}

cfgs = {}
for name, p in RUNS.items():
    if not os.path.exists(p):
        print(f"{name}: no run.json at {p}")
        continue
    with open(p) as f:
        cfgs[name] = json.load(f)["config"]

names = list(cfgs)


def get(c, path):
    cur = c
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return "-"
        cur = cur[part]
    return cur


rows = [(k, [get(cfgs[n], [k]) for n in names]) for k in FLAT]
for sect, keys in NESTED.items():
    rows += [(f"{sect}.{k}", [get(cfgs[n], [sect, k]) for n in names])
             for k in keys]

w = max(len(r[0]) for r in rows) + 2
print(" " * w + "".join(f"{n:>26}" for n in names))
for key, vals in rows:
    shown = [str(v)[:24] for v in vals]
    mark = "  " if len(set(shown)) == 1 else "* "
    print(f"{mark}{key:<{w-2}}" + "".join(f"{s:>26}" for s in shown))
