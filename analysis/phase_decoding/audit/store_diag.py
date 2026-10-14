"""Why does the collector's exploit fail? Exploit trials via agent_step with
(a) the collector's store-on-fire, (b) stores suppressed, (c) (b) + displacement/heading reset."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/orcd/home/002/jackking/cls")
from analysis.phase_decoding.rollout import RolloutEngine
from hopfield_nav.evaluation.metrics import agent_step, random_start
from hopfield_nav.rollout.distractors import goal_encoding
from hopfield_nav.world.env import at_goal

eng = RolloutEngine(sys.argv[1], num_arenas=10); b = eng.build_bundle()
def run(mode, n_trials=20, max_steps=100):
    rng = np.random.RandomState(0); succ = []; fires = []; steps_all = 0
    for ai, env in enumerate(b.envs):
        off = b.offsets[ai]; goal = env.goal_location
        for _ in range(n_trials):
            hop = eng.make_hopfield()
            hop.input_memory(torch.from_numpy(goal_encoding(eng.vh, off, goal)).float())
            env.set_position(random_start(env.size, goal, rng))
            if mode == "reset":
                env._last_displacement = np.zeros(2); env._heading_rad = 0.0
            h = pr = pa = None; ok = False; nf = 0
            for t in range(max_steps):
                o = agent_step(eng.agent, env, off, eng.vh, hop, h, eng.cfg, eng.device,
                               deterministic=False, goal_local=goal, goal_in_memory=True,
                               prev_reward=pr, prev_action=pa)
                if o["store_action"] > 0.5:
                    nf += 1
                    if mode == "store": hop.input_memory(o["store_embedding"][0])
                h, pr, pa = o["h_rnn"], o["next_prev_reward"], o["next_prev_action"]
                steps_all += 1
                if at_goal(env): ok = True; break
            succ.append(ok); fires.append(nf)
    return {"success": float(np.mean(succ)), "store_fires_per_step": float(np.sum(fires) / steps_all)}
out = {m: run(m) for m in ("store", "nostore", "reset")}
print(json.dumps({"ckpt": sys.argv[1], **out}, indent=1))
