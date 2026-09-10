"""Does an encoder ALIAS between two envs, amplified by repeated stores, kill one?

The continual protocol (analysis/continual/agenthash.py) reported one env of
five -- and two of sixteen -- that never became solvable. The first explanation
on record was the section 15.4 perimeter signature: readout clean, agent held
against the wall. That was WRONG, and this module is what falsified it.

Measured with no policy and no rollouts, so nothing depends on what any agent
learned:

  1. CODE OVERLAP between each env's cell codes and every stored goal. Env 0
     and env 2 alias at 0.33-0.39 where every other pair sits at 0.066-0.089.
  2. GOAL-ABSENT ||q|| per env. Env 2 reads 0.138 mean / 0.207 p90 against
     0.055-0.085 elsewhere -- inside the gate band of DUAL_TRAINING section
     9.3, so the agent standing in env 2 is above the following threshold with
     nothing of its own in memory.
  3. THE FIELD with the env's own goal stored: basin 1.000 from all 400 cells,
     q_accuracy 0.992. The readout is fine, which is what rules the perimeter
     story out.

What is left is MULTIPLICITY, which this module sweeps. `--oracle_store_at_goal`
fires a store on EVERY at-goal step, so an env solved 40/40 writes its goal ~40
times while the failing env managed 1-3. The Hopfield is linear (tanh inert,
recall = one step of power iteration), so W = sum_i p_i p_i^T and k copies scale
that term by k.

RESULT: fine at 2 copies, basin 0.013 at 5, 0.007 at 40. The control env that
does NOT alias holds 1.000 with forty copies in memory, so it is the alias and
not the load. It is symmetric -- forty copies of env 2 break env 0 instead. And
parity rescues it (20:40 -> 1.000), so the failure is the RATIO between two
aliasing patterns rather than the alias itself.

Confirmed end to end at the protocol level: --lock_store_after_goal writes each
goal once, and the dead env goes to 0.85 in its own block and 1.000 thereafter.
"""

from __future__ import annotations

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_world_for_split,
)
from hopfield_nav.rollout.distractors import goal_encoding
from analysis.nav_tri.readout_field import field_over_cells, integrate

def main():
    CKPT = ("/orcd/pool/003/jackking/cls_runs/agent_ckpts/"
            "navigate_navp2_d0_base_s42_22133273/navigate_u725.pt")
    N, DEV = 5, "cpu"

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = N
    enc, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, DEV,
                                      getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    envs, vh, offsets = eval_world_for_split(
        cfg, enc, DEV, ckpt_path=CKPT, split="place=held_out", val_seed=0)
    envs, offsets = envs[:N], offsets[:N]
    R, size = float(cfg.env.goal_radius), envs[0].size
    goals = [goal_encoding(vh, offsets[j], envs[j].goal_location) for j in range(N)]


    def basin(j, mult):
        """mult: {env_idx: n_copies}. Returns goal-basin of env j's field."""
        hop = Hopfield(D, beta=cfg.hopfield.beta, device=DEV)
        for k, n in mult.items():
            for _ in range(int(n)):
                hop.input_memory(torch.from_numpy(goals[k]).float())
        fld = field_over_cells(vh, hop, size, offsets[j], DEV)
        _e, reached, _ = integrate(fld, size, envs[j].goal_location, R)
        return float(reached.mean()), hop.num_memories


    print("Each row: env 2's own goal stored ONCE (what it managed in the")
    print("protocol) against k copies of env 0's goal (which it solved 40/40).")
    print("  %-8s %10s %12s" % ("k(env0)", "n_mem", "env2 basin"))
    for k in (0, 1, 2, 5, 10, 20, 40):
        m = {2: 1}
        if k:
            m[0] = k
        b, nm = basin(2, m)
        print("  %-8d %10d %12.3f" % (k, nm, b))

    print()
    print("CONTROL -- env 3 does not alias with env 0 (foreign overlap 0.089).")
    print("Same sweep; if multiplicity alone were the problem it would break too.")
    print("  %-8s %10s %12s" % ("k(env0)", "n_mem", "env3 basin"))
    for k in (0, 1, 10, 40):
        m = {3: 1}
        if k:
            m[0] = k
        b, nm = basin(3, m)
        print("  %-8d %10d %12.3f" % (k, nm, b))

    print()
    print("SYMMETRY -- env 0 aliases env 2 just as much. Does the reverse break?")
    print("  %-8s %10s %12s" % ("k(env2)", "n_mem", "env0 basin"))
    for k in (0, 1, 10, 40):
        m = {0: 1}
        if k:
            m[2] = k
        b, nm = basin(0, m)
        print("  %-8d %10d %12.3f" % (k, nm, b))

    print()
    print("RESCUE -- give env 2 the same multiplicity as env 0. If parity fixes")
    print("it, the failure is the RATIO, not the alias itself.")
    print("  %-14s %10s %12s" % ("k2 : k0", "n_mem", "env2 basin"))
    for k2, k0 in ((1, 40), (5, 40), (10, 40), (20, 40), (40, 40), (40, 1)):
        b, nm = basin(2, {2: k2, 0: k0})
        print("  %-14s %10d %12.3f" % ("%d : %d" % (k2, k0), nm, b))


if __name__ == "__main__":
    main()
