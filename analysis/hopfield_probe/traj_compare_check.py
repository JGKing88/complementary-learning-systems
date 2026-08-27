"""Real-space motion of the recall dynamics, across recall regimes."""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np

from analysis.hopfield_probe.attractor import trajectory_probe
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (
    ProbeConfig, build_cell_bank, build_memory, load_probe_encoder,
    sample_worlds,
)

R = "/orcd/pool/003/jackking/cls_runs"
V35 = f"{R}/encoders/run_20260422_185816/encoder_best.pt"
L7 = f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt"

# label, ckpt, encoder-gain override, beta override, alpha
ARMS = [
    ("v35 production", V35, None, None, 1.0),
    ("v35 g100+b1e6", V35, 100.0, 1e6, 1.0),
    ("v35 g300+b1e6", V35, 300.0, 1e6, 1.0),
    ("L7 production", L7, None, None, 1.0),
    ("L7 alpha 0.9", L7, None, None, 0.9),
    ("L7 g300+b1e6", L7, 300.0, 1e6, 1.0),
]
STEPS = 8
K = 5

print(f"Mean distance from the GOAL, in cells, after each recall step (K={K}).")
print("Step 0 is where the cue starts. Production takes exactly one step.\n")
hdr = f"{'arm':16s} {'start':>6s} " + " ".join(f"{'s'+str(i):>6s}"
                                               for i in range(1, STEPS + 1))
print(hdr)
print("-" * len(hdr))
rows = {}
for label, path, g_ov, b_ov, alpha in ARMS:
    enc, cfg_m, gain, fwhm, _ = load_probe_encoder(path)
    if g_ov:
        enc.gain = g_ov
        gain = g_ov
    cfg = ProbeConfig(Npos=1716, env_size=20, n_worlds=2, n_envs_per_world=10,
                      k_values=(K,), steps=(1,), n_alias=2000, n_score_envs=1,
                      n_cont_samples=1, n_cont_annulus=0,
                      beta_override=b_ov, alpha=alpha)
    cfg.validate()
    field = Field(enc, list(cfg_m.lambdas), fwhm, gain, cfg.Npos)
    acc_goal, acc_step, acc_in = [], [], []
    for w in sample_worlds(cfg):
        rng = np.random.RandomState(w.seed)
        mem = build_memory(field, w, K, cfg, rng)
        bank = build_cell_bank(field, w, K, cfg, rng)
        t = trajectory_probe(field, w, 0, mem, bank, cfg, max_steps=STEPS)
        acc_goal.append(t["goal_dist"])
        acc_step.append(t["step_dist"])
        acc_in.append(t["in_env"])
    rows[label] = (np.nanmean(acc_goal, 0), np.nanmean(acc_step, 0),
                   np.nanmean(acc_in, 0), t["start_dist_mean"])
    gd = rows[label][0]
    print(f"{label:16s} {rows[label][3]:6.2f} " +
          " ".join(f"{v:6.2f}" for v in gd))

print(f"\n\nDistance TRAVELLED by each step, in cells.")
print(hdr.replace("start", "     "))
print("-" * len(hdr))
for label in rows:
    print(f"{label:16s} {'':6s} " +
          " ".join(f"{v:6.2f}" for v in rows[label][1]))

print(f"\n\nFraction still inside the test env.")
print(hdr.replace("start", "     "))
print("-" * len(hdr))
for label in rows:
    print(f"{label:16s} {'':6s} " +
          " ".join(f"{v:6.2f}" for v in rows[label][2]))
