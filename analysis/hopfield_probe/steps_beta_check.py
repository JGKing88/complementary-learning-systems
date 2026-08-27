"""Does a saturated recall hold up over steps, or degrade like the linear one?

The catch in "just raise beta": production takes ONE step, and at one step beta
barely matters -- the earlier sweep showed acc45 flat at 99.6% from beta 300 to
1e6. The attractor property only shows up in the multi-step limit, which
production never uses.

So the recipe is only worth anything if the saturated dynamics also *hold* over
steps. In the linear regime they do not: acc45 falls monotonically because every
step is power iteration toward the blend most shared among stored patterns. If
saturation stops that decay, the attractor regime is real and usable. If acc45
still falls, raising beta buys nothing anyone can spend.
"""
import dataclasses
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (
    ProbeConfig, build_memory, load_probe_encoder, local_cells, sample_worlds,
)
from analysis.hopfield_probe.qfield import GridAcc, cell_q_field

R = "/orcd/pool/003/jackking/cls_runs"
enc, cfg_m, g0, fwhm, _ = load_probe_encoder(
    f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/encoder_final.pt")

STEPS = (1, 2, 3, 5, 10, 20)
CASES = [
    ("gain 100, beta 100  (production)", 100.0, 100.0),
    ("gain 300, beta 300  (linear)", 300.0, 300.0),
    ("gain 300, beta 1e6  (saturated)", 300.0, 1e6),
    ("gain 300, beta 1e7  (saturated)", 300.0, 1e7),
]

base = ProbeConfig(Npos=1716, env_size=20, n_worlds=3, n_envs_per_world=5,
                   k_values=(5,), steps=STEPS, n_alias=2000, n_score_envs=3,
                   n_cont_samples=1, n_cont_annulus=0)
base.validate()
worlds = sample_worlds(base)
cells = local_cells(base.env_size)

print("acc45 (%) by recall steps, K=5\n")
print(f"{'case':34s} " + " ".join(f"s={s:<5d}" for s in STEPS))
print("-" * 84)
for name, egain, beta in CASES:
    enc.gain = egain
    field = Field(enc, list(cfg_m.lambdas), fwhm, egain, base.Npos)
    cfg = dataclasses.replace(base, beta_override=beta)
    accs = {s: GridAcc(cfg) for s in STEPS}
    for w in worlds:
        rng = np.random.RandomState(w.seed)
        mem = build_memory(field, w, 5, cfg, rng)
        for e in range(3):
            qf, _c, _b = cell_q_field(field, w, e, mem, cfg)
            for s in STEPS:
                accs[s].add_env(qf[s], cells, w.specs[e].goal)
    row = [accs[s].scalars.to_json()["acc45"]["mean"] * 100 for s in STEPS]
    print(f"{name:34s} " + " ".join(f"{v:6.1f}" for v in row))

print("\nFalling with steps = power iteration eating the readout (the linear")
print("regime). Flat = the saturated dynamics are holding the memory.")
