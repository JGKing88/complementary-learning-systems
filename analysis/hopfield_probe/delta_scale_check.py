"""Why does 10% damping destroy retrieval?

x <- (1-a)x + a*tanh(beta*Wx). The retained-cue term has norm (1-a)*||x|| = 1-a.
The recall term has norm a*||tanh(beta*Wx)||. If the recall signal is much
weaker than the cue, then even a small (1-a) dominates the sum and the endpoint
is mostly the cue -- so retrieval returns the cue's own cell.

Measure the two norms at production settings.
"""
import sys
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np

from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (
    ProbeConfig, build_memory, load_probe_encoder, local_cells, sample_worlds,
)

R = "/orcd/pool/003/jackking/cls_runs"
CASES = [
    ("L7-s42 production", f"{R}/sweeps/w53_attract_knee/004_att16_seed=42/"
     "encoder_final.pt", None, None),
    ("v35 production", f"{R}/encoders/run_20260422_185816/encoder_best.pt",
     None, None),
    ("v35 g100 + b1e6", f"{R}/encoders/run_20260422_185816/encoder_best.pt",
     100.0, 1e6),
]

print("||recall term|| vs ||retained cue|| per step, at alpha=1 scale\n")
print(f"{'case':22s} {'||tanh(bWx)||':>14s} {'ratio to cue':>13s} "
      f"{'(1-a) that ties':>16s}")
print("-" * 70)
for name, path, gain_ov, beta_ov in CASES:
    enc, cfg_m, gain, fwhm, _ = load_probe_encoder(path)
    if gain_ov:
        enc.gain = gain_ov
        gain = gain_ov
    cfg = ProbeConfig(Npos=1716, env_size=20, n_worlds=1, n_envs_per_world=10,
                      k_values=(5,), steps=(1,), n_alias=10, n_score_envs=1,
                      n_cont_samples=1, n_cont_annulus=0,
                      beta_override=beta_ov)
    cfg.validate()
    w = sample_worlds(cfg)[0]
    field = Field(enc, list(cfg_m.lambdas), fwhm, gain, cfg.Npos)
    mem = build_memory(field, w, 5, cfg, np.random.RandomState(0))

    cells = local_cells(cfg.env_size)
    z = field.encoded_state(cells, w.specs[0].offset).astype(np.float64)
    W = mem.hopfield.W.cpu().numpy().astype(np.float64)
    beta = float(mem.hopfield.beta)

    delta = np.tanh(beta * (z @ W.T))
    dn = np.linalg.norm(delta, axis=1).mean()      # ||a * delta|| at a=1
    cue = 1.0                                       # ||x|| after normalisation
    # (1-a) at which the retained cue matches the recall term in norm
    tie = dn / (1.0 + dn)
    print(f"{name:22s} {dn:14.4f} {dn/cue:13.4f} {tie:16.4f}")

print("\n'(1-a) that ties' is the damping at which the retained cue equals the")
print("recall term. Below alpha = 1 minus that, the endpoint is mostly cue.")
