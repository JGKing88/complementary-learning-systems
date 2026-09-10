"""Check the analytical regime boundary against measurement.

For unit-norm patterns, W = s*(Z^T Z - diag) and ||W x|| ~ s for a cue aligned
with the memory subspace. Per COORDINATE that is ~s/sqrt(D), so the tanh
argument is

    u  =  beta * s / sqrt(D)

and the regime boundary is u ~ 1, i.e. beta*s ~ sqrt(D). At the production
scale s = 1/D that means beta ~ D^1.5.

The rescue tables reported `tanh_arg = beta * s * sqrt(D)`, which is D times u.
If that is right, the "optimal band tanh_arg 1-10" is u = 1e-3 to 1e-2 -- deep
in the LINEAR regime, not near saturation. Check both against the measured
recall-term norms.
"""
import numpy as np

D = 1024
print(f"D = {D},  sqrt(D) = {np.sqrt(D):.1f},  D^1.5 = {D**1.5:.0f}\n")
print(f"{'case':22s} {'beta':>10s} {'s':>10s} {'u = b*s/sqrt(D)':>16s} "
      f"{'||recall||':>11s} {'measured':>9s}")
print("-" * 84)
CASES = [
    ("v35 production", 3.699, 1 / D, 0.0034),
    ("L7 production", 100.0, 1 / D, 0.0881),
    ("v35 g100+b1e6", 1e6, 1 / D, 31.96),
    ("rescue 'tanh_arg'=1", 1.0, 1 / np.sqrt(D), None),
    ("rescue 'tanh_arg'=10", 10.0, 1 / np.sqrt(D), None),
    ("saturation threshold", D ** 1.5, 1 / D, None),
]
for name, beta, s, meas in CASES:
    u = beta * s / np.sqrt(D)
    # linear prediction ||tanh(bWx)|| ~ beta*s*0.9 ; saturated caps at sqrt(D)
    pred = min(beta * s * 0.9, np.sqrt(D))
    m = f"{meas:9.4f}" if meas is not None else "        -"
    print(f"{name:22s} {beta:10.4g} {s:10.5f} {u:16.4g} {pred:11.4f} {m}")

print("\nu << 1 -> linear (power iteration, one attractor, memories are saddles)")
print("u >> 1 -> saturated (x -> sign(Wx)/sqrt(D); memories fixed iff corners)")
print(f"\nreported tanh_arg = beta*s*sqrt(D) = D*u, so tanh_arg 1-10 is "
      f"u = {1/D:.2e} to {10/D:.2e}")
print(f"capacity bound for random corners, D/(2 ln D) = "
      f"{D/(2*np.log(D)):.0f}  (nav_p2 measured 50-100)")
