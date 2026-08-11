"""Two nearby views are a WARP of each other, and the warp reads out position.

Read-only probe against the production env -- no prototypes, no local copy of
the ray-caster.

Facing North from (x, y), ray i reads W(x + D*tan(theta_i)) with
D = size - 0.5 - y the distance to the wall. So an observation is the wall's
barcode resampled onto the ray-angle axis, and moving reparametrises that
sampling rather than perturbing its values:

    translate by dx   ->  every ray's read slides by dx        (a SHIFT)
    change D          ->  the sampling dilates about the centre (a SCALE)

Two consequences, and they are why this file exists.

1. Zero-lag cosine is the wrong statistic for this env. It is the pixelwise
   correlation of two frames of a panning shot: near zero however similar the
   content. Slide the vectors against each other and the similarity reappears.
   Any claim about this sensory code resting on plain cosine similarity should
   be re-examined -- see docs/sensory_code.md for ones that did and were wrong.

2. The shift is a readout of displacement. Near the cone centre
   du/di = D*sec^2(theta)*dtheta, so a translation dx appears as a lag of
   ~dx/(D*dtheta) ray indices. Cross-correlating two views therefore recovers
   the parallel displacement from the lag and the perpendicular one from the
   dilation -- without localising either view. That is a concrete mechanism for
   "given a view here and a view there, how do I get there", and it is the kind
   of computation an architecture has to be *able* to express: a per-timestep
   linear map over the ray vector cannot.

Usage:
    python -m analysis.scaffold_experiments.warp_structure
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import FOVEAL_HALF_ANGLE_DEG, make_env, raycast_codes

SIZE = 8


def env_at(res, seed, obs):
    return make_env(EnvConfig(size=SIZE, observation_size=obs,
                              wall_resolution=res), "discrete", seed=seed)


def view(env, x, y, n_rays, psi=0.0):
    """One view from a continuous position, through the production ray-caster."""
    return raycast_codes(env._wall_code, env.size, np.array([x]), np.array([y]),
                         np.array([psi]), n_rays, env.wall_resolution)[0]


def best_lag(a, b, max_lag):
    """Lag maximising normalised overlap correlation, and that correlation."""
    best, bl = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            u, v = a[lag:], (b[:len(b) - lag] if lag else b)
        else:
            u, v = a[:lag], b[-lag:]
        if len(u) < 40:
            continue
        c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        if c > best:
            best, bl = c, lag
    return bl, best


def main() -> int:
    n_rays = 240                     # lag resolution; see the note at the end

    print("1. THE STRUCTURE IS THERE, under the right statistic")
    print("   Two cells one apart in x, facing North.")
    print(f"   {'wall_resolution':>16} {'zero-lag cos':>13} {'best-lag cos':>13}"
          f" {'lag':>6}")
    print("   " + "-" * 52)
    for res in (1, 8):
        zs, bs, ls = [], [], []
        for seed in range(8):
            env = env_at(res, seed, n_rays)
            for x in (2.0, 3.0, 4.0):
                for y in (1.0, 2.0, 3.0):
                    a = view(env, x, y, n_rays)
                    b = view(env, x + 1, y, n_rays)
                    zs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
                    L, c = best_lag(a, b, n_rays // 2)
                    bs.append(c); ls.append(L)
        print(f"   {res:16d} {np.mean(zs):13.3f} {np.mean(bs):13.3f} "
              f"{np.mean(ls):6.1f}")
    print("\n   At res=1 that is ~0.02 against ~0.85: apparently unrelated views")
    print("   that are in fact the same content, displaced.")
    print("\n   Note res=8 matches far worse under this test (~0.38). The true")
    print("   warp is a shift PLUS a dilation, and a pure roll cannot express")
    print("   the dilation -- at coarse stripe scale the residual misalignment")
    print("   is sub-stripe and harmless, at fine scale it is not. That is a")
    print("   limit of the roll model here, NOT evidence the structure is gone;")
    print("   a shift+scale estimator is untested. Worth knowing if you pair")
    print("   wall_resolution=8 with a flow-style reader.")

    dtheta = 2 * np.deg2rad(FOVEAL_HALF_ANGLE_DEG) / n_rays

    print("\n\n2. THE LAG IS A DISPLACEMENT READOUT")
    print(f"   predicted lag = dx / (D * dtheta),  dtheta={dtheta:.5f} rad/ray")
    print(f"\n   lag vs dx   (y=2.0, so D={SIZE - 0.5 - 2.0})")
    print(f"   {'dx':>6} {'measured':>9} {'predicted':>10} {'peak corr':>10}")
    D = SIZE - 0.5 - 2.0
    for dx in (0.25, 0.5, 1.0, 1.5, 2.0):
        lags, cors = [], []
        for seed in range(12):
            env = env_at(1, seed, n_rays)
            L, c = best_lag(view(env, 3.0, 2.0, n_rays),
                            view(env, 3.0 + dx, 2.0, n_rays), n_rays // 2)
            lags.append(L); cors.append(c)
        print(f"   {dx:6.2f} {np.mean(lags):9.1f} {dx / (D * dtheta):10.1f} "
              f"{np.mean(cors):10.3f}")
    print("\n   Linear in dx with a constant ~0.75 factor -- the small-angle step")
    print("   drops sec^2(theta), which averages 1.65 over a +/-60 deg cone, so")
    print("   the true ray spacing is wider and the lag correspondingly smaller.")

    print(f"\n   lag vs distance-to-wall D   (dx = 1.0)")
    print(f"   {'D':>6} {'measured':>9} {'predicted':>10} {'peak corr':>10}")
    for y in (1.0, 2.0, 3.0, 4.0, 5.0):
        D = SIZE - 0.5 - y
        lags, cors = [], []
        for seed in range(12):
            env = env_at(1, seed, n_rays)
            L, c = best_lag(view(env, 3.0, y, n_rays),
                            view(env, 4.0, y, n_rays), n_rays // 2)
            lags.append(L); cors.append(c)
        print(f"   {D:6.1f} {np.mean(lags):9.1f} {1.0 / (D * dtheta):10.1f} "
              f"{np.mean(cors):10.3f}")

    print("\n\n3. THREE LIMITS, before leaning on any of this")
    print("   * Confidence decays with distance, though the estimate does not")
    print("     bias: the lag stays on the line out to dx=2 while the peak")
    print("     correlation falls 0.97 -> 0.72. It gets less certain, not wrong.")
    print("   * Degrades near walls, where little wall is in view -- the same")
    print("     regime that fails identifiability (positional_identifiability).")
    print("     The measured/predicted ratio drifts 0.81 -> 0.42 as D goes")
    print("     6.5 -> 2.5, because the small-angle step gets worse as the cone")
    print("     spans more of the wall.")
    print("   * observation_size sets displacement PRECISION, not just")
    print("     identifiability: dx=1 is ~15 lags at 240 rays and ~4 at 60, so")
    print("     at the narrower width nothing finer than ~0.25 cells resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
