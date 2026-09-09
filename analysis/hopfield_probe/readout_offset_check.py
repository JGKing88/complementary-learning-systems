"""Why the current readout loses the bearing on a binary code -- in closed form.

Both readouts are finite differences of the same scalar field
``s(p) = <zhat, z(p)>``. With ``d_i = z(p+e_i) - z(p)`` and unit codes,
``<d_i, zhat> = s(p+e_i) - s(p)`` and ``<d_i, z(p)> = C(1) - 1``, so

    q_i^now = <d_i_hat, zhat - z(p)> = [s(p+e_i) - s(p) + (1 - C(1))] / ||d_i||

a FORWARD difference plus a constant, while Sec 7.1's (iii-c) is

    q_i^new = [s(p+e_i) - s(p-e_i)] / 2

a CENTRAL difference of the same field. The constant in the first is a
correction calibrated for a QUADRATIC similarity profile:

    continuous, 1 - C(j) = a j^2   ->  q^now propto  2 a r_i          correct
    binary,     1 - C(j) = (2m/D) j -> q^now propto (2m/D)(1 + cos t) WRONG

For the linear profile there is nothing for the constant to cancel, so it
survives as the SAME additive term in both components -- a translation of the
2-vector, not a scaling, which is why it destroys the bearing rather than the
magnitude. The central difference never picks it up.

This script evaluates both readouts on the exact profiles, with no encoder and
no fitting, and checks three numbers against the measured ones:

    acc45 for the binary code under the current readout   vs  0.392 (arm B)
    q_north, goal due north, binary, current readout      vs  0.267 (Sec 10.20)
    acc45 for the binary code under (iii-c)               -- the prediction
"""
from __future__ import annotations

import numpy as np

D, M = 1024, 18.4                 # dims; coords flipped per cell (Sec 10.20)
A = 0.1 / 49.0                    # continuous: 1 - C(j) = a j^2, res90 7


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def profiles(j):
    """1 - C(j) for the continuous (quadratic) and binary (linear) codes."""
    return A * j ** 2, (2.0 * M / D) * j


def readouts(r, kind):
    """(q_now, q_new) at displacement -r from the goal, i.e. goal at +r.

    The agent sits at the origin and the goal is at ``r``; ``s`` at an offset
    ``u`` from the agent is ``C(||r - u||)``.
    """
    def C(u):
        dist = np.linalg.norm(r - np.asarray(u, dtype=float), axis=-1)
        return 1.0 - profiles(dist)[0 if kind == "cont" else 1]

    e = np.eye(2)
    one = profiles(1.0)[0 if kind == "cont" else 1]        # 1 - C(1)
    dnorm = np.sqrt(2.0 * one)                             # ||d_i||
    s0 = C([0.0, 0.0])
    now = np.array([(C(e[i]) - s0 + one) / dnorm for i in range(2)])
    new = np.array([(C(e[i]) - C(-e[i])) / 2.0 for i in range(2)])
    return now, new


def acc45(kind, radius=12.0, n=20000):
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ok_now = ok_new = 0
    for t in th:
        r = radius * np.array([np.cos(t), np.sin(t)])
        now, new = readouts(r, kind)
        ok_now += abs(wrap(np.arctan2(now[1], now[0]) - t)) < np.pi / 4
        ok_new += abs(wrap(np.arctan2(new[1], new[0]) - t)) < np.pi / 4
    return ok_now / n, ok_new / n


def main() -> None:
    print("closed form, no encoder, nothing fitted\n")
    print(f"  {'code':<12s}{'readout':<10s}{'acc45':>8s}   measured")
    for kind, label, meas in (("cont", "continuous", 0.995),
                              ("bin", "binary", 0.392)):
        a_now, a_new = acc45(kind)
        print(f"  {label:<12s}{'current':<10s}{a_now:>8.3f}   {meas:.3f}"
              f"   {'<- arm B' if kind == 'bin' else '<- production'}")
        print(f"  {'':<12s}{'(iii-c)':<10s}{a_new:>8.3f}   "
              f"{'PREDICTION' if kind == 'bin' else ''}")

    # The other anchor: goal due north, current readout, binary code. Sec 10.20
    # measured q_north flat at 0.267 over k = 1..8.
    print(f"\n  q_north, binary, current readout, vs Sec 10.20's 0.267 flat:")
    for k in (1.0, 2.0, 4.0, 8.0, 16.0):
        now, new = readouts(np.array([0.0, k]), "bin")
        print(f"    k={k:>5.0f}   q_now = {now[1]:.4f}   q_new = {new[1]:.4f}")
    print(f"    closed form sqrt(4m/D) = {np.sqrt(4 * M / D):.4f}")

    # And the window that sets acc45 for the binary code.
    th = np.linspace(-np.pi, np.pi, 200000, endpoint=False)
    err = wrap(np.arctan2(1 + np.sin(th), 1 + np.cos(th)) - th)
    ok = np.abs(err) < np.pi / 4
    lo, hi = np.degrees(th[ok].min()), np.degrees(th[ok].max())
    print(f"\n  bearing of (1+cos t, 1+sin t) is within 45 deg of t for "
          f"t in ({lo:.0f}, {hi:.0f}) deg")
    print(f"  window {hi - lo:.0f}/360 = {(hi - lo) / 360:.3f}")


if __name__ == "__main__":
    main()
