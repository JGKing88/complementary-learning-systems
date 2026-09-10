"""Per-update, per-regime rollout diagnostics.

`docs/DUAL_TRAINING.md` §7 asks for these and says why: evals run every 25-50
updates and do not split by regime, so the corner trap (D2) has only ever been
seen after the fact. Phase 1 diagnosed it from a finished run -- exploit
installs persistent q-following, in an explore rollout q points at distractors,
the agent drives into a wall. The prediction on record is that `chase_q` rises
BEFORE `edge_frac` does, and no run so far could test that, because nothing
logged either one per update.

Everything here is already computed inside the collector's step loop. This is
bookkeeping, not new computation, and it turns the corner trap from a post-hoc
diagnosis into a time series with an onset.

**`follow_q` and `chase_q` are the same cosine.** `cos(a, q)` is one quantity;
which name it takes is a property of the ROLLOUT's regime, not of the
statistic. So the collector emits `cos_aq` and the trainer labels it -- exactly
the split `behavior_probe` makes between `follow_q` (goal present) and
`chase_q` (goal absent). Their difference is `regime_gap`, the direct measure
of whether the policy knows which regime it is in.

Definitions are pinned to `analysis/nav_tri/behavior_probe.py` so a
training-time number and a probe number mean the same thing:

  * `edge_frac`  -- share of steps on a perimeter cell (uniform is 0.19 at
                    size 20, so this is NOT a number to drive to zero;
                    `p20_e_kcap` sits at 0.061 and covers 12% less).
  * `clip_frac`  -- share of steps with `realized < 0.9 * commanded`. Read it
                    with `realized_mag`: a policy parked past
                    `max_action_norm` reads 1.000 here with no wall involved.
  * `pin_frac`   -- share of ROWS matching §18.7's wall-pin signature,
                    `clip_frac > 0.5` AND realized speed `< 0.5`. This is the
                    clamp-immune one, and it is gate 1 in the plan: 100% of
                    episodes at u25/u50, 0% by u150.
"""
from __future__ import annotations

import numpy as np

# A recall shorter than this carries no usable direction, so including it in
# the cosine would average in the angle of a numerical accident. §5.8 puts the
# goal-absent ||q|| at ~0.086 and goal-present at ~0.30, both far above this.
Q_EPS = 1e-3
# A step the policy did not choose (epsilon, auto-nav) says nothing about
# whether the POLICY follows q, which is the whole question.
ACT_EPS = 1e-6
# behavior_probe.py: `clipped = realized < 0.9 * max(want, 1e-8)`.
CLIP_RATIO = 0.9
# §18.7's pin signature, both conditions required.
PIN_CLIP = 0.5
PIN_SPEED = 0.5


def on_perimeter(pos: np.ndarray, size: int) -> np.ndarray:
    """(B, 2) snapped positions -> (B,) bool, is this a perimeter cell?

    One definition, used by BOTH `wall_penalty` and the diagnostic, so the
    number the reward charges for and the number logged can never drift apart.
    Matches `behavior_probe`'s `edge`.
    """
    p = np.asarray(pos)
    xs, ys = p[:, 0], p[:, 1]
    return (xs == 0) | (xs == size - 1) | (ys == 0) | (ys == size - 1)


class RegimeDiagnostics:
    """Accumulate per-step rollout statistics, summarised at the end.

    One instance per `collect_rollout` call, i.e. per (regime, world) rollout.
    `observe` is called once per timestep with whole-batch arrays.
    """

    __slots__ = ("_n_rows", "_cos_sum", "_cos_n", "_q_sum", "_q_n",
                 "_edge_sum", "_clip_sum", "_cmd_sum", "_real_sum", "_steps",
                 "_row_clip", "_row_real", "_row_n")

    def __init__(self, n_rows: int):
        self._n_rows = int(n_rows)
        self._cos_sum = 0.0
        self._cos_n = 0
        self._q_sum = 0.0
        self._q_n = 0
        self._edge_sum = 0.0
        self._clip_sum = 0.0
        self._cmd_sum = 0.0
        self._real_sum = 0.0
        self._steps = 0.0
        # Per-row, because the pin is an episode property. A rollout where
        # half the rows are pinned and half are free has the same pooled
        # clip_frac as one where every row is half-pinned, and only the first
        # is the §18.7 basin.
        self._row_clip = np.zeros(self._n_rows, dtype=np.float64)
        self._row_real = np.zeros(self._n_rows, dtype=np.float64)
        self._row_n = np.zeros(self._n_rows, dtype=np.float64)

    def observe(
        self,
        *,
        q: np.ndarray,            # (B, 2) recalled displacement, local frame
        action: np.ndarray,       # (B, 2) commanded move, pre-clamp
        realized: np.ndarray,     # (B, 2) displacement the env produced
        at_edge: np.ndarray,      # (B,) bool
        alive: np.ndarray | None = None,     # (B,) bool; None = all alive
        from_policy: np.ndarray | None = None,  # (B,) bool; None = all policy
    ) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        action = np.asarray(action, dtype=np.float64).reshape(-1, 2)
        realized = np.asarray(realized, dtype=np.float64).reshape(-1, 2)
        at_edge = np.asarray(at_edge).reshape(-1).astype(bool)

        B = q.shape[0]
        live = (np.ones(B, dtype=bool) if alive is None
                else np.asarray(alive).reshape(-1).astype(bool))
        pol = (np.ones(B, dtype=bool) if from_policy is None
               else np.asarray(from_policy).reshape(-1).astype(bool))

        qn = np.linalg.norm(q, axis=-1)
        an = np.linalg.norm(action, axis=-1)
        rn = np.linalg.norm(realized, axis=-1)

        # cos(a, q) -- the one statistic that becomes follow_q or chase_q
        # depending on the regime of the rollout it was collected in.
        ok = live & pol & (qn > Q_EPS) & (an > ACT_EPS)
        if ok.any():
            cos = np.sum(action[ok] * q[ok], axis=-1) / (an[ok] * qn[ok])
            self._cos_sum += float(cos.sum())
            self._cos_n += int(ok.sum())

        if live.any():
            self._q_sum += float(qn[live].sum())
            self._q_n += int(live.sum())
            self._edge_sum += float(at_edge[live].sum())
            self._cmd_sum += float(an[live].sum())
            self._real_sum += float(rn[live].sum())
            clipped = rn < CLIP_RATIO * np.maximum(an, 1e-8)
            self._clip_sum += float(clipped[live].sum())
            self._steps += float(live.sum())

            idx = np.flatnonzero(live)
            self._row_clip[idx] += clipped[idx]
            self._row_real[idx] += rn[idx]
            self._row_n[idx] += 1.0

    def summary(self) -> dict[str, float]:
        """Means over live steps, plus the per-row pin fraction.

        An empty rollout returns zeros rather than NaN, matching
        `mean_steps`'s convention (tri finding 8) -- a NaN here would poison
        the wandb series and hide the very updates worth looking at.
        """
        n = max(self._steps, 1.0)
        seen = self._row_n > 0
        if seen.any():
            row_clip = self._row_clip[seen] / self._row_n[seen]
            row_real = self._row_real[seen] / self._row_n[seen]
            pin = float(np.mean((row_clip > PIN_CLIP) & (row_real < PIN_SPEED)))
        else:
            pin = 0.0
        return {
            "cos_aq": self._cos_sum / max(self._cos_n, 1),
            "cos_aq_frac": self._cos_n / n,
            "q_mag": self._q_sum / max(self._q_n, 1),
            "edge_frac": self._edge_sum / n,
            "clip_frac": self._clip_sum / n,
            "cmd_mag": self._cmd_sum / n,
            "realized_mag": self._real_sum / n,
            "pin_frac": pin,
            "steps": self._steps,
        }


def merge(summaries: list[dict[str, float]]) -> dict[str, float]:
    """Pool several rollouts' summaries, weighting by live steps.

    A plain mean over rollouts would weight a rollout that died early the same
    as a full one. `cos_aq` is weighted by the steps that actually entered it
    (`cos_aq_frac * steps`), not by total steps, because it is conditioned on
    a usable recall and a policy-chosen action.
    """
    if not summaries:
        return {}
    tot = sum(s.get("steps", 0.0) for s in summaries)
    if tot <= 0:
        return {k: 0.0 for k in summaries[0] if k != "steps"}

    out: dict[str, float] = {}
    for k in summaries[0]:
        if k in ("steps", "cos_aq"):
            continue
        out[k] = sum(s[k] * s["steps"] for s in summaries) / tot

    cw = sum(s["cos_aq_frac"] * s["steps"] for s in summaries)
    out["cos_aq"] = (
        sum(s["cos_aq"] * s["cos_aq_frac"] * s["steps"] for s in summaries) / cw
        if cw > 0 else 0.0
    )
    out["steps"] = tot
    return out


__all__ = ["RegimeDiagnostics", "merge", "on_perimeter", "Q_EPS",
           "CLIP_RATIO", "PIN_CLIP", "PIN_SPEED"]
