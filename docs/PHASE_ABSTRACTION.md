# Is explore/exploit an abstract variable in the RNN? (redo, 2026-10-14)

Replaces the `analysis/phase_decoding` results voided by
`docs/PHASE_DECODING_AUDIT.md`.

## Summary

RESULTS_PENDING

## 1. Method

### 1.1 Models

| name | checkpoint | hidden | trained regime |
|---|---|---|---|
| d0_base | `navigate_navp2_d0_base_s42_22133273/navigate_u725.pt` | 1024, rnn | interleaved explore (no goal event) and exploit (goal preloaded) rollouts |
| task3r | `navigate_navp2_task3r_k2_h128_s42_22883646/navigate_u3000.pt` | 128, rnn | task-faithful: search, oracle store at first touch, teleport, continue; second visit from a fresh state with the memory kept |

Each has a random-weight control: same config, freshly initialised weights
(`torch.manual_seed(0)`).

Arenas: 100 minted at `place=held_out,wall=held_out,goal=held_out`
(`val_seed=0`), the base-validation distribution. The recorded validation
worlds hold only 6 arenas. Policy sampled throughout. Distractors uniform on
0 to 10, as in training.

### 1.2 Two definitions of phase

**A, set by memory content.** Fresh RNN state, random non-goal start. The
Hopfield holds n distractors (explore) or the goal and n distractors, shuffled
(exploit). Nothing is written during a trial. The trial ends on the first step
after which the agent stands on the goal, or at 60 steps. This is
`evaluate_navigation`'s contract, d0_base's exploit rollouts and task3r's
revisit rollouts. 40 trials per condition per arena.

**B, the agent's own switch (task3r only).** Distractors only. On the first
step whose input is at the goal the move is discarded, the oracle writes the
goal-cell pattern, and the agent is teleported (uniform non-goal cell, heading
0, zero displacement) with RNN state and previous reward and action kept.
Later arrivals teleport without writing. Phase 0 is every step before the first
arrival, phase 1 every step after. A **sham** arm is identical except that
nothing is written. Search capped at 300 steps, 60 steps recorded after the
switch. 40 trials per arm per arena.

The policy's store head is never honoured: training ignores it in both
regimes.

### 1.3 Validation of the collector

Before collecting, each run compares the collector with the training eval on
10 arenas × 40 trials at 5 distractors and aborts on a gap over 4 binomial
standard errors.

| run | collector | training eval |
|---|---|---|
| d0_base, A, exploit success (steps) | 0.995 (12.5) | 1.000 (12.2) (`evaluate_navigation`) |
| task3r, A, exploit success (steps) | 0.998 (11.3) | 0.993 (11.4) (`evaluate_navigation`) |
| task3r, B, first-touch success within 300 (steps) | 0.703 (107.7) | 0.723 (108.7) (`evaluate_task`, one visit) |

In the full B collection, store and sham trials find the goal equally often
(69.5% each). In the 60 steps after the switch, store trials reach the goal
4.6 times per trial and sham trials 0.21 times: the write is what makes the
agent exploit.

### 1.4 Features

- `h`: the RNN state after the step.
- `x`: the policy input vector for the step.
- `qn`: the length of each Hopfield readout channel in `x`
  (`hopfield_signal`, `hopfield_multistep_1`). A linear decoder on `x` cannot
  compute it; the exploit policy is known to gate on it.
- `xq`: `x` with `qn` appended.

Features are z-scored over the selected rows; constant features are dropped.

### 1.5 Row selections

Rows whose input is at the goal (definition B arrival steps, where the goal
reward enters the input) are excluded.

- `balanced`: equal rows per phase per arena, at most 1500.
- `matched`: per arena, the same joint histogram of time since the last
  teleport (bins of 3 to 6 steps) and distance to goal (2-cell bins) in both
  phases.
- `matched_dist_ge1`: `matched`, trials with at least one distractor only, so
  explore `q` is never exactly zero.
- `matched_qnorm`: per arena, matched on time and on deciles of the current
  readout length `||q||`, with at least one distractor. Distance is not matched
  here (the `||q||` distributions overlap only partly).
- `matched_abs_t` (B only): matched on time since the trial began, in place of
  time since the last teleport.

Each selection also reports a behaviour-only decoder (time since teleport,
distance to goal, speed, wall contact, distractor count; trained on 80% of
arenas), which measures how much phase the selection leaves predictable from
behaviour.

### 1.6 Metrics

As in Bernardi et al. 2020 (Cell). Code: `analysis/phase_decoding/geometry.py`,
validated on synthetic geometries in `hopfield_nav/tests/test_phase_geometry.py`.

- **Parallelism score.** One coding vector per arena (mean of phase 1 minus
  mean of phase 0), mean cosine over all arena pairs. Null: phase labels
  permuted across whole segments within each arena, 1000 shuffles.
- **CCGP.** L2 logistic regression (C=1, class-balanced) trained on k arenas
  (k = 1, 2, 5, 10, 25, 50), tested on 40 held-out arenas, 10 draws per k.
  Null: the geometric random model (condition centroids replaced by isotropic
  Gaussian points with the same total variance, residuals kept), 10 draws per k.
- **Within-arena decoding.** 5-fold, whole segments held out. The ceiling.

A segment is a (trial, phase) pair; it is the resampling unit throughout.
Nothing is split by timestep.

RESULTS_SECTIONS_PENDING
