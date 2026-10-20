# Is explore/exploit an abstract variable in the RNN? (redo, 2026-10-14)

Replaces the `analysis/phase_decoding` results voided by
`docs/PHASE_DECODING_AUDIT.md`.

## Summary

The claim this supports: the memory circuit supplies a recall-strength signal,
and a trained recurrent network turns it into an arena-invariant
explore/exploit representation with the geometric signatures of abstraction
reported in hippocampus and PFC (Bernardi et al. 2020). The representation is
absent from the raw input and from untrained networks, it integrates recall
strength over the trial, and the same network also builds an abstract task
state the circuit never signals.

- **Abstract, and not present in the input.** Both trained RNNs hold explore
  versus exploit (goal in memory or not) along one arena-invariant direction.
  With behaviour matched, a decoder trained on one arena reads phase in
  held-out arenas at 0.88 to 0.91, against a geometric null at or below 0.60.
  In the raw policy input phase is not abstract (parallelism 0.07 to 0.09;
  0.61 from the current input, 0.75 to 0.80 from a 20-step window).
  Random-weight networks given the same input do not form it (0.53 to 0.61).
- **Its content is recall strength over the trial.** The arena-invariant
  part of the input is the length of the Hopfield readout, `||q||`, a
  nonlinear feature the network has to compute. `||q||` alone decodes phase
  across arenas at 0.93; its history over the trial (20-step window plus
  running mean, max and min) decodes at 0.97 to 0.98, matching `h` (0.97).
  `h` adds nothing on top of that history, and `h` with it regressed out
  falls to 0.53 to 0.56 (section 2.5). d0_base and task3r agree within 0.01.
  This is the representational side of the known policy result that the
  explore/exploit switch is a gate on `||q||`.
- **The network also builds a state the circuit does not supply.** In task3r
  sham trials (the agent reaches the goal and is teleported, memory unchanged)
  `h` keeps an abstract record of the arrival (section 2.6). More than 30
  steps later, when the 20-step input window no longer contains the event,
  `h` still separates before from after arrival across arenas at 0.92 (one
  training arena) to 0.95 (50), against 0.56 to 0.78 for the input window
  and chance for `||q||`, with time since the trial began matched. It lasts
  at least the 60 steps recorded.
- **Not yet shown:** whether abstraction is specific to this circuit. A
  network trained on the same task without the memory circuit (the
  `rollout_rnn.py` baseline) is the comparison that would say so.

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

## 2. Results

Outputs in `$CLS_RUNS/results/phase_abstraction/`: `analysis_v3/`
(balanced, matched, matched_dist_ge1), `analysis_v3_qnorm/`,
`analysis_v3_all/`, `analysis_v3_abs_t/`. Each holds one JSON per run, the
config it ran with, and a `summary.md` with every number. CCGP is given as
"train on 1 arena / train on 50 arenas".

### 2.1 Definition A, every control applied (`matched_all`)

Matched per arena on time (6-step bins), distance to goal (4-cell bins) and
`||q||` quintiles, trials with at least one distractor. Behaviour alone then
decodes phase at 0.54 (d0_base) and 0.56 (task3r).

| run | feature | parallelism (shuffle 95%) | CCGP 1 / 50 | geometric null 95% | within-arena |
|---|---|---|---|---|---|
| d0_base | `h` | 0.63 (0.01) | 0.82 / 0.84 | 0.52 to 0.55 | 0.87 |
| d0_base | `x` + `||q||` | 0.09 | 0.54 / 0.67 | 0.53 to 0.57 | 0.67 |
| d0_base | `||q||` | | 0.55 / 0.51 | 0.52 to 0.58 | 0.58 |
| d0_base random | `h` | 0.01 | 0.54 / 0.60 | 0.51 to 0.52 | 0.54 |
| task3r | `h` | 0.74 (0.01) | 0.80 / 0.85 | 0.53 to 0.59 | 0.87 |
| task3r | `x` + `||q||` | 0.12 | 0.57 / 0.65 | 0.52 to 0.57 | 0.68 |
| task3r | `||q||` | | 0.56 / 0.54 | 0.51 to 0.58 | 0.57 |
| task3r random | `h` | 0.07 | 0.54 / 0.61 | 0.51 to 0.52 | 0.54 |

`h` CCGP is flat in the number of training arenas: one arena is enough.
That is the signature of a factorised geometry, where the phase direction does
not depend on the arena.

Matching on `||q||` was incomplete for the random-weight agents: their `||q||`
features still decode phase at 0.81 to 0.84 in this selection, because they
wander and their readout channels behave differently. The input carries phase
there and their RNN state still does not.

### 2.2 What each control removes (definition A)

CCGP for task3r (d0_base in brackets), train on 1 / 50 arenas:

| selection | behaviour only | `h` | `x` + `||q||` | `||q||` |
|---|---|---|---|---|
| balanced | 0.87 (0.87) | 0.93 / 0.98 (0.96 / 0.98) | 0.82 / 0.92 (0.82 / 0.92) | 0.89 / 0.89 (0.89 / 0.90) |
| matched (time, distance) | 0.56 (0.54) | 0.88 / 0.97 (0.91 / 0.97) | 0.89 / 0.94 (0.86 / 0.95) | 0.93 / 0.93 (0.92 / 0.93) |
| matched_qnorm (time, `||q||`) | 0.85 (0.85) | 0.86 / 0.91 (0.86 / 0.90) | 0.59 / 0.74 (0.58 / 0.71) | 0.51 / 0.52 (0.53 / 0.53) |
| matched_all | 0.56 (0.54) | 0.80 / 0.85 (0.82 / 0.84) | 0.57 / 0.65 (0.54 / 0.67) | 0.56 / 0.54 (0.55 / 0.51) |

Matching behaviour costs `h` about 0.05 at one training arena and nothing at
50. Matching `||q||` removes the input's advantage and leaves `h` above 0.8. With behaviour matched but not
`||q||`, the input with `||q||` decodes phase as well as `h` does: under the
analysis in its usual form, the RNN would add little over its input.

### 2.3 Definition B, the agent's own switch (task3r)

Store and sham trials find the goal equally often (69.5%). After the switch,
store trials reach the goal 4.6 times in 60 steps and sham trials 0.21 times.

| arm | selection | behaviour only | `h` parallelism | `h` CCGP 1 / 50 | `x` + `||q||` | `||q||` |
|---|---|---|---|---|---|---|
| store | matched (time since teleport, distance) | 0.57 | 0.89 | 0.96 / 0.99 | 0.83 / 0.95 | 0.94 / 0.94 |
| store | matched_all | 0.56 | 0.82 | 0.88 / 0.97 | 0.57 / 0.66 | 0.54 / 0.51 |
| sham | matched | 0.57 | 0.99 | 0.94 / 0.97 | 0.56 / 0.57 | 0.51 / 0.52 |
| sham | matched_all | 0.58 | 0.98 | 0.96 / 0.97 | 0.56 / 0.58 | 0.50 / 0.50 |
| sham | matched_abs_t (time since trial start, distance) | 0.89 | 0.98 | 0.94 / 0.96 | 0.59 / 0.61 | 0.51 / 0.52 |

`fig_B_switch` shows a decoder on `h` fit to store rows of 50 arenas and
applied to the other 50, aligned on the first arrival. It sits at -12 logits
before arrival in both arms and jumps to +10 at arrival in both. In store
trials it stays near +10 for all 40 steps shown. In sham trials it falls to
about +1 within 10 steps and crosses zero at about 35. The jump is the arrival
(the goal reward enters the input and the agent is teleported); the lasting
part depends on the write.

The sham result needs care. Post-arrival rows are later in the trial and
closer to their last teleport. Matching time since the trial began leaves
decoding at 0.94, so this is not a clock that runs from the start of the
trial. In that selection, time since the last teleport is unmatched
(behaviour-only 0.89), but the RNN can only know that time by registering the
arrival. The state is an arrival trace.

### 2.5 Is the part beyond `||q||` just past `||q||`? (definition A)

Matching equalises only the current `||q||`; the decoders see one timestep.
`analysis/phase_decoding/history.py` asks whether the input history accounts
for what `h` adds. Rows: `matched` (time since teleport and distance to goal),
all of them. Three measures, each as cross-arena decoding trained on 1 / 10 /
50 arenas:

- input-history baselines: `||q||` or `x` over the last 5, 10 or 20 steps of
  the trial with a lag mask; `q_hist` is the 20-step `||q||` window plus the
  running mean, max and min of `||q||` since the trial began;
- incremental: a spline basis of the `||q||` history alone, then the basis
  plus `h`; the gain is what `h` adds, on every row;
- residual: `h` minus its ridge prediction from that basis. This measures the
  same thing as matching, and rows where `||q||` already settles the phase
  carry no residual signal, which pulls it toward 0.5.

Results at 50 training arenas (1 arena in brackets):

| | task3r | d0_base |
|---|---|---|
| `h` | 0.97 (0.89) | 0.97 (0.95) |
| current `||q||` | 0.93 (0.92) | 0.93 (0.93) |
| `||q||`, last 5 steps | 0.95 (0.94) | 0.96 (0.95) |
| `||q||`, last 20 steps | 0.95 (0.94) | 0.96 (0.95) |
| `||q||` history incl. since-start summaries | 0.97 (0.96) | 0.98 (0.96) |
| `x`, last 20 steps, no norms | 0.80 (0.63) | 0.75 (0.63) |
| basis of current `||q||` → + `h` | 0.93 → 0.97 | 0.93 → 0.97 |
| basis of `||q||` history → + `h` | 0.97 → 0.97 | 0.98 → 0.97 |
| `h` residual given current `||q||` | 0.72 | 0.73 |
| `h` residual given `||q||` history | 0.56 | 0.53 |

Given the current `||q||`, `h` adds 0.04; given the `||q||` history it adds
nothing. The history of the readout length decodes phase across arenas as
well as `h` from 50 training arenas and better from one (0.96 vs 0.89 for
task3r). Most of it is within 5 steps; the since-start summaries (running
mean and maximum) add the rest. The two models agree to within 0.01 on every
measure. The residual of 0.53 to 0.56 is all that is left of `h` beyond the
history. Dilution could make it understate what `h` holds on the ambiguous
rows, but the undiluted incremental measure shows no gain from `h` either.

This says what 2.1 and 2.2's abstract variable contains, not that it is
absent. Phase is not arena-invariant in the raw input (parallelism 0.07 to
0.09); the network computes the readout length, a nonlinear feature, integrates
it over the trial and holds it along a fixed direction. Random-weight networks
with the same input do not.

### 2.6 Arrival memory without a memory write (definition B, sham)

In sham trials nothing in memory changes at the first arrival; the goal
reward enters the input for a step and the agent is teleported. `history.py`
asks how long `h` keeps this, against input windows. Rows matched on time
since the trial began and distance to goal (`matched_abs_t`; `matched` gives
the same picture). Decoders trained on 1 / 10 / 50 arenas.

All post-arrival rows:

| feature | 1 | 10 | 50 |
|---|---|---|---|
| `h` | 0.94 | 0.96 | 0.96 |
| current input | 0.58 | 0.61 | 0.61 |
| input, last 20 steps | 0.67 | 0.82 | 0.84 |
| `||q||`, current or history | 0.50 | 0.51 | 0.52 |

The 20-step window contains the reward event for the first 20 steps after
arrival, so this table favours it. The fair test drops post-arrival rows
closer than 21 (or 31) steps to the arrival, re-matches, and trains and tests
every decoder on the remaining rows only:

| post rows kept | `h` | input, last 20 steps | + `||q||` since start | current input |
|---|---|---|---|---|
| > 20 steps after arrival | 0.92 / 0.95 / 0.95 | 0.58 / 0.74 / 0.78 | 0.57 / 0.75 / 0.78 | 0.59 / 0.63 / 0.64 |
| > 30 steps after arrival | 0.92 / 0.94 / 0.95 | 0.56 / 0.74 / 0.78 | 0.56 / 0.75 / 0.78 | 0.59 / 0.64 / 0.65 |

By steps since arrival (decoder trained on 50 arenas, rows > 30 steps): `h`
0.96 at 31 to 45 and 0.94 at 46 to 60; input window 0.79 and 0.76.

`h` keeps the arrival for at least the 60 steps recorded, in a format that
generalises from one arena (0.92). The input window keeps 0.74 to 0.78 with
many training arenas and 0.56 from one; that residue is most likely the
agent's own changed behaviour after the arrival coming back as input
(previous action, displacement, view), and it is barely arena-invariant.
Neither memory (`||q||` at chance) nor a clock from the trial start (matched)
explains `h`. The arrival is a task event the memory circuit does not signal;
the network builds an abstract state from it.

Store trials, for comparison: `||q||` history decodes post-store rows at 0.98
and `h` at 0.998, as in definition A.

Scope: task3r only (d0_base never trained with a mid-episode arrival).
Results: `analysis_v3_history_b/`, `analysis_v3_history_b_late21/`,
`analysis_v3_history_b_late31/`.

### 2.7 Figures

- `analysis_v3_all/fig_A_d0_A_matched_all.pdf`, `fig_A_t3_A_matched_all.pdf`:
  CCGP against training arenas for `h`, `x`, random-weight `h` and `||q||`,
  with geometric-null 95% lines; parallelism with shuffle 95% lines.
- `analysis_v3/fig_A_*_matched.pdf`: the same without the `||q||` match.
- `analysis_v3/fig_B_switch.pdf`: switch-aligned decoder output, store and
  sham, for decoders on `h` and on `x`.

## 3. Limits

- One seed per model, and one checkpoint each.
- Arenas are minted from the base-validation distribution, not the training
  arenas (task3r trained on 3).
- The parallelism score of the two-dimensional `||q||` feature is near 1 by
  construction (two coordinates of consistent sign) and is not compared with
  the others.
- Matching discards rows: `matched_all` keeps about 5,600 rows per phase out of
  about 45,000 exploit rows, mostly early-trial rows where the two phases
  overlap in distance and `||q||`.
- Definition B cannot match time since trial start and time since teleport
  together; the arrival-trace reading rests on the two selections and the
  switch traces jointly.

## 4. Reproduce

```bash
# collection (validates against the training eval first; aborts on mismatch)
NAME=d0_A CKPT=checkpoint/navigate_navp2_d0_base_s42_22133273/navigate_u725.pt \
  DEFINITION=A MAX_STEPS=60 POST_STEPS=0 \
  sbatch --time=02:00:00 analysis/phase_decoding/run_collect.sh
# (d0_A_random: add RANDOM_AGENT=1; t3_*: the task3r checkpoint;
#  t3_B: DEFINITION=B MAX_STEPS=300 POST_STEPS=60)

# metrics
ONLY=d0_A sbatch --time=06:00:00 analysis/phase_decoding/run_analysis.sh
CONFIG=analysis/phase_decoding/configs/abstraction_v3_all.json \
  OUT=$CLS_RUNS/results/phase_abstraction/analysis_v3_all \
  ONLY=d0_A,d0_A_random,t3_A,t3_A_random,t3_B \
  sbatch --time=08:00:00 analysis/phase_decoding/run_analysis.sh

# figures
python -m analysis.phase_decoding.figures \
  --results $CLS_RUNS/results/phase_abstraction/analysis_v3_all --selection matched_all
```
