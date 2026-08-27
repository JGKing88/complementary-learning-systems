# Encoder-Hopfield probe — results

Running log. The harness is `analysis/hopfield_probe/`, specified in
`ENCODER_HOPFIELD_EVAL.md` (tests) and `ENCODER_HOPFIELD_EVAL_VIZ.md` (report).
Everything below is 2026-08-27, `Npos=1716`, `size=20`, 8 worlds × 5 scored
envs, `steps` 1–15, `K` 1–20, four encoders.

---

## 0. Summary

**Production's Hopfield is a linear matched filter that degrades cues.** At the
live operating point a cue corrupted to cos 0.70 comes back *worse* — 0.54 at
K=5, 0.27 at K=10 — and the `q` readout decays monotonically with recall depth.
Both are properties of the linear regime, not of the architecture.

**Two independent conditions have to hold at once**, and every earlier attempt
moved one:

| condition | knob | production | what it buys |
|---|---|---|---|
| loop gain / saturation | `β` *or* storage `scale` (same knob twice) | `tanh_arg` 0.12–3.1 | a nonzero fixed point instead of decay |
| patterns near a hypercube corner | encoder `gain` only | cos-to-binarisation 0.81–0.90 | that the fixed point is *your* memory |

Raising `β` alone is a **net loss** (§2). Raising both together works, unevenly
(§3). The rescue sweep finds settings with genuine basins that hold to K=10
(§4).

**The rule that survived:** raise inference gain until cos-to-binarisation
≈ 0.96 — encoder-specific in gain (v35 ~100, L7 ~300), universal in `cos_bin` —
**and** saturate `β` alongside it. Not "use gain 300", not "raise β" alone, and
**not** the rescue sweep's `alpha` optimum on its own — alpha is a time
constant that only means anything once the loop gain makes the recall term
comparable to the cue (§4.2–4.3).

**Best measured setting: v35 at gain 100 with β=1e6** (§3.1) — acc45 100.0% at
every step count from 1 to 15, retrieval 97.5%, reach 99.6%, for 1.2° of mean
angular error against production.

---

## 1. Production baseline

| encoder | gain | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|---|
| v35 | 3.70 | 8.81° | 99.4% | 74.3% | 14.15 | 87.2% | 86.2% |
| L7-s42 | 100 | 11.81° | 97.0% | 57.8% | 11.03 | 78.4% | 76.5% |
| L7-s43 | 100 | 11.84° | 96.9% | 60.9% | 11.35 | 76.9% | 77.2% |
| untrained | 5 | 94.70° | 22.6% | 0.0% | none | 6.9% | 22.6% |

Controls at K=5: oracle 7.2°, local oracle 6.9°, Gram-Schmidt swap 5.9°,
`use_tanh=False` identical to production, `tanh |arg|` max 3.7e-4. The basis is
sound and the recall is linear.

`unique_radius` does **not** rank encoders the way `q` accuracy does — v35 leads
every probe metric with the worse published `r_min` — but it has 8× the
parameters, so capacity is an unresolved confound.

## 2. β alone — a net loss

`--beta 1e6`, each encoder at its own gain.

| K=5, s=1 | \|err\| | acc45 | exact | basin | reach |
|---|---|---|---|---|---|
| v35 | 8.81→10.38 | 99.4→98.7% | 74.3→**43.7%** | 14.15→**5.83** | 87.2→**62.6%** |
| L7-s42 | 11.81→13.23 | 97.0→96.7% | 57.8→**18.6%** | 11.03→**1.75** | 78.4→**45.5%** |
| L7-s43 | 11.84→14.95 | 96.9→94.4% | 60.9→**12.9%** | 11.35→**0.77** | 76.9→**37.0%** |

Buys a partial flattening of the steps decay (L7-s42 76.5→92.6% at s=15) and
costs roughly half the retrieval, most of the basin and a quarter of the flow.
Exactly `EXPERIMENTS_NAV_P2.md` §5.7's prediction: the loop gain "does not make
a continuous vector into a corner", and retrieval is scored against a
*continuous* cell bank, so a binarised recall is far from every cell.

## 3. Both conditions — helps, unevenly

`--encoder_gain 300 --beta 1e6`.

| encoder | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| v35 | 8.81→19.15 | 99.4→95.1% | 74.3→**100.0%** | 14.15→**21.62** | 87.2→**97.4%** | 86.2→95.0% |
| L7-s42 | 11.81→**10.32** | 97.0→**99.7%** | 57.8→**72.6%** | 11.03→**13.85** | 78.4→**80.7%** | 76.5→**99.6%** |
| L7-s43 | 11.84→**10.00** | 96.9→**99.5%** | 60.9→52.3% | 11.35→8.18 | 76.9→**81.0%** | 77.2→**99.3%** |

The steps decay is **gone** — flat 99.6% to s=15 — which is the clearest result
in the set. But it is not uniform:

- **L7-s42** improves on every metric.
- **L7-s43** improves direction and flow, *loses* retrieval and basin. Same
  architecture, same config, different seed, opposite sign. Do not read this
  off one encoder.
- **v35** trades: retrieval and flow improve sharply, angular accuracy falls.
  Partly self-inflicted — gain 300 is L7's optimum, and v35's own sweep puts it
  at ~100 (cos_bin 0.968, `|err|` 7.5°, acc45 100.0%) against 300's 0.989 /
  14.0° / 98.4%.

### 3.1 v35 at its *own* optimal gain — the best result in the campaign

§3 used gain 300 for all four encoders. That is L7's optimum; v35's is ~100
(`gain_probe_check.py`). Re-run at gain 100 with β=1e6:

| v35, K=5 | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| production (g3.7) | **8.81°** | 99.4% | 74.3% | 14.15 | 87.2% | 86.2% |
| g300 + sat | 19.15° | 95.1% | 100.0% | 21.62 | 97.4% | 95.0% |
| **g100 + sat** | 10.03° | **100.0%** | **97.5%** | **20.20** | **99.6%** | **100.0%** |

The 19° in §3 was the wrong gain, not an intrinsic trade — 9° of angular error
bought nothing. At gain 100: **acc45 is 100.0% at every step count from 1 to
15**, retrieval 74.3→97.5%, basin 14.15→20.20, reach 87.2→**99.6%**.

The only regression against production is mean `|err|`, 8.81→10.03°. Note
`acc45` *rises* (99.4→100.0%), so production has ~0.6% of cells beyond 45° and
this has none: the mean is slightly worse while the tail is gone. Strictly
better-behaved distribution.

**One config change buys near-perfect retrieval, +6 cells of basin, +12 points
of reach and complete step-invariance, for 1.2° of mean angular error.** It
changes every embedding, so it needs a training run rather than a config edit,
and it is one encoder at one seed.

## 4. Rescue sweep — real basins exist, and production is nowhere near them

504 cells per encoder: `zero_diag` × `alpha` × `scale` × `β`, with each
encoder's own gain in the grid so β=gain is an explicit anchor row. Recovery is
a cue corrupted to cos 0.70; **below 0.70 means recall made it worse.**

**Production anchor** — degrades cues, worse with load:

| encoder | `tanh_arg` | K=3 | K=5 | K=10 |
|---|---|---|---|---|
| v35 | 0.116 | +0.698 | +0.543 | **+0.269** |
| L7-s42 | 3.12 | +0.718 | +0.443 | **+0.218** |
| L7-s43 | 3.12 | +0.700 | +0.436 | **+0.222** |

**Best passing cell** — holds to K=10:

| encoder | K=3 | K=5 | K=10 |
|---|---|---|---|
| v35 | 10 / α0.1 / **+0.980** | 10 / α0.1 / **+0.977** | 10 / α0.1 / **+0.968** |
| L7-s42 | 10 / α0.1 / +0.979 | 10 / α0.1 / +0.969 | 1 / α0.5 / **+0.947** |
| L7-s43 | 10 / α0.1 / +0.971 | 10 / α0.1 / +0.962 | 1 / α0.5 / **+0.939** |
| untrained | none passed | none passed | none passed |

Recovery pooled by `tanh_arg` peaks in the **1–10** band for all three trained
encoders (0.72–0.76) and falls at both ends. Two routes reach it —
`β=100, scale=1/D` and `β=1, scale=1/√D` — which is the
`(p → λp, β → β/λ²)` invariance showing up in the data.

**The L7 pair is already inside the optimal `tanh_arg` band at β=gain=100**, so
on this test their binding knob looks like `alpha`, not the loop gain — a
different knob from the one this campaign spent most of its time on.

> **Tested, and it does not transfer.** §4.2 runs the full suite at
> `alpha=0.5`: `exact_hit` reads 0.2% at s=1. §4.3 then shows why that is a
> *delay* rather than damage, and why alpha and the loop gain cannot be moved
> independently. Read this section's optima as hypotheses about *this test*,
> not as settings.

### 4.1 The untrained control earns its place

It scores **perfect recovery — +1.000, +0.999, +0.999** — and is correctly
rejected at every cell, because `mean_pairwise_cos` catches the collapse
(pairwise 0.9986, effective rank 1.00 of 25). A collapsed encoder sends every
cue to one vector, so a corrupted cue "recovers" perfectly onto it.

Recovery alone would have named the untrained encoder the best in the study.
This is the third time in this campaign a single attractor statistic was
maximised by a degenerate memory — after collapse and after stasis — and it is
why the criterion is all three of self-consistency, pairwise cosine and
recovery, never one.

### 4.2 Rescue's optimum does not transfer — alpha 0.5 on the full suite

§4 said the L7 pair's binding knob is `alpha`, since their `tanh_arg` is
already in the optimal band. Tested directly — L7-s42, `alpha=0.5`, nothing
else changed:

| L7-s42, K=5 | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| production | 11.81° | 97.0% | 57.8% | 11.03 | 78.4% | 76.5% |
| **alpha 0.5** | 11.75° | 97.1% | **0.2%** | **0.00** | 77.9% | 93.9% |
| both (g300+sat) | **10.32°** | **99.7%** | **72.6%** | **13.85** | **80.7%** | **99.6%** |

Alpha 0.5 delivers what rescue promised on the steps axis (76.5→93.9%) and
leaves direction untouched — and drops `exact_hit` 57.8→0.2% with basin
11.03→**zero** *at s=1*.

> **Overstated, corrected in §4.3.** That is not annihilation, it is delay:
> alpha moves the retrieval peak from s=1 to s=5 or later. Reading a slow
> process at the one step production happens to use made it look broken.

**The flaw is in the rescue test, not in alpha.** With `alpha=0.5` the update
keeps half the *cue*, so the endpoint is a blend of cue and memory. The two
tests start from different places:

- rescue's cue is the goal pattern corrupted to cos 0.70 — already nearly
  right, so damping refines it toward the goal and recovery reads 0.97;
- the task's cue is a *different cell's* embedding, far from the goal, so
  damping keeps the state near where it started and retrieval returns the cue's
  own cell.

Rescue measures **cleanup of a nearly-correct cue**; navigation needs
**retrieval from an arbitrary position**. Damping helps the first and destroys
the second, and the recovery guard cannot see it because from cos 0.70 the
state genuinely does improve.

This is the fourth degenerate route in this campaign and the first the
three-part criterion missed. The lesson is not a fourth criterion: it is that
**rescue's cue distribution is unrepresentative of the task, so its optima are
hypotheses to test on the full suite, never recommendations.**

## 4.3 The alpha sweep — a time constant, not a destroyer

§4.2 said `alpha=0.5` "annihilates retrieval". Wrong, and the trajectory probe
(§6) is what showed it. Sweeping alpha on L7-s42, `exact_hit` by step at K=5:

| alpha | s=1 | s=2 | s=3 | s=5 | s=10 | s=15 |
|---|---|---|---|---|---|---|
| **1.0** (production) | **57.8** | 42.1 | 34.1 | 24.2 | 9.0 | 6.5 |
| 0.9 | 0.7 | 2.8 | 7.4 | **28.3** | 25.4 | 17.3 |
| 0.75 | 0.3 | 0.4 | 1.1 | 2.8 | 17.5 | **28.3** |
| 0.5 | 0.2 | 0.3 | 0.3 | 0.4 | 1.5 | 3.2 *(still climbing)* |

Alpha moves the retrieval **peak later** — s=1 → s=5 → s=15 → past 15 — and
roughly halves it. Reading it at s=1 alone, which is where production lives,
makes a slow process look like a broken one. Still not a win, but the reason
matters: it is a time constant, not damage.

**Why the cliff is at alpha=1.0 exactly.** In
`x ← (1−α)x + α·tanh(β·Wx)` the retained-cue term has norm `(1−α)`, and the
recall term has norm `≈ β/D` in the linear regime (`delta_scale_check.py`):

| | β/D | measured ‖recall‖ | (1−α) at which the cue ties |
|---|---|---|---|
| v35 production | 0.0036 | 0.0034 | 0.0034 |
| L7-s42 production | 0.0977 | 0.0881 | 0.081 |
| v35 g100+β1e6 | *saturated* | **31.96** (= √D) | 0.97 |

So at production the recall signal is **0.3–9% of the cue** — which is just the
loop gain `β·S/D` restated, with `S=1` from unit-norm storage and `β` three
orders below `D=1024`. Any `(1−α)` above that swamps the memory with the cue,
which is why alpha 0.9 already fell off. Saturated, the recall term is capped
at `√D` = 32 and outweighs the cue 32:1, so alpha only becomes a real
integration knob there.

**Alpha and loop gain are not independent.** That is why the rescue sweep's
alpha optimum always came paired with `tanh_arg` 1–10, and why lifting alpha
out of that pairing (§4) was meaningless.

## 4.4 β sets step-invariance and nothing else

v35 at gain 300, β swept over three decades:

| arm | \|err\| | acc45 | exact | basin | acc45 @ s=15 |
|---|---|---|---|---|---|
| g300, β1e3 | 19.57 | 93.8% | 100.0% | 21.62 | 79.4% |
| g300, β1e4 | 19.54 | 93.8% | 100.0% | 21.62 | 83.3% |
| g300, β1e6 | 19.15 | 95.1% | 100.0% | 21.62 | **95.0%** |
| **g100, β1e6** | **10.03** | **100.0%** | 97.5% | 20.20 | **100.0%** |

A 1000× change in β leaves `|err|` (19.57→19.15), `exact_hit` (100.0%) and
basin (21.62) **untouched**, and moves only `acc45 @ s=15` (79.4→95.0%).
Changing gain 300→100 at fixed β moves `|err|` 19.15→10.03.

- **encoder gain** sets angular error *and* retrieval
- **β** sets step-invariance *and nothing else*

Retrieval at s=1 depends on where a single application lands, which the
patterns decide; step-invariance needs the dynamics to have a fixed point,
which needs saturation.

## 5. Caveats

- One scaffold, one seed per encoder, `K ≤ 20`. The two L7 seeds already
  disagree on retrieval sign in §3.
- `β = 1e6` and `alpha = 0.1` are far outside anything run end-to-end.
- An inference-gain change alters every embedding a trained policy was fitted
  to. Nothing here is a config edit for an existing checkpoint; the next step
  is a training run.
- §4's grid never varies encoder gain, so it cannot see §3's regime at all. The
  two sections answer different questions and their optima are not comparable.

## 6. Real-space trajectory — retrieval is a jump, not a walk

`attractor.trajectory_probe` decodes **every step** of the recall trajectory to
its nearest cell and measures the motion in cells. Nothing else in A–D does
this: A decodes only the endpoint, B measures `q`'s angle, D follows the `q`
field with an agent rather than following the recall. Mean distance from the
goal, K=5, start ≈ 8.2 cells:

| arm | s1 | s2 | s3 | s5 | s8 |
|---|---|---|---|---|---|
| v35 production | **0.00** | 0.00 | 0.35 | 0.60 | 1.00 |
| v35 g100+β1e6 | **0.00** | 0.00 | 0.00 | 0.00 | **0.00** |
| L7 production | 0.01 | 0.29 | 0.46 | 1.00 | 1.75 |
| L7 alpha 0.9 | **6.01** | 3.08 | 1.66 | 0.60 | 0.67 |
| L7 g300+β1e6 | 0.35 | 0.50 | 0.50 | 0.50 | 0.50 |

**The first application travels ~10 cells and lands on the goal.** Retrieval is
a one-step jump. Production then *drifts away* — 0.00 → 1.00 by s=8 — which is
the matched-filter decay of §0 made concrete in real space. The saturated arm
lands and **stays at 0.00**: a fixed point, visible as a distance rather than
inferred from a cosine.

And alpha 0.9 turns the jump into a walk — 6 cells out after one application,
arriving around s=5. That is §4.3's correction in one row: the retrieval is not
gone, it has not happened yet.

`ProbeConfig.trajectory_steps` (default 15) runs it on `n_map_worlds`, since it
costs one bank retrieval per step.

## 7. Reproducing

    ./analysis/hopfield_probe/run_probe.sh                      # Sec 1
    ... --beta 1e6                                              # Sec 2
    ... --beta 1e6 --encoder_gain 300                           # Sec 3
    ... --beta 1e6 --encoder_gain 100                           # Sec 3.1
    ... --rescue --skip a --skip bc --skip d --skip controls    # Sec 4
    ... --alpha 0.5 / 0.75 / 0.9                                # Sec 4.2-4.3
    ... --encoder_gain 300 --beta 1e3 / 1e4 / 1e6               # Sec 4.4

Per-encoder diagnostics: `gain_gap_check.py`, `gain_probe_check.py`
(`PROBE_CKPT` / `PROBE_GAINS`), `gain_crosstalk_check.py`, `crosstalk_check.py`,
`localchart_check.py`, `contamination_check.py`, `steps_beta_check.py`,
`decouple_check.py`, `corner_check.py`, `delta_scale_check.py`.
