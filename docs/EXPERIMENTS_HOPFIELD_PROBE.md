# Encoder-Hopfield probe — results

Running log. The harness is `analysis/hopfield_probe/`, specified in
`ENCODER_HOPFIELD_EVAL.md` (tests) and `ENCODER_HOPFIELD_EVAL_VIZ.md` (report).
Everything below is 2026-08-27, `Npos=1716`, `size=20`, 8 worlds × 5 scored
envs, `steps` 1–15, `K` 1–20, four encoders.

---

## 0. Summary

> ## ▸ Read §10.11 first
>
> **The mechanism, and the open questions.** Everything in §10 reduces to one
> variable — the code's **effective dimension** `d_eff`, the participation ratio
> of its covariance. Far-field cosine spread is `1/√d_eff` to within a few
> percent over a 20× range, so aliasing is a tail event in a `d_eff`-dimensional
> space; `attract_lambda` and the coding-rate term compete to set it; and
> `r_min` fails because it is a **product of two opposing functions of `d_eff`**
> and therefore peaks at a different value than reach does.
>
> §10.11 also lists the **four things this does not derive** — including the
> res90 ≈ 5 floor that bounds the entire trade, which is the least understood
> number in this document.
>
> **Best encoder:** `sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt`
> at its own gain 100 with `β = gain`. Continuous reach **0.987**
> (0.977 across three scaffolds × four seeds) against level 7's 0.806.
> Spec sheet: https://claude.ai/code/artifact/db70ecb9-ca16-4f8b-a897-5dfa0a01d198


**Production's Hopfield is a linear matched filter that degrades cues.** At the
live operating point a cue corrupted to cos 0.70 comes back *worse* — 0.54 at
K=5, 0.27 at K=10 — and the `q` readout decays monotonically with recall depth.
Both are properties of the linear regime, not of the architecture.

**Two independent conditions have to hold at once**, and every earlier attempt
moved one:

| condition | knob | production | what it buys |
|---|---|---|---|
| loop gain / saturation | `β` *or* storage `scale` (same knob twice) | `β·s` 0.004–0.10, tanh arg `u` 1e-4–3e-3 | a nonzero fixed point instead of decay |
| patterns near a hypercube corner | encoder `gain` only | cos-to-binarisation 0.81–0.90 | that the fixed point is *your* memory |

Raising `β` alone is a **net loss** (§2). Raising both together works, unevenly
(§3). The rescue sweep finds settings with genuine basins that hold to K=10
(§4).

**The rule that survived:** raise inference gain until cos-to-binarisation
≈ 0.96 — encoder-specific in gain (v35 ~100, L7 ~300), universal in `cos_bin` —
**and** saturate `β` alongside it. Not "use gain 300", not "raise β" alone, and
**not** the rescue sweep's `alpha` optimum on its own — alpha is a time
constant that only means anything once the loop gain makes the recall term
comparable to the cue (§4.2–4.3, §7).

**Cross-talk, not coding radius, is what the readout reads** (§4.5). At
identical architecture and seed, `attract_lambda` 16→32 leaves `r_min` flat,
raises worst-pair overlap 0.578→0.699, and loses on every probe metric —
retrieval hardest, from 57.8% to 38.8%.

**Best measured setting: v35 at gain 100 with β=1e6** (§3.1) — acc45 100.0% at
every step count from 1 to 15, retrieval 97.5%, reach 99.6%, for 1.2° of mean
angular error against production.

**And the reach objective decomposes** (§10). Continuous reach is per-environment
and nearly binary: a goal dies when **one** co-stored competitor crosses cos
≈ 0.25, that competitor sits ~370 cells away, and at K=1 no encoder loses a
single environment. `r_min` cannot see any of it — six candidates share
`r_min` 12.0 while their far-field alias rate spans seven-fold — and the rate
of distant pairs above 0.25 predicts measured reach at **ρ = −0.92**, and
**−0.85 once the confirmation run's eight arms are added** (§10.8).

**The screen's nomination was run and it won** (§10.8), and then §10.9 found a
better arm needing no inference override at all: `w52_attract_fwhm/*_att0.5` at
its own gain 100 with **β = gain**. Reach rises monotonically as
`attract_lambda` *falls* — 0.806 / 0.931 / 0.972 / 0.987 at 16 / 2 / 1 / 0.5 —
which is the axis w52–w54 spent three waves climbing the other way because
`r_min` rewards it.

**But §10.10 puts a floor on how finely any of this can be read.** The same arm
varies **0.959–0.988** across three scaffold draws, so `att0.5`'s lead over
Level 6 at gain 300 is +0.009 on average, winning two draws and losing one.
Large orderings hold; gaps under ~0.02 between the top arms do not resolve on
one scaffold, and §10.8–§10.9 quote them to three decimals as though they did.

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

On this pair `unique_radius` and `q` accuracy **agree**. v35 has the better
radius — `r_min` **16**, named as the best of the 407-checkpoint audit in
`EXPERIMENTS_UNIQUE_RADIUS.md` §1 — and it also leads every probe metric:

| | `r_min` | \|err\| | acc45 | exact | reach | pairwise cos |
|---|---|---|---|---|---|---|
| v35 | **16** | **8.81°** | **99.4%** | **74.3%** | **87.2%** | **0.005** |
| L7-s42 | 9.0 | 11.81° | 97.0% | 57.8% | 78.4% | 0.027 |

> **This reverses an earlier claim in this document.** It previously read
> "`unique_radius` does *not* rank encoders the way `q` accuracy does — v35
> leads every probe metric with the worse published `r_min`." That was
> unsupported: v35's checkpoint carries no stored `unique_radius`, and rather
> than look the number up I inferred a worse radius from v35 predating the
> campaign. The doc had 16 on record the whole time.

Two encoders is not a trend, and they differ on more than radius: v35 has 8×
the parameters and **5× lower cross-talk** (worst pair 0.264 vs 0.524). Since
it leads on radius *and* orthogonality, this cannot separate which one the
readout is actually reading — and §4/§7 argue the binding constraint is
cross-talk, which enters `unique_radius` only through the alias ceiling.
A param-matched pair that differs on one of the two would settle it —
**§4.5 is that pair**, and the answer is cross-talk.

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

| encoder | loop gain `β·s` | tanh arg `u` | K=3 | K=5 | K=10 |
|---|---|---|---|---|---|
| v35 | 0.0036 | 1.1e-4 | +0.698 | +0.543 | **+0.269** |
| L7-s42 | 0.098 | 3.1e-3 | +0.718 | +0.443 | **+0.218** |
| L7-s43 | 0.098 | 3.1e-3 | +0.700 | +0.436 | **+0.222** |

**Best passing cell** — holds to K=10:

| encoder | K=3 | K=5 | K=10 |
|---|---|---|---|
| v35 | 10 / α0.1 / **+0.980** | 10 / α0.1 / **+0.977** | 10 / α0.1 / **+0.968** |
| L7-s42 | 10 / α0.1 / +0.979 | 10 / α0.1 / +0.969 | 1 / α0.5 / **+0.947** |
| L7-s43 | 10 / α0.1 / +0.971 | 10 / α0.1 / +0.962 | 1 / α0.5 / **+0.939** |
| untrained | none passed | none passed | none passed |

Recovery pooled by the sweep coordinate `β·s·√D` peaks in its **1–10** band
for all three trained encoders (0.72–0.76) and falls at both ends. Two routes
reach it — `β=100, scale=1/D` and `β=1, scale=1/√D` — which is the
`(p → λp, β → β/λ²)` invariance showing up in the data.

> **Units, corrected.** These tables originally called `β·s·√D` the "tanh
> argument". It is **D times** the per-coordinate argument `u = β·s/√D`, so the
> 1–10 band is `u` = 1e-3 to 1e-2 — firmly **linear**, not partial saturation
> as the name implied (§7). The coordinate is monotonic in `β·s` so every
> ranking here stands; only the regime label was wrong. Later runs emit
> `loop_gain` and `tanh_u` instead.

**The L7 pair is already inside the optimal band at β=gain=100**, so
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

§4 said the L7 pair's binding knob is `alpha`, since their loop gain is
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

### 4.3 The alpha sweep — a time constant, not a destroyer

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
alpha optimum always came paired with a higher loop gain, and why lifting alpha
out of that pairing (§4) was meaningless.

### 4.4 β sets step-invariance and nothing else

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

## 4.5 attract_lambda 16 vs 32 — cross-talk is the binding variable

`w54_attract_far` extends level 7's `attract_lambda` from 16 to 32. It has 12
finished encoders and appears nowhere in `EXPERIMENTS_UNIQUE_RADIUS.md`.
Architecture is identical to att16 — λ [11,12,13], out 1024, hidden 256, 0.572M
params, gain 100, fwhm 0.25 — so paired by seed, `attract_lambda` is the **only**
thing that differs. That is the controlled comparison the v35/L7 pair could not
be (§1).

| | \|err\| | acc45 | exact | basin | reach | acc45 @ s=15 |
|---|---|---|---|---|---|---|
| att16-s42 | **11.81°** | **97.0%** | **57.8%** | **11.03** | **78.4%** | **76.5%** |
| att32-s42 | 15.00° | 94.2% | **38.8%** | **6.35** | **58.0%** | 62.9% |
| att16-s43 | **11.84°** | **96.9%** | 60.9% | 11.35 | 76.9% | 77.2% |
| att32-s43 | 14.63° | 94.7% | 58.8% | 11.25 | 73.9% | 72.0% |

**att32 is worse on every metric on both seeds**, and the radius gain that
motivated it evaporates when paired: `r_min` 12→12 and 15→14. The 4-seed median
advantage quoted from the stored metrics (13 vs 12) came from seeds 44/45; seed
variance within att16 (12 to 15) is larger than the arm difference.

### The two failure modes read different statistics of cross-talk

Direction degrades on both seeds, retrieval only on s42 — which looks like noise
until sorted by **worst-pair** overlap rather than mean:

| | mean pairwise | worst pair | \|err\| | exact |
|---|---|---|---|---|
| att16-s43 | 0.0188 | 0.440 | 11.84° | 60.9% |
| att16-s42 | 0.0275 | 0.578 | 11.81° | 57.8% |
| att32-s43 | 0.0337 | 0.575 | 14.63° | 58.8% |
| att32-s42 | 0.0387 | **0.699** | 15.00° | **38.8%** |

`exact_hit` is monotone in the **worst pair** with a cliff past ≈0.6;
`|err|` is monotone in the **mean**. That is what §7's two conditions predict:
retrieval is an argmax, so it fails when the single worst competitor beats the
self term in condition (b); direction is an average, so it tracks average
interference. s43 survives because its worst pair stays under the cliff even
though its mean rose 79%.

**So cross-talk, not coding radius, is what the readout reads.** §1 could not
separate them because v35 led on both. Here radius is flat and cross-talk is
41% worse, and every probe metric follows cross-talk.

Practical: **`attract_lambda` 16 → 32 is a regression for navigation** even
where `r_min` calls it flat or better. The arm worth trying instead is
`w54_attract_far`'s `rep0.25`, whose alias ceiling (0.833) is the lowest in
w53/w54 — the direction that lowers the worst pair.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/att16_vs_att32/`.

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

## 7. When is it an attractor network? The analysis

Clean in both limits, not in the middle. `regime_check.py` checks the algebra
against the measurements.

**The regime parameter.** For unit-norm patterns `W = s(ZᵀZ − diag)` and
`‖Wx‖ ≈ s` for a cue with support in the memory subspace, so per *coordinate*
the tanh argument is

    u = β·s/√D          transition at u ≈ 1, i.e. β·s ≈ √D

At production's `s = 1/D` that is **β ≈ D^1.5 = 32 768**. The linear-regime
prediction `‖tanh(βWx)‖ ≈ β·s` and the saturated cap `√D` match measurement to
three significant figures:

| | `u` | predicted | measured |
|---|---|---|---|
| v35 production | 1.1e-4 | 0.0033 | 0.0034 |
| L7 production | 3.1e-3 | 0.0879 | 0.0881 |
| v35 g100+β1e6 | 30.5 | 32.0 (`= √D`) | 31.96 |

**Linear limit, `u ≪ 1`.** `x ← normalize(Wx)` is power iteration: the only
fixed points are eigenvectors of `W`, and only the leading one is stable.
Stored patterns are **saddles**. There is exactly one attractor and it is not a
memory. Production sits here, at `u` = 1e-4 to 3e-3.

**Saturated limit, `u ≫ 1`.** `x ← sign(Wx)/√D`, so a stored `z₁` is a fixed
point iff both:

- **(a)** `z₁` is itself a corner, `z₁ = sign(z₁)/√D` — an *encoder* property,
  measured as cos-to-binarisation;
- **(b)** `sign(Wz₁) = sign(z₁)` — a *capacity* property. For corner patterns
  `Wz₁ = s[(1 − K/D)z₁ + Σ_{k≠1} c_k z_k]`, and a coordinate flips when the
  cross-term beats the self-term there. Union-bounding over `D` coordinates
  gives capacity ≈ **D/(2 ln D) = 74** at D=1024.

That is the two-condition structure of §0 *derived* rather than observed, and
(b) predicts a number it was not fitted to: `nav_p2` §5.7 measured capacity
between **50 and 100**.

**What the analysis does not cover.**

1. The intermediate regime `u ≈ 1`. No clean treatment; partial saturation.
2. **The continuous cell bank.** Retrieval is scored against non-corner cell
   embeddings, so a binarised recall is far from *all* of them. This is what
   collapsed `exact_hit` in §2 while the dynamics behaved exactly as predicted,
   and the fixed-point analysis says nothing about it. It is the single largest
   gap between "is an attractor network" and "is a useful position readout".
3. The angular cost of raising encoder gain (§3.1) — that is the geometry of
   the encoder's chart, not the Hopfield.
4. The correlation penalty on (b). Overlap is measured (`alias_ceiling` 0.88)
   but its effect on capacity is not derived.

## 8. Where things live

Raw result JSON, permanent:
`$CLS_RESULTS/hopfield_probe/20260827/` — ten arms plus a `README.md` naming
each. `report.build <dir>` regenerates a page from any of them; nothing is
recomputed, so restyling costs no recall.

The derivation in §7 is also written up as a standalone page, source at
`analysis/hopfield_probe/report/analysis_page.html`.

## 9. Reproducing

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
`decouple_check.py`, `corner_check.py`, `delta_scale_check.py`,
`traj_compare_check.py`, `regime_check.py` (Sec 7's algebra vs
measurement).

Sec 10's measurements: `dead_goal_check.py` (per-env reach, dead fraction vs K),
`dead_overlap_check.py` (which goal dies, and why not distance),
`dead_threshold_check.py` (where the overlap line sits),
`alias_lattice_check.py` (the module-realignment falsification),
`gain_tradeoff_check.py` (alias rate against res90),
`nav_screen_check.py` (the pool screen), `nav_screen_validate.py`
(the screen against measured reach), `load_curve_check.py`
(dead fraction against load, on a constant env set),
`l6_confirm_check.py` (Sec 10.8's out-of-sample test). The run itself is
`run_l6.sh`.

## 10. What sets continuous reach — the synthesis

Written 2026-08-30, stepping back from the arms above to ask what would make a
**10% coverage, `exclude_cross_env_pairs=True`** encoder best on *continuous*
reach. Re-analysis of the archived arms plus new measurements over the encoder
pool. Nothing here needed a new probe run.

> **The `reach` column in §1–§4.5 is the discrete rate.** The continuous one is
> the stated objective and it differs — usually higher, because sub-cell steps
> escape sinks a cardinal classifier falls into, but *lower* at the top
> (`v35 g100+β1e6`: 0.996 discrete, 0.974 continuous).

### 10.1 Reach is per-environment and nearly binary

`Scalars` keeps one value per scored environment, so the distribution is
recoverable from the archive. It is a mixture, not a spread:

| arm, K=5 | mean | envs at ≥0.95 | envs <0.5 | worst |
|---|---|---|---|---|
| v35 g100+β1e6 | 0.974 | 90% | **0%** | 0.56 |
| production v35 | 0.867 | 80% | 12% | 0.003 |
| production att16-s42 | 0.806 | 72% | 20% | 0.003 |
| att32-s42 | 0.631 | 53% | 40% | 0.003 |
| β1e6 alone, att16-s42 | 0.525 | 38% | 47% | 0.003 |

Reach against **start distance is flat** — att16-s42 scores 0.83 from cells
adjacent to the goal and 0.77 from 21 cells away. So the failures are not
accumulated error and not a last-half-cell precision problem: in a fifth of
environments the goal is unreachable *from the cell next to it*.

### 10.2 Every failure is interference between stored goals

At **K=1 the dead fraction is 0.00 for every arm**, including the worst. It
then climbs with load — but the load axis needs care, because `scored_envs`
measures `min(K, n_score_envs)` environments, so K=1 and K=3 score a *smaller
env set* than K≥5. That is the same population-grows-with-K confound
`n_score_envs` was added to kill on the retrieval axis, and it reaches the K<5
columns anyway. Fixed by selecting a constant env subset at every K
(`load_curve_check.py`; all 8 worlds, since this needs reach values only):

| dead fraction, envs 0–2, n=24 | K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| **v35 g100+β1e6** | 0.00 | **0.00** | **0.00** | **0.00** | **0.00** |
| v35 g300+β1e6 | 0.00 | 0.04 | 0.04 | 0.04 | 0.00 |
| v35 production | 0.00 | 0.12 | 0.17 | 0.12 | 0.33 |
| att16-s42 g300+β1e6 | 0.00 | 0.21 | 0.17 | 0.17 | 0.42 |
| att16-s42 production | 0.00 | 0.17 | 0.25 | 0.29 | 0.75 |
| att16-s43 production | 0.00 | 0.17 | 0.25 | 0.33 | 0.71 |
| att32-s42 production | 0.00 | 0.21 | 0.46 | 0.42 | 0.75 |
| att16-s42, β1e6 alone | 0.50 | 0.50 | 0.54 | 0.58 | 0.71 |

**The encoder changes the slope, not the intercept.** Every arm starts at 0.00
and they diverge, so this is a per-competitor probability that differs by
encoder rather than a fixed penalty. Inverting `P(dead) = 1 − (1−p)^(K−1)`:

| implied per-competitor `p` | K=5 | K=10 | K=20 |
|---|---|---|---|
| v35 g100+β1e6 | <0.01 | <0.01 | <0.01 |
| v35 g300+β1e6 | 0.011 | 0.005 | <0.01 |
| v35 production | 0.045 | 0.015 | 0.021 |
| att16-s42 g300+β1e6 | 0.045 | 0.020 | 0.028 |
| att16-s42 production | 0.069 | 0.038 | 0.070 |
| att32-s42 production | 0.142 | 0.058 | 0.070 |

`p` spans about seven-fold, it is roughly K-stable within an arm, and it tracks
the far-field alias rate of §10.5 at a **ratio of 1.6–3.0** across every arm —
which is what identifies the screen as measuring exactly this quantity. (The
ratio exceeds 1 because "dead" is reach <0.5, which catches partial collapse
as well as outright aliasing.)

So the two factors are **`p` from the encoder and K in the exponent**.
Saturation does not merely lower the rate: `v35 g100+β1e6` is **flat at 0.00
across the whole range to K=20** — condition (b) of §7 paying off, and the
capacity bound `D/(2 ln D) = 74` says that should hold well past 20.

### 10.3 A goal dies when one co-stored competitor crosses cos ≈ 0.25

`multi_env_goals` stores the first K goals of a world, so the competitors are
computable from the checkpoint. Worst overlap separates dead from live at
**AUC 0.83–1.00** (`dead_overlap_check.py`), and a single threshold
misclassifies 0–2 of 20 (`dead_threshold_check.py`, best split 0.18–0.36):

| encoder | max_cos, dead | live | min_sep, dead | live | AUC |
|---|---|---|---|---|---|
| att16-s42 | 0.327 | 0.090 | 383 | 350 | 0.96 |
| att16-s43 | 0.362 | 0.067 | 371 | 354 | 0.89 |
| att32-s42 | 0.325 | 0.103 | 365 | 358 | 0.83 |
| att32-s43 | 0.386 | 0.077 | 371 | 354 | 1.00 |

**Real-space separation predicts nothing** — 383 against 350, and the wrong
sign. The killer sits ~370 cells away: far-field aliasing, hundreds of cells
outside any coding radius. It is one competitor and not an average, which is
§4.5's argmax-versus-mean split again.

> **Falsified: the killers are not on the module lattice.** The obvious story —
> aliases where a grid module realigns exactly — looks strong in the 20 probe
> pairs and dies at scale. Of 3,861 far pairs, 42% of the high-overlap ones
> have an exact module alignment against a **40% base rate**
> (`alias_lattice_check.py`). That is evidence against λ being the visible
> lever, contrary to `EXPERIMENTS_UNIQUE_RADIUS.md` §10.3 item 4.

### 10.4 `r_min` is priced for a different problem

Six constrained candidates carry the **same stored `r_min` of 12.0** while
their far-field alias rate spans seven-fold. And the campaign's last axis moved
the wrong way on purpose: `attract_lambda` 8 → 64 buys res90 12 → 18, exactly
what §4.4b's law rewards, and pays alias rate 0.0123 → 0.0610.

Inference gain is the same trade on a fixed checkpoint
(`gain_tradeoff_check.py`) — att16-s42, gain 1 → 1000: far-field alias rate
0.111 → 0.0094 (12×), res90 22 → 8.

**So `r_min` and continuous reach pull in opposite directions along both of the
axes that have been swept most recently.** `r_min` prices res90 and the alias
ceiling as comparable factors; navigation reads the ceiling almost alone,
because the Gram-Schmidt basis only needs one-cell neighbours.

### 10.5 A screen that costs seconds

If dead goals are goals with a competitor above ~0.25, the rate of *distant*
pairs above that line should predict reach. Against the arms where reach was
measured — nine encoder × setting combinations, two encoder families, gain 3.7
to 300, β 100 to 1e6 — **Spearman = −0.917, p = 0.0005**
(`nav_screen_validate.py`). The threshold came from §10.3, not from this fit.

Over the pool (`nav_screen_check.py`, seed 42, res90 as the local guard):

| encoder | `r_min` | far>.25 @100 | res90 | far>.25 @300 | res90 |
|---|---|---|---|---|---|
| *v35 (out of brief)* | — | *0.0036* | *8* | *0.0031* | *6* |
| *w21 arena-spread ×2 (out of brief)* | *5.0* | *0.0009* | *3* | *0.0009* | *3* |
| L5 `sm50_b4096` | 12.0 | 0.0043 | 6 | 0.0042 | 5 |
| **L6 `eps1_rate0.5`** | 12.0 | 0.0088 | 10 | **0.0057** | **8** |
| w54 `rep0.25` | 12.0 | 0.0106 | 12 | 0.0083 | 9 |
| w53 `att8` | 12.0 | 0.0123 | 12 | 0.0094 | 10 |
| **L7 `att16` (headline)** | 12.0 | **0.0170** | 14 | 0.0125 | 11 |
| w54 `att32` | 12.0 | 0.0269 | 16 | 0.0170 | 12 |
| w54 `att64` | 12.0 | 0.0610 | 18 | 0.0388 | 13 |

The ranking is unchanged at any threshold from 0.15 to 0.35.

### 10.6 What to do

1. **Select on the alias rate, not `r_min`**, with a res90 floor. Free, and it
   ranks a pool where `r_min` is constant.
2. **Best available legal encoder: Level 6 at inference gain 300**, with
   saturated β. Nominated off the screen at (0.0057, res90 8), the only
   constrained arm inside v35's box — **and confirmed in §10.8**: continuous
   reach 0.981 over four seeds, zero dead goals at K=3/5/10 on every seed.
   Unsaturated it already reaches 0.931, which beats level 7 *with* the full
   treatment.
3. **Reverse the last two waves.** `attract_lambda` down (8 beats 16 beats 32
   beats 64; 4/2/1 untested), `repel_weight` down (`rep0.25` already beats
   `att16`, and `EXPERIMENTS_UNIQUE_RADIUS.md` §10.2 flagged the axis).
4. **Re-open §5.6l — its verdict was priced in `r_min`.** The arena-spread
   diagnostic scores **0.0009**, the best measured anywhere, four times better
   than v35. It was dismissed because `r_min` stayed at 6, all of the gap being
   res90. It is out of brief (the spread term sees positions outside the
   patches), so treat it as the bracket — and if res90 6 is survivable, the job
   becomes reaching that ceiling legally.
5. **Settle the res90 floor.** Direction is fine at res90 8, 11 and 14 and
   collapses (10.0° → 19.2°) at 6 — four points from two encoders, and
   everything above is bounded by it. A gain sweep with the full probe at each
   step replaces it with a curve.
6. **Scoping the memory to the current environment is a fallback, not a
   lever.** At K=1 every encoder solves every environment, so it would work —
   but §10.2 shows the encoder route already gets there without it: the good
   arm is flat at 0.00 through K=20. K sets the exponent, the encoder sets the
   base, and only one of the two is free of cost. Worth keeping in mind if a
   configuration turns out to need loads far beyond 20, where §7's capacity
   bound of 74 starts to bind.

### 10.7 What this rests on

- The 0.25 line is 20 goals per encoder across four encoders; the *ranking* is
  threshold-insensitive but the number is soft.
- The res90 = 8 floor is four data points. §10.6 item 5 exists for this.
- **v35 is a target, not a candidate**: 20.4% coverage, 60 patches,
  `hidden_dim` 1024, 4.64M params, no cross-env constraint. It shows the box is
  reachable, not that it is reachable under the brief.
- One scaffold, one seed per screened arm, K ≤ 20. The dead-goal *joins* need
  goal positions, so they use the four worlds serialised into the archive
  (n = 20 per encoder); the load curves need only reach values and use all
  eight (n = 24).
- Raising inference gain changes every embedding: none of this is a config edit
  for a trained policy.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/l6_production/` and
`l6_g300_sat/`.

Page: https://claude.ai/code/artifact/d65ad39b-eb99-4ae3-808c-52223bec874a

### 10.8 The confirmation run — Level 6 probed, and it is the best in the campaign

§10.6 item 2 nominated `w49_g100_knee/*_eps1_rate0.5` off the screen, with a
prediction on record before the run: continuous reach 0.92–0.95 saturated,
0.86–0.91 unsaturated. Four seeds, both arms, settings identical to every
archived arm.

| K=5, s=1 | \|err\| | acc45 | exact | basin | disc | **cont** | acc45 @ s=15 |
|---|---|---|---|---|---|---|---|
| **L6 production** (β=gain=100) | | | | | | | |
| s42 | 7.13° | 99.9% | 90.4% | 17.57 | 0.948 | 0.951 | 92.9% |
| s43 | 7.57° | 99.2% | 88.0% | 18.43 | 0.939 | 0.910 | 92.6% |
| s44 | 6.33° | 99.9% | 90.3% | 18.70 | 0.957 | 0.960 | 95.1% |
| s45 | 8.32° | 99.2% | 85.6% | 17.55 | 0.901 | 0.880 | 87.2% |
| *median* | | | | | | **0.931** | |
| **L6 gain 300 + β1e6** | | | | | | | |
| s42 | 9.09° | 100.0% | 97.4% | 20.38 | 1.000 | 0.986 | 100.0% |
| s43 | 9.20° | 99.8% | 90.2% | 18.23 | 0.997 | 0.976 | 99.8% |
| s44 | 8.59° | 100.0% | 93.1% | 18.90 | 1.000 | 0.964 | 100.0% |
| s45 | 9.15° | 99.4% | 93.6% | 19.25 | 0.974 | 0.987 | 99.3% |
| *median* | | | | | | **0.981** | |

Reference, same settings: att16 production 0.806, v35 production 0.867,
att16 g300+sat 0.890, v35 g100+sat 0.974.

**Both predictions were too low, in the same direction.** The screen ranked
correctly and the mapping from alias rate to reach was pessimistic, which is
what calibrating it on worse encoders would do.

Three things this settles.

**Level 6 unsaturated already beats every constrained arm ever measured** —
0.931 against att16's 0.806 at the same settings, and against att16's 0.890
*with* the full gain-300 + saturation treatment. It leads every other column
too: 6.3–8.3° against 11.8°, retrieval 85.6–90.4% against 57.8%, basin 17.6–18.7
against 11.0. So the campaign's step from level 6 to level 7 cost about 0.12 of
continuous reach, and `r_min` scored that step as an improvement (floor 7 → 9).

**And it does this in the linear regime.** `tanh` argument max **0.0044**, four
orders below the `u ≈ 1` transition, loop gain `β·s` = 0.098 — identical to
att16 production, the arm §0 calls a matched filter that degrades cues. What
level 6 has is condition (a) *baked in at training time*: it is trained at final
gain 100 (§6.10h). Since §6 established retrieval is a one-step jump rather
than a relaxation, a linear filter makes that jump fine when the patterns are
separable enough, and saturation is what stops the state drifting afterwards.

**Saturation still adds, and it is not redundant.** 0.931 → 0.981, and it flattens
the load curve outright: dead fraction at K=20 goes 0.29–0.50 → 0.00–0.12, and
**all four saturated seeds have zero dead goals at K=3, 5 and 10**, matching
`v35 g100+β1e6`. So both conditions earn their place — (a) from the encoder, (b)
from the loop gain.

#### The screen, out of sample

Adding the eight new arms to §10.5's nine:

| | ρ | p | n |
|---|---|---|---|
| original nine arms | −0.917 | 0.0005 | 9 |
| the eight L6 arms alone | −0.762 | 0.028 | 8 |
| **all seventeen** | **−0.853** | **0.00001** | 17 |

It held. The degradation from −0.917 is the expected cost of an out-of-sample
test rather than a fit.

Its rank prediction at the very top did **not** resolve, though: L6 g300+sat has
a *worse* alias rate than `v35 g100+sat` (0.0054–0.0062 against 0.0036) and
scores at least as well (0.981 against 0.974). v35 is one seed and 0.974 sits
inside L6's four-seed range of 0.964–0.987, so this is a tie rather than a
reversal — but the screen has no resolving power left at this level, and it
should be read as a filter, not a ranking, below about 0.006.

> **A §10.2 claim does not survive.** That section reported `p` tracking the
> alias rate at a ratio of 1.6–3.0 over seven arms. The eight new arms give
> 0.60, 1.09, 1.70 and 2.82 for the unsaturated seeds, and the saturated ones
> have `p = 0` exactly, so there is no ratio to take. `p` at K=10 is estimated
> from 24 samples and is quantised in steps of 0.005, which the original seven
> arms did not make obvious. The honest statement is that `p` is within a factor
> of about three of the alias rate, not that it tracks it.

#### What binds now

At the top the failure mode has moved off aliasing entirely. The saturated arms
post **discrete reach 1.000 with continuous 0.964–0.987**, and continuous reach
from cells adjacent to the goal is 0.995–0.998. So 1.4–3.6% of starts reach the
goal *cell* and never get within 0.5 of the goal *point*.

That is the sub-cell approach, and `ARRIVAL_RADIUS = 0.5` against a float
position is as much a property of the measurement as of the encoder. Anything
further on this arm should establish which before spending runs on it.

### 10.9 Selecting on the nav objective: `attract_lambda` **down**, and gain is spent

§10.8 confirmed the screen's nomination. This round asks what is left, and the
answer changes the recommendation: the best encoder is not Level 6 at all, and
it needs no inference override.

Everything below is at **β = gain** — no saturation anywhere.

#### The pool screen

`nav_screen_all.py` walks every arm in w39/w45–w54 at gains 100/300/1000. Two
things fall out immediately.

**The spread term does nearly all the far-field suppression.** `w48 rate0` —
the same config with the coding-rate term off — has an alias rate of **0.2059**
against 0.004–0.06 for everything else. A 25–50× effect, and the largest number
in the screen. That is §5.6j's mechanism confirmed from the other side.

**And `attract_lambda` is monotone the wrong way.** All eight arms share one
base config, so this is a clean axis:

| `attract_lambda` | 0.5 | 1 | 2 (L6) | 4 | 8 | 16 (L7) | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| alias @ g100 | 0.0059 | 0.0072 | 0.0082 | 0.0095 | 0.0123 | 0.0170 | 0.0269 | 0.0610 |
| res90 @ g100 | 7 | 9 | 10 | 11 | 12 | 14 | 16 | 18 |
| `r_min` | 6.0 | 10.0 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 |

w52 → w53 → w54 climbed from 2 to 64 because `r_min` rewards res90. On the
alias rate that is a 7× regression, and the untested direction was down.

> **A one-seed reading retracted.** `fwhm_ratio` 0.5 looked 15% better than
> Level 6 at seed 42. Over four seeds it is 0.0080 against 0.0082 — nothing,
> and Level 6's own seed range (0.0071–0.0088) is wider than the effect
> (`fwhm_seeds_check.py`). §6.9's "0.25 and 0.5 are equivalent" holds on this
> metric too. `steps3x` survives at gain 100 (0.0073 vs 0.0082) and vanishes at
> gain 300, where every arm converges near 0.0055.

#### The gain ladder: an interior optimum, not a floor

Four seeds of Level 6, β = gain (`run_ladder.sh`):

| L6 | res90 | alias | \|err\| | exact | **cont** | dead @ **K=1** |
|---|---|---|---|---|---|---|
| g100 | 10 | 0.0082 | 7.3° | 88% | 0.931 | 0.00 |
| **g300** | 8 | 0.0056 | 9.0° | 96% | **0.971** | 0.00 |
| g1000 | 6 | 0.0052 | 19.3° | 99% | 0.954 | 0.03 |
| g3000 | 5 | ~0.005 | 41.7° | **100%** | **0.608** | **0.34** |

Gain 3000 collapses to 0.608 while retrieval hits 100% and the basin maxes out:
the Hopfield finds the right goal every time and the readout cannot use it. The
`K=1` column is the clean diagnostic — no cross-talk is possible there, so those
failures are pure local chart.

**It is an optimum because the alias rate saturates.** res90 8 → 6 → 5 buys
0.0056 → 0.0052 → ~0.005 while `|err|` goes 9° → 19° → 42°. Past gain 300 the
trade stops being a trade.

> **The `res90 ≥ 8` guard of §10.6 was calibrated on the wrong quantity** —
> angular error, not reach — and it is also too high. The cross-encoder ladder
> has `att0.5` fine at res90 7 and `L5` at 0.977 with res90 6; only `rate3` at
> res90 3 dies (`|err|` 80.5°, reach 0.077, **dead fraction 1.00 at K=1**). The
> real floor is nearer 4–5. It lands on the right answer for the wrong reason,
> because the alias rate flattens at the same place.
>
> It also conflated two failures. An encoder *trained* into a short chart is
> fine (`L5`, res90 6, 0.977); an encoder *read through* a chart it was not
> trained for degrades much faster (L6 forced to res90 6 by gain, 0.954, and to
> res90 5, 0.608).

#### `att0.5` is the best encoder measured

Four seeds, own gain 100, β = 100 — no override of any kind
(`run_att_low.sh`):

| | \|err\| | acc45 | exact | basin | **cont** (4 seeds) |
|---|---|---|---|---|---|
| **`att0.5`** | 7.0–8.7° | 99.5–100% | 96.2–98.9% | 20.7–21.4 | **0.987** (0.984–0.993) |
| `att1` | 6.5–7.6° | 99.6–100% | 90.6–96.9% | 18.5–20.0 | 0.972 (0.963–0.993) |

`att0.5` has the tightest seed spread in the campaign. `att1`'s 0.993 was one
lucky seed — the fifth time a one-seed reading has not survived here.

| | cont | needs |
|---|---|---|
| **`att0.5` @ own gain 100** | **0.987** | nothing |
| L6 @ g300 + β1e6 | 0.981 | override + saturation |
| `att1` @ g100 | 0.972 | nothing |
| L6 @ g300 | 0.971 | override |
| L6 @ g100 | 0.931 | nothing |
| att16 (level 7) @ g100 | 0.806 | nothing |

Reach rises monotonically as `attract_lambda` falls — 0.806 / 0.931 / 0.972 /
0.987 at 16 / 2 / 1 / 0.5 — while `r_min` falls from 12.0 to 6.0.

**Attract and inference gain are substitutes.** Each arm's optimal gain falls as
its `attract_lambda` falls: `att2` and `att1` want 300, `att0.5` wants 100
(0.987 at g100 against 0.977 at g300). Both knobs shorten the chart, so an
encoder that spent the budget in training gains nothing by spending it again at
inference — which is why the best config is also the simplest.

**Saturation is unnecessary.** L6 at g300 scores 0.971 with β = gain against
0.981 saturated, a gap smaller than either arm's seed spread. What saturation
buys is step-invariance (s15 0.90–0.96 against 0.99–1.00), and the policy reads
s=1.

Raw: `$CLS_RESULTS/hopfield_probe/20260827/` — `l6_g300`, `l6_g1000`,
`l6_g3000`, `ladder_g100`, `attlow_g100`, `attlow_g300`, plus
`screen_all_seed42.txt` and `screen_fwhm_steps_4seed.txt`.

### 10.10 The scaffold is a variance source, and it was never varied

Every reach number in §1–§10.9 comes from `--seed 0`: one draw of 8 worlds × 20
envs. Encoder-training seed spread has been measured repeatedly here; the
probe's own world and goal sampling never had been. Two more draws, on the two
leading arms, four encoder seeds each (`run_probeseed.sh`):

| probe seed | `att0.5` @ g100 | L6 @ g300 | leader |
|---|---|---|---|
| 0 | 0.987 | 0.971 | att0.5 |
| 1 | 0.988 | 0.961 | att0.5 |
| **2** | **0.959** | **0.972** | **L6** |
| mean | 0.978 | 0.968 | att0.5, +0.009 |

> **§10.9 overstated this.** It said `att0.5`'s worst encoder seed beats L6's
> median "on either draw", which was true of draws 0 and 1 and was written as a
> property of the arms. Draw 2 breaks it: `att0.5`'s worst is 0.939 against L6's
> median 0.972. The defensible statement is that **`att0.5` leads on two of
> three scaffolds by ~0.02 and trails on the third by 0.013**, averaging +0.009
> — a real but small edge, inside scaffold variation.

**The variance itself is the result.** One arm swings **0.959–0.988** across
three world draws, which is larger than most of the differences §10.9 ranks. So:

- the large orderings are safe — level 7 at 0.806 against everything else near
  0.97 survives any draw;
- **gaps of 0.01–0.02 between the top arms are not resolvable on one scaffold**,
  and §10.8–§10.9 report them to three decimals as though they were.

Anything meant to separate the leaders needs all three draws. `n_worlds` 8 is
also low; raising it is the cheaper fix than repeating draws, and the probe
costs ~3 min per encoder either way.

Raw: `att0.5_ps1`, `att0.5_ps2`, `l6_g300_ps1`, `l6_g300_ps2`.

### 10.11 The mechanism: one variable, two metrics, two different optima

**This is the section to read if you read one.** §10.1–§10.10 are a sequence of
measurements; this is the account that makes them a single thing, and it is what
makes the result transferable to a config nobody has run.

#### Effective dimension is the variable

`why_attract_check.py`, one seed per arm, 4000 random positions. **Effective
dimension** is the participation ratio of the code covariance,
`(Σλ)² / Σλ²` — how many of the 1024 output directions the code actually
occupies:

| arm | eff dim | PR/D | far-cos sd | **1/√d_eff** | alias rate |
|---|---|---|---|---|---|
| `att16` (level 7) | 131 | 0.128 | 0.0825 | 0.0874 | 0.0194 |
| `att4` | 193 | 0.189 | 0.0675 | 0.0720 | 0.0109 |
| `att2` (level 6) | 211 | 0.206 | 0.0659 | 0.0688 | 0.0114 |
| `att1` | 235 | 0.230 | 0.0639 | 0.0652 | 0.0106 |
| **`att0.5`** | **297** | 0.290 | 0.0577 | 0.0580 | 0.0083 |
| `att0.25` | 345 | 0.337 | 0.0534 | 0.0538 | 0.0083 |
| `rate0` (no spread term) | **16** | 0.016 | 0.2259 | 0.2500 | 0.2115 |

**The far-field cosine spread is `1/√d_eff` to within a few percent over a 20×
range.** Distant pairs behave like random vectors in a `d_eff`-dimensional
space, which is what §10.3's "one competitor above 0.25" is a tail event of.

#### Why `att0.5` works

`attract_lambda` and the coding-rate term **compete for `d_eff`**. Attract asks
nearby positions to be similar, so the code varies slowly and collapses into few
directions; the rate term pushes it to spread over the sphere. Lowering attract
shifts the balance: 16 → 0.25 raises `d_eff` 131 → 345. Removing the rate term
altogether collapses the code to **16 of 1024 directions**, which is the whole
explanation of `rate0`'s 0.21 alias rate.

The chain to reach is then:

1. a goal dies when **one** co-stored competitor exceeds cos ≈ 0.25 (§10.3);
2. competitor cosines are ≈ `N(0, 1/√d_eff)`;
3. so raising `d_eff` pushes that tail down fast, and the dead-goal rate with it.

`att0.5` sits where `d_eff` is high enough that the tail is nearly empty and not
so high that the chart is too short to build a local basis from.

#### Why `r_min` is the wrong signal

§4.4b's law is `r_min ≈ res90 · √(ln(1/C)/ln(1/0.9))` — **a product of two
opposing functions of `d_eff`**. Raising `d_eff` shortens the chart (res90 down,
first factor down) and suppresses aliases (`C` down, second factor up). So
`r_min` has an **interior optimum in `d_eff`** and peaks around `att2`–`att16`,
while reach keeps improving to `att0.5`. One underlying variable, two metrics,
two different peaks — and w52–w54 climbed toward `r_min`'s peak for three waves.

And navigation does not need res90 the way `r_min` does. The readout is
`W_basis @ (recalled − current)` with the basis built from **±1-cell**
neighbours, so the chart only has to be valid at one cell. res90 7 leaves the
1-cell cosine at ~0.998. `r_min` maximises res90 because it asks a *decoding*
question — how far can a position move and stay identifiable; navigation asks a
*gradient* question and needs res90 only as a floor.

#### What this explains that was previously just observed

* **Every knob is the same trade** (§10.9) — attract, gain, `rate_lambda` and
  patch count all move `d_eff`, so all of them buy alias rate with res90.
* **Attract and inference gain are substitutes** (§10.9) — same variable, so a
  budget spent in training cannot be spent again at inference.
* **Combinations do not compound** (§10.10) — `a0.5_sm30` is worse than either
  ingredient because you cannot move one variable twice and then trim back.
* **res90 and the alias rate are rank-correlated across every arm measured** —
  they are two readouts of `d_eff`.

#### Open questions — three things this does *not* derive

1. **Why `attract_lambda` sets `d_eff` quantitatively.** The qualitative account
   is solid and the relation is monotone over six arms across a 64× range, but
   there is no derivation of `d_eff(attract_lambda)`, so the optimum has to be
   found empirically for any new geometry or loss.
2. **Where the 0.25 threshold comes from.** It should fall out of the argmax
   against the self-term at given K — the same condition (b) algebra as §7 — but
   it was fitted from 20 goals per encoder, not derived. Deriving it would give
   a *predicted* dead-goal rate from `d_eff` and K alone, which would remove the
   probe from the loop entirely.
3. **Why direction collapses below res90 ≈ 5.** At res90 5 the one-cell cosine
   is still ~0.996, so the Gram-Schmidt basis should be fine — and yet `|err|`
   reaches 42° (§10.9's gain-3000 row). Something about the basis conditioning,
   or about the `recalled − current` difference vector, degrades faster than the
   raw cosine implies. This is the floor that bounds the whole trade, and it is
   the least understood number in §10.

A fourth, from §10.9: an encoder **trained** into a short chart behaves better
than one **read through** a short chart (`L5` at res90 6 scores 0.977; level 6
forced to res90 6 by gain scores 0.954, and to res90 5 collapses to 0.608). If
`d_eff` were the whole story those would match, so inference gain and training
are not perfectly interchangeable and something beyond `d_eff` distinguishes
them.
