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
then put the loop gain in the `tanh_arg` 1–10 band. Not "use gain 300", and not
"raise β".

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

**The L7 pair is already inside the optimal `tanh_arg` band at β=gain=100.**
Their problem is not the loop gain; it is `alpha=1`. That is a different knob
from the one this campaign spent most of its time on.

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

## 5. Caveats

- One scaffold, one seed per encoder, `K ≤ 20`. The two L7 seeds already
  disagree on retrieval sign in §3.
- `β = 1e6` and `alpha = 0.1` are far outside anything run end-to-end.
- An inference-gain change alters every embedding a trained policy was fitted
  to. Nothing here is a config edit for an existing checkpoint; the next step
  is a training run.
- §4's grid never varies encoder gain, so it cannot see §3's regime at all. The
  two sections answer different questions and their optima are not comparable.

## 6. Reproducing

    ./analysis/hopfield_probe/run_probe.sh                      # Sec 1
    ... --beta 1e6                                              # Sec 2
    ... --beta 1e6 --encoder_gain 300                           # Sec 3
    ... --rescue --skip a --skip bc --skip d --skip controls    # Sec 4

Per-encoder diagnostics: `gain_gap_check.py`, `gain_probe_check.py`
(`PROBE_CKPT` / `PROBE_GAINS`), `gain_crosstalk_check.py`, `crosstalk_check.py`,
`localchart_check.py`, `contamination_check.py`, `steps_beta_check.py`,
`decouple_check.py`, `corner_check.py`.
