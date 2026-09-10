# Encoder-Hopfield eval — reporting layer

Status: **implemented**, 2026-08-27. Companion to `ENCODER_HOPFIELD_EVAL.md`,
which defines *what is measured*. This document defines *what the reader sees*,
and it is deliberately separable: it depends on the test layer only through the
result JSON, and the test layer never imports from it.

The output is a set of **static, self-contained HTML pages** — one per test,
plus an index — generated from saved results. No server, no build step, no
network at render time.

    python -m analysis.hopfield_probe.report.build RESULTDIR

**One change the implementation made.** Heatmap cells carry `data-c="i,j"` and
the values ride in the chart's JSON payload, formatted in JS on hover, rather
than a `data-tip` string per rect. Same tooltip and same `n`; at the sub-cell
map's 25 600 cells the string-per-rect form was megabytes of redundant text in
a file whose whole point is being one thing you can email.

---

## 0. Contract with the test layer

```
analysis/hopfield_probe/
    report/
        __init__.py
        theme.py       # the palette and chrome tokens of Sec 1, in one place
        figures.py     # every chart, as inline SVG strings
        page.py        # HTML shell, the interaction layer, cards and tiles
        build.py       # CLI: python -m analysis.hopfield_probe.report.build RESULTDIR
```

The planned `schema.py` was not built. The result JSON is produced by
`stats.py`'s accumulators, which already have exactly one `to_json` per
quantity, so a second set of dataclasses restating the same shape would be a
copy to keep in sync rather than a contract. `build.py` reads the JSON
defensively instead — a missing test simply drops its page.

Four rules, and they are what make the split worth having:

1. **`report/` imports nothing from `attractor.py` / `qfield.py` / `flow.py`.**
   The tests run headless on a compute node; the pages are built afterwards,
   on a laptop, from JSON.
2. **`build.py` recomputes nothing.** If a number is on a page, it is in the
   JSON. A figure that needs a quantity the tests did not save is a bug in the
   test layer's schema, not something the plotter derives.
3. **Regenerating is idempotent and cheap.** Restyling every page must never
   mean re-running a single recall.
4. **Every page renders standalone.** Inline SVG, inline CSS, no external
   assets, no CDN. A page is one file you can email.

---

## 1. Visual system

The palette is the **reference instance from the `dataviz` skill, used
unchanged**. `theme.py` holds it in one place; no hex appears anywhere else.

> **Validation status.** These are documented-validated configurations of the
> reference palette, not novel ones — the first three categorical slots are
> recorded as passing all-pairs in both modes (CVD ΔE 9.2 light / 9.4 dark),
> and the adjacent pairlist passes for the full eight. `node` is not installed
> on the cluster, so `scripts/validate_palette.js` was **not** re-run here;
> nothing below deviates from the documented instance, so nothing needed it.
> **Any substitution — a brand ramp, a fourth categorical hue, a re-stepped
> ordinal — must run the validator before it ships.**

### 1.1 Slots and what they mean here

| role | encoding | value |
|---|---|---|
| `K` (memory load) | **ordinal ramp**, blue, light→dark | steps 250, 300, 350, 400, 450, 500, 550 for K = 1, 2, 3, 5, 10, 20, 50 |
| `steps` (recall iterations) | **not a color** — facet | small multiples, one panel per `s` |
| encoder identity | categorical slot 1 / slot 2 | v35 `#2a78d6`, L7 `#eb6834` |
| untrained floor | muted ink | `#898781`, 2px dashed |
| oracle controls (§6.1, §6.1b) | secondary ink | `#52514e`, dashed (oracle) / dotted (local oracle) |
| `|err|`, `retr_dist`, `cos_goal` | **sequential**, blue 100→700 | magnitude, one hue |
| signed `err` | **diverging**, blue ↔ red, gray midpoint | `#f0efec` light / `#383835` dark at 0 |
| `other_env` outcome | categorical slot 2 | `#eb6834` |
| `alias` outcome | categorical slot 3 | `#1baf7a` |

**Why `K` is a ramp and not eight hues.** `K` is an ordered magnitude, so the
ordinal ramp is the correct encoding rather than a workaround — the
value-ramp anti-pattern applies to *nominal* categories, which this is not. It
also keeps every chart inside the three-slot all-pairs budget. The ordinal
floors are respected: on the light surface no step lighter than 250, on dark no
step darker than 600.

**Why `steps` is faceted.** Six step counts × seven `K` values is 42 series.
Colour cannot carry both. `K` gets the ramp because it is the axis the reader
compares within a panel; `steps` becomes the panel grid.

### 1.2 Chrome

Surfaces `#fcfcfb` / `#1a1a19`; primary ink `#0b0b0b` / `#ffffff`; muted
`#898781`; hairline grid `#e1e0d9` / `#2c2c2a`. 2px lines, ≥8px markers, 2px
surface gap between adjacent fills, recessive axes. System sans throughout;
`tabular-nums` only in tables and axis ticks.

**Dark mode is selected, not flipped** — its own steps from the same ramps,
declared under both `@media (prefers-color-scheme: dark)` and
`:root[data-theme="dark"]` so a toggle wins in both directions.

### 1.3 Interaction, applied to these pages

Every chart is interactive by default:

- **Line charts** (`|err|` vs. distance, `acc_45` vs. `steps`, `exact_hit` vs.
  distance): vertical crosshair snapping to the nearest distance bin, one
  tooltip listing **every `K` at that bin** — the reader never has to hit a
  line. Values lead, series names follow, series keyed by a short stroke.
- **Heatmaps**: the cell is the hit target, with a ≥24px transparent hit area.
  Tooltip gives `Δ = (dx, dy)`, the value, and **`n`** — the sample count in
  that cell, which is what tells the reader whether to believe it.
- **Every chart has a table view**, collapsed behind a `<details>`. Tooltips
  enhance; they never gate. This is also what makes the pages useful in a
  paper draft — the table is copy-pasteable.
- **One filter row above the content**, scoping the whole page: encoder,
  memory mode, seed. Never per-chart controls.
- Marks lift on hover/focus; keyboard focus shows the same tooltip as pointer.

### 1.4 Rules that apply to every figure on every page

- **`n` is always visible.** Every binned curve carries its per-bin sample
  count — as a faint bar strip under the axis, or in the tooltip and table.
  The near-goal bins are the sparsest and the most interesting, and a curve
  that hides its `n` there is actively misleading (`EVAL.md` §10).
- **Chance is drawn, not described.** 90° mean `|err|`, `acc_45 = 0.25`,
  `acc_90 = 0.5` are hairlines in muted ink with a direct label, on every plot
  where they apply.
- **Controls are reference lines, not series.** Oracle and local-oracle appear
  on the aggregate curves and in the summary table. They do **not** appear on
  heatmap panels, where they would not vary.
- **Spread, not just means.** Anything averaged over worlds carries an IQR
  band or per-world faint lines behind the mean. Per
  `feedback_eval_point_threshold`, a bare mean is not reportable here.
- **No dual axes, ever.** Two measures of different scale become two charts.

---

## 2. Page set

```
index.html          run header, headline stat tiles, cross-test summary table
test_a.html         attractor + basin
test_b.html         q accuracy, grid positions
test_c.html         q accuracy, continuous positions
test_d.html         flow
controls.html       Sec 6 controls, all together
rescue.html         only when --rescue ran
```

Shared furniture on every page: a sticky run header (encoder name, `gain`,
`fwhm_ratio`, `out_dim`, params, `lambdas`/`Npos`, memory mode, seeds, git SHA,
timestamp), the filter row, and a footer linking the result JSON that produced
the page.

---

## 3. `index.html` — the answer in one screen

**Hero row — four stat tiles**, not charts. These are single numbers and a
one-bar bar chart would be the anti-pattern:

| tile | value | sublabel |
|---|---|---|
| Basin radius | `r_exact_95` at production `K`, `s=1` | cells, ± across worlds |
| Direction accuracy | `acc_45` at `s=1`, pooled | vs. 0.25 chance |
| Snap cost | mean `excess` over `d < 2` | degrees added by continuous positions |
| Flow | `reach_rate` at `s=1` | fraction of starts that arrive |

Each tile shows the v35 number large, with the L7 and untrained values beneath
it in muted ink as a two-row comparison — the comparison is the point, and a
tile that shows one encoder alone makes the reader open another page.

**Below:** one **capacity summary** chart — `exact_hit` and `acc_45` against
`K` on a shared x-axis, two small panels side by side, blue ordinal ramp
retired here in favour of encoder colour since `K` is now the x-axis. Then the
full cross-test table: rows = `(encoder, K, steps)`, columns = the headline
metric of each test, sortable, `tabular-nums`.

---

## 4. `test_a.html` — attractor and basin

**Section 1 — is it an attractor at all.**
- Line chart: `frac_self_consistent` vs. `steps`, one line per `K` (blue ramp).
  This is the plot that either shows §1.3's collapse or refutes it.
- Beside it, the same x-axis: `mean_pairwise_cos` vs. `steps`. Two charts, not
  one dual-axis chart — they share `steps` and nothing else.
- A stat tile: `sign_flip` rate. If it is nonzero this is the most important
  number on the page and it deserves to be a number, not a bar.

**Section 2 — the basin, in real space.**
- **`cos_goal` vs. distance**: median line with a 10–90 band, one panel per
  `steps`, one line per `K`. The continuous quantity underneath everything
  else on the page.
- **`exact_hit` vs. distance**: same layout. Chance is not meaningful here;
  instead draw the `1/|CELLBANK|` floor as the hairline.
- **`retr_dist` vs. distance**: median + 90th percentile. Log y — it spans
  "one cell off" to "across the arena".

**Section 3 — the outcome map.** The signature figure of this page: a
`size × size` grid per env, cell coloured by outcome.

- In-env outcomes use the **sequential blue ramp on `retr_dist`** — so `exact`
  is the darkest step and a near-miss is visibly near. This is one continuous
  encoding rather than three arbitrary categories.
- `other_env` cells: slot 2 orange. `alias` cells: slot 3 aqua. Three colour
  families total, which is exactly the all-pairs budget.
- Goal cell marked with a 2px ring in primary ink, never by colour alone.
- Laid out as **small multiples**: rows = `K`, columns = `steps`, one
  representative env, with a control to step through envs.

**Section 4 — anisotropy and confusion.**
- Polar chart: `r_by_direction` over 8 sectors, one trace per `K`. A radar is
  acceptable here precisely because the axes *are* angular — this is a physical
  direction, not an arbitrary multi-attribute comparison.
- `confusion(j)` as a `K × K` matrix heatmap, sequential blue, rows and columns
  ordered by **scaffold-offset distance from the test env** rather than by env
  index. That ordering is what makes the diagnostic legible: aliasing shows as
  mass on the diagonal-adjacent band, interference as a uniform field.

---

## 5. `test_b.html` — q accuracy on the grid

**Section 1 — the headline curve.** `|err|` vs. distance-to-goal: median line,
IQR band, one line per `K`, one panel per `steps`. Chance at 90°. Oracle and
local-oracle as reference lines.

**Section 2 — what the agent actually consumes.** `acc_45` and `acc_90` vs.
distance, two panels. `acc_45` is the discrete agent's real accuracy and is
labelled as such on the chart, not only in a caption.

**Section 3 — the `steps` question.** The plot this whole axis exists for:
`acc_45` vs. `steps`, one line per `K`, with the three interpretations from
`EVAL.md` §4 named in a short legend note — flat means the extra multistep
channels are redundant, falling means iteration destroys the readout, rising
then falling means `steps=1` is not the optimum. Annotate the production
channels (`s = 1, 2, 3`) with a shaded band so the reader sees which part of
the curve is currently wired to the policy.

**Section 4 — the heatmaps.** Three, in this order, each a small-multiple grid
over `(K, steps)`:

1. **Goal-relative**, 39×39, signed `err`, diverging blue↔red, gray at 0. The
   goal is at the centre by construction, marked with a ring. This is the
   primary map (`EVAL.md` §4).
2. **Goal-relative**, `|err|`, sequential blue — same geometry, magnitude only,
   for readers who want "how bad" without "which way".
3. **Env-absolute**, `|err|`, sequential blue, aggregated over envs and
   ignoring goal position — the wall-and-corner view that goal-relative
   coordinates average away.

Then one **single-env panel** in absolute coordinates with its real goal
marked, as the sanity anchor: aggregates hide structure, and one raw example is
what catches a harness bug.

**Section 5 — magnitude and bias.**
- `‖q‖` vs. distance, one panel per `steps`, with a note on the chart that
  magnitudes are not comparable across `steps`.
- Signed `err` by 8 sectors as a polar bar chart — the Gram–Schmidt North-bias
  check (`EVAL.md` §1.5). A systematic lobe on the North axis is the signature.

---

## 6. `test_c.html` — continuous positions

Mirrors `test_b.html` section for section, so the two can be read side by side,
plus the two figures that only exist here.

**The decomposition chart — the page's headline.** One panel, three lines on
one axis, against distance with the fine near-zero bins:

- `|err|_C` — what the agent actually gets (blue, solid, 2px)
- `err_geom` — the analytic snap ceiling with a perfect readout (muted ink,
  dashed)
- `excess` — `|err|_C − |err|_B`, the snap-attributable part (orange, solid)

All three are angles in degrees, so this is legitimately one axis. Log-x, since
the whole story is in `d < 3`. This single chart is what separates "the encoder
degrades near the goal" from "quantisation degrades near the goal".

**The sub-cell heatmap.** Mean `|err|` over the continuous plane at 8 bins per
cell, sequential blue, with the snap-cell boundaries drawn as a hairline
overlay. The field is piecewise constant within a cell by construction, and the
figure's job is to show that visibly — if it is not piecewise constant, the
harness is not snapping the way the env does.

`n`-per-bin is mandatory here and shown as a bar strip beneath the
decomposition chart: uniform-area sampling gives `∝ d` samples per bin, so the
near-goal bins carrying the headline are the sparsest on the page.

---

## 7. `test_d.html` — flow

**Hero:** `reach_rate` stat tiles, discrete and continuous variants side by
side, each with its `mean_steps` and — always adjacent, never separated — the
success count it was computed over. The tile template puts them in one line
specifically so `mean_steps` can never be read without its denominator
(`project_nav_tri_failure_modes`).

**Section 1** — `reach_rate` vs. start distance, one line per `K`, panel per
`steps`.

**Section 2 — the trajectory figure.** Per env: the `q` field as a quiver
overlay on the `size × size` grid, arrows in muted ink with length ∝ `‖q‖`,
over a sequential-blue background of `|err|`. Successful trajectories drawn in
slot 1 blue at low opacity, failed ones in slot 8 red. Spurious sinks marked
with a filled ring, sized by basin, in slot 2 orange; limit cycles drawn as
closed loops.

This is the one place in the whole report where individual trajectories are
drawn, and it earns it: a sink is a *location*, and no aggregate statistic can
point at one. It is explicitly **not** a substitute for the aggregate curves —
per `feedback_no_squinting`, the sink locations and basin sizes are recorded as
numbers by `flow.py` and this figure illustrates them rather than being the
means of discovering them.

**Section 3** — sink inventory table: env, sink cell, basin size, distance from
the true goal, sorted by basin size.

---

## 8. `controls.html`

All of `EVAL.md` §6 on one page, because their value is comparative:

- **Oracle vs. local oracle vs. Hopfield**, as `|err|` vs. distance on one
  chart. The three-way gap is the attribution: local-oracle error is the basis
  itself, oracle-minus-local is manifold curvature, Hopfield-minus-oracle is
  recall.
- **Gram–Schmidt order swap**: paired bars, original vs. swapped, per encoder.
- **Empty memory** (`K = 0`): a pass/fail tile. It must sit on chance; anything
  else is a leak.
- **Untrained encoder**: appears as the muted dashed line on every other page's
  aggregate curves, and here as its own column.
- **`use_tanh=False`**: paired bars against the default, plus the `tanh_arg`
  distribution as a histogram, annotated with where v35 (`β = 3.70`) and L7
  (`β = 100`) each sit. That histogram is the evidence for or against §1.3's
  linearity claim, and it is the one figure that most directly motivates the
  rescue page.

---

## 9. `rescue.html` — only when `--rescue` ran

Carries a **banner** at the top, in `warning` status colour with an icon and
label: *these settings are not the production operating point and these numbers
are not encoder-quality numbers.* Status colour with icon and text, never
colour alone.

Main figure: `frac_self_consistent` against `β·scale` (the product, since only
the product matters), one line per `zero_diag` setting, faceted by `alpha`.
Second: `mean_pairwise_cos` on the same grid — a setting only counts as success
if the first is high **and** the second is low, so the two are shown adjacent
with a shared x-axis and the success region shaded on both.

If any setting succeeds, a callout links to the re-run of §3.2 and §4 under it,
since a fixed point with a zero-radius basin would be worse for navigation than
what we have today.

---

## 10. Anti-pattern checklist

Checked against the `dataviz` catalogue before shipping any page:

| risk on these pages | ruling |
|---|---|
| `steps` and `K` both as colour | avoided — `K` is the ramp, `steps` is the facet |
| dual axis for `cos_goal` and `retr_dist` | forbidden — two panels sharing an x-axis |
| 5 outcome categories as 5 hues on a map | avoided — sequential on `retr_dist` + 2 slots |
| ramp on nominal categories | not applicable — `K` is ordered |
| rainbow heatmap | forbidden — one hue for magnitude, blue↔red for signed |
| 8 hues when the story is one number | index hero row is stat tiles |
| trajectory plots as the means of discovery | forbidden — numbers first, figure illustrates |
| a mean without its spread | forbidden — IQR band or per-world lines |
| a curve without its `n` | forbidden — §1.4 |

**Last step, every time: render the pages and look at them.** The palette
validator checks colour, not layout — label collisions, panel overflow at the
39×39 heatmap grid, and legend wrapping only show up on screen.
