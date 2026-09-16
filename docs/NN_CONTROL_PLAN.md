# Goal-conditioned NN control: can a plain network navigate from encoded states?

**Status (2026-09-16).** A0, A1, A1x, A2, B1x, B2, B3 and their diagnostics
are done; findings in §1. Next (§6): the representation probes. Branch `worktree-nn-generalization-control`;
run-by-run record in `NN_CONTROL_LOG.md`; code map in §7.

---

## 0. The question

The attractor route to a goal is: store the goal's code, recall it from the
current code, read the difference off as a direction. It works on any
(start, goal) pair without ever having seen the pair, in any region of the
scaffold, because nothing about the pair is learned — the local frame at
every cell is built from Φ.

This experiment asks whether a network with no attractor can do the same
when it is simply *told* the goal. `GridEnv` has no obstacles, so the
optimal action is `normalize(g − p)` (`bfs_action_batch_continuous`
literally returns the unit vector). The task is therefore exact:

> **Can a network compute `normalize(g − p)` from `enc(p)` and `enc(g)`,
> for encodings it has not seen?**

There are two claims inside "can it": the **instant** claim (one pair,
no history — what the attractor does), tested by a memoryless network on
i.i.d. pairs (Experiment A); and the **given-time** claim (put the
network in the environment and let it move — can it learn what it needs
in context, from its own trajectory), tested by a recurrent network on
lifetimes of episodes (Experiments B, B2). Both are scored on one table
(§2.5), and B is read against A.

---

## 1. Findings

Angular error against the true direction; random is 90°. All held-out
numbers are on environments (walls, scaffold positions) never trained on.

### 1.1 A memoryless net learns the code it was shown — and only that

**Grid mode**, `[gbook(p), gbook(g)]`, 64 envs scattered over the scaffold
(A1): **0.23°** mean / 0.17° median on held-out envs, held-out start
*and* goal cells, every pair enumerated; discrete actions 1.000; two seeds
agree to 0.01°. The nearest-neighbour lookup line on the same cells is
58–83°, so this is learned structure, not a table of seen codes. Best
config 5×768 relu, 8000 updates, lr 1e-3 with ×0.1 at 70% (constant lr
destabilises at ~6000 updates; cosine from the start hurts; tanh is 10×
worse; weight decay does nothing).

**Regular mode**, `[omni(p), omni(g)]` (A2): **5.5°** on never-seen
barcodes (5.65 / 5.35 two seeds; discrete 0.991; NN line 38°), flat in
`|g − p|`, and still **22°** when `omni(p)` and `omni(g)` come from two
*different* unseen walls (D2). The network inverts the ray projection view
by view — run boundaries sit at angles fixed by position — and uses the
shared bits for the last 17°. It learned the observation's generating
rule; no ray-axis architecture was needed.

**Confine grid-mode training to a 400×400 corner** (A1x) and it fails
outside: **44°** (two seeds; K = 160: 57–71°). Monotone in cycle
coverage. By displacement the failure is the mid-range Chinese-remainder
band (14° at `d = 1`, 63° at `d = 9`, 30° where the modules wrap): the
per-module periodicity is learned, the cross-module combination is not.

### 1.2 What the two models compute (range and band probes)

The grid code has two parts: a per-module phase (local, translation-
invariant) and the cross-module combination that resolves a displacement
beyond half a module period. A network can implement the second as a
**rule on phase differences** (residue triple → Δ; 39 entries per axis,
all present in any one 20×20 env, valid at every position) or as a
**lookup over the absolute phases it saw**. Synthetic-pair probes on the
checkpoints (log, 2026-09-14) separate them:

| model | random positions over the whole cycle, \|Δ\| ≤ 18 | \|Δ\| = 20 / 24 / 35 / 50 | on coordinates it saw, \|Δ\| ≥ 20 |
|---|---|---|---|
| A1 scattered (fixed lattice) | **0.2–0.4°** | 5 / 22 / 101 / 140 | — |
| B2 `dist@90` (translation-trained) | **0.5–1.0°** | 9 / 22 / 87 / 140 | — |
| A1x corner | 15° at d = 1, 60° at d = 8 — per axis: **0.2°** if X *and* Y were seen values, 44–54° if one was, 92° if neither | — | **86–89°** |

So the scattered model learned **the rule** — position-free, but limited
to the displacement range it trained on (±19; anti-aligned at 50–60,
where the residues read the wrong sign). The corner model learned **a
per-axis lookup over the coordinate pairs it saw**, neither position- nor
range-general. Same architecture, same objective: the training geometry
chose the solution. The rule fits a corner's data exactly too; SGD took
the lookup because at corner scale it is cheaper.

A1x's reported 0.2° on held-out envs *inside* the corner was a placement
artifact: the generator draws held-out envs from the training envs'
lattice (pitch 47, ±3 jitter in a dense packing), so their coordinate
values were all seen. The real boundary is seen vs unseen coordinate
value, inside or outside; the 44° outside stands. (Caveat for any
"held-out place inside the region" claim under a dense packing.)

Nothing here learned the full-range decode — a fixed linear map on module
phase angles mod 2π, unique to |Δ| < 858 — and nothing about ±19 data
pushes a network toward it.

### 1.3 History does not build a map on the real code (B1x)

Three arms trained on lifetimes inside the corner, scored outside it
(readout 1 = static, `h = 0`; readout 2 = by episode over 20-episode
lifetimes, sampled):

| arm | R1 outside | R2 by episode | slope |
|---|---|---|---|
| A1x reference (MLP, i.i.d.) | 44° | — | — |
| `dist` (MLP 5×768, rollouts) | 22.5° (median 6°) | 70–76°, flat | 0 |
| `full` (GRU 1×512 + prev_action) | 44.5° | 55–65°, flat | −0.24°/ep |
| `rec` (GRU, no prev_action) | 44.9° | 53–65°, flat | −0.25°/ep |

No arm accumulates anything across goal changes. The GRUs refine for a
few steps within an episode (46° → 31°) and collapse on long ones (110°
at step 40). `dist`'s 22.5° is the Gaussian-NLL head hedging the
mid-range band to a flat ~25° (D1), not better disambiguation —
trajectory-shaped pairs through the `1 − cos` loss stay at 44° (A1y).

This null is structural. With one fixed lattice, the frame at any cell is
a deterministic function of the observable code, so for every region
training covers the weights learn it outright and no lifetime ever
rewards inferring it from `(Δgbook, action)`. Supervised training on the
real code cannot select for in-context frame learning; the pressure
exists only at test time.

### 1.4 Forcing it: an unseen lattice is learned in context (B2)

B2 randomises the lattice per lifetime — rotation θ (excluding a ±15°
band around the standard lattice), a translation, one lattice per row —
so the frame is unknowable from the weights and only the trajectory
carries it. Test: the standard lattice, never trained on. Design in §5.

**Gates, all passed.** Told θ, a memoryless MLP is at **1.0°** on the
held-out lattice (the task is well-posed). A scripted two-step estimator
through the same lifetime evaluator: **92 → 88 → 5.4 → 0.1 → 0.0°** by
step, ~2° flat over 19 goal changes, identical on every lattice (the
information is in two steps of trajectory and lasts the lifetime). The
memoryless null is **90°** on every readout (nothing but the trajectory
carries θ).

**A GRU on the raw code does not learn it.** GRU 2×512 + prev_action:
89° flat on every readout for 8000 updates. With half the lifetimes at a
fixed training orientation as a foothold: 72° on that orientation, still
90° on the held-out one. Diagnosis: the memoryless MLP learns the
translation-invariant decode from rollouts in 600 updates (`dist@90`:
0.7° on its orientation, and exactly |θ − 90°| elsewhere); the GRU cannot
fit that function at its input. A 5×768 MLP encoder trained *jointly* in
front of the GRU collapses — dead ReLUs; input-independent with
LayerNorm; loss spikes even on fully determined data — because the GRU's
BPTT gradient destabilises the shared encoder at lr 1e-3.

**With the decode given, the recurrent net learns the frame.** The
`dist@90` trunk frozen as encoder, GRU 2×512 + prev_action trained on
top over random-lattice lifetimes. Held-out envs, held-out lattice, two
seeds:

| | no history (R1) | 1 step | 10 steps | lifetime (e10–19) |
|---|---|---|---|---|
| frozen decode, s0 / s1 | 150° / 139° | 56° / 81° | 18° / 23° | **14.2° / 14.7°** |
| `rec` (no prev_action), frozen decode | 103° | 105° | 100° | 45° at e19, rising across episodes only |

With no history the network commits to the training-mean orientation,
nearly opposite at θ = 0; one `(Δgbook, action)` pair takes it to ~60°;
ten steps to ~20°; it holds ~14° over the lifetime, against 90°
memoryless and the estimator's 5° at two steps. On trained orientations
it sits at 8°, the sampled-policy floor. Without `prev_action` it can only
integrate across episodes — the action is what makes frame estimation a
one-step measurement, as the estimator's algorithm says.

**And from scratch.** The same MLP → GRU trained jointly works once the
encoder has its own learning rate (1e-4, 10× below the GRU's) and a
3000-update warm-up on one fixed orientation before the lattices vary:

| | R1 | 1 step | 10 steps | lifetime |
|---|---|---|---|---|
| S1 joint | 94° | **31°** | **12.5°** | 15.7° |
| S2 joint, encoder detached from the GRU gradient | 96° | 81° | 26° | **10.3°** |

Both learned the decode in the warm-up (2.0° / 0.5° on the warm-up
orientation, |θ − 90°| elsewhere) and the frame within ~1000 updates of
the switch. The joint failures were the optimiser, not the task.

### 1.5 What B2 does and does not show

A translation over the full period is a move to another scaffold
position, so a B2 network sees the equivalent of every position: **B2
holds out orientation, not region.** It says nothing about learning an
unseen scaffold region, in weights or in context. What it establishes:
(i) a plain network can learn the relative-phase decode when the data
close the lookup route (the capacity is there; the bias is not — §1.2);
(ii) the one property of a grid code that no invariance fixes, its
orientation, a recurrent network can measure from its own trajectory,
frozen-decode or from scratch. A rotation is a one-parameter in-context
target (the estimator needs two steps); "a new region under the same
lattice" needs no in-context learning at all once the decode is the rule
— and none if it is the lookup, because there is nothing to measure.

### 1.6 From a corner to an unseen region under an unseen orientation (B3)

B3 (§6.1) keeps the region holdout B2 gave up: training envs inside the
400×400 corner, per-row lattices at every orientation with the
translation drawn so each *rotated* footprint stays inside the corner —
so the network never sees a phase combination from outside it — and
test envs whose rotated footprints sit 300 cells outside on both axes
(`far@θ`). From scratch, S1's recipe (warm-up 3000 at θ = 90°, then the
0.5 mix), 8000 updates:

**The decode learned from the corner is the rule.** During the warm-up
`far@90` — every coordinate unseen, the orientation trained — reached
**1.6°** (4.5° at u = 1000), while `far@45` sat at 46° and `far@0` at
90°, the |θ − 90°| signature. A1x's fixed placement on the same corner
produced the lookup; corner-confined translation produced the rule.

**Then the frame, in context, in the unseen region.** At u = 8000:

| set | region | orientation | no history | e0 → e4 → e9 → e19 | ep0 s0 → s1 → s2 → s5 → s10 |
|---|---|---|---|---|---|
| `far@0` | unseen | unseen | 94° | 23.5 → 20 → 15 → **13.9** | 94 → 34 → 25 → 14 → 14 |
| `heldout_out` / `heldout_in` | outside / inside | unseen | 94° | 23 → 19 → 15 → 14.4 / 14.5 | 95 → 34 → 24 → 14 → 13 (both) |
| `far@45` / `far@90` | unseen | trained | 49° / 6° | 15.5 → 13 → 9 → 7.5 / 8.6 → 14 → 9 → 7.3 | 50 → 24 → 18 → 10 / 11 → 11 → 9 → 6 |

Inside and outside the corner are identical to the decimal at every
orientation: the decode is fully position-general, and the only cost
left is the held-out orientation (~7°, as in B2). A network trained on
nothing but a corner's phase combinations navigates a region it never
saw under an orientation it never saw, to ~34° after one step of its own
trajectory, ~14° after five, and ~14° over the lifetime, against 94°
with no history, 90° memoryless, and 44° for A1x's weights outside the
same corner. Open: a transient at episode 1 on the trained-orientation
sets (`far@45` 15.5 → 30.5 → 26.5 → 12.7) — the first goal change
disturbs something a fixed orientation had made free — which the probes
(§6.2) should explain.

### 1.7 Standing conclusions

1. A memoryless network does not learn the attractor's given frame; it
   learns the code it was shown — the rule on differences from scattered
   coverage or from corner-confined translations, a lookup from a fixed
   corner placement — and neither extrapolates in displacement range.
2. On the real code, no training regime can make history build a frame,
   because the weights always have the shorter route.
3. Remove that route and a recurrent network learns an unseen code's
   frame from its trajectory — as a two-part computation, memoryless
   decode then in-context rotation, learnable jointly from scratch — and
   with the decode learned from a corner under corner-confined
   translations, it does so in a region it never saw: ~150° → ~15° over a
   lifetime, ~20° in ten steps, ~14° in five on B3.
4. What the attractor has built in — a translation-invariant code and a
   frame — a plain network can acquire: the invariance from data that
   deny it a lookup, the frame from lifetimes that deny it a fixed
   lattice. What it does not acquire from either is the full-range
   decode (§1.2).

---

## 2. The task, precisely (shared by A, B, B2)

### 2.1 Environments

`GridEnv`, size `S = 20`, no obstacles, ±1 barcode walls at
`wall_resolution = 1`, `observation_size = 120` rays over 120°. Envs are
placed on the `lambdas = [11, 12, 13]` scaffold (`Npos = 1716`) by the
declared-domain generator (`world/generate.py`), `place_margin = 20`.

### 2.2 Encodings

| name | `enc(c)` | width |
|---|---|---|
| **gbook** | smoothed grid code at the cell's *global* scaffold position | `Ng = 434` |
| **omni** | all four cardinal ray-cast views at the cell | `4 · 120 = 480` |
| **xy** | `(x, y) / S` | 2 |

**Grid mode** is `[gbook(p), gbook(g)]`, **regular mode** `[omni(p),
omni(g)]`, **xy mode** the ceiling. Grid mode has no sensory channel (it
would be a second encoding of `p` carrying env identity) and there is no
reward channel (arrival is the only event and it carries nothing).
`prev_action` is absent from A and a factor in B.

The grid code is *not* translation-invariant: `gbook(p) − gbook(p + d)`
for the same `d` at two base points has cosine −0.33 (0.73 only across a
module period), so `(gbook(p), gbook(g)) → g − p` depends on the phase of
`p` in each module. The attractor sidesteps this with a hand-built frame
at every cell; a network has to learn the phase geometry. `omni` for both
ends keeps the task heading-free and symmetric.

### 2.3 Output and teacher

Discrete: 4 cardinal logits against the **optimal set** (actions that
reduce Manhattan distance; loss and metric both use the set).
Continuous: a 2-vector against the unit vector `(g − p)/‖g − p‖`; B runs
the env with `continuous_normalize = True` so a step is a unit vector.

### 2.4 Holdouts

| holdout | mechanism | grid mode tests | regular mode tests |
|---|---|---|---|
| **H-env** — held-out environments | `make_val_set(wall: held_out, place: held_out)` | a scaffold region never seen | a barcode never seen |
| **H-goal** — cells never a goal | `goal_val_frac = 0.2` | goal-input generalisation | same |
| **H-region** — cells never a start or a goal | `region_val_frac = 0.1`, `region ⊂ goal_cells_val` | a phase combination never seen | an observation never seen |
| **H-lattice** — an orientation never trained (B2) | per-lifetime θ, held-out band \|θ\| < 15° | the frame itself | n/a |

`place` has two strengths. *Scattered* (A1): training envs anywhere in
the scaffold, held-out envs elsewhere — unseen phase combinations are
interleaved with seen ones. *Corner* (A1x): training inside
`rect:0,0,K,K`, tests outside (`OutsideRect` with the margin) — unseen
combinations are systematically far. `base_val` inside the corner
separates the wall effect from the phase effect, **but** shares the
training placement lattice, so it tests unseen cells on seen coordinate
values (§1.2); a real inside test samples coordinates the training envs
never covered.

B honours H-goal (goals from `goal_cells_train`) but not H-region (its
trajectories walk through region cells); A and B share the `start_train`
row of the table.

### 2.5 The table

For every env set (train, H-env, and `same` — a fixed subset of the
training envs, the env-side probe): the **start × goal quadrant table**,
start ∈ {train, region} × goal ∈ {train, goal-heldout, region}. Each cell:
mean angular error (median, fraction < 30°) or optimal-set accuracy, with
three reference lines scored on the same pairs — teacher (0°), uniform
random (90° / ≈0.48), and a **nearest-neighbour decoder** (decode `p`, `g`
to the nearest training cell by cosine, subtract): the lookup line a
network must beat to have learned structure. The final table enumerates
every pair. Checkpoint selection uses the H-env train × train cell only.

Decision rules: *generalises* — H-env region × region within 10° / 0.05
of train × train and ≤ 20° / ≥ 0.90; *interpolates only* — region row
fails; *memorises* — H-env near random, `same` ≫ H-env.

---

## 3. Experiment A — memoryless, i.i.d. pairs

**Model.** `PairRegressor`: a plain MLP (`num_layers` × `hidden_size`,
relu | tanh) with a linear head — 4 logits, or a 2-vector normalised to a
direction. A deterministic regression, no distribution head, no `act()`;
it shares the `FeedForwardCore` trunk and the input layout with
`RNNAgent` so readout 1 can run B's weights through the same evaluator.

**Data.** Per env, every cell's encodings precomputed once. A batch is
`pairs_per_env = 512` draws of `p ∈ start_train`, `g ∈ goal_train`,
`p ≠ g` from each of 64 envs — 32k pairs, one forward, no env stepping.
At 8000 updates that is ~35 epochs of the train quadrant, with
replacement. `--pair_sampler trajectory` (A1y) instead emits every cell
on the straight line from `p₀` to `g`.

**Loss.** Discrete: cross-entropy against a uniform distribution over the
optimal set. Continuous: `1 − cos(out/‖out‖, u)`, so the loss is the
metric. (This loss sits on a plateau at loss ≈ 1 when the conditional
mean of the target is 0 — B2's oracle run escaped it after ~200 updates
at the real batch size; a toy did not.)

**Evaluation.** The table every `eval_every` updates on sampled pairs,
enumerated at the end. `eval_goal_pairs --by_distance` bins error by
Chebyshev `|g − p|`; `--mismatched_walls` scores `omni(p)` from wall *i*
against `omni(g)` from wall *j*; `--eval_thetas` adds env sets on
rotated lattices.

**Arms run.** A0 xy (pipeline gate: ≤ 5° everywhere — passed at 0.3°);
A1 grid, depth/width/schedule sweep; A1x grid on the corner (K = 400,
K = 160; two seeds); A1y trajectory pairs on the corner; A2 regular
(l5h768 and l4h512, both action modes). A3 (ray-axis encoders) and A4
(scaling) were conditional on failures that did not happen.

Script `hopfield_nav/train_goal_pairs.py`, launcher `run_goal_pairs.sh`.

---

## 4. Experiment B — recurrent, lifetimes of episodes

**What it tests.** Whether a network given time in an environment reaches
goals a memoryless one cannot — and whether that is memory or the
rollout data.

**Arms.** `full` (GRU + prev_action, the hypothesis), `rec` (GRU, no
prev_action: recurrence alone), `dist` (MLP on the same rollouts: the
data-distribution control with no memory). If `dist ≈ A`, any `full` gain
is memory; if `dist > A`, the data did it. The `dist` vs A comparison is
made on H-env envs only.

**Data.** An *episode* is one `(p → g)` attempt, ended by goal-reach or
the 60-step cap; a *chunk* is 64 steps of one env, the unit of one BC
update; a *lifetime* is 32 chunks on one env with the hidden state
carried throughout, then the env is swapped and `h` zeroed. **On
goal-reach a row gets a fresh start and a fresh goal (from
`goal_cells_train`), keeping `h`** — so an episode is one `(p, g)` pair as
in A, goal-memorisation across episodes is impossible, and the only
thing worth holding in `h` is the env. DAgger against the teacher;
truncated BPTT within a chunk from the detached `initial_h`. 64 lifetimes
× 8 envs per update, round robin.

**Evaluation — three readouts, one metric.** Every readout scores the
policy's own action against `normalize(g − p)`; the teacher never acts.

- **Readout 1 — instant.** A's static table at `h = 0`, `prev_action = 0`,
  mean direction. B's score on the attractor's own claim.
- **Readout 2 — with experience.** On one H-env env, 64 independent
  lifetimes of 20 episodes with fresh goals, actions **sampled** (the
  mean of a Gaussian fit to an uncertain target collapses toward zero).
  Every score lands at an `(episode, step)` cell. Two marginals: *by
  episode* — accumulation across goal changes; *by step within episode 0*
  — how many steps are needed. Three shapes: flat in both (memoryless-
  equivalent); rises with step and resets each episode (episode-local);
  rises with episode (a map, or a frame, accumulating). Episode 0, step 0
  is readout 1 by construction (gate C16).
- **Readout 3 — behaviour.** Success and steps-to-goal, secondary.

**Reading.** `full ≈ A` and readout 2 flat → history does not help;
readout 2 rising with `dist ≈ A` → in-context learning; `full > A` on
readout 1 → the static evaluator leaks; `dist > A` → fix A's sampler
first. B is read only where A failed.

**Continual plot.** `train_goal_pairs.py --mode sequential` and B's
sequential mode slot into the project's success-vs-time figure (one block
per env, every seen env re-scored); not run in this line.

Script `hopfield_nav/train_goal_lifetimes.py`, launcher
`run_goal_lifetimes.sh`; readout 2 in `evaluation/lifetime.py`.

---

## 5. Experiment B2 — the grid code under lattice randomisation

### 5.1 Why

§1.3: on one fixed lattice no lifetime rewards inferring the frame. B2
gives every lifetime its own lattice, so the weights have nothing to
memorise and the only route to the arena-frame direction is to measure
this lifetime's frame from `(Δgbook, action)` and carry it — the
attractor's `d_N, d_E`, read off Φ there, measured from the trajectory
here.

### 5.2 The code family

Standard code: module *m*'s bump sits at phases `(X mod λ_m, Y mod λ_m)`
of the global position. Randomised, per row (per lifetime):

    φ_m = ( R_θ (X, Y) / s  +  T )  mod λ_m         R_θ = [[cos θ, −sin θ], [sin θ, cos θ]]

then the same toroidal Gaussian (`gridcode/lattice.py::gbook_at`; equals
`smooth_gbook` exactly at the identity — gate B2-C1). Three ingredients,
each necessary:

- **Rotation θ** ~ U[0, 2π) minus the band |θ| < 15°. The essential one:
  a memoryless net can recover the *rotated* displacement `R_θ(g − p)`
  from one pair (three coprime moduli pin it), but the arena-frame
  direction needs θ, which no single pair contains — the best memoryless
  output averages 90°. Scale `s` alone would not do this (direction is
  scale-invariant); it is an optional realism knob, `s = 1` in every run.
- **Translation T** ~ U[0, 1716)², the combined period, after rotation.
  Without it the absolute phases `R_θ(O + p)` at the 64 fixed env offsets
  `O` are a weights route to θ on training envs. With it, absolute phases
  are uniform whatever θ is; phase *differences* — the decode and the
  trajectory's frame signal — are untouched (tested).
- **One lattice per row, not per env.** A lattice held fixed for a
  lifetime is fitted in weights across the ~256 updates the lifetime lasts
  (the memoryless null's loss fell to −0.2 on its own training lifetimes
  under per-env lattices). 64 rows × 64 envs = 4096 concurrent lattices
  is past what weights can fit; per-env tables become `(64, 400, 434)`,
  2.8 GB for 64 envs.

Test lattice: the standard code, θ = 0. A second point inside the band
(θ = 7°) checks the band is held out, not the one value; a training
orientation (45°) is the in-distribution check.

### 5.3 Training and arms

Scattered placement (64 envs), continuous actions, 8000 updates, lr 1e-3
with ×0.1 at 70%, BPTT 64, otherwise as §4; at each lifetime boundary
every row draws a lattice and its `(S², Ng)` table is synthesised.

| arm | encoder | core | trained on |
|---|---|---|---|
| `dist` | — | MLP 5×768 | the null |
| `full` | — | GRU 2×512 + prev_action | raw code |
| `full`, anchor-mix | — | GRU 2×512 | half the lifetimes at a fixed training θ = 90° |
| `dist@90` | — | MLP 5×768 | all lifetimes at θ = 90° with per-row translation: the decode alone |
| **frozen decode** | `dist@90` trunk, frozen; prev_action bypasses it | GRU 2×512 + prev_action | random lattices |
| `rec`, frozen decode | same | GRU 2×512, no prev_action | random lattices |
| **S1 from scratch** | 5×768 relu + LayerNorm + skip to the head, lr 1e-4 | GRU 2×512 + prev_action, lr 1e-3 | 3000-update warm-up at θ = 90°, then mix 0.5 |
| S2 from scratch | as S1, GRU gradient detached from the encoder | same | same |

Flags: `--lattice_theta_random --lattice_theta_holdout_deg 15
--lattice_translate --lattice_per_row --eval_thetas 0,7,45[,90]`;
`--lattice_mix_standard_frac`, `--lattice_mix_theta_deg`,
`--lattice_mix_warmup_updates`; `--encoder_layers 5 --encoder_hidden 768
[--no-encoder_norm] [--encoder_detach] [--encoder_init <dist ckpt>
--encoder_freeze] [--encoder_lr 1e-4]`; `--scripted` runs the estimator.

### 5.4 Gates

Passed before any GRU is read, in order:

| gate | what | pass | result |
|---|---|---|---|
| B2-C1 | `gbook_at(0, 1)` equals the scaffold's smoothed book | 1e-5 | exact |
| B2-C2 | per-module centroid shift for a unit step equals `R_θ a / s` | 0.05 cells | 5e-7 |
| B2-C3 | oracle-θ MLP (`(cos θ, sin θ)` appended) on the held-out lattice: the task is well-posed | ≤ 5° | **1.0°** |
| B2-C4 | scripted two-step estimator (`evaluation/scripted_frame.py`: centroid shifts under two axis steps → `R_θ/s`; CRT on phase differences → Δ′; un-rotate) through readout 2 | ≤ 5° from step 2, flat by episode | 5.4° at step 2, 0.0° at step 5, ~2° by episode |
| B2-C5 | memoryless null on the held-out lattice | ~90° both readouts | 89–90° |
| B2-C6 | `full`'s readout 2 at (episode 0, step 0) equals its readout 1 | C16 | holds |

### 5.5 Readouts and reading

As §4, on the standard lattice (held-out envs, and the training envs — the
code is unseen there too) plus `heldout@7` and `heldout@45`. The by-step
marginal of episode 0 (`ep0_by_step`) is the "how many steps" number; the
by-episode marginal says whether the frame is kept across goal changes.

| `dist` | estimator | `full`, held-out lattice | reading |
|---|---|---|---|
| ~90° | ≤ 5° | falls within episode 0, holds after | in-context learning of the frame |
| ~90° | ≤ 5° | falls, plateaus 30–60° | learnable in part; a cruder estimator |
| ~90° | ≤ 5° | ~90° flat | a capability/optimisation limit, not impossibility |
| ~90° | fails | — | the design is broken |
| < 90° | — | — | θ leaks through something; find it first |

Results: §1.4. Note the reference numbers are on different holdouts from
A1x's 44° (region vs orientation, §1.5).

---

## 6. Next

### 6.1 B3 — the corner, with lifetimes (done; results in §1.6)

The version of the original question that survives §1.5: train **only on
the corner's phase combinations**, at every orientation and every
translation *within* the corner, and test outside it. Run 2026-09-15/16;
(1) the pure-MLP control never left the loss plateau (uninformative; a
second seed is running), (2) from scratch answered both questions — the
rule from the corner, the frame in the unseen region — so (3) is not
needed. Design kept here for the record.

**Design.** Envs inside `rect:0,0,400,400` (A1x's 64), per-row lattices
with random θ and a translation drawn so the rotated env's footprint stays
inside the corner (`--lattice_translate_region rect:0,0,400,400`; ~370
values per axis, so a single code still does not reveal θ). Eval sets:
`heldout_in` (inside; plus the band probe on coordinates the training
envs never covered), `heldout_out` at θ = 0 (all coordinates unseen, the
orientation unseen), `heldout_out@45` (coordinates unseen, orientation
trained), `heldout_out@90`.

**Runs, in order.**
1. **Memoryless control.** `dist`, all lifetimes at θ = 90° with corner
   translations, 1000 updates (~15 min). Read `heldout_out@90` vs
   `heldout_in@90`: ≪ 90° means the encoder learned the **rule** from a
   corner; ~44–90° means the lookup again. Within-corner translation makes
   the lookup ~400 × 39 entries per axis — between A1x's 9k (lookup won)
   and A1's 36k (rule won) — so the outcome is genuinely open.
2. **From scratch MLP → GRU**, S1's recipe (encoder lr 1e-4, warm-up 3000
   at 90°, then mix 0.5), 8000 updates. `heldout_out@45` isolates the
   decode's position generalisation; `heldout_out` at θ = 0 is the
   composite — a new region under an unseen orientation, learned from the
   trajectory.
3. If (1) takes the lookup: the **capacity sweep** on A1x's setting —
   hidden 64 / 128 / 256 × depth 3 / 5, one weight-decay variant — reading
   `heldout_out` and the inside band probe. The lookup is the cheapest fit
   only while the network has room for it.

**Reading.** (1) rule + (2) `heldout_out` falls like §1.4 → a recurrent
net trained on a corner navigates a new region under a new orientation,
the strongest form of the claim — *which is what happened (§1.6)*.
(1) lookup → the corner question is an inductive-bias question (§1.2)
and (3) is the test; (2) is then read on `heldout_in` only. This
supersedes the earlier B2-mix idea, which had no real holdout.

### 6.2 Representation probes — what the encoder and the GRU hold

Behaviour matches the decoupled algorithm (§1.4); the mechanism is
inferred. Four probes, on synthetic lattices (`gbook_at`, no scaffold),
run on the **frozen-decode `full`** (s0, s1), **S1**, **S2**, and the
raw-code `full` as a null; `analysis/b2_probes.py`, ~1–2 h total.

| probe | what | reads as |
|---|---|---|
| **P-Δ** | linear regression from encoder features to the rotated displacement Δ′ (and to per-module phase differences) | the decode is explicit if R² ≈ 1; where it lives (encoder vs GRU state) |
| **P-θ** | linear regression from the GRU hidden state to `(cos θ, sin θ)` of the current lifetime, by step in episode 0 and by episode | an explicit frame estimate if decodable; its sharpening over steps is the measurement curve |
| **P-swap** | run a lifetime at θ₁, switch the code to θ₂ mid-lifetime without resetting `h` | a decoupled estimator keeps un-rotating by θ₁ — wrong by exactly θ₂ − θ₁ — until it re-measures; an entangled solution fails otherwise |
| **P-act** | feed a wrong `prev_action` for one step | if the frame comes from `(Δcode, action)`, the output rotates by a predictable amount; `rec`'s insensitivity is the control |

Also on `dist@90` and A1 (P-Δ only): whether the rule's residue lookup
is linearly readable, and where it breaks past |Δ| = 19. And on B3-2:
what the episode-1 transient on trained orientations is (§1.6) — P-θ by
episode should show whether the frame estimate is disturbed by the goal
change.

### 6.3 Open, lower priority

- From scratch with mix 0 after the warm-up (pure random lattices) — the
  09-13 attempt failed for the lr reason, not the mix.
- The scale arm (`s ~ U[0.7, 1.4]`), a second in-context parameter.
- A second seed of S1/S2.
- Range: no network here decodes past ±19; a Fourier-feature front end
  could learn the full-range map (linear on the circle) — a different
  question.
- The continual plot (§4).

---

## 7. Code map

Everything below exists and is tested (`hopfield_nav/tests/test_goal_pairs.py`,
29 tests; `test_lattice.py`, 17; layering unchanged).

| what | where |
|---|---|
| lattice synthesis and reading: `gbook_at`, `module_phases`, `code_phases`, `torus_centroid`, `wrapped_phase_diff`, `crt_displacement` | `gridcode/lattice.py` |
| input layout (`rnn_input_layout`; new channels append: `xy_state`, `goal_grid_state`, `goal_sensory`, `lattice_oracle`), `build_rnn_input`, `table_gather` (per-row tables), `collect_rollout_rnn(gbook_table=)` | `policy/agent_rnn.py`, `rollout/rnn.py` |
| trunks: `FeedForwardCore(norm=)`, `EncodedRecurrentCore(skip, detach, bypass)`, `build_recurrent_core` | `policy/recurrent.py` |
| A's model `PairRegressor` | `policy/pair_regressor.py` |
| config fields `input_goal_grid_state`, `goal_sensory`, `input_sensory`, `sensory_mode`, `input_xy_state`, `input_lattice_oracle`, `input_encoder_*`; `region_val_frac`, `pairs_per_env`, `resample_goal_on_reach` | `config.py` |
| split: `region_cells`, `CellSets`; `generate_split(region_frac=)` | `world/spec.py`, `world/generate.py` |
| per-row goals: `VecEnv._goals`, `set_goal_pool`, `set_goals`; `_at_goal_l2` on `(B, 2)`; oracles on `(B, 2)` | `world/vec_env.py`, `world/env.py`, `rollout/oracles.py` |
| pairs, targets, the table, reference lines, `RNNAgentAsPairModel`, by-distance, mismatched walls | `evaluation/goal_pairs.py` |
| readout 2 `evaluate_lifetime_direction(gbook_table=)`, `ep0_by_step` | `evaluation/lifetime.py` |
| the scripted estimator | `evaluation/scripted_frame.py` |
| world/mode setup, `EnvSet.lattice_gbook/with_lattice`, `LatticeSampler`, `ARMS` | `training/goal_pairs_setup.py` |
| trainers and launchers | `train_goal_pairs.py`, `train_goal_lifetimes.py`, `eval_goal_pairs.py`, `run_goal_pairs.sh`, `run_goal_lifetimes.sh`, `run_eval_goal_pairs.sh` (all take `EXTRA=`) |
| collectors | `analysis/goal_pairs_results.py`, `analysis/b2_results.py` |
| pre-flight | `scripts/goal_nav_preflight.py` |
| probes (ad hoc, 09-14) | job tmp: `range_probe.py`, `range_probe_corner.py`, `band_probe.py`, `probe_encoder.py` — to be folded into `analysis/b2_probes.py` (§6.2) |

Not changed: the Hopfield stack, `bc_rnn_update`, `VecEnv.step_batch` and
the at-goal contract, `train_rnn.py`'s protocols; every existing
checkpoint and `world.json` loads. Ray-axis encoders (`sensory_encoder`)
were never built — A2 made them unnecessary.

Fixed settings: `size 20`, `lambdas 11 12 13`, `fwhm_ratio 0.25`,
`observation_size 120`, `wall_resolution 1`, 64 train / 16 H-env / 8
`same` envs at `place_margin 20`, `goal_val_frac 0.2`,
`region_val_frac 0.1`; A: 64 × 512 pairs per update; B: 64 lifetimes ×
8 envs, 64-step chunks, 32 chunks per lifetime, 60-step episode cap,
`init_log_std −1`; eval: 64 lifetimes × 20 episodes on 8 envs per set.

---

## 8. Sanity gates

Each is a script or a logged number with a pass condition; a failed gate
stops the wave it guards.

**Pre-flight** (`scripts/goal_nav_preflight.py`, once per world):

| id | check | pass |
|---|---|---|
| C1 | split invariants: `region ⊂ goal_cells_val`, `start_train ∩ region = ∅`, H-env walls and boxes disjoint from train, `same ⊂ train` | all hold |
| C2 | exact-twin rate of `omni` and `gbook` per env | 0 |
| C3 | teacher vs brute-force recomputation on 10k pairs | exact |
| C4 | uniform-random baseline on enumerated pairs | 90° ± 2 / 0.48 ± 0.02 |
| C5 | NN decoder exact on train × train in every mode; ~5° on xy held-out | as stated |
| C6 | `pair_inputs` bit-identical to `build_rnn_input` at `prev_action = 0` | exact |
| C7 | grid-code non-invariance: cosine of `gbook(p) − gbook(p + d)` across base points | well below 1 (−0.03 measured) |
| C8 | min toroidal gap between every train / H-env / `same` box | ≥ `place_margin` |

**During A** (every eval): C9 loss and metric move together; C10 `same`
≈ train; C11 teacher / random / NN recomputed on the scored pairs (a
metric bug shows as the teacher failing); C12 exposure counter; C13 A0
≤ 5° / ≥ 0.98 everywhere before A1.

**Before B is read**: C14 per-row goals under the default reproduce a
recorded rollout bit-for-bit; C15 every reset row's goal is in the pool
and ≠ its start; C16 readout 2's (episode 0, step 0) equals readout 1;
C17 lifetime horizon covers the eval's 20 episodes; C18 sampled-vs-mean
gap reported; C19 a rise is not confined to the first BPTT window.

**B2**: C1–C6 of §5.4; plus the lattice histogram in every run's final
JSON — zero lifetimes inside the held-out band (or the mix fraction).

**Before any comparison**: C20 same env set, cell set and enumeration
(enforced by the table's key); C21 checkpoint selected on H-env
train × train only; C22 seed spread reported, single seeds labelled.

Two evaluator properties to read every B curve against: continuous
at-goal is an L2 ball of 0.5, so a row can stand on the goal *cell*
without being at goal and the teacher there is the zero vector (scored
90° for any agent; the estimator takes a random step in that case); and
readout 2 samples actions, so a policy's floor is ~6–8°, not 0.

---

## 9. Predictions and outcomes

| | prediction | outcome |
|---|---|---|
| P1 | A0 solved in < 200 updates | ✓ 0.3° |
| P2 | (withdrawn: assumed a translation-invariant code) | — |
| P3 | A2 near random on held-out walls (no cross-correlation in a linear read) | ✗ 5.5° — the map is learned directly (P11 ✓) |
| P4 | encoders close A2's gap | not needed |
| P5 | `dist ≈ A` on H-env | ✓ in B1x's static table (22.5 vs 44 is the loss hedge, not the sampler) |
| P8 | discrete and continuous rank arms identically | ✓ |
| P9 | region × region the worst cell | ✗ every cell equal |
| P10 | A1x: outside within 0.1° of inside | ✗ 44° outside |
| P12 | B1x: `full` rises by episode outside the corner | ✗ flat, −0.24°/ep |
| P13 | D1: `dist`'s gain is short-range; A1y reproduces it | half: the gain is the NLL hedge, not the distribution |
| P14 | D2: mismatched walls well below random, flat in distance | ✓ 22°, flat |
| P15 | B2 gates pass | ✓ 1.0° / 0.0° from step 3 / 90° |
| P16 | B2 `full`: R1 ~90°, R2 falls in episode 0, final 10–25° | shape ✓ and level ✓ — but only with the decode decoupled; the raw-code GRU stayed at 89° |
| P17 | `rec` between `full` and `dist`, nearer `full` | partly: 45° at e19, but by cross-episode accumulation only — no within-episode measurement, a different shape from `full` |
| P18 | B2-mix on the corner | superseded by B3 (§6.1) |
| P19 | B3 (1): a corner-trained decode with corner translations learns the rule | ✓ `far@90` 1.6° in B3-2's warm-up (the pure-MLP control stalled on the plateau; second seed running) |
| P20 | B3 (2): `heldout_out` at θ = 0 falls to ~20° within a lifetime if P19 holds | ✓ 14° by episode 4, 14 by step 5; `far@0` the same |
| P21 | probes: θ is linearly decodable from the GRU state after 1–2 steps in the frozen-decode and S1 models; Δ′ from the encoder at R² > 0.95; P-swap error ≈ θ₂ − θ₁ for a few steps | open |

---

## 10. Lessons and risks

Design traps met in B2, each caught by a null that moved:

- **A randomisation held fixed longer than the weights take to fit it is
  not a randomisation** — per-env lattices were fitted in weights over the
  ~256 updates a lifetime lasted; per-row lattices fixed it.
- **A fixed structure the weights can index is a leak** — env offsets
  under rotation-only lattices; translation closes it.
- **A shared encoder under a recurrent gradient collapses** — dead units,
  then input-independence, then loss spikes even on determined data; a
  10× lower encoder learning rate and a determined-data warm-up fix it.
- **Check that the null moves for the right reason** — the memoryless
  arm's falling loss was memorisation, not a leak.
- **Placement lattices leak coordinates** — held-out envs drawn from the
  training lattice share its coordinate bands under dense packing.
- `1 − cos` on a normalised output plateaus when the conditional mean is
  0; constant lr destabilises long runs (step at 70%); evaluate uncertain
  policies sampled, never by the mean.

Out of scope, by design: multi-goal memory, capacity, interference — the
axes the attractor is built for. Telling the network the goal removes
them; if a plain net wins here, the attractor's claim relocates to those
axes rather than disappearing.
