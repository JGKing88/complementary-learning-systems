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

**What "corner scale" means — the dense-tiling control (A1xd, 09-16).**
The same recipe with 384 envs tiling the corner edge to edge at fixed
placement (every coordinate value in [0, 400) seen, pairs still within
an env) reaches **0.7–1.7° outside the corner** (two seeds; A1x: 44°), and the
decode probe puts it at the rule everywhere (far rect 1.2–2.4°, the
rule's range profile). So it is not the absolute
codes being fixed, nor how many positions are seen, but how they are
arranged: A1x's 64 envs sat on a pitch-47 placement lattice, so the
seen X values formed 12 clusters of 20 and "which cluster, where in it"
was a cheap absolute coordinate that undercut the residue rule. One
contiguous run of 400 values (A1xd) or 64 scattered clusters (A1) offer
no such coordinate, and the rule wins. B3-1b's corner-confined
translation worked by filling the corner, not by decoupling cells from
codes. The lookup is a *structure* effect — few clusters — not a
quantity effect; §6.3 tests that directly: on 8 scattered clusters
neither input noise nor a smaller network recovers the rule (noise
gets the per-module half, small networks build a smaller table).

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

**The decode learned from the corner is the rule.** The memoryless
control — `dist` alone, every lifetime at θ = 90° with corner-confined
translations — reaches `far@90` **0.3°** enumerated at u = 3000 (0.8° by
u = 800; coordinates 300 cells outside the corner on both axes), with `far@45` at 45° and `far@0`
at 90°, the |θ − 90°| signature; B3-2's warm-up phase gives the same
(1.6°). A1x's fixed placement on the same corner produced the lookup
(44° outside); corner-confined translation produces the rule — because
it fills the corner: the dense-tiling control (§1.2, A1xd) gets the rule
from a fixed placement too.

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

### 1.7 What the encoder and the GRU hold (probes, §6.2)

Linear readouts on synthetic lifetimes (`analysis/b2_probes.py`, log
2026-09-16), on the frozen-decode `full` (s0, s1), S1, S2, B3-2, the
raw-code `full` as the null and `rec` as the no-action control:

- **Encoder → code-frame displacement, explicitly.** Ridge from the
  encoder's features to `R_θ(g − p)`: R² 0.986 (frozen trunk), 0.998
  (S1), 0.999 (B3-2), 0.991 (S2); per-module phase differences at R²
  0.88–0.99; the env-frame direction at R² 0 (needs θ). A1's features read
  the same at every orientation. The readout breaks past |Δ| = 19 where
  the behaviour does. The jointly trained encoders are cleaner than the
  frozen `dist` trunk (1.0° / 0.8° vs 3.2°).
- **GRU state → θ, linearly.** R² 0.86–0.93 in late episodes; measured
  in the first 3–5 steps of episode 0 in step with the policy (frozen s0:
  82 → 50 → 37 → 22 → 14° at steps 0/2/3/5/10, policy 83 → 30 → 21 → 15 →
  13; S1: 88 → 33 → 19 → 12 → 11) and refined to 6–7° over the lifetime.
  The same state carries the un-rotated direction the head reads (9–11°,
  R² 0.92–0.96). S2 (detached encoder) holds θ far less linearly with an
  equally good policy. `rec`: cross-episode only. Raw: nothing.
- **P-swap: a decoupled estimator.** Switch the lattice by +90° at
  episode 10 with `h` kept and every model's action is off by **+87 to
  +90°** — the swap angle — for the whole episode, then re-measures over
  5–10 episodes: the frame estimate is sticky once formed (the first
  measurement took ~5 steps). Raw: no effect (89° throughout).
- **P-act: the frame comes from (Δcode, prev_action).** One step of
  `prev_action` rotated by +90° rotates the output in the predicted sense
  by up to +57° at step 1 (frozen s0), +15–25° at step 3, ~10–15° at step
  6, and 0° by episode 5, with a 3–5-step tail — evidence weighted down
  as the state fills.
- **B3-2's episode-1 transient** is a frame-estimate transient: its θ
  readout stalls over episodes 1–3 (28 / 18 / 17° vs S1's 18 / 13 / 9)
  in step with the policy dip, and is gone by episode 4. Probably the
  reset's teleport entering the (Δcode, action) integrator; untested.

The mechanism inferred from behaviour in §1.4 is what the state holds.

### 1.8 The decode from self-motion alone (phase 1, §6.4)

A1's network and loss, trained on pairs of steps of random walks in 64
scattered envs with the walker's own recorded displacement as the
target — no teacher, no goal, no position (log 2026-09-16):

| arm | held-out envs, 131M env-steps | env-steps to 10° / 5° / 2° / 1° |
|---|---|---|
| grid code, displacement-balanced pairs, two seeds | **0.48 / 0.46°** (every quadrant 0.5) | **8.2M / 11.5M / 18–20M / 39–41M** |
| grid code, raw random-walk pairs (k ≤ 30), two seeds | 8.2 / 10.4° (train = held-out) | 62M / – |
| grid code, 32 envs, raw pairs | 9.1° | 40M / – |
| grid code, balanced, 8-heading labels | 10.7° = the label's floor | 10° at 8.2M, as the full label |
| ray-cast views (no scaffold), balanced, two seeds | 5.7 / 6.3° | 18–25M / – |
| A1, teacher-labelled i.i.d. pairs (reference) | 0.23° | ~50M pairs to ≤ 1° |

- **A plain MLP acquires the translation-invariant decode from its own
  motion**, to 0.5° on envs it never walked (decode probe: never-walked
  coordinate values 0.6°, far rect 0.6°, A1's range profile to the
  degree), at 41M env-steps to 1° — the same order as A1's ~50M
  teacher-labelled pairs. The supervision that was "free" is enough.
- **The one condition is which experiences it learns from.** A raw
  random walk's displacements are mostly 1–5 cells; the decode learned
  from them is the rule out to ~10 cells and 8–10° on the enumerated
  table, train = held-out — a data limit, not a generalisation limit.
  Keeping pairs uniform over displacement size (the walker's own choice;
  no new information) gives the full decode.
- **A coarse label is enough.** Eight headings per pair learn the rule
  on the same curve as the full direction, to their own 11° floor.
- **No scaffold at all**: from views, balanced self-motion reaches
  5.7–6.3° on unseen walls (A2's teacher number: 5.5°).
- **Against the encoder.** Agent-HaSH's "encoder" is the same kind of
  object — a learned decode of the grid code (its direction signal is
  `W · (z(goal) − z(current))`, `z = encoder(gbook)`, `W` the local
  frame from the scaffold at the agent's true coordinates) — trained on
  a proximity bit per pair over 60 patches of 100×100 positions (600k
  unique positions, 600M draws). Phase 1 gets the full decode from 64
  arenas of 400 cells (25.6k unique positions), 41M env-steps, a
  direction per pair from odometry, and no frame.
- **The encoder's objective on the same walks (09-17; `encoder_training.train
  --walk_data` on `hopfield_nav.dump_walks` dumps — its own trainer,
  its own batching; figures `$CLS_RUNS/figures/nn_control/p1_size20.png`,
  `p1_size50.png`).** One walker per arena, its visited cells as rows,
  near/far masks over the walker's own moments (odometry labels; with
  perfect odometry identical to true-coordinate labels), read out with the
  harness's frame projection on the decode's held-out arenas. The encoder
  reaches its ceiling from **0.5–2M env-steps** — as soon as the walker
  has covered its arena: 50×50 **10.5 → 8.8 → 8.6°** within 19 cells (24.5
  → 20.2 → 20.1° within 49) at 0.5M / 2M / 8M steps; 20×20 **16.3°** (with
  radius 10; radius 20 covers the arena and gives 62°). At those step
  counts the decode is near chance; it reaches 10° at ~10M steps, 1° at
  40–60M, and 0.5° at 131M at every range it trained on. **A crossover:**
  proximity structure is learned from very little experience, the full
  displacement table from 10–50× more, and the table ends 15× more
  accurate within the encoder's radius and 40× beyond it. (A first
  reimplementation of the encoder loop differed from its trainer in batch
  composition and produced eroding 24–48° readouts; superseded — log
  09-17, correction.) Their trainer verbatim on 64 patches of 50×50
  reproduces the pre-trained att0.5 (7.1° / 8.2° / 19.0°); att0.5's own
  25-patch recipe reruns at 15–17° for the same seed. Trained **online**
  on the decode's axis (`--buffer visited`: one walker per arena, 32,768
  new steps per update, batches from the cells visited so far) it ends at
  24.9° (20×20) and 14.7° / 27.6° (50×50 within 19 / 49), 6–8° above
  their trainer on the same walker's dumps, and its early readout follows
  its gain schedule rather than its data — the dump points are its honest
  early numbers. With the decode on the same visited-set buffer (one
  walker per arena; identical rows into both models) it reaches 0.45° /
  0.51°, 1° at 34M / 46M steps — indifferent to the replay policy. The
  encoder on the decode's sliding window (one walker's recent moments per
  arena per batch) is 24.0° at 20×20 (as on the visited set) but 23.8° /
  45.8° at 50×50 (vs 14.7 / 27.6): a compact recent window starves its
  near/far objective of far pairs, which the decode's direction target
  does not need. Balancing the encoder's rows (endpoints of
  displacement-balanced pairs) helps at 20×20 (19.6°) but not at 50×50
  (28.7 / 48.4°) — the window's positional coverage matters too. With no
  memory at all (a fresh 512-step rollout per update, nothing kept) both
  models fail: decode 0.87° / 47°, encoder 62° / 73–80° — recent
  experience kept eligible for ~20 updates is load-bearing. Clean figures
  (window / visited set / no memory, both models): `p1_size20_clean.png`,
  `p1_size50_clean.png`.

### 1.9 Standing conclusions

1. A memoryless network does not learn the attractor's given frame; it
   learns the code it was shown — the rule on differences from scattered
   or dense coverage (fixed or translated), a lookup when the seen
   positions form a few clusters that give a cheap absolute coordinate —
   and neither extrapolates in displacement range.
2. On the real code, no training regime can make history build a frame,
   because the weights always have the shorter route.
3. Remove that route and a recurrent network learns an unseen code's
   frame from its trajectory — as a two-part computation, memoryless
   decode then in-context rotation, learnable jointly from scratch — and
   with the decode learned from a corner under corner-confined
   translations, it does so in a region it never saw: ~150° → ~15° over a
   lifetime, ~20° in ten steps, ~14° in five on B3.
4. What the attractor has built in — a translation-invariant code and a
   frame — a plain network can acquire: the invariance from positions
   that leave it no cheap absolute coordinate — and from nothing but its
   own motion, at the cost of the teacher-labelled version (§1.8) — the
   frame from lifetimes that deny it a fixed lattice. What it does not
   acquire from either is the full-range decode (§1.2).

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
(1) the pure-MLP control's first seed never left the loss plateau
(stochastic; the second seed did, at u = 400, and gives 0.8° outside),
(2) from scratch answered both questions — the rule from the corner, the
frame in the unseen region — so (3) is not needed. Design kept here for the record.

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

### 6.2 Representation probes — what the encoder and the GRU hold (done; results in §1.7)

Behaviour matches the decoupled algorithm (§1.4); the mechanism was
inferred; run 2026-09-16, every prediction met. Four probes, on synthetic lattices (`gbook_at`, no scaffold),
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

### 6.3 Memorisation test — is the shortcut about data quantity or data structure? (run 2026-09-16; result below)

Asked 2026-09-16, after the dense-tiling control (§1.2): *if the corner
MLP's failure was memorisation, can an MLP trained on even less data than
the corner — not corner-shaped — generalise, given input noise or a much
smaller network?*

**What the controls say the shortcut is.** A1x (64 envs on a 12 × 12
placement lattice inside the corner) took the lookup; A1xd (384 envs
tiling the same corner edge to edge, fixed placement) and A1 (64 envs
scattered) took the rule. What the lookup needs is a *cheap absolute
coordinate*: with the seen X values in 12 clusters of 20, "which cluster,
where in it" is a 12-way plus 20-way read, cheaper than the residue
rule; with one contiguous run of 400 values, or 64 scattered clusters, it
is not. So the lever is the number of clusters the training positions
form, not the number of positions. Few scattered envs = few clusters =
the cheap coordinate: the regime in which to ask whether noise or
capacity can push the network back to the rule.

**Runs (A1m).** Scattered placement (A1's `anywhere`, margin 20), 16
held-out envs anywhere, 8000 updates, A1's schedule, pairs per update
held at ~32k.
- *Wave 1, the baseline curve:* 5×768 at K = 4 / 8 / 16 training envs
  (1,600 / 3,200 / 6,400 cells; A1 and A1x had 25,600). Where does the
  big network switch to the shortcut?
- *Wave 2, at the largest K where the baseline fails:* input noise on
  the bump inputs, σ = 0.1 / 0.3 (peak 1) — the exact template becomes
  unreliable while phase differences survive; smaller networks 2×64,
  2×128, 3×256; dropout 0.2 as a third regulariser. Two seeds where an
  arm crosses.

**Readouts.** `heldout` (new walls, new positions) enumerated; the
decode probe (`analysis/decode_probe.py`): far rect, pairs split by
whether their coordinate values were seen, error by |Δ|. The lookup and
the rule are told apart by the seen/unseen split, not by the mean.

**Predictions.** P23: the 5×768 baseline is at the lookup by K = 8
(≥ 30° on unseen coordinate values, < 1° on seen). P24: noise at σ = 0.3
moves it to the rule at the same K. P25: a small network either fails
outright (no room for either solution) or takes the rule — the outcome
worth having is a small network at the rule where the big one is at the
lookup. P26: dropout does less than input noise (it perturbs the
features, not the template).

**Result (log 2026-09-16).** Error on far-rect pairs whose coordinate
values were never in a training footprint, at K = 8 (3,200 cells):
baseline **84°** (K = 4: 83°, K = 16: 70°, A1's K = 64: 0.3°); input
noise 0.1 → 56°, noise 0.3 → 44°, dropout 0.2 → 61°; 2×64 / 2×128 /
3×256 → **89° / 89° / 87°** while fitting train to 1–5°. The range
profile says what noise buys: the per-module, short-range decode (4–6°
at |Δ| = 1, from 66°) and not the cross-module combination (68–75° at
|Δ| = 10, A1x's Chinese-remainder band). **Answer: no.** On few
clusters neither regularisation nor capacity recovers the rule; smaller
networks build a smaller table. The residue-to-displacement table needs
the phase triples seen in enough combinations — coverage structure
(A1xd, A1), not regularisation. P23 ✓, P24 ✗ (partway), P25 ✗, P26 ✓.

### 6.4 Phase 1 — the self-taught decode (the sample-efficiency comparison with Agent-HaSH) (run 2026-09-16; results in §1.8)

Asked 2026-09-16: a direct sample-efficiency comparison between the MLP
and Agent-HaSH. Worked through with Jack; what survives is this.

**Why not an RL comparison.** The two systems differ in exactly two
places. The *memory*: a learned associative map in Agent-HaSH versus, for
an MLP agent, a goal slot written by the harness at contact — and for one
goal per env the slot *is* the memory, nothing to learn on either side.
The *decode*: analytic (scaffold geometry) in Agent-HaSH, learned in
weights for the MLP. An MLP agent with a pre-learned decode fed as a 2-D
channel sees the same input as Agent-HaSH's `q`; a controller trained on
either learns to follow a vector, and the curves would overlap for a
reason unrelated to the question. So the only content is **the cost of
acquiring the decode** — and the version worth having is the one that
needs no teacher, no goal and no position: learned from the agent's own
motion, the way the encoder is learned from proximity.

**Why many envs.** Agent-HaSH's one-arena headline is a statement about
its controller; the env-specific parts (encoder, many envs; decode, built
in) were paid for elsewhere. The MLP's decode pays the same way. One env
gives a within-env table (§6.3); re-placing one env at random scaffold
offsets would give the rule (it is B3's translation) but injects the
attractor's frame prior through the data. Many envs is what an agent
experiences.

**Data.** Goal-free random walks: N scattered training envs (A1's
placement, margin 20; N = 64 and 32), 16 held-out envs never walked.
Discrete unit steps, uniform random actions, walls block, `goals_active`
off (no goal, no reward, no teleport). 8 walkers per env × 64 steps per
update, appended to a replay buffer of the last 20 updates. Experience
is counted in env-steps walked.

**Pairs and target.** Steps `(t, t + k)` of one walk, `k ~ U[1, 30]`,
kept if the displacement is non-zero and within Chebyshev 19 (A1's
range). Input `[code(p_t), code(p_{t+k})]`; target the unit vector of the
recorded displacement — the walker's own odometry. 32k pairs per update
(A1's batch), sampled from the buffer.

**Model and loss.** A1's, unchanged: 5×768 ReLU, `1 − cos`, Adam 1e-3
with the step to 1e-4 at 70%. Two input modes: **grid** (`gbook`, the
encoder's own input — the like-for-like arm) and **regular** (`omni`
views, learns perception and geometry together — no scaffold at all, the
reference for what a network gets from self-motion with no code).

**Evaluation.** Every 50 updates on the held-out envs: the enumerated
direction error over every pair (readout 1, A1's metric), and the far-rect
probe on coordinate values never walked. Reported: env-steps to 5° and
1° on held-out envs, final error, seen/unseen split — against A1's
supervised number (~50M teacher pairs, 64 envs) and the encoder's cost
on the same axis.

**The encoder, precisely (read off the headline checkpoint,
`sweeps/ur_loss2_repel_low/029`).** It is not perception: its input is
the grid code (`Phi_flat`), and Agent-HaSH's direction signal is
`q = W · (z(goal) − z(current))` with `z = encoder(gbook)` and `W` the
Gram-Schmidt frame of the encoded codes at the two neighbouring scaffold
positions of the agent's *true* coordinates (`rollout/signal.py`,
`scaffold.project_displacement`). So encoder + projection is a learned
displacement decode of the grid code — the same object phase 1 learns —
with three differences: the supervision is a near/far proximity bit per
pair (near = within 10 cells; `mse_contrastive`, attract 2, repel 2), the
local frame is supplied by the scaffold (privileged), and the readout is
linear in the embedding. Its cost: a 4×512 GELU MLP → 1024-d, trained on
**60 patches of 100×100 positions** (600k unique positions; our 64
arenas hold 25.6k), batch 8192, 1000 epochs ≈ 600M position draws;
unique radius 21–28 cells (the decode's range, against our 19). The
comparison is therefore like-for-like in what is learned; it is reported
as unique positions covered and samples drawn, with the supervision and
the frame stated. Perception, in Agent-HaSH, is not learned at all: the
memory stores it.

**Arms.** (1) grid, 64 envs, two seeds — the number. (2) grid, 32 envs —
the env count. (3) regular, 64 envs, two seeds — total learned content.
(4) grid, 64 envs, target quantised to 8 headings — label richness.

**Predictions.** P27: arm 1 reaches ≤ 1° on held-out envs, at an
env-step cost within 3× of `dist@90`'s rollout BC (~600 updates). P28:
32 envs lands between A1m's K = 16 (70°) and 64 (rule) — most likely the
rule, at a slower curve. P29: regular mode reaches A2's ~5° from
self-motion alone. P30: 8-heading targets cost ≤ 2× the steps of full
directions — the decode is not label-limited.

**What is fed to each model, exactly (plain terms).**

*Data, shared.* 64 arenas (20×20 or 50×50 cells) at random spots on the
scaffold. In each, a walker takes random unit steps (walls block). Every
step is one record: `(arena, cell, grid code of that cell)` — the 434-number
bump vector the agent's scaffold gives. No goals, no rewards, no true
coordinates reach either model.

*The MLP (decode).* Input: two codes from one walk, `[code(p_t),
code(p_{t+k})]` (868 numbers). Target: the direction the walker actually
moved between the two moments — the sum of its own steps, as a unit
vector. Each of 4,000 updates: every walker takes 64 more steps (32,768
new records over all arenas) and 32,768 pairs `(t, t+k)` are drawn from
the recent history, chosen so displacement sizes 1..19 (or 1..49) are
equally represented. Loss `1 − cos`. Test: in 16 arenas never walked,
every pair of cells `(p, g)` → `[code(p), code(g)]` → compare the output
direction with `g − p`.

*The encoder.* Input: one code (434 numbers). Output: a 1024-number
embedding `z`. No per-sample target: on a batch of 4,096 codes, for every
pair of rows from the same walker, if the two cells were within 20 cells
of each other (from the walker's own displacement between the moments)
pull the embeddings together (cosine → 1), otherwise push them apart
(cosine → 0); pairs from different walkers get no label; plus a small
coding-rate term that keeps the embedding spread. Its trainer
(`encoder_training.train`) reads a dataset from a file, so the walkers
walk for a fixed budget (0.5M / 2M / 8M steps) and every (walker, visited
cell, code) row is written to a dump (`hopfield_nav.dump_walks`); one
walker per arena, so "same walker" = "same arena". Each epoch shuffles
the N rows into batches of 4,096 rows — **one gradient update per batch,
the loss over all 4,096² pairs in it, of which the ~64 rows per arena give
~64 × 63 labelled pairs per arena (~260k per batch)** — for as many
epochs as give the same gradient budget as its real encoders (39k updates
at 50×50, 20k at 20×20; N = 160k / 25.6k rows). Test: embed every cell of
the held-out arenas and read direction the way the agent does — project
`z(g) − z(p)` onto the embedding's own east/north step directions at `p`
(`rollout/signal.py`) — same pairs, same angular error.

*The one asymmetry that matters:* the MLP's label is a direction, the
encoder's is a proximity bit; each objective is built around its own.

**Result.** §1.8. One design change on the way: raw random-walk pairs
(k ≤ 30) are mostly 1–5 cells apart and the decode learned from them is
short-range (8–10°); the arms were rerun on one code path with a
continuous per-walker history and `--balance_range` (pairs uniform over
|Δ| = 1..19, k ≤ 400). P27 ✓ (0.5°; 1° at 39–41M env-steps, ~A1's cost),
P28 ✓ for plain walks (32 envs = 64 envs at 9°; the balanced 32-env arm
was not run), P29 ✓ (5.7–6.3°), P30 ✓ (same curve to the label's floor).

### 6.5 Open, lower priority

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
| P19 | B3 (1): a corner-trained decode with corner translations learns the rule | ✓ `far@90` 0.3° enumerated (memoryless `dist`, seed 1; seed 0 stalled on the plateau), 1.6° in B3-2's warm-up |
| P20 | B3 (2): `heldout_out` at θ = 0 falls to ~20° within a lifetime if P19 holds | ✓ 14° by episode 4, 14 by step 5; `far@0` the same |
| P21 | probes: θ is linearly decodable from the GRU state after 1–2 steps in the frozen-decode and S1 models; Δ′ from the encoder at R² > 0.95; P-swap error ≈ θ₂ − θ₁ for a few steps | ✓ θ at 33–50° after 2 steps, 12–22° after 5; Δ′ at R² 0.986–0.999; P-swap +87–90° for the *whole* episode, not a few steps (§1.7) |
| P22 | A1xd: dense fixed tiling of the corner still takes the lookup (coverage alone is not enough) | ✗ 3.6° / 6.4° outside by u = 500, 0.7 / 1.7° final — the rule; the arrangement of the seen values is the lever (§1.2, §6.3) |
| P23–P26 | A1m memorisation test (§6.3) | P23 ✓ (84° unseen at K = 8); P24 ✗ noise 0.3 reaches 44°, the short-range half only; P25 ✗ small nets 85–89°, a smaller table; P26 ✓ dropout 61° ≈ noise 0.1 |
| P27–P30 | Phase 1, the self-taught decode (§6.4) | P27 ✓ 0.48 / 0.46°, 1° at 41M / 39M env-steps; P28 ✓ (plain walks: 32 envs 9.1° = 64 envs 8–10°); P29 ✓ 5.7 / 6.3° from views; P30 ✓ 8 headings learn on the same curve to their 11° floor |
| P31 | encoder objective on the same walks: within 2× of the decode's env-steps to its own 6° | ✗ in the other direction: in its own trainer it reaches 8.6–10.5° (within radius) from 0.5–2M steps, where the decode is near chance; the decode passes it at ~10M steps and ends at 0.5° (a crossover; a first reimplementation's 24–48° was a batching artefact) |

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
