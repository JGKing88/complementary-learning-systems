# Train/eval splits and OOD evaluation — analysis and design sketch

Status: **proposal**, 2026-08-07. Nothing here is implemented. Part 1 is an
audit of the data-generating process as it stands; Part 2 says what each
requested split costs; Part 3 sketches the machinery; Part 4 lists OOD probes
that fell out of the audit.

---

## Part 1 — What actually generates the data

### 1.1 The four independent factors

An "environment" in this codebase is the product of four things drawn in four
different places, from three different RNG streams. They are worth naming
separately because the splits the project wants are exactly one-factor
holdouts.

| factor | drawn where | from what |
|---|---|---|
| **wall pattern** | `world/env.py:159` | `GridEnv.rng` = `RandomState(seed)` |
| **goal (local)** | `world/env.py:170`, `env.py:231` | same `GridEnv.rng` |
| **scaffold position** | `world/scaffold.py:116` `_spread_offsets` | **global** `np.random` |
| **env size** | `config.py:24` `EnvConfig.size` | not drawn — one global scalar |

**Wall pattern.** `_wall_code = rng.choice([-1, 1], size=(4, size))` — four
walls, one ±1 bit per cell along each. This is the *only* source of
env-to-env sensory variation. `_codebook` (`env.py:167`) is then a pure
function of `(_wall_code, size, observation_size)`: a 120° foveal raycast,
`observation_size` rays, each returning the ±1 code of the wall segment it
hits. So env identity ≡ `(seed, size, observation_size)`, and two envs with
the same seed are the same env.

**Goal.** Drawn from the *same* rng immediately after the wall code
(`env.py:170`), so goal and wall pattern are entangled in one stream: you
cannot vary one without perturbing the other unless you set the goal
explicitly. `reset_goal()` (`env.py:231`) redraws uniformly over the whole
local grid, and `train_navigate.py:262` calls it every explore rollout when
the stage asks for it.

**Scaffold position.** `register_envs` assigns each env a top-left `offset`
into the `Npos × Npos` scaffold; `get_encoded_state` then reads
`encoded_Phi[local + offset]`. The offset is what "where in the grid
scaffold" means. `_spread_offsets` lays a `rows × cols` lattice over
`[0, Npos-size]²` and jitters each point by up to `0.4·(spacing-size)/2` —
using **global** `np.random`, not the per-run `RandomState`. That detail
causes two of the problems below.

**Size.** `cfg.env.size` is a single scalar shared by every env, the
`VectorHash` placement geometry, and the distractor-exclusion test. There is
no per-env size anywhere.

### 1.2 What is deterministic and what is not

`encoded_Phi` is a **pure function of `(lambdas, Npos, fwhm_ratio, encoder)`** —
`gen_gbook_2d` (`gridcode/codebook.py:13`) has no randomness at all. This is
already relied on: `prep_scaffold.py:49` content-addresses the cache on exactly
those four things. Consequence: the training world and the eval world share
one embedding field. Only the *offsets into it* differ.

The non-static `build_scaffold` path does consume global `np.random`
(`scaffold.py:83-86`), but only for `pbook`/`Wgp`/`Wsp`, which feed `recall()`
— not the embeddings the policy reads.

### 1.3 Train/val separation today

`setup_world` (`training/world_setup.py:35`) is called twice off one
`RandomState(cfg.seed)`: train worlds first, then the eval world. Val envs
therefore get different seeds → different wall codes and different goals.
**That part is sound, and it is the "no leakage" the project believes it
has.** But each call builds its *own* `VectorHash` and each runs
`_spread_offsets` over the **full** `[0, Npos-size]²`. Nothing keeps a val
env off a train env's patch.

How bad this is depends entirely on `Npos`, and the two regimes differ sharply.

At the *dataclass* defaults (`lambdas=[11,12]` → `Npos=132`, `size=20`,
20 train / 10 val envs):

```
10 / 10 val envs overlap at least one train env footprint
train footprints cover 45.9% of the scaffold
```

At the *working* configuration — which is what every launcher actually uses:
`lambdas=[11,12,13]` → `Npos=1716`, `size=20`, 80 train / 4 val
(`navigate_explore_min_s2`):

```
0 train/val footprint overlaps
train footprints cover 1.09% of the scaffold
min Chebyshev origin distance train <-> val: 39 cells (envs are 20 wide)
```

So at the scale the project actually runs, Euclidean position leakage is not
occurring today — there is room to spare. But nothing *guarantees* it, the
nearest train/val pair sits 19 cells apart edge-to-edge, and there is no way to
deliberately place a held-out region far from training. Since
`input_encoded_state` is on by default (`policy/channels.py:71`), the policy
reads the absolute position code directly, so this is a channel it could in
principle exploit. It needs to be declared and checked, not left to luck.

**And Euclidean distance is the wrong metric anyway** — see §1.6.

`train.py:244-253` makes it worse when enabled: `refresh_envs_each_update`
re-places the train envs with `placement="random"` every update, so over a
long run training sweeps the whole scaffold.

### 1.4 Val worlds are not reproducible from a checkpoint

`build_eval_world` (`evaluation/checkpoint_io.py:72`) rebuilds the "training-time
eval world" by replaying `RandomState(cfg.seed)` past a skip loop of
`envs_per_world × num_worlds` draws. That recovers the val **seeds** — hence
wall codes and goals — correctly. It does **not** recover offsets, because
offsets come from global `np.random`, whose state at that point in training
depended on everything the train world consumed first. Every post-hoc driver
reseeds independently (`eval_all.py:110`, `agenthash.py:392`,
`prep_scaffold.py:143` all do `np.random.seed(0)`).

Measured, same configuration:

```
val offsets @ training time : [(5,3), (10,35), (10,75), (3,106), (53,3), ...]
val offsets @ eval_all time : [(8,5), (9,39), (6,75), (6,111), (63,3), ...]
per-axis deltas up to 10 cells — half an env width at size=20
```

This is not leakage. It *is* the reason the position axis cannot currently be
controlled or even reported honestly: `eval_all`'s `scaffold_layout` plot draws
placements the training run never used. The skip loop is also silently fragile
— add one `rng.randint` anywhere in `setup_world` and every checkpoint's
"reproduced" val set shifts.

### 1.5 Where the size assumption is baked in

`observation_size` (the ray count, default 60) is **independent of grid size**,
and `compute_input_dim` reads only `(embed_dim, observation_size, flags)`.
So the policy's input width does not depend on env size — **size
generalization needs no architectural change at all.** What it needs is for
these sites to stop reading the global scalar and read `env.size` instead:

- `evaluation/metrics.py:321, 448` — distractor exclusion box
- `evaluation/metrics.py:613` — `grid_size` for coverage
- `training/exploit.py:62`, `training/explore.py:61` — distractor exclusion
- `rollout/collector.py:175, 335, 349, 524, 549` — `visited_cells`, novelty
- `analysis/trajectories.py:168, 284, 316, 339`
- `analysis/phase_decoding/rollout.py:129, 166`
- `world/scaffold.py` — `VectorHash.size` is one scalar used by
  `register_envs`/`_spread_offsets`/`_overlaps`; needs per-env size

`cfg.env.observation_size` sites must stay global — that one really is the
policy's input width.

### 1.6 Euclidean distance is not the right separation metric

The grid code repeats. `gen_gbook_2d` sets module *m*'s active row from
`x % λ_m`, so two envs at offsets *o* and *o'* have **cell-for-cell identical
codes in module m** whenever `(o_x - o'_x) % λ_m == 0` **and**
`(o_y - o'_y) % λ_m == 0` — no matter how far apart they are in the scaffold.
Module λ=12 is 144 of the 434 grid dimensions; sharing it exactly means a third
of the code is not held out at all.

Measured on the same working configuration (80 train, 4 val, `Npos=1716`, the
same offsets that showed **zero** Euclidean overlap):

```
module lambda=11: 1 of 320 train/val pairs share the module exactly
module lambda=12: 2 of 320   e.g. train (313,1675) vs val (1033,1267)
module lambda=13: 1 of 320
```

A val env 900 cells away can share 144/434 of its grid code, exactly, with a
train env. A Euclidean-margin check would pass it.

**But measured in embedding space, that turns out not to be a leak.** The policy
never sees `gbook`; it sees `encoded_Phi`, and the encoder is a nonlinear
434→1024 map. Cosine similarity between unit-normalized embeddings, working
encoder (`run_20260422_185816`, `out_dim=1024`, `fwhm_ratio=0.25`):

```
random pairs                          : mean +0.006  p99 +0.261  max +0.723
share module lam=11 exactly, >100 apart: mean +0.001  p99 +0.126  max +0.400
share module lam=12 exactly, >100 apart: mean +0.001  p99 +0.112  max +0.331
share module lam=13 exactly, >100 apart: mean +0.000  p99 +0.117  max +0.305
```

Exact module sharing leaves pairs **less** similar than a typical random pair.
One module of three, matched exactly, contributes essentially nothing after the
nonlinearity. A per-module phase rule would also be *infeasible* at the project's
env counts: the λ=11 phase torus has only 121 cells, so 80 train envs saturate
it. Requiring per-module phase distance ≥ 2 from every train env accepts **0%**
of candidate offsets at N=80 (18.7% at distance ≥ 1). Phase separation is
neither necessary nor achievable — drop it.

**What does predict embedding similarity is distance — toroidal, not flat.**

```
offset dx=  1: cos +0.999    dx= 20: cos +0.732    dx=200: cos -0.006
offset dx=  5: cos +0.980    dx= 50: cos +0.142
```

The correlation length is ~30–50 cells, i.e. **longer than one env** (size 20).
Two envs placed 20 apart still sit at cos ≈ 0.73 — the same patch, for practical
purposes. And because `Npos = prod(lambdas) = 1716` *exactly*, the window is a
full period: the scaffold is a **torus**, and `x≈0` is adjacent to `x≈1715` in
code space. The worst flat-far pair found was `(1715,987)` vs `(4,989)` —
flat Chebyshev distance 1711, **cos +0.972** — which is simply displacement
(−5,−2) on the torus.

**Residual long-range aliases survive even the toroidal metric.** Over 8M
random pairs at toroidal Chebyshev distance > 200:

```
max cos +0.850     ~0.5% of pairs exceed cos 0.3     p99.99 +0.703
```

So *no coordinate-space rule is sufficient*. Toroidal distance is the right
cheap pre-filter; the guarantee has to come from measuring cosine similarity on
`encoded_Phi` itself. (Independently of splits, that 0.5%-above-0.3 tail is the
Hopfield's intrinsic false-recall floor, and worth knowing for the distractor
work.)

Related: `Npos` must satisfy `Npos <= prod(lambdas)`, or two distinct scaffold
positions get *identical* codes in every module and the scaffold aliases
outright. `Npos=None → prod(lambdas)` sits exactly at the boundary, so runs are
safe — but `eval_all --Npos` can raise it past that with no warning.

### 1.7 A quarter of the env-identity bits are perceptually dead

The foveal cone is fixed North at ±60° (`env.py:18, 270-272`), so every ray has
`dy = cos θ ≥ 0.5 > 0`. Wall 2 (South) requires `dy < 0` and is therefore
**never hit by any ray from any cell**. Measured at `size=20`,
`observation_size=12` by flipping each wall bit and diffing the codebook:

```
wall N: 20/20 bits live, mean 140.2 / 400 cells affected per bit
wall E: 20/20 bits live, mean  34.9 / 400
wall S:  0/20 bits live   <-- dead
wall W: 20/20 bits live, mean  34.9 / 400
TOTAL live: 60 of 80 bits
distinct observations: 351 of 400 cells (12.3% perceptually aliased)
```

Effective env identity is **60 bits, not 80**, and they are unequally weighted:
an N-wall bit moves four times as many cells as an E/W bit. Any Hamming-distance
construction must be defined over the *live* bits or a quarter of its flips are
silent no-ops and the difficulty axis is meaningless.

### 1.8 Summary of the gap, per requested split

| requested split | wall patterns differ? | held out today? | blocker |
|---|---|---|---|
| new scaffold positions | yes | **no** — 100% footprint overlap at defaults, and offsets aren't reproducible | offsets uncontrolled + un-serialized |
| new env sizes | yes | **no** — one global size | ~15 `cfg.env.size` sites; no per-env size |
| unseen goal locations | yes | **no** — `reset_goal` draws uniformly over every cell | no goal domain concept |
| new wall patterns | yes | yes (by seed disjointness) | — |

---

## Part 2 — Implementation sketch: one env generator

**Agreed on the shape.** One generator, three consumers (train, in-training
validation, post-hoc validation), producing *specs* that a separate builder
turns into runtime objects. The spec/builder split is what makes the rest
possible: specs are serializable, checkable for disjointness, and rebuildable
later without re-running an encoder. Everything below is that one generator.

Three layers, strictly separated:

```
VectorHash       the existing class, built once, shared by every world/split
   └─ EnvGenerator   traits -> list[EnvSpec]        (pure, no torch)
        └─ WorldBuilder   EnvSpec + VectorHash -> GridEnv
```

**No new scaffold type.** `VectorHash` (`world/scaffold.py`) already is it. The
change is to remove a field, not to add a class — see §2.1.

### 2.1 Layer 1 — one shared `VectorHash`

`encoded_Phi` is a pure function of `(lambdas, Npos, fwhm_ratio, encoder)`
(§1.2). Today `setup_world` builds one per train world **plus one for the eval
world**, all bit-identical. At the working config that is not a rounding error:

```
Npos=1716, out_dim=1024, float32  ->  12.06 GB per copy
num_worlds=1 today therefore holds ~24 GB of duplicated encoded_Phi
```

Building it once and handing the same object to every world and every split is
a prerequisite for a shared generator *and* frees 12 GB per extra world.
`prep_scaffold.py` already content-addresses exactly this tuple; the trainer
should use the same cache instead of rebuilding.

**The cause is one field, not a missing abstraction.** `VectorHash` conflates:

- *the shared field* — `gbook`, `encoded_Phi`, `lambdas`, `Npos`,
  `module_sizes`: a pure function of `(lambdas, Npos, fwhm_ratio, encoder)`
- *per-world placement* — `env_offsets`, `size`, `register_envs`

A second world needs its own `env_offsets`, and the only way to get one is to
construct a second `VectorHash` — which drags a duplicate `encoded_Phi` along.
So the fix is to **delete a field, not add a class**: once `EnvSpec.offset`
carries placement, `vh.env_offsets` is redundant and the field object is shared
by reference.

**The cut follows a boundary the code already has.** State, by the method that
sets it:

| method | sets | depends on envs? |
|---|---|---|
| `build_scaffold()` | `gbook`, `module_sizes`, `pbook`, `Wpg`, `Wgp` | no |
| `precompute_encoded_phi()` | `encoded_Phi` | no |
| `register_envs()` | `env_offsets`, `Wsp`, `Wps`, `Ns` | **yes** |

Row 3 splits further, because in **static mode `register_envs` assigns
`env_offsets` and returns** (`scaffold.py:203-206`) — `Wsp`/`Wps`/`Ns` are never
assigned, do not exist as attributes, and have no `hasattr` guards anywhere. So
the env-dependent state is really two different things:

- **`EnvSpec.offset`** — single source of truth for placement, all modes.
  Putting offsets in a second object would duplicate them in the only mode that
  runs, with two copies to keep in sync.
- **`EnvAssoc{Wsp, Wps, Ns}`** — the sensory↔place weights fitted to one env
  set. Non-static only; simply absent in static mode.

So: one shared field object, offsets on the specs, and an `EnvAssoc` that exists
only on the non-static path. **This preserves non-static exactly** — each world
keeps its own `Wsp`/`Wps` and its own `_test_scaffold`, while `encoded_Phi`
(12 GB) and `gbook` (10 GB) are shared by reference. Merely sharing one
`VectorHash` would *not* preserve it: one object holds one `Wsp`, and a second
`register_envs` call overwrites the first.

`EnvAssoc` is **not** connected to the generator. It is fitted from
`env.fully_explore_random()`, so it needs *built* envs and sits strictly
downstream of the builder:

```
generator (pure -> EnvSpecs)  ->  builder -> GridEnvs  ->  [non-static] fit EnvAssoc
```

The generator depends on the shared field only — `Npos` and `prod(lambdas)` for
the wrap and capacity math, `encoded_Phi` for margin derivation and the cosine
diagnostic. One feedback edge exists and is an error path, not a coupling:
`_test_scaffold` raises below 95% grid recovery, so in non-static mode a
generated env set can be rejected at build time (remedy: `Np`,
`observation_size`, or regenerate).

Blast radius of the non-static path, verified:

- **Nothing outside `scaffold.py` reads it.** `VectorHash.recall` /
  `recall_batch` — the only consumers of `pbook`/`Wsp`/`Wps`/`Wgp`/`Wpg` — are
  called from exactly one site in the repo: `scaffold.py:503`, inside
  `_test_scaffold`. Every other `recall` hit is the *Hopfield* class
  (`rollout/signal.py:112`, `nav_eval/nav.py:114`) or the separate
  `encoder_scaffold.py` class in `analysis/scaffold_experiments/`. The policy,
  rollout and eval paths never touch it.
- **Infeasible at the working scaffold**: `pbook` is `(Np, Npos, Npos)` =
  1600 × 1716² × 8 B = **37.7 GB**, plus gbook 10.2 and encoded_Phi 12.1, plus
  32,000 Python-loop recalls in `_test_scaffold`. It is a small-`Npos` mode
  (0.22 GB at `Npos=132`).
- **Never used**: 354 of 355 run manifests are static — 326
  `static_vectorhash: true` plus 28 legacy `gbook_only: true`. Zero non-static.

Design rule that follows: **one `Placement` per env set, never merged.**
Registering train and val envs together would put 32k patterns through one
`_test_scaffold` instead of two smaller sets, and could fail where two separate
registrations pass.

Remaining cost: **27 read sites** of `.env_offsets` outside tests, nearly all
`vh.env_offsets[global_idx]`. Mechanical, but it is the bulk of this step.

Consequence for "a world is a full VectorHash scaffold": structurally true, but
the *content* is identical across worlds, so `num_worlds` is currently nothing
but a grouping of offsets over one shared field. **Decided: collapse it.** One
`VectorHash`, envs grouped into worlds for bookkeeping only. `num_worlds` stays
on the CLI so existing launchers parse, but it no longer multiplies memory.

### 2.2 Layer 2 — `EnvSpec` and independent trait streams

```python
@dataclass(frozen=True)
class EnvSpec:
    wall_seed: int                 # -> _wall_code -> _codebook
    size:      int
    offset:    tuple[int, int]     # top-left in scaffold coords
    goal:      tuple[int, int]     # local coords
```

The traits must come from **independent** RNG streams, because today wall code
and goal share `GridEnv.rng` (§1.1) and refreshing one perturbs the other. Each
stream is derived, not advanced:

```python
rng(trait, tick) = RandomState(stable_hash(run_seed, trait, tick))
```

Deriving per `(trait, tick)` rather than drawing from one running stream is what
makes "refresh placement only, hold walls and goals" reproducible without
replaying everything else. A running stream re-couples the traits.

`GridEnv` needs one new setter, `set_goal(pos)`, mirroring the existing
`set_position`. Construction stays `GridEnv(seed=wall_seed, ...)` so `_wall_code`
is bit-identical to today's for a given seed; the constructor's own goal draw
becomes dead entropy and is overwritten.

### 2.3 Layer 3 — domains, and `complement()` as the OOD mechanism

One domain object per trait, with `sample(rng, n)`, `contains(v)`, and
`complement(margin)`:

```python
PlaceDomain :  Anywhere | Rect(x0,y0,w,h) | Union[...] | Complement(d, margin)
GoalDomain  :  Any | Ring(w) | Interior(w) | Quadrant(q) | Cells([...]) | Complement(d)
WallDomain  :  SeedRange(lo, hi)
SizeDomain  :  Sizes([...])
```

`complement()` is exactly the "automatically use the other rule" behaviour asked
for. Both OOD requests reduce to it:

- trained with `goal=Ring(1)` → its goal-OOD val set is `Complement(Ring(1))`
  = `Interior(1)`, derived from the serialized train domain, not restated;
- trained with `place=Rect(x0,y0,w,h)` → every val set for that checkpoint gets
  `Complement(Rect, margin)` and cannot sample inside the training region.

Because the train domains are serialized (§2.7), a later eval derives all of
this from the checkpoint. Nothing has to be remembered by hand.

### 2.4 Mix and match — three levels per trait

Each trait has two things: a **region** (declared at train time; unrestricted by
default) and the **values** actually used inside it. That gives three levels:

| level | sampled from | is |
|---|---|---|
| `same` | the recorded train values (the D4 union) | memorization probe |
| `held_out` | `region \ train_values`, at margin | **base validation** |
| `ood` | `complement(region, margin)` | OOD probe |

- **Base validation during training** = `held_out` on all four traits: inside
  the declared region, disjoint from every value training used. That is the
  "different in every respect" requirement, and it is one call.
- **Post-hoc isolation** = any combination. "train scaffold locations, val goals
  and patterns" is `place=same, wall=held_out, goal=held_out, size=same`.
- **OOD is a post-hoc parameter**, per trait. Training only ever builds
  `held_out`. Later, `--ood place` switches that one trait to the region
  complement and leaves the others at whatever level was asked for — so an OOD
  number always has a same-checkpoint in-distribution control beside it.

`ood` is only available for a trait whose region was actually restricted at
train time. Asking for `--ood place` on a model trained with
`place_region=Anywhere` must be a clear error, not a silently empty set.

One generator, one call signature, all three uses.

### 2.5 What "no overlap" means, per trait

`novel` needs a decidable separation test against the train set. Per trait:

**wall** — disjoint seed ranges is the *mechanism*; the *check* must be on the
codebook, and per §1.7 only 60 of 80 bits are live (the South wall is never hit
by any ray). Assert no exact codebook collision; report min Hamming over live
bits as the margin actually achieved.

**goal** — disjoint cell sets **in env-local coordinates**. If any training env
ever places its goal at local cell (12, 9), no base-val env may use (12, 9).
The forbidden set is the union over all train envs and all refresh ticks.

This collides with goal refresh, and the collision forces **two branches at
build time** — which resolve to **one identical object** in the spec file.

At the working config the local grid has 400 cells and there are 80 train envs.
With goal refresh on, training consumes 80 fresh cells per update and exhausts
all 400 within ~5–10 updates; after that the forbidden set is the whole grid and
no legal base-val goal exists. So:

```python
if refresh.goal is not None:
    # Pre-partition the region: train can never exhaust it.
    S_train, S_val = split(region_cells, val_frac=0.2, rng)   # 320 / 80
else:
    # Train goals are drawn once at build; the rest is free.
    S_train = drawn_train_goals                                # <= n_envs cells
    S_val   = region_cells - S_train                           # ~320+
goal_domain_val = Cells(S_val)      # <-- both branches land here
```

Both branches serialize as a plain `Cells([...])` list, so **post-hoc val
generation is branch-blind**: it reads a cell list and samples. All the extra
logic lives at train time, once. Under the refresh branch, `reset_goal` samples
only from `S_train`, so the recorded union (D4) is a subset of `S_train` by
construction and `val goals ∩ train goals = ∅` holds by domain rather than by
luck.

`val_frac=0.2` is a default, not a decision. The split is a **random scatter**,
not a region — the local goal distribution stays matched between train and val,
which keeps ring/interior free as a separately declarable OOD region.

**size** — trivially disjoint sets.

**place** — **decided: Euclidean distance in scaffold coordinates.** Grid-space
proximity is explicitly not the criterion; two envs may be near in code space
provided they are far in real space. §1.6's measurement supports this — exact
module sharing is not an embedding leak.

Two details inside that decision:

1. **The distance must wrap.** `Npos == prod(lambdas) == 1716` exactly, so the
   scaffold *is* a full period: `x≈0` and `x≈1715` are the same point of the
   coordinate system, and the policy receives bit-identical encoded states
   there. This is not "close in grid space" — it is the same place. A flat
   check rates the measured worst case, `(1715,987)` vs `(4,989)` at cos +0.972,
   as 1711 cells apart. So: Euclidean, taken mod `prod(lambdas)`. One
   `np.minimum(d, PROD-d)`, same metric family, correct answer at the seam.
2. **The margin should be ~50, not ~20.** Embedding correlation length is 30–50
   cells (§1.6): two envs 20 apart still sit at cos ≈ 0.73, i.e. effectively the
   same patch. Today's de-facto train↔val separation is 39 cells (§1.3), which
   is inside that. Capacity is not the constraint — the full scaffold holds 841
   envs at margin 40.

Cosine similarity on `encoded_Phi` is **not** a gate. It is worth computing once
per generation and *reporting* (a 400×32k matmul, seconds on GPU): it costs
nothing and it is the only way to notice a long-range alias — max cos +0.850,
~0.5% of pairs above 0.3, which no coordinate rule catches. Reported, not
enforced, so the split stays encoder-independent and reproducible from
coordinates alone.

### 2.6 Capacity and the `Npos` check

The generator asserts feasibility *before* building, with an error naming which
constraint failed and which knob fixes it. Measured capacities, `size=20`:

```
full scaffold Npos=1716, margin  0 : 7225 envs
                         margin 20 : 1849
                         margin 40 :  841
Rect 200x200,            margin 20 :   25
Rect 400x400,            margin 20 :  100
Rect 800x800,            margin 20 :  400
```

So the standing 80 train + 4 val fits with room to spare on the full scaffold,
but a *restricted* OOD placement region can run out fast — a 200×200 region
holds 25 envs at margin 20, and fewer once cosine rejection bites. Checks:

- `Npos <= prod(lambdas)` (§1.6) — currently unchecked anywhere
- `capacity(domain, size, margin) >= n_envs`
- bounded rejection-sampling attempts, then a specific error: raise `Npos`,
  enlarge the region, lower the margin, or ask for fewer envs

### 2.7 Refresh — per-trait, with two consequences

```python
@dataclass
class RefreshPolicy:
    place: int | None      # every N updates; None = never
    wall:  int | None
    goal:  int | None
    size:  int | None
```

On a tick for trait *t*, resample only *t* from the **train** domain, holding
the rest. This replaces both of today's mechanisms: `refresh_envs_each_update`
(train.py only — **not wired into `train_navigate` at all**, so the current
trainer has no placement refresh) and the per-stage `reset_goal` in the explore
regime (`stages.py` `RolloutSpec.reset_goal`).

Two consequences to design around:

1. **Separation must be re-asserted at every refresh.** Today's refresh uses
   `placement="random"` over the whole scaffold (`train.py:253`), so a refreshed
   train env can land on the val region. Sampling from the train *domain* makes
   this structurally impossible, but the assertion should still run — it is
   cheap and it is the exact bug class this whole design exists to remove.
2. **Wall refresh is the expensive one.** Place and goal refresh are a few ints.
   Refreshing walls rebuilds `_codebook`: `size² × observation_size` raycasts in
   a Python triple loop (`env.py:276-283`) — 400×12 = 4.8k per env, ×80 envs =
   384k per tick at the working config. Needs measuring, and probably
   vectorizing, before it is promised at every-update cadence.

Note also that `batch_envs` shares one codebook and one goal across all B
episodes (`vec_env.py:31-32`), so it is parallel episodes *within* an env — it
cannot carry trait variation and is not part of this design.

### 2.8 Serialization

`world.json` beside the manifest, and the same dict embedded in the `.pt` under
`world_spec` so a bare checkpoint is self-describing:

```json
{"spec_version": 1,
 "scaffold": {"lambdas": [11,12,13], "Npos": 1716, "fwhm_ratio": 0.25,
              "encoder": {"path": "...", "sha256": "...", "out_dim": 1024, "gain": 3.699}},
 "domains":  {"place": {...}, "wall": {...}, "goal": {...}, "size": {...}},
 "separation": {"toroidal_margin": 50, "max_cos": 0.30, "wall_min_hamming": 12},
 "refresh":  {"place": null, "wall": null, "goal": 1, "size": null},
 "resolved": {"train": [EnvSpec, ...], "base_val": [EnvSpec, ...]},
 "spec_hash": "…"}
```

Both halves are load-bearing: the **domains** let a later eval mint fresh envs
(you will want 200 OOD envs, not the 4 that happened to exist at train time) and
derive complements; the **resolved** list lets `same` work and lets a val
sampler exclude what training actually saw.

`resolved.train` is the **union over refresh ticks**, not a snapshot — the set
of every value each trait ever took. At 300 ticks × 80 envs that is ~200 KB, and
it is what makes both `same` ("anything training saw") and `held_out` ("nothing
training saw") well-defined for a refreshed trait.

### 2.9 Env size OOD — verification result

**Verified: no architectural change needed.** `compute_input_dim` reads only
`(embed_dim, observation_size, flags)`, and `observation_size` is the ray count,
independent of grid size. A policy trained at size 20 can be run at any size.

**But it is not zero-work, and the failure mode is silent.** Six sites read the
global `cfg.env.size` where they need the env's own; at a val size ≠ train size
they produce wrong numbers rather than an exception:

- `metrics.py:613` — `grid_size` → `total_positions`, the **denominator of
  `mean_coverage`**. Coverage silently mis-normalized.
- `metrics.py:321, 448`, `exploit.py:62`, `explore.py:61` — distractor exclusion
  box; distractors get drawn from inside the env, or a band outside it is
  wrongly excluded.
- `collector.py:175, 335, 349, 524, 549` — `visited_cells` shape and novelty
  bookkeeping.
- `scaffold.py` — `VectorHash.size` is one scalar used by
  `register_envs`/`_spread_offsets`/`_overlaps`.

If val size is uniform (a val *set* at one size, differing from train), only the
scalar has to be threaded, not per-env sizes — materially less work than mixed
sizes within a set. See **D3**/scope.

### 2.10 Consumers, back-compat, tests

- `train_navigate` builds train + base-val from one generator call; writes
  `world.json`.
- `eval_all` / `agenthash` / `analysis.trajectories` grow
  `--world_spec PATH` plus per-trait level flags
  (`--place same --goal novel --wall novel`). Results keyed by the combination.
- Absent `--world_spec` → today's replay path, warned (offsets are not the
  training-time ones — §1.4).
- `scripts/backfill_world_spec.py` for the 309 existing run dirs: recovers wall
  seeds and goals via the replay, records `offset: "unrecoverable"` rather than
  inventing them.
- Tests: domain sampling determinism; round-trip equality on `_wall_code`
  arrays (not just seeds); separation assertions per trait; capacity errors
  fire with the right message; `complement(complement(d)) == d`.

### 2.11 Decisions taken, and what is still open

Settled 2026-08-07:

- **D1 — goal holdout is by env-local cell**, union over all train envs and all
  refresh ticks. Implemented as an up-front domain cap, not as
  exclude-what-was-used (§2.5).
- **D2 — placement separation is Euclidean in scaffold coordinates**, not phase
  or embedding distance. Wrapped mod `prod(lambdas)`; margin ~50; cosine
  reported but not enforced (§2.5).
- **D3 — `num_worlds` collapses** into one shared `Scaffold` with envs grouped
  for bookkeeping only (§2.1).
- **D4 — the refresh union is recorded**, so `same` means "any value that trait
  actually took during training" (§2.8).

- **D1' — the train goal cell set** is a random scatter, chosen by branch:
  80/20 pre-partition when goals refresh, `region \ train_goals` when they do
  not. Both serialize identically (§2.5).
- **D5 — training builds only the in-region `held_out` val.** OOD is a
  post-hoc per-trait parameter that switches one trait to the region
  complement (§2.4).

- **Place margin is an edge-to-edge gap derived from the scaffold**, not a
  function of env size and not a constant. `encoded_Phi` is
  `(Npos, Npos, embed_dim)` — env size never enters it. Correlation length is
  set by `lambdas` and `fwhm_ratio`.

  **Corrected 2026-08-07.** The first version of this bullet sampled *diagonal*
  displacement and concluded ~50. Diagonal is the **best** case, not the worst —
  two envs side by side are axis-aligned — and the mean hides the tail.
  Re-measured over the full toroidal Chebyshev ring (all directions), working
  encoder:

  ```
     d     mean      p99      max     axis(d,0)  diag(d,d)
    30   +0.396   +0.686   +0.824      +0.502     +0.241
    50   +0.071   +0.355   +0.512      +0.137     +0.006
    60   +0.012   +0.226   +0.440      +0.044     +0.000
    80   -0.009   +0.143   +0.273      -0.011     -0.001
   140   +0.005   +0.271   +0.481      +0.047     +0.027
   200   -0.000   +0.079   +0.122      -0.004     -0.001
  ```

  Three consequences. (1) Axis-aligned decorrelates ~3× slower than diagonal at
  every gap, so a margin derived from one direction is not a margin. (2) **The
  tail does not vanish with distance** — max is still +0.273 at d=80, and the
  bump at d=140 is §1.6's long-range alias structure reappearing. No Euclidean
  margin makes worst-case similarity small. (3) So derive on a **quantile, not
  the mean**: `mean < 0.05` → d≈60, `p99 < 0.15` → d≈80, `p99 < 0.10` → never.

  Still raise rather than clamp when the curve never crosses: at
  `fwhm_ratio=0.5` the mean plateaus at +0.12 and no margin separates it, which
  a hardcoded constant would hide.

  The surviving tail is what the per-split cosine diagnostic (§2.7) is for, and
  it is a stronger argument for that diagnostic than the original estimate
  suggested — see the open question in `docs/ENV_GENERATOR_STATUS.md`.
- **`val_frac` = 0.2** for the goal cell split.
- **Size OOD ships as a uniform val size differing from train**, not mixed sizes
  within a set. Only the scalar is threaded; the ~6 sites in §2.9 still need
  fixing (they read the *train* size while val envs have another), but no
  per-env size arrays.
---

## Part 3 — OOD probes that came out of the audit

Ranked by information-per-unit-work. Several are nearly free *today*.

### Tier 1 — cheap and high-yield

**1. Perimeter vs interior goal split.** `project_hopfield_nav_perimeter_basin`
records that the navD "breakout" is perimeter-orbit search and that an interior
goal at (2,4) fails 100%. Train goals ∈ `Ring(1)`, test ∈ `Interior(1)` — and
run the reverse direction too. If interior→perimeter transfers but
perimeter→interior does not, the perimeter basin is a training-distribution
artifact, not a representational limit. This is the single best-motivated split
in the project and needs only `goal_domain`.

**2. Size extrapolation, with a step-budget control.** Architecturally free
(§1.5). Train at 20, test at {8, 12, 16, 24, 32}. **Report at both fixed
`max_steps` and `max_steps ∝ size²`** — otherwise "worse at 32" conflates
capability with budget, and coverage is a per-cell fraction so the confound is
guaranteed.

**3. Embedding-geometry–stratified position OOD.** Better than splitting the
scaffold by Euclidean region: stratify patches by a *measured* property of
`encoded_Phi` there — local isometry error `‖Φ(x+e)−Φ(x)‖` anisotropy, or the
conditioning of the `gram_schmidt_2d_batch` frame. `encoded_Phi` is precomputed
and static, so this is one numpy pass. Train on well-conditioned patches, test
on ill-conditioned ones. This converts position-OOD from a nominal split into a
graded difficulty axis, and it separates *policy* failure from *readout
geometry* failure — the open question behind
`project_hopfield_nav_disambig`.

**4. Held-out encoder.** `--encoder_override` already exists. Evaluating with a
different same-architecture, different-seed encoder asks whether the policy
learned properties of *the* embedding or of *an* embedding. Costs one flag.

### Tier 2 — small extensions, sharp questions

**5. Distractor *similarity*, not just count.** `val_n_distractors_list` varies
count only, and `sample_distractors` draws uniformly outside the footprint.
Add a distractor policy: `uniform` (today) | `near_footprint(r)` |
`cosine_band(lo, hi)` (patterns whose cosine to the true goal encoding lands in
a band) | `other_goals`. The cosine-band version yields a difficulty *curve*
where there is now a binary, and it probes the goal-vs-distractor
disambiguation hypothesis directly.

**6. Wall-code Hamming interpolation.** Env identity is only `4·size` ±1 bits.
Mint eval envs at controlled Hamming distance *d* from a specific training env,
sweeping d from 0 (same env, new goal) to `2·size` (independent). A
memorization→generalization curve instead of a point estimate. The d=0 case is
a clean test of the exact shortcut `train_navigate.py:257-261` already worries
about ("in env X go to position Y" from the fingerprint alone).

**7. Wall-code distribution shift.** Training codes are iid Bernoulli(½). OOD:
biased *p*, low switch-rate codes (long same-sign runs → locally aliased
views), or periodic codes. A `wall_code_policy` field on `EnvSpec`; stresses
the sensory→position pathway specifically.

**8. Aliasing-stratified val.** The raycast can map distinct cells to identical
observations; distinct rows of `_codebook` gives a per-env aliasing rate. Use
it two ways: as a reported difficulty covariate, and as a *filter* — a
high-aliasing env may be unsolvable and is currently silently dragging the val
mean.

### Tier 3 — worth noting

**9. Factorial holdout.** Train on (small ∧ perimeter) ∪ (large ∧ interior);
test the two unseen combinations. Asks whether size and goal-position are
learned as independent factors.

**10. `Npos` shift and the wrap boundary.** `eval_all --Npos` exists already.
Raising `Npos` beyond training puts envs on grid phases the encoder saw but the
policy never visited. Sub-case worth its own row: place an eval env straddling
the `Npos` truncation corner — the encoded field is not periodic there, so the
local geometry genuinely differs. Free to construct.

**11. Non-square envs.** `size` is a scalar; `(w, h)` would be a small
generalization and a real shape-OOD axis. Later.

---

## Open questions for the experiment design

1. Is the target claim "generalizes to unseen scaffold positions" or "is
   invariant to scaffold position"? The second is stronger and suggests also
   *removing* `input_encoded_state` as an ablation rather than only holding out
   regions.
2. Held-out region size: at `Npos=132`, `size=20`, train footprints already
   cover 46% of the scaffold. A meaningful position holdout with a margin will
   force either fewer train envs or a larger `Npos` (i.e. more `lambdas`).
   Worth deciding before, not after.
3. Should `val_iid` keep today's semantics (new walls, arbitrary position) as
   the continuity baseline? Recommended yes — it is what every existing number
   means, and dropping it makes the new results incomparable to the run history.
