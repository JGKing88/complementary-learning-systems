# Env generator — status and per-phase plans

Handoff note for the env-generator / train-eval-splits work. The roadmap lives at
`~/.claude/plans/ok-write-the-full-memoized-barto.md`; the audit and design rationale
live in `docs/EVAL_SPLITS_DESIGN.md`. **Read this file first** — the roadmap is
deliberately coarse past Phase 1, and each phase gets a detailed plan here before it
starts.

Branch `env-generator`, from `a3e940d` on `main`.

Baseline established before any refactor edit (2026-08-07):

```
./run_tests.sh                                        438 passed
python -m hopfield_nav.tests.gen_golden --check       all goldens match
```

| Phase | What | Status |
|---|---|---|
| 0 | Branch; in-flight explore-min work committed and isolated | **done** |
| 1 | Plumbing refactor, behavior-frozen | **done** — `8db2930` |
| 2 | Generator, domains, separation | **done** — see outcome below |
| 3 | Serialization + train wiring | **done** — `901e802` |
| 4 | Per-trait refresh | **done** — see outcome below |
| 5 | Eval CLIs, mix-and-match | **done** — see outcome below |
| 6 | Size OOD | planned — see below |

---

## Phase 1 — detailed plan

**Goal.** Split `VectorHash` into an env-independent shared field plus a per-env-set
`EnvAssoc`, and make env offsets a first-class list threaded through every call site.
No generator, no new behavior.

**Acceptance gate.** `./run_tests.sh` at 438 passed, and `gen_golden --check`
byte-identical. Any golden movement is a bug in this phase, not an intended change.
This is the whole value of doing the phase separately: nothing should move, so
anything that moves is a signal.

**Why this is the right cut.** The `env_offsets: list[tuple[int,int]]` convention
already exists half-adopted — `evaluation/protocols.py:188`, `evaluation/rnn.py:144`
and `training/rnn_sequential.py:50` take it directly, and `metrics.py:1063` is
literally an adapter between the two conventions. This phase finishes the migration
and deletes `env_global_indices` / `env_indices`, which is a vestigial identity map
(`list(range(n))` at every one of its construction sites).

### Step order

Ordered so the tree compiles at each step and the suite can be run between them.

**1.1 — `world/scaffold.py`: split the class.**

`VectorHash` keeps only env-independent state and **drops the `size` constructor
argument** (size is an env property, and its only use was placement):

- keeps: `cfg, lambdas, Np, Ng, Npos, thresh, c`, `gbook`, `module_sizes`,
  `encoded_Phi`, and non-static `pbook`, `Wpg`, `Wgp`
- keeps: `build_scaffold`, `precompute_encoded_phi`, `get_encoded_state`,
  `get_store_patterns`, `gram_schmidt_projection`, `project_displacement`,
  `g_to_position`, `g_to_position_fast`
- loses: `size`, `env_offsets`, `register_envs`, `Wsp`, `Wps`, `Ns`, `recall`,
  `recall_batch`, `_test_scaffold`, `get_goal_encodings`

New in the same module:

```python
class EnvAssoc:            # non-static only; fitted to ONE env set
    Wsp, Wps, Ns
    recall(obs) / recall_batch(obs_batch)

def fit_env_assoc(field, envs, offsets) -> EnvAssoc | None    # None under static
def spread_offsets(n_envs, size, Npos, jitter, rng) -> list[tuple[int,int]]
def random_offsets(n_envs, size, Npos, rng)          -> list[tuple[int,int]]
def goal_encodings(field, envs, offsets)             -> list[np.ndarray]
```

`fit_env_assoc` carries the `fully_explore_random` → `sbook` → `pseudotrain_Wsp/Wps`
→ `_test_scaffold` body verbatim, so non-static keeps its own weights *and* its own
self-test per env set.

**Behavior-freeze constraint:** `spread_offsets` / `random_offsets` take an `rng`
parameter, and in this phase every caller passes **the `np.random` module itself**.
Draw order and values are then identical to today. Phase 2 passes a real
`RandomState`, which is where §1.4's reproducibility bug actually gets fixed. Same
signature either way — no throwaway adapter.

Safe to move `recall`: it and `recall_batch` are the sole consumers of
`pbook/Wsp/Wps/Wgp/Wpg`, and they are called from exactly one site in the repo —
`scaffold.py:503`, inside `_test_scaffold`. Every other `recall` hit is the
*Hopfield* class (`rollout/signal.py:112`, `nav_eval/nav.py:114`) or the separate
`encoder_scaffold.py` class under `analysis/scaffold_experiments/`.

**1.2 — `world/world.py`: the `World` container.**

```python
@dataclass
class World:
    envs:    list[GridEnv]
    offsets: list[tuple[int, int]]
    field:   VectorHash          # shared across worlds
    assoc:   EnvAssoc | None
```

Replaces the `{"envs", "vectorhash", "env_indices"}` dict returned by
`training/world_setup.py:49`. `env_indices` is dropped outright.

**1.3 — `setup_world` builds the field once and shares it.**

Numerically inert: `gen_gbook_2d` is deterministic, `load_encoder` calls
`encoder.eval()`, `smooth_gbook` is deterministic, and `precompute_encoded_phi`
consumes no RNG. Under `static_vectorhash` — 354 of 355 recorded runs — the
`build_scaffold` body draws nothing from global `np.random`, so the stream that
placement consumes is untouched. Expected effect: one 12 GB `encoded_Phi` instead of
`num_worlds + 1` copies.

**1.4 — `evaluation/metrics.py`: offsets in, indices out.**

All six evaluators take `env_global_indices: list[int]` as parameter 4. Change to
`env_offsets: list[tuple[int,int]]` and replace the six
`vectorhash.env_offsets[env_global_indices[i]]` lookups (lines 305, 443, 627, 830,
839, 975) with `env_offsets[i]`. Delete the adapter at line 1063.

Signatures: `evaluate_navigation` (254), `evaluate_goal_discovery` (371),
`evaluate_exploration` (551), `evaluate_realistic` (698), `evaluate_repeat` (883),
`evaluate_sequential_episodes` (1000).

**1.5 — update the 25 evaluator call sites and the remaining offset readers.**

Call sites pass `list(range(n))` today, so each becomes the offsets list the caller
already holds: `train.py:357,362,367,372,419`; `eval_all.py:137,147,156,165,175,200,215`;
`training/world_setup.py:158,162,165`; `tests/gen_golden.py:325,332,353`;
`tests/test_goal_contract.py:689,713,858`; `tests/test_evaluator_correctness.py:153,226,244,268`.

Other `vh.env_offsets[...]` readers: `checkpoint_io.py:140`,
`training/world_setup.py:76` (`make_hops`), `train_navigate.py:247`,
`train_phased.py:71`, `train.py:262`, `train_store.py:90`,
`analysis/trajectories.py:391`, `analysis/continual/agenthash.py:242,484,501`,
`analysis/phase_decoding/rollout.py:134`, `analysis/continual/baseline.py:286`,
`train_rnn.py:207`.

**1.6 — tests.** Eight files assign `vh.env_offsets` directly and now pass offsets
explicitly: `gen_golden.py`, `test_batched_eval.py`, `test_protocols.py`,
`test_goal_contract.py`, `test_smoke_train.py`, `test_evaluator_correctness.py`,
`test_phase_decoding.py`. Golden *values* must not move — the offsets they pin are
unchanged, only how they reach the evaluator is.

**1.7 — `assert Npos <= prod(lambdas)`** in `VectorHash.__init__`. Above that bound
two distinct scaffold positions get identical codes in every module and the scaffold
aliases outright. Unchecked anywhere today; `eval_all --Npos` can walk past it
silently. Pure addition — no valid config is affected.

### Phase 1 outcome (2026-08-07, `8db2930`)

Gate met: **438 passed, all goldens byte-identical, 32/32 entry points import.**
Field sharing verified by object identity — `num_worlds=2` plus the eval world now
holds **one** `encoded_Phi`, not three.

Two things worth carrying forward:

- **`spread_offsets` jitter can collapse on a small scaffold.** At `Npos=12,
  size=4, n=2` the per-axis jitter is ±0.8, which `int(round(...))` flattens, so two
  independently-placed worlds land on identical offsets. Pre-existing (the goldens
  confirm behavior did not change), harmless at the working `Npos=1716`, and exactly
  what an explicit generator removes. Worth an assertion in Phase 2.
- **Two pre-existing unused imports** surfaced while linting the diff and were left
  alone: `train_store.py` imports five config dataclasses it never uses, and
  `agenthash.py` imports `agent_step`. Not this phase's business.

### Risk register for this phase

| Risk | Mitigation |
|---|---|
| Placement RNG order shifts, moving every offset | Callers pass `np.random` module in this phase; the fix is deliberately deferred to Phase 2 |
| Sharing the field perturbs the global stream in non-static mode | Accepted: infeasible at working `Npos` (pbook alone is 37.7 GB) and unused in 354/355 runs |
| A golden moves | Stop. It means the refactor changed behavior; do not regenerate |
| `analysis/` entry points break silently (not covered by tests) | `python scripts/check_entry_points.py` after each step |

### Verification

```bash
./run_tests.sh
python -m hopfield_nav.tests.gen_golden --check
python scripts/check_entry_points.py
```

Plus a memory check: build with `num_worlds=2` and confirm one `encoded_Phi`
allocation, not three.

---

## Phase 2 — detailed plan

**Goal.** The env generator: declared per-trait domains, independent trait streams,
separation checking, and the two entry points training and post-hoc eval both call.
Library only — **nothing is wired into training in this phase.**

**Acceptance gate.** New `hopfield_nav/tests/test_splits.py` passes; the existing 438
tests still pass and `gen_golden --check` still matches. Phase 2 adds files and one
method (`GridEnv.set_goal`); it must not perturb any existing path.

### What Phase 1 left in place

```python
# world/scaffold.py
VectorHash(cfg)                       # field: gbook, encoded_Phi, (non-static pbook/Wpg/Wgp)
place_envs(n, size, Npos, rng, ...)   # -> list[tuple[int,int]]
fit_env_assoc(field, envs, offsets)   # -> EnvAssoc | None
goal_encodings(field, envs, offsets)

# world/world.py
World(envs, offsets, field, assoc)
build_world(field, envs, *, offsets=None, ...)   # offsets= is the generator hook
```

`build_world(..., offsets=...)` already accepts explicit offsets, so Phase 2 needs no
change to `world.py` — a generated `EnvSpec` list feeds straight in.

### Module layout

```
world/domains.py     domain families, stable_hash, trait_rng
world/spec.py        EnvSpec, TraitDomains, GeneratedSplit
world/generate.py    margin, capacity, separation, diagnostics, generate_split, make_val_set
world/env.py         + GridEnv.set_goal
tests/test_splits.py
```

### 2.1 `GridEnv.set_goal`

Mirror of the existing `set_position` (`env.py:218`). Construction stays
`GridEnv(seed=wall_seed)` so `_wall_code` is bit-identical for a given seed; the
constructor's own goal draw becomes dead entropy and is overwritten. Per-env size
comes from `dataclasses.replace(env_cfg, size=spec.size)` — `make_env` reads
`env_cfg.size`, so there is no need to widen its signature.

### 2.2 Trait streams

```python
def stable_hash(*parts) -> int          # blake2b over a canonical string
def trait_rng(run_seed, trait, tick) -> np.random.RandomState
```

**Not Python's `hash()`** — it is salted per process, so a run would not reproduce
across invocations. Derived per `(trait, tick)` rather than advanced from one stream:
that is what makes "refresh placement only, hold walls and goals" reproducible
without replaying the others.

### 2.3 Domains

Each with `sample(rng, n, ...)`, `contains(v)`, `complement(...)`, `to_json` /
`from_json`.

| trait | family |
|---|---|
| place | `Anywhere` \| `Rect(x0,y0,w,h)` \| `Complement(inner, margin)` |
| goal | `AnyCells` \| `Cells(frozenset)` \| `Ring(w)` \| `Interior(w)` \| `Quadrant(q)` |
| wall | `SeedRange(lo, hi)` |
| size | `Sizes([...])` |

`Complement(Anywhere)` must raise — there is nothing outside it. That is the "asking
for `--ood place` on a model that trained everywhere" error, caught at the domain
layer rather than at the CLI.

### 2.4 Separation

**place — toroidal, edge-to-edge, max over axes.** Per axis, circular origin
distance `dc = min(|Δ| mod P, P - |Δ| mod P)` with `P = prod(lambdas)`; gap on that
axis is `dc - size`; separation is `max(gap_x, gap_y)`. Max, not min: two AABBs are
disjoint as soon as they separate on *one* axis, and the ring measurement below is
Chebyshev, so the two agree by construction. Wrapping mod `P` is always correct — it
simply has no effect when `Npos` is well below the period.

Train↔val uses the full margin. **Train↔train uses the same margin** — corrected
during implementation. The plan said margin 0 on the reasoning that two train envs
near each other is coverage, not leakage. That reasoning is right and the conclusion
was still wrong: the *joint* packing needs a common basis, or train envs placed with
no clearance can occupy every slot a margin-separated val env could use. It costs
nothing (margin 80 admits 289 envs against the 84 used) and moves train placement
*towards* the historical spread-lattice, not away from it. See the Phase 3 outcome.

**wall** — disjoint `SeedRange`s as the mechanism; assert no exact codebook
collision; report min Hamming over **live bits only**. The South wall is structurally
dead: `FOVEAL_HALF_ANGLE_DEG = 60.0`, so every ray has `dy = cos θ ≥ 0.5 > 0` and
wall 2 can never be hit. That is 60 live bits of 80 at `size=20`. Derive `S` as dead
structurally, but *measure* N/E/W by flipping each bit and diffing the codebook, and
cache per `(size, observation_size)` — a bit-flip sweep is ~2 s at `size=20,
observation_size=12`.

**goal** — cell-set disjointness in env-local coordinates. **size** — set
disjointness.

### 2.5 Margin derivation — corrected

`derive_margin(field, rng, *, quantile, threshold)`: sample position pairs at
toroidal Chebyshev gap `d` **over all displacement directions**, return the smallest
`d` clearing the threshold. Measured (working encoder, full ring):

```
   d     mean      p99      max     axis(d,0)  diag(d,d)
  30   +0.396   +0.686   +0.824      +0.502     +0.241
  50   +0.071   +0.355   +0.512      +0.137     +0.006
  60   +0.012   +0.226   +0.440      +0.044     +0.000
  80   -0.009   +0.143   +0.273      -0.011     -0.001
 140   +0.005   +0.271   +0.481      +0.047     +0.027
 200   -0.000   +0.079   +0.122      -0.004     -0.001
```

The roadmap's "~50" came from the diagonal curve and is wrong: **diagonal is the
best case**, axis-aligned decorrelates ~3× slower, and two envs side by side are
axis-aligned. Derive on a **quantile, not the mean**: `mean<0.05` → d≈60,
`p99<0.15` → d≈80, `p99<0.10` → never reached.

**Decided: `p99 < 0.15`, giving margin ≈ 80 at the working config.** The tighter of
the two candidates, chosen deliberately because the cosine check only reports
(§2.7) — the margin is the whole guarantee, so it carries the tail as far as a
coordinate rule can.

Capacity is not the constraint at either candidate (`Npos=1716`, `size=20`, need 84):

```
  margin  40: 29 per axis ->  841 envs
  margin  60: 22 per axis ->  484 envs
  margin  80: 17 per axis ->  289 envs
```

Raise rather than clamp when the curve never crosses — at `fwhm_ratio=0.5` the mean
plateaus at +0.12 and no margin separates it.

### 2.6 Capacity and preflight

`place_capacity(domain, size, margin, Npos) = (floor((W - size)/(size+margin)) + 1)²`
for a `Rect`, checked before sampling. Bounded rejection attempts, then an error
naming the knob: raise `Npos`, enlarge the region, lower the margin, or ask for fewer
envs. Also assert the Phase-1 `Npos <= prod(lambdas)` guard is satisfied (it is
enforced in `VectorHash.__init__`, so this is a cheap re-check with a
generator-specific message).

### 2.7 Cosine diagnostic

`cosine_report(field, offsets_a, size_a, offsets_b, size_b) -> {max, p99, frac>0.3}`
over normalized `encoded_Phi` rows of the two footprint sets. At the working config
that is 80×400 = 32k vectors against 4×400 = 1600, `d=1024` — one matmul.

**Decided: it never gates.** Generation is never rerun on the strength of a cosine
value, so a split stays derivable from coordinates alone — no encoder in the loop,
and re-deriving a split under a different encoder cannot change which envs it
contains. Implemented as a pure function returning the numbers; the caller prints
them and stores them under `separation.diagnostics`.

The cost is accepted knowingly: at margin 80 a val env can still sit at cos ≈ 0.27
to a train env, and an unlucky draw near the d=140 bump can reach ≈ 0.48. That shows
up in the report rather than being silently fixed, which is the point — a number you
can see beats a reroll you cannot.

### 2.8 Goal cells — two branches, one object

```python
if refresh_goal:
    S_train, S_val = split(region_cells, val_frac=0.2, rng)   # 320 / 80
else:
    S_train = drawn_train_goals                               # <= n_envs cells
    S_val   = region_cells - S_train
goal_domain_val = Cells(S_val)          # both branches land here
```

Both serialize as a plain `Cells([...])`, so post-hoc generation is branch-blind.
Random scatter, not a region — keeps ring/interior free as a separately declarable
OOD region.

### 2.9 Entry points

```python
generate_split(field, env_cfg, domains, n_train, n_val, seed, *, refresh_goal)
    -> GeneratedSplit(domains, train, base_val, goal_cells_train, goal_cells_val,
                      margin, diagnostics)

make_val_set(split, n_envs, levels, seed, ...) -> list[EnvSpec]
    levels: {"place"|"wall"|"goal"|"size": "same" | "held_out" | "ood"}
```

`generate_split` draws train and base_val **together** so separation is enforced
jointly rather than checked afterwards — sampling val without knowing train is what
produces today's overlap. `generate_split` is the special case
`levels = all held_out`.

`GeneratedSplit` is the Phase-2 stand-in for the Phase-3 `world.json`: it carries
domains + resolved lists + the per-trait union, so `make_val_set` is testable now and
Phase 3 only has to serialize it.

### 2.10 Tests (`tests/test_splits.py`)

- `stable_hash` identical across separate processes (subprocess check — this is the
  one that catches accidentally using `hash()`)
- domain `sample` determinism given `(domain, seed)`; `complement(complement(d)) == d`;
  `Complement(Anywhere)` raises
- **structural**: the South wall contributes zero live bits at several sizes, and
  N/E/W contribute all of theirs — the §1.7 claim, asserted rather than assumed
- separation holds per trait between `train` and `base_val` for a range of seeds
- toroidal gap: two envs straddling the `Npos` seam are correctly reported as near
- capacity error fires with the right message when a `Rect` is too small
- `EnvSpec` round-trip: build → serialize → rebuild gives identical `_wall_code`
  **arrays**, offsets and goals (not just equal seeds)
- goal branch: with `refresh_goal=True` train and val cell sets partition the grid
  and are disjoint; with `False`, val is the complement of what train drew

### Decisions for this phase (settled 2026-08-07)

- **Cosine never gates** (§2.7). Report and store; do not reroll, do not error. A
  split stays derivable from coordinates alone.
- **Margin is `p99 < 0.15`** (§2.5), ≈ 80 at the working config — the tighter
  candidate, since with no cosine gate the margin is the whole guarantee.

Together these say: the placement rule is purely geometric and fully specified by
the domain plus the margin, and the cosine numbers are evidence *about* a split
rather than an input *to* it. The residual — max cos ≈ 0.27 at margin 80, ≈ 0.48 on
an unlucky draw near the d=140 bump — is accepted and visible in the report.

### Risk register

| Risk | Mitigation |
|---|---|
| `hash()` used instead of `stable_hash` | Cross-process test in the suite |
| Margin derived from one direction again | `derive_margin` samples the full ring; test asserts axis-aligned is the slow direction |
| Rejection sampling loops forever on a tight region | Bounded attempts + capacity preflight with a specific error |
| Generated envs silently reuse a train wall seed | Disjoint `SeedRange` by construction, plus an explicit assertion on codebooks |

### Phase 2 outcome (2026-08-07)

Gate met: **481 passed** (438 pre-existing + 43 new in `tests/test_splits.py`), all
goldens byte-identical. Phase 2 added files and one method; nothing existing moved.

Shipped: `world/domains.py`, `world/spec.py`, `world/generate.py`,
`GridEnv.set_goal`, `tests/test_splits.py`.

**`derive_margin` validated against the real encoder.** Run through a lazy
`encoded_Phi` proxy (per-module phase tables + encoder, identical arithmetic to
`precompute_encoded_phi`, no 12 GB allocation), it reproduces the hand measurement
exactly:

```
derive_margin(quantile=0.99, threshold=0.15) = 80      <- the decided rule
derive_margin(quantile=0.50, threshold=0.05) = 60      <- the looser candidate
```

**The dead South wall is now asserted, not assumed.** `live_wall_bits` measures
`3*size` live bits at sizes 4, 6 and 8 — wall 2 contributes zero, walls N/E/W
contribute all of theirs. Wall Hamming margins are reported over live bits only, so
a quarter of the flips are no longer silent no-ops.

**One design asymmetry worth naming.** Only two traits have a bounded universe and
therefore a meaningful complement:

| trait | universe | `ood` level |
|---|---|---|
| place | `[0, Npos)^2` | yes — `Rect.complement(margin)` |
| goal | `[0, size)^2` | yes — `complement_for(domain, size)` |
| wall | unbounded | **no** — novelty *is* `held_out` (a seed training never drew) |
| size | unbounded above | **no** — named outright via `make_val_set(..., size=N)` |

That is encoded as errors rather than left implicit: `Anywhere.complement()`,
`AnyCells.complement()`, `SeedRange.complement()` and `Sizes.complement()` each raise
with the knob to use instead. So "`--ood place` on a model trained everywhere" fails
at the domain layer with an explanation, not as a silently empty sample.

**Testing note.** The `derive_margin` fixture needs a field that genuinely
decorrelates. A sum of fixed cosines does not — it stays quasi-periodic and the p99
never falls, which initially read as a bug in `derive_margin` and was not. Random
Fourier features give `cos ≈ exp(-d²/2ℓ²)`, and the test now asserts that a longer
correlation length yields a larger margin rather than pinning one number.

---

## Phase 3 — detailed plan

**Goal.** Make a run's world *recorded* rather than *implied*: emit `world.json`,
carry a pointer in the checkpoint, and let `train_navigate` optionally source its
envs from the Phase-2 generator.

**Acceptance gate.** 481 tests still pass, goldens byte-identical, plus new
serialization tests. Goldens are unaffected by construction here — `gen_golden`
builds its own envs and sets offsets explicitly, so it never touches
`train_navigate`'s world setup.

**This is the first phase where numbers legitimately move** — but only under the new
flag. With `--env_generator` off, `setup_world` still calls
`place_envs(..., np.random)` exactly as it does now, so a given `--seed` produces the
same envs as Phase 1 and 2. With it on, placement comes from the generator's derived
streams and the §1.4 reproducibility bug is fixed.

### The design decision that shapes this phase

**`world.json` is written on *both* paths, always.** The legacy path is perfectly
expressible as a spec: its domains are just the permissive defaults
(`Anywhere` / `AnyCells` / the full seed range) and its `resolved` lists are the
offsets, goals and seeds it actually drew.

That matters more than it sounds. §1.4's bug is that a checkpoint's val offsets are
*unrecoverable* — `build_eval_world` replays the seed stream but placement came from
global `np.random`, so every post-hoc eval scores a checkpoint on scaffold patches
training never used. Recording the resolved specs fixes that for **every new run
immediately**, whether or not anyone opts into declared domains. The flag then only
controls whether the generator *chooses* the envs, not whether they are *recorded*.

### Config surface — flat fields, string grammar

Domains reach the config as compact strings, parsed at startup, mirroring how
`schedule` already works (`stages.py` parses `"explore:100,novelty=0.1"`):

```
--env_generator / --no-env_generator     default OFF this phase
--place_region   anywhere | rect:X0,Y0,W,H          default anywhere
--goal_region    any | ring:W | interior:W | quadrant:Q   default any
--wall_seeds     LO,HI                               default 0,100000000
--place_margin   auto | N                            default auto (derive_margin)
--goal_val_frac  F                                   default 0.2
```

Flat `TrainConfig` fields, not a nested dataclass. Two reasons: `asdict(cfg)` stays
JSON-native (domain *objects* would not serialize), and it avoids touching
`cfg_from_checkpoint`'s hand-written nested reconstruction, which is the riskiest
function in the checkpoint path. Each flag gets a `CFG_FIELDS` entry
(`train_navigate.py:443`) so `--load_checkpoint` inheritance works unchanged.

### Step order

**3.1 — `GridEnv.seed`.** Record the constructor's seed as an attribute. Needed to
write a legacy-path env's `wall_seed` into `world.json`; today the seed is consumed
and discarded. Consumes no RNG, so behavior-neutral.

**3.2 — `world/spec.py`: `WorldSpec` + `spec_hash`.** Wraps a `GeneratedSplit` with
scaffold identity and provenance:

```json
{"spec_version": 1,
 "scaffold": {"lambdas": [...], "Npos": N, "fwhm_ratio": F,
              "static_vectorhash": true,
              "encoder": { ...run_manifest.encoder_identity()... }},
 "generator": "declared" | "legacy",
 "split":     { ...GeneratedSplit.to_json()... },
 "spec_hash": "sha256 of the canonical split+scaffold JSON"}
```

`encoder_identity` (`run_manifest.py:127`) already yields path + sha256 + out_dim +
lambdas + gain, so this reuses it rather than hashing anything itself.

**3.3 — `world_setup.write_world_spec(save_dir, ...)`.** One helper, called by
`train_navigate`. Kept in `training/` rather than inside the trainer so
`train_phased` / `train_store` / `train` can adopt it later without a second copy.

**3.4 — `train_navigate` wiring.** After `build_field`, branch:

- `--env_generator`: parse domains, `generate_split(field, ...)` once for train +
  base_val together, `build_envs`, then `build_world(field, envs, offsets=...)`.
- else: today's `setup_world` calls, then derive `EnvSpec`s from the resulting
  `World`s (now possible thanks to 3.1).

Either way, write `world.json` next to `run.json` after `save_dir` resolves. In this
phase the union never changes, so one write suffices; Phase 4 adds the rewrite at
`ckpt_every` when refresh starts moving it.

**3.5 — checkpoint payload.** Both `torch.save` sites
(`train_navigate.py:336` periodic, `:429` final) gain a `world_spec` key holding
domains + `spec_hash` + the `world.json` path — **not** the resolved union, which
grows once refresh exists and would bloat every checkpoint. Safe against
`cfg_from_checkpoint`, which only walks the `config` sub-dict.

**3.6 — `checkpoint_io` loader + legacy fallback.**
`load_eval_world(ckpt_or_dir, field, env_cfg, movement_mode)` prefers `world.json`
and returns `(envs, offsets)` exactly as recorded. Missing → today's replay path with
a printed warning naming what is approximate (the offsets, per §1.4). The 355
existing run dirs keep working, and no new guarantee is claimed for them.

### Tests

Extend `tests/test_splits.py` (or a new `test_world_spec.py`):

- `WorldSpec` JSON round-trip; `spec_hash` stable under key reordering and
  changing under a real edit
- write → read → rebuild yields identical `_wall_code` arrays, offsets and goals
- the **legacy path** produces a valid, loadable `world.json` (this is the one that
  proves §1.4 is fixed for ordinary runs)
- missing `world.json` falls back and warns rather than raising
- a short end-to-end `train_navigate` run with `--env_generator` writes a
  `world.json` whose `train` and `base_val` satisfy `verify_split`

### Risk register

| Risk | Mitigation |
|---|---|
| Numbers move silently for existing launchers | `--env_generator` defaults OFF; the legacy path keeps drawing from `np.random` unchanged |
| `asdict(cfg)` breaks on a non-serializable field | Domains are strings in the config; objects exist only after parsing |
| `world.json` and the `.pt` disagree | `spec_hash` in both; loader compares and warns on mismatch |
| Two sources of truth for offsets once specs exist | `World.offsets` stays the single runtime source; the spec is a record of it, asserted equal at write time |

### Noticed, not fixed

`cfg_from_checkpoint` reconstructs `env`/`vectorhash`/`hopfield`/`agent`/`ppo` but
**not `bc`**, so `cfg.bc` comes back as a raw dict rather than a `BCConfig`
(`checkpoint_io.py:58-68`). Dormant — nothing in the navigate path reads
`cfg.bc.<field>` after a load, and it would crash loudly if it did. Left alone
because it is unrelated to this work; worth its own one-line commit.

### Phase 3 outcome (2026-08-07)

Gate met: **493 passed** (481 + 12 new in `tests/test_world_spec.py`), goldens
byte-identical.

Shipped: `WorldSpec` + `spec_hash` in `world/spec.py`; `setup_worlds_declared`,
`specs_from_world`, `legacy_split`, `write_world_spec` in `training/world_setup.py`;
`GridEnv.seed`; six CLI flags with `CFG_FIELDS` entries; `world_spec_for` /
`eval_world_from_spec` and a warned fallback in `evaluation/checkpoint_io.py`.

**`world.json` is written on both paths**, so §1.4 is fixed for every new run
whether or not it opts into declared domains. `--env_generator` only controls
whether the generator *chooses* the envs; the recording happens either way.

**The legacy recording immediately earned its keep.** On the first end-to-end run
(toy config, `Npos=12`, 2 train + 2 val) it reported:

```
world.json: generator=legacy margin=0 min_place_gap=-4 min_wall_hamming=2 max_cos=1.0
  train: [(4, 1), (4, 8)]      val: [(4, 1), (4, 7)]
```

A val env at **the same offset** as a train env — `min_place_gap=-4`, `max_cos=1.0`.
That is §1.3's overlap, surfaced automatically by an ordinary run instead of having
to be found by an audit. Every new run now says out loud how close its val envs came
to its train envs.

**A bug the end-to-end run caught, in this phase's own code.** The Phase-2 capacity
preflight checked train and val separately — `capacity(margin) >= n_val` and
`capacity(0) >= n_train + n_val`. Both passed on a configuration that then failed to
place a *single* val env. The check was unsound: train envs placed with no mutual
clearance can sit on every margin-separated slot validation needs, so their capacity
at margin 0 says nothing about what is left. Fixed by making the bound joint
(`capacity(margin) >= n_train + n_val`) **and** placing train envs on the same margin
basis. The preflight now names all four knobs:

```
ValueError: place domain Anywhere() holds ~4 envs of size 4 at margin 25, need
4 train + 4 val = 8. Raise Npos, enlarge the region, lower the margin, or ask
for fewer envs.
```

Worth noting the shape of the miss: the unit tests passed because they exercised
`generate_split` at configurations with room to spare. It took a deliberately tight
end-to-end run to find it. Phase 4's tests should include a tight-packing case.

**Still deferred:** `train_phased`, `train_store` and `train` do not yet write a
spec — the helpers live in `training/world_setup.py` precisely so they can adopt it
without a second copy. And `cfg_from_checkpoint` still returns `cfg.bc` as a raw
dict (`checkpoint_io.py:61-73`), noted in the Phase 3 plan and still unfixed.

### Ahead of Phase 4: env memory is derived, not pooled (2026-08-07)

Phase 4.2 in the roadmap called for an `ExploitRegime.rebuild_pools(worlds)` hook,
called after any refresh tick touching place or goal. **That item is now obsolete** —
the underlying problem is solved structurally instead.

`ExploitRegime` cached one Hopfield per env at construction, holding
`encoded_Phi[goal + offset]`. Both inputs are things a refresh moves, and a cached
memory survives the move pointing at the old cell. The failure is worse than noise:
the reward still fires at the *real* goal, so PPO receives a consistent signal that
following the recall channel does not pay — training the agent to ignore the very
thing the exploit regime exists to teach, with no error raised.

Nor could it be dodged by refreshing only explore envs: regime assignment is by
index against `n_pre_now`, which moves as `empty_frac` anneals, so an env that was
refreshed while exploring becomes exploit later and is looked up in a pool built from
its original state.

`regime.spec()` now always constructs. Verified safe before changing:

- **Nothing was ever written to the pooled objects.** `collector.py:79` sets
  `shared_hopfield = not isinstance(hopfields, list)`, and the only `input_memory`
  call in a rollout sits inside `if not shared_hopfield`. So rebuilding is
  content-identical, not merely equivalent — no persisted state existed to lose.
- **No RNG moves.** `Hopfield.__init__` and `input_memory` draw nothing, and the
  distractor branch keeps its draws gated on `use_distractors` exactly as before, so
  an unconstrained run consumes no distractor randomness. Pinned by a test.
- **Nothing external held the pools** — only the two regime classes.
- Cost: ~309 ms per update at 80 envs and `embed_dim=1024`, ~93 s over a 300-update
  run, about 1.3% of a 2-hour budget.

`train_phased` and `train_store` keep their own `make_hops` pools. Neither refreshes,
so neither can go stale; left alone rather than widened into this change.

The general lesson, worth keeping: the pool's docstring justified sharing as *"only
sound because nothing writes to it."* The real condition is "nothing writes **and**
nothing it depends on moves." Refresh breaks the second half, which nobody had
written down.

---

## Phase 4 — detailed plan

**Goal.** Per-trait refresh: re-draw an env's placement, wall pattern, goal or size
on a cadence, from the **train** domain, with every value it ever takes recorded.

**Acceptance gate.** 511 tests pass, goldens byte-identical. Refresh is off by
default, so a run that does not ask for it is unchanged.

### What the roadmap got wrong, and what replaced it

The roadmap's **4.2 (exploit pool invalidation)** is **obsolete**. It called for a
`rebuild_pools` hook after any tick touching place or goal, because
`ExploitRegime` cached one Hopfield per env holding `encoded_Phi[goal + offset]`.
Two changes since removed the problem rather than managing it:

- `69d0e63` — memory is **derived per rollout**, never cached. A refresh cannot
  produce a stale pattern because there is no pattern to go stale.
- `36b59db` — `allow_store` is **declared**, so refresh no longer interacts with an
  undeclared write path.

So Phase 4 does not need a hook, a generation counter, or an invalidation rule.

**The load-bearing item is now a different one.**

### 4.1 The invariant that must not be got wrong

**Every refresh tick must fold its new values into `split.used`.**

`make_val_set` excludes against `split.used[trait]` — the union of everything
training touched. If a tick re-draws a placement and does not call
`split.record_used`, that offset silently drops out of the exclusion set, and a
`held_out` validation env can later be placed exactly where training was. Nothing
raises; the split just quietly stops meaning anything.

This is the same shape as the bug 4.2 used to be, one level up: a cached fact that
a refresh invalidates. The mitigation is the same in spirit — make it structural.
The refresh step should return the new specs and the caller should have no way to
apply them without recording them, rather than `record_used` being a second call
that can be forgotten.

### 4.2 Refresh requires `--env_generator`

Without declared domains there is nothing to re-draw *from*: the legacy path has no
place domain, no goal-cell partition, and no seed range. Setting any refresh cadence
without `--env_generator` must be a startup error naming that, not a silent no-op.

### 4.3 Config surface

Four flat `int | None` fields on `TrainConfig`, matching the Phase-3 style
(JSON-native, no nested dataclass, one `CFG_FIELDS` entry each):

```
--refresh_place N     re-draw train placements every N updates   (None = never)
--refresh_wall  N     re-draw train wall seeds every N updates
--refresh_goal  N     re-draw train goals every N updates
--refresh_size  N     re-draw train sizes every N updates
```

**Validation is not fixed** stays true: only train envs refresh. `base_val` is drawn
once and held. A validation set that moved under the model would make every
in-training curve uninterpretable — you could not tell improvement from an easier
draw.

### 4.4 The refresh step

At the top of the update loop (`train_navigate.py:196`), before the rollout loop.
For each trait due on this tick, draw from `split.domains` using
`trait_rng(cfg.seed, trait, tick=update)` — the derived stream, so changing one
trait's cadence cannot move another trait's values.

| trait | what it re-draws | cost |
|---|---|---|
| place | all train offsets jointly, excluding `base_val` at margin | **12 ms** / tick (84 envs, margin 80, Npos 1716) |
| goal | a cell from `split.goal_cells_train`, via `env.set_goal` | negligible |
| wall | a seed from the domain minus `used["wall"]`, then rebuild the env | **4.1 s** at `obs=12`, **10.2 s** at `obs=60`, 80 envs |
| size | a size from the domain, then rebuild the env | as wall |

Placement re-draws train against a fixed `base_val`, so the margin still holds in
the direction that matters. `env.set_goal` already exists (Phase 2); `env.reset_goal`
must **not** be used, because it draws uniformly over the whole grid and would walk
straight out of the declared goal domain.

### 4.5 `refresh_goal` must reach `generate_split`

`setup_worlds_declared` never passes `refresh_goal`, so `generate_split`'s
goal-partition branch has never run. With goal refresh on and the partition off,
training consumes 80 fresh cells per update and exhausts all 400 in ~5 updates —
after which no legal held-out goal exists and generation fails (or worse, has
already silently overlapped).

So: `setup_worlds_declared` passes `refresh_goal = cfg.refresh_goal is not None`,
which switches the 80/20 up-front cap on. This is a bug-in-waiting that only becomes
reachable in this phase.

### 4.6 Vectorize `_build_sensory_codebook`

The triple Python loop (`world/env.py`) is the whole cost of wall and size refresh.
It vectorizes cleanly over cells × rays.

**Gate on bit-identical output** against the loop version. Any drift changes every
env's codebook and would invalidate the goldens, `wall_code_for`, and
`live_wall_bits` simultaneously — and would do it silently, since a different-but-
plausible codebook still trains.

### 4.7 Delete `--randomize_goal_per_rollout`

Per the decision to replace it outright. Removes `TrainConfig.
randomize_goal_per_rollout`, `ExploreRegime.randomize_goal`, `RolloutSpec.
reset_goal`, and the `env.reset_goal()` call at `train_navigate.py:286`.

Cadence is equivalent (`train_navigate` runs one rollout per env per update, so
per-rollout and per-update coincide). **Scope is not**: the old flag re-drew goals
for explore-regime envs only; `--refresh_goal 1` is world-wide. Accepted — it
changes the explore regime's behaviour, and the v35 lineage is already
unreproducible from the `freeze_log_std` fix.

### 4.8 `world.json` rewritten on the checkpoint cadence

Phase 3 writes it once at startup, which was correct while nothing moved. Now the
`used` union grows, so rewrite at each `ckpt_every` alongside the checkpoint. Size
stays trivial: 300 ticks × 84 envs × 2 ints ≈ 200 KB.

### Tests

- **refresh off → byte-identical**: goldens, plus a short run against a no-refresh
  baseline
- **traits are independent**: refreshing `place` must leave goals and wall seeds
  untouched across ticks — the derived-stream property, and the thing that makes
  per-trait refresh meaningful at all
- **the union accumulates**: after N ticks `len(used["place"]) > n_train`
- **the invariant holds end to end**: mint a `held_out` val set *after* refresh and
  assert it excludes everything in the union — this is the test that would have
  caught a missing `record_used`
- refreshed placements stay ≥ margin from `base_val`
- goal refresh draws only from `goal_cells_train`
- refresh without `--env_generator` errors
- vectorized codebook bit-identical to the loop, at several sizes and ray counts
- **a tight-packing case**, per the Phase 3 lesson: unit tests that all have room to
  spare miss placement bugs

### Risk register

| Risk | Mitigation |
|---|---|
| A tick forgets `record_used`; val exclusion silently narrows | Make applying a refresh and recording it the same operation, not two calls |
| Wall refresh dominates wall clock | Vectorize first; measure before promising every-update cadence |
| Goal refresh exhausts the cell grid | 4.5 — `refresh_goal` switches on the up-front partition |
| Refreshed placement drifts onto the val set | Re-draw excludes `base_val` at margin; assert per tick |
| Vectorized codebook differs subtly | Bit-identity test; goldens as backstop |

---

## Phase 4 — outcome

`hopfield_nav/training/refresh.py`, wired into `train_navigate`. 678 tests pass
(was 651); goldens byte-identical. Refresh is off by default and a run that does
not ask for it takes the same code path it always did.

```
--refresh_place N   re-draw train placements every N updates   (None = never)
--refresh_wall  N   re-draw train wall seeds every N updates
--refresh_goal  N   re-draw train goals every N updates
--refresh_size  N   re-draw the train env size every N updates
```

All four require `--env_generator` and are checked before the encoder loads.
Only the train set moves; `base_val` is drawn once and held.

### The invariant, made structural

`_draw` writes `split.train` **and** calls `record_used`; `_apply` takes no
arguments and reads `split.train`. So the only thing appliable is what was
recorded, and "apply a refresh nobody recorded" has no spelling. `maybe_refresh`
is the only public entry.

The test that guards it (`test_the_union_is_what_a_later_val_set_avoids`) had to
be rewritten once: the first version compared a `held_out` val set against
`split.used`, which is circular — dropping `record_used` makes the union
*smaller*, so the val set trivially clears it while sitting exactly where
training just was. It now accumulates what the **live worlds** held on each tick
and compares against that. Six mutations were run against the final suite (drop
`record_used`; place ignores `base_val`; size does not drag place+goal; goal
draws from the whole domain instead of the partition; apply skips offsets; wall
reuses old seeds) and each is caught.

The wall-exclusion mutation initially was **not** caught: at the default 10M-wide
seed range two ticks never collide by chance. It needed a test on a 30-wide
range, where a reissued seed actually stops the union growing.

### Two things only reachable in this phase

**`refresh_goal` now reaches `generate_split`.** `setup_worlds_declared` never
passed it, so the 80/20 goal partition had never executed. With goal refresh on
and the partition off, `n_train` envs consume `n_train` fresh cells per update
and a size-8 arena's 64 are gone in eight updates.

**`--randomize_goal_per_rollout` is deleted**, replaced by `--refresh_goal N`.
Cadence is equivalent (one rollout per env per update); scope is not — the old
flag was explore-regime only and drew uniformly over the arena, so it could land
on a cell reserved for validation. Two recorded runs used it, so
`coerce_legacy_cfg` warns on load rather than dropping it silently (an unknown
top-level key is ignored by `cfg_from_checkpoint`, so it would otherwise vanish).

### A placement bug the refresh exposed

`sample_places` failed on the first real refresh with *"no lattice of that pitch
fits either"* when one plainly did. Two causes in `_lattice_places`, both only
reachable with a **non-empty `exclude`** — which is to say only from a refresh,
since `generate_split` draws train and val in one call with nothing pre-placed:

1. **Jitter slack came from the pitch.** `_axis_slots` drops trailing slots until
   the wrap clears, so the survivors are not evenly spaced: at `Npos=15,
   pitch=10` the slots `[0, 10]` are 7 apart going up and **2** apart going
   round. Pitch-based slack licensed a jitter of 2, which closed the wrap gap to
   1 — and the jittered env then blocked a *later* lattice slot. Four slots, four
   envs, one lost. Fixed by `_axis_slack`, which reads the actual slot gaps
   including the wrap. Same seam that bit `_axis_slots` in Phase 3.
2. **No fallback to the bare slot.** Eight jitter tries, and if all eight hit
   `exclude` the slot was abandoned even though it was legal. Now the ninth
   attempt is the unjittered slot.

Verified: 559 lattice configurations across `Npos ∈ {60,132,264,1716}`,
`size ∈ {3,6,8,20}`, `margin ∈ {0,2,6,12,20,80}` — **0 separation violations**, 5
configurations that used to fail now succeed, 0 regressions. The change alters
which offsets the *lattice* returns, but that path is reached in 2 of 15 joint
draws and never at the working config (`Npos=1716, margin=80, 94 envs`), where
rejection sampling always succeeds. `sample_places` is on-branch code that has
never produced a recorded run, and the legacy `place_envs` path does not use it.

### Costs, re-measured — the plan's figures were wrong

| trait | plan said | measured (80 envs, `Npos=1716`) |
|---|---|---|
| place | 12 ms | **3.9 ms** |
| goal | negligible | **0.3 ms** |
| wall | 4.1 s @ obs=12, 10.2 s @ obs=60 | **56 ms** @ obs=12, **129 ms** @ obs=60 |

**Plan item 4.6 (vectorize `_build_sensory_codebook`) was already done** — `47cc228`
rewrote it as a single `raycast_codes` call over cells × headings. It is not a
triple Python loop and has not been one since before the Phase 4 plan was
written. Wall refresh is 30–80× cheaper than the plan assumed, so every cadence
including `1` is affordable and no bit-identity gate was needed.

`split_diagnostics` — the whole cost of a `world.json` rewrite — is **0.31 s**, so
rewriting on `ckpt_every` is free.

### Size refresh

Implemented, including the coupling: a new size drags **place and goal** with it,
because offsets were spaced for the old footprint and a goal cell can fall
outside the new arena outright. It refuses at startup when fewer than two sizes
are declared, which is every configuration today — nothing produces a multi-value
`Sizes` domain yet (that is Phase 6). Tested by building the multi-size split by
hand, so the code is exercised rather than speculative.

### Other changes

- `world.json` is rewritten on the checkpoint cadence when a run refreshes, and
  once more at the end. The `used` union grows; `base_val` never moves, so every
  *evaluation* use of the file is unaffected. An earlier checkpoint's recorded
  `spec_hash` describes a prefix no longer on disk — the latest always matches.
- `split.diagnostics["refresh"]` records the cadence, tick and per-trait counts,
  so the file can say whether the world stood still.
- `make_val_set` now refuses `size=None` when training used more than one size,
  instead of picking one by set-iteration order.
- Both regimes read `env.size` rather than `cfg.env.size` when sampling
  distractors — the latter goes stale under size refresh.

### Left undone

- `train_phased` / `train_store` / `train.py` still do not write `world.json`,
  and do not refresh. Unchanged from Phase 3.
- `make_val_set` still does not run `verify_split` on its own output.

### Looking at a generated world

`analysis/world_layout.py` renders a split on the full scaffold — train and
validation footprints, and the square each env forbids to every other offset.
Placement is a pure function of coordinates, so it skips `encoded_Phi` entirely
and runs in about a second.

```bash
python -m analysis.world_layout --seeds 0 1 2
python -m analysis.world_layout --configs working rect --refresh 20
```

Two things the drawing has to get right, and both are easy to get wrong. The
forbidden zone is a **square around the offset** of side `2*(size + margin)`, not
the footprint dilated by the margin — separation is `max` over axes, so B is
illegal exactly when it is close on *both*. And the halos **wrap**: drawn flat
they would show clearance that is not there, which is the same mistake that makes
`(1715, 987)` and `(4, 989)` look 1711 cells apart when they are 5.

What it shows at `lambdas=[11,12,13]`:

| config | placed by | min gap achieved |
|---|---|---|
| anywhere, size 8, margin 80, 90 envs | rejection | 80 — the margin binds exactly |
| rect:200,200,700,700, 48 envs | **lattice** | 81–83 |
| anywhere, margin 40, 220 envs | rejection | 40 |
| anywhere, size 20, margin 80, 70 envs | rejection | 80–81 |

The `rect` row is visibly a regular grid: 48 envs into ~64 slots is 75%
occupancy, which is where rejection sampling gives up and `_lattice_places`
takes over. That is the regime the seam bug above lived in, and it is not
otherwise obvious from any number the generator reports.

### A consequence of place refresh, measured

> **Corrected 2026-08-12 (Phase 5).** This section first reported that
> `place='held_out'` becomes *infeasible* after ~20–30 refresh ticks, on the
> evidence that `make_val_set` raised. That was wrong: `make_val_set` was raising
> because rejection sampling could not *find* the surviving offsets, not because
> none existed. I had verified the legal-offset **count** by exhaustive scan and
> then inferred infeasibility from the sampler's failure — the count was right,
> the inference was not. With Phase 5's `legal_mask` the draw succeeds at every
> tick tried, out to 2000. The corrected numbers are below.

`--refresh_place 1` at the working config draws ~80 fresh offsets per tick and
almost none repeat: 20 ticks give **1679 distinct** offsets. Each forbids a
176×176 square of candidate offsets on a 1716² torus, so the legal set collapses
fast — and then **stops**:

```
tick      |used[place]|     legal offsets     max packing     make_val_set(10)
   0                80           944,117             187      ok
   8               720            10,808              11      ok
  24             1,999             2,463              10      ok
 100             8,070               101              10      ok
 300            23,961                12              10      ok
1000            78,863                10              10      ok
2000           155,254                10              10      ok
```

Seeds 0–2 at 300 ticks: 12 / 24 / 17 legal, max packing **10** in all three.

**It plateaus rather than exhausting**, and the reason is worth keeping. Once only
a dozen offsets remain legal, a refresh tick — which draws from the ~2.9M
positions merely clear of `base_val`, not clear of `used` — hits one of those
dozen with probability ~10⁻⁵. So the survivors are never consumed. The legal set
converges to a small fixed point instead of emptying.

**In-training validation was never affected** either way: `base_val` is drawn at
generation and held, which is the whole reason it is held.

What is true, and is what the Phase 5 preflight should report, is that the
**headroom collapses**: after a few hundred ticks the run supports a held-out
place set of about 10 envs and no more. Asking for 11 would fail. That is a real
constraint on `--num_val_envs` in a post-hoc eval, and it is invisible without
being measured.

`place='same'` is unaffected. `place='ood'` is unaffected and stays roomy when a
`Rect` was declared, since the complement is never touched.

---

## Phase 5 — detailed plan

**Goal.** Make the split reachable from evaluation: point a CLI at a run's
`world.json` and ask for any per-trait level combination, in one invocation.

**Acceptance gate.** 683 tests pass, goldens byte-identical. Every existing
`eval_all` command line keeps working and its output JSON keeps its current shape.

### 5.0 The §1.4 bug is still live in every eval CLI — fix this first

Phase 3 built `eval_world_from_spec` and taught `build_eval_world` to take a
`spec`. **Nothing passes one.** All four call sites still call it with three
arguments:

```
hopfield_nav/eval_all.py:112
analysis/trajectories.py:723
analysis/phase_decoding/rollout.py:106
analysis/continual/agenthash.py:418
```

So every post-hoc evaluation still takes the RNG-replay branch, prints the "env
offsets are not exact" warning, and scores checkpoints on scaffold patches
training never used — measured deltas up to 10 cells, the whole reason
`world.json` exists. Runs written since `901e802` have a truthful record sitting
unread beside them.

This is the highest-value item in the phase and it is nearly free. The fix has to
be structural rather than four remembered edits, because a fifth call site would
silently rejoin the legacy path: **`build_eval_world` gains a `ckpt_path`
parameter and discovers the spec itself** via `world_spec_for`, which already
accepts a checkpoint or a run directory. Callers pass the path they already hold.
The legacy warning then fires only when a run genuinely has no record.

Worth doing as its own commit, before any new surface: it changes what existing
numbers mean, and that should not be tangled up with a feature.

### 5.1 Level flags on `eval_all`

```
--split recorded                                  # base_val exactly as trained against
--split place=same,wall=held_out,goal=held_out    # the memorization probe
--split place=ood                                 # unspecified traits default to held_out
```

Repeatable. Same compact grammar as `--schedule` and `--place_region`, so it
survives `asdict(cfg)` and reaches a results file as a string.

**`recorded` is not the same as all-`held_out`, and the distinction is the point.**
`recorded` replays the run's own validation envs. All-`held_out` mints *fresh*
envs that are also disjoint from training. The first answers "how did this
checkpoint do on its own val set", the second "does that generalize to another
draw from the same rule". Both are wanted; a CLI that could only express one
would quietly conflate them.

**Multiple combos per invocation, not per process.** The scaffold is 12 GB at
`Npos=1716` and the encoder load is not cheap; re-running the CLI per combo pays
that per combo. So `eval_checkpoint` splits into

```
prepare(...)   -> cfg, encoder, field, agent, embed_dim      # once
run_suite(...) -> the results dict for one env set           # per combo
```

with the combo loop between them. `build_field` is already the shared-field entry
point Phase 1 created; this is the same move one layer up.

**Output shape — nine readers, none of which may break.** `analysis/continual/
baseline.py`, `analysis/continual/plotting.py`, `train.py`, `train_rnn.py`,
`evaluation/rnn.py`, `tests/test_smoke_train.py` and three `run_*.sh` scripts all
read today's top-level `nav_det` / `discovery` / `exploration` keys. So: the first
combo's results stay at the top level exactly as now, and **every** combo also
appears under `results["splits"][key]`. A single-combo run is then byte-compatible
with today's output, and the combination × metric table is a view over `splits`.

### 5.2 What a minted val set must assert about itself

`make_val_set` does not currently run `verify_split` on its own output — noted as
undone in Phases 3 and 4, and this is where it becomes load-bearing. Once a CLI
mints an env set on request, the separation claim *is* the claim the eval is
making, so it has to be checked and reported, not assumed:

- run the pairwise checks on the minted set against `split.used`
- put the achieved minimum place gap and wall Hamming into `results["splits"][key]`,
  alongside the level combination, so a results file says what its envs were

### 5.3 `sample_places` cannot answer this question at the size it is now asked

`exclude` is the set of already-placed envs a draw must clear. Three callers, and
the sizes are the story:

| caller | `exclude` | size |
|---|---|---|
| `generate_split` | nothing — train and val are one draw, separated by `self_margin` | 0 |
| `Refresher._draw` | the fixed `base_val` footprints | 10 |
| `make_val_set`, `place=held_out`/`ood` | `split.used["place"]`, every offset training ever held | 80 → **23,961** |

The inner loop is `any(toroidal_gap(cand, size, o, s, period) < margin for o, s in
exclude)`, so a candidate costs O(|exclude|) in Python and `_lattice_places`
repeats that per slot per jitter attempt. **Measured: one failing
`make_val_set` at 23,961 excludes takes 78 s.** That is not a preflight problem —
it is what a user waits to be told "no" by `eval_all --split place=held_out`.

**The fix: a mask path, opted into by the caller.** Clearance is a purely
geometric predicate, so the legal set can be computed once instead of re-tested
per candidate. `toroidal_gap(cand, size, o, s, P) >= m` fails on an axis exactly
when `cand` lies in the wrapped interval

```
[ o - size - m + 1 ,  o + s + m - 1 ]          length size + s + 2m - 1
```

so each excluded env forbids one wrapped **rectangle** of candidate offsets. Paint
them into a 2D difference array, 2D cumsum, `legal = (coverage == 0)`. Draw from
`flatnonzero(legal)`, filter through `domain.contains` so `OutsideRect` keeps
working, and enforce `self_margin` incrementally among the few being drawn.

Measured **~50–77 ms at any `|exclude|`**, and exact — 0 mismatches against
`toroidal_gap` across 300 sampled cells at four tick counts.

```
ticks=  0   |used|=    80    legal offsets = 944,117
ticks=  8   |used|=   720                     10,808
ticks= 24   |used|= 1,999                      2,463
ticks=300   |used|=23,961                         12
```

**Opted into, not threshold-triggered.** A hidden `len(exclude) > N` switch would
make a run's offsets depend on which side of a magic number it landed, and
`Refresher` calls this every tick with 10 excludes — a flat 50 ms there would add
15 s to a 300-tick run against today's 3.9 ms. So `make_val_set` asks for the mask
and `Refresher` does not, and the choice is visible where it is made.

With the mask, the error can finally name the cause instead of suggesting capacity
remedies for a problem that is not capacity (~400 slots for 90 envs): *N legal
offsets remain of 2.94M; only M of those are also `self_margin` apart; training
used K offsets.*

**The mask also corrected a Phase 4 finding.** `place='held_out'` after a
refreshing run is **not** infeasible — see the correction note in the Phase 4
outcome. The rejection sampler was failing to *find* the ~10 surviving offsets,
at a hit rate of about 10⁻⁵ per probe, and I read that as their absence. The mask
finds them every time, out to 2000 ticks. What is real is that the headroom
collapses to ~10 envs, which is a constraint on `--num_val_envs`, not a failure.

**`ood` on an undeclared trait** already raises with the right explanation from the
domain layer (`Anywhere.complement()`, `SeedRange.complement()`). The CLI should
let that propagate rather than wrapping it.

### 5.3b The startup preflight

Refreshed placements are a pure function of `(seed, tick, place domain, size,
Npos, period, base_val, margin)` — nothing training does enters. So a preflight
does not *estimate* the end-state union, it **replays the exact ticks**: 1.15 s
for 300.

**What it reports is headroom, not a predicted failure.** The first version of
this plan expected it to catch an outright exhaustion; with the mask, that does
not happen (see the correction in the Phase 4 outcome — the legal set plateaus
around 10 rather than emptying). What does happen is that the largest held-out
place set a run can support falls from ~187 envs to ~10 over a few hundred ticks,
and that ceiling is invisible unless measured. A post-hoc eval asking for more
than it than would fail, at the end of a run, for a reason decided at the start.

So: replay to the end of the schedule, compute the legal mask, greedily pack, and
report the ceiling. Compare it against `cfg.num_val_envs`. ~2 s.

The mask answers *coverage*, not *packing* — 12 legal offsets is not 12 usable
envs — so the check is mask → greedy pack. Both the preflight and the eval run
**that same code**, which is the only arrangement in which the prediction cannot
disagree with what later happens.

Decided:

- **checks against `cfg.num_val_envs`**, the natural size of an eval set
- **records and proceeds** — a warning plus a `preflight` block in `world.json`,
  not a raise. A run that only ever uses `--split recorded` is genuinely fine with
  a tight union, and that is not the trainer's call to veto.

### 5.4 Out of scope, deliberately

**`--val_size` is Phase 6, and must error here rather than half-work.** Six sites
read the global `cfg.env.size` where they need the val set's, and every one fails
*silently*: `metrics.py:614` (`grid_size` → the denominator of `mean_coverage`),
`metrics.py:323,449` and both regimes' distractor exclusion box, and
`collector.py:192,352,366,541,566` (`visited_cells` shape and novelty). A val set
at a different size would return plausible numbers computed against the wrong
denominator. So Phase 5 accepts `size=same` only.

### 5.4 Out of scope, deliberately

**`--val_size` is Phase 6, and must error here rather than half-work.** Six sites
read the global `cfg.env.size` where they need the val set's, and every one fails
*silently*: `metrics.py:614` (`grid_size` → the denominator of `mean_coverage`),
`metrics.py:323,449` and both regimes' distractor exclusion box, and
`collector.py:192,352,366,541,566` (`visited_cells` shape and novelty). A val set
at a different size would return plausible numbers computed against the wrong
denominator. So Phase 5 accepts `size=same` only.

### 5.5 `train_rnn` and the RNN baseline build envs the same way — parity

Everything in Phases 1–4 landed on `train_navigate` alone:

| entry point | envs from | writes `world.json` |
|---|---|---|
| `train_navigate.py` | **generator** or legacy | **yes**, both paths |
| `train.py` / `train_phased.py` / `train_store.py` | legacy | no |
| `train_rnn.py` | `rnn_setup.build_envs` + `place_envs(np.random)` | no |
| `analysis/continual/baseline.py` | same | no |

`train_rnn.py:183` draws seeds from `RandomState(cfg.seed)` and, when
`input_grid_state=True`, places with `place_envs(..., np.random, "spread")` at
line 207 — the unrecorded global-RNG placement §1.4 is about. `baseline.py:270`
does the same. So today the baseline aligns with `agenthash` only through the
hand-matched draw-order convention documented at `agenthash.py:325-333`.

**Decided: both move onto the same generator path as `train_navigate`**, so a
baseline and an agent-hash run can be given the same declared world rather than
being talked into agreement. That means:

- `RNNTrainConfig` gains the same six generator fields `TrainConfig` has
  (`env_generator`, `place_region`, `goal_region`, `wall_seeds`, `place_margin`,
  `goal_val_frac`), as flat JSON-native values
- both call `generate_split` / `build_envs` and write `world.json`, on both the
  declared and legacy paths, exactly as `train_navigate` does
- **the `build_envs` name collision gets resolved, not shadowed.**
  `training/rnn_setup.build_envs(cfg, rng)` and
  `world/generate.build_envs(specs, env_cfg, mode)` are unrelated functions with
  one name, and `train_rnn.py:32` imports the first. The config-driven one becomes
  `rnn_setup.build_envs_from_config`; the spec-driven one keeps the name, since it
  is the one everything is converging on.

Two honest limits. The RNN stack builds a scaffold only under
`input_grid_state=True`, so **place is an axis it can generalize along only
then** — without grid state its envs' offsets are unobservable to it, and a
place split is a distinction it cannot see. And `RNNTrainConfig` has no encoder,
so `derive_margin` is unavailable there: `place_margin` must be given explicitly
or default to the same value the agent run used.

`agenthash`'s `--env_seed` stays, as the explicit baseline-matching path for runs
that predate this.

### Build order

Each of the first three is its own commit, because each changes something
different about what existing numbers mean.

1. **5.0** — spec discovery inside `build_eval_world`. Fixes the live §1.4 bug at
   four call sites at once.
2. **5.3** — the mask path and the error message. Must precede the preflight, or
   the preflight is either a heuristic or 12 minutes.
3. **5.3b** — the preflight, sharing 5.3's code.
4. **5.1 / 5.2** — `--split` on `eval_all`, the `prepare` / `run_suite` cut,
   `results["splits"]`, verify-and-report.
5. **5.5** — `agenthash --split`, then `train_rnn` / `baseline` parity.

### Tests

- `build_eval_world` prefers a discovered spec over the replay, and the offsets it
  returns equal the recorded ones — the §1.4 regression test
- a run with no `world.json` still evaluates, and still warns
- `recorded` reproduces `base_val` exactly; all-`held_out` does not, and is disjoint
- each level combination lands under `results["splits"]` with its achieved gaps
- a single-combo run's JSON is key-for-key what today's is
- the field is built **once** across N combos (object identity)
- `place=held_out` after refresh exhaustion names the union
- `size` other than `same` errors and points at Phase 6
- **the mask agrees with the scan** — same legality verdict on every cell, over
  several sizes, margins and periods, including the torus seam and mixed excluded
  sizes. This is the one that matters: the mask is a rewrite of the predicate, and
  a subtly wrong one would silently pass envs that overlap what training used
- the preflight's verdict equals what `make_val_set` actually does at that tick —
  same code, asserted rather than assumed
- `train_rnn` / `baseline` write a `world.json` whose specs rebuild their envs
  bit-identically

### Risk register

| Risk | Mitigation |
|---|---|
| 5.0 silently changes every downstream number | Its own commit, and it is a *fix* — the numbers were wrong before |
| A fifth call site rejoins the legacy path | Discovery lives inside `build_eval_world`, not at the call sites |
| Output shape breaks nine readers | First combo stays at top level; `splits` is additive |
| 12 GB scaffold rebuilt per combo | `prepare` / `run_suite` split, pinned by an identity test |
| Minted set is assumed separated rather than checked | 5.2 — verify and report the achieved gaps |
| `--val_size` ships half-working | 5.4 — rejected outright, with the six sites named |
| The mask disagrees with the predicate it replaces | Equivalence test against the scan path, not just a spot check |
| Preflight predicts something the eval then contradicts | They run the same code; pinned by a test |
| Renaming `build_envs` breaks an unnoticed importer | Grep before, full import sweep after |

---

## Phase 5 — outcome

Four commits, in the order the plan set: `a9b6c6a` spec discovery, `47a019c` the
mask, `d20bf83` the preflight, and this one for `--split` and RNN parity.

### 5.0 — the bug that was already there

All four eval CLIs called `build_eval_world` with three arguments, so every
post-hoc evaluation since Phase 3 took the RNG replay and scored checkpoints on
scaffold patches training never used, while a truthful `world.json` sat unread
beside them. `build_eval_world` now takes `ckpt_path` and finds the record
itself — discovery lives there rather than at the call sites so a fifth cannot
quietly rejoin the legacy path. Since "nobody passed it" and "this run has no
record" are indistinguishable at runtime, a test reads the source of all four.

### 5.3 — `sample_places` could not answer the question at the size it was asked

One *failing* `make_val_set` at 23,961 excludes took **78 s**, because each
candidate was tested against every excluded env in Python. That is what a user
waited to be told "no" by `eval_all --split place=held_out`.

Clearance is closed-form: an excluded env blocks candidate offsets over the
wrapped interval `[o - size - margin + 1, o + other_size + margin - 1]` per axis
— one rectangle, **asymmetric** when the two envs differ in size. `legal_mask`
paints them into a 2D difference array and cumsums: **~50–77 ms at any
`|exclude|`**, exact against `toroidal_gap` on every cell of every scaffold
tried. Opted into by the caller, not switched on a size threshold — `Refresher`
calls this every tick with ten excludes, where a flat 50 ms would add 15 s to a
300-tick run, and a hidden threshold would make a run's offsets depend on which
side of a magic number it landed.

**This corrected a Phase 4 finding** — see the correction note there.
`place=held_out` is not infeasible after refresh; the sampler could not find the
~10 survivors at a hit rate near 10⁻⁵.

### 5.3b — the preflight

Replays the exact ticks (they are a pure function of seed and tick), **1.4 s**
for a 300-update run, draw-only so no envs are built. Reports a *ceiling*, and
records it in `world.json` under `diagnostics.preflight`; never refuses.

Two things it got wrong first, both found by measuring:

- **Greedy packing is order-dependent** and `make_val_set` shuffles with the
  *val* seed, which the preflight cannot know. A verdict from one fixed order
  over-promised: at seeds 1 and 4 with `n=4`, one order reported room and
  `make_val_set` then failed for two of four val seeds. It now takes the worst
  of five orders, capped at `n_val_envs`. It errs toward *no* — a spurious
  warning costs a line, a spurious all-clear costs a re-run.
- **A domain that runs dry is a different failure, and the only fatal one.** It
  raises partway through, hours in, at a tick decided before the run started, so
  the replay catches it and `train_navigate` **refuses at startup**. A shrinking
  eval ceiling is still recorded-and-proceed — a run that only ever evaluates on
  `--split recorded` is fine with a tight union, and that is not the trainer's
  call to veto. These are not the same kind of thing, and only one of them is
  worth stopping for.

### 5.1 / 5.2 — `--split`

```
--split recorded                                  # the run's own base_val
--split place=same,wall=same,goal=same            # the memorization probe
--split place=ood                                 # unnamed traits -> held_out
```

Repeatable; `eval_checkpoint` splits into the env-independent half and
`run_suite`, so **one scaffold serves every combination** (pinned by an identity
test — at `Npos=1716` a rebuild per combination would cost more than every
evaluator put together).

`recorded` is deliberately not a synonym for all-`held_out`: the first asks how
a checkpoint did on the set it was scored against, the second whether that
survives another draw from the same rule.

Output stays compatible for the nine readers of this file: the first combination
sits at the top level exactly as before, `splits` is additive. Each minted set's
separation is **measured** into `split_report` — and a violated `held_out` claim
raises, because a number in a report is not a strong enough place for a
generator bug.

`--val_size` is rejected outright with the six silently-failing sites named.

### 5.5 — `train_rnn` and the RNN baseline on the same record

Both now call one `rnn_world`, take the same six declared-domain flags, and
write the same `world.json`. So a baseline run and an agent-hash run can be
handed one record instead of being talked into agreement by the draw-order
convention at `agenthash.py:325-333` — a comment, enforced by nothing.

The `build_envs` collision is resolved rather than shadowed:
`rnn_setup.build_envs_from_config` draws envs from a config,
`world.generate.build_envs` rebuilds them from a record.

**The scaffold is built for the generator, not only for the agent.** Under
`input_grid_state=False` the RNN never observes where its envs sit — but the
placement is still part of the world's identity, an agent-hash run on the same
`world.json` *does* observe it, and the separation guarantees are stated in
scaffold coordinates either way. So `--env_generator` builds a scaffold whenever
asked and records the offsets regardless. Verified: the same config with grid
state on and off draws the **same** envs at the **same** offsets — the declared
world is a property of the config, not of what the agent happens to see. Only
the *legacy* path stays conditional, because its historical draw must not move.

One limit, refused rather than faked: **`--place_margin` must be explicit.**
`derive_margin` reads the scaffold's cosine-vs-distance curve, which needs an
encoder; this stack has none, and a borrowed constant would be wrong at a
different `Npos`.

`split_diagnostics` now reports an empty `cosine` block when the field has no
`encoded_Phi`, rather than denying the RNN stack a world record for lacking an
encoder it never had.

### `--split` everywhere, through one resolver

`analysis/trajectories.py` and `analysis/phase_decoding` (`exp1`, `exp2`, and
`RolloutEngine`) take `--split` too, so a trajectory grid can be drawn in arenas
whose goal region the run never trained on, and phase decodability can be asked
of held-out or OOD envs rather than only of the run's own validation set.

All five drivers resolve their env set through **one** call —
`checkpoint_io.eval_env_set`, reached directly by `eval_all` for its
N-combination table and via `eval_world_for_split` by the four single-set
drivers. That matters more than the flags: each used to build its own eval
world, so a level would otherwise have meant whatever that driver's copy of the
logic did. Two source-reading tests keep it that way — one that no driver
resolves a world without the record, one that every driver still accepts a
split. Both were mutation-checked.

### Left undone

- `train.py` / `train_phased.py` / `train_store.py` still do not write
  `world.json`. They are the remaining legacy trainers.

---

## Phase 6 — detailed plan

**Goal.** Evaluate a checkpoint on validation envs of a size it never trained on:
`--val_size N`, which Phase 5.4 currently refuses.

**Acceptance gate.** 743 tests pass, goldens byte-identical, and `--val_size`
equal to the training size returns *exactly* what `--split` returns today. That
last one is the real gate: it says the size plumbing changed nothing for every
run that does not use it.

### 6.0 What is actually free, and what the roadmap over-counted

**The architecture is genuinely free.** `channels.input_dim(cfg, embed_dim,
sensory_dim)` sums channel widths over `embed_dim` and `sensory_dim`, and
`sensory_dim` is the *ray count* (`observation_size`), not the grid width. A
size-20 env feeds a size-8-trained agent an identically shaped observation.
Confirmed by running one: `env.size=12` against `cfg.env.size=6` produced a
12-dim observation and completed every evaluator without an error.

**The roadmap's site list is wrong in both directions.** It named six; the eval
path has **three**, and five of the six it listed are in `rollout/collector.py`,
which **no evaluator uses** — the collector is training-only (`train_navigate`,
`train`, `train_phased`, `train_store`). Those five matter for *training* at a
new size, which this phase does not do. Meanwhile it missed the analysis drivers,
which have five more.

Most of `metrics.py` already reads `env.size` per env — `random_start` does so at
lines 330, 455, 506, 761, 808, 916, 955. The global reads are the exception, not
the rule.

### 6.1 The sites that actually matter, and which fail silently

| site | what it is | failure |
|---|---|---|
| `metrics.py:614` | `grid_size` → `total_positions`, the **denominator of `mean_coverage`** | **silent** |
| `metrics.py:323, 449` | distractor exclusion box | **silent** — distractors can land inside a larger val env's own footprint |
| `analysis/trajectories.py:316` | coverage denominator | **silent** |
| `analysis/trajectories.py:168, 284, 339` | distractor exclusion box | **silent** |
| `analysis/trajectories.py:764, 767` | plot grid extent | visual only |
| `analysis/phase_decoding/rollout.py:138, 175` | quadrant split + distractors | **silent** |
| `checkpoint_io.py:355` | `scaffold_layout_dict`'s recorded `env_size` | **silent** |

Measured, not assumed: an `env.size=12` set evaluated under `cfg.env.size=6`
reported `mean_coverage` against a denominator of 36 instead of 144 — **4× the
truth**, no error. That is the number the whole phase exists to not get wrong.

The fix is uniformly "read it from the env, not the config", and the envs already
carry it (`EnvSpec.size` → `dataclasses.replace(env_cfg, size=s.size)` in
`build_envs`). Coverage should be computed **per env** and then averaged, not
against one global denominator — that is also what makes a mixed-size set
meaningful later, without doing mixed sizes now.

### 6.2 A bug in `make_val_set`, found while planning this

`make_val_set` passes the excluded *train* envs to `sample_places` labelled with
the **validation** size:

```python
exclude=[(o, env_size) for o in used_place]      # env_size is the VAL size
```

`used_place` holds training offsets, whose envs are the *training* size. The
forbidden span is `[o - size - margin + 1, o + other_size + margin - 1]` —
asymmetric in the two sizes — so mislabelling inflates the excluded region.
Measured at `Npos=132`, train size 6, val size 20, margin 10: **6030 legal
offsets instead of the true 8581**, a 30% over-exclusion. Invisible today only
because every val set has so far been the training size, where the two labels
coincide.

The immediate fix is to label them with the train size. The deeper one is that
`split.used["place"]` stores bare offsets while `used["size"]` stores a separate
set, so the pairing is lost — if size refresh ever produces mixed train sizes,
`used["place"]` has to become offset→size pairs. Worth doing now rather than
leaving a second latent version of the same bug.

### 6.3 Wall codes of different lengths are not comparable

`wall_hamming` **raises** across sizes (`operands could not be broadcast
together with shapes (4,8) (4,20)`), because a wall code is
`(4, size * wall_resolution)`. This is loud rather than silent, which is
fortunate, but three functions assume one size and would hit it:
`split_diagnostics:682`, `verify_split:706`, `val_set_report:799`.

There is no correct Hamming number to report here — the two codes do not live in
the same space. So the honest treatment is: **across sizes, wall novelty is seed
disjointness only**, the Hamming margin is reported as `None`, and the reason
says so rather than the field being quietly absent.

### 6.4 A larger val set needs scaffold room the split was never packed for

`generate_split`'s capacity preflight is run at the *training* size. A bigger
validation env forbids a much larger region, and the train envs already placed
may leave nothing. Observed immediately: 2 envs of size 12 on an `Npos=35`
scaffold holding 3 train envs of size 6 at margin 3 → **0 legal offsets**, and
the error (correctly) blamed the exclusion set.

That message is right but incomplete for this case. It should name the size
change as a candidate cause, the way the Phase 4 size-refresh preflight does —
the same numbers passed at the training size.

### 6.5 Held-out goals get confined to the old arena

`make_val_set` filters the goal pool to `c[0] < env_size and c[1] < env_size`.
Every cell in `goal_cells_val` was drawn at the *training* size, so at a larger
validation size the filter is a no-op and **goals never appear outside the
original footprint** — the outer band of a size-20 arena would never hold a goal.

Silent, and it would make "size OOD" quietly test something narrower than it
claims. Three options, and the choice belongs in the plan rather than in code:

1. **Rescale** the held-out cell set into the new arena — keeps the *rule*
   (a ring stays a ring) but the specific cells are no longer held out.
2. **Extend**: hold out the train-size cells, and treat every cell beyond the old
   footprint as also legal. Goals reach the new region; disjointness survives.
3. **Declare it**: require `goal=same` or an explicit region at a new size.

**Recommendation: (2).** It is the only one where "held out" keeps meaning what
it means elsewhere — disjoint from what training used — while letting the new
geometry actually appear.

### 6.6 The step-budget confound

A bigger arena has more cells, so `mean_coverage` at a fixed `max_steps` must
fall whether or not anything about the policy generalizes. Reporting one number
would confound capability with budget.

So report at **both** a fixed `max_steps` and one scaled by `size²` (the cell
count, which is what coverage is a fraction of), and put both in the results
under their own keys. Navigation is a different story — path length grows like
`size`, not `size²` — so nav gets `size`-scaled budgets. Neither scaling is
"correct"; the point is that a reader can see which claim survives which.

### 6.7 CLI surface

`--val_size N` on `eval_all`, replacing the Phase 5.4 refusal. It reaches
`make_val_set(..., size=N)`, which already exists and is tested. Results are
keyed by `(split, size)` so a size sweep is a column in the same table `--split`
already produces.

`analysis/trajectories.py` and `phase_decoding` take it too, once 6.1 lands —
their reads are on the same list.

### 6.8 Out of scope, and why

- **Mixed sizes within one env set.** `generate_split._single_size` refuses them
  and stays refusing. Uniform-val-size is the decided ship (roadmap), and it is
  what keeps `sample_places`' single `size` parameter honest.
- **Training at a new size.** That is `--refresh_size`, which Phase 4 built and
  gated off pending a multi-value `Sizes` domain. It needs the five
  `rollout/collector.py` sites, which this phase does not touch. Keeping them
  apart is deliberate: an eval-side size change cannot corrupt a training run.

### Build order

1. **6.2** — the exclusion-size bug and the `used["place"]` pairing. It is a live
   bug, independent of the feature, and everything else sits on top of it.
2. **6.1** — read size from the env; per-env coverage denominators.
3. **6.3 / 6.4** — the diagnostics and the placement message.
4. **6.5** — the goal-cell decision.
5. **6.6 / 6.7** — budgets and the CLI.

### Tests

- **`--val_size` equal to the training size is a no-op**: byte-identical results
  to the same `--split` without it. The gate for the whole phase.
- **the coverage denominator is the val set's**: a size-12 set under a size-6
  config reports coverage against 144, and the pre-fix behaviour (36) fails it
- distractors never land inside a val env's own footprint, at either size
- `legal_offsets` with correctly-labelled excludes matches the ground-truth
  predicate over the whole scaffold, at differing sizes (the 6030-vs-8581 case)
- `wall_hamming` across sizes reports `None` and says why, rather than raising
  from inside a diagnostic
- a val size the scaffold cannot hold names the size change
- held-out goals reach cells outside the training footprint (6.5, option 2)
- the architecture claim, asserted rather than assumed: a size-N val set feeds
  the same `input_dim` as the training size

### Risk register

| Risk | Mitigation |
|---|---|
| A size change silently rescales a metric | 6.1, and the equal-size no-op gate |
| `used["place"]` keeps losing the size pairing | 6.2 fixes the representation, not just the call site |
| Wall diagnostics raise from inside a report | 6.3 — report `None` with a reason |
| Coverage read as capability when it is budget | 6.6 — both scalings, both reported |
| "Size OOD" quietly tests only the old footprint | 6.5 — decided before code, option (2) |
| Eval-side size work leaks into training | 6.8 — the collector sites stay untouched |
