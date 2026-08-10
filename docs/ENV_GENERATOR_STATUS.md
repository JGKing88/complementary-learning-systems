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
| 4 | Per-trait refresh | not started |
| 5 | Eval CLIs, mix-and-match | not started |
| 6 | Size OOD | not started |

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
