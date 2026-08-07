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
| 2 | Generator, domains, separation | not started |
| 3 | Serialization + train wiring | not started |
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
