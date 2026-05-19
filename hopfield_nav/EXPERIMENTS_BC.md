# Hopfield-Nav DAgger / BC progress log

Tracks the supervised-training (`--training_mode bc`) experiments added
2026-04-24. All runs use `run_bc.sh` parameterized by env vars; configs
below show the differences from the baseline. Project on wandb:
`hopfield-nav-bc`.

Common defaults (unchanged across A–AO unless noted):
- encoder `encoders/run_20260422_185816/encoder_best.pt`, fwhm=0.25
- size=8, observation=12, lambdas=11/12/13, Np=400, static-vectorhash
- `movement_mode=continuous`, `init_log_std=-0.5`
- 600 update budget (most runs killed by 2h SBATCH wall well before this)
- batch_envs=16, steps_per_rollout=128, envs_per_world=20
- val_distractors `0 1 3`, eval_every=25, n_val_trials=32
- `bc_supervise_explore` ON, `bc_n_minibatches=4`, `bc_store_weight=1.0`
- `realistic_steps_per_env=1000`

Eval columns below: `gf@d=N` is `goal_find_rate` at distractor count N
(eval expl block); `cov` is mean_coverage at d=0; `upd` is the last
training update reached before TIMEOUT.

---

## Phase 1 — initial hyperparam scan (A–L), zero distractors

Goal: find a baseline that learns at all. Inputs were the lean default
(`hs/pa/sn/-enc`, no `prev_reward` or `hopfield_raw`). Most A–E were
killed at update 50 to free GPUs; F–L ran the 2h budget.

| tag | wandb    | lr   | ep | sw  | ent  | other        | upd | gf@d=0 | cov  | notes |
|-----|----------|-----:|---:|----:|-----:|--------------|----:|-------:|-----:|-------|
| A   | 9s57fbkr | 3e-4 | 2  | 1.0 | 0    | baseline     |  50 |  0.33 | 0.27 | early kill |
| B   | 6ps7ge9o | 3e-4 | 2  | 3.0 | 0    | sw bump      |  50 |  0.15 | 0.16 | early kill |
| C   | ispcqaio | 3e-4 | 2  | 1.0 | 0    | sup=0        |  50 |  0.16 | 0.16 | early kill |
| D   | of2j2mr7 | 1e-3 | 4  | 1.0 | 0    | lr+ep bump   |  50 |  0.47 | 0.43 | early hint  |
| E   | rpfj4kso | 3e-4 | 2  | 1.0 | 0.01 | small ent    |  50 |  0.25 | 0.21 | early kill |
| F   | l9rq0tfu | 3e-4 | 2  | 1.0 | 0    | A re-run     | 350 |  0.10 | 0.13 | collapsed |
| G   | w0wk426t | 3e-4 | 2  | 3.0 | 0    | B re-run     | 330 |  0.07 | 0.12 | collapsed |
| **H** | kkbiwe3s | 1e-3 | 4 | 1.0 | 0    | D re-run     | 340 |  **0.78** | 0.52 | first solid run |
| I   | qecqlinm | 3e-4 | 2  | 1.0 | 0.01 | E re-run     | 350 |  0.16 | 0.17 | collapsed |
| J   | j3i2t71d | 3e-4 | 2  | 1.0 | 0    | discrete     | 340 |  0.12 | 0.14 | discrete worse |
| K   | 3cflzm91 | 1e-3 | 4  | 1.0 | 0.02 | H + ent      | 330 |  0.69 | 0.50 | ent=0.02 fine |
| L   | nfcojfeh | 1e-3 | 4  | 1.0 | 0.10 | discrete+ent | 340 |  **0.88** | 0.55 | high single point but discrete-only and d=1/3 still 0.09 |

**Findings:**
- `lr=1e-3, ep=4` is the floor. `lr=3e-4, ep=2` is unstable — runs A/E/I
  start fine then collapse by update 200+ (over-fit move head with weak
  signal-to-noise).
- `bc_store_weight` doesn't need to grow when pos_weight is correct
  (memory note `project_hopfield_nav_bc.md`).
- Discrete movement underperforms continuous on goal_find at matched
  ent unless ent is very high (L) — not worth pursuing further.
- **All distractor evals stuck at ~0.10** since training had zero distractors.

## Phase 2 — H replication across seeds (M–O)

| tag | wandb    | seed | upd | gf@d=0 | gf@d=1 | gf@d=3 | cov |
|-----|----------|-----:|----:|-------:|-------:|-------:|----:|
| M   | 97cazbwz | 0 | 220 | 0.79 | 0.13 | 0.12 | 0.50 |
| N   | wvcumtkv | 1 | 350 | 0.81 | 0.07 | 0.08 | 0.46 |
| O   | vdl0bdw3 | 2 | 350 | 0.54 | 0.10 | 0.08 | 0.47 |

H config replicates at d=0 (mean ~0.71) but seed=2 is weak (0.54). The
d=1/d=3 floor confirmed: with no distractor exposure, the agent
follows whatever Hopfield points at — including noise from a
distractor-populated memory.

## Phase 3 — input enrichment (P, Q, R)

Added `--input_prev_reward --input_hopfield_raw` (input_dim 14 → 18) to
match the PPO phased recipe. P–R kept zero distractors.

| tag | wandb    | enrich extras | upd | gf@d=0 | cov | notes |
|-----|----------|---------------|----:|-------:|----:|-------|
| P   | vfr0bswa | pr+hr         | 350 |  0.41  | 0.36 | enrichment alone hurt |
| Q   | 1rnxoi5f | (none)        | 260 |  0.65  | 0.45 | re-baseline |
| R   | 69lhn89i | pr+hr         |  60 |  0.08  | 0.15 | early kill |

**Finding:** input enrichment in isolation didn't help — the extra
channels need a teacher signal that uses them. That comes from Phase 4.

## Phase 4 — distractor training (S, T, U) — the breakthrough

Added `--n_train_distractors N` so the world starts with N pre-stored
non-goal patterns in Hopfield. Teacher logic still says "follow the
Hopfield direction post-memory", but the pattern of stored cells now
includes non-goals — the student must learn to *not* always trust
hopfield-derived signals.

| tag | wandb    | dist | enrich | upd | gf@d=0 | gf@d=1 | gf@d=3 | cov |
|-----|----------|-----:|--------|----:|-------:|-------:|-------:|----:|
| S   | n28csekd | 1 | (none) | 340 | **0.62** | **0.64** | **0.61** | 0.51 |
| T   | 2joy5h0c | 3 | (none) | 300 | 0.61 | 0.62 | 0.58 | 0.42 |
| U   | 11upoig8 | 3 | pr+hr  | 300 | 0.65 | 0.69 | 0.66 | 0.53 |

**Result:** distractor-eval gap effectively closed in a single change.
gf@d=1 climbed from ~0.10 (Phase 2) to ~0.65. This invalidates the
prior memory note claim that "with distractors reach_rate drops to
~13%" — that was about a model with `n_train_distractors=0`.

## Phase 5 — tuning around distractor + entropy (V–AC)

Sweeping `lr ∈ {3e-4, 5e-4, 1e-3}`, `ent ∈ {0.02, 0.03, 0.05}`,
`dist ∈ {1, 2, 3}`, with enrichment now standard.

| tag | wandb    | dist | lr   | ent  | seed | upd | gf@d=0/1/3       | cov |
|-----|----------|-----:|-----:|-----:|-----:|----:|------------------|----:|
| V   | llbl4us9 | 1 | 1e-3 | 0.02 | 0 | 200 | 0.69/0.70/0.66   | 0.57 |
| W   | e8b44nl5 | 3 | 1e-3 | 0.02 | 0 | 200 | 0.65/0.64/0.66   | 0.48 (no enrich) |
| X   | j69l3fw5 | 1 | 1e-3 | 0.05 | 0 | 200 | 0.52/0.52/0.46   | 0.41 (ent too high) |
| Y   | 58tquj0c | 1 | 3e-4 | 0.02 | 0 | 200 | **0.79/0.80/0.74** | 0.57 |
| **Z** | urlt9egv | 1 | 1e-3 | 0.02 | 1 | 200 | **0.81/0.82/0.82** | 0.47 |
| AA  | 0qo65rjh | 2 | 1e-3 | 0.02 | 0 | 200 | 0.73/0.70/0.69   | 0.53 |
| AB  | ankvgoc5 | 1 | 1e-3 | 0.03 | 0 | 200 | 0.55/0.53/0.53   | 0.44 |
| AC  | nuo6461m | 1 | 1e-3 | 0.02 | 0 | 170 | 0.81/0.78/0.78   | 0.54 (V-equivalent rerun) |

**Best so far: bc-Z (`urlt9egv`)** at gf=0.81/0.82/0.82 across all
distractor counts after only 200 updates — tied for the strongest
distractor-balanced result. ent=0.02 is the sweet spot; ent=0.03–0.05
hurts.

## Phase 6 — dist=2 family with seed sweep (AD–AK)

Trying to find a `dist=2` recipe that's competitive with the dist=1
result and replicates across seeds.

| tag | wandb    | dist | lr   | ep | ent  | seed | enrich | upd | gf@d=0/1/3 | cov |
|-----|----------|-----:|-----:|---:|-----:|-----:|--------|----:|------------|----:|
| AD  | utoqv61z | 2 | 1e-3 | 4 | 0.02 | 1 | pr+hr | 200 | 0.72/0.73/0.74 | 0.49 |
| AE  | ne6rtsqc | 3 | 1e-3 | 4 | 0.02 | 0 | pr+hr | 200 | 0.74/0.72/0.69 | 0.51 |
| AF  | g8yl9ryi | 2 | 1e-3 | 4 | 0.02 | 0 | (none)| 200 | 0.55/0.51/0.52 | 0.47 (no enrich → drop) |
| AG  | 2idb3mwh | 2 | 5e-4 | 4 | 0.02 | 0 | pr+hr | 200 | 0.73/0.78/0.75 | 0.55 |
| AH  | hfpbc0sg | 2 | 1e-3 | 8 | 0.02 | 0 | pr+hr | 200 | 0.73/0.74/0.77 | 0.53 |
| AI  | fvwcikwy | 2 | 1e-3 | 4 | 0.02 | 0 | pr+hr | 130 | **0.82/0.76/0.78** | 0.50 (only 130 upd) |
| AJ  | 4z49qlgt | 2 | 1e-3 | 4 | 0.02 | 2 | pr+hr | 200 | 0.49/0.53/0.53 | 0.40 (seed=2 weakest, again) |
| AK  | czw4cufp | 2 | 1e-3 | 4 | 0.02 | 0 | pr+hr | 210 | 0.61/0.68/0.66 | 0.51 |

**Findings:**
- enrichment matters at dist=2 (AF without enrichment dropped 0.20 vs
  AI/AK with enrichment).
- Seed=2 underperforms across multiple recipes (O 0.54, AJ 0.49) —
  consistent seed brittleness, mirrors the V3 phased-PPO finding that
  seed=2 has a different attractor.
- `ep=8` (AH) doesn't beat `ep=4` (AI/AK) — wasted compute.

## Phase 7 — last batch (AL, AM, AN, AO) — submitted 2026-04-25 ~01:52 EDT

Attempted ablation around the AI/AK setpoint to nail down lr × epochs ×
ent, all at `dist=2 + enriched`. All four hit the 2h SBATCH wall at
~03:50 EDT, reaching only ~200/600 updates.

| tag | wandb    | lr   | ep | ent  | seed | upd | gf@d=0/1/3   | cov |
|-----|----------|-----:|---:|-----:|-----:|----:|--------------|----:|
| AL  | cjcgqedq | 5e-4 | 4 | 0.02 | 0 | 200 | 0.72/0.69/0.74 | **0.60** |
| AM  | w1v7qr4h | 1e-3 | 8 | 0.02 | 0 | 220 | 0.70/0.74/0.70 | 0.48 |
| AN  | bpow8s05 | 1e-3 | 4 | 0.04 | 0 | 220 | 0.58/0.59/0.58 | 0.49 |
| AO  | y7ol325s | 1e-3 | 4 | 0.02 | 1 | 220 | **0.80/0.79/0.79** | 0.40 |

**AL/AM/AN/AO ablation factors** (vs the AI/AK setpoint = lr=1e-3,
ep=4, ent=0.02, seed=0, dist=2, enriched):
- **AL**: drop lr to 5e-4 → similar gf, **highest cov of the batch
  (0.60)**, dist=3 even slightly higher than dist=0.
- **AM**: bump ep to 8 → no clear benefit; gf flat, cov dropped from
  AI's 0.50 to 0.48.
- **AN**: bump ent to 0.04 → clear regression (gf ~0.58 vs 0.70+
  baseline). Confirms ent=0.04 is past the sweet spot.
- **AO**: seed=1 → top gf of the batch (~0.80) but lowest cov (0.40),
  matching the seed=1 = "fast convergence" pattern from Z (`urlt9egv`).

**Loss curves still trending down at 220 updates.** None of these
runs converged. To get a clean read we need to either bump
`--time` past 2h or drop `N_UPDATES` to 300 (still well past the
plateau most runs show by ~150 updates, where mean_reward stops
moving and only loss continues to settle).

---

## Summary across all 41 runs

**Strongest configs by metric (all dist=eval, dist-balanced):**
- gf-balanced: **bc-Z** (urlt9egv) gf=0.81/0.82/0.82, 200 updates,
  lr=1e-3 ep=4 ent=0.02 dist=1 seed=1 enriched.
- cov-leading: **bc-AL** (cjcgqedq) cov=0.60 with gf=0.72/0.69/0.74,
  200 updates, lr=5e-4 ep=4 ent=0.02 dist=2 seed=0 enriched.
- gf-d=0 single point: **bc-L** (nfcojfeh) gf=0.88, but discrete-only
  and d=1/3 collapse to 0.09.

**Recipe consensus** (extracted across the H, S, V, Y, Z, AI lineage):
- `lr=1e-3, bc_epochs=4, bc_n_minibatches=4`
- `bc_store_weight=1.0` (with pos_weight inside the store BCE — see
  `project_hopfield_nav_bc.md`)
- `bc_move_ent_coef=0.02` (0 → unstable, ≥0.04 → regress)
- `init_log_std=-0.5`, `bc_supervise_explore` ON
- `n_train_distractors ∈ {1, 2}` (3 also fine, 0 → fails distractor eval)
- inputs enriched with `--input_prev_reward --input_hopfield_raw`
- `movement_mode=continuous`

**Open questions:**
1. **Convergence**: every "best" run was killed by the 2h wall at 200–340
   updates; the 600-update budget has never been exhausted. Need to bump
   `--time` to 4h or run `--n_updates 300` to confirm where these
   plateau.
2. **Seed=2 brittleness**: O (0.54) and AJ (0.49) consistently lag
   seeds 0/1. Same seed pathology as the V3 phased PPO. May indicate a
   bad early-trajectory attractor that more BC updates would escape, or
   a genuine init dependence.
3. **Coverage vs goal_find tradeoff**: AO (high gf, low cov) and AL
   (high cov, mid gf) suggest these don't move together. Eval expl
   coverage at d=0 isn't a primary metric per project memory, but the
   gap is suspicious — does AO succeed by short, direct goal-finding
   trajectories that don't visit much?

## Phase 5 (2026-04-30): minimal-input regime, dilution, and unification

Targets shifted to `mean_steps < 10` (deterministic nav follow) and
`mean_coverage > 0.50` (per-rollout pre-memory exploration), 8×8 grid.
Input set frozen during this phase to:
`current_reward + prev_reward + raw_hopfield_q + sensory` (no prev_action,
no encoded_state). Distractor range broadened to `n_train_distractors_min=0
n_train_distractors_max=10` to test robustness.

### Lineage

| tag | recipe Δ vs prev | best sr | best ms (sr≥.5) | best cov | outcome |
|---|---|---|---|---|---|
| AP | inputs trimmed (no sensory, no prev_action) | 0.40 | — | 0.13 | **collapsed** sr→0.05–0.20 → confirmed agent needs position info |
| AQ | AP + sensory back | 0.78 | 73 | 0.48 | random walk + lucky goal hits, ms ~75 plateau |
| AR | AQ + ent_coef 0.02→0.01 | 0.78 | 75 | 0.47 | indistinguishable from AQ |
| AS | AQ + dist_max 10→3 | 0.84 | 72 | 0.51 | small lift on cov, ms unchanged |
| AT | AS + unit-vector hopfield (`--no-input_hopfield_raw`) | 0.97 (u=20) | 24 (u=20) | 0.40 | **breakthrough at u=20**, then dilution collapse to AS-territory |
| AU | AT + `--no-bc_supervise_explore` | 1.00 | **4.7** ✓ | 0.10 | **nav target hit**, locked through u=750. cov stuck at random-walk floor |
| AV | AT + nav_weight=5 (per-step trust_hop loss weight) | 0.66 | 65 | 0.44 | weight too low to fight dilution |
| AW | AT + nav_weight=20 | 0.80 | 51 | 0.54 ✓ | hits cov target; nav still 5× optimal |
| AX | resume AU u=100 + supervise_explore | 0.71 | 79 | 0.39 | **bug**: optimizer state pinned lr=1e-3 — fixed in train.py |
| AY | AX with lr=5e-5 (lr-fix applied) | 0.77 (u=20) | 19 (u=20) | 0.12 | nav preserved at low lr but cov barely moved |
| **AZ** | AT + `--input_goal_in_memory` (1-bit phase indicator) | **1.00** | **5.8** ✓ | **0.50** ✓ | **both targets hit** at u=170 simultaneously. Nav locked from u=20, cov climbs steadily. |

### Key findings

1. **`q_full` magnitude is the nav-learning blocker in raw mode.** Probe
   (`probe_alignment.py`) on AQ.s0 u=700 showed `cos(policy, q_full) ≈ 0.03`
   (random-baseline) while `cos(q_full, optimal) ≈ 0.99` — agent ignored a
   correct signal because `‖q_full‖ ≈ 0.15` was indistinguishable from
   pre-memory zero. Switching to unit-vector mode (drop
   `--input_hopfield_raw`) gave nav at u=20.

2. **Dilution from novelty labels destroys learned Hopfield-following.**
   Without a phase indicator, the policy must average two different mappings
   over the same input distribution. Most label-mass is novelty (only post-
   store-at-goal steps fire trust_hop), so novelty wins. Demonstrated in AT
   collapse (u=20 → u=70 regression), AV (`nav_weight=5`), AW (`nav_weight=20`).

3. **Per-step label upweighting ≠ phase indicator.** AV and AW tested
   `bc_nav_weight ∈ {5, 20}`; both still showed nav decay over training. The
   policy can't selectively engage Hopfield-following without a discriminator
   in the input — weighting just slows the dilution.

4. **Curriculum (load-then-supervise) does not preserve nav.** AX/AY tested
   resuming AU u=100 with `supervise_explore=True`. Even at lr=5e-5,
   novelty-supervision gradients perturb shared-policy weights enough to
   slip nav.

5. **The `input_goal_in_memory` 1-bit phase indicator is the unification
   knob.** Per-rollout state `agent_goal_store_fired` is fed as a single
   input bit. Pre-store: bit=0, agent does novelty; post-store: bit=1, agent
   follows q_full. Same network learns both in parallel without dilution.
   AZ.s0/s1 both reproduced sr=1.00 / ms=5.8 / cov=0.50 within 200 updates.

### Bug fix (train.py)

`optimizer.load_state_dict()` was silently overriding CLI `--bc_lr` /
`--ppo lr` flags when resuming from a checkpoint, because Adam's saved
state pins per-group `lr`. Now we re-apply CLI lr after loading optimizer.

### Code changes for this phase

- `bc.py`: per-step `nav_weight` for trust_hop move labels.
- `ppo.py:RolloutBatch`: new `trust_hop_mask` field (BC mode only).
- `rollout.py`: writes `trust_hop_mask[b,t] = (~at_goal & trust_hop)`.
- `config.py`: `BCConfig.nav_weight: float = 1.0`.
- `train.py`: `--bc_nav_weight` flag; lr-override after optim load;
  `evaluate_union_coverage` plumbed (`num_trials=10`).
- `inspect_trajectories.py`: added `--no_goal` mode.
- `probe_alignment.py`: new diagnostic for policy/q_full/optimal alignment.
- `run_bc.sh`: env-var hooks for new knobs; wall bumped to 12h.

## Operational notes

- All runs are submitted via `run_bc.sh` with env-var overrides. Tags
  (RUN_TAG=A,B,...) are recorded only in the wandb run name `bc-<TAG>-
  seed<N>`; not in any standalone manifest. Reconstruction here was done
  by parsing `wandb-metadata.json` files in
  `/orcd/home/002/jackking/cls/wandb/run-2026042*-*/`.
- `run_bc.sh` SBATCH time: `0-02:00:00`. Scaled rollout (200 steps × 16
  envs × 4 epochs × 4 minibatches × 600 updates) does not fit; ~220
  updates is the practical ceiling at current settings.
- No saved checkpoints are systematically promoted — each tag's best
  eval is whatever survived in the slurm log before TIMEOUT.
