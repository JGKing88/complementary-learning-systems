# Explore-first training — plan

Written 2026-09-17. Experiment log: `docs/EXPERIMENTS_EXPLORE_FIRST.md`.
Branch/worktree: `explore-first` (cut from `main` at 77cf411, which contains
the merged `nav-tri-metric` line). Predecessors: `docs/DUAL_TRAINING.md`
(the one-model result and the ‖q‖ gate), `docs/EXPERIMENTS_NAV_TRI.md` §wave 3
(the refuted explore-first ordering), `docs/TASK_FAITHFUL_PLAN.md` (the task
this line trains on), `docs/CONTINUAL_CONTROLS_PLAN.md` (the regularizers
this line borrows).

> **Status: §9 answered 2026-09-17, wave 1 in flight.** Decisions: the task
> regime only; the frozen-adapter arm dropped; the explorer retrained under
> this line (`xf_explorer`, job 22889945); goal reward stays at the
> baseline's 2.0. Jack: *"I would love if this worked without the continual
> learning algos, so it's a good arm but is not the central arm"* — so the
> **naive fork is the central arm**, EWC and the KL are supporting arms.
> Implementation is done and tested (§6). Log: `EXPERIMENTS_EXPLORE_FIRST.md`.

## 0. What this is

Jack, 2026-09-17: *"I'd like to try training explore before exploit again.
Explore is largely task agnostic as a behavior we would expect in an agent.
So the interpretation would be that an agent already knows how to explore
from a lifetime of being placed into environments, and it is just exploit
that needs to be learned in the water maze context. Hopefully we can then get
much better sample efficiency on learning exploit. The trick I think will be
to prevent catastrophic forgetting of explore behavior. It would be great if
you could do this without interleaving explore trials into exploit trials."*

So the claim under test has three parts, and each gets its own measurement:

1. **Exploit is cheap given explore.** A pretrained explorer finds the goal on
   most first visits, so the exploit signal is dense from update 0 and the
   only thing left to learn is "follow `q` when it is there". Measured as
   trajectories-to-criterion on the exploit metrics, against the from-scratch
   task-faithful line at the same shape.
2. **Explore survives exploit training.** Measured every 25 updates on the
   held-out explore evals and on first-visit search, and at the end by the
   continual protocol.
3. **Without explore trials.** Phase 2 contains no explore-regime rollout and
   no novelty reward anywhere. The only goal-absent inputs the policy sees are
   the search segments of the task itself (memory holds distractors only
   until the first touch), and those steps carry no explore shaping.

### 0.1 Why this is not wave 3 again

Wave 3 arm A (2026-08-13) warm-started from the explore specialist and
**annealed exploit rollouts in while interleaving**, full fine-tuning, no
protection: coverage held 0.31–0.37 for 350 updates and then fell to 0.068 in
one 50-update step when `empty_frac` reached 0.5. Arm C (explore block then
exploit block, no interleave) slid 0.351 → 0.223 through its exploit block.
Both were recorded as "only simultaneous exposure holds both", and the
mechanism as: *exploit installs persistent q-following; in an explore rollout
q points at distractor phantoms; the agent drives into a wall* (D1/D2).

What is different here, on purpose:

| | wave 3 A / C | this line |
|---|---|---|
| phase-2 data | explore + exploit regimes, novelty on in explore | the task regime only: search → store → navigate, novelty **off** |
| goal-absent inputs in phase 2 | dedicated explore rollouts | the search segment of every rollout |
| forgetting protection | none | the arms in §4 |
| what was measured | coverage at the end | coverage every 25 updates + search competence + continual protocol |

The wave-3 collapse is the *prediction* for the unprotected control (§4 E0).
If E0 does not collapse, the earlier reading was about novelty-vs-goal reward
interference rather than about ordering, and that is worth knowing too.

### 0.2 What the ‖q‖ gate says about this

`DUAL_TRAINING.md` §9: the one model's regime cue is a scalar gate on ‖q‖
(goal-present ‖q‖ ≈ 0.27, goal-absent 0.05–0.09, gate at 0.09–0.20), and
**the explore specialist has no gate** — `chase_q` is flat over a 10× ‖q‖
sweep; it ignores the readout entirely. So phase 2 has to build the gate from
one side only: it sees goal-present inputs after the store and goal-absent
inputs before it, both inside the same task, with reward only for the former.
Whether a gate can be learned from that (rather than unconditional
q-following, which is the corner trap) is the mechanistic question behind
part 2 of the claim.

## 1. Phase 1 — the explorer (retrained under this line)

Launcher variant **`xf_explorer`**, job 22889945, seed 42: the `p20_e_kcap`
recipe — `explore:700` on 20 arenas, hidden 1024, polar action head,
state-dependent spread, `LOG_KAPPA_MAX 2.5`, `input_hopfield_raw`, encoder
`w52_attract_fwhm/001_att0.5_seed=43` gain 100, shuffle regime assignment,
U[0,10] distractors in memory, novelty 0.3 / wall −0.1 / persistence +0.2 /
time −0.05, ε 0.1→0 over 200 updates, PPO 4×4 at 3e-4 — run under **today's
launcher defaults**. Jack asked for the retrain; the concrete reason it is
right: the Aug 31 `p20_e_kcap` checkpoint carries `input_hopfield_multistep
[1, 2, 3]` (74 input dims), while every run since 2026-09-06, `d0_base`
included, uses `"1"` (70 dims). A fork of the old explorer would have
inherited the old channel layout. Everything else in its config was checked
against `d0_base`'s and `task1r_k4_h128`'s and is identical
(`ent_coef 0.005`, `clip 0.15`, `explore_goals_off`, `rnn/relu`).

The κ cap is kept at 2.5 (not `p20_e`'s 5.0) so it is constant across
phases: the cap is exploit's single largest unlock (§2.5 of DUAL_TRAINING)
and changing it at the boundary would confound the forgetting measurement
with the orbit-trap effect the cap has on explore.

Its cost (700 updates × 20 envs × 64 = 896k explore trajectories, ~4 h) is
**recorded and not charged**: under the hypothesis it is the lifetime prior.
The sample-efficiency claim is about phase 2 alone. Both numbers go in the
table so a reader can charge it if they disagree. The checkpoint the arms
fork is the one at the coverage plateau of its own `expl` evals (u700
unless the series says the plateau came earlier and then eroded).

**u0 baseline.** Every fork now scores its parent **before its first
gradient step, on the fork's own held-out envs** (`[navigate_u0]` block,
`train_navigate` change in this line) — so every later eval of a run is a
delta against what that run started from, on the same envs, not against a
number measured elsewhere. Expected: coverage ≈ 0.6 at d=0 and flat in
distractors; first-visit `found_rate` high; revisit navigation at chance
(the explorer has no gate, so a stored goal does not change its behaviour).

## 2. Phase 2 — the task

The task-faithful regime (`TASK_FAITHFUL_PLAN.md` §0), at the shape the task
line currently favours, with the explore shaping removed:

```
SCHEDULE='task:1000,visits=4,novelty=0,eps=0'
ENVS_PER_WORLD=1  BATCH_ENVS=64  ENV_REPEATS=4  REDRAW_GOAL_PER_ROLLOUT=1
PPO 10 epochs x 8 minibatches, lr 1e-4, target_kl 0.1   (the SE optimizer)
LOG_KAPPA_MAX=2.5  GOAL_REWARD=2.0  wall -0.1 / persistence +0.2 / time -0.05
EVAL_SCOPE=task  EVAL_EVERY=25  CKPT_EVERY=25
```

- One arena, goal redrawn per visit sequence: the goal cannot be memorised,
  so search is honest every sequence (the task line's wave-3/4 finding).
- `visits=4`: visit 1 is search → store → navigate; visits 2–4 arrive with
  the memory visit 1 left. Three of four rollouts start goal-present. This is
  what makes the exploit signal dense; it is also the continual protocol's
  revisit trial exactly.
- `novelty=0, eps=0` on the stage: the search segment pays only the run-wide
  wall/persistence/time terms plus the eventual goal. No novelty, no ε
  actions. **Nothing in phase 2 rewards covering the arena.** (Persistence and
  wall are physical costs the explorer was trained with; they stay so the
  reward the explorer sees before the store is a subset of what it was trained
  on rather than a new shaping. Dropping them is a lever, not the default.)
- 256 trajectories per update (1 × 4 × 64), 800 serial steps per update.
  `cum_episodes` / `cum_env_steps` are in every checkpoint, so the cost axis
  is read from the run, not reconstructed.

**Goal reward 2.0, not 5.0.** The task line's wave 4 tests 5.0 under the
*other* rule (novelty kept on after the store, `_nv_c1_g5`), so its result
would not transfer to this rule; and the baseline this line is measured
against, `task1r_k4_h128`, is at 2.0. Matching the baseline keeps the
sample-efficiency comparison like-for-like.

**Not run (Jack, §9 Q1: default only): the pre-stored exploit regime.**
`exploit:N` with the goal oracle-stored before every rollout, as in
`d0_base`'s exploit half — no search segment, so the policy never sees a
goal-absent input in phase 2 and the gate would have to come from the
mechanism alone. The harsher forgetting test; available if E0 holds.

## 3. What "no forgetting" means, measured

Four readings, all against the u0 baseline of §1:

| reading | where | criterion |
|---|---|---|
| **held-out explore coverage** at d = 0 / 5 / 10 | every 25 updates, `EVAL_SCOPE=task` exploration evals | within 10% of the explorer's own, at every eval — a slide counts as forgetting, not only a cliff |
| **first-visit search** (`found_frac`, `steps_first`, `cov_first`) | every 25 updates, task evals | not below the explorer's u0 |
| **corner-trap signature** on search steps | `behavior_probe` at the final checkpoint, pre-store steps only | `chase_q` on search steps stays at the specialist's level (≈0.01–0.03), `edge_frac` does not rise |
| **continual protocol** | `agenthash` 5 envs × 100 iterations × 500 cap, `--stochastic_policy`, `--lock_store_after_goal` | primary (search) trials keep finding the goal; revisits ≥ 0.95; retention delta ≈ 0 |

And "exploit learned" is the task line's criterion so the two lines compare:
held-out **revisit success ≥ 0.95 at ≤ 20 steps** and **`follow_q` ≥ 0.80**.
`task1r_k4_h128` (from scratch, novelty on before the store) reached revisits
1.00 at 18.9 steps and `follow_q` 0.80 at **u500 = 128k trajectories**. The
sample-efficiency number is the update (and `cum_episodes`) at which each
arm first crosses that, with the eval-point rule from memory: report the
series, no directional claim from fewer than 4 points.

## 4. The arms — mechanisms against forgetting

All arms fork the `xf_explorer` checkpoint into the §2 recipe. Names are
launcher variants under a new `xf_*` family (`x`plore-`f`irst).

**The central arm is E0** — the plain fork, no algorithm. Jack: *"I would
love if this worked without the continual learning algos, so it's a good
arm but is not the central arm."* It gets the one lever that is not a
continual-learning method, the learning rate. E1 and E3 are the supporting
arms: if E0 forgets, they say whether the forgetting is stoppable and at
what cost to exploit; if E0 holds, they are the controls that show the
holding was not luck.

| arm | mechanism | trains | what it tests |
|---|---|---|---|
| **E0 `xf_naive`** *(central)* | none | everything | does the plain fork keep exploring? The wave-3 prediction is a collapse |
| **E0' `xf_naive_lr03`** *(central)* | none, lr 3e-5 | everything | the plain lever: does a 3× smaller step keep the explorer without any algorithm? |
| **E1 `xf_ewc_<λ>`** | online EWC toward the explorer's weights. Fisher estimated **once**, on the first update's rollouts before the first step, on their search steps (`explore_mask`) — the explorer, on the task, searching. No explore-regime collection anywhere | everything, penalised | a weight-space prior; λ is a trade-off knob, two log-spaced values (1e3, 1e4) then a third if the two bracket nothing |
| **E3 `xf_kl_<β>`** | full fine-tune + **search-masked distillation**: `β · KL(π_explorer ‖ π)` on the steps where `explore_mask = 1` (before the store), against a frozen copy of the explorer | everything, penalised on search steps | behaviour-space prior placed exactly where explore is supposed to be used; no explore rollouts, but the explorer re-enters as a teacher on the task's own search steps (β = 1, 10) |
| **B0 `task1r_k4_h1024`** | from scratch, novelty on before the store (the task line's rule) | everything | the matched-shape baseline for part 1 of the claim. The task line has this at h128 only; h1024 is the explorer's width, so it is the honest comparison |
| **B1 `xf_scratch_nonov`** | from scratch, `novelty=0, eps=0` | everything | what the prior buys: exploit-only reward with no explorer |

**Dropped (Jack, §9 Q2): E2 `xf_adp`**, the frozen explorer with a residual
exploit module. It cannot forget by construction, which made it the cleanest
reading of part 1 — and two modules, which is not the object under study.
The argument it rested on stays in the record because it is a prediction
about E0/E3's mechanism: an additive `W·q` term gets a magnitude gate for
free (small ‖q‖ perturbs the explorer's heading little, large ‖q‖ overrides
it), and that is the same mechanism §9 of DUAL_TRAINING found in the one
model. If E0 learns exploit without losing search, the probe should find
exactly that.

## 5. Pre-registered predictions

- **P1 (part 1).** Every `xf_*` arm reaches revisit ≥ 0.95 at ≤ 20 steps
  before u200 (≤ 51k trajectories), i.e. ≥ 2.5× fewer than B0's 128k. Basis:
  first-visit `found_frac` starts near the explorer's value (expected ≥ 0.9)
  instead of B0's 0.52–0.59, so goal-present rollouts with reward are the
  majority from u0. *Falsifier:* any `xf_*` arm slower than B0.
- **P2 (E0).** Coverage falls > 30% by u500. Prediction is a slide (arm C's
  shape), not arm A's cliff, because nothing in phase 2 switches regime
  fractions. *Falsifier:* E0 holds coverage within 10% — then ordering was
  never the problem and the wave-3 collapse was the novelty/goal-reward
  interference of the interleave.
- **P3 (E0').** lr 3e-5 slows both: forgetting *and* exploit. Prediction is
  that it does not change the ordering — coverage still falls > 30% by u500,
  later — because the drift that removes explore is the same drift that
  installs q-following. *Falsifier:* E0' holds coverage and still meets the
  criterion before u500 — then the whole thing was a step-size problem.
- **P4 (E3).** Holds coverage within 10% and meets the exploit criterion; the
  mechanism in the probe is a ‖q‖ gate on search steps (low `chase_q` before
  the store, high after).
- **P5 (E1).** A monotone trade-off in λ: high λ holds coverage and slows
  exploit, low λ ≈ E0's numbers, no λ dominating E3.
- **P6 (B1).** Slow or never: without novelty the from-scratch searcher rarely
  touches the goal in 200 steps, so the signal is sparse.

## 6. Implementation (built 2026-09-17, commit 39c6f5b and after)

1. **`hopfield_nav/training/prior.py` — `ExplorerPrior`.** Anchored to the
   weights as loaded from `--load_checkpoint`, before any step. EWC:
   `0.5·λ·Σ Fᵢ(θᵢ−θ*ᵢ)²`, F a diagonal *true* Fisher of the movement
   log-prob, estimated once on the first update's rollouts before the first
   gradient step, on their search steps only (`explore_mask × alive ×
   policy_action`, ≤ `--fisher_trajectories` rows). Value and store heads get
   zero importance (the movement log-prob has no gradient there), so the
   value is free to re-learn the new objective. KL: `coef ·
   KL(π_explorer ‖ π)` on the search steps of every rollout against a frozen
   copy, teacher run once per update over the pooled buffer.
2. **`policy/polar_head.py` — `vonmises_kl`, `polar_kl`.** torch registers
   no KL for VonMises and `PolarMove` is not a `Distribution`, so the
   continual suite's `kl_divergence` path cannot serve the polar head.
   Analytic `log I0(κ₂) − log I0(κ₁) + A(κ₁)(κ₁ − κ₂ cos(μ₁−μ₂))` with the
   scaled Bessels (κ = 148 stays finite), plus torch's Beta KL for speed.
   Checked against Monte Carlo at four (μ, κ) pairs.
3. **`updates/ppo.py` — `ppo_update(..., prior=None)`.** The terms join
   every gradient step's loss; reported as `prior_ewc` / `prior_kl`; the
   trainer adds `prior_drift` (RMS distance from the anchor). Refused
   without `--load_checkpoint` and under `--continue_from` (the anchor would
   be the resumed weights).
4. **`train_navigate` — the u0 eval.** A fork scores its parent before its
   first step on its own held-out envs (`[navigate_u0]`).
5. **Launcher.** `xf_explorer`, `xf_naive`, `xf_naive_lr03`, `xf_ewc_<λ>`,
   `xf_kl_<β>`, `xf_scratch_nonov`; `EWC_LAMBDA` / `PRIOR_KL_COEF` /
   `FISHER_TRAJECTORIES` pass-throughs; `DRY_RUN=1` prints the assembled
   command. Not built: the adapter (dropped).
6. **Tests.** `test_explorer_prior.py` (25: KL vs Monte Carlo, zero at the
   anchor, grows with drift, heads free, mask selects steps, teacher frozen,
   pooled indexing == direct forward, inside `ppo_update` a large λ ends
   nearer the anchor and a large β nearer the teacher) and
   `test_explore_first_smoke.py` (4, end to end: u0 block before u1, Fisher
   once, terms in the log, no-parent refused). Suite: 1,652 pass.

## 7. Wave 1 (submitted 2026-09-17)

| arm | job | shape | updates | trajectories |
|---|---|---|---|---|
| phase 1 `xf_explorer` | 22889945 | 20 × 64, h1024 | 700 | 896k (not charged) |
| B0 `task1r_k4_h1024` | 22891316 | 1 × 4 × 64, h1024, from scratch | 1000 | 256k |
| B1 `xf_scratch_nonov` | 22891317 | same, from scratch | 1000 | 256k |
| E0 `xf_naive` | after phase 1 | same, fork | 1000 | 256k |
| E0' `xf_naive_lr03` | after phase 1 | same | 1000 | 256k |
| E1 `xf_ewc_1e3`, `xf_ewc_1e4` | after phase 1 | same | 1000 | 256k each |
| E3 `xf_kl_1`, `xf_kl_10` | after phase 1 | same | 1000 | 256k each |

Eight phase-2 jobs, seed 42; the central arm and the best supporting arm
get seeds 43/44 in wave 2. 1000 updates is 2× the task line's u500 crossing
so the forgetting curve has room after the exploit criterion is met. Timing:
`task1r_k4_h128` ran ~8 s/update; h1024 is expected 15–20 s/update, so 1000
updates is 4–6 h on the 12 h `ou_bcs_normal` partition with
`CKPT_EVERY=25`; a TIMEOUT is a normal outcome (compare at the largest
common checkpoint). The λ and β values are first guesses on the scale the
continual suite found (λ = 1e4 partial, 1e5 frozen, under a BC loss); the
penalty and the PPO loss are both in the log so the ratio is visible, and
wave 2 moves them if the two values bracket nothing.

## 8. What is deliberately not in wave 1

- The pre-stored exploit regime (§2; Jack: default only).
- The frozen-explorer adapter (§4; dropped).
- The three-arena fixed-goal setting: the redraw arena is the task line's
  current rule and the one where search is honest.
- EWC on the value head; distillation on post-store steps (that would teach
  the explorer's *non*-following after the store, which is the wrong prior).
- Lifting the κ cap for the explorer (`p23_kanneal`'s explore-safe anneal):
  exploit needs the cap; changing it at the boundary confounds the reading.
- Dropping persistence/wall from the search segment (§2): a lever for wave 2
  if E0's search behaviour drifts in a way the shaping explains.

## 9. Questions for Jack — asked and answered 2026-09-17

1. **Phase-2 task.** Default (the task regime with novelty and ε off) only,
   or also the pre-stored exploit regime? — **Default only.**
2. **Which mechanisms count.** E2 freezes the explorer; E3 uses it as a
   teacher on search steps; E1 is weight-space only. — **Drop E2.** And:
   *"I would love if this worked without the continual learning algos, so
   it's a good arm but is not the central arm."* → E0 is the central arm.
3. **The explorer.** Fork the existing `p20_e_kcap u700`, or retrain? —
   **Retrain** (§1; the channel-layout drift made this the right call).
4. **Goal reward.** 2.0, or 5.0 if the task line's wave 4 lands first? —
   **"Sure"**; resolved to 2.0 because the 5.0 arm runs under a different
   reward rule and the baseline is at 2.0 (§2).
