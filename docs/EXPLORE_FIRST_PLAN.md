# Explore-first training — plan

Written 2026-09-17. Experiment log: `docs/EXPERIMENTS_EXPLORE_FIRST.md`.
Branch/worktree: `explore-first` (cut from `main` at 77cf411, which contains
the merged `nav-tri-metric` line). Predecessors: `docs/DUAL_TRAINING.md`
(the one-model result and the ‖q‖ gate), `docs/EXPERIMENTS_NAV_TRI.md` §wave 3
(the refuted explore-first ordering), `docs/TASK_FAITHFUL_PLAN.md` (the task
this line trains on), `docs/CONTINUAL_CONTROLS_PLAN.md` (the regularizers
this line borrows).

> **Status: plan written, awaiting Jack's answers to §9 before code or jobs.**

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

## 1. Phase 1 — the explorer (exists; not trained here)

`/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navp2_p20_e_kcap_s42_21695408/navigate_u700.pt`

Recipe `p20_e_kcap`: `explore:700` on 20 arenas, hidden 1024, polar action
head, state-dependent spread, `LOG_KAPPA_MAX 2.5`, `input_hopfield_raw`,
encoder `w52_attract_fwhm/001_att0.5_seed=43` gain 100, shuffle regime
assignment, distractors in memory. The same channels and the same κ cap as
`d0_base`, so it forks into the exploit recipe with no architecture change
(`--load_checkpoint`, which drops Adam's moments by design — correct here,
the objective changes).

Chosen over `p20_e` (uncapped κ, `log_kappa_max 5.0`) so the cap is constant
across phases: the cap is exploit's single largest unlock (§2.5 of
DUAL_TRAINING) and changing it at the phase boundary would confound the
forgetting measurement with the orbit-trap effect the cap has on explore.

Its cost (700 updates × 20 envs × 64 = 896k explore trajectories) is
**recorded and not charged**: under the hypothesis it is the lifetime prior.
The sample-efficiency claim is about phase 2 alone. Both numbers go in the
table so a reader can charge it if they disagree.

**u0 baseline.** Before anything trains, the explorer is scored on the full
task eval scope (`EVAL_SCOPE=task`: nav + exploration + task-faithful) so
every later eval is a delta against it. Expected: coverage ≈ 0.63 at d=0 and
flat in distractors; first-visit `found_frac` high; revisit navigation at
chance (it has no gate, so a stored goal does not change its behaviour).

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

**Alternative (not default): the pre-stored exploit regime.** `exploit:N`
with the goal oracle-stored before every rollout, as in `d0_base`'s exploit
half. It has no search segment, so the policy never sees a goal-absent input
in phase 2 and the gate would have to come from the mechanism alone. It is the
cleaner "exploit only" reading and the harsher forgetting test. §9 Q1.

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

All arms fork `p20_e_kcap u700` into the §2 recipe. Names are launcher
variants under a new `xf_*` family (`x`plore-`f`irst).

| arm | mechanism | trains | what it tests |
|---|---|---|---|
| **E0 `xf_naive`** | none | everything | the unprotected control; the wave-3 prediction is a collapse |
| **E1 `xf_ewc_<λ>`** | online EWC toward the explorer's weights. Fisher estimated **once, at u0**, from the explorer's own explore-regime rollouts (one collection of the training shape, used for the Fisher only, never for a gradient step) | everything, penalised | a weight-space prior; λ is a trade-off knob, two log-spaced values then a third if the two bracket nothing |
| **E2 `xf_adp`** | **frozen explorer + residual exploit module**: rnn, direction head and spread heads frozen; a zero-initialised MLP on `[h_frozen, x]` adds to the direction logits (and optionally to `log κ`); value head retrained | adapter + value | zero forgetting by construction; the question becomes whether a residual on the explorer's own features can express the gate and q-following. A magnitude gate is *free* for an additive `W·q` term — small ‖q‖ perturbs the explorer's direction little, large ‖q‖ overrides it — which is the same mechanism §9 of DUAL_TRAINING found in the one model |
| **E2b `xf_adp_gru`** | as E2 with a small recurrent column (64) in the adapter | adapter + value | whether the d=10 tail needs memory the frozen trunk does not provide |
| **E3 `xf_kl_<β>`** | full fine-tune + **search-masked distillation**: `β · KL(π_explorer ‖ π)` on the steps where `explore_mask = 1` (before the store), against a frozen copy of the explorer | everything, penalised on search steps | behaviour-space prior placed exactly where explore is supposed to be used; no explore rollouts, but the explorer re-enters as a teacher on the task's own search steps |
| **B0 `task1r_k4_h1024`** | from scratch, novelty on before the store (the task line's rule) | everything | the matched-shape baseline for part 1 of the claim. The task line has this at h128 only; h1024 is the explorer's width, so it is the honest comparison |
| **B1 `xf_scratch_nonov`** | from scratch, `novelty=0, eps=0` | everything | what the prior buys: exploit-only reward with no explorer |

E2 and E3 are both "the explorer stays": E2 by freezing, E3 by teaching. E3 is
the arm closest to Jack's sentence — one network, nothing frozen, no explore
trials — and E2 is the arm that cannot fail on part 2, which makes it the
cleanest reading of part 1. Both are labelled for what they import.

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
- **P3 (E2).** Coverage identical to u0 (by construction); d=0 revisit
  criterion met; d=10 `mean_steps_all` worse than `d0_base`'s tail. If E2b
  closes that gap, the tail needs memory.
- **P4 (E3).** Holds coverage within 10% and meets the exploit criterion; the
  mechanism in the probe is a ‖q‖ gate on search steps (low `chase_q` before
  the store, high after).
- **P5 (E1).** A monotone trade-off in λ: high λ ≈ E2's numbers, low λ ≈ E0's,
  no λ dominating E3.
- **P6 (B1).** Slow or never: without novelty the from-scratch searcher rarely
  touches the goal in 200 steps, so the signal is sparse.

## 6. Implementation (what has to be built)

1. Launcher: `xf_*` family in `hopfield_nav/run_nav_p2.sh` — forks the
   explorer, sets the §2 shape, parses `_ewc<λ>`, `_adp`, `_adp_gru`,
   `_kl<β>` levers. `task1r_k4_h1024` already parses.
2. `NavAgent`: `--exploit_adapter {none,mlp,gru}` — a zero-initialised
   residual on the direction logits reading `[features.detach(), x]`, plus
   `--freeze_trunk` (rnn + direction head + spread heads) in
   `set_phase_freeze`. Checkpoint load with the new module absent is a
   fork, so it is created fresh at zero.
3. PPO hooks: `ppo_update(..., extra_loss=...)` taking a callable over
   `(agent, minibatch, new_dist)`; EWC penalty from
   `hopfield_nav/continual/regularize.OnlineEWC` (its Fisher estimator, run
   once at u0 on explore rollouts) and the masked KL against a frozen copy
   (`continual/distill._frozen_copy`, `_masked_kl` already exist for the
   polar distribution). Both are logged per update.
4. Accounting: nothing new — `cum_episodes` / `cum_env_steps` are in the
   checkpoints; the u0 eval is `eval_all` on the explorer.
5. Tests: adapter at init is the identity policy; frozen params have no grad
   after a step; EWC penalty is 0 at θ* and > 0 after a step; masked KL is 0
   on post-store steps and equals the unmasked KL when the mask is all ones.

## 7. Wave 1

| arm | job shape | updates | trajectories |
|---|---|---|---|
| explorer u0 eval | eval only | — | — |
| E0 `xf_naive` | 1 × 4 × 64, h1024 | 1000 | 256k |
| E1 `xf_ewc_lo`, `xf_ewc_hi` | same | 1000 | 256k each |
| E2 `xf_adp` | same | 1000 | 256k |
| E3 `xf_kl_lo`, `xf_kl_hi` | same | 1000 | 256k each |
| B0 `task1r_k4_h1024` | same, from scratch | 1000 | 256k |
| B1 `xf_scratch_nonov` | same, from scratch | 1000 | 256k |

Eight training jobs, seed 42; the winner and E0 get seeds 43/44 in wave 2.
1000 updates is 2× the task line's u500 crossing so the forgetting curve has
room after the exploit criterion is met. Timing: `task1r_k4_h128` ran ~10
s/update; h1024 is expected 15–20 s/update, so 1000 updates is 4–6 h —
submit on the 12 h partition with `CKPT_EVERY=25`, and a TIMEOUT is a normal
outcome (compare at the largest common checkpoint).

## 8. What is deliberately not in wave 1

- Re-training the explorer on the current recipe (Jack's "lifetime" prior is
  whatever explorer exists; the checkpoint's recipe is recorded).
- The three-arena fixed-goal setting: the redraw arena is the task line's
  current rule and the one where search is honest.
- Adapter capacity sweeps, EWC on the value head, distillation on post-store
  steps (that would teach the explorer's *non*-following after the store,
  which is the wrong prior).
- Lifting the κ cap for the explorer (`p23_kanneal`'s explore-safe anneal):
  exploit needs the cap; changing it at the boundary confounds the reading.

## 9. Questions for Jack (answers change what gets built)

1. **Phase-2 task.** Default: the water-maze task regime with novelty and ε
   off (§2) — the search segment is the task's own, not an explore trial.
   The alternative is the pre-stored exploit regime with no search at all
   (§2, last paragraph). Run the default only, or both?
2. **Which mechanisms count.** E2 freezes the explorer (a residual module
   learns exploit — nothing can be forgotten, but it is two modules). E3
   keeps one network and uses the explorer as a *teacher on search steps*
   (no explore rollouts, but explore knowledge re-enters through the KL).
   E1 is weight-space only. Are all three acceptable under "without
   interleaving explore trials", or should any be dropped?
3. **The explorer.** Default: `p20_e_kcap u700` (exists, κ cap 2.5, 20
   arenas). Alternative: train a fresh explorer on the current recipe first
   (~3 h) so phase 1 is recorded under this line's rules.
4. **Goal reward.** Default 2.0 (the task line's). Wave 4 of the task line is
   testing 5.0 for the post-store tie; if that lands before wave 1 launches,
   adopt it?
