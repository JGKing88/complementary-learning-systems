# Training one policy to explore AND exploit

**Standalone synthesis, written 2026-09-06.** Everything here is drawn from
`EXPERIMENTS_NAV_TRI.md` (phase 1) and `EXPERIMENTS_NAV_P2.md` (phase 2);
every claim carries its section. **New experiments continue to be logged in
`EXPERIMENTS_NAV_P2.md`** — this document is the plan and the map, not a lab
notebook, and it is rewritten rather than appended to.

The one-model claim is the whole point: a single policy that explores,
navigates, and discriminates while **inferring** which regime it is in. That is
why `input_goal_in_memory` is banned — handing the agent the regime bit deletes
the problem (`EXPERIMENTS_NAV_P2` §0.0).

**Reading order for someone new:** §1 (the chain) → §5 (dual failure modes) →
§7 (the metric panel) → §9 (waves). §2–§4 are reference.

**Published page:**
[bb0e31c3](https://claude.ai/code/artifact/bb0e31c3-4aca-4710-b57f-e71a227ff994)
— the same content, laid out. Re-publish it when this file changes.

---

## 0. The one-paragraph state of play

Both specialists are solved to a known ceiling. Exploit reaches **1.000 success
at 10.95 steps, 1.10× optimal** (`p10_pol_v1`, §9.8) and its residual failures
at ten distractors are mode A — following a readout that is wrong. Explore
reaches **swept 0.644 @200 steps, 0.911 @1000** (`p20_e`, §22.3.1) which is *at*
the billiard ceiling, and its residual failure is that the policy is a
**memoryless vector field** — it replays the same action at the same
(position, heading) regardless of history. Phase 1 produced a combined model at
coverage 0.315 / success 0.849 / 16.5 steps at d=10. What phase 2 added is that
the two specialists' *knob optima conflict on at least three axes* (§4.3), and
that the regime signal a combined model needs is **available but inverts on a
successful exploiter** (§6.3). Those two facts are the dual problem.

---

## 1. The chain, and every place it breaks

An exploit episode succeeds when four links hold in sequence; an explore episode
has its own two-link chain. Their fixes are mutually exclusive, so attributing a
failure to the wrong link wastes a training run.

```
EXPLOIT   memory  ->  readout  ->  policy  ->  execution
          is the      does the     does the    does following
          goal        recall       action      it actually
          separable   point at     follow q?   arrive?
          from the    the goal?
          distractors?

EXPLORE   motor   ->  strategy
          does the    does the path
          agent       cover new
          actually    ground?
          move?
```

The explore chain is short and the first link is the one that fails: **100% of
episodes at u25/u50 are wall-pinned** (§18.7), commanding 0.79 and realizing
0.09. Nothing about strategy is measurable until the motor link clears.

| link | diagnostic | fix domain |
|---|---|---|
| memory | `margin` = ⟨p_goal,x⟩ − max_i⟨p_dist_i,x⟩ vs its distance-matched baseline | encoder |
| readout | `q_accuracy` = cos(q, goal−pos) | encoder / projection |
| policy | `follow_q` = cos(a, q), **read against `align_true`** | training |
| execution | `clip_frac`, `straddled`, horizon | config |
| motor | commanded ‖a‖ vs realized ‖a‖ | reward shaping |
| strategy | `swept_coverage`, recurrence dip | objective / representation |

---

## 2. Exploit failure modes

### 2.1 The two headline modes — same cost, opposite cause

**This is the single most important distinction in the project** (phase-1
finding 20). They cost about the same (success 0.828 exploit-only vs 0.849
combined at d=10), so **a success rate cannot tell them apart**. Only
`follow_q_fail` and `q_accuracy_fail` together can, and they point in opposite
directions.

| at d=10, on **failed** trials | combined model | exploit-only |
|---|---|---|
| `follow_q_fail` — did it follow `q`? | **0.011** | **0.311** |
| `q_accuracy_fail` — was `q` worth following? | **0.437** | **0.179** |
| `final_dist_fail` | 9.4 | 5.6 |
| `fail_frac_at_edge` | 0.389 | 0.289 |

> **Mode A — trusting a broken readout.** `q_accuracy` low, `follow_q` high. The
> policy does its job; the signal is wrong. **Encoder-limited — no policy change
> fixes it.** The agent walks faithfully into a phantom and stops one
> distractor's distance out.

> **Mode B — ignoring a usable readout.** `q_accuracy` high, `follow_q` low. The
> signal is fine; the policy will not use it. **Regime-detection-limited — a
> policy problem.** The agent ends far away, against a wall 39% of the time:
> explore behaviour running inside a nav episode.

**Explore training converts A into B at roughly constant cost** (finding 21).
That is why the ~15% a combined model misses at d=10 is *not* the encoder's
fault and is the one lead a policy fix could take.

The sharpest mode-A evidence in the dataset is `p10_pol_v1` at d=10, where
`follow_q` (0.630) **exceeds** `q_accuracy` (0.450) and `align_true` collapses
to 0.257 as a direct result — more committed to the readout than the readout
deserves (§9.8).

### 2.2 Splitting mode A further — and the trap in doing so

The Hopfield is linear (§5.4), so the weight of stored pattern `i` under query
`x` is just ⟨p_i, x⟩ and `margin` is computable offline. Mode A splits into:

* **`readout_memory`** — margin well below the distance-matched baseline. A
  contaminated blend. Wants a better encoder or fewer stored patterns.
* **`readout_decode`** — margin on par, `q_accuracy` still poor. The right
  pattern won cleanly and the projection is wrong.

**Split on the margin's *deficit*, never on an absolute floor.** Measured across
four arms, the goal pattern wins the competition on **101/101 failures and
667/667 successes**. So "did a distractor win" separates nothing and would label
every failure a decode error. What separates them is the *level*: failures sit
at median margin ~0.60 against ~0.93 for successes at the same distance
(`EXPLOIT_DIAGNOSTIC` §3).

### 2.3 The symptom and motion taxonomies

Symptom, from `d(t)` alone on float positions, first rule that matches:
`straddled` → `approached_left` → `blocked` → `timeout_converging` →
`never_approached`. Motion, orthogonal to it: `pinned` → `committed` →
`looping` → `oscillating` → `meandering`.

**`straddled` is a near-goal *heading* diagnostic, not a step-length one.** The
policy's speed is bounded below by `min_action_norm`, so the agent cannot creep;
landing inside the capture ball needs `d² + L² − 2dL·cos φ ≤ R²`, and the
tolerated heading error `φ` shrinks as `d` grows. It has a sharp signature —
`d_min` piling up just above `R` — which makes it cheap to kill rather than
argue about. Predicted worse at `goal_radius` 0.5 than at 1.0.

**There are no interior obstacles.** `pinned` and `blocked` therefore always
mean *held against the perimeter*, which connects them to the explore-side
wall-pin basin rather than to scenery.

### 2.4 Three measurement requirements, each of which has already produced a wrong conclusion

1. **`align_true` is the baseline for `follow_q`, not zero.** When
   `q_accuracy ≈ 1` the relation `follow_q ≈ align_true` is *forced by geometry*
   and proves nothing. The informative regime is where they diverge — i.e. at
   high distractor counts. Reporting `follow_q` alone is how this project
   concluded a model "barely follows the readout" when it had a 100% success
   rate (§9.7).
2. **Matched baselines are mandatory.** Failed episodes are far from the goal
   *by construction*. Every cause statistic must be reported against the same
   statistic on successes **restricted to the same distance bin**, so zero means
   "behaved like a success at this distance".
3. **Distances on `pos_f`, never on the snapped cell.** `at_goal` is an L2 ball
   on the float position; mixing in `current_location` reintroduces the
   snap-square/L2-ball mismatch. `behavior_probe` gets this right for the cosine
   bins and wrong for the sigma/kappa bins — two ladders that disagree by up to
   ~0.7 cells.

### 2.5 What has worked on exploit

| intervention | effect | section |
|---|---|---|
| **`LOG_KAPPA_MAX` 5.0 → 2.5** | **the largest single unlock in the project**: success 0.375@u475 → 1.000@u125, ~4× in updates | §17.9 |
| bounded step + both `prev_action` channels | 10.16 → 6.25 steps at d=0, ~38% fewer at both distractor levels, higher success | §8.1 |
| polar action parameterization | closes the magnitude channel (1.234× → 1.018×); κ becomes usable state-dependently | §9.6 |
| curriculum | stability outright — never below 0.979 after breakthrough vs control's 0.490; costs a later start (u300 vs u150) | §9.9 |
| longer training | `follow_q` 0.558 (u300) → 0.819 (u900) → **0.911** (u2000). Most "the policy won't follow `q`" readings were budget readings | §9.8 |

**What did not work:** cheaper failure (`time_penalty` 0.02) gives 24× more
success at u50 — at 59 steps per goal against the control's 10. It is stumbling
onto targets inside the budget, not steering, and it then has the *worst*
stability of four arms. **Early goal discovery is not the convergence
bottleneck** (§9.9).

---

## 3. Explore failure modes

### 3.1 Mode 1 — the wall pin. Rare at convergence, universal early, and *paid for*.

| ckpt | coverage | realized speed | `clip_frac` | `edge_frac` | `straightness` | pinned episodes |
|---|---|---|---|---|---|---|
| u25 | 0.048 | **0.088** | **0.914** | 0.928 | **0.985** | **64/64 (100%)** |
| u50 | 0.049 | 0.105 | 0.906 | 0.926 | 0.981 | **64/64 (100%)** |
| u75 | 0.194 | 0.520 | 0.445 | 0.533 | 0.944 | 20/64 (31%) |
| u150 | 0.326 | 0.868 | 0.070 | 0.241 | 0.951 | **0/64 (0%)** |
| u700 | 0.386 | 0.963 | 0.031 | 0.126 | 0.958 | 0/64 |

**The un-pinning IS the learning curve** — pinned fraction 100% → 31% → 0%
across u50–u150, with coverage tracking it exactly. The honest budget is ~150
updates to escape the basin, ~100 more to plateau, then a long slow tail. The
thing that looks expensive is not the exploring (§18.7).

**Why it happens: the persistence bonus pays for it.** Every shaping term,
priced per step:

| ckpt | novelty | **persistence** | wall | time | predicted | logged `mean_r` |
|---|---|---|---|---|---|---|
| u50 **pinned** | +0.030 | **+0.196** | −0.093 | −0.050 | +0.084 | +0.074 |
| u150 free | +0.236 | +0.190 | −0.024 | −0.050 | +0.352 | +0.334 |
| u700 final | +0.292 | +0.192 | −0.013 | −0.050 | +0.421 | +0.416 |

The model reproduces logged reward across a 5.7× range with error ≤0.018. **At
the pin, persistence pays +0.196/step against `wall_penalty`'s −0.093 — a ratio
of 2.1.** The pin is a *rewarded* state, and the term rewarding it is the one
meant to encourage ballistic exploration. Cause: persistence scores
`cos(a_t, a_{t−1})` on the **commanded** action, and a pinned agent commands a
rock-steady heading while realizing 0.09 (§18.8).

**Do not raise `wall_penalty`.** It would need >0.24 merely to make the pin
unprofitable; at 0.24 it charges a *healthy* policy −0.031/step for legitimate
perimeter work; and the perimeter is 19% of the arena and **must** be visited.
We have already measured what an edge-avoiding explorer looks like — it is
`p20_e_kcap`, at 12% less coverage.

**The fix is `--persistence_realized`** (staged as `p21_pr`, **not launched**).
Score the bonus on realized displacement: a pinned agent's cosine collapses; an
unobstructed policy is untouched, because realized == commanded on 97% of steps
for a converged model.

### 3.2 Mode 2 — the systematic curl

`p20_e_kcap` learned a near-constant **0.121 rad/step** — one revolution every
52 steps, ~four circuits per episode. It traces an **annulus**, which explains
both halves of its deficit at once: it never reaches the perimeter
(`edge_frac` 0.061) *and* it dwells in the middle (0.445 of steps on old ground
against `p20_e`'s 0.310).

It is **82% systematic turning** (6.9° of 8.4° per step) against `p20_e`'s
0.3%. Locally this reads as clean straight running — the drift is below the 15°
that counts as straight — so nothing in the paths looks like a circle.

> **Both arms loop. The cost is not looping; it is not *leaving*.** 100% of
> episodes in both arms re-cross their own path, and the uncapped arm does it
> **more** often (12.5 events vs 8.9) while spending *less* of the episode on
> old ground. It cuts across and keeps going.

**The diagnostic is the recurrence curve** — `mean |p(t) − p(t+τ)|` against τ.
An orbit of period T shows a clear minimum at τ = T. `p20_e_kcap` is 14.5 cells
from where it was 30 steps ago and back within 4.06 cells 54 steps later: dip
depth **10.66** in 98/100 trajectories, against a curl-predicted period of 51.9.

**Every other instrument failed, and this is the reusable lesson:**

| instrument | why it failed |
|---|---|
| `signed_turn_mean` | an episode mean — a path that circles both ways cancels to ~0 |
| `straightness` | unsigned cosine — reads a 6.9°/step curl as straight, and reads a **wall-pinned agent as 0.985, the highest value in the project** |
| windowed \|Σdθ\| | contaminated by wall bounces; the billiard null scores 59% |
| revisit lag | blind to precession — the orbit returns to the same *region*, a different snapped cell |

Three separate times a *higher* `straightness` has accompanied *worse*
coverage. **For turning behaviour, report the signed / unsigned / windowed
triple together — no one of them is safe alone.**

**Fix that worked:** anneal `LOG_KAPPA_MAX` 2.5 → 5.0 over 400 of 700 updates
(`p23_kanneal`). The deterministic/sampled gap collapsed **20% → 3.3%**,
matching the uncapped arm. The closed orbit is gone (§26).

### 3.3 Mode 3 — the memoryless vector field

The policy is close to a fixed function of (position, heading). At a state
repeat — back within 0.5 cells with heading within 15°, ≥20 steps later — the
continuations stay ~8× closer than chance for 25 steps:

| divergence after k steps | `p20_e` repeat | random null |
|---|---|---|
| k=1 | 0.12 cells | 1.25 |
| k=10 | **1.28** | 10.22 |
| k=24 | **3.05** | 15.93 |

The RNN hidden state is definitely different on the second visit — it carries
the entire history, including having been there — **and it barely changes the
action** (§22.1).

**Nothing has broken this.** Replay ratio: control 0.125, κ-anneal 0.118, aux
visitation head 0.115. And `p24_aux` proved the information was *there* —
`aux_visited_loss` fell 0.632 → 0.367 predicting 8-direction visitation from
the trunk's own features — while the policy head, reading the identical vector,
ignored it, and coverage fell 13.7% with 3.3× the volatility (§27).

**Two things constrain how to read this:**

1. **Memorylessness cannot be the coverage cap.** A boustrophedon is a function
   of position alone (row parity comes off the y-coordinate) and scores ~0.9 at
   200 steps. So a near-optimal *memoryless* policy exists, and that is the
   basin PPO settles into (§29.3).
2. **But history would still pay.** The policy passes within a cell of its own
   earlier track on **~33% of steps**, and every one costs novelty. The
   information is in the state (occupancy content 0.044 on the control, up to
   0.116 on `p34`), and occupancy's absolute influence on the action is
   **pinned at 0.024–0.030 across five arms** regardless of intervention.

> **That is a credit-assignment problem, not an information problem.** Avoiding
> a revisit pays diffusely, as slightly more novelty over later steps;
> persistence pays immediately and certainly on the very next step.

### 3.4 Four explore-metric traps

1. **`mean_coverage` hides the speed axis.** Cell coverage says speed barely
   matters (E[T] flat at ~157 over a 6× speed range); **`swept_coverage`** — the
   union of `goal_radius` discs along the path, which *equals* P(the goal was
   findable) — says speed dominates (E[T] 127 → 88 → 70 from speed 1 to 3).
   §2.1/§18.2's "the speed cap is free" is **retracted**; it costs ~30% of
   expected discovery time (§19.2).
2. **`revisit_frac` ≡ 1 − `cells_per_step`.** Correlation −1.0000, algebraically:
   `coverage = (1 − revisit_frac)/2` exactly. It is coverage restated and carries
   no independent information. Use `analysis/nav_tri/proximity.py` —
   share of steps passing within `r` of ground held ≥`lag` steps earlier.
3. **`strategy_efficiency` must reference the *realized* magnitude.** Referenced
   to the commanded one it read **3.97** for a policy merely sitting at its speed
   cap (§9.1).
4. **`union_swept_coverage` is a mode-collapse detector, not a discriminator.**
   A collapsed policy scores `union == per_trial` exactly; healthy policies all
   saturate it.

---

## 4. The parameter map

### 4.1 Knobs with measured interior optima

| knob | optimum | effect | evidence |
|---|---|---|---|
| `LOG_KAPPA_MAX` | **2.5** early | the default e⁵=148 locks the policy into a wall; 2.5 is worth 4× in updates on exploit | §17.9 |
| `INIT_LOG_STD` | σ = **0.50** (Cartesian) | biggest explore lever — ~2× coverage. ε steps are masked out of the movement surrogate, so σ is the **only** channel for step magnitude | tri finding 1 |
| `GOAL_REWARD` | **2.0** | inert *within* a regime, decisive *between* them: both regimes share one pooled advantage normalization | tri finding 2 |
| `EPSILON_EXPLORE` | **0.1** | 0.4 is over-bought, by three independent mechanisms | tri finding 4 |
| `PERSISTENCE_BONUS` | 0.20, **realized** | carries `m²/(m²+σ²)` of signal — untestable at small step size, well-conditioned later | tri finding 6, §18.8 |
| `empty_frac` | 0.5 | moves along a coverage/steps frontier, not up it: 0.5→0.7 buys +28% coverage for −41% steps | tri finding 13 |
| `time_penalty` | 0.05 | 0.15 buys the best `mean_steps` (8.01) and the worst success (0.875) — a real trade, not a win | §8.1 |

### 4.2 Knobs that are no-ops, and why that matters

| knob | why it does nothing |
|---|---|
| `MOVE_ENT_COEF` under `FREEZE_LOG_STD=1` | Gaussian entropy depends on σ alone. **The whole v35 lineage swept a dead knob.** |
| `--epsilon_explore` on an exploit-only schedule | `exploit.py` hard-codes `epsilon=0.0`. Accepted, echoed, discarded — produced bit-identical numbers to the control. Also retracts an earlier claim that ε-annealing contributed to the u150–u250 swings: **there was never any ε in any exploit run in this phase.** |
| `hopfield_beta` | the tanh argument is ~1e-4, so tanh is inert and recall is power iteration; β is a no-op and `steps=1` is the only setting that retrieves |
| `WALL_PENALTY`, raising it | see §3.1 — it bids against +0.196/step of persistence income and taxes work the agent must do |
| **`input_hopfield_multistep` depths 2 and 3** | **DROPPED 2026-09-06** — see §4.2.1 |

### 4.2.1 Multi-step Hopfield — dropped

Jack, 2026-09-06: *"i don't see multi step hopfield mentioned and i think you
should drop that."* The evidence was already in and it agrees.

**What it was.** `--input_hopfield_multistep 1 2 3` projects the recall at
three iteration counts and feeds each as a 2-D input, on the theory that the
policy can read recall-*convergence dynamics*. Every run from P1 to P35 used
all three.

**Why it goes.** Three independent reasons, in increasing order of how much
they should have settled it earlier:

1. **It adds nothing measurable.** §7.7 ablated the depth channel directly.
   Group C *is* the depth-2 and depth-3 statistics, so `A∪B` is the
   depth-{1}-only classifier — and it scores **0.869 against 0.858** at ten
   distractors and t=8, **0.888 against 0.884** at t=64, and 0.763 against
   0.760 at t=1. **Depth {1} is never worse and is usually slightly better.**
2. **The premise is false.** §5.4: the recall does **not** converge, it
   *drifts*. The network is a linear associative memory in which iterating is
   power iteration toward the top eigenvector, and pooled direction quality
   degrades from 1.46% bad at one step to 12.27% at twelve. Depths 2 and 3 are
   not better-converged states, they are strictly **degraded** ones sampling
   the transient of a walk *away* from the answer. `steps=1` is the only
   setting that retrieves at all.
3. **It creates an intervention trap.** `q` reaches the policy through **four**
   channels, not one — the raw signal plus three `multistep_q` channels
   (`EXPLOIT_DIAGNOSTIC` §7). Intervening on the obvious one leaves the other
   three carrying the contaminated recall and contradicting it, and the run
   looks like it worked. Dropping to depth {1} removes a whole class of silent
   experimental error.

**It is measured, not assumed.** `INPUT_HOPFIELD_MULTISTEP` now defaults to
`"1"`, and wave 1 carries **`d1_ms3`**, which is `d0_base` with `"1 2 3"` and
nothing else moved. The honest qualification in §7.7 is the reason that arm
exists: the ablation removed *four summary statistics* of the depth channel,
while the policy receives `q²` and `q³` as raw 2-D vectors and could in
principle use them some other way. There is no evidence it should — but "no
evidence for" is not "evidence against", and one arm settles it.

**Cost of the change, stated because this launcher moves a default:** every run
up to and including P35 trained with `"1 2 3"`, so a phase-2 number and a
post-2026-09-06 number are not comparable on this axis. It also removes 4 of
the policy's 74 input dims.

### 4.3 The knobs whose optima CONFLICT between the regimes

**This table is the dual-training problem stated in one place.** Every row is a
knob where the exploit specialist and the explore specialist want different
things, on evidence.

| knob | exploit wants | explore wants | resolution | status |
|---|---|---|---|---|
| **`LOG_KAPPA_MAX`** | **cap at 2.5** — the unlock, 1.000@u125 vs 0.375@u475 | **lifted** — the cap costs 3.2% sampled coverage and traps the mean policy in a closed orbit (20% det/sampled gap) | **anneal 2.5 → 5.0**; `p23_kanneal` measured it explore-safe (gap 20% → 3.3%) | **explore-verified, exploit UNTESTED** — the single highest-value open test |
| **speed** | **frozen at 1.0** — 1.000 success, 10.95 steps, 1.10× optimal | **fast** — swept coverage is monotone increasing in speed | polar + learned `[0.5, 2.0]`; note §12 — step count tracks the *cap*, not navigation quality | untested jointly |
| **`persistence_bonus`** | risk: rewards ballistic commitment, which is right on approach and wrong at the doorstep (`straddled`) | +0.20 but it **pays for the wall pin** unless realized | `--persistence_realized` | `p21_pr` staged, **not launched** |
| **`INIT_LOG_STD`** | mostly sets clamp depth, not exploration — a 3.0× nominal σ range collapses to 1.33× effective angular noise | σ = 0.50 is a genuine 2× coverage lever | **polar parameterization decouples them** — magnitude and heading as separate distributions | resolved in principle (§9.6) |
| **`epsilon_explore`** | inert by construction | 0.1 | already regime-scoped | fine |

### 4.4 The two reward-arithmetic constraints

**`revisit_penalty` has a hard ceiling at the novelty reward.** Positive reward
needs coverage rate `c > rp/(0.3 + rp)`, and the agent *starts* pinned at
`c ≈ 0.10`. So every increment raises the bar the agent must clear **before**
reward turns positive, while making the pin itself more punishing. Measured
dose-response: 0.15 escaped slowly (and was the only arm ever to kill the
orbit), 0.25 stalled at u200, 0.40 never moved at all (§34.3).

**Shaping is mostly degenerate under pooled advantage normalization.** Only
*ratios* matter, which makes `revisit_penalty` largely redundant with novelty —
and makes `goal_reward` a between-regime knob rather than a within-regime one.

---

## 5. Dual-training failure modes

### 5.1 Established — measured in phase 1

| # | failure | signature | mitigation |
|---|---|---|---|
| **D1** | **Schedule collapse.** Explore-first collapses (coverage 0.068), exploit-first mirrors it (0.062), and *blocked* — which sees both regimes but never together — behaves like explore-first. | one metric near zero at the end of a run that looked fine mid-way | **Interleave within the same PPO update.** Only simultaneous exposure holds both. |
| **D2** | **The corner trap.** Exploit installs persistent `q`-following; in an explore rollout `q` points at distractors; the agent drives into a wall. `edge_frac` 0.82, `clip_frac` 0.65. **CONFIRMED §5.3.4** — collapsed episodes chase 4.1× harder than the rest. | sustained `chase_q`; and per-episode, the collapsed tail (§5.3.2) | per-update per-regime logging (§7); reduce exploit weight |
| **D3** | **Env-identity leak.** Regime assignment was *positional*, so at fixed `empty_frac` the same envs were always exploit and the policy could gate on env identity instead of on the recall signal. | a regime gap that vanishes on fresh envs | **`--regime_assignment shuffle`** (fixed; default stays `index` for back-compat) |
| **D4** | **Peak-then-degrade.** Every interleaved run peaks mid-run and degrades. | joint metric falling after a mid-run maximum | select with `joint_curve.py`, not the last checkpoint; **LR 1e-4** fixes it at a small cost in peak |
| **D5** | **Mode conversion.** Explore training does not add exploit failures — it converts mode A into mode B at roughly constant cost. | `follow_q_fail` collapsing while `q_accuracy_fail` stays healthy | this is the *tractable* failure; see §6 |

### 5.2 Predicted from the specialists — NOT yet measured

These follow from phase-2 results and are stated as predictions with their
falsifiers, so they can be checked rather than assumed.

**D6 — the κ schedule conflict is a *timing* conflict, not a value conflict.**
Exploit needs the cap and locks at ~u125; explore needs it lifted by ~u400.
`p23_kanneal` shows the anneal is explore-safe, and §26.3 flags explicitly that
it says nothing about whether exploit still converges at u125 under it.
*Prediction:* a fixed-update ramp will be either too early for exploit or too
late for explore, and the ramp should be **triggered on exploit's first
sustained success ≥0.85** rather than on an update index.
*Falsifier:* if a fixed 400-update ramp holds both, the trigger is unnecessary.

**D7 — the persistence bonus is a cross-regime coupling.** It is the largest
per-step term (+0.19, roughly two-thirds of total reward at the pin) and it is
constant across both regimes. In explore it pays for the wall pin; in exploit
it rewards ballistic commitment, which is correct on the approach and wrong at
the doorstep.
*Metric:* the `straddled` share of exploit failures, against `persistence_bonus`.
*Falsifier:* if `straddled` is flat in `persistence_bonus`, the coupling is
explore-only and `--persistence_realized` closes it.

**D8 — the regime detector inverts on a successful exploiter.** See §6.3. This
is the most specific and most consequential prediction P3 hands forward, and it
breaks the obvious implementation of a regime gate.

**D9 — asymmetric convergence times make the joint peak land where neither
specialist peaks.** Exploit locks at u125–u300; explore plateaus at ~u250 with a
tail to u700 and beyond. Combined with D4, the joint optimum is a compromise
checkpoint, not a coincidence of two maxima.
*Metric:* per-regime metric curves on one axis, plus `joint_curve.py`.

### 5.3 Interference, as a first-class number

For every metric: **interference = specialist ceiling − interleaved value.**
Report it per metric, every wave. Current reference lines:

| metric | specialist ceiling | best combined | interference |
|---|---|---|---|
| **swept @200 (raw)** | 0.626 (`p20_e`) | **0.680** (`w6_pers`) | **−8.6% — the combined model WINS** |
| **swept efficiency** | **0.965** (`p20_e`) | **0.873** (`w6_pers`) | **+9.5%** |
| cell coverage d=0 | 0.385 | 0.315 | 18% |
| `mean_steps` d=0 | 10.95 @ 1.10× optimal (`p10_pol_v1`) | 7.6 *(at higher speed — not comparable)* | confounded by stride |
| success d=10 | 0.958 | 0.849 | 11 pts |

#### 5.3.1 MEASURED — and the raw number says the opposite of the right one

Wave 0.2, job 22134356. Sampled, `n_dist=0`, 200 steps, `place=held_out`,
144 matched trials per model, reduced through the online evaluator's own
`SweptArea`:

| model | swept | sd | union | realized speed | billiard @ that speed | **swept_eff** |
|---|---|---|---|---|---|---|
| `w6_pers` u1950 — phase-1 combined | **0.680** | 0.093 | 1.000 | **1.417** | 0.779 | **0.873** |
| `p20_e` u700 — explore specialist | 0.626 | 0.046 | 0.997 | 0.961 | 0.649 | **0.965** |

*(`p20_e` reads 0.626 here against the 0.644 quoted in §0 and §22.3.1. Not a
correction — a different protocol: 6 envs × 24 trials **sampled** on
`place=held_out` here, 8 × 8 **deterministic** there. §5.3's table uses this
one because an interference number must be measured the same way on both
sides.)*

> **Raw swept says the combined model beats the explore specialist by 8.6%.
> Efficiency against a speed-matched billiard says it loses by 9.5%. The entire
> raw advantage is speed — 1.42 against 0.96.**

This is §19.2's warning arriving as a concrete near-miss: swept is monotone in
speed, so a model that sweeps more *because it moves faster* has not explored
better, and `w6_pers` moves faster only because phase 1 had no `[0.5, 1.0]`
action bound. Quoting the raw number would have recorded "interleaving costs the
explore half nothing" — the opposite of what happened.

**A second thing the ratio exposes.** `p20_e`'s *cell* `strategy_efficiency` is
**1.038** — it beats a billiard — while its *swept* efficiency is **0.965**, and
it does not. Those are consistent rather than contradictory: §22 found `p20_e`
slides along walls rather than bouncing off them (9× fewer sharp reversals than a
billiard), and cell-counting rewards visiting distinct cell *centres* while swept
only rewards covering distinct *area*. **The explore specialist's edge over a
billiard is in where it lands, not in what it sweeps.**

`w6_pers` also carries **twice the trial-to-trial spread** (sd 0.093 vs 0.046).

**Caveats, both load-bearing.** These are cross-phase — different encoders
(`ur_loss2_repel_low` vs w52) and different action bounds. They could not be
rolled on shared trajectories at all (`explore_traj` refuses: "does not share a
world"), which is precisely why the billiard ratio exists. `d0_base` supersedes
this with a same-encoder, same-bound number.

#### 5.3.2 The distractor cost is a 15% TAIL, not a shift — and it is D2

The d=10 numbers looked like a modest degradation and are not. Per-trial swept
on the same 144 episodes:

| `w6_pers` | mean | sd | p5 | **p50** | p95 | frac < 0.35 |
|---|---|---|---|---|---|---|
| d = 0 | 0.671 | 0.092 | 0.471 | **0.688** | 0.778 | **0.000** |
| d = 10 | 0.608 | **0.199** | **0.163** | **0.668** | 0.820 | **0.146** |

**The median barely moves — 0.688 → 0.668, −3%.** The mean falls only because
**14.6% of episodes collapse outright**, and the p95 actually *rises*. A mean
over this mixture reports "distractors cost the explore half 9%", which is not
what happens to any individual episode: 85% are unaffected and one in seven is
destroyed.

That is the fourth time in this project a mean over a mixture has told the
opposite of the truth (§5.1's eight-env mean, §5.2.1's two-draw fluke, §7.7's
pooled `b2`, and now this).

**What the collapsed episodes do**, against the 123 that did not:

| | collapsed (21) | rest (123) |
|---|---|---|
| swept | 0.193 | 0.678 |
| realized speed | **0.702** | 1.522 |
| share of steps < 0.3 cells | **0.496** | 0.058 |
| **edge occupancy** | **0.858** | 0.300 |
| path length | **139.8** | 302.9 |
| span traversed | 17.9 | 19.0 |

**They ride the wall.** 86% of steps on the perimeter ring — against a uniform
occupancy of 0.19 — with half of all steps barely moving, and half the path
length of a healthy episode. They are not trapped in a corner from the start:
`span` is 17.9 of a 20-cell arena, so the agent crosses the arena, reaches a
boundary, and then stays on it.

**And there are zero such episodes at d = 0.** The distractors cause it.

This is phase-1 finding 12 — **D2, the corner trap** — measured directly for the
first time, and the mechanism reads exactly as that finding predicted.
Distractors in this codebase are **memory-only**: `sample_distractors` draws
encoded patterns from grid positions *outside* the test env, so a phantom recall
points at something that is literally not in the arena. A policy carrying
exploit's persistent `q`-following drives at it, arrives at the boundary, and
has nowhere further to go.

**Consequence for wave 1, and it is why §7's instrumentation was built first.**
`pin_frac` and `edge_frac`, split by regime and logged every update, are exactly
the instrument for this population — and D2's prediction on record is that
`chase_q` rises *before* `edge_frac` does. That is now testable. What this
section adds to the prediction is **where to look**: not at the mean of any of
them, but at the tail.

**The control settles it: this is INTERFERENCE, not the task.** `p20_e`, same
protocol, same 144 trials, same reduction:

| | mean | sd | p5 | p50 | p95 | **frac < 0.35** |
|---|---|---|---|---|---|---|
| `p20_e` d = 0 | 0.625 | 0.046 | 0.545 | 0.632 | 0.687 | **0.000** |
| `p20_e` d = 10 | 0.631 | 0.041 | **0.562** | 0.638 | 0.686 | **0.000** |
| `w6_pers` d = 0 | 0.671 | 0.092 | 0.471 | 0.688 | 0.778 | **0.000** |
| `w6_pers` d = 10 | 0.608 | 0.199 | **0.163** | 0.668 | 0.820 | **0.146** |

**The explore specialist is completely flat in distractors** — its mean is
*marginally higher* at ten (0.631 against 0.625), its spread *narrower* (0.041
against 0.046), its p5 *higher* (0.562 against 0.545), and it has **not one
collapsed episode at either level**. That is the behavioural counterpart of the
`chase_q ≈ 0.000` both explore arms report, and it reproduces `p5_e`'s phase-1
result that coverage at ten distractors equals coverage at zero.

So the 15% tail is not something the task does to any policy. **It is what
training on exploit does to the explore half**, and it is D2 measured cleanly
for the first time:

> A specialist that has never learned to follow `q` cannot be lured by a
> phantom. Interleaving installs the following, and the following is what
> drives the agent into the wall on one episode in seven.

**This sharpens what wave 1 has to show.** The question is no longer whether the
corner trap exists but whether it can be *held at zero* while exploit still
converges. `pin_frac` and `edge_frac` on the **explore** rollouts are the live
readout, and the target is the specialist's number, not an improvement on
`w6_pers`.

**A caveat on the size of the tail.** `w6_pers` is a phase-1 model with no
`[0.5, 1.0]` action bound, running at 1.40. A faster policy reaches a wall
sooner and has more of its episode left to spend on it, so 14.6% is this
model's number and not a constant of interleaving. `d0_base` measures it at
matched speed.

#### 5.3.3 Deterministically it is worse still — and that is the deployed regime

All four conditions, job 22134356:

| model | eval | n_dist | swept | sd | speed | billiard | **swept_eff** |
|---|---|---|---|---|---|---|---|
| `w6_pers` | sampled | 0 | 0.680 | 0.093 | 1.417 | 0.779 | 0.873 |
| `w6_pers` | sampled | 10 | 0.616 | 0.201 | 1.402 | 0.763 | 0.807 |
| `w6_pers` | **det** | 0 | 0.623 | 0.130 | 1.303 | 0.729 | 0.855 |
| `w6_pers` | **det** | **10** | **0.556** | **0.216** | 1.236 | 0.711 | **0.781** |
| `p20_e` | sampled | 0 | 0.626 | 0.046 | 0.961 | 0.649 | 0.965 |
| `p20_e` | sampled | 10 | 0.632 | 0.041 | 0.962 | 0.649 | 0.975 |
| `p20_e` | **det** | 0 | 0.631 | 0.065 | 0.957 | 0.646 | 0.977 |
| `p20_e` | **det** | **10** | **0.637** | **0.050** | 0.963 | 0.645 | **0.988** |

**Under deterministic evaluation the combined model is worse than the
specialist on the RAW number too** — 0.556 against 0.637 at ten distractors —
so the speed advantage that made the sampled comparison look favourable is gone.
The efficiency gap widens from 9.5% to **21%**.

**This is the deployment-relevant comparison, not the charitable one.** §24
records why: exploit deploys deterministically — that is where §17.10's 1.013
beeline comes from — and the whole reason `p20_e_kcap` mattered was that a
deterministic deficit is invisible during sampled training. The same asymmetry
applies here, in the same direction.

**And `p20_e` is monotonically *better* under every hardening.** Its best number
of all four conditions is deterministic at ten distractors (0.988): more
distractors, less noise, higher efficiency. Its spread never exceeds 0.065. The
explore specialist is not merely unharmed by distractors — it is completely
insensitive to them.

Whereas `w6_pers` degrades on every axis at once: raw swept 0.680 → 0.556 across
the sampled/d=0 to det/d=10 corners, efficiency 0.873 → 0.781, and spread
**0.093 → 0.216**. The spread is the tell throughout: every one of its numbers
is a mixture.

#### 5.3.4 CONFIRMED on `chase_q` — and there are TWO collapse modes, not one

§5.3.2 *inferred* the corner-trap mechanism from edge occupancy. `explore_traj`
records `chase_q` = mean `cos(a, q)` per trial, which measures it directly.
Collapsed episodes (swept < 0.35) against the rest, same runs:

| run | n collapsed | **chase_q** collapsed | **chase_q** rest | edge collapsed | edge rest |
|---|---|---|---|---|---|
| `p20_e` samp d=0 | **0** | — | 0.000 | — | 0.126 |
| `p20_e` samp d=10 | **0** | — | 0.021 | — | 0.127 |
| `p20_e` det d=0 | 1 | **0.000** | 0.000 | 0.940 | 0.129 |
| `p20_e` det d=10 | **0** | — | 0.021 | — | 0.129 |
| `w6_pers` samp d=0 | **0** | — | 0.000 | — | 0.255 |
| `w6_pers` det d=0 | 8 | **0.000** | 0.000 | 0.655 | 0.226 |
| `w6_pers` samp d=10 | 21 | **0.275** | 0.120 | 0.841 | 0.325 |
| `w6_pers` det d=10 | 25 | **0.453** | 0.110 | 0.928 | 0.308 |

**The mechanism is confirmed.** At ten distractors the collapsed episodes chase
the recall **4.1× harder** than the rest deterministically (0.453 against 0.110)
and 2.3× sampled. And the chain is measurable end to end across all 144 trials:

    corr(chase_q, edge_frac) = +0.408
    corr(edge_frac, swept)   = −0.792
    corr(chase_q, swept)     = −0.315

Chasing raises edge occupancy, and edge occupancy is what destroys the sweep.

**But there are TWO collapse modes and only one is D2.** Look at the rows with
`chase_q` **exactly 0.000**:

* **`w6_pers` det d=0 — 8 collapsed, no chasing at all.** At zero distractors
  there is no phantom to chase, yet 5.6% of episodes still collapse. That is
  §18.6's **mode-1 wall pin**, the motor failure, and it appears here because
  *deterministic* evaluation traps a policy that its own sampling noise would
  free — the same asymmetry §23 measured on `p20_e_kcap`. Note it is absent
  from the sampled run at d=0 (0 collapsed).
* **`p20_e` det d=0 — 1 collapsed, `chase_q` 0.000**, `edge` 0.940, speed 0.10.
  One episode in 144, the specialist's own rare wall pin, and §18.6 reports
  exactly that rate.

So the two failure modes that both read as "a collapsed episode" are separable,
and `chase_q` is what separates them:

> **collapse WITHOUT chasing = the wall pin** (mode 1). Present in both models,
> rare, and induced by *deterministic* evaluation rather than by distractors.
>
> **collapse WITH chasing = the corner trap** (D2). Only in the combined model,
> only with distractors, and it is the 15%.

That distinction matters for wave 1 because the two want different fixes:
`--persistence_realized` addresses the first (§3.1) and does nothing for the
second, while the second is a regime-detection failure and is what §6 is about.
Reading `pin_frac` without `chase_q` beside it would conflate them — which is
the same shape of error as reading `follow_q` without `align_true`.

---

## 6. Is the agent exploring or exploiting? — independent of the rollout's regime

This is the question that decides whether mode B is fixable, and P3 (§7)
answers most of it.

### 6.1 What the agent COULD know — the ideal-observer bound

Three targets, all classified from statistics that are **functions of what the
policy already sees**, cross-validated over held-out envs, with a
label-permutation control at 0.452–0.549 and a constant-feature control at
exactly 0.500.

| target | question | AUC at 10 distractors |
|---|---|---|
| **Q_ep** | is anything from this env in memory at all? | **0.87 @ t=1, 0.99 @ t=4, 0.996 @ t=64** |
| **Q_trust** | is *this* recall's direction reliable right now? | **0.76 @ t=1, 0.88 @ t=8, 0.884 @ t=64** |
| Q_step | is the current recall the goal? | ≡ Q_ep to three decimals — **retired** |

**The base rate alone almost settles it.** With the goal in memory at ten
distractors, `q` points within **30° on 96.2% of steps** and 15° on 88.3%,
median error 5.3°. A policy that ignores `q` is not being cautious about an
unverifiable signal; it is declining a signal that is right 24 times in 25.

And the goal-absent control came out perfect: a recall that locked onto a
foreign pattern is **directionally indistinguishable from noise** (measured
0.320 / 0.165 / 0.080 against the uniform-2D predictions 1/3, 1/6, 0.083).

> **Mode B is a learning problem, not an information problem.** Three
> independent numbers say so and none depends on the others.

### 6.2 Which cue carries it

| cue | statistic | Q_ep t=1 | t=64 | note |
|---|---|---|---|---|
| **D** | `d1_chart` — recall's residual from the env's pattern subspace | **1.000** | **1.000** | needs a per-env fit |
| **D'** | **`chart_frac` = ‖q‖ / ‖recall − x‖** | — | **0.974 / 0.988** at d=10 | **one scalar, NO fit required** |
| A | `a6_q_max` — running max of ‖q‖ | 0.887 | **0.963** | the robust cue |
| A | `a1_qnorm` — ‖q‖ itself | 0.887 | 0.881 | what the policy has today |
| B | `b2_spread` — allocentric spread | 0.500 | 0.587 pooled | **1.000 on unclipped episodes** |

Three things follow:

1. **The discrimination is ‖q‖, and extra steps help by sampling *more cells*,
   not by generating motion evidence.** `still`, which visits one cell,
   saturates at 0.945 and never crosses 0.95; every probe that moves at all
   crosses within 2–3 steps.
2. **`b2` is a perfect regime detector — until the agent touches a wall.** It
   goes **1.000 → 0.611** the moment an episode meets a clip, because `b2` needs
   *net displacement*: `ĝ = Σd + q` is constant when the goal is stored and
   drifts when it is not. **The arena clip attacks the cue by suppressing
   motion, not by corrupting the channel** — so no input channel can fix it, and
   behaviour can (§7.10).
3. **`chart_frac` is the one genuinely missing input channel.** ‖q‖ is the 2-D
   projection of a 1024-dim recall; 1022 dimensions are thrown away. The single
   scalar "how much of this recall does the local chart explain" recovers them
   at **+0.276 AUC over ‖q‖** on the P2 encoder, needs no env-specific basis
   (`W` is already computed every step), and **beats the fitted 64-dim version**
   (§7.7.2).

### 6.3 The inversion — and the design constraint it imposes

**The single most actionable thing P3 found.** The probe-fitted classifier
applied to a *successful exploiter*'s own trajectories does not merely lose
power. It **inverts**:

| target | frozen classifier on `p4_x` (exploit specialist), d=10 | t=1 | t=8 | t=64 |
|---|---|---|---|---|
| Q_ep | frozen | 0.857 | 0.574 | **0.208** |
| Q_ep | **refit** | 0.841 | 0.985 | **0.998** |
| Q_trust | frozen | 0.743 | 0.575 | **0.114** |
| Q_trust | **refit** | 0.728 | 0.917 | **0.983** |

**Mechanism:** ‖q‖ scales with distance to the goal — median 0.041 at one cell
against 0.393 beyond twelve — so **‖q‖ next to the goal is smaller than the
goal-absent value of 0.086**. An agent that walks to the goal spends its episode
in the one region where the regime cue is not merely weak but *reversed*.

The explore specialist is the control that makes this airtight: its mean
distance to the goal stays at 10.1–11.6 for the whole episode, not one of its
1024 rows is masked because it never arrives, and the frozen classifier does
**not** invert on it (0.904 at t=64).

> **Design constraint: a gate or auxiliary head trained on exploration rollouts
> will invert on a successful exploiter.** Any regime signal must be built from
> statistics that *survive the approach to the goal* — the running maximum
> `a6_q_max` and running s.d. `a5_q_std` do, because they remember the large ‖q‖
> from early in the episode; the current `a1_qnorm` does not.

**Two more consequences, both counterintuitive and both load-bearing:**

**The information is all there — the agent is already collecting it, by
accident.** Refitted, the exploiter reaches Q_ep 0.998 and Q_trust **0.983** —
*ten points better than a billiard*. Why: the best probing behaviour is
`along_q` (the only probe family to reach AUC 0.95 at all, at t≈9, against
*never* for everything else), and an exploit policy executes `along_q` as a side
effect of exploiting.

**So do NOT design a separate verify-then-exploit phase.** The behaviour that
gathers the most evidence about whether the memory is real *is* the behaviour
that acts on the memory. The explore specialist, which moves but not along `q`,
collects **less** information than the exploiter (Q_trust refit 0.838) and less
than a billiard (0.884).

### 6.4 The behavioural panel — what the agent actually does

| quantity | reads | current values |
|---|---|---|
| `follow_q` (exploit rollouts) | is it exploiting? | 0.911 @ d=0 → 0.630 @ d=10 (`p10_pol_v1`) |
| `chase_q` (explore rollouts) | is it exploiting when it shouldn't? | **≈0.000** in both explore arms — the explore-side distractor problem is solved |
| **`regime_gap` = `follow_q` − `chase_q`** | **direct measure of regime discrimination** | **not yet computed — propose adding** |
| near-goal κ gradient | does it hedge on final approach? | grows with training: 1.03× (u300) → 1.24× (u900) → **1.65×** (u2000) |
| `follow_q` vs distractor count | does it withdraw trust as the readout degrades? | 0.558 → 0.427 as `q_accuracy` 0.989 → 0.711 |

The last two are the same finding from two sides: **"be more uncertain where the
readout is bad" and "rely on the readout less where it is bad" are both valid
responses to the same information, and this policy does both** — κ near the goal
and `follow_q` against distractor count. §9.1 looked only for the first, which
is why §9.3 read as a flat negative.

---

## 7. The metric panel

Three tiers. Tier 1 says whether the run is good; tier 2 says why; tier 3 stops
you fooling yourself. **Every tier-1 metric has at least one tier-3 guard, and
each guard exists because it has already caught a wrong conclusion.**

### Tier 1 — headline, scored at every eval

| regime | metric | target | ceiling |
|---|---|---|---|
| explore | **`swept_coverage` @200** | ≥ 0.58 (90% of specialist) | 0.644 |
| explore | `union_swept_coverage` | > per-trial (else mode collapse) | — |
| exploit | `success_rate` @ d=0 / 5 / 10 | 1.00 / ≥0.95 / ≥0.90 | 1.000 / 0.995 / 0.958 |
| exploit | `mean_steps` @ d=0, **with `mean_speed` beside it** | ≤ 12 at speed ≈1 | 10.95 |
| both | **interference** = specialist − interleaved, per metric | ≤ 10% | — |

### Tier 2 — mechanism, scored at every eval and on the final checkpoint

| reads | metric | healthy | pathological |
|---|---|---|---|
| exploit mode A vs B | **`follow_q_fail` × `q_accuracy_fail`** | both moderate | A: follow high, acc low · B: **follow ≈0, acc ≥0.4** |
| readout following | `follow_q`, **always with `align_true`** | diverge at high d | equal → uninformative |
| memory vs decode | `margin` deficit vs distance-matched baseline | ≈0 | ≤0.6 vs 0.93 |
| corner trap | **`chase_q`** | ≈0 | sustained elevation (NOT a lead over `edge_frac` — §7.1) |
| regime discrimination | **`regime_gap`** = `follow_q` − `chase_q` | large | → 0 |
| motor link | **commanded ‖a‖ vs realized ‖a‖** | ratio ≈1 | ratio ≫1 = pinned |
| wall pin | `clip_frac` × realized speed | <0.1, ≈1 | **>0.5, <0.5** |
| **which collapse mode** | **`pin_frac` × `chase_q`** (§5.3.4) | — | chase ≈0 → motor wall pin · chase ≫0 → **corner trap** |
| orbiting | **recurrence dip depth / period / IQR** | no post-rise dip | depth >3, tight IQR |
| retracing | `proximity_revisit` (**not** `revisit_frac`) | — | ~0.33 now |
| memory use | replay divergence ratio at k=10 | > 0.3 | **0.115–0.125 now** |
| map influence | occupancy **absolute** = share × `state_influence` | > 0.05 | **0.024–0.030, immovable** |

### Tier 3 — guards

| guard | catches |
|---|---|
| `align_true` printed beside every `follow_q` | the geometric identity at `q_accuracy ≈ 1` |
| `chase_q` printed beside every `pin_frac` | conflating the two collapse modes (§5.3.4) — they want opposite fixes |
| distance-matched baselines on every failure statistic | failures being far from the goal by construction |
| `pos_f`, never the snapped cell | the snap-square / L2-ball mismatch (~0.7 cells) |
| realized magnitude in `strategy_efficiency` | the 3.97 artifact |
| signed **and** unsigned **and** windowed turning together | three separate wrong claims about circling |
| `swept_coverage` not `mean_coverage` as the headline | the speed axis |
| ≥8 distractor draws before any separability number | the 23.3% two-draw fluke that misdirected a whole section |
| ≥4 eval points before any directional claim | evals swing 30+ points |
| sampled, not deterministic, evaluation for explore | the κ-cap gap read 14% deterministic, 3.2% sampled |
| both training-eval and probe success quoted as a bracket | 0.958 vs 0.901 on the same checkpoint |
| held-out split, not `recorded` | every §9.6–9.8 number is on the run's own validation set |

### The instrumentation — **BUILT 2026-09-06**

`hopfield_nav/rollout/diagnostics.py`, wired through `RolloutBatch.diag` and
logged by `train_navigate` **every update, split explore/exploit** as
`train/expl/*` and `train/expt/*`, plus `train/regime_gap`. Continuous movement
only. 24 tests in `test_rollout_diagnostics.py`.

Evals run every 25–50 updates and do not split by regime, so the corner trap
had only ever been seen after the fact — and D2's prediction, that `chase_q`
rises *before* `edge_frac`, was untestable. It is now a time series with an
onset. The cost is one dot product and four norms per step.

Emitted per regime: `cos_aq`, `cos_aq_frac`, `q_mag`, `edge_frac`, `clip_frac`,
`cmd_mag`, `realized_mag`, `pin_frac`, `steps`.

Three decisions worth knowing:

1. **`follow_q` and `chase_q` are one statistic.** `cos(a, q)` is emitted once
   as `cos_aq`; which name it takes is a property of the *rollout's regime*,
   not of the statistic. The trainer labels it. Their difference is
   `regime_gap`.
2. **ε and auto-nav steps are excluded from `cos_aq`.** A step the policy did
   not choose says nothing about whether the *policy* follows `q`, and
   including them makes a policy that ignores the recall read as partly
   following it.
3. **`pin_frac` is per-row, and `clip_frac` is not enough on its own.** A
   policy parked past `max_action_norm` reads `clip_frac` 1.000 with no wall
   involved (the probe's own docstring warns about this). `pin_frac` requires
   *both* `clip_frac > 0.5` and realized speed `< 0.5`, which is clamp-immune,
   and it is per-row because a rollout with half its rows pinned and one with
   every row half-pinned have identical pooled `clip_frac` — and only the first
   is §18.7's basin.

**A guard found by running it, not by a unit test.** On a first live rollout the
explore side came back `q_mag` **0.000** and `cos_aq_frac` **0.000**: with no
distractors the goal-absent memory is *empty*, so `q = 0` exactly and `chase_q`
is **undefined, not zero** — §7.5's degenerate condition. Ungated, `regime_gap`
would silently have reported exploit's own `cos_aq` as a discrimination.
`regime_gap` is now emitted only when *both* regimes have a usable recall on at
least half their steps.

#### 7.1 What this instrument can and cannot test — D2's lead/lag is NOT one

D2's prediction on record is that **`chase_q` rises before `edge_frac` does**,
and building the per-update logger was justified partly by making that testable.
Having run it: **it is not testable at this resolution, and that is a property
of the quantity, not of the run.**

Both statistics are **episode means over the same 200-step rollout**, and the
causal chain D2 describes — follow a phantom, arrive at the boundary, stay
there — happens *within* an episode. A cause and its effect that both complete
inside one update land in the same logged row, so a per-update cross-correlation
puts them at lag 0 by construction. Testing the lead needs *within-episode*
timing, which this instrument does not produce.

A fixed-threshold onset test is worse than useless here, and worth recording
because it is the mistake this project keeps making in new clothes: `edge_frac`
starts at **0.445** because the early policy is wall-pinned by default (§18.7:
100% of episodes pinned at u25/u50), so "first `edge_frac` > 0.25" fires at
u = 0 for reasons that have nothing to do with chasing. **A statistic compared
against zero when its baseline is not zero** — the same shape as reading
`follow_q` without `align_true`.

**What the lagged test does say**, on the post-pin window (after the last update
with `pin_frac` > 0.10), against a shuffle null that fixes the marginals and
destroys only the timing:

| arm | window | peak r | at lag | null 95th pct | reading |
|---|---|---|---|---|---|
| `d0_base` | u14–74 | **+0.666** | **0** | 0.359 | strongly coupled, **simultaneous** |
| `d1_kanneal` | u48–107 | +0.255 | 0 | 0.399 | **below the null — no relationship** |

So chasing and edge occupancy co-move strongly in the baseline and not
detectably in the κ-anneal arm. That is *suggestive* and no more: the two
windows cover different update ranges and the arms are at different stages, so
this is not a matched comparison and must not be read as one.

**What the instrument IS good for**, and what it should be judged on: a
*sustained* regime — `chase_q` elevated across many updates — and its onset
relative to a coverage collapse. That is the "corner trap as a time series"
use, and it needs a long run, which is what wave 1 is.

---

## 8. Logic gates

Read these as: *observe X → conclude Y → do Z*, never as a knob list.

**Gate 1 — is the motor link clear?** If pinned-episode fraction > 0.2 after
u150, nothing downstream is interpretable. → `--persistence_realized`. Do **not**
touch `wall_penalty`.

**Gate 2 — which exploit mode?** Read `follow_q_fail` × `q_accuracy_fail` on
failures only.
- follow high, accuracy low → **mode A, encoder-limited.** Stop tuning the
  policy. Check `margin` deficit to split memory from decode.
- follow ≈0, accuracy ≥0.4 → **mode B, policy-limited.** Proceed to gate 3.
- both healthy and it still fails → **execution.** Check `straddled`, `clip_frac`,
  horizon.

**Gate 3 — is mode B an information problem?** It is not (§6.1), and the check
is cheap: refit the P3 feature set on the agent's own rollouts. Refit AUC ≥0.95
→ the information is present and the failure is credit assignment or policy
search. Refit AUC <0.8 → something has changed about the setup and P3 needs
re-running.

**Gate 4 — is the regime signal inverting?** Compare frozen vs refit AUC on the
agent's own trajectories. **A frozen classifier that inverts is a *success*
signal** — it means the agent is spending its episode next to the goal. Judge on
refit; never deploy a gate fitted on explore rollouts.

**Gate 5 — is explore orbiting or just wandering?** Recurrence dip depth >3 with
tight `period_iqr` → orbit → anneal `LOG_KAPPA_MAX`. No post-rise dip → not an
orbit, and the ceiling is elsewhere. **Trust the aggregate curve, not the
per-trajectory count** — `p20_e` shows 91/100 "orbiting" per trajectory while
its aggregate correctly shows none.

**Gate 6 — did an intervention actually change the mechanism?** Coverage is not
the test; every oracle arm moved coverage without moving the mechanism. The
tests are the **replay divergence ratio** (memory use) and **occupancy's
absolute influence** (map use). Use share × `state_influence`, never share
alone — `p32_headdrop`'s share fell 0.275 → 0.071 purely because its denominator
grew.

**Gate 7 — is a number safe to quote?** ≥4 eval points, ≥8 distractor draws,
held-out split, sampled eval for explore, and a stated ratio-vs-null. Run-to-run
noise on the state-probe ratios is ~25% on an identical checkpoint.

---

## 9. Waves

Ordered by information per GPU-hour. **Nothing here is launched.**

### Wave 0 — instrument, no training

1. **Per-update per-regime diagnostics** (§7). Prerequisite for reading any
   interleaved run.
2. **Score the existing combined model on `swept_coverage`.** Every combined
   number to date is `mean_coverage`. This is a probe pass, minutes on CPU, and
   it fills the blank row in §5.3.
3. **Refit the P3 feature set on the combined model's own rollouts**, both
   regimes. Gives `regime_gap` and the frozen/refit pair with no training.

### Wave 1 — resolve the knob conflicts (§4.3) · **BUILT AND LAUNCHED**

All four arms are `interleave:1200,empty_frac=0.5`, `regime_assignment shuffle`,
`EVAL_SCOPE=navexpl`, w52 encoder, polar, `LOG_KAPPA_MAX=2.5`, speed `[0.5,1.0]`.
Each differs from `d0_base` by exactly one knob.

| arm | change | tests | falsifier |
|---|---|---|---|
| **`d0_base`** | nothing — **the P6 baseline that never ran** | the control for the other three, and the first interleaved run in phase 2 | — |
| `d1_kanneal` | `LOG_KAPPA_MAX` 2.5→5.0 over 400 | **D6** — the highest-value open test. Explore-verified (§26), exploit unknown (§26.3 says so outright) | exploit fails to lock by u300 → trigger the ramp on exploit's first sustained success instead of on an update index |
| `d1_persr` | `--persistence_realized` | **D7** and gate 1 together | pin clears but coverage *falls* → the bonus was doing real work at the walls; try a smaller realized bonus |
| `d1_ms3` | keeps `input_hopfield_multistep 1 2 3` | §4.2.1 — makes the multistep drop measured rather than assumed | `d1_ms3` beats `d0_base` → depths 2/3 carry something the §7.7 summaries missed, and the drop is reverted |

`p21_pr` (explore-only, `persistence_realized`, `explore:300`) runs alongside as
the clean single-regime control for D7 — `p20_e`'s own eval series is its
control, so it needs no re-run of the baseline.

**Speed is deliberately NOT an arm here.** §12 showed step count tracks the
speed *cap* rather than navigation quality, so a `[0.5, 2.0]` arm would move
`mean_steps` for a reason that is not about interleaving. It belongs in a wave
of its own, scored on swept coverage.

#### 9.1 LIVE — provisional, and D6's worry may have the sign backwards

**Two arms running, `d0_base` at u100 and `d1_kanneal` at u150. Nothing here is
a conclusion — it is 4 and 6 eval points, two of them matched, and this
document's own guard is ≥4 eval points *and* a matched comparison before a
directional claim.** Recorded now because the direction is unexpected.

| u | `d0_base` succ@10 / steps@10 / swept@0 | `d1_kanneal` succ@10 / steps@10 / swept@0 |
|---|---|---|
| 25 | 0.583 / 59.1 / 0.171 | 0.583 / 59.1 / 0.171 |
| 50 | 0.927 / 56.3 / 0.196 | 0.667 / 63.1 / 0.077 |
| **75** | **0.844 / 54.0** / 0.171 | **0.969 / 36.6** / 0.112 |
| **100** | **0.917 / 50.9** / 0.091 | **0.990 / 30.6** / 0.218 |
| 125 | — | 0.979 / 26.8 / 0.223 |
| 150 | — | 0.979 / 29.8 / 0.169 |

**At u25 the two arms are bit-identical** (0.583 / 59.1 / 0.171 / 0.152), which
is the check that they are actually matched: same seed, one knob.

**The exploit half is where the difference is, and it is large.** `steps@10`
falls 59.1 → 50.9 on the baseline over 75 updates (−14%) and 59.1 → 26.8 on the
anneal arm over 100 (−55%). At the two matched points the anneal arm is **32%
and 40% faster**.

**The explore half is not readable.** Both oscillate between 0.08 and 0.24 and
the sign flips between the matched points (u75 favours the baseline, u100 the
anneal arm). Nothing to say yet.

**Why the direction is a surprise, and the reading that would explain it.** D6
was framed as a risk that lifting the cap would cost exploit the unlock §17.9
measured. The data so far points the other way, and §9.8.1 is why: on a
converging exploit arm **κ grows with training** — ≈38 at u300, ≈93 at u900,
i.e. `log κ` ≈ 3.6 then 4.5. `LOG_KAPPA_MAX = 2.5` caps κ at **12.2**. So
holding 2.5 fixed does not merely protect the early policy, it **caps the
converged one**, and the anneal releases a ceiling the policy is pressing
against.

Both facts are consistent: §17.9 measured that the *default* 5.0 is too loose
**early**, and §9.8.1 that a converged exploiter wants to be sharper than 2.5
allows. Wanting the cap early and gone later is exactly what an anneal is, and
D6's proposed refinement — trigger the ramp on exploit's first sustained
success rather than on an update index — becomes more attractive rather than
less if this holds.

**What would overturn it:** the baseline catching up by u400–600 (it is only
slower, not stuck), or the anneal arm's explore half degrading once the ramp
completes at u400. Both are inside the run.

> **UPDATE, u125 — the first falsifier is already firing.** `d0_base` reached
> **1.000 / 1.000 success at 34.4 steps**, up from 50.9 at u100, so the matched
> gap has gone **32% (u75) → 40% (u100) → 22% (u125)** and the baseline now has
> the *better* success rate (1.000 against 0.979). The anneal arm is still
> ahead on steps and got there sooner, but "the baseline catches up" is the
> live hypothesis, not a hypothetical one. This is precisely the behaviour
> finding 16 warns about — the exploit eval swings enormously at a fixed seed
> and **no exploit conclusion is safe before ~500 updates.** Read §9.1 as a
> record of what the series looked like early, not as a result.

##### 9.1.1 WHICH LINK moved — a null on `follow_q` with a 40% effect on steps

The per-regime diagnostics answer something the eval metrics cannot: the arms
differ by 40% on `mean_steps`, so *which link of §1's chain* is different?
Matched on update index over the overlapping range, late window u85–u128, mean
± sd over ~44 logged updates each:

| | `d0_base` | `d1_kanneal` | Δ |
|---|---|---|---|
| **exploit `follow_q`** | 0.462 ± 0.08 | 0.500 ± 0.06 | **+0.038** |
| **`regime_gap`** | 0.434 ± 0.12 | 0.437 ± 0.11 | **+0.003** |
| exploit `edge_frac` | 0.189 ± 0.03 | 0.153 ± 0.02 | −0.036 |
| explore `chase_q` | 0.028 ± 0.10 | 0.063 ± 0.07 | +0.034 |
| explore `edge_frac` | 0.488 ± 0.07 | 0.444 ± 0.10 | −0.044 |
| explore `pin_frac` | 0.024 ± 0.03 | 0.038 ± 0.08 | +0.014 |

**Every difference is inside one standard deviation, and `regime_gap` is
identical to three decimals — while `mean_steps@10` differs by 40%.**

That is a null on the mediator with a large effect on the outcome, and it says
what the κ anneal is *not* doing:

> The anneal arm does not consult the readout more (`follow_q` +0.038 on a
> ±0.08 spread) and does not discriminate the regimes better (`regime_gap`
> +0.003). It navigates faster because it can **point more precisely**.

Which is exactly what the knob governs. κ is a directional-precision parameter;
it has no term that would make a policy trust a recall more. §9.8.1's
measurement — a converged exploiter running at κ ≈ 93 against a 2.5 cap's
ceiling of 12.2 — predicts a precision effect and nothing else, and that is what
shows up. **The mechanism and the knob agree, which is the check that a
40%-on-one-metric result most needs.**

Without this instrument the steps difference would have been attributable to
anything — better following, better regime detection, a different exploration
schedule. It is none of them.

##### 9.1.2 And the interference is visible live, in `edge_frac`

`explore edge_frac` sits at **0.444–0.488 in both interleaved arms**. Uniform
occupancy of the perimeter ring is **0.19**, and the explore specialist runs at
**0.241 at u150** and **0.126 converged.**

So both interleaved arms spend roughly **twice** as much of their explore
rollouts on the perimeter as the specialist does at a comparable stage. That is
D2's signature as a live time series rather than a post-hoc autopsy — the thing
§7's instrumentation was built for — and it is present in the baseline as much
as in the anneal arm, so it is a property of *interleaving*, not of the κ knob.

It is also the number wave 1 should ultimately be judged on for the explore
half, alongside the collapsed-tail fraction of §5.3.2: **the target is the
specialist's 0.126, and neither arm is near it yet.**

##### 9.1.3 D2 caught live in wave 1 — and the d=0/d=10 dissociation is exact

Both u125 checkpoints rolled against `p20_e` on **matched trajectories** (same
w52 encoder, so `explore_traj` puts all three in one process on identical envs,
starts and memory contents — the comparison wave 0.2 could not have). Sampled,
144 trials, threshold = ½ × billiard at each model's own speed:

| | n_dist | swept | speed | **swept_eff** | **frac collapsed** | **chase in tail** | chase in body |
|---|---|---|---|---|---|---|---|
| `d0_base` u125 | 0 | 0.375 | 0.702 | 0.692 | 0.083 | **0.000** | 0.000 |
| `d1_kanneal` u125 | 0 | 0.460 | 0.746 | 0.811 | 0.028 | **0.000** | 0.000 |
| `p20_e` u700 | 0 | 0.636 | 0.962 | **0.982** | **0.000** | — | 0.000 |
| `d0_base` u125 | **10** | 0.422 | 0.711 | 0.770 | **0.090** | **0.487** | −0.018 |
| `d1_kanneal` u125 | **10** | 0.452 | 0.729 | 0.805 | **0.083** | **0.522** | 0.040 |
| `p20_e` u700 | **10** | 0.639 | 0.963 | **0.990** | **0.000** | — | 0.014 |

**The dissociation §5.3.4 predicted is exact.** The same arms, the same
checkpoints, the same reduction:

* **at d = 0** they collapse a little (2.8–8.3%) and the tail chases at
  **exactly 0.000** — early-training motor failure, not the corner trap;
* **at d = 10** they collapse about as often (8.3–9.0%) and the tail chases at
  **0.487–0.522** against a body chasing at −0.018 to 0.040 — a **12–25×
  separation**;
* **`p20_e` has zero collapsed episodes at either level**, and its body chase is
  0.000 / 0.014.

So D2 is not a phase-1 artefact and it is not something the phase-2 config
avoided. It is present in the very first interleaved run this phase has, at
u125, in both arms, at essentially the same rate — **and the κ anneal does not
touch it (0.090 vs 0.083)**, which is what should be expected: κ is a
directional-precision knob and this is a regime-detection failure.

**The threshold that made this readable, and the one that hid it.** A first
pass used an absolute cut of 0.35, which flagged **37.5%** of `d0_base`'s d = 0
episodes — every one with `chase_q` exactly 0.000. That arm runs at realized
speed 0.702, where a billiard sweeps only 0.542, so 0.35 was 65% of *chance*:
the cut was labelling a slow policy as a broken one. Against the billiard at
each model's own speed the same data reads 8.3%, and the real signal at d = 10
stops being buried in it.

That is the third instance in this document of the same error class — a
statistic compared against a constant when its baseline moves (`follow_q`
without `align_true`, `edge_frac` against zero when uniform is 0.19, and now
this). **The §5.3.2 headline survives the change**: `w6_pers` at d = 10 reads
0.160 under the speed-relative rule against 0.146 absolute, with the chase
separation intact (0.263 vs 0.119).

**Where this leaves wave 1.** The explore half of both arms is well short of the
specialist on every axis — swept 0.42–0.45 against 0.64, efficiency 0.77–0.81
against 0.99, and a 9% chasing tail against zero. These are u125 checkpoints of
a 1200-update run, so the honest statement is that **the corner trap is open in
both arms at u125 and neither κ knob addresses it**; whether training closes it
is what the rest of the run answers.

### Wave 2 — the regime signal

Gated on wave 1, because §27 is a warning: the aux head proved the information
was in the trunk and the policy ignored it anyway.

| arm | change | rationale |
|---|---|---|
| `d2_chart` | **`chart_frac` as an input channel** | the single largest measured information gap: +0.276 AUC over ‖q‖ at d=10, one scalar, no per-env fit. Needs `‖recall − x‖`, a rollout change |
| `d2_qstats` | `a6_q_max` and `a5_q_std` as input channels | the two statistics that **survive the approach to the goal** (§6.3) — the inversion-proof regime cue |

**Caveat carried forward:** `feedback_hopfield_nav_bc_inputs` froze the input
set for the bc-AQ line. Adding a channel is a decision, not an obvious win, and
§27 is direct evidence that availability ≠ use.

### Wave 3 — credit assignment

The explore-side wave already written up as `EXPERIMENTS_NAV_P2` §36
(`p35_alias`, `p36_rpanneal`, `p37_aliasaux`) belongs here, because its target —
making memory's value **immediate** rather than diffuse — is the same problem
mode B has on the exploit side. Run it after wave 1 so the wall pin is not
confounding it.

---

## 10. What would falsify this plan

* **If wave 0 shows `regime_gap` is already large** on the combined model, mode
  B is not a regime-detection failure and §6 is aimed at the wrong thing.
* **If the κ anneal breaks exploit**, the conflict in §4.3 is a genuine
  incompatibility rather than a scheduling problem, and the one-model claim
  needs either a longer schedule or a different action parameterization.
* **If `chart_frac` as an input changes nothing**, that is the third time
  (after `p24_aux` and `p25_visin`) that supplying information has failed to
  change behaviour, and the conclusion is that the bottleneck is how the policy
  *reads* its state — an architectural problem, not an input one.
* **If occupancy's absolute influence stays at 0.024–0.030** through wave 3,
  that constant is structural, and the next question is whether PPO can move it
  at all or whether this needs a different objective.
