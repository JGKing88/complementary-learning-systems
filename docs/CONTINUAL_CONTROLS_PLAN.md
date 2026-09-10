# A continual-learning control suite for the Hopfield-nav claim

Written 2026-08-30. **Revised 2026-08-30 (v2)** after review — see §0.3 for what
changed.

> **Status: in flight.** This is no longer plan-only. What exists:
>
> | piece | state |
> |---|---|
> | Wave 0 (T0.1 / T0.3 / T0.4) | run; T0.3 = 1.000, T0.4 landed, T0.1 **inconclusive** and re-running at 64× the budget |
> | The nine methods (§4) | implemented and tested in `hopfield_nav/continual/` |
> | Wave 1 (Tier-1 sweeps + ER + online EWC) | on the cluster |
> | Wave 2 (CLEAR, DER++, SI, LwF) | implemented, launcher ready |
> | Wave 3 (HNET, multi-head, XdG) | **done** — jobs 21653228 (144) + 21654989 (40), all OK |
> | §5.2 in-context zero-update control | implemented and on the cluster |
> | §5.1 meta-pretraining | not started |
> | Metrics + results page | implemented, generated from the data |
>
> **Wave 3 was missed once.** It is not in the sequencing table below because
> the suite was called complete when Wave 2's corrections finished rather than
> when this document was finished, and nobody noticed until a reader asked
> where HNET had gone. See the log's 2026-08-31 entries for both halves of that
> — the omission, and what was built in response.
>
> It also changed the answer. **XdG + SI reaches 0.739 retained**, against
> Experience Replay's 0.579 — so the best classic result on this task is a
> parameter-isolation method, not a replay one, and the family that was left
> out was the family that wins. It needs an oracle task id, which
> `task_identifiability` shows is worth a great deal here (0.43 against a
> chance of 0.20), so it is reported as an upper bound and the results page
> splits its frontier table rather than ranking the two groups together.
>
> The running record — including three bugs this uncovered in shared code, and
> one wrong verdict caught before it shipped — is
> [CONTINUAL_CONTROLS_LOG.md](CONTINUAL_CONTROLS_LOG.md). Read that for what
> actually happened; this document is what was intended.

Companion reading: `docs/CODEBASE_MAP.md` §"final_plotting", `analysis/continual/`,
`hopfield_nav/training/rnn_sequential.py`.

---

## 0. What this document is for

`analysis/continual/` currently supports one comparison:

| run | protocol | env0 | env1 | env2 | env3 | env4 (current) |
|---|---|---|---|---|---|---|
| `agenthash_w_oracle` | frozen policy, Hopfield store only | **0.98** | 1.00 | 1.00 | 1.00 | 1.00 |
| `baseline_regular_final` (30 seeds) | pretrain → sequential BC finetune | 0.05 | 0.18 | 0.22 | 0.28 | 0.99 |
| `baseline_regular_200steps` (30 seeds) | same, 200-step rollouts | 0.08 | 0.18 | 0.24 | 0.28 | 1.00 |
| `20x20_pretrained_10_full_iters` (10 seeds) | pretrain → finetune, 100 upd/env | 0.06 | 0.26 | 0.35 | 0.29 | 0.99 |

(`reached` averaged over the last 20 % of the final block; 5 envs × 200 updates,
`size=20`, `movement_mode=continuous`. Extracted from
`analysis/continual/histories/`.)

The RNN control reaches **0.99 on the env it is currently training on** and
**~0.19 on everything else**, with a clean recency gradient. Plasticity is
fine; retention is gone. The Hopfield agent is at ~1.0 everywhere.

That is a real result, but it is one control — a *naive* one — and a referee's
first question writes itself: *"you compared against SGD with no
continual-learning method at all; a 2019 replay buffer would close this."*
This document is the answer.

### 0.1 The strategic point, stated up front

**A replay buffer with an unbounded budget will probably close most of this
gap, and the plan is built on the assumption that it will.** One trajectory is
`200 steps × 60 floats` ≈ 48 KB; the entire 5-env × 200-update stream is
**~192 MB**. Storing every observation the agent has ever seen is free at this
scale, and Experience Replay over a perfect buffer *is* joint training with a
delay. If it retains, that is the expected result, not a failure.

So the deliverable is **not** "no classic method retains." It is a
**cost-matched frontier**. Every method is scored on five axes at once:

| axis | what it means | Hopfield agent's value |
|---|---|---|
| **retention** | mean `reached` on envs `0..i-1` at end of block `i` | ~1.0 |
| **gradient steps at deployment** | weight updates needed to acquire a new env | **0** |
| **episodes to acquire** | rollouts needed before the new env is solved | **1** (one store) |
| **stored bytes** | replay data + per-task params + importance matrices | fixed `O(D²)` matrix, no data |
| **task-ID requirement** | must the method be told which env it is in | **none** |

A method that matches retention but needs 200 gradient steps, 200 episodes and
a growing archive of raw trajectories has not refuted the claim — it has
*located* it.

### 0.2 The one result that would genuinely hurt

If a **pretrained recurrent policy adapts to a new env in-context, with no
weight updates at all** (§5.2), the claim that a separate associative memory is
*needed* weakens sharply. Confirmed in scope; it is the only control that
competes on the Hopfield agent's own terms.

### 0.3 What changed in v2

Six decisions came back from review. All are now folded in; §10 records them
rather than asking them.

1. **`input_prev_action` is ON.** W8 resolved.
2. **N=5 headline, N=20 scaling panel.**
3. **Movement mode stays `continuous`.** v1 recommended `discrete`. Checked:
   every headline `agenthash` run is `continuous`, so `discrete` would compare
   two different tasks unless the Hopfield side is re-run too. `discrete` *is*
   the easier setting and easier-for-the-control is the right direction — but
   not at the price of alignment. Kept as a robustness check only. See §3.3.
4. **`batch_envs=1` is a deliberate regime, not a defect.** v1 was wrong about
   this twice; §3.1 W1 now states the correction and turns it into a
   sensitivity measurement instead of a fix.
5. **The continual-RL framing is cut.** v1 leaned on the continual-RL
   literature. This is supervised learning — cross-entropy on oracle actions,
   no reward, no value head, no policy gradient. The CRL taxonomy and the
   Continual World numbers do not transfer and are gone. See §1.3 for the two
   ways this genuinely does differ from a supervised CL benchmark, and §4.1 for
   CLEAR restated by mechanism rather than by pedigree.
6. **The suite is cut from ~20 methods to 9.** §4.6 lists every drop and why.
   §5.1 and §5.2 (meta-pretraining and in-context) are promoted to first-class
   deliverables rather than stretch goals.

---

## 1. What the setting actually is

### 1.1 The task

`GridEnv` (`hopfield_nav/world/env.py:278`) is an empty `size × size` arena
whose four boundary walls carry a **random ±1 barcode**, `(4, size)`, drawn
from the env's seed. The observation is a foveal ray-cast: `observation_size=60`
rays fanned around the heading, each returning the ±1 code of the wall segment
it hits. The goal is a random cell and is **never observed** — only the BFS
oracle sees it.

So env `i` differs from env `j` in two ways:

1. **a different random re-encoding of position** (different barcode), and
2. **a different target function** (different goal).

### 1.2 Scenario: supervised, domain-incremental, recurrent

The learning algorithm is **plain supervised learning** — masked cross-entropy
(discrete) or Gaussian log-likelihood (continuous) on the oracle's action,
`updates/bc_rnn.py`. No reward enters the loss; there is no value head and no
policy gradient. The right literature is therefore **supervised continual
learning**, and within it:

- **Domain-incremental.** Shared output space, shifting input distribution,
  task identity not given at test. Item (1) above makes this structurally a
  permuted-MNIST-style problem — the input space is randomly relabelled per
  task. Item (2) makes it strictly harder, because in permuted MNIST the label
  function is constant and here it is not.
- **Task identity is inferable in principle.** 60 rays against an 80-bit
  barcode is highly discriminative. An agent *could* recognise which maze it is
  in. It could not recover the *goal* that way — the goal is pure episodic
  content, which is precisely the thing the Hopfield store holds.
- **Task boundaries are known at training time.** We hand this to the baselines
  for free; the generosity only strengthens the control.
- **The learner is recurrent** — the most consequential fact for method
  selection.

**Good news for the controls.** In domain-IL permuted MNIST, EWC is a serious
method, not a strawman: 94.3 % against 78.5 % for naive fine-tuning, with a
97.6 % joint ceiling (van de Ven & Tolias 2019, Table 5). This is not the
class-incremental regime where parameter regularisation collapses to chance.

**Bad news.** Ehret et al. (ICLR 2021), the definitive study of continual
learning in *recurrent* networks, found weight-importance methods degrade
specifically as **working-memory demand** rises — not with sequence length, but
with how much sample-specific state the hidden vector must carry. Fisher
importance spikes with memory load, forcing stability and destroying
plasticity. A policy that integrates a trajectory to localise itself is exactly
that. Their recommendation — task-conditioned hypernetworks — is §4.4, and is
the most important single competitor in this document.

### 1.3 The two ways this is not a standard supervised CL benchmark

Worth stating once, because they change how results should be read, and then
not dwelt on:

1. **We score closed-loop rollouts, not per-sample loss.** `reached` is the
   outcome of running the policy for up to `max_steps`. Small parameter drift
   that barely moves a cross-entropy can compound over 200 steps into a total
   navigation failure — which is one reason forgetting looks so sharp here.
2. **The data is self-generated and non-i.i.d.** DAgger collects under the
   *student's* policy and labels with the oracle, so the training distribution
   moves as the policy moves. A replay buffer therefore stores states from a
   policy that no longer exists — the standard off-policy caveat, and the
   reason CLEAR's distillation term (§4.1) is worth having on top of plain ER.

Neither makes this reinforcement learning. They are the two footnotes.

---

## 2. Tier 0 — the measurements that must exist before any method is judged

None of these are continual-learning methods. They are the axes of the plot,
and three of the four have never been run.

| # | run | what it bounds | status |
|---|---|---|---|
| **T0.1** | **Joint / multi-task ceiling** — one policy on all N envs simultaneously (`train_rnn --mode mixed` on the *evaluation* envs), to convergence | The **ceiling for every weight-based method**. No CL algorithm beats joint training on its own tasks. | ✗ missing |
| **T0.2** | **Per-env expert** — N independent policies, each trained alone | Per-task ceiling; `T0.1 − T0.2` measures capacity interference at N envs | ✗ missing |
| **T0.3** | **BFS oracle** — `teacher_force=True` in `collect_rollout_rnn` | The task's own ceiling under the eval step cap; catches "the eval is impossible" | ✗ not recorded |
| **T0.4** | **Sequential from scratch** — no pretraining, naive SGD | The **floor**. Every recorded history has `mode=finetune`; the from-scratch control the paper describes appears never to have been run at final settings | ✗ missing |

**T0.1 is the most important run in this document.** If the joint ceiling at
N=5, `hidden=128` is 0.6, then no continual method can exceed 0.6, the "CL
methods fail" result is really a *capacity* result, and the honest fix is a
bigger network — which a referee will ask for. If it is ~1.0, every point of
the gap is genuinely forgetting. **Run it before writing any method code.**

Sweep it over capacity — `hidden ∈ {128, 256, 512, 1024}`, `layers ∈ {1, 2}` —
and over N ∈ {5, 20}. The resulting curve (joint ceiling vs. number of envs,
per capacity) is a figure in its own right, and it is the fair way to state
what the store buys: the Hopfield agent's ceiling does not bend down as N
grows, because nothing about its policy is per-env.

---

## 3. Tier 1 — the two controls, at their best

### 3.1 The naive control

`analysis/continual/run_baseline.sh` overrides the library defaults on several
knobs. Which of those are deliberate and which are oversights:

| # | current | verdict | action |
|---|---|---|---|
| **W1** | `BATCH_ENVS=1`, `EPOCHS=1`, `N_MINIBATCHES=1` | **Deliberate, and correct — keep it.** One rollout → one update is the online regime, and it is what makes the x-axis read as *episodes consumed*: 200 for the RNN against 1 for the Hopfield store. Batching would destroy the "episodes to acquire" axis in §0.1. Two corrections to v1: `n_minibatches` is a **no-op** at `batch_envs=1` (the batch dimension is 1, so `N // n_mb` cannot split it), so the only live knob is `epochs`; and framing this as a reduced "budget" was wrong. | Keep `batch_envs=1` as the headline regime. Run **one side condition at `batch_envs=16`** purely as a sensitivity check — a single 200-step trajectory is highly autocorrelated, and high-variance updates are themselves a known cause of forgetting. If retention is unchanged, the regime choice is vindicated for free; if it improves a lot, we need to know that before a referee does. Sweep `epochs ∈ {1, 2, 4}` (more passes over the *same* rollout adds no data and does not break the regime). |
| **W2** | `Adam` built once per seed, outside the block loop (`baseline.py:337`) | **Oversight worth testing.** Second moments carry across env boundaries; stale `v` from env i−1 makes the first steps in env i maximally destructive — the mechanism behind the *stability gap*. | Sweep: reset optimizer state per block · SGD+momentum · lower β₂ · LR warm-up over the first K updates of each block. |
| **W3** | `LR=1e-3`, constant | **Untuned.** The single strongest forgetting/plasticity knob. | Sweep `1e-4 … 3e-3` per decade plus within-block cosine decay. The retention/plasticity trade-off curve is itself a result. |
| **W4** | `hidden=128`, 1 layer, GRU | **Untested.** The net must memorise N barcode→position decoders *and* N goals. | Tied to T0.1. Sweep hidden, layers, GRU/LSTM. **Report the control at its best capacity, not at 128.** |
| **W5** | `init_log_std` **not exposed on `baseline.py`'s CLI** → pinned at the dataclass default `0.0`, i.e. σ = 1.0 on a unit-magnitude action, and learnable | **Genuine bug-shaped gap.** The DAgger student explores with noise the size of the action itself, and the run script cannot set it. | Expose `--init_log_std` / `--freeze_log_std`; sweep `init_log_std ∈ {0, −0.5, −1.0, −1.5}` × frozen/learned. |
| **W6** | no weight decay | Untested, cheap. | Small sweep. |
| **W7** | `ONLY_TRAIN_ON_REACHED` | Measured, and it **hurts** — current-env `reached` drops to 0.09–0.25 in every such history. | Keep off; report as a recorded ablation. |
| **W8** | `INPUT_PREV_ACTION=0` | **Resolved: turn it on.** | `--input_prev_action` on for the whole suite, including the Hopfield-side comparison runs if they are re-run. `input_prev_reward` stays off (reward is a time penalty plus a goal bonus; it leaks goal proximity, which the Hopfield agent does not get). |

### 3.2 The pretraining control

- **P1 — more, and more diverse, pretraining envs.** The structure worth
  learning is only extractable from many envs. Sweep pool size over decades
  (10 / 100 / 1000) and pretraining length.
- **P2 — measure the transfer, don't assume it.** Report forward transfer,
  `FT_i = (AUC_i − AUC_i^scratch)/(1 − AUC_i^scratch)`. There is currently **no
  evidence pretraining helps at all** — the pretrained run retains ~0.06 on env
  0. It may be doing nothing, and we would not know, because T0.4 has never
  been run to compare against.
- **P3 — optimizer state.** `baseline.py:285` deliberately drops Adam moments.
  Sweep both ways.
- **P4 — freeze depth.** Frozen trunk + per-env head is a much stronger and far
  more common transfer recipe than full finetuning, and it bridges to §4.5.
- **P5 — the limit case is §5.1**, meta-pretraining. "Pretrain to learn the
  structure of the task" has a principled maximum and it is OML/ANML.

### 3.3 Movement mode: continuous stays

`discrete` is easier to learn — a 4-way categorical instead of a 2-D Gaussian
with a learned σ, no exploration-scale problem, exact labels — and easier for
the control is the *right* direction for a control. But every headline
`agenthash` run is `movement_mode=continuous`, so switching the suite to
`discrete` would compare two different tasks unless the Hopfield side is
re-run too.

**Decision: continuous is the headline.** The method translations I worried
about in v1 are mostly closed-form for a Gaussian head:

| method | continuous form |
|---|---|
| CLEAR distillation | closed-form KL between two diagonal Gaussians |
| LwF | same |
| DER++ | MSE on the stored `(μ, log σ)` instead of on logits |
| EWC / online EWC | Fisher of the Gaussian log-likelihood; unchanged in form |
| EWC-DR (Logits Reversal) | **no Gaussian analogue** → discrete-only |

`discrete` survives as a robustness check on the two or three survivors, and as
the only place EWC-DR can run.

---

## 4. Tier 2 — the algorithm suite (cut to 9)

### 4.1 Wave 1 — the two that define the frontier

**ER — Experience Replay, reservoir buffer, per-env balanced sampling.**
Keep an `RNNRolloutBatch` buffer; each update trains on the new rollout plus
`k` sampled stored trajectories. Balanced sampling (uniform *per env*, not over
the stream) is the default, not a separate method — it removes the buffer's
recency bias and is uniformly better. Sweep buffer size across decades
**including unbounded**, and `k`. Unbounded ER should approach T0.1 and is
simultaneously the "perfect memory" bound and the GDumb answer. Report bytes.

**Online EWC.** Single running diagonal Fisher with decay γ; one anchor
maintained across blocks. **Compute the Fisher correctly** — the true Fisher
`E_s E_{a∼π}[(∇ log π(a|s))²]`, sampling actions from the model, not the
squared gradient of the BC loss at the teacher action (that is the *empirical*
Fisher, a different and worse estimator). Document which is used. **Sweep λ
over decades** (1e-1 … 1e5) — under-tuned λ is the standard way EWC gets
accidentally strawmanned, and the stability/plasticity curve as λ rises is a
figure.

### 4.2 Wave 2 — the competition

**CLEAR**, stated by mechanism: ER plus a distillation term to the *past self*
on replayed states — KL(π_old ‖ π_now), coefficient ~0.01, at a 50:50
new:replay ratio. Rolnick et al. framed this as continual RL, but strip
V-trace and the value-cloning term (we have no value head) and what remains is
a supervised loss. It is here because §1.3's second footnote is real: replayed
states came from a policy that no longer exists, and anchoring to the old
policy's *outputs* rather than only to the oracle's labels is the standard fix.
Needs no task boundaries or identity.

**DER++.** Store the policy's own output `(μ, log σ)` alongside the teacher
action at collection time; replay loss = BC(teacher) + α·MSE(stored, current).
Boundary-free, and the difference from CLEAR is instructive — DER++ anchors to
the output at *collection* time, CLEAR to the previous *task's* converged
policy.

**SI — Synaptic Intelligence.** Path-integral importance accumulated online, so
no separate Fisher pass and no clean boundary needed. Beat EWC in domain-IL on
both split MNIST (65.4 vs 64.0) and permuted MNIST (95.3 vs 94.3), and it is
the cheap boundary-free counterweight to online EWC.

**LwF.** Distil the pre-block model's action distribution on the *current*
env's rollouts. **No buffer at all** — the cleanest measurement of what pure
functional regularisation buys at zero storage, and a repeatedly strong
domain-IL baseline.

### 4.3 Wave 3 — the structural methods

**HNET — task-conditioned hypernetwork** (von Oswald et al., ICLR 2020).
**The headline competitor.** Don't store the policy's weights; store a small
per-task embedding and generate the weights from it, regularised so old
embeddings still produce old weights. Two reasons:

1. **It is the empirical winner for recurrent networks.** Ehret et al.
   benchmarked online EWC, SI, masking, masking+SI, generative replay, coresets,
   multitask and from-scratch across four RNN benchmarks and found HNET
   *"consistently outperformed weight-importance methods,"* especially as task
   complexity rose. We have a recurrent policy; this is the method the
   literature points at.
2. **It is the closest classical thing to what the Hopfield agent does.** Both
   keep a small addressable per-task code and recover task-specific behaviour
   from it instead of overwriting shared weights. The difference is *how the
   code gets written*: HNET learns each embedding by gradient descent over a
   whole block; the Hopfield store writes one in a **single Hebbian
   outer-product update from one episode**. That reframes the result from "we
   beat a baseline" to "we occupy the same functional niche as the best
   classical method at 0 gradient steps instead of 200."

Caveat from the same paper: hypernetworks *"introduce additional optimization
challenges, especially in conjunction with vanilla RNNs."* Budget tuning time.

**Multi-head with oracle task ID.** Shared trunk, per-env output head, told
which env it is in. This single method bounds the entire parameter-isolation
family: if isolation with a free task ID does not retain, nothing in that
family will. Report it explicitly as an upper bound, alongside its parameter
count. Pair it with the **inferred** condition — a small learned env classifier
over the observation stream, then route — because the gap between the two
measures how much of the problem is task inference rather than forgetting,
which is exactly the job the Hopfield store does in one shot.

> **What was built instead of the inferred condition, 2026-08-31.** Routing a
> learned classifier through the protocol is a second training loop with its own
> failure modes, and the *number* it would produce can be measured directly:
> how identifiable is the env from what the agent sees at all?
> `analysis/continual/task_identifiability.py` fits observations to env index,
> sweeping linear and MLP readouts over windows of 1–64 observations and
> splitting by trajectory so correlated neighbouring frames cannot land on both
> sides of it.
>
> **Best over every window and readout: 0.426, against a chance of 0.200.** The
> environments are barely identifiable from their observations, so an inferred
> task ID would be nowhere near a free substitute for the oracle — which means
> the oracle is worth a great deal and every arm in this wave is an upper bound
> rather than a peer. That is carried as a column in the tables, not a
> footnote. Building the router would refine this number; it would not change
> the conclusion that the arms need one.

**XdG — context-dependent gating** (Masse, Grant & Freedman, PNAS 2018). For
each task, zero a fixed random fraction of hidden units. ~10 lines, and it
**composes with SI**: XdG+EWC learned 100 sequential permuted-MNIST tasks at
52.4 % where either alone is near-chance. Kept over PackNet/HAT because it is
the conceptually interesting one — sparse, non-overlapping addressing is the
closest classical analogue to content-addressable storage.

### 4.4 What was cut, and why

Every drop is dominated by something already in the suite.

| dropped | why |
|---|---|
| vanilla EWC | Online EWC is the same code path at γ=1 and is usually ≥ it. Report vanilla as a γ setting if wanted, not a method. |
| MAS | Same family as EWC/SI with no distinct prediction here. |
| EWC-DR | Genuinely the best-known EWC fix, but Logits Reversal has no Gaussian analogue. Moved to the discrete robustness check (§3.3). |
| L2-init, continual backprop, shrink&perturb, ReDo | Plasticity maintenance. At N=5 the control is not plasticity-limited (`reached ≈ 0.99` on the current env). Revisit **only if** the N=20 panel shows the current-env curve degrading. |
| MESU | Interesting and boundary-free, but a larger implementation than its marginal information given SI already covers the boundary-free regularisation slot. Note as future work. |
| A-GEM | Dominated by ER in every published comparison. |
| MIR | Retrieval refinement on top of ER; only pays when the buffer is capacity-limited, and at 192 MB ours is not. |
| GDumb | Answered by ER's unbounded-buffer setting, which is the same question. |
| P&C | = online EWC + distillation, both already present separately. |
| PackNet, HAT, Progressive Nets | Multi-head with oracle task ID already bounds the isolation family. These add engineering, not information. |
| GPM | The only "store bases, not data" point on the memory axis, which is a real gap. Kept as an **optional Wave-3 stretch** if Wave 2 finishes early. |

**Nine implementations:** ER, online EWC, CLEAR, DER++, SI, LwF, HNET,
multi-head(+task inference), XdG. Plus the two Tier-3 bounds in §5.

---

## 5. Tier 3 — the bounds that decide the framing

The first two were confirmed in scope at review and are first-class
deliverables, not stretch goals. The third was added on 2026-09-01, after
Waves 0-3 had run; it is not a stretch goal either, and the fact that nothing
in the original plan asked for it is itself worth noticing.

### 5.1 Meta-pretraining — the strongest possible form of control #2

**OML** (Javed & White, NeurIPS 2019) and **ANML** (Beaulieu et al., ECAI 2020)
meta-learn a representation — ANML, a neuromodulatory gating network — *such
that subsequent online SGD does not interfere*. ANML sequentially learned 600
classes over ~9 000 SGD updates without catastrophic forgetting.

This is the principled maximum of "pretrain to learn the structure of the
task": rather than hoping features transfer, meta-learn features whose *update
dynamics* are non-interfering. Meta-train over a large pool of `GridEnv`s with
the sequential BC protocol as the inner loop; meta-test on the held-out N-env
stream.

Implementation note: the inner loop is already the thing we run
(`run_sequential_blocks` over a short block), so the meta-outer-loop is the new
part. Start with OML (simpler: split the network into a meta-learned
representation and a fast-adapting head) before attempting ANML's
neuromodulation.

### 5.2 In-context adaptation with zero weight updates

Pretrain the recurrent policy across many envs **with the hidden state carried
across episodes within an env** (RL²-style, though here the inner objective is
still supervised BC). At test, freeze the weights entirely and let the agent
adapt to a new env purely through recurrent activity — matching the Hopfield
agent's "no gradient steps at deployment" condition exactly.

Forgetting is then impossible by construction for *both* models, and the
comparison becomes the sharp one: **can an RNN's activation memory do what an
explicit associative store does?** If yes, the framing has to change. If no —
and the failure mode is legible (capacity? interference? horizon?) — that is
the strongest positive result available here.

Note this needs a change to the collector: `collect_rollout_rnn` currently
zeroes `h` on goal-reach so each per-env trajectory is an independent episode.
In-context pretraining needs the opposite — carry `h` across the episode
boundary within an env, and zero it only at the env boundary. That is a flag on
the collector, not a rewrite.

### 5.3 The substrate control — MLP and the frame-stack ladder

*Added 2026-09-01, after Waves 0-3. Not in the original plan, and it should
have been.*

Every method in §4 was published and calibrated on **feedforward** networks —
mostly MLPs on permuted or split MNIST. This suite runs all of them on a GRU.
That is not a neutral change, and nothing in Waves 0-3 measures its effect:

- **Importance estimation is scale-broken on recurrent weights.** A weight
  applied once per forward pass and a weight applied *T* times per sequence do
  not produce comparable Fisher information or SI path integrals. Part of what
  the λ sweeps over decades are finding is plausibly a scale mismatch that
  would not exist in an MLP — the same failure class as the β calibration that
  nearly buried HNET (§4.3).
- **XdG gates inside the recurrence** in this implementation. Masse et al.
  gated feedforward activations. Owning a task's *dynamics* is a strictly
  stronger intervention than owning its activations, so the suite's current
  best classic result is not the published method.
- **HNET generates GRU weights.** von Oswald generated feedforward target
  weights; recurrent weight generation is a harder conditioning problem.

So the MLP arm is the **method-fidelity control**. It separates "these methods
underperform because this task is hard" from "these methods underperform
because they are being applied to a substrate nobody characterised them on."
Those are currently confounded in every number the suite reports.

Two further things it buys, neither of which motivated it:

**It decomposes the multi-head result.** Wave 3 puts 74 % of attainable
performance loss in the shared trunk — but that trunk is a GRU. If an MLP
forgets as much, the loss is in the input→feature map, and every
recurrence-specific account of the fragility is dead.

**It is a structural zero for §5.2.** `memory_lift` conditions on "the previous
episode found the goal", which is also true of lifetimes that merely drew an
easy goal. The episodic control bounds that confound at +0.031, so it is not
unaddressed — but a memoryless policy has *no channel* for cross-episode state,
so its `memory_lift` should be exactly zero. If it is not, the statistic is
measuring something other than memory. The current design cannot run that test.

**What it does not do.** It is an added arm, not a swap. §5.2's result — carry
+0.330, in-context +0.190 — exists *because* activations persist across episode
boundaries. An MLP does not have a worse version of that; it has none.

#### Implementation shape

One agent class, and a trick that keeps the diff small: **carry the frame stack
in the hidden-state slot**. `h` already threads through `collect_rollout_rnn`
(`initial_h` / `final_h`), `bc_rnn_update` (`h0[:, idx]`) and
`evaluate_nav_one_env` (`h = out["h_next"]`). If the MLP's "hidden state" is a
rolling shift register of the last *k*−1 raw observations, shaped to the
existing `(layers, B, width)` contract, then the rollout collector, the
evaluator, the BC update and every continual method work unchanged.

It stays honestly feedforward: a shift register of raw observations is not
learned state, there is no gradient through time, and the memory horizon is
hard-bounded at exactly *k* steps. At *k*=1 it is zero. Register as another
entry in `ARCHITECTURES` in `analysis/continual/baseline.py`, sized by a
`--frame_stack` flag.

#### Phase A — 72 runs, and it gates everything else

Naive SGD only (`R_none`), otherwise the standard protocol: N=5, 200 updates
per env, `batch_envs=1`, `n_eval_trials=1`.

| axis | values |
|---|---|
| frame stack *k* | 1, 4, 8 |
| learning rate | 3e-4, 1e-3, 3e-3 |
| seeds | 8 |

**The lr sweep is not optional.** Wave 1 tuned lr for the GRU (§3.1). Comparing
a tuned recurrent net against an untuned MLP is the unfair-comparison failure
this project keeps catching in itself, and it would produce exactly the
conclusion the experiment is supposed to test.

Phase A alone answers the first-order question — *does this task need history,
and how much* — which nothing in the suite currently measures.

#### Phase B — conditional on Phase A, ~300 runs

Run only if the MLP is competitive. Methods chosen so the design carries an
internal control:

| method | why it is in |
|---|---|
| ER | Replay is substrate-agnostic in principle, so its MLP↔GRU gap is the **baseline** gap the others are read against |
| SI, EWC (4 λ each) | Where recurrent importance estimation is most suspect. Coefficients do **not** transfer across substrates and must be re-swept, not reused |
| XdG, XdG+SI (2 gating values) | The most interesting cell in the design: gating a feedforward net is what Masse et al. actually published, so this is the *faithful* version of the suite's current best classic result |

#### What must be matched, or the wave is void

1. **Parameter count.** The GRU trunk is 73,220. Input width is 62 per frame
   (60 rays + `prev_action`), so it grows to 496 at *k*=8 and the hidden width
   must shrink to compensate — solved per *k*, not once. Unmatched capacity
   confounds every comparison in both phases.
2. **A fresh joint ceiling (T0.1).** "Is this forgetting or capacity?" *resets*
   for a new architecture. Without an MLP joint-training run there is no
   denominator for an MLP retention number. Cheap, easy to forget, fatal if
   forgotten.
3. **Wave-4 histories carry `optimal_to_goal`; Waves 0-3 do not.** Route
   efficiency will be available for the MLP arms and not for their GRU
   comparators. Say so rather than quietly comparing.

#### Registered readings

| outcome | reading |
|---|---|
| *k*=1 ≈ GRU on current-env | The task is Markov in one observation. Recurrence buys nothing on the supervised task, and every recurrence-specific claim about the continual results is unsupported. |
| ladder saturates by *k*=4 | Short-window integration only. The GRU is not doing the long integration its architecture permits. |
| ladder never reaches the GRU | Recurrence earns its place, and the residual gap quantifies what unbounded integration is worth. |
| MLP retains **more** at matched current-env | Forgetting is recurrence-specific; §5 of the results page is about dynamics, not features. |
| SI/EWC gap ≫ ER gap | Importance estimation is substrate-broken and the parameter-regularisation result is understated. |
| `memory_lift`(*k*=1) ≠ 0 under §5.2's protocol | The statistic is confounded by lifetime difficulty and §5.2 needs revisiting. At *k*>1 there is a bounded *k*-step leak across the episode boundary, so *k*=1 is the clean cell. |

**Prediction, registered before the runs exist.** The MLP loses on current-env
— single observations look aliased, since a classifier on the agent's own
observations separates the five environments at 0.426 against 0.200 chance. But
the ladder saturates early, around *k*=4, which would mean the GRU is
exploiting a short window rather than the long integration it is capable of.

#### Cost

Phase A is 72 runs, against Wave 1's 272 and Wave 3's 184 — small, and it
decides whether Phase B is worth its ~300. One sbatch array each.

---

## 6. Sequencing and the Wave-1 schedule

### 6.1 Measured cost

Timed on one CPU core, `batch_envs=1`, `size=20`, `hidden=128`, 200-step
rollouts: **~1000 environment-steps per second**. The workload is
env-stepping-bound at `batch_envs=1`, so the scaling axis is **CPU fan-out, not
GPUs** — a 128-unit GRU on a batch of 1 barely touches an accelerator. (This
likely explains why the existing 30-seed job needed a 12 h allocation: 30
processes contending on one GPU. Verify against a GPU run in the first launch.)

Per-seed cost, counting `200 × (rollout + eval over envs-so-far)` per block:

| protocol | env-steps / seed | wall / seed |
|---|---|---|
| N=5, 200 upd/env | 800 k | **~13 min** |
| N=20, 200 upd/env | 9.2 M | **~2.6 h** |

### 6.2 Resources

| partition | limit | nodes | per node | use |
|---|---|---|---|---|
| `ou_bcs_normal` | **1 day** | ~20 | A100×8 or H100×4, **192–256 CPU**, ~1 TB | **Wave-1 workhorse.** The 1-day limit and the CPU count are both what we need. |
| `pi_fiete` | **7 days** | 1 | A100×8, 192 CPU | Long runs — Wave 3 HNET tuning, Wave 4 meta-pretraining. |
| `mit_normal_gpu` | 6 h | ~66 (10 idle) | L40S×4 / H200×8, 64–120 CPU | Short parallel sweeps that fit in 6 h; good for burst capacity. |

One 192-CPU node runs ~150 N=5 seeds concurrently. Wave 1's whole grid is a few
node-hours; **queue time, not compute, is the binding constraint on the 24 h
deadline**, so submit early and wide rather than deep.

### 6.3 Waves

**Wave 0 — the axes. Blocking.** T0.1 joint ceiling (capacity sweep × N ∈ {5,20})
· T0.2 per-env experts · T0.3 oracle · T0.4 from-scratch sequential.
*Nothing else is interpretable until these land.*

**Wave 1 — target: 24 h.** Tier-1 fixes W1–W6 (W2 and W5 first — cheapest,
largest expected effect) · pretraining P1–P4 · **ER** with buffer-size sweep
including unbounded · **online EWC** with λ over decades.
*Exit: we know the ceiling, the floor, and where a good buffer and a good
regulariser sit between them.*

**Wave 2.** CLEAR · DER++ · SI · LwF.
*Exit: the best classic method on each of the five cost axes is identified.*

**Wave 3.** HNET (oracle and inferred embeddings) · multi-head with oracle and
inferred task ID · XdG (+SI) · GPM if there is room.

**Wave 4.** §5.3 substrate control, Phase A gating Phase B · §5.1
meta-pretraining · the N=20 scaling panel across the surviving methods · the
N=20 `agenthash` run §10 assigns here · plasticity maintenance only if N=20
shows the current-env curve degrading.
*§5.2 was pulled forward and answered on 2026-09-01 — it is no longer part of
this wave; see the log.*

---

## 7. Evaluation protocol

### 7.1 Metrics

| metric | definition |
|---|---|
| **A_N** | mean `reached` over all N envs at the end of the stream |
| **Forgetting** | `mean_i max(p_ii − p_Ni, 0)` — peak minus final, per env |
| **BWT** | `1/(N−1) Σ (p_Ni − p_ii)` |
| **FT_i** | `(AUC_i − AUC_i^scratch)/(1 − AUC_i^scratch)` — the only metric that scores pretraining, and the one currently missing |
| **Stability gap** | the *transient* drop on env `i-1` in the first updates of block `i`. The per-update trace already records this at full resolution; nobody has plotted it |
| **Episodes-to-criterion** | rollouts in env `i` before `reached ≥ 0.9`. **The axis on which the Hopfield agent's advantage is largest** (1 vs ~200), and which the current figures do not show at all. `batch_envs=1` is what keeps this axis clean (§3.1 W1) |
| **Stored bytes** | replay data + per-task params + importance matrices + masks |
| **Parameters** | flat for most; grows with N for multi-head |

### 7.2 Matched budgets

Report every method at **matched gradient steps** and **matched environment
interactions**. Replay methods see more data per step; say which is matched. A
retention number obtained at 4× the compute is not a comparison.

### 7.3 Hyperparameters

Wang et al. 2024, *Hyperparameters in Continual Learning: A Reality Check* —
across 8 000+ experiments, *"most state-of-the-art algorithms fail to replicate
their reported performance"* once hyperparameters are tuned on one stream and
evaluated on another, because the conventional protocol tunes on the very
stream it evaluates.

**Adopt their two-phase protocol.** Tune on a *tuning stream* of envs from the
same generator; evaluate frozen on a disjoint *evaluation stream*. `rnn_world`
already supports declared domains and held-out envs (`--n_val_envs`), so this
is cheap to do correctly. Report each method's grid in an appendix — an
under-tuned EWC is not a result.

### 7.4 Seeds and eval cadence

≥ 20 seeds, report SEM. Single-seed histories swing violently: the same config
ranges from `{0, 0, 0, 0.098, 1}` to `{0.049, 0.976, 0.805, 0.317, 1}`.

Eval every update at N=5 (keeps stability-gap resolution). At N=20 eval is
~85 % of the cost; if the panel needs to be cheaper, drop to every 5 updates
there and keep N=5 at full resolution.

---

## 8. Implementation shape

`run_sequential_blocks` already takes `on_update` / `on_block_start` callbacks
and drives `collect_rollout_rnn → bc_rnn_update → evaluate_nav_all`. Every
method in §4 except HNET is a modification of two things: **what loss the
update adds**, and **what happens at a block boundary**.

```python
class ContinualMethod(Protocol):
    def on_block_start(self, block, agent, envs) -> None: ...
    def extra_batches(self, rollout) -> list[RNNRolloutBatch]: ...  # ER, CLEAR, DER++
    def penalty(self, agent) -> torch.Tensor: ...                   # EWC, SI, LwF
    def after_step(self, agent) -> None: ...                        # SI path integral
    def on_block_end(self, block, agent, envs) -> None: ...         # Fisher, anchors, gating
    def state_bytes(self) -> int: ...                               # the memory axis
```

`bc_rnn_update` needs one change: accept an optional list of extra batches and
an optional penalty term. Everything else is additive. HNET replaces the agent
rather than the update, so it gets its own driver reusing
`run_sequential_blocks`.

Layout, respecting the layering rules in `hopfield_nav/tests/test_layering.py`
(`analysis/` may import `hopfield_nav`, never the reverse):

```
hopfield_nav/continual/
    base.py           # ContinualMethod protocol + a no-op
    replay.py         # ER, CLEAR, DER++
    regularize.py     # online EWC, SI
    distill.py        # LwF
    isolate.py        # multi-head, XdG, task inference
    hypernet.py       # HNET (its own agent + driver)
analysis/continual/
    baseline.py       # gains --method / --method_args
    metrics.py        # NEW: A_N, FG, BWT, FT, stability gap, bytes
    run_suite.sh      # sweeps the suite over seeds and methods
```

`baseline.py` grows `--method` and `--method_args` and records both in
`history["metadata"]`, so every existing plotting and merging path keeps
working and old histories stay readable.

---

## 9. Risks

| risk | mitigation |
|---|---|
| **The joint ceiling is low** — the RNN cannot represent N envs at any tested capacity | Then this is a capacity result, not a forgetting result, and must be *reported as such*. T0.1's capacity sweep is what distinguishes them. This is why Wave 0 blocks. |
| **Unbounded ER matches the Hopfield agent on retention** | Expected. The claim moves to the cost frontier — 0 gradient steps, 1 episode, no stored data. Build the frontier figures from the start. |
| **In-context adaptation also retains** (§5.2) | The most serious risk to the framing. Better found in Wave 4 than in review. |
| **HNET beats everything** | Fine, and interesting — same functional niche. The differentiator is one-shot Hebbian write vs. a per-task gradient-descent inner loop. Frame it that way from the start. |
| **`batch_envs=1` gradient noise is doing part of the forgetting** | The `batch_envs=16` side condition in W1 measures exactly this. If it matters, report both regimes rather than switching. |
| **Queue time eats the 24 h** | Submit Wave 0 and Wave 1 together, wide rather than deep; `ou_bcs_normal` for the 1-day limit, `mit_normal_gpu` for burst. Compute is a few node-hours. |

---

## 10. Decisions taken, and what is still open

**Settled in review:**

| # | decision |
|---|---|
| 1 | `--input_prev_action` **on**; `input_prev_reward` stays off (it leaks goal proximity). **Never took effect** — see below. |
| 2 | **N=5 headline, N=20 scaling panel** |
| 3 | **`continuous` movement mode**, matching the existing `agenthash` runs; `discrete` as a robustness check and the only home for EWC-DR |
| 4 | `batch_envs=1` **kept as the regime**, with a `batch_envs=16` sensitivity condition |
| 5 | Continual-RL framing **cut**; this is supervised domain-incremental CL |
| 6 | Suite **cut to 9** methods; §5.1 and §5.2 promoted to first-class |
| 7 | Compute: `ou_bcs_normal` (1-day) primary, `pi_fiete` (7-day) for long runs, `mit_normal_gpu` for burst; Wave 1 target 24 h |

**Correction to decision #1 (2026-09-02).** The channel was on in exactly one
arm of the whole suite, A2, and off in 616 of 664 continuous runs and 424 of
472 discrete ones. The pretraining checkpoints were built without it, and
`restore_arch_from_ckpt` takes `input_prev_action` from the checkpoint because
the two imply different input widths — so every arm loading a checkpoint ran
without the channel whatever its command line said. No script passes the flag
*and* a checkpoint, so nothing was silently dropped; the decision was simply
not implementable for the pretrained arms without rebuilding the checkpoint,
which nobody noticed was required.

It has since been measured rather than assumed. From scratch in discrete the
channel is worth +0.079 to +0.121 retained across the replay family and flat
elsewhere, and it closes almost the whole gap to pretraining — ER reaches 0.945
from random weights with the channel against 0.961 pretrained without it. Two
checkpoints have been rebuilt with the channel and the full method suite run on
top of them, so all eight cells of the 2x2x2 (action space x initialisation x
channel) now exist.

**Still open — none blocking, defaults chosen:**

- **Slurm account / QoS.** `run_baseline.sh` uses `--partition=pi_evelina9` with
  no account flag. Assuming partition alone suffices for `ou_bcs_normal` and
  `pi_fiete`; will surface the error immediately if not.
- **N=20 eval cadence.** Defaulting to every update for comparability; will drop
  to every 5 if the panel is the long pole (§7.4).
- **Where the N=20 Hopfield comparison comes from.** The existing `agenthash`
  histories are all N=5. The scaling panel needs an N=20 `agenthash` run to
  compare against — cheap (frozen policy, no training), but it is a run that
  does not exist yet. Folding it into Wave 4.

---

## 11. Reading list

**Positioning**
- van de Ven, Tuytelaars & Tolias 2022, *Three types of incremental learning*, Nat. Mach. Intell. — the scenario taxonomy
- van de Ven & Tolias 2019, *Three scenarios for continual learning* — arXiv 1904.07734 (the domain-IL numbers in §1.2)
- Wang et al. 2023, *A Comprehensive Survey of Continual Learning* — arXiv 2302.00487

**Recurrent-specific — read these first**
- **Ehret et al., ICLR 2021, *Continual Learning in Recurrent Neural Networks* — arXiv 2006.12109.** The single most relevant paper to this setting.
- Cossu et al. 2021, *Continual Learning for RNNs: an Empirical Evaluation* — arXiv 2103.07492
- von Oswald et al., ICLR 2020, *Continual learning with hypernetworks* — arXiv 1906.00695

**The nine methods**
- Chaudhry et al. 2019 (ER) · Rolnick et al. 2019, arXiv 1811.11682 (CLEAR) · Buzzega et al. 2020, arXiv 2004.07211 (DER++)
- Schwarz et al. 2018 (online EWC) · Zenke et al. 2017 (SI) · Li & Hoiem 2016 (LwF)
- Masse, Grant & Freedman, PNAS 2018, arXiv 1802.01569 (XdG)

**Tier-3 bounds**
- Javed & White 2019, *Meta-Learning Representations for Continual Learning* (OML) — arXiv 1905.12588
- Beaulieu et al. 2020, *Learning to Continually Learn* (ANML) — arXiv 2002.09571

**Methodology**
- Wang et al. 2024, *Hyperparameters in Continual Learning: A Reality Check* — arXiv 2403.09066
- De Lange et al. 2023, *Continual evaluation for lifelong learning: identifying the stability gap* — arXiv 2205.13452

**Cut but worth citing if a referee asks**
- Kirkpatrick et al. 2017 (EWC) · Aljundi et al. 2018 (MAS) · Liu et al. CVPR 2026, arXiv 2603.18596 (EWC-DR) · Bruno et al. Nat. Commun. 2025, arXiv 2504.13569 (MESU) · Saha et al. ICLR 2021, arXiv 2103.09762 (GPM) · Dohare et al. Nature 2024 (loss of plasticity)
