# Cosyne 2027 figures

Each figure has a script here that writes its PDF and PNG next to itself.
Shared colors and fonts live in `style.py`, so an arena keeps its color
across figures.

```
~/.conda/envs/cls/bin/python cosyne2027/fig1_task_schematic.py
```

Panels are drawn at final size with 9 pt text and almost no margin.
Insert them at 100%; scaling a panel changes its font size.

## Figure 1: sequential water maze task

`fig1_task_schematic.py` draws the protocol in
`hopfield_nav/evaluation/protocols.py`, 4.2 x 1.3 in. It shows the task
only: nothing in it, or in the caption, depends on the model (no memory
writes). The paths are illustrations on the 20x20 cell lattice, not agent
rollouts.

Draft caption. Values in braces depend on which run the results figures use:
`run_cl_ood.sh` uses 6 arenas, 40 iterations and 200 steps, and
`run_agenthash.sh` uses 5, 200 and 100.

> **Sequential water maze.** The agent meets {N} arenas one at a time.
> Each is 20x20 cells with one hidden goal, and each has its own random
> wall pattern, the only visual cue that tells arenas apart. Each trial starts at a random cell
> and ends when the agent reaches the goal or after {max_steps} steps. In
> block *i*, each of {iters} iterations runs one trial in every arena seen
> so far; each column is one block, each square one trial. The first trial
> in a new arena is a search, and later trials can go straight to the goal.
> Trials in earlier arenas (pale) test whether the agent still knows those
> goals after learning new ones.

## Figure 2: model

`fig2_model_schematic.py` redraws `model_schematic.pdf` (the hand-made
version, kept here as the reference) in the style of figure 1 at exactly
3:1 (5.58 x 1.86 in). Arrows carry words, not symbols, and each box names
its brain region. Box fill gives the module type, and the key sits under
the embedding net. The Vector-HaSH modules are gray because they do not
learn; the Hebbian link between hippocampus and sensory input is drawn in
the plastic color. There is no direction-decoder box: the attractor's
recall direction goes straight to the controller.

## Font

The PDFs name the core fonts Helvetica and Symbol without embedding them,
so Illustrator opens them with the Mac's own Helvetica and the text stays
editable. Greek letters and the minus sign come from Symbol. The PNG
previews render with Nimbus Sans, URW's metric-identical clone. Set math
with `style.rich`, not mathtext: mathtext places glyphs from its own font,
which Illustrator flags as missing.

## Figure 3: main results

`cl_results.py` builds every number from one matrix and writes
`data/cl_results.json`; `fig3_main_results.py` draws from that file only.
a[k, j] is success on arena j over the last 20% of arena k's 200-update
window, where success is a held-out rollout from a random start reaching the
goal (radius 0.5) within T_eval = 100 steps. P = mean_j a[j, j] (plasticity),
ACC = mean_j a[4, j] (final success), F = P - ACC (forgetting). Path ratio is
path length over the straight-line optimum on successful final-window
rollouts; it is reported as a median because starts next to the goal make
single ratios huge.

- Controls: the continuous, pretrained, previous-action-channel cell of the
  continual control suite (`analysis/continual/run_wave_prev.sh`, 8 seeds,
  one deterministic held-out rollout per seen arena after every update,
  200-step cap). Success within 100 steps is read from those rollouts as
  `steps_to_goal < 100`, which is exact for a deterministic policy, so no
  control was retrained. Each method's configuration is re-selected at
  T_eval = 100 with the suite's own rule (current arena >= 0.5, then best
  retention); that moves naive SGD to lr 3e-4 and XdG + SI to lambda 1e4.
- Agent-HaSH (d0_base u725): `run_agenthash_heldout.sh`, the same schedule
  (one store-allowed episode per update, 200-step cap; held-out no-store
  episodes on every seen arena, 100-step cap), sampled actions, one store per
  arena at the goal, 8 seeds with `--env_seed` = the controls' seeds.
- Each model keeps the wall resolution it was trained on (controls 1 barcode
  segment per cell, Agent-HaSH 4), so the arenas are different random draws
  from the same generator; the goal radius is common.

`fig3_a2_per_env.py` is the per-arena alternative to panel a2: one small
multiple per model (Agent-HaSH, naive SGD, XdG + SI), each 35 x 25 mm with a
shared y axis, one line per arena in the fig 1 arena colors (A-E; D and E
extend the palette with its next two slots, yellow and magenta, so the
figure carries an arena key).

## Encoder figure

`encoder_similarity.py` (run as `python -m cosyne2027.encoder_similarity`)
writes `data/encoder_similarity.json`; `fig_encoder.py` draws it. Raw
cosine similarity against axis-aligned distance (0-100 cells, offsets along
x or y) for d0_base's trained encoder, the smoothed grid code it reads
(lambdas 11/12/13) and an untrained encoder of the same architecture (torch
seed 0, gain 100); mean +-1 SD over 2000 position pairs per distance, CPU
only. The grid code is translation-invariant, so its curve is exact (SD 0,
no band): 0.91 at lambda = 12, 0.70 at 24, partial realignments out to 100.
Axis-aligned rather than all-direction Euclidean: averaged over a distance
ring, the lattice code's direction dependence gives a jagged curve (real
structure, not sampling: the exact ring average is as rough) that dilutes
the lambda bump to 0.25. Raw rather than
mean-centred because that is what the attractor compares: the untrained
encoder maps every position to nearly one vector (cosine ~1 everywhere),
and centred it would retrace the grid-code curve.

No place-field inset. The clean single fields in
`encoder_training/notebooks/explore.ipynb` (cell 5) come from the April
encoder `encoders/run_20260422_185816` (gain 3.7) after mean-subtraction, a
floor, a sigma-30 high-pass and a sigma-2 blur. d0_base's encoder runs at
gain 100: its units are near-binary sign codes over large regions (98.7%
touch the wall of a 50-cell arena), and the same recipe gives multi-blob
patterns, not single fields.
