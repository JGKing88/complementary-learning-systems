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

> **Sequential water maze.** The agent meets {N} arenas, each 20x20 cells
> with one hidden goal, one at a time. Each trial starts at a random cell
> and ends when the agent reaches the goal or after {max_steps} steps. In
> block *i*, each of {iters} iterations runs one trial in every arena seen
> so far; each column is one block, each square one trial. The first trial
> in a new arena is a search, and later trials can go straight to the goal.
> Trials in earlier arenas (pale) test whether the agent still knows those
> goals after learning new ones.

## Figure 2: model

`fig2_model_schematic.py` redraws `model_schematic.pdf` (the hand-made
version, kept here as the reference) in the style of figure 1, without the
small network drawings, at exactly 3:1 (6.5 x 2.17 in). Box fill gives the
module type; the key is in the top right.

## Font

Both figures use Helvetica as Nimbus Sans, URW's clone with the same design
and metrics (licensed Helvetica is not on the cluster). `style.apply()`
loads it from `/usr/share/fonts/urw-base35` and the PDFs embed it.
