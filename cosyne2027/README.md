# Cosyne 2027 figures

Each figure has a script here that writes its PDF and PNG next to itself.
Shared colors and fonts live in `style.py`, so an arena keeps its color
across figures.

```
~/.conda/envs/cls/bin/python cosyne2027/fig1_task_schematic.py
```

Panels are drawn at final size (6.5 in wide, 7 pt text). Scaling a panel
in the layout changes its font size.

## Figure 1: sequential water maze task

`fig1_task_schematic.py` draws the protocol in
`hopfield_nav/evaluation/protocols.py`. The paths are illustrations on the
20x20 cell lattice, not agent rollouts.

Draft caption. Values in braces depend on which run the results figures use:
`run_cl_ood.sh` uses 6 arenas, 40 iterations and 200 steps, and
`run_agenthash.sh` uses 5, 200 and 100.

> **Sequential water maze.** The agent meets {N} arenas, each 20x20 cells
> with one hidden goal, one at a time. Each trial starts at a random cell
> and ends when the agent reaches the goal or after {max_steps} steps. In
> block *i*, each of {iters} iterations runs one trial in every arena seen
> so far. The first trial in a new arena is a search. On reaching the goal,
> its location is written to memory (star), and later trials can go
> straight to it. Only the newest arena can write to memory, and memory is
> never cleared, so trials in earlier arenas (pale) measure retention.
