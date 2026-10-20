"""Figure 1: the sequential water maze task.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig1_task_schematic.py

Draws the task in hopfield_nav/evaluation/protocols.py, model-free: nothing
here depends on how an agent solves it. Each column is one block. Top row:
the arena introduced in that block, with its wall pattern, hidden goal, a
first-trial search path and a later direct path. Bottom row: the block's trials in
order. Block i runs one trial in every arena seen so far (A, then A B, then
A B C, repeated); trials in earlier arenas are revisits.

The paths are illustrations, not agent rollouts. They live on the real
20x20 cell lattice: the search path is a novelty-seeking walk that ends at
the goal, and the later path is a shortest 4-connected path. The wall
pattern is the env's ±1 barcode (GridEnv._wall_code: one segment per wall
cell, drawn per arena from its own rng). It is the only thing that tells
arenas apart, so each arena gets its own draw here.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

import style

style.apply()

ARENA_CELLS = 20
MOVES = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

# (goal, search start, later-trial start) in cell coordinates, origin bottom-left
ARENAS = [
    ((14, 13), (3, 4), (4, 15)),
    ((5, 6), (15, 15), (16, 3)),
    ((12, 4), (4, 16), (2, 6)),
]
NAMES = "ABC"
ITERS_SHOWN = [4, 3, 2]         # iterations drawn per block before the '...'

# Layout, in inches (the axes spans the figure with 1 data unit = 1 inch).
# FIG_W and FIG_H are the final size; keep them snug to the key's text.
FIG_W, FIG_H = 4.18, 1.26
MARGIN = 0.02                   # keeps edge strokes inside the canvas
ARENA_SIDE = 0.9
WALL = 0.05                     # wall-pattern band, inside ARENA_SIDE
COL_X = [0.45, 1.5, 2.55]       # column centers
ELLIPSIS_X = 3.14
ARENA_Y0 = 0.22                 # arena bottom edge
TICK_Y = 0.07                   # trial-tick centers
TICK = 0.1
TICK_GAP = 0.03
DOTS_W = 0.15                   # room for the '...' after each block's ticks
KEY_X = 3.32
KEY_PITCH = 0.162


def search_path(start, goal, rng, max_steps=400):
    """Walk that prefers unvisited cells and keeps its heading; stops at goal."""
    pos = np.array(start)
    path = [pos.copy()]
    seen = {tuple(pos)}
    heading = None
    for _ in range(max_steps):
        nxt = pos + MOVES
        inside = np.all((nxt >= 0) & (nxt < ARENA_CELLS), axis=1)
        w = np.where([tuple(n) in seen for n in nxt], 0.05, 1.0) * inside
        if heading is not None:
            w = w * np.where(np.all(MOVES == heading, axis=1), 3.0, 1.0)
        k = rng.choice(4, p=w / w.sum())
        heading = MOVES[k]
        pos = nxt[k]
        path.append(pos.copy())
        seen.add(tuple(pos))
        if tuple(pos) == tuple(goal):
            return np.array(path)
    return None


def find_search_path(start, goal, lo=60, hi=100):
    """First seed whose walk reaches the goal in lo..hi steps (deterministic)."""
    for seed in range(10_000):
        p = search_path(start, goal, np.random.default_rng(seed))
        if p is not None and lo <= len(p) <= hi:
            return p
    raise RuntimeError("no search path in range")


def direct_path(start, goal):
    """Shortest 4-connected path that interleaves x and y moves evenly."""
    (x0, y0), (x1, y1) = start, goal
    dx, dy = x1 - x0, y1 - y0
    n = abs(dx) + abs(dy)
    path = [(x0, y0)]
    x, y = x0, y0
    for i in range(1, n + 1):
        # take an x move whenever x is behind its share of the straight line
        if abs(x - x0) < round(abs(dx) * i / n):
            x += np.sign(dx)
        else:
            y += np.sign(dy)
        path.append((x, y))
    return np.array(path)


def smooth(path, passes=3):
    """Chaikin corner cutting, endpoints fixed, so grid paths read as motion."""
    p = np.asarray(path, float)
    for _ in range(passes):
        q = np.empty((2 * len(p) - 2, 2))
        q[0::2] = 0.75 * p[:-1] + 0.25 * p[1:]
        q[1::2] = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], q, p[-1:]])
    return p


def segment(ax, x, y, w, h, color, bit):
    """One wall segment: +1 in the arena color, -1 in its tint."""
    ax.add_patch(Rectangle((x, y), w, h, ec="none", zorder=5,
                           fc=color if bit > 0 else style.tint(color, 0.72)))


def draw_walls(ax, x0, y0, color, code):
    """±1 barcode around the floor, one segment per wall cell, plus corners."""
    inner = ARENA_SIDE - 2 * WALL
    c = inner / ARENA_CELLS
    far = x0 + WALL + inner, y0 + WALL + inner
    for k in range(ARENA_CELLS):
        segment(ax, x0 + WALL + k * c, far[1], c, WALL, color, code[0, k])
        segment(ax, far[0], y0 + WALL + k * c, WALL, c, color, code[1, k])
        segment(ax, x0 + WALL + k * c, y0, c, WALL, color, code[2, k])
        segment(ax, x0, y0 + WALL + k * c, WALL, c, color, code[3, k])
    for cx in (x0, far[0]):
        for cy in (y0, far[1]):
            ax.add_patch(Rectangle((cx, cy), WALL, WALL, fc=color, ec="none",
                                   zorder=5))
    for off, side in ((0, ARENA_SIDE), (WALL, inner)):   # outer, floor edge
        ax.add_patch(Rectangle((x0 + off, y0 + off), side, side, fc="none",
                               ec=color, lw=0.5, zorder=6))


def draw_arena(ax, cx, color, name, goal, s_start, d_start, seed):
    x0, y0 = cx - ARENA_SIDE / 2, ARENA_Y0
    inner = ARENA_SIDE - 2 * WALL
    c = inner / ARENA_CELLS
    fx0, fy0 = x0 + WALL, y0 + WALL       # floor origin
    ax.add_patch(Rectangle((fx0, fy0), inner, inner,
                           fc=style.WATER, ec="none", zorder=0))
    for k in range(1, ARENA_CELLS):
        ax.plot([fx0 + k * c] * 2, [fy0, fy0 + inner],
                color=style.GRID, lw=0.25, zorder=1)
        ax.plot([fx0, fx0 + inner], [fy0 + k * c] * 2,
                color=style.GRID, lw=0.25, zorder=1)
    code = np.random.default_rng(seed).choice([-1, 1], size=(4, ARENA_CELLS))
    draw_walls(ax, x0, y0, color, code)

    def xy(p):
        p = np.asarray(p, float)
        return fx0 + (p[..., 0] + 0.5) * c, fy0 + (p[..., 1] + 0.5) * c

    ax.plot(*xy(smooth(find_search_path(s_start, goal))),
            color=style.INK_MUTED, lw=0.7, zorder=2)
    ax.plot(*xy(smooth(direct_path(d_start, goal))), color=color, lw=1.5,
            zorder=3)
    for start in (s_start, d_start):
        ax.add_patch(Circle(xy(start), 0.025, fc="white", ec=style.INK,
                            lw=0.7, zorder=4))
    ax.add_patch(Circle(xy(goal), 1.6 * c, fc="none", ec=style.INK,
                        lw=0.9, ls=(0, (2, 1.2)), zorder=4))

    ax.text(cx, y0 + ARENA_SIDE + 0.03, f"Arena {name}", ha="center",
            va="bottom")


def tick(ax, x, y, color, revisit):
    ax.add_patch(Rectangle((x, y - TICK / 2), TICK, TICK,
                           fc=style.tint(color, 0.7) if revisit else color,
                           ec=color, lw=0.7, zorder=3))


def draw_block(ax, i):
    """Trial ticks for block i: iterations over arenas 0..i, then '...'."""
    seq = [j for _ in range(ITERS_SHOWN[i]) for j in range(i + 1)]
    width = len(seq) * TICK + (len(seq) - 1) * TICK_GAP + DOTS_W
    x = COL_X[i] - width / 2
    for j in seq:
        tick(ax, x, TICK_Y, style.ARENA_COLORS[j], revisit=j != i)
        x += TICK + TICK_GAP
    ax.text(x, TICK_Y, "...", ha="left", va="center", color=style.INK_2)


def draw_key(ax):
    gx, tx = KEY_X, KEY_X + 0.22
    y = ARENA_Y0 + ARENA_SIDE - 0.05 - KEY_PITCH * np.arange(7)
    for k, bit in enumerate([1, -1, 1, 1, -1, 1, -1, -1]):
        segment(ax, gx + k * 0.02, y[0] - WALL / 2, 0.02, WALL,
                style.INK_MUTED, bit)
    labels = ["wall pattern", "search", "later trial", "hidden goal",
              "start", "new arena", "revisit"]
    for yy, label in zip(y, labels):
        ax.text(tx, yy, label, va="center", color=style.INK_2)
    y = y[1:]

    t = np.linspace(0, 1, 30)
    ax.plot(gx + 0.16 * t, y[0] + 0.03 * np.sin(t * 4 * np.pi),
            color=style.INK_MUTED, lw=0.7)
    ax.plot([gx, gx + 0.16], [y[1]] * 2, color=style.INK_2, lw=1.5)
    ax.add_patch(Circle((gx + 0.08, y[2]), 0.055, fc="none", ec=style.INK,
                        lw=0.9, ls=(0, (2, 1.2))))
    ax.add_patch(Circle((gx + 0.08, y[3]), 0.025, fc="white", ec=style.INK,
                        lw=0.7))
    tick(ax, gx + 0.08 - TICK / 2, y[4], style.INK_MUTED, revisit=False)
    tick(ax, gx + 0.08 - TICK / 2, y[5], style.INK_MUTED, revisit=True)



def main():
    fig = plt.figure(figsize=(FIG_W + 2 * MARGIN, FIG_H + 2 * MARGIN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-MARGIN, FIG_W + MARGIN)
    ax.set_ylim(-MARGIN, FIG_H + MARGIN)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (goal, s_start, d_start) in enumerate(ARENAS):
        draw_arena(ax, COL_X[i], style.ARENA_COLORS[i], NAMES[i],
                   goal, s_start, d_start, seed=i)
        draw_block(ax, i)
    ax.text(ELLIPSIS_X, ARENA_Y0 + ARENA_SIDE / 2, "...", ha="center",
            va="center", color=style.INK_2)

    draw_key(ax)
    style.save(fig, "fig1_task_schematic")


if __name__ == "__main__":
    main()
