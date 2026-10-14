"""Figure 1: the sequential water maze task.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig1_task_schematic.py

Draws the protocol in hopfield_nav/evaluation/protocols.py, model-free.
Each column is one block. Top row: the arena introduced in that block, with
its hidden goal, a first-trial search path and a later direct path. Bottom
row: the block's trials in order. Block i runs one trial in every arena seen
so far (A, then A B, then A B C, repeated); only the newest arena may store
its goal, and the others are revisits.

The paths are illustrations, not agent rollouts. They live on the real
20x20 cell lattice: the search path is a novelty-seeking walk that ends at
the goal, and the later path is a shortest 4-connected path.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

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

# Layout, in inches (the axes spans the figure with 1 data unit = 1 inch)
FIG_W, FIG_H = 6.5, 1.85
ARENA_SIDE = 1.05
COL_X = [0.62, 2.17, 3.72]      # column centers
ELLIPSIS_X = 4.55
ARENA_Y0 = 0.60                 # arena bottom edge
TICK_Y = 0.30                   # trial-tick centers
TICK = 0.105
TICK_GAP = 0.035
KEY_X = 5.12
KEY_PITCH = 0.15


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


def draw_arena(ax, cx, color, name, goal, s_start, d_start):
    x0, y0 = cx - ARENA_SIDE / 2, ARENA_Y0
    c = ARENA_SIDE / ARENA_CELLS
    ax.add_patch(Rectangle((x0, y0), ARENA_SIDE, ARENA_SIDE,
                           fc=style.WATER, ec="none", zorder=0))
    for k in range(1, ARENA_CELLS):
        ax.plot([x0 + k * c] * 2, [y0, y0 + ARENA_SIDE],
                color=style.GRID, lw=0.25, zorder=1)
        ax.plot([x0, x0 + ARENA_SIDE], [y0 + k * c] * 2,
                color=style.GRID, lw=0.25, zorder=1)
    ax.add_patch(Rectangle((x0, y0), ARENA_SIDE, ARENA_SIDE,
                           fc="none", ec=color, lw=1.4, zorder=5))

    def xy(p):
        p = np.asarray(p, float)
        return x0 + (p[..., 0] + 0.5) * c, y0 + (p[..., 1] + 0.5) * c

    sp = find_search_path(s_start, goal)
    ax.plot(*xy(smooth(sp)), color=style.INK_MUTED, lw=0.6, zorder=2)
    dp = direct_path(d_start, goal)
    ax.plot(*xy(smooth(dp)), color=color, lw=1.4, zorder=3)
    for start in (s_start, d_start):
        ax.add_patch(Circle(xy(start), 0.022, fc="white", ec=style.INK,
                            lw=0.6, zorder=4))
    ax.add_patch(Circle(xy(goal), 1.6 * c, fc="none", ec=style.INK,
                        lw=0.8, ls=(0, (2, 1.2)), zorder=4))

    ax.text(cx, y0 + ARENA_SIDE + 0.05, f"Arena {name}", ha="center",
            va="bottom", color=style.INK)


def draw_block(ax, i, n_iters):
    """Ticks for block i: n_iters iterations over arenas 0..i, then '...'."""
    seq = [j for _ in range(n_iters) for j in range(i + 1)]
    width = len(seq) * TICK + (len(seq) - 1) * TICK_GAP
    x = COL_X[i] - width / 2 - 0.06   # leave room for the trailing '...'
    first_new = True
    for j in seq:
        new = j == i
        color = style.ARENA_COLORS[j]
        ax.add_patch(Rectangle((x, TICK_Y - TICK / 2), TICK, TICK,
                               fc=color if new else style.tint(color, 0.7),
                               ec=color, lw=0.6, zorder=3))
        if new and first_new:
            ax.plot(x + TICK / 2, TICK_Y + TICK / 2 + 0.07, marker="*",
                    ms=5, color=style.INK, mec="none", zorder=3)
            first_new = False
        x += TICK + TICK_GAP
    ax.text(x + 0.02, TICK_Y, "...", ha="left", va="center",
            color=style.INK_2)
    ax.text(COL_X[i], 0.08, f"block {i + 1}", ha="center", va="center",
            color=style.INK_2)


def draw_key(ax):
    rows_top = ARENA_Y0 + ARENA_SIDE - 0.1 - KEY_PITCH * np.arange(4)
    gx, tx = KEY_X, KEY_X + 0.22

    y = rows_top[0]
    t = np.linspace(0, 1, 30)
    ax.plot(gx + 0.16 * t, y + 0.03 * np.sin(t * 4 * np.pi),
            color=style.INK_MUTED, lw=0.6)
    ax.text(tx, y, "first trial (search)", va="center", color=style.INK_2)

    y = rows_top[1]
    ax.plot([gx, gx + 0.16], [y, y], color=style.INK_2, lw=1.4)
    ax.text(tx, y, "later trial", va="center", color=style.INK_2)

    y = rows_top[2]
    ax.add_patch(Circle((gx + 0.08, y), 0.055, fc="none", ec=style.INK,
                        lw=0.8, ls=(0, (2, 1.2))))
    ax.text(tx, y, "hidden goal", va="center", color=style.INK_2)

    y = rows_top[3]
    ax.add_patch(Circle((gx + 0.08, y), 0.022, fc="white", ec=style.INK,
                        lw=0.6))
    ax.text(tx, y, "random start", va="center", color=style.INK_2)

    gray = style.INK_MUTED
    rows_bot = TICK_Y + KEY_PITCH * np.array([1, 0, -1])
    y = rows_bot[0]
    ax.plot(gx + 0.08, y, marker="*", ms=5, color=style.INK, mec="none")
    ax.text(tx, y, "goal stored", va="center", color=style.INK_2)
    y = rows_bot[1]
    ax.add_patch(Rectangle((gx + 0.08 - TICK / 2, y - TICK / 2), TICK, TICK,
                           fc=gray, ec=gray, lw=0.6))
    ax.text(tx, y, "trial in newest arena", va="center", color=style.INK_2)
    y = rows_bot[2]
    ax.add_patch(Rectangle((gx + 0.08 - TICK / 2, y - TICK / 2), TICK, TICK,
                           fc=style.tint(gray, 0.7), ec=gray, lw=0.6))
    ax.text(tx, y, "revisit", va="center", color=style.INK_2)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (goal, s_start, d_start) in enumerate(ARENAS):
        draw_arena(ax, COL_X[i], style.ARENA_COLORS[i], NAMES[i],
                   goal, s_start, d_start)
        draw_block(ax, i, n_iters={0: 4, 1: 3, 2: 2}[i])

    ax.text(ELLIPSIS_X, ARENA_Y0 + ARENA_SIDE / 2, "...", ha="center",
            va="center", fontsize=10, color=style.INK_2)
    ax.add_patch(FancyArrowPatch(
        (COL_X[0] - ARENA_SIDE / 2, 0.17), (ELLIPSIS_X, 0.17),
        arrowstyle="-|>,head_length=3,head_width=1.6", color=style.INK_MUTED,
        lw=0.5, shrinkA=0, shrinkB=0))
    ax.text(ELLIPSIS_X + 0.04, 0.17, "time", va="center", color=style.INK_2)

    draw_key(ax)
    style.save(fig, "fig1_task_schematic")


if __name__ == "__main__":
    main()
