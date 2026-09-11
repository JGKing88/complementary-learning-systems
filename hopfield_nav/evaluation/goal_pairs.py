"""Pair sampler and static evaluator for the goal-conditioned control (plan §5.6).

Experiment A trains a memoryless network on i.i.d. (start, goal) pairs and
scores it on a **quadrant table**: start ∈ {train, region} × goal ∈ {train,
goal_heldout, region}, on each env set. Both the sampler that produces
training batches and the evaluator that produces the table live here, on
the same `CellSets`, so they cannot disagree about which cells are held out.

There is no env stepping anywhere in this module. Every encoding of every
cell is precomputed once per env (`EnvTensors`), and a batch is an index
gather.

Models are scored through two methods, `predict_direction(x) -> (B, 2)` and
`predict_logits(x) -> (B, 4)`. `PairRegressor` has them natively;
`RNNAgentAsPairModel` gives an `RNNAgent` the same two so Experiment B's
weights can be scored on the same table (readout 1) by the same code; and
`NearestNeighbourDecoder` implements them with no network at all, as the
line that separates "learned structure" from "learned a lookup table".

Every table entry carries a key `(env_set, start_set, goal_set, n_pairs)`.
`diff_tables` refuses to subtract entries whose keys differ (gate C20).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import RNNAgentConfig
from ..policy.agent_rnn import rnn_input_layout
from ..policy.pair_regressor import normalize_direction
from ..rollout.oracles import bfs_action_batch_continuous  # noqa: F401  (documented teacher)
from ..rollout.rnn import goal_sensory_vec, grid_state_vec, sensory_vec, xy_vec
from ..world.env import CARDINAL_ACTIONS, GridEnv
from ..world.spec import CellSets


# ---------------------------------------------------------------------------
# Per-env encodings, computed once
# ---------------------------------------------------------------------------

@dataclass
class EnvTensors:
    """Every cell's encodings for one env, as float32 numpy, indexed by cell id.

    Cell id is ``x * size + y``. ``gbook`` is None when no scaffold was built.
    """
    size: int
    gbook: np.ndarray | None        # (S*S, Ng)
    omni: np.ndarray                # (S*S, 4*obs)
    north: np.ndarray               # (S*S, obs)
    xy: np.ndarray                  # (S*S, 2)

    @staticmethod
    def build(env: GridEnv, env_offset: tuple[int, int] | None,
              sgb: np.ndarray | None) -> "EnvTensors":
        S = env.size
        cells = np.array([(x, y) for x in range(S) for y in range(S)], dtype=np.int64)
        gb = None
        if sgb is not None and env_offset is not None:
            gb = grid_state_vec(cells, env_offset, sgb)
        return EnvTensors(
            size=S, gbook=gb,
            omni=sensory_vec(env, cells, "omni"),
            north=goal_sensory_vec(env, cells, "north"),
            xy=xy_vec(cells, S),
        )

    def encoding(self, kind: str) -> np.ndarray:
        if kind == "gbook":
            if self.gbook is None:
                raise ValueError("gbook encoding requested but no scaffold was built")
            return self.gbook
        return {"omni": self.omni, "north": self.north, "xy": self.xy}[kind]


def cell_ids(cells, size: int) -> np.ndarray:
    """A set of (x, y) -> sorted int64 array of ``x * size + y``."""
    return np.array(sorted(x * size + y for x, y in cells), dtype=np.int64)


# ---------------------------------------------------------------------------
# Sampling and enumeration
# ---------------------------------------------------------------------------

def sample_pairs(cells: CellSets, starts: str, goals: str, n: int,
                 rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """``n`` pairs ``(p, g)`` as cell ids, ``p`` from one set, ``g`` from another, ``p != g``."""
    S = cells.size
    ps = cell_ids(cells.starts(starts), S)
    gs = cell_ids(cells.goals(goals), S)
    if len(ps) == 0 or len(gs) == 0:
        raise ValueError(f"empty cell set: starts={starts} ({len(ps)}), goals={goals} ({len(gs)})")
    p = ps[rng.randint(len(ps), size=n)]
    g = gs[rng.randint(len(gs), size=n)]
    # Resample collisions; with |starts ∩ goals| small this is a handful.
    bad = p == g
    while bad.any():
        g[bad] = gs[rng.randint(len(gs), size=int(bad.sum()))]
        bad = p == g
    return p, g


def enumerate_pairs(cells: CellSets, starts: str, goals: str) -> tuple[np.ndarray, np.ndarray]:
    """Every ``(p, g)`` with ``p`` in one set, ``g`` in another, ``p != g``."""
    S = cells.size
    ps = cell_ids(cells.starts(starts), S)
    gs = cell_ids(cells.goals(goals), S)
    P, G = np.meshgrid(ps, gs, indexing="ij")
    P, G = P.ravel(), G.ravel()
    keep = P != G
    return P[keep], G[keep]


# ---------------------------------------------------------------------------
# Inputs and targets
# ---------------------------------------------------------------------------

def input_kinds(cfg: RNNAgentConfig) -> list[tuple[str, str]]:
    """The layout, as ``(channel_name, encoding_kind_and_end)`` for a pair model.

    Mirrors ``rnn_input_layout`` exactly -- same order, same enabling flags --
    and says for each channel which encoding to gather and whether it is the
    start's or the goal's. Channels that only make sense in a rollout
    (``prev_action``, ``prev_reward``) are zero-filled, which is the
    first-step state every episode starts in and what makes this the same
    tensor ``build_rnn_input`` builds at ``h = 0`` (gate C6).
    """
    out: list[tuple[str, str]] = []
    if getattr(cfg, "input_sensory", True):
        mode = getattr(cfg, "sensory_mode", "ego")
        # Under "ego" the first-step view is the North view (heading resets
        # to North on placement), so the pair model reads `north`.
        out.append(("sensory", "omni:p" if mode == "omni" else "north:p"))
    if cfg.input_prev_action:
        out.append(("prev_action", "zeros:4" if cfg.movement_mode == "discrete" else "zeros:2"))
    if cfg.input_prev_reward:
        out.append(("prev_reward", "zeros:1"))
    if cfg.input_grid_state:
        out.append(("grid_state", "gbook:p"))
    gc = getattr(cfg, "goal_channel", "none")
    if gc == "abs":
        out.append(("goal_vec", "xy:g"))
    elif gc == "rel":
        out.append(("goal_vec", "xyrel"))
    if getattr(cfg, "input_xy_state", False):
        out.append(("xy_state", "xy:p"))
    if getattr(cfg, "input_goal_grid_state", False):
        out.append(("goal_grid_state", "gbook:g"))
    gs = getattr(cfg, "goal_sensory", "none")
    if gs == "omni":
        out.append(("goal_sensory", "omni:g"))
    elif gs == "north":
        out.append(("goal_sensory", "north:g"))
    return out


def pair_inputs(tensors: EnvTensors, cfg: RNNAgentConfig,
                p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``(B, D)`` float32 input for pairs of cell ids, in ``rnn_input_layout`` order."""
    parts = []
    for name, spec in input_kinds(cfg):
        if spec.startswith("zeros:"):
            parts.append(np.zeros((len(p), int(spec.split(":")[1])), dtype=np.float32))
        elif spec == "xyrel":
            parts.append(tensors.xy[g] - tensors.xy[p])
        else:
            kind, end = spec.split(":")
            parts.append(tensors.encoding(kind)[p if end == "p" else g])
    x = np.concatenate(parts, axis=1).astype(np.float32)
    # Width agreement with the layout the RNN stack would build.
    obs = tensors.north.shape[1]
    gdim = 0 if tensors.gbook is None else tensors.gbook.shape[1]
    want = sum(w for _, w in rnn_input_layout(cfg, obs, gdim))
    if x.shape[1] != want:
        raise ValueError(f"pair_inputs built width {x.shape[1]}, layout says {want}")
    return x


def pair_displacement(p: np.ndarray, g: np.ndarray, size: int) -> np.ndarray:
    """``(B, 2)`` float ``g - p`` in cell units, from cell ids."""
    px, py = p // size, p % size
    gx, gy = g // size, g % size
    return np.stack([gx - px, gy - py], axis=1).astype(np.float32)


def unit_vectors(p: np.ndarray, g: np.ndarray, size: int) -> np.ndarray:
    d = pair_displacement(p, g, size)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return (d / np.maximum(n, 1e-8)).astype(np.float32)


def optimal_action_set(p: np.ndarray, g: np.ndarray, size: int) -> np.ndarray:
    """``(B, 4)`` bool: which cardinal actions reduce Manhattan distance to ``g``.

    Wall-clipped, so at the arena edge an action that would not move is never
    optimal (it does not reduce the distance). Matches
    ``bfs_action_batch_discrete``'s candidate set exactly.
    """
    px, py = p // size, p % size
    gx, gy = g // size, g % size
    cur = np.abs(px - gx) + np.abs(py - gy)
    out = np.zeros((len(p), 4), dtype=bool)
    for a, (dx, dy) in enumerate(CARDINAL_ACTIONS):
        nx = np.clip(px + dx, 0, size - 1)
        ny = np.clip(py + dy, 0, size - 1)
        out[:, a] = (np.abs(nx - gx) + np.abs(ny - gy)) < cur
    return out


def pair_targets(p: np.ndarray, g: np.ndarray, size: int, movement_mode: str) -> np.ndarray:
    """Training targets: unit vectors, or a uniform distribution over the optimal set."""
    if movement_mode == "continuous":
        return unit_vectors(p, g, size)
    opt = optimal_action_set(p, g, size).astype(np.float32)
    return opt / np.maximum(opt.sum(axis=1, keepdims=True), 1.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-row angle in degrees between two ``(B, 2)`` direction arrays."""
    pn = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    cos = np.clip((pn * target).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def score_pairs(pred: np.ndarray, p: np.ndarray, g: np.ndarray, size: int,
                movement_mode: str) -> dict[str, float]:
    """Summary statistics for a block of predictions on pairs."""
    if movement_mode == "continuous":
        ang = angular_error_deg(pred, unit_vectors(p, g, size))
        return {"metric": float(ang.mean()), "mean_deg": float(ang.mean()),
                "median_deg": float(np.median(ang)),
                "frac_lt_30": float((ang < 30.0).mean()), "n": int(len(p))}
    opt = optimal_action_set(p, g, size)
    choice = pred.argmax(axis=1)
    hit = opt[np.arange(len(p)), choice]
    return {"metric": float(hit.mean()), "acc": float(hit.mean()), "n": int(len(p))}


# ---------------------------------------------------------------------------
# Models seen through one interface
# ---------------------------------------------------------------------------

class RNNAgentAsPairModel:
    """Give an ``RNNAgent`` the ``predict_*`` interface, at the first-step state.

    ``h = None`` (zeros), one step, the head's mean direction (continuous) or
    logits (discrete). This is readout 1 for Experiment B.
    """

    def __init__(self, agent) -> None:
        self.agent = agent
        self.movement_mode = agent.cfg.movement_mode

    # `eval_all` toggles train/eval mode on whatever it is handed.
    def eval(self):
        self.agent.eval(); return self

    def train(self, mode: bool = True):
        self.agent.train(mode); return self

    @torch.no_grad()
    def _dist(self, x: torch.Tensor):
        dist, _ = self.agent(x.unsqueeze(1), None)
        return dist

    @torch.no_grad()
    def predict_direction(self, x: torch.Tensor) -> torch.Tensor:
        dist = self._dist(x)
        mean = dist.mean[:, 0] if hasattr(dist, "mean") else dist.mode[:, 0]
        return normalize_direction(mean)

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist(x).logits[:, 0]


class NearestNeighbourDecoder:
    """Decode ``p`` and ``g`` to the nearest *training* cell, then subtract.

    The lookup-then-subtract strategy with no learning. Nearest by cosine on
    the same encoding the model reads (the ``p``-side encoding of the first
    non-zero channel). A network at this line on held-out cells has learned
    a table; above it, structure. On ``xy`` the decode is exact and this
    scores the teacher's 0° / 1.0 (gate C5).
    """

    def __init__(self, tensors: EnvTensors, cfg: RNNAgentConfig, cells: CellSets,
                 movement_mode: str) -> None:
        self.t = tensors
        self.size = tensors.size
        self.movement_mode = movement_mode
        kinds = [spec for _, spec in input_kinds(cfg) if ":" in spec and not spec.startswith("zeros")]
        p_kind = next((s.split(":")[0] for s in kinds if s.endswith(":p")), None)
        g_kind = next((s.split(":")[0] for s in kinds if s.endswith(":g")), None)
        if p_kind is None or g_kind is None:
            raise ValueError(f"cannot decode: no p-side or g-side encoding in {kinds}")
        self.p_kind, self.g_kind = p_kind, g_kind
        # The dictionary: training-start cells for p, training-goal cells for g.
        self.p_ids = cell_ids(cells.start_train, self.size)
        self.g_ids = cell_ids(cells.goal_train, self.size)

    def _decode(self, enc: np.ndarray, ids: np.ndarray, kind: str) -> np.ndarray:
        lib = self.t.encoding(kind)[ids]
        if kind == "xy":
            d = ((enc[:, None, :] - lib[None, :, :]) ** 2).sum(-1)
            return ids[d.argmin(axis=1)]
        a = enc / np.maximum(np.linalg.norm(enc, axis=1, keepdims=True), 1e-8)
        b = lib / np.maximum(np.linalg.norm(lib, axis=1, keepdims=True), 1e-8)
        return ids[(a @ b.T).argmax(axis=1)]

    def decode_pairs(self, p: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ph = self._decode(self.t.encoding(self.p_kind)[p], self.p_ids, self.p_kind)
        gh = self._decode(self.t.encoding(self.g_kind)[g], self.g_ids, self.g_kind)
        return ph, gh

    def predict_for_pairs(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        ph, gh = self.decode_pairs(p, g)
        if self.movement_mode == "continuous":
            d = pair_displacement(ph, gh, self.size)
            n = np.linalg.norm(d, axis=1, keepdims=True)
            same = n[:, 0] < 1e-8
            out = d / np.maximum(n, 1e-8)
            # Decoded onto the same cell: no direction. Emit the true one's
            # orthogonal so it scores 90°, i.e. "no information", not 0°.
            if same.any():
                u = unit_vectors(p[same], g[same], self.size)
                out[same] = np.stack([-u[:, 1], u[:, 0]], axis=1)
            return out.astype(np.float32)
        opt = optimal_action_set(ph, gh, self.size).astype(np.float32)
        # Ties inside the decoded optimal set are broken by index, as argmax
        # would; a decoded pair with an empty set (same cell) scores 0.
        return opt


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

QUADRANTS: tuple[tuple[str, str], ...] = tuple(
    (s, g) for s in CellSets.START_SETS for g in CellSets.GOAL_SETS)


def quadrant_key(env_set: str, starts: str, goals: str, n: int) -> tuple:
    return (env_set, starts, goals, int(n))


@torch.no_grad()
def _predict(model, x: np.ndarray, movement_mode: str, device, batch: int = 65536) -> np.ndarray:
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device)
        if movement_mode == "continuous":
            outs.append(model.predict_direction(xb).cpu().numpy())
        else:
            outs.append(model.predict_logits(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_pairs(
    model, tensors: EnvTensors, cfg: RNNAgentConfig, cells: CellSets, *,
    movement_mode: str, device, env_set: str = "?",
    n_per_quadrant: int | None = 4096, rng: np.random.RandomState | None = None,
    reference: bool = True,
) -> dict:
    """The six-cell quadrant table for one env.

    ``n_per_quadrant=None`` enumerates every pair in every quadrant; otherwise
    that many are sampled with ``rng``. With ``reference`` the teacher,
    uniform-random and nearest-neighbour lines are scored on the SAME pairs
    and returned beside the model (gate C11), so a metric bug shows up as the
    teacher failing.

    Returns ``{(starts, goals): {"key": ..., "model": {...}, "teacher": {...},
    "random": {...}, "nn": {...}}}``.
    """
    rng = rng if rng is not None else np.random.RandomState(0)
    S = tensors.size
    nn_dec = NearestNeighbourDecoder(tensors, cfg, cells, movement_mode) if reference else None
    out = {}
    for starts, goals in QUADRANTS:
        if len(cells.starts(starts)) == 0 or len(cells.goals(goals)) == 0:
            continue
        if n_per_quadrant is None:
            p, g = enumerate_pairs(cells, starts, goals)
        else:
            p, g = sample_pairs(cells, starts, goals, n_per_quadrant, rng)
        x = pair_inputs(tensors, cfg, p, g)
        pred = _predict(model, x, movement_mode, device)
        row = {"key": quadrant_key(env_set, starts, goals, len(p)),
               "model": score_pairs(pred, p, g, S, movement_mode)}
        if reference:
            if movement_mode == "continuous":
                teach = unit_vectors(p, g, S)
                rnd = rng.standard_normal((len(p), 2)).astype(np.float32)
            else:
                teach = optimal_action_set(p, g, S).astype(np.float32)
                rnd = rng.standard_normal((len(p), 4)).astype(np.float32)
            row["teacher"] = score_pairs(teach, p, g, S, movement_mode)
            row["random"] = score_pairs(rnd, p, g, S, movement_mode)
            row["nn"] = score_pairs(nn_dec.predict_for_pairs(p, g), p, g, S, movement_mode)
        out[(starts, goals)] = row
    return out


def aggregate_tables(tables: list[dict]) -> dict:
    """Mean of each statistic over envs, per quadrant, weighted equally per env.

    Keys must agree on ``(starts, goals)``; the env_set of the key is kept
    from the first table and ``n`` is summed.
    """
    if not tables:
        return {}
    out = {}
    for q in tables[0]:
        rows = [t[q] for t in tables if q in t]
        agg = {"key": (rows[0]["key"][0], q[0], q[1], sum(r["key"][3] for r in rows)),
               "n_envs": len(rows)}
        for who in ("model", "teacher", "random", "nn"):
            if who not in rows[0]:
                continue
            stats = {}
            for k in rows[0][who]:
                vals = [r[who][k] for r in rows]
                stats[k] = float(np.sum(vals)) if k == "n" else float(np.mean(vals))
                if k == "metric":
                    stats["metric_std"] = float(np.std(vals))
            agg[who] = stats
        out[q] = agg
    return out


def diff_tables(a: dict, b: dict, who: str = "model") -> dict:
    """``a - b`` on ``metric``, per quadrant; refuses mismatched keys (gate C20)."""
    out = {}
    for q in a:
        if q not in b:
            continue
        ka, kb = a[q]["key"], b[q]["key"]
        if ka[1:3] != kb[1:3] or ka[3] != kb[3]:
            raise ValueError(f"refusing to compare {ka} with {kb}: keys differ")
        out[q] = a[q][who]["metric"] - b[q][who]["metric"]
    return out


def format_table(agg: dict, movement_mode: str, title: str = "") -> str:
    """A fixed-width text table: rows = start set, cols = goal set, cell = model (nn)."""
    unit = "deg" if movement_mode == "continuous" else "acc"
    lines = [f"{title}  [{unit}; model  (nn-decoder)]" if title else f"[{unit}]"]
    hdr = "start\\goal".ljust(14) + "".join(g.ljust(22) for g in CellSets.GOAL_SETS)
    lines.append(hdr)
    for s in CellSets.START_SETS:
        row = s.ljust(14)
        for g in CellSets.GOAL_SETS:
            r = agg.get((s, g))
            if r is None:
                row += "-".ljust(22)
                continue
            m = r["model"]["metric"]
            nn = r.get("nn", {}).get("metric", float("nan"))
            sd = r["model"].get("metric_std")
            cell = (f"{m:6.1f}" if unit == "deg" else f"{m:6.3f}")
            if sd is not None:
                cell += (f"±{sd:4.1f}" if unit == "deg" else f"±{sd:.3f}")
            cell += (f" ({nn:5.1f})" if unit == "deg" else f" ({nn:.3f})")
            row += cell.ljust(22)
        lines.append(row)
    return "\n".join(lines)
