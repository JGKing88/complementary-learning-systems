"""Build the report pages from result JSON.

    python -m analysis.hopfield_probe.report.build RESULTDIR [--out DIR]

**Recomputes nothing.** If a number is on a page it is in the JSON; a figure
that needs a quantity the tests did not save is a bug in the test layer's
schema, not something this module derives. That is what makes restyling free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..harness import OUTCOMES
from .figures import (
    grouped_bars, heatmap, line_chart, polar_bars, quiver_over_heat,
    stacked_bars, stat_tile, table,
)
from .page import (
    banner, card, esc, facets, filter_row, run_header, shell, single_page,
)
from .theme import CATEGORICAL, ordinal_colors

CAT1, CAT2, CAT3, CAT8 = (CATEGORICAL[0][0], CATEGORICAL[1][0],
                          CATEGORICAL[2][0], CATEGORICAL[7][0])
MUTED = "var(--muted)"
INK2 = "var(--ink2)"

# Outcome colours: the three in-env outcomes share the blue family (ordered by
# retr_dist), the two out-of-env ones take slots 2 and 3. Three colour families
# total, which is exactly the all-pairs budget.
OUTCOME_COLORS = ["#0d366b", "#3987e5", "#b7d3f6", CAT2, CAT3]


# ---------------------------------------------------------------------------
# small helpers over the result schema
# ---------------------------------------------------------------------------

def centers(edges: list[float]) -> list[float]:
    return [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]


def ks(res: dict) -> list[str]:
    """K values as strings, in ascending order -- the ordinal ramp's order."""
    return [str(k) for k in sorted(res["config"]["k_values"], key=int)]


def steps(res: dict) -> list[str]:
    return [str(s) for s in res["config"]["steps"]]


def k_series(res: dict, getter, *, dark=False) -> list[dict]:
    """One line per ``K``, coloured by the ordinal ramp.

    An all-``None`` series is dropped rather than drawn: it would occupy a
    legend slot and a colour while showing nothing. The real case is
    ``mean_pairwise_cos`` at ``K=1``, where the quantity is genuinely
    undefined -- one stored pattern has no other cue to be compared with.
    Colour still follows ``K``, not the surviving row number, so a series
    disappearing never repaints the others.
    """
    kk = ks(res)
    cols = ordinal_colors(len(kk), dark)
    out = []
    for i, k in enumerate(kk):
        vals = getter(k)
        if vals is None or all(v is None for v in vals):
            continue
        out.append({"label": f"K={k}", "color": cols[i], "values": vals})
    return out


def scal(node: dict, name: str, field: str = "mean"):
    s = (node or {}).get("scalars", {}).get(name)
    return None if not s else s.get(field)


def pct(v, nd=1) -> str:
    return "--" if v is None else f"{v * 100:.{nd}f}%"


def deg(v, nd=1) -> str:
    return "--" if v is None else f"{v:.{nd}f}°"


def num(v, nd=2) -> str:
    return "--" if v is None else f"{v:.{nd}f}"


def radius(v, nd=2) -> str:
    """A basin radius, with ``first_failure_radius``'s sentinel spelled out.

    ``-1`` means the condition failed at ``r=0`` -- the goal cell does not even
    retrieve itself -- which is a categorically different statement from "the
    radius is small", and printing it as ``-1.00 cells`` reads as neither.
    """
    if v is None:
        return "--"
    if v < 0:
        return "none"
    return f"{v:.{nd}f}"


def ref_k(res: dict) -> str:
    """The K a headline quotes: production runs 0-10 distractors, so 5."""
    kk = [int(k) for k in ks(res)]
    for target in (5, 3, 10):
        if target in kk:
            return str(target)
    return str(kk[len(kk) // 2])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_index(results: list[dict], src: str) -> str:
    primary = results[0]
    rk, rs = ref_k(primary), steps(primary)[0]

    def tile_for(label, fn, fmt):
        vals = [(r["header"].get("label", "?"), fn(r)) for r in results]
        head = vals[0]
        cmp = [(n, fmt(v)) for n, v in vals[1:]]
        return label, fmt(head[1]), cmp

    def basin(r):
        try:
            return scal(r["test_a"]["k"][rk]["per_step"][rs], "r_exact_95")
        except (KeyError, TypeError):
            return None

    def acc(r):
        try:
            return scal(r["test_bc"]["k"][rk]["per_step"][rs]["grid"], "acc45")
        except (KeyError, TypeError):
            return None

    def snap(r):
        try:
            return scal(r["test_bc"]["k"][rk]["per_step"][rs]["continuous"],
                        "excess_near_mean")
        except (KeyError, TypeError):
            return None

    def flow(r):
        try:
            return scal(r["test_d"]["k"][rk][rs]["discrete"], "reach_rate")
        except (KeyError, TypeError):
            return None

    tiles = []
    for label, fn, fmt, sub in (
        ("Basin radius", basin, lambda v: radius(v, 2),
         f"cells, r_exact_95 at K={rk}, s={rs}"),
        ("Direction accuracy", acc, pct, "acc45 vs. 25% chance"),
        ("Snap cost", snap, lambda v: deg(v, 2),
         "excess degrees at d < 2"),
        ("Flow", flow, pct, "discrete reach rate"),
    ):
        lab, val, cmp = tile_for(label, fn, fmt)
        tiles.append(stat_tile(lab, val, sub, cmp))

    # Capacity: K on the x-axis, so encoder identity takes the colour here.
    kk = ks(primary)
    kx = [float(k) for k in kk]
    cols = [CAT1, CAT2, MUTED]
    ex_series, acc_series = [], []
    for i, r in enumerate(results):
        name = r["header"].get("label", f"enc{i}")
        c = cols[i % len(cols)]
        ex_series.append({"label": name, "color": c, "values": [
            scal(r.get("test_a", {}).get("k", {}).get(k, {})
                 .get("per_step", {}).get(rs, {}), "exact_frac") for k in kk]})
        acc_series.append({"label": name, "color": c, "values": [
            scal(r.get("test_bc", {}).get("k", {}).get(k, {})
                 .get("per_step", {}).get(rs, {}).get("grid", {}), "acc45")
            for k in kk]})

    cap = f"""<div class="grid2">
{card("Exact retrieval vs. memory load",
      line_chart(kx, ex_series, xlabel="K (stored goals)",
                 ylabel="exact_hit", ylim=(0, 1)))}
{card("Direction accuracy vs. memory load",
      line_chart(kx, acc_series, xlabel="K (stored goals)", ylabel="acc45",
                 ylim=(0, 1), chance=0.25, chance_label="chance"))}
</div>"""

    rows = []
    for r in results:
        name = r["header"].get("label", "?")
        for k in kk:
            for s in steps(r):
                a = r.get("test_a", {}).get("k", {}).get(k, {}) \
                    .get("per_step", {}).get(s, {})
                b = r.get("test_bc", {}).get("k", {}).get(k, {}) \
                    .get("per_step", {}).get(s, {})
                d = r.get("test_d", {}).get("k", {}).get(k, {}).get(s, {})
                rows.append([
                    name, k, s,
                    radius(scal(a, "r_exact_95"), 2),
                    pct(scal(a, "exact_frac")),
                    deg(scal(b.get("grid", {}), "abs_err_mean")),
                    pct(scal(b.get("grid", {}), "acc45")),
                    deg(scal(b.get("continuous", {}), "abs_err_mean")),
                    deg(scal(b.get("continuous", {}), "excess_near_mean"), 2),
                    pct(scal(d.get("discrete", {}), "reach_rate")),
                ])
    tbl = table(
        ["encoder", "K", "steps", "r_exact_95", "exact_hit", "B |err|",
         "B acc45", "C |err|", "C excess<2", "D reach"], rows,
        summary="cross-test summary")

    body = f"""
<h1>Encoder &rarr; Hopfield readout</h1>
<p class="lede">What the navigation stack needs from an encoder is a direction
field. These four tests take <code>q</code> apart in the order the errors
compound: is the goal an attractor at all, how far off does <code>q</code>
point on the grid, what the continuous snap adds on top, and whether the
resulting field actually carries trajectories to the goal.</p>
<div class="tiles">{"".join(tiles)}</div>
<h2>Capacity</h2>
{cap}
<h2>Everything</h2>
{card("", tbl)}
"""
    return shell("Hopfield probe · overview", "index.html",
                 run_header(primary["header"],
                            _kv_extra(results)), body, src)


def _kv_extra(results: list[dict]) -> str:
    if len(results) < 2:
        return ""
    names = ", ".join(esc(r["header"].get("label", "?")) for r in results[1:])
    return f'<span class="kv">also <b>{names}</b></span>'


def page_test_a(res: dict, src: str) -> str:
    A = res["test_a"]
    kk, ss = ks(res), steps(res)
    rs = ss[0]

    fp_self = line_chart(
        [float(s) for s in ss],
        k_series(res, lambda k: [
            A["k"][k]["per_step"][s]["fixed_point"]
            .get("frac_self_consistent", {}).get("mean") for s in ss]),
        xlabel="recall steps", ylabel="frac_self_consistent", ylim=(0, 1.02))
    fp_pair = line_chart(
        [float(s) for s in ss],
        k_series(res, lambda k: [
            A["k"][k]["per_step"][s]["fixed_point"]
            .get("mean_pairwise_cos", {}).get("mean") for s in ss]),
        xlabel="recall steps", ylabel="mean_pairwise_cos", ylim=(0, 1.02))

    sf = max((A["k"][k]["per_step"][s]["fixed_point"]
              .get("sign_flip_frac", {}).get("mean") or 0.0)
             for k in kk for s in ss)
    sf_tile = stat_tile(
        "Sign flips", pct(sf),
        "recall landing on -z. Nonzero inverts q.",
        [("threshold", "0%")])

    t = A.get("tanh_arg", {})
    tanh_tile = stat_tile(
        "tanh argument", f'{t.get("abs_p99", 0):.2e}' if t else "--",
        "|beta*(Wx)| p99 over real cues",
        [("max", f'{t.get("abs_max", 0):.2e}' if t else "--"),
         ("linear below", "~1e-2")])

    def curve(field, ylab, ylim=None, chance=None, log=False):
        panels = []
        for s in ss:
            ed = A["k"][kk[0]]["per_step"][s][field]["edges"]
            xc = centers(ed)
            series = k_series(res, lambda k, s=s: A["k"][k]["per_step"][s]
                              [field]["mean"][:len(xc)])
            n = A["k"][kk[-1]]["per_step"][s][field]["n"][:len(xc)]
            panels.append((f"steps = {s}", line_chart(
                xc, series, xlabel="distance to goal (cells)", ylabel=ylab,
                ylim=ylim, chance=chance, n_per_bin=n, width=420, height=250)))
        return facets(panels, lambda lab: f'data-steps="{lab.split("= ")[1]}"')

    outcome_panels = []
    for s in ss:
        oc = A["k"][ref_k(res)]["per_step"][s]["outcome"]
        xc = centers(oc["edges"])
        outcome_panels.append((f"steps = {s}", stacked_bars(
            xc, list(OUTCOMES), oc["frac"][:len(xc)], OUTCOME_COLORS,
            xlabel="distance to goal (cells)",
            counts=[sum(r) for r in oc["counts"][:len(xc)]],
            width=430, height=250)))

    map_html = ""
    for m in A["k"][ref_k(res)].get("maps", [])[:2]:
        size = m["size"]
        rd = [[None] * size for _ in range(size)]
        oc = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                rd[i][j] = m["retr_dist"][idx]
                oc[i][j] = m["outcome"][idx]
        # In-env outcomes are drawn as a sequential ramp on retr_dist, so
        # "exact" is the darkest step and a near-miss is visibly near; the two
        # out-of-env categories get their own hues via the overlay.
        overlay = []
        for i in range(size):
            for j in range(size):
                if oc[i][j] == OUTCOMES.index("other_env"):
                    overlay.append({"cell": (i, j), "fill": CAT2,
                                    "tip": f"({i}, {j}) — other env"})
                elif oc[i][j] == OUTCOMES.index("alias"):
                    overlay.append({"cell": (i, j), "fill": CAT3,
                                    "tip": f"({i}, {j}) — alias cell"})
        map_html += card(
            f'world {m["world"]}, env {m["env"]} — retrieved cell distance',
            heatmap(rd, kind="sequential", unit="cells",
                    mark=tuple(m["goal"]), cell=14, xlabel="x", ylabel="y",
                    overlay=overlay),
            note="Blue ramp is how far the retrieved cell sits from the goal "
                 "(darkest = exact). Orange dots retrieved into another env, "
                 "aqua onto an alias cell. Ring marks the goal.")

    rbd = A["k"][ref_k(res)]["per_step"][rs].get("r_by_direction", {})
    polar = polar_bars(rbd.get("mean", []), unit="cells",
                       title=f"r_exact_all by sector (K={ref_k(res)}, s={rs})")
    polar_min = polar_bars(rbd.get("min", []), unit="cells", color=CAT2,
                           title="worst sector across worlds")

    conf = A["k"][ref_k(res)]["per_step"][rs].get("confusion", [])
    conf_html = ""
    if conf and len(conf) > 1:
        od = A["k"][ref_k(res)].get("offset_distances") or []
        order = list(range(len(conf)))
        if od:
            order = sorted(order, key=lambda i: sum(od[i]))
        m = [[float(conf[a][b]) for b in order] for a in order]
        conf_html = card(
            "Confusion — which env the other_env retrievals landed in",
            heatmap(m, kind="sequential", unit="cells", cell=12,
                    xlabel="retrieved env", ylabel="test env"),
            note="Rows and columns ordered by scaffold-offset distance, not "
                 "env index. Mass on the near-offset band is scaffold "
                 "aliasing, which more encoder capacity will not fix; a "
                 "uniform field is memory interference, which it will.")

    body = f"""
<h1>Test A &middot; attractor and basin</h1>
<p class="lede">Whether a stored goal is a fixed point of the recall dynamics,
and over what real-space disc a cue relaxes to it. Retrieval is decided against
every <em>cell</em> in the world, not against the stored goals &mdash; that is
what makes this a position readout rather than a K-way choice.</p>
{filter_row([("steps", "recall steps",
              [("*", "all")] + [(s, s) for s in ss])])}

<h2>1 &middot; Is it an attractor at all?</h2>
<div class="tiles">{sf_tile}{tanh_tile}</div>
<div class="grid2">
{card("Does a stored pattern stay on itself?", fp_self)}
{card("Do different cues land in the same place?", fp_pair,
      note="Rising toward 1 with steps is the collapse: normalised linear "
           "iteration is power iteration, so every cue converges on the same "
           "top eigenvector and per-goal attractors cannot exist.")}
</div>

<h2>2 &middot; The basin, in real space</h2>
{card("cos(recall, goal) vs. distance", curve("cos_goal", "cos_goal"),
      note="The continuous quantity underneath every binary predicate here. "
           "It degrades smoothly where exact_hit cliffs.")}
{card("exact_hit vs. distance", curve("exact", "exact_hit", (0, 1)))}
{card("retr_dist vs. distance", curve("retr_dist", "retr_dist (cells)"),
      note="exact_hit says whether retrieval was right; this says how wrong "
           "it was when it was not. NaN wherever retrieval left the test env "
           "-- across two rooms a real-space distance is not a quantity.")}

<h2>3 &middot; Outcomes</h2>
{card(f"Outcome composition by distance (K={ref_k(res)})",
      facets(outcome_panels,
             lambda lab: f'data-steps="{lab.split("= ")[1]}"'),
      note="Splitting 'near' from 'far_same_env' is the point of retr_dist: a "
           "readout consistently one cell off is a usable direction signal, "
           "one that lands across the room is not, and both were a single "
           "'miss' under a nearest-stored-goal definition.")}
{map_html}

<h2>4 &middot; Anisotropy and confusion</h2>
<div class="grid2">{card("", polar)}{card("", polar_min)}</div>
{conf_html}
"""
    return shell("Hopfield probe · Test A", "test_a.html",
                 run_header(res["header"]), body, src)


def page_test_b(res: dict, src: str) -> str:
    BC = res["test_bc"]
    kk, ss = ks(res), steps(res)
    ch = res["test_bc"]["chance"]

    def grid(k, s):
        return BC["k"][k]["per_step"][s]["grid"]

    def curve(field, ylab, ylim=None, chance=None, band=False):
        panels = []
        for s in ss:
            ed = grid(kk[0], s)[field]["edges"]
            xc = centers(ed)
            series = []
            cols = ordinal_colors(len(kk))
            for i, k in enumerate(kk):
                node = grid(k, s)[field]
                d = {"label": f"K={k}", "color": cols[i],
                     "values": node["mean"][:len(xc)]}
                if band and i == len(kk) - 1:
                    d["band"] = (node["p25"][:len(xc)], node["p75"][:len(xc)])
                series.append(d)
            panels.append((f"steps = {s}", line_chart(
                xc, series, xlabel="distance to goal (cells)", ylabel=ylab,
                ylim=ylim, chance=chance, n_per_bin=grid(kk[-1], s)[field]["n"],
                width=420, height=250)))
        return facets(panels,
                      lambda lab: f'data-steps="{lab.split("= ")[1]}"')

    step_x = [float(s) for s in ss]
    steps_chart = line_chart(
        step_x,
        k_series(res, lambda k: [scal(grid(k, s), "acc45") for s in ss]),
        xlabel="recall steps", ylabel="acc45", ylim=(0, 1),
        chance=ch["acc45"], chance_label="chance")
    steps_err = line_chart(
        step_x,
        k_series(res, lambda k: [scal(grid(k, s), "abs_err_mean")
                                 for s in ss]),
        xlabel="recall steps", ylabel="mean |err| (deg)",
        chance=ch["abs_err_deg"], chance_label="chance")

    rk = ref_k(res)
    maps = []
    for s in ss:
        g = grid(rk, s)
        m = g["map_goal_relative_signed"]
        size = (len(m["mean"]) + 1) // 2
        maps.append((f"steps = {s}", heatmap(
            m["mean"], kind="diverging", unit="deg",
            mark=(size - 1, size - 1), counts=m["n"], cell=6,
            x_origin=-(size - 1), y_origin=-(size - 1),
            xlabel="dx to goal", ylabel="dy to goal")))
    maps_abs = []
    for s in ss:
        m = grid(rk, s)["map_goal_relative_abs"]
        size = (len(m["mean"]) + 1) // 2
        maps_abs.append((f"steps = {s}", heatmap(
            m["mean"], kind="sequential", unit="deg",
            mark=(size - 1, size - 1), counts=m["n"], cell=6,
            x_origin=-(size - 1), y_origin=-(size - 1))))
    env_abs = []
    for s in ss:
        m = grid(rk, s)["map_env_absolute_abs"]
        env_abs.append((f"steps = {s}", heatmap(
            m["mean"], kind="sequential", unit="deg", counts=m["n"], cell=9,
            xlabel="x", ylabel="y")))

    ex = ""
    for m in BC["k"][rk].get("example_maps", [])[:2]:
        size = m["size"]
        grid2 = [[m["signed_err_deg"][i * size + j] for j in range(size)]
                 for i in range(size)]
        ex += card(
            f'world {m["world"]}, env {m["env"]} — single env, absolute coords',
            heatmap(grid2, kind="diverging", unit="deg",
                    mark=tuple(m["goal"]), cell=14, xlabel="x", ylabel="y"),
            note="Aggregates hide structure; one raw example is what catches a "
                 "harness bug.")

    sec = grid(rk, ss[0])
    sector = polar_bars(
        [abs(v) if v is not None else None
         for v in sec.get("sector_signed_deg", [])],
        unit="deg", color=CAT2,
        title="mean |signed err| by true-bearing sector")

    body = f"""
<h1>Test B &middot; q accuracy on the grid</h1>
<p class="lede">At every grid cell, the angle between <code>q</code> and the
true bearing to the goal. This is the floor Test C is measured against.</p>
{filter_row([("steps", "recall steps",
              [("*", "all")] + [(s, s) for s in ss])])}

<h2>1 &middot; Headline</h2>
{card("|err| vs. distance to goal", curve("abs_err", "|err| (deg)",
      chance=ch["abs_err_deg"], band=True),
      note="Band is the IQR at the largest K. Chance for a uniform bearing is "
           "90 degrees.")}

<h2>2 &middot; What the agent actually consumes</h2>
<div class="grid2">
{card("acc45 — the discrete agent's real accuracy",
      curve("acc45", "acc45", (0, 1), ch["acc45"]),
      note="Fraction of cells whose q bins to the correct cardinal under "
           "classify_direction_batch. Not recoverable from mean angle error.")}
{card("acc90 — does following q reduce distance at all?",
      curve("acc90", "acc90", (0, 1), ch["acc90"]))}
</div>

<h2>3 &middot; The steps question</h2>
<div class="grid2">
{card("acc45 vs. recall steps", steps_chart,
      note="Flat means the extra multistep channels are redundant. Falling "
           "means iteration destroys the readout. Rising then falling means "
           "steps=1 is not the optimum. Production feeds s = 1, 2, 3 to the "
           "policy.")}
{card("mean |err| vs. recall steps", steps_err)}
</div>

<h2>4 &middot; Heatmaps</h2>
{card(f"Goal-relative, signed error (K={rk})", facets(maps,
      lambda lab: f'data-steps="{lab.split("= ")[1]}"'),
      note="Primary map. Every cell is re-indexed by its offset from that "
           "env's goal, so the goal sits at the centre by construction and "
           "the average is over a single well-defined quantity. An "
           "env-absolute per-cell average would have a different goal in "
           "every sample.")}
{card(f"Goal-relative, |error| (K={rk})", facets(maps_abs,
      lambda lab: f'data-steps="{lab.split("= ")[1]}"'))}
{card(f"Env-absolute |error| (K={rk})", facets(env_abs,
      lambda lab: f'data-steps="{lab.split("= ")[1]}"'),
      note="A different question: is q worse near walls and in corners? "
           "Goal-relative coordinates average that away.")}
{ex}

<h2>5 &middot; Magnitude and bias</h2>
<div class="grid2">
{card("||q|| vs. distance", curve("qnorm", "||q||"),
      note="input_hopfield_raw feeds unnormalised q to the policy, so whether "
           "||q|| carries distance is a real question. Magnitudes are not "
           "comparable across steps -- recall saturates.")}
{card("Gram-Schmidt bias check", sector,
      note="North is kept exactly by the orthogonalisation and East is "
           "whatever survives projecting North out. A systematic lobe on the "
           "North axis is that asymmetry showing up.")}
</div>
"""
    return shell("Hopfield probe · Test B", "test_b.html",
                 run_header(res["header"]), body, src)


def page_test_c(res: dict, src: str) -> str:
    BC = res["test_bc"]
    kk, ss = ks(res), steps(res)
    ch = BC["chance"]
    rk = ref_k(res)

    def cont(k, s):
        return BC["k"][k]["per_step"][s]["continuous"]

    decomp = []
    for s in ss:
        c = cont(rk, s)
        xc = centers(c["abs_err"]["edges"])
        series = [
            {"label": "|err| continuous", "color": CAT1,
             "values": c["abs_err"]["mean"][:len(xc)]},
            {"label": "err_geom (snap ceiling)", "color": MUTED,
             "dash": "4 3", "values": c["err_geom"]["mean"][:len(xc)]},
            {"label": "excess (snap-attributable)", "color": CAT2,
             "values": c["excess"]["mean"][:len(xc)]},
        ]
        decomp.append((f"steps = {s}", line_chart(
            xc, series, xlabel="distance to goal (cells)",
            ylabel="degrees", log_x=True,
            n_per_bin=c["excess"]["n"][:len(xc)], width=430, height=280)))

    def curve(field, ylab, ylim=None, chance=None):
        panels = []
        for s in ss:
            ed = cont(kk[0], s)[field]["edges"]
            xc = centers(ed)
            series = k_series(res, lambda k, s=s: cont(k, s)[field]["mean"]
                              [:len(xc)])
            panels.append((f"steps = {s}", line_chart(
                xc, series, xlabel="distance to goal (cells)", ylabel=ylab,
                ylim=ylim, chance=chance, log_x=True,
                n_per_bin=cont(kk[-1], s)[field]["n"][:len(xc)],
                width=420, height=250)))
        return facets(panels,
                      lambda lab: f'data-steps="{lab.split("= ")[1]}"')

    sub = cont(rk, ss[0])["map_subcell_abs"]
    sub_html = heatmap(sub["mean"], kind="sequential", unit="deg",
                       counts=sub["n"], cell=3.0, xlabel="x", ylabel="y")

    cmp_rows = []
    for s in ss:
        g = BC["k"][rk]["per_step"][s]["grid"]
        c = cont(rk, s)
        cmp_rows.append([
            s, deg(scal(g, "abs_err_mean")), deg(scal(c, "abs_err_mean")),
            deg(scal(c, "excess_near_mean"), 2),
            pct(scal(g, "acc45")), pct(scal(c, "acc45")),
        ])

    body = f"""
<h1>Test C &middot; q accuracy at continuous positions</h1>
<p class="lede">Identical to Test B except the position is continuous and
reaches the encoder only through the env's <code>round()</code> snap. The true
bearing is measured from the continuous position; <code>q</code> is read at the
snapped cell. C minus B is the cost of the snap alone.</p>
{filter_row([("steps", "recall steps",
              [("*", "all")] + [(s, s) for s in ss])])}

<h2>1 &middot; The decomposition</h2>
{card(f"Where the near-goal error comes from (K={rk})",
      facets(decomp, lambda lab: f'data-steps="{lab.split("= ")[1]}"'),
      note="Three angles on one axis, log-x because the whole story is at "
           "d &lt; 3. err_geom is what a perfect readout at the snapped cell "
           "would still get wrong; excess is what the Hopfield adds on top, "
           "differenced per-sample at the cell that sample snapped to. This "
           "is what separates 'the encoder degrades near the goal' from "
           "'quantisation degrades near the goal'.")}

<h2>2 &middot; The same curves as Test B</h2>
{card("|err| vs. distance", curve("abs_err", "|err| (deg)",
      chance=ch["abs_err_deg"]))}
{card("acc45 vs. distance", curve("acc45", "acc45", (0, 1), ch["acc45"]))}

<h2>3 &middot; Sub-cell structure</h2>
{card(f"Mean |err| over the continuous plane (K={rk}, s={ss[0]})", sub_html,
      note="Eight bins per cell. The field is piecewise constant within a cell "
           "by construction -- q depends on a continuous position only through "
           "snap(p) -- and this figure's job is to show that visibly. If it is "
           "not piecewise constant, the harness is not snapping the way the "
           "env does.")}

<h2>4 &middot; B against C</h2>
{card("", table(["steps", "B |err|", "C |err|", "C excess (d&lt;2)",
                 "B acc45", "C acc45"], cmp_rows, summary="grid vs continuous"))}
<p class="note">No goal-radius masking. <code>goal_radius</code> is a
reward-shaping knob that can change between training runs, and no encoder
metric may depend on the current value of one; the near-goal region is reported
in full with its per-bin n.</p>
"""
    return shell("Hopfield probe · Test C", "test_c.html",
                 run_header(res["header"]), body, src)


def page_test_d(res: dict, src: str) -> str:
    D = res["test_d"]
    BC = res.get("test_bc", {})
    kk, ss = ks(res), steps(res)
    rk = ref_k(res)

    tiles = []
    for name in ("discrete", "continuous"):
        node = D["k"][rk][ss[0]][name]
        tiles.append(stat_tile(
            f"{name} reach", pct(scal(node, "reach_rate")),
            f"K={rk}, s={ss[0]}, arrival radius {D['arrival_radius']}",
            [("mean steps", num(scal(node, "mean_steps"), 1)),
             ("successes", num(scal(node, "n_success"), 1))]))

    panels = []
    for name, col in (("discrete", CAT1), ("continuous", CAT2)):
        for s in ss:
            node = D["k"][rk][s][name]["reach_by_dist"]
            xc = centers(node["edges"])
            series = k_series(res, lambda k, s=s, nm=name:
                              D["k"][k][s][nm]["reach_by_dist"]["mean"]
                              [:len(xc)])
            panels.append((f"{name}, steps = {s}", line_chart(
                xc, series, xlabel="start distance (cells)",
                ylabel="reach rate", ylim=(0, 1),
                n_per_bin=node["n"][:len(xc)], width=420, height=240)))

    traj = ""
    maps = BC.get("k", {}).get(rk, {}).get("example_maps", [])
    if maps:
        m = maps[0]
        size = m["size"]
        heat = [[abs(m["signed_err_deg"][i * size + j])
                 if m["signed_err_deg"][i * size + j] is not None else None
                 for j in range(size)] for i in range(size)]
        sinks = [s for s in D["k"][rk][ss[0]]["sinks"]
                 if s.get("world") == m["world"] and s.get("env") == m["env"]]
        traj = card(
            f'The field itself — world {m["world"]}, env {m["env"]}, '
            f'K={rk}, s={ss[0]}',
            quiver_over_heat(m["q"], heat, size, tuple(m["goal"]),
                             sinks=sinks, cell=18),
            note="Arrows are q, background is |err|, ring is the goal, orange "
                 "discs are spurious sinks sized by basin. A sink is a "
                 "location and no aggregate can point at one -- but the sink "
                 "cells and basins are recorded as numbers by flow.py, and "
                 "this illustrates them rather than being how they were "
                 "found.")

    rows = []
    for s in ss:
        for sk in D["k"][rk][s]["sinks"][:20]:
            rows.append([s, sk.get("world", "-"), sk.get("env", "-"),
                         str(sk["cells"][0]), sk["basin"],
                         num(sk["dist_from_goal"], 1)])
        for lc in D["k"][rk][s]["limit_cycles"][:10]:
            rows.append([s, lc.get("world", "-"), lc.get("env", "-"),
                         f'cycle len {len(lc["cells"])}', lc["basin"],
                         num(lc["dist_from_goal"], 1)])
    inv = table(["steps", "world", "env", "location", "basin", "dist to goal"],
                rows, summary="sink and limit-cycle inventory") if rows else \
        '<p class="note">No spurious sinks or limit cycles recorded.</p>'

    body = f"""
<h1>Test D &middot; flow</h1>
<p class="lede">Angle error per cell is a local statistic; whether the
trajectories it induces arrive is a global property of the field. No policy and
no encoder calls &mdash; this consumes the <code>q</code> field Test B already
built. The goal is absorbing: an agent that arrives stops, rather than the
field needing a fixed point there.</p>
{filter_row([("steps", "recall steps",
              [("*", "all")] + [(s, s) for s in ss])])}
<div class="tiles">{"".join(tiles)}</div>
<p class="note">Mean steps sits beside its success count by construction:
computed over a shrinking success set it is the classic trap, since a field
that only succeeds from nearby posts an excellent one.</p>

<h2>1 &middot; Reach rate vs. start distance</h2>
{card("", facets(panels,
      lambda lab: f'data-steps="{lab.split("= ")[1]}"'))}

<h2>2 &middot; The field</h2>
{traj}

<h2>3 &middot; Sink inventory</h2>
{card("", inv)}
"""
    return shell("Hopfield probe · Test D", "test_d.html",
                 run_header(res["header"]), body, src)


def page_controls(res: dict, src: str) -> str:
    C = res["controls"]
    ref = C["reference"]

    ed = C["oracle"]["abs_err"]["edges"]
    xc = centers(ed)
    three = line_chart(
        xc,
        [{"label": "Hopfield", "color": CAT1,
          "values": C["hopfield_reference"]["abs_err"]["mean"][:len(xc)]},
         {"label": "oracle (perfect recall)", "color": INK2, "dash": "5 3",
          "values": C["oracle"]["abs_err"]["mean"][:len(xc)]},
         {"label": "local oracle (one cell)", "color": INK2, "dash": "2 3",
          "values": C["local_oracle"]["abs_err"]["mean"][:len(xc)]}],
        xlabel="distance to goal (cells)", ylabel="|err| (deg)",
        chance=90.0, chance_label="chance",
        n_per_bin=C["oracle"]["abs_err"]["n"][:len(xc)],
        width=560, height=320)

    swap = grouped_bars(
        ["mean |err|"],
        [{"label": "original basis", "color": CAT1,
          "values": [scal(C["oracle"], "abs_err_mean")]},
         {"label": "swapped order", "color": CAT2,
          "values": [scal(C["gram_schmidt_swapped"], "abs_err_mean")]}],
        ylabel="degrees", width=380, height=260)

    tanh = grouped_bars(
        ["mean |err|"],
        [{"label": "tanh (production)", "color": CAT1,
          "values": [scal(C["hopfield_reference"], "abs_err_mean")]},
         {"label": "linear (use_tanh=False)", "color": CAT2,
          "values": [scal(C["no_tanh"], "abs_err_mean")]}],
        ylabel="degrees", width=380, height=260)

    em = C["empty_memory"]
    naive_acc = (em.get("naive_acc45") or {}).get("mean")
    leak_ok = naive_acc is None or naive_acc < 0.35
    empty_tile = stat_tile(
        "Empty memory", pct(naive_acc),
        "acc45 with K=0 — must sit at chance (25%)",
        [("verdict", "no leak" if leak_ok else "LEAK"),
         ("max |q| naive", num((em.get("naive_max_abs_q") or {}).get("mean"),
                               4)),
         ("max |q| production", "0.0000")])

    t = res.get("test_a", {}).get("tanh_arg", {})
    hist_html = ""
    if t:
        edges = t["hist_edges"]
        xc2 = [(edges[i] * edges[i + 1]) ** 0.5
               for i in range(len(edges) - 1)]
        hist_html = card(
            "Distribution of |beta * (W x)| over real cues",
            line_chart(xc2, [{"label": "count", "color": CAT1,
                              "values": [float(v) for v in t["hist"]]}],
                       xlabel="|beta * (W x)|", ylabel="count", log_x=True,
                       width=560, height=260),
            note=f'p99 {t["abs_p99"]:.2e}, max {t["abs_max"]:.2e}. tanh is '
                 f'linear to well under a percent below ~1e-2, so this is the '
                 f'evidence for or against the claim that the nonlinearity is '
                 f'inert at this operating point -- and beta differs 27x '
                 f'between the encoders under test, so it is measured rather '
                 f'than assumed.')

    body = f"""
<h1>Controls</h1>
<p class="lede">Without these, none of Tests A&ndash;D is attributable. All at
the reference operating point K={ref['k']}, steps={ref['steps']}.</p>

<h2>1 &middot; The three-way gap</h2>
{card("Hopfield vs. oracle vs. local oracle", three,
      note="The oracle is a CEILING under the same projection, not a ground "
           "truth: z_goal - z_c is a large displacement and W is a local "
           "tangent frame, so it degrades with distance for reasons unrelated "
           "to the encoder. It is the right control because the Hopfield path "
           "has the identical pathology. Local-oracle error is the basis "
           "itself; oracle minus local is manifold curvature; Hopfield minus "
           "oracle is recall.")}

<h2>2 &middot; Basis and dynamics</h2>
<div class="grid2">
{card("Gram-Schmidt order swap", swap,
      note="Orthogonalise East-first instead of North-first. A large "
           "difference means the reported angles are substantially an "
           "artifact of the basis construction.")}
{card("Linear control", tanh,
      note="If use_tanh=False changes nothing, beta is doing nothing at this "
           "operating point and recall is a linear readout.")}
</div>

<h2>3 &middot; Leak check</h2>
<div class="tiles">{empty_tile}</div>
<p class="note">The production path short-circuits on <code>num_memories ==
0</code> and emits an all-zero signal. Recall through an all-zero W does not:
normalising the zero vector gives zero, so <code>recalled - current</code> is
<code>-current</code> and q points <em>away</em> from where the agent stands.
Both are measured, because the gap is exactly the leak this control exists to
catch.</p>

<h2>4 &middot; Is the tanh doing anything?</h2>
{hist_html}
"""
    return shell("Hopfield probe · controls", "controls.html",
                 run_header(res["header"]), body, src)


def page_rescue(res: dict, src: str) -> str:
    R = res["rescue"]
    rows = R["rows"]
    by_bs: dict[float, dict] = {}
    for r in rows:
        by_bs.setdefault(r["beta_scale"], {})[
            (r["zero_diag"], r["alpha"])] = r
    xs = sorted(by_bs)

    def chart(field, ylab):
        series = []
        cols = [CAT1, CAT2, CAT3, CAT8, MUTED, INK2]
        combos = sorted({(r["zero_diag"], r["alpha"]) for r in rows})
        for i, cb in enumerate(combos):
            series.append({
                "label": f"zero_diag={cb[0]}, alpha={cb[1]}",
                "color": cols[i % len(cols)],
                "values": [by_bs[x].get(cb, {}).get(field) for x in xs],
            })
        return line_chart(xs, series, xlabel="beta * scale", ylabel=ylab,
                          log_x=True, ylim=(0, 1.02), width=560, height=300)

    succ_rows = [[r["zero_diag"], r["alpha"], f'{r["scale"]:.3g}',
                  f'{r["beta"]:.3g}', r["k"],
                  num(r["frac_self_consistent"], 3),
                  num(r["mean_pairwise_cos"], 3)]
                 for r in R["success"]]
    succ = (table(["zero_diag", "alpha", "scale", "beta", "K",
                   "self-consistent", "pairwise cos"], succ_rows,
                  summary=f'{R["n_success"]} settings met both criteria')
            if succ_rows else
            '<p class="note">No setting in the sweep produced per-goal '
            'attractor behaviour.</p>')

    body = f"""
<h1>Rescue sweep</h1>
{banner(esc(R["note"]), "Not the production operating point.")}
<p class="lede">Whether the Hopfield layer is fixable at all. Only the product
<code>beta * scale</code> reaches the argument of the tanh, so that is the
x-axis; <code>scale</code> is the main suspect, since it is what makes
<code>||W x||</code> about 1e-3 and therefore what makes the tanh inert.</p>
<div class="grid2">
{card("Does a pattern stay on itself?", chart("frac_self_consistent",
      "frac_self_consistent"))}
{card("Do different cues collapse together?", chart("mean_pairwise_cos",
      "mean_pairwise_cos"))}
</div>
<p class="note">Success is <b>both</b>: high self-consistency <em>and</em> low
pairwise cosine. Either alone is meaningless &mdash; a single global attractor
scores a perfect self-consistency at K=1.</p>
<h2>Settings that worked</h2>
{card("", succ)}
"""
    return shell("Hopfield probe · rescue", "rescue.html",
                 run_header(res["header"]), body, src)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(result_dir: Path, out_dir: Path | None = None) -> Path:
    result_dir = Path(result_dir)
    out_dir = Path(out_dir) if out_dir else result_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        files = [result_dir / e["file"] for e in manifest["encoders"]]
    else:
        files = sorted(p for p in result_dir.glob("*.json")
                       if p.name != "manifest.json")
    if not files:
        raise SystemExit(f"no result JSON in {result_dir}")

    results = [json.loads(p.read_text()) for p in files]
    src = str(result_dir)
    primary = results[0]

    pages = {"index.html": page_index(results, src)}
    if "test_a" in primary:
        pages["test_a.html"] = page_test_a(primary, src)
    if "test_bc" in primary:
        pages["test_b.html"] = page_test_b(primary, src)
        pages["test_c.html"] = page_test_c(primary, src)
    if "test_d" in primary:
        pages["test_d.html"] = page_test_d(primary, src)
    if "controls" in primary:
        pages["controls.html"] = page_controls(primary, src)
    if "rescue" in primary:
        pages["rescue.html"] = page_rescue(primary, src)

    for name, html_text in pages.items():
        (out_dir / name).write_text(html_text)

    # One file with everything, tabs becoming in-page anchors. The multi-page
    # set stays the default -- test_c alone is large enough that stacking it
    # with the rest is a slower first paint -- but a single file is what
    # travels, and it is what a host with its own document skeleton can take.
    sections = []
    for name, html_text in pages.items():
        sid = name.replace(".html", "")
        label = dict((f, n) for f, n in _TAB_NAMES).get(name, sid)
        sections.append((sid, label, _body_of(html_text)))
    hdr = run_header(primary["header"], _kv_extra(results))
    (out_dir / "report.html").write_text(
        single_page("Hopfield probe", hdr, sections, src))
    (out_dir / "report.fragment.html").write_text(
        single_page("Hopfield probe", hdr, sections, src, fragment=True))
    return out_dir


_TAB_NAMES = [
    ("index.html", "Overview"), ("test_a.html", "A · attractor"),
    ("test_b.html", "B · q on grid"), ("test_c.html", "C · continuous"),
    ("test_d.html", "D · flow"), ("controls.html", "Controls"),
    ("rescue.html", "Rescue"),
]


def _body_of(page: str) -> str:
    """The content of a rendered page, without its shell.

    Cheaper and less brittle than threading a `fragment` flag through every
    page builder: the pages are generated here, so the markers are known
    exactly rather than parsed out of arbitrary HTML.
    """
    start = page.find('<div class="wrap">')
    end = page.rfind('<div class="footer">')
    if start < 0 or end < 0:
        return page
    return page[start + len('<div class="wrap">'):end]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = build(Path(a.result_dir), Path(a.out) if a.out else None)
    print(f"pages: {out}")
    for f in sorted(out.glob("*.html")):
        print(f"  {f}  ({f.stat().st_size // 1024} KB)")
    print(f"\nopen: {out / 'index.html'}")
    print(f"one file: {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
