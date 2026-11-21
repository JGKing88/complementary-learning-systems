"""Render the literature report markdown as one self-contained artifact page.

    python build_report_page.py REPORT.md OUT.html

Handles exactly what the report uses: #/##/### headings, paragraphs, pipe
tables, **bold**, [text](url) links (URLs may contain parentheses), and the
verification tags such as [full text] or [abstract; inference], which become
chips. Status columns in tables become chips too.
"""
import html
import re
import sys

import heatmap

KINDS =["full text", "abstract", "secondary", "unverified", "preprint", "derived", "inference", "figure-read"]
CHIP_CLASS = {"full text": "ft", "abstract": "ab", "secondary": "se", "unverified": "un",
              "preprint": "pp", "derived": "de", "inference": "in", "figure-read": "de", "mixed": "mx"}
TAG_RE = re.compile(r"\[(" + "|".join(KINDS) + r")([^\]\[]*)\]")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def links(text):
    """Replace [label](url) with anchors, matching parentheses inside the URL."""
    out, i = [], 0
    while True:
        j = text.find("](", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        k = text.rfind("[", i, j)
        if k < 0:
            out.append(text[i:j + 2])
            i = j + 2
            continue
        depth, p = 1, j + 2
        while p < len(text) and depth:
            if text[p] == "(":
                depth += 1
            elif text[p] == ")":
                depth -= 1
            p += 1
        url = text[j + 2:p - 1]
        label = text[k + 1:j]
        if not url.startswith("http"):
            out.append(text[i:p])
        else:
            out.append(text[i:k])
            out.append(f'<a href="{url.replace(chr(34), "%22")}" target="_blank" rel="noopener">{label}</a>')
        i = p


def chip(kind, rest=""):
    cls = CHIP_CLASS.get(kind, "mx")
    return f'<span class="chip {cls}">{kind}{rest}</span>'


def inline(text):
    t = html.escape(text, quote=False)
    t = links(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = TAG_RE.sub(lambda m: chip(m.group(1), m.group(2)), t)
    return t


def status_cell(text):
    low = text.strip().lower()
    for k in KINDS + ["mixed"]:
        if low.startswith(k):
            return chip(k, html.escape(text.strip()[len(k):], quote=False))
    return inline(text)


def table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    status_cols = {i for i, h in enumerate(head) if h.lower() in ("status", "how firm")}
    th = "".join(f"<th>{inline(h)}</th>" for h in head)
    trs = []
    for r in body:
        tds = "".join(f"<td>{status_cell(c) if i in status_cols else inline(c)}</td>"
                      for i, c in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    return (f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def render(md):
    lines = md.split("\n")
    title, blocks, toc, para, i = "", [], [], [], 0
    used = set()
    grids, cur_h3, cur_sid, grid_sids = [], "", "", {}

    def unique_id(text):
        """Heading ids must be unique; region headings such as CA3 repeat across sections."""
        base = sid = slug(text)
        n = 2
        while sid in used:
            sid, n = f"{base}-{n}", n + 1
        used.add(sid)
        return sid

    def flush():
        if para:
            text = " ".join(para).strip()
            cls = ""
            if text.startswith("**Full citations.**"):
                cls = ' class="cites"'
            elif text.startswith("Sources"):
                cls = ' class="sources"'
            blocks.append(f"<p{cls}>{inline(text)}</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            flush()
            tl = []
            while i < len(lines) and lines[i].startswith("|"):
                tl.append(lines[i])
                i += 1
            if tl[0].strip().startswith("| Recording ("):
                # A width / grid-spacing grid: drawn as a heat map at the top of the page.
                grid = heatmap.parse(tl, cur_h3)
                grids.append(grid)
                grid_sids.setdefault(grid["kind"], cur_sid)
                title_id = heatmap.KINDS[grid["kind"]]["title_id"]
                what = cur_h3.lower() + (" kernel" if grid["kind"] == "kernel" else "")
                blocks.append(f'<p class="hm-pointer">The {html.escape(what)} grid is drawn as a '
                              f'heat map at the <a href="#{title_id}">top of the page</a>.</p>')
            else:
                blocks.append(table(tl))
            continue
        if ln.startswith("### "):
            flush()
            t = ln[4:].strip()
            cur_h3 = t
            blocks.append(f'<h3 id="{unique_id(t)}">{inline(t)}</h3>')
        elif ln.startswith("## "):
            flush()
            t = ln[3:].strip()
            sid = unique_id(t)
            cur_sid = sid
            toc.append((sid, t))
            blocks.append(f'</section><section class="sec" aria-labelledby="{sid}"><h2 id="{sid}">{inline(t)}</h2>')
        elif ln.startswith("# "):
            flush()
            title = ln[2:].strip()
        elif not ln.strip():
            flush()
        else:
            para.append(ln)
        i += 1
    flush()
    return title, blocks, toc, grids, grid_sids


def main():
    src, out = sys.argv[1], sys.argv[2]
    title, blocks, toc, grids, grid_sids = render(open(src, encoding="utf-8").read())
    counts_sid = next((s for s, t in toc if t.startswith("Fields per cell per grid period")), None)
    heat = heatmap.block_html(grids, grid_sids, counts_sid)
    # The report's first two paragraphs are the overall summary and the tag key.
    first_sec = next(k for k, b in enumerate(blocks) if b.startswith("</section>"))
    summary, legend_p = blocks[0], blocks[1]
    body = "".join(blocks[first_sec:])[len("</section>"):] + "</section>"
    q1, q2 = toc[0][0], toc[1][0]
    nav = "".join(f'<li><a href="#{s}">{html.escape(t)}</a></li>' for s, t in toc)
    page = TEMPLATE.format(h1=html.escape(title), summary=summary, legend=legend_p,
                           body=body, nav=nav, q1=q1, q2=q2, heatmaps=heat)
    open(out, "w", encoding="utf-8").write(page)
    n_chips, n_links, n_tables = page.count('class="chip'), page.count("<a href="), page.count("<table>")
    print(f"{out}: {len(page.encode()) / 1e3:.0f} KB, {len(toc)} sections, "
          f"{n_chips} chips, {n_links} links, {n_tables} tables")


TEMPLATE = """<title>Place Fields and Population Similarity</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
/* Layout: contents rail on the left, one reading column; two answer cards open the page. */
:root {{
  --bg: #f5f6f8;
  --surface: #ffffff;
  --ink: #171a21;
  --muted: #5c6474;
  --line: #dde1e8;
  --accent: #4f4296;
  --accent-soft: #ece9f8;
  --amber: #7a4f00; --amber-bg: #faefd6;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #111318; --surface: #191c23; --ink: #e6e8ee; --muted: #9aa2b2; --line: #2b303a;
    --accent: #b4a8f0; --accent-soft: #262240; --amber: #f2c66d; --amber-bg: #33280f;
    color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #111318; --surface: #191c23; --ink: #e6e8ee; --muted: #9aa2b2; --line: #2b303a;
  --accent: #b4a8f0; --accent-soft: #262240; --amber: #f2c66d; --amber-bg: #33280f;
  color-scheme: dark;
}}
html {{ scroll-behavior: smooth; }}
@media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
body {{ background: var(--bg); color: var(--ink); font: 17px/1.62 var(--serif); }}
a {{ color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }}
.page {{
  max-width: 1280px; margin: 0 auto;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 32px 64px;
  display: grid; gap: 32px; grid-template-columns: minmax(0, 1fr);
}}
@media (min-width: 1100px) {{
  .page {{ grid-template-columns: 250px minmax(0, 1fr); align-items: start; }}
  nav.toc {{ position: sticky; top: calc(20px + env(safe-area-inset-top, 0px)); max-height: calc(100vh - 40px); overflow-y: auto; }}
}}
nav.toc {{ font: 13px/1.4 var(--sans); }}
nav.toc .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }}
nav.toc ol {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }}
nav.toc a {{ display: block; padding: 4px 8px; border-radius: 4px; color: var(--muted); text-decoration: none; }}
nav.toc a:hover {{ color: var(--ink); background: var(--surface); }}
nav.toc a.on {{ color: var(--accent); background: var(--accent-soft); }}
main {{ min-width: 0; display: grid; gap: 28px; }}
header.top {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 500 12px/1.3 var(--sans); text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
h1 {{ font: 600 2.05rem/1.18 var(--sans); letter-spacing: -0.015em; margin: 0; text-wrap: balance; max-width: 26ch; }}
h2 {{ font: 600 1.42rem/1.25 var(--sans); letter-spacing: -0.01em; margin: 0 0 4px; text-wrap: balance; }}
h3 {{ font: 600 1.08rem/1.3 var(--sans); margin: 8px 0 0; text-wrap: balance; }}
p {{ margin: 0; max-width: 72ch; }}
strong {{ font-weight: 600; }}
.answers {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
.answer {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; display: grid; gap: 8px; align-content: start; }}
.answer .q {{ font: 500 13px/1.35 var(--sans); color: var(--muted); }}
.answer .a {{ font: 600 1.15rem/1.3 var(--sans); }}
.answer p {{ font-size: 15.5px; }}
.answer .more {{ font: 500 13px/1.3 var(--sans); }}
.summary {{ display: grid; gap: 14px; }}
.legend {{ font-size: 15px; color: var(--muted); }}
section.sec {{ display: grid; gap: 14px; padding-top: 28px; border-top: 1px solid var(--line); scroll-margin-top: 16px; }}
section.sec:first-of-type {{ border-top: 0; padding-top: 0; }}
h2, h3 {{ scroll-margin-top: 16px; }}
p.cites, p.sources {{ font: 13px/1.55 var(--sans); color: var(--muted); max-width: none; }}
p.cites strong {{ color: var(--ink); }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font: 13.5px/1.45 var(--sans); font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; vertical-align: top; padding: 9px 12px; border-bottom: 1px solid var(--line); }}
th {{ font-weight: 600; font-size: 12px; color: var(--muted); background: var(--bg); position: sticky; top: 0; }}
tbody tr:last-child td {{ border-bottom: 0; }}
td {{ min-width: 9ch; }}
.chip {{ display: inline-block; font: 500 10.5px/1.5 var(--mono); padding: 0 6px; border-radius: 4px; white-space: nowrap; vertical-align: 1px; border: 1px solid transparent; }}
.chip.ft {{ color: var(--ink); background: var(--bg); border-color: var(--line); }}
.chip.ab {{ color: var(--muted); border-color: var(--muted); }}
.chip.se {{ color: var(--muted); border: 1px dashed var(--muted); }}
.chip.un {{ color: var(--amber); background: var(--amber-bg); }}
.chip.pp {{ color: var(--accent); background: var(--accent-soft); }}
.chip.de, .chip.in, .chip.mx {{ color: var(--muted); font-style: italic; }}
footer {{ font: 13px/1.5 var(--sans); color: var(--muted); border-top: 1px solid var(--line); padding-top: 18px; }}

/* Heat maps: field width over grid spacing. Cell colours are computed per mode (--bl/--tl light,
   --bd/--td dark) from a diverging blue / grey / red scale on log10(ratio), centred on 1. */
.hm-block {{ display: grid; gap: 28px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 20px 20px 18px; }}
.hm-group-block {{ display: grid; gap: 14px; }}
.hm-group-block + .hm-group-block {{ border-top: 1px solid var(--line); padding-top: 24px; }}
.hm-block h2 {{ margin: 0; }}
.hm-sub {{ font: 14.5px/1.55 var(--sans); color: var(--muted); max-width: 88ch; }}
.hm-legend {{ display: flex; flex-wrap: wrap; gap: 10px 28px; align-items: flex-end; font: 12px/1.3 var(--sans); color: var(--muted); }}
.hm-scale {{ flex: 1 1 320px; max-width: 520px; display: grid; gap: 4px; }}
.hm-bar {{ height: 12px; border-radius: 3px; background: var(--gl); }}
.hm-ticks {{ position: relative; height: 15px; font-variant-numeric: tabular-nums; }}
.hm-ticks span {{ position: absolute; transform: translateX(-50%); }}
.hm-ticks span:first-child {{ transform: none; }}
.hm-ticks span:last-child {{ transform: translateX(-100%); }}
.hm-ends {{ display: flex; justify-content: space-between; gap: 12px; }}
.hm-nckey {{ display: flex; align-items: center; gap: 8px; }}
.hm-nckey i {{ width: 22px; height: 14px; border-radius: 3px; border: 1px solid var(--line);
  background: repeating-linear-gradient(135deg, transparent 0 4px, var(--line) 4px 5px); }}
.hm-figs {{ display: grid; gap: 22px; }}
.hm-fig {{ margin: 0; display: grid; gap: 8px; }}
.hm-fig figcaption {{ font: 600 15px/1.3 var(--sans); }}
.hm-wrap {{ overflow-x: auto; }}
table.hm {{ border-collapse: separate; border-spacing: 2px; width: 100%; font: 500 12.5px/1.3 var(--sans); font-variant-numeric: tabular-nums; }}
table.hm th, table.hm td {{ border: 0; padding: 7px 9px; vertical-align: middle; position: static; }}
table.hm thead th {{ background: transparent; color: var(--ink); font: 600 12.5px/1.25 var(--sans); text-align: center; white-space: nowrap; vertical-align: bottom; }}
table.hm thead th span, table.hm tbody th span {{ display: block; font-weight: 400; color: var(--muted); font-size: 11.5px; }}
table.hm thead th.hm-corner {{ text-align: left; }}
table.hm tbody th {{ text-align: left; font-weight: 400; color: var(--ink); background: var(--surface); position: sticky; left: 0; z-index: 1; min-width: 190px; max-width: 280px; }}
table.hm tbody th b {{ font-weight: 600; }}
table.hm tbody th span.w {{ color: var(--ink); }}
table.hm tr.hm-group th {{ position: static; padding: 10px 2px 2px; font: 600 11px/1.2 var(--sans); text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); background: transparent; }}
table.hm td {{ text-align: center; white-space: nowrap; border-radius: 4px; min-width: 9ch; }}
table.hm td.v {{ background: var(--bl); color: var(--tl); cursor: default; }}
table.hm td.v:hover, table.hm td.v:focus-visible {{ outline: 2px solid var(--ink); outline-offset: -2px; }}
table.hm td.nc {{ color: var(--muted); font-weight: 400; font-size: 11.5px; border: 1px solid var(--line);
  background: repeating-linear-gradient(135deg, transparent 0 6px, var(--line) 6px 7px); }}
table.hm td.nc span {{ background: var(--surface); padding: 0 3px; border-radius: 2px; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) table.hm td.v {{ background: var(--bd); color: var(--td); }}
  :root:not([data-theme="light"]) .hm-bar {{ background: var(--gd); }}
}}
:root[data-theme="dark"] table.hm td.v {{ background: var(--bd); color: var(--td); }}
:root[data-theme="dark"] .hm-bar {{ background: var(--gd); }}
.hm-tip {{ position: fixed; z-index: 10; pointer-events: none; max-width: 340px; display: grid; gap: 2px;
  background: var(--surface); color: var(--muted); border: 1px solid var(--line); border-radius: 8px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.14); padding: 9px 11px; font: 12.5px/1.4 var(--sans); }}
.hm-tip strong {{ color: var(--ink); font-size: 15px; font-variant-numeric: tabular-nums; }}
.hm-tip[hidden] {{ display: none; }}
p.hm-pointer {{ font: 14px/1.5 var(--sans); color: var(--muted); }}
</style>

<div class="page">
  <nav class="toc" aria-label="Contents">
    <div class="label">Contents</div>
    <ol>{nav}</ol>
  </nav>
  <main>
    <header class="top">
      <div class="eyebrow">Literature review · real neural recordings only · November 2026</div>
      <h1>{h1}</h1>
    </header>

    {heatmaps}

    <div class="answers">
      <div class="answer">
        <div class="q">Is there evidence for very large place fields?</div>
        <div class="a">Yes in metres; in ventral CA3 one field can span most of a track</div>
        <p>Bat CA1 fields reach 32 m in a 200 m tunnel, about 17% of the flight path. In ventral rat CA3 on an 18 m track, fields averaged 5.5 m and one drawn field ran about 15.7 m, roughly 87% of the track (read from the figure). No single dorsal field covers a third of an environment larger than about 6 m or 2 m². Broad tuning also appears in marmoset hippocampus and in the subiculum's diffuse firing.</p>
        <a class="more" href="#{q1}">The evidence</a>
      </div>
      <div class="answer">
        <div class="q">Does population similarity fall off across a large part of an environment?</div>
        <div class="a">Smoothly in one study, in ventral rat CA3; anterior marmoset hippocampus shows one block per track arm</div>
        <p>On an 18 m track, ventral CA3 population-vector correlation fell to 0.5 about 3.3 m from a position and reached zero near 10.5 m, about 64% of the 16.5 m analysed; intermediate CA3 reached zero near 7 m. The vectors pool cells across rats, and no one has replicated it. In anterior marmoset hippocampus on a 4.26 m L-shaped track, correlation fell to 0.5 about 1.2 to 1.3 m away and stayed near +0.3 beyond, but the high values form one block per arm, bounded by the corner; the vectors pool units across sessions and the values are read from a figure. Every dorsal rodent curve with a field width from the same study falls to half within about one field width, including dorsal CA3 on the 18 m track.</p>
        <a class="more" href="#{q2}">The evidence</a>
      </div>
    </div>

    <div class="summary">{summary}<div class="legend">{legend}</div></div>

    {body}

    <footer>Compiled on 15 November 2026 from eleven sets of research notes covering primary papers, supplements, abstracts and dataset landing pages, then updated from full texts: Kjelstrup 2008 and its supplement, Brun 2008, Jung, Wiener &amp; McNaughton 1994, Terrazas 2005, Muller &amp; Kubie 1987, O'Keefe &amp; Burgess 1996, Sharp &amp; Green 1994, Olson 2017, Lever 2009, Mainali 2025 and a Maurer &amp; Nadel review. The field-width over grid-spacing tables were added on 20 November 2026 from four further sets of notes and an independent audit. On 21 November 2026 three audited sweeps of rat and mouse field widths, field counts and population kernels replaced the per-study grid rows with pooled rows and added the kernel grids and the fields-per-period table. Each chip says how a value was checked: full text, abstract only, a secondary source, unverified, a preprint, read off a figure, or arithmetic derived from reported numbers. A markdown copy of this report is kept in the repository.</footer>
  </main>
</div>

<script>
(() => {{
  const links = [...document.querySelectorAll("nav.toc a")];
  const byId = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
  const heads = [...document.querySelectorAll("section.sec h2")];
  if (!("IntersectionObserver" in window) || !heads.length) return;
  const vis = new Set();
  const io = new IntersectionObserver(entries => {{
    for (const e of entries) (e.isIntersecting ? vis.add(e.target.id) : vis.delete(e.target.id));
    let cur = null;
    for (const h of heads) {{ if (vis.has(h.id)) {{ cur = h.id; break; }} }}
    if (!cur) {{
      for (const h of heads) {{ if (h.getBoundingClientRect().top < 80) cur = h.id; }}
    }}
    links.forEach(a => a.classList.toggle("on", a === byId.get(cur)));
  }}, {{ rootMargin: "0px 0px -60% 0px" }});
  heads.forEach(h => io.observe(h));
}})();
(() => {{
  // Heat-map tooltip: value first, then the midpoint, the row, the column and the row's studies.
  const tip = document.querySelector(".hm-tip");
  if (!tip) return;
  const line = (text, tag) => {{ const el = document.createElement(tag || "span"); el.textContent = text; return el; }};
  const show = td => {{
    const parts = [line(td.dataset.r, "strong"), line("Geometric midpoint " + td.dataset.m),
                   line(td.dataset.row), line("over " + td.dataset.col)];
    if (td.dataset.s) parts.push(line(td.dataset.s));
    tip.replaceChildren(...parts);
    tip.hidden = false;
    const r = td.getBoundingClientRect(), t = tip.getBoundingClientRect();
    let x = r.left + r.width / 2 - t.width / 2, y = r.top - t.height - 8;
    if (y < 8) y = r.bottom + 8;
    x = Math.max(8, Math.min(x, window.innerWidth - t.width - 8));
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }};
  const hide = () => {{ tip.hidden = true; }};
  for (const td of document.querySelectorAll("table.hm td.v")) {{
    td.addEventListener("pointerenter", () => show(td));
    td.addEventListener("pointerleave", hide);
    td.addEventListener("focus", () => show(td));
    td.addEventListener("blur", hide);
  }}
  window.addEventListener("scroll", hide, {{ passive: true }});
}})();
</script>
"""

if __name__ == "__main__":
    main()
