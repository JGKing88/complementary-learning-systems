"""HTML assembly: the shell, the interaction layer, and the six pages.

Every page is standalone -- inline SVG, inline CSS, inline JS, no external
asset -- so one file can be emailed or dropped into a slide deck.

The interaction layer is deliberately small and generic. Each chart emits a
``<script type="application/json" class="chartdata">`` payload beside its SVG;
one ``wire()`` pass finds them and attaches either a crosshair (line charts) or
per-mark tooltips (everything else). Labels arriving from the CLI are inserted
with ``textContent``, never by string concatenation into ``innerHTML``.
"""
from __future__ import annotations

import html

from .theme import css

TABS = [
    ("index.html", "Overview"),
    ("test_a.html", "A · attractor"),
    ("test_b.html", "B · q on grid"),
    ("test_c.html", "C · continuous"),
    ("test_d.html", "D · flow"),
    ("controls.html", "Controls"),
]

JS = r"""
(function () {
  var tip = document.getElementById('tip');
  function show(x, y) {
    tip.style.opacity = '1';
    var r = tip.getBoundingClientRect();
    var nx = x + 14, ny = y + 14;
    if (nx + r.width > window.innerWidth - 8) nx = x - r.width - 14;
    if (ny + r.height > window.innerHeight - 8) ny = y - r.height - 14;
    tip.style.left = nx + 'px'; tip.style.top = ny + 'px';
  }
  function hide() { tip.style.opacity = '0'; }

  function rows(head, items) {
    tip.textContent = '';
    var h = document.createElement('div');
    h.className = 'th'; h.textContent = head; tip.appendChild(h);
    items.forEach(function (it) {
      var r = document.createElement('div'); r.className = 'tr';
      var l = document.createElement('span');
      if (it.color) {
        var k = document.createElement('i');
        k.className = 'tk'; k.style.borderTopColor = it.color;
        l.appendChild(k);
      }
      l.appendChild(document.createTextNode(it.label));
      var v = document.createElement('span');
      v.className = 'tv'; v.textContent = it.value;
      r.appendChild(l); r.appendChild(v); tip.appendChild(r);
    });
  }

  function fmt(v) {
    if (v === null || v === undefined) return '--';
    if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0))
      return v.toPrecision(3);
    return (Math.round(v * 100) / 100).toString();
  }

  document.querySelectorAll('.chartdata').forEach(function (node) {
    var d; try { d = JSON.parse(node.textContent); } catch (e) { return; }
    var fig = node.closest('.chartbox');
    if (!fig) return;
    var svg = fig.querySelector('svg');
    if (!svg) return;

    if (d.type === 'line') {
      var hit = svg.querySelector('.hit');
      var cross = svg.getElementById(d.id + '-cross');
      if (!hit) return;
      function nearest(clientX) {
        var b = svg.getBoundingClientRect();
        var sx = (clientX - b.left) / b.width * d.w;
        var best = 0, bd = Infinity;
        for (var i = 0; i < d.x.length; i++) {
          var xv = d.x[i]; if (xv === null) continue;
          var px;
          if (d.cat) { px = d.ml + i / Math.max(d.x.length - 1, 1) * d.iw; }
          else if (d.logx) {
            var lo = Math.log10(Math.max(Math.min.apply(null,
                  d.x.filter(function (v) { return v > 0; })), 1e-6));
            var hi = Math.log10(Math.max.apply(null, d.x));
            if (xv <= 0) continue;
            px = d.ml + (Math.log10(xv) - lo) / Math.max(hi - lo, 1e-9) * d.iw;
          } else {
            var x0 = Math.min.apply(null, d.x.filter(function (v) {
              return v !== null; }));
            var x1 = Math.max.apply(null, d.x.filter(function (v) {
              return v !== null; }));
            px = d.ml + (xv - x0) / Math.max(x1 - x0, 1e-9) * d.iw;
          }
          var dd = Math.abs(px - sx);
          if (dd < bd) { bd = dd; best = i; }
        }
        return best;
      }
      function move(ev) {
        var i = nearest(ev.clientX);
        var items = d.series.map(function (s) {
          return { label: s.label, color: s.color, value: fmt(s.values[i]) };
        });
        if (d.n && d.n.length > i) {
          items.push({ label: 'n', value: String(d.n[i]) });
        }
        rows((d.xlabel ? d.xlabel + ' ' : '') + fmt(d.x[i]), items);
        if (cross) {
          var b = svg.getBoundingClientRect();
          var scale = b.width / d.w;
          cross.setAttribute('opacity', '0.5');
          var px = (ev.clientX - b.left) / scale;
          cross.setAttribute('x1', px); cross.setAttribute('x2', px);
        }
        show(ev.clientX, ev.clientY);
      }
      hit.addEventListener('pointermove', move);
      hit.addEventListener('pointerleave', function () {
        hide(); if (cross) cross.setAttribute('opacity', '0');
      });
      svg.setAttribute('tabindex', '0');
      svg.addEventListener('focus', function () {
        var items = d.series.map(function (s) {
          return { label: s.label, color: s.color, value: fmt(s.values[0]) };
        });
        rows(String(d.x[0]), items);
        var b = svg.getBoundingClientRect(); show(b.left + 20, b.top + 20);
      });
      svg.addEventListener('blur', hide);
    } else if (d.type === 'grid') {
      svg.querySelectorAll('[data-c]').forEach(function (el) {
        el.addEventListener('pointermove', function (ev) {
          var ij = el.getAttribute('data-c').split(',');
          var i = +ij[0], j = +ij[1];
          var v = d.v[i] ? d.v[i][j] : null;
          var n = (d.n && d.n[i]) ? d.n[i][j] : null;
          tip.textContent = '';
          var head = document.createElement('div');
          head.className = 'th';
          head.textContent = '(' + (d.x0 + i) + ', ' + (d.y0 + j) + ')';
          tip.appendChild(head);
          var row = document.createElement('div'); row.className = 'tr';
          var lab = document.createElement('span');
          lab.textContent = d.unit || 'value';
          var val = document.createElement('span'); val.className = 'tv';
          val.textContent = (v === null || v === undefined)
            ? 'no samples' : fmt(v);
          row.appendChild(lab); row.appendChild(val); tip.appendChild(row);
          if (n) {
            var r2 = document.createElement('div'); r2.className = 'tr';
            var l2 = document.createElement('span'); l2.textContent = 'n';
            var v2 = document.createElement('span'); v2.className = 'tv';
            v2.textContent = String(n);
            r2.appendChild(l2); r2.appendChild(v2); tip.appendChild(r2);
          }
          show(ev.clientX, ev.clientY);
        });
        el.addEventListener('pointerleave', hide);
      });
      svg.querySelectorAll('[data-tip]').forEach(function (el) {
        el.addEventListener('pointermove', function (ev) {
          tip.textContent = '';
          var r = document.createElement('div');
          r.textContent = el.getAttribute('data-tip');
          tip.appendChild(r);
          show(ev.clientX, ev.clientY);
        });
        el.addEventListener('pointerleave', hide);
      });
    } else {
      svg.querySelectorAll('[data-tip]').forEach(function (el) {
        el.addEventListener('pointermove', function (ev) {
          tip.textContent = '';
          var r = document.createElement('div');
          r.textContent = el.getAttribute('data-tip');
          tip.appendChild(r);
          show(ev.clientX, ev.clientY);
        });
        el.addEventListener('pointerleave', hide);
      });
    }
  });

  // Filters scope everything below them: each control hides every element
  // whose data-<key> disagrees with the current selection.
  document.querySelectorAll('.filters select').forEach(function (sel) {
    function apply() {
      var key = sel.getAttribute('data-key');
      var val = sel.value;
      document.querySelectorAll('[data-' + key + ']').forEach(function (el) {
        el.style.display =
          (val === '*' || el.getAttribute('data-' + key) === val)
            ? '' : 'none';
      });
    }
    sel.addEventListener('change', apply);
    apply();
  });
})();
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def shell(title: str, active: str, header: str, body: str,
          source: str = "") -> str:
    tabs = "".join(
        f'<a href="{f}" class="{"on" if f == active else ""}">{esc(n)}</a>'
        for f, n in TABS)
    foot = (f'<div class="footer">Generated by '
            f'<code>analysis.hopfield_probe.report.build</code> from '
            f'<code>{esc(source)}</code>. Spec: '
            f'<code>docs/ENCODER_HOPFIELD_EVAL.md</code>; figures: '
            f'<code>docs/ENCODER_HOPFIELD_EVAL_VIZ.md</code>.</div>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{css()}</style></head>
<body>
<header class="run">{header}<nav class="tabs">{tabs}</nav></header>
<div class="wrap">{body}{foot}</div>
<div id="tip" role="status" aria-live="polite"></div>
<script>{JS}</script>
</body></html>
"""


def single_page(title: str, header: str, sections: list[tuple[str, str, str]],
                source: str = "", *, fragment: bool = False) -> str:
    """Every test on one page, tabs becoming in-page anchors.

    ``fragment=True`` emits the body only -- no doctype, html, head or body
    tags -- for hosts that supply their own document skeleton. The style and
    script still ride along, because "self-contained" has to survive that too.
    """
    tabs = "".join(f'<a href="#{esc(sid)}">{esc(name)}</a>'
                   for sid, name, _ in sections)
    body = "".join(
        f'<section id="{esc(sid)}" class="page">{content}</section>'
        for sid, _name, content in sections)
    foot = (f'<div class="footer">Generated by '
            f'<code>analysis.hopfield_probe.report.build</code> from '
            f'<code>{esc(source)}</code>. Spec: '
            f'<code>docs/ENCODER_HOPFIELD_EVAL.md</code>; figures: '
            f'<code>docs/ENCODER_HOPFIELD_EVAL_VIZ.md</code>.</div>')
    inner = (f'<header class="run">{header}'
             f'<nav class="tabs">{tabs}</nav></header>'
             f'<div class="wrap">{body}{foot}</div>'
             f'<div id="tip" role="status" aria-live="polite"></div>')
    style = f"<style>{css()}\n.page {{ scroll-margin-top: 64px; }}</style>"
    if fragment:
        return f"{style}{inner}<script>{JS}</script>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{style}</head>
<body>
{inner}
<script>{JS}</script>
</body></html>
"""


def run_header(header: dict, extra: str = "") -> str:
    ur = header.get("unique_radius") or {}
    bits = [
        f'<span class="name">{esc(header.get("label", "encoder"))}</span>',
        _kv("gain / &beta;", _num(header.get("gain"))),
        _kv("fwhm", _num(header.get("fwhm_ratio"))
            + (" (override)" if header.get("fwhm_was_overridden") else "")),
        _kv("out_dim", header.get("out_dim")),
        _kv("params", f'{(header.get("n_params") or 0) / 1e6:.2f}M'),
        _kv("&lambda;", ",".join(str(v) for v in header.get("lambdas", []))),
    ]
    if ur.get("r_min") is not None:
        bits.append(_kv("r_min / r_med",
                        f'{_num(ur.get("r_min"))} / {_num(ur.get("r_median"))}'))
    if ur.get("alias_ceiling_max") is not None:
        bits.append(_kv("alias ceiling", _num(ur.get("alias_ceiling_max"), 3)))
    if extra:
        bits.append(extra)
    return "".join(bits)


def _kv(k: str, v) -> str:
    return f'<span class="kv">{k} <b>{esc(v)}</b></span>'


def _num(v, nd=2) -> str:
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.{nd}f}".rstrip("0").rstrip(".") if f != int(f) else str(int(f))


def card(title: str, inner: str, note: str = "", attrs: str = "") -> str:
    n = f'<p class="note">{note}</p>' if note else ""
    t = f"<h3>{esc(title)}</h3>" if title else ""
    return f'<section class="card" {attrs}>{t}{n}{inner}</section>'


def facets(items: list[tuple[str, str]], attrs_fn=None) -> str:
    out = []
    for label, svg in items:
        a = attrs_fn(label) if attrs_fn else ""
        out.append(f'<div {a}><p class="facet-title">{esc(label)}</p>'
                   f'{svg}</div>')
    return f'<div class="facets">{"".join(out)}</div>'


def filter_row(controls: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
    """One row above the content it scopes. Never inside a card, never
    per-chart."""
    out = []
    for key, label, options in controls:
        opts = "".join(f'<option value="{esc(v)}">{esc(t)}</option>'
                       for v, t in options)
        out.append(f'<label>{esc(label)} <select data-key="{esc(key)}">'
                   f'{opts}</select></label>')
    return f'<div class="filters">{"".join(out)}</div>'


def banner(text: str, strong: str = "", icon: str = "&#9888;") -> str:
    return (f'<div class="banner"><span class="icon">{icon}</span>'
            f'<span><b>{esc(strong)}</b>{esc(text)}</span></div>')


__all__ = ["banner", "card", "esc", "facets", "filter_row", "run_header",
           "shell", "single_page"]
