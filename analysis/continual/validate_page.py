"""Structural check on the generated page: unclosed tags, stray closes.

The artifact renderer will not tell me if a <div> never closes; it will just
lay the page out wrong. This is the cheapest way to catch that without a
browser.
"""
import sys
from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
        # SVG leaves used by the figures
        "path", "circle", "line", "rect", "polyline", "use", "stop"}


class Check(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}> at {self.getpos()}")
            return
        top, pos = self.stack.pop()
        if top != tag:
            self.errors.append(
                f"</{tag}> at {self.getpos()} closes <{top}> opened at {pos}")


src = open(sys.argv[1]).read()
c = Check()
c.feed(src)
for tag, pos in c.stack:
    c.errors.append(f"<{tag}> opened at {pos} never closed")

print(f"{len(src)} bytes, {src.count('<h2')} sections, "
      f"{src.count('<table')} tables, {src.count('<svg')} figures")
if c.errors:
    print(f"{len(c.errors)} structural problems:")
    for e in c.errors[:15]:
        print("  " + e)
    sys.exit(1)
print("structure OK")

# A few content invariants that would be silently wrong rather than broken.
# Look for values in *data* positions, not prose. "None of these are
# continual-learning methods" is a legitimate sentence; ">None<" is a leak.
import re as _re
problems = []
leaks = _re.findall(r">\s*(None|nan|NaN|inf)\s*<", src)
if leaks:
    problems.append(f"missing values rendered as data: {sorted(set(leaks))}")
if _re.search(r"(None|nan)</(td|span|strong)>", src):
    problems.append("a None/nan reached a table cell")
if src.count("var(--") < 40:
    problems.append("suspiciously few theme tokens; colours may be hardcoded")
if "prefers-color-scheme" not in src:
    problems.append("no dark-mode block")
if problems:
    print("content warnings:")
    for p in problems:
        print("  ! " + p)
else:
    print("content OK")
