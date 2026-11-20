"""OKLab colour helpers for the ratio heat maps (diverging blue / grey / red, log scale)."""
import math


def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gam(c):
    c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return c


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_oklab(rgb):
    r, g, b = (_lin(v) for v in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklab_to_lin(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def in_gamut(lab, eps=1e-4):
    return all(-eps <= v <= 1 + eps for v in oklab_to_lin(lab))


def oklab_to_hex(lab):
    r, g, b = (min(1.0, max(0.0, v)) for v in oklab_to_lin(lab))
    return "#" + "".join(f"{round(_gam(v) * 255):02x}" for v in (r, g, b))


def hex_to_oklab(h):
    return rgb_to_oklab(hex_to_rgb(h))


def lch(lab):
    L, a, b = lab
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def from_lch(L, C, H):
    return (L, C * math.cos(math.radians(H)), C * math.sin(math.radians(H)))


def at_lightness(hex_hue, L):
    """The hue and chroma of hex_hue moved to OKLab lightness L, chroma cut until it fits sRGB."""
    _, C, H = lch(hex_to_oklab(hex_hue))
    while C > 0 and not in_gamut(from_lch(L, C, H)):
        C -= 0.002
    return oklab_to_hex(from_lch(L, C, H))


def mix(h1, h2, t):
    a, b = hex_to_oklab(h1), hex_to_oklab(h2)
    return oklab_to_hex(tuple(x + (y - x) * t for x, y in zip(a, b)))


def luminance(h):
    r, g, b = (_lin(v) for v in hex_to_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(h1, h2):
    a, b = sorted((luminance(h1), luminance(h2)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# Palette roles (dataviz reference palette): grey midpoint per mode, blue ramp ends,
# and a red arm built at the blue end's lightness so the two arms weigh the same.
NEUTRAL = {"light": "#f0efec", "dark": "#383835"}
BLUE_END = {"light": "#184f95", "dark": "#5598e7"}
RED_HUE = "#e34948"
RED_END = {m: at_lightness(RED_HUE, hex_to_oklab(BLUE_END[m])[0]) for m in BLUE_END}
INK = {"light": ("#0b0b0b", "#ffffff"), "dark": ("#0b0b0b", "#ffffff")}
DECADES = 1.0  # the colour saturates at a ratio of 10 or 0.1


def ratio_color(ratio, mode):
    """Diverging colour for a ratio on a log scale centred on 1."""
    x = max(-1.0, min(1.0, math.log10(ratio) / DECADES))
    end = BLUE_END[mode] if x < 0 else RED_END[mode]
    return mix(NEUTRAL[mode], end, abs(x))


def ink_for(bg, mode):
    dark_ink, light_ink = INK[mode]
    return dark_ink if contrast(bg, dark_ink) >= contrast(bg, light_ink) else light_ink


if __name__ == "__main__":
    for mode in ("light", "dark"):
        print(mode, "neutral", NEUTRAL[mode], "blue", BLUE_END[mode], "red", RED_END[mode])
        for r in (0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.4, 2, 3, 5, 10):
            c = ratio_color(r, mode)
            ink = ink_for(c, mode)
            print(f"  {r:5}: {c} L={hex_to_oklab(c)[0]:.3f} ink {ink} contrast {contrast(c, ink):.1f}")
