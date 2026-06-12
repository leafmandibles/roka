"""Annotation-based test reimplementation of RK frame helpers."""

try:
    from .node_api import node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def _rk_parse_aspect_ratio(value):
    import re

    text = str(value or "1:1").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"Invalid aspect ratio '{value}'. Use W:H, e.g. 4:3.")
    w = float(match.group(1))
    h = float(match.group(2))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid aspect ratio '{value}'. Values must be positive.")
    return w, h


def _rk_snap_int(value, snap):
    snap = max(1, int(snap or 1))
    return max(snap, int(round(float(value) / snap) * snap))


def _rk_aspect_label(width, height):
    import math
    from fractions import Fraction

    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    gcd = math.gcd(width, height)
    simple_w, simple_h = width // gcd, height // gcd
    if simple_w <= 100 and simple_h <= 100:
        return f"{simple_w}:{simple_h}"
    frac = Fraction(width, height).limit_denominator(100)
    return f"{frac.numerator}:{frac.denominator}"


@node("roka/resolution/RK_Frame", returns=("width", "height"))
def frame(
    MP: float = 1.0,
    AspectRatio: str = "1:1",
    snap: int = 2,
) -> tuple[int, int]:
    import math

    ar_w, ar_h = _rk_parse_aspect_ratio(AspectRatio)
    area = max(1.0, float(MP) * 1_000_000.0)
    width = math.sqrt(area * (ar_w / ar_h))
    height = width / (ar_w / ar_h)
    return (_rk_snap_int(width, snap), _rk_snap_int(height, snap))


@node("roka/resolution/RK_AspectRatio", returns=("MP", "aspect_ratio"))
def aspect_ratio(
    width: int = 1024,
    height: int = 1024,
) -> tuple[float, str]:
    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    mp = (width * height) / 1_000_000.0
    return (round(mp, 6), _rk_aspect_label(width, height))


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
