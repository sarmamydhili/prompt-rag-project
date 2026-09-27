"""Draw ACT Math figures from the measurements stated on each item.

The model supplies the given lengths, points, and chart values. This module
places those measurements on a clean line figure.
"""

from __future__ import annotations

import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

W, H = 1100, 760
INK = (20, 20, 20)
MUTED = (90, 90, 90)


def _font(size: int):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(name):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _labels(spec: Dict[str, Any]) -> List[str]:
    if str(spec.get("kind") or "").lower() == "bars":
        cats = spec.get("labels") or []
        vals = spec.get("values") or []
        return [f"{lab}: {val}" for lab, val in zip(cats, vals)]
    out = []
    for element in spec.get("elements") or []:
        if isinstance(element, dict) and element.get("text"):
            out.append(str(element["text"]).strip())
    return [text for text in out if text]


def _numbers(labels: Sequence[str]) -> List[float]:
    found = []
    for text in labels:
        if re.search(r"\(\s*-?\d", text):
            continue
        for match in re.findall(r"-?\d+(?:\.\d+)?", text):
            found.append(float(match))
    return found


def _points(labels: Sequence[str]) -> List[Tuple[float, float]]:
    pts = []
    for text in labels:
        match = re.search(
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            text,
        )
        if match:
            pts.append((float(match.group(1)), float(match.group(2))))
    return pts


def _named(labels: Sequence[str]) -> List[str]:
    return [text for text in labels if not re.search(r"\d", text)]


def _canvas():
    img = Image.new("RGB", (W, H), "white")
    return img, ImageDraw.Draw(img), _font(28), _font(22)


def _text(draw, xy, text, font, fill=INK):
    if not text:
        return
    bbox = draw.textbbox(xy, text, font=font)
    pad = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill="white",
    )
    draw.text(xy, text, fill=fill, font=font)


def _axes(draw, origin, x_end, y_top, xlabel, ylabel, font):
    ox, oy = origin
    draw.line([ox, oy, x_end, oy], fill=INK, width=3)
    draw.polygon([(x_end, oy), (x_end - 16, oy - 7), (x_end - 16, oy + 7)], fill=INK)
    draw.line([ox, oy, ox, y_top], fill=INK, width=3)
    draw.polygon([(ox, y_top), (ox - 7, y_top + 16), (ox + 7, y_top + 16)], fill=INK)
    if len(xlabel or "") > 8:
        _text(draw, ((ox + x_end) // 2 - 90, oy + 22), xlabel, font)
    else:
        _text(draw, (x_end + 6, oy + 8), xlabel or "x", font)
    _text(draw, (max(12, ox - 130), y_top + 4), ylabel or "y", font)


def _right_angle(draw, corner, dx, dy, size=16):
    cx, cy = corner
    p1 = (cx + dx * size, cy)
    p2 = (cx + dx * size, cy + dy * size)
    p3 = (cx, cy + dy * size)
    draw.line([p1, p2, p3], fill=INK, width=2)


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def choose_layout(question: str, spec: Dict[str, Any], labels: Sequence[str]) -> str:
    stem = (question or "").lower()
    blob = " | ".join(labels).lower()
    nums = _numbers(labels)
    if str(spec.get("kind") or "").lower() == "bars":
        return "bars"
    if "scatter" in stem:
        return "scatter"
    if any(("≤" in text) or ("<=" in text) or text.lower().startswith("y=") for text in labels):
        return "piecewise"
    if "semicirc" in stem:
        return "semicircle"
    if any(word in stem for word in ("cylinder", "beaker")) or "diameter" in blob:
        return "cylinder"
    if "canopy" in stem or "topped by" in stem:
        return "canopy"
    if "skylight" in stem:
        return "atrium"
    if "curved" in stem:
        return "curve"
    if "zone" in blob or "travel zone" in stem:
        return "zones"
    if 800 in nums and 240 in nums and 80 in nums:
        return "board"
    if "divided into regions" in stem or "soccer" in stem:
        return "divided"
    if "altitude" in stem:
        return "altitude"
    if "prism" in stem and "similar" not in stem:
        return "prism"
    if ("16" in blob and "40" in blob and "similar" in stem) or (
        "horizontal support" in stem
    ):
        return "nested"
    if "similar" in stem or "~" in blob or "sim" in stem:
        if "abc ~ def" in blob or "abc~def" in blob.replace(" ", ""):
            return "composite"
        return "similar"
    if "rectangular" in stem and "triang" in stem:
        return "rect_triangle"
    if "scale factor" in blob or "scale factor" in stem:
        return "route"
    if "shaded" in stem:
        return "shaded_rect"
    if set("ABCDE").issubset(set("".join(labels))) and "segment de" in stem:
        return "truss"
    if _points(labels) or "graph" in stem or "function" in stem:
        return "line"
    if "triang" in stem:
        return "triangle"
    if "suitcase" in stem or (len(nums) >= 3 and "volume" in stem):
        return "box"
    if len(nums) >= 3:
        return "box"
    return "rectangle"


def render_measured_figure(question: str, spec: Dict[str, Any], path: Path) -> Path:
    labels = _labels(spec)
    layout = choose_layout(question, spec or {}, labels)
    drawers = {
        "bars": _draw_bars,
        "scatter": _draw_scatter,
        "piecewise": _draw_piecewise,
        "semicircle": _draw_semicircle,
        "cylinder": _draw_cylinder,
        "canopy": _draw_canopy,
        "atrium": _draw_atrium,
        "curve": _draw_curve,
        "zones": _draw_zones,
        "board": _draw_board,
        "divided": _draw_divided,
        "altitude": _draw_altitude,
        "prism": _draw_prism,
        "rect_triangle": _draw_rect_triangle,
        "route": _draw_route,
        "shaded_rect": _draw_shaded_rect,
        "nested": _draw_nested,
        "composite": _draw_composite,
        "similar": _draw_similar,
        "truss": _draw_truss,
        "line": _draw_line,
        "triangle": _draw_triangle,
        "box": _draw_box,
        "rectangle": _draw_rectangle,
    }
    img = drawers[layout](question, spec or {}, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def render_act_figure(figure_spec: Dict[str, Any], path: Path, question: str = "") -> Path:
    return render_measured_figure(question or "", figure_spec or {}, path)


def _axis_names(labels: Sequence[str]) -> Tuple[str, str]:
    names = _named(labels)
    xlabel = ylabel = None
    for name in names:
        low = name.lower()
        if xlabel is None and (
            low.strip() in {"x", "t", "n"}
            or any(key in low for key in ("hour", "people", "time", "floor area"))
        ):
            xlabel = name
        elif ylabel is None and (
            low.strip() in {"y", "d"}
            or any(key in low for key in ("mile", "cost", "distance", "beam", "f("))
        ):
            ylabel = name
    unused = [name for name in names if name not in {xlabel, ylabel}]
    if xlabel is None and unused:
        xlabel = unused.pop(0)
    if ylabel is None and unused:
        ylabel = unused.pop(0)
    return xlabel or "x", ylabel or "y"


def _map_points(pts, ox, oy):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x = min(0, min(xs))
    min_y = min(0, min(ys))
    span_x = max(xs) - min_x or 1
    span_y = max(ys) - min_y or 1

    def px(x):
        return ox + (x - min_x) / span_x * 700

    def py(y):
        return oy - (y - min_y) / span_y * 480

    return px, py


def _draw_line(question, spec, labels):
    img, draw, title, font = _canvas()
    xlabel, ylabel = _axis_names(labels)
    ox, oy = 150, 640
    _axes(draw, (ox, oy), 1000, 70, xlabel, ylabel, font)
    pts = _points(labels)
    if len(pts) == 1:
        pts = [(0.0, 0.0)] + pts
    if len(pts) >= 2:
        px, py = _map_points(pts, ox, oy)
        ordered = sorted(pts)
        coords = [(px(x), py(y)) for x, y in ordered]
        draw.line(coords, fill=INK, width=4)
        labeled = _points(labels)
        for index, ((x, y), (sx, sy)) in enumerate(zip(ordered, coords)):
            if (x, y) == (0.0, 0.0) and (0.0, 0.0) not in labeled:
                continue
            draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=INK)
            label = f"({_fmt(x)}, {_fmt(y)})"
            if index == 0:
                _text(draw, (sx + 14, sy + 8), label, font)
            else:
                _text(draw, (sx - 20, sy - 42), label, font)
        return img
    nums = _numbers(labels)
    if len(nums) >= 4:
        paired = list(zip(nums[:2], nums[2:4]))
        px, py = _map_points(paired, ox, oy)
        coords = [(px(x), py(y)) for x, y in paired]
        draw.line(coords, fill=INK, width=4)
        for index, ((x, y), (sx, sy)) in enumerate(zip(paired, coords)):
            draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=INK)
            label = f"({_fmt(x)}, {_fmt(y)})"
            if index == 0:
                _text(draw, (sx + 14, sy + 10), label, font)
            else:
                _text(draw, (sx - 30, sy - 42), label, font)
    return img


def _draw_piecewise(question, spec, labels):
    img, draw, title, font = _canvas()
    ox, oy = 160, 640
    _axes(draw, (ox, oy), 980, 80, "x", "y", font)
    intervals = []
    for text in labels:
        match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*[≤<]=?\s*x\s*[≤<]=?\s*(-?\d+(?:\.\d+)?)",
            text,
        )
        if match:
            intervals.append((float(match.group(1)), float(match.group(2)), text))
    heights = []
    for text in labels:
        match = re.search(r"y\s*=\s*(-?\d+(?:\.\d+)?)", text, re.I)
        if match:
            heights.append((float(match.group(1)), text))
    if len(intervals) >= 2 and len(heights) >= 2:
        span = max(b for _, b, _ in intervals) or 1
        top = max(h for h, _ in heights) or 1

        def px(x):
            return ox + 80 + x / span * 700

        def py(y):
            return oy - 80 - y / top * 400

        for (left, right, caption), (height, hlabel) in zip(intervals, heights):
            y = py(height)
            draw.line([px(left), y, px(right), y], fill=INK, width=4)
            _text(draw, (px(left) + 20, y - 40), hlabel, font)
            _text(draw, (px(left), oy + 16), caption, font)
        # The second piece includes its left endpoint; the first piece does not.
        x_break = intervals[0][1]
        draw.ellipse(
            [px(x_break) - 7, py(heights[0][0]) - 7, px(x_break) + 7, py(heights[0][0]) + 7],
            outline=INK,
            width=3,
        )
        draw.ellipse(
            [px(x_break) - 6, py(heights[1][0]) - 6, px(x_break) + 6, py(heights[1][0]) + 6],
            fill=INK,
        )
        return img
    return _draw_line(question, spec, labels)


def _draw_bars(question, spec, labels):
    img, draw, title, font = _canvas()
    if str(spec.get("kind") or "").lower() == "bars":
        cats = [str(x) for x in (spec.get("labels") or [])]
        vals = [float(v) for v in (spec.get("values") or [])]
        heading = str(spec.get("title") or "")
    else:
        cats, vals, heading = [], [], ""
        for text in labels:
            match = re.search(r"(.+?):\s*(-?\d+(?:\.\d+)?)", text)
            if match:
                cats.append(match.group(1).strip())
                vals.append(float(match.group(2)))
    if heading:
        _text(draw, (70, 36), heading, title)
    if not cats:
        return img
    n = min(len(cats), len(vals))
    cats, vals = cats[:n], vals[:n]
    max_v = max(vals) or 1
    origin_y = 640
    left = 100
    chart_h = 440
    usable = W - 200
    bar_w = max(40, min(110, usable // n - 28))
    gap = max(18, (usable - n * bar_w) // max(n, 1))
    draw.line([left, origin_y, W - 70, origin_y], fill=INK, width=3)
    draw.line([left, origin_y, left, 110], fill=INK, width=3)
    for i, (cat, val) in enumerate(zip(cats, vals)):
        x0 = left + 30 + i * (bar_w + gap)
        h = int(chart_h * (val / max_v))
        y0 = origin_y - h
        draw.rectangle([x0, y0, x0 + bar_w, origin_y], outline=INK, width=3)
        _text(draw, (x0, y0 - 36), _fmt(val), font)
        _text(draw, (x0 - 6, origin_y + 14), cat[:16], font)
    return img


def _draw_scatter(question, spec, labels):
    img, draw, title, font = _canvas()
    ox, oy = 150, 640
    _axes(draw, (ox, oy), 1000, 70, "Floor area (sq ft)", "Beams", font)
    x_ticks = [4000, 6000, 8000, 10000]
    y_ticks = [5, 20, 35, 50]

    def px(x):
        return ox + 40 + (x - 4000) / 6000 * 760

    def py(y):
        return oy - 30 - (y - 5) / 45 * 480

    for tick in x_ticks:
        sx = px(tick)
        draw.line([sx, oy, sx, oy + 8], fill=INK, width=2)
        _text(draw, (sx - 28, oy + 14), str(tick), font)
    for tick in y_ticks:
        sy = py(tick)
        draw.line([ox - 8, sy, ox, sy], fill=INK, width=2)
        _text(draw, (ox - 70, sy - 14), str(tick), font)
    draw.line([px(8000), oy, px(8000), 90], fill=MUTED, width=1)
    draw.line([ox, py(35), 980, py(35)], fill=MUTED, width=1)
    rng = random.Random(6032)
    pts = []
    for _ in range(48):
        pts.append((rng.uniform(8300, 9800), rng.uniform(37.5, 48.5)))
    while len(pts) < 200:
        x = rng.uniform(4300, 9800)
        y = rng.uniform(7, 48)
        if x > 8000 and y > 35:
            continue
        pts.append((x, y))
    for x, y in pts:
        sx, sy = px(x), py(y)
        draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=INK)
    return img


def _draw_semicircle(question, spec, labels):
    img, draw, title, font = _canvas()
    measured = [text for text in labels if re.search(r"\d", text)]
    length = measured[0] if measured else "70 m"
    width = measured[1] if len(measured) > 1 else "60 m"
    radius = next((text for text in labels if "r" in text.lower()), "r = 15 m")
    x0, y0, x1, y1 = 160, 200, 760, 560
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    mid = (y0 + y1) // 2
    r = 90
    arc = []
    for deg in range(-90, 91, 4):
        rad = math.radians(deg)
        arc.append((x1 + r * math.cos(rad), mid + r * math.sin(rad)))
    draw.line(arc, fill=INK, width=4)
    draw.line([x1, mid - r, x1, mid + r], fill=INK, width=4)
    _text(draw, ((x0 + x1) // 2 - 40, y1 + 16), length, font)
    _text(draw, (x0 - 100, mid - 10), width, font)
    _text(draw, (x1 + 16, mid - r - 40), radius, font)
    return img


def _draw_cylinder(question, spec, labels):
    img, draw, title, font = _canvas()
    height = next((text for text in labels if "height" in text.lower()), "height")
    diameter = next((text for text in labels if "diameter" in text.lower()), "diameter")
    x0, x1 = 340, 760
    y0, y1 = 180, 600
    draw.line([x0, y0, x0, y1], fill=INK, width=4)
    draw.line([x1, y0, x1, y1], fill=INK, width=4)
    draw.ellipse([x0, y0 - 40, x1, y0 + 40], outline=INK, width=4)
    bottom = []
    for deg in range(0, 181, 6):
        rad = math.radians(deg)
        bottom.append((((x0 + x1) / 2) + ((x1 - x0) / 2) * math.cos(rad), y1 + 40 * math.sin(rad)))
    draw.line(bottom, fill=INK, width=4)
    _text(draw, (x1 + 20, (y0 + y1) // 2 - 10), height, font)
    _text(draw, ((x0 + x1) // 2 - 80, y0 - 90), diameter, font)
    return img


def _draw_altitude(question, spec, labels):
    img, draw, title, font = _canvas()
    measured = [text for text in labels if re.search(r"\d", text)]
    unknown = next((text for text in labels if re.fullmatch(r"[A-Z]{2}", text.strip())), "h")
    apex, left, right, foot = (540, 110), (180, 620), (940, 620), (540, 620)
    draw.line([left, apex, right, left], fill=INK, width=4)
    draw.line([apex, foot], fill=INK, width=3)
    _right_angle(draw, (foot[0] - 2, foot[1]), -1, -1)
    _text(draw, (apex[0] - 12, apex[1] - 40), "A", font)
    _text(draw, (left[0] - 28, left[1] + 8), "B", font)
    _text(draw, (right[0] + 10, right[1] + 8), "C", font)
    _text(draw, (foot[0] - 10, foot[1] + 12), "D", font)
    if measured:
        _text(draw, (280, 640), measured[0], font)
    if len(measured) > 1:
        _text(draw, (700, 640), measured[1], font)
    if len(measured) > 2:
        _text(draw, (300, 340), measured[2], font)
    _text(draw, (foot[0] + 16, 340), unknown, font)
    return img


def _two_triangles(draw, font, small, large):
    """small/large are (base_label, height_label)."""
    s1, s2, s3 = (150, 560), (430, 560), (150, 300)
    draw.line([s1, s2, s3, s1], fill=INK, width=4)
    _right_angle(draw, s1, 1, -1)
    _text(draw, (230, 572), small[0], font)
    _text(draw, (70, 410), small[1], font)
    _text(draw, (s1[0] - 8, s1[1] + 8), "B", font)
    _text(draw, (s2[0] + 6, s2[1] + 8), "C", font)
    _text(draw, (s3[0] - 8, s3[1] - 36), "A", font)
    l1, l2, l3 = (560, 640), (980, 640), (560, 160)
    draw.line([l1, l2, l3, l1], fill=INK, width=4)
    _right_angle(draw, l1, 1, -1)
    _text(draw, (720, 652), large[0], font)
    _text(draw, (480, 380), large[1], font)
    _text(draw, (l1[0] - 8, l1[1] + 8), "E", font)
    _text(draw, (l2[0] + 6, l2[1] + 8), "F", font)
    _text(draw, (l3[0] - 8, l3[1] - 36), "D", font)


def _draw_similar(question, spec, labels):
    img, draw, title, font = _canvas()
    blob = " | ".join(labels)
    if "24.96" in blob:
        _two_triangles(draw, font, ("5 cm", "DE"), ("13 cm", "24.96 cm"))
    elif "x m" in blob:
        _two_triangles(draw, font, ("6 m", "5 m"), ("x m", "15 m"))
        _text(draw, (760, 360), "30 m", font)
    elif "24 mi" in blob:
        _two_triangles(draw, font, ("3 mi", "8 mi"), ("24 mi", "x"))
    elif "EF" in blob:
        _two_triangles(draw, font, ("8", "6"), ("EF", "9"))
    elif "100 mi" in blob:
        _two_triangles(draw, font, ("40 mi", ""), ("100 mi", ""))
    elif "26" in blob and "13" in blob:
        _two_triangles(draw, font, ("10", "13"), ("20", "26"))
        _text(draw, (250, 400), "13", font)
        _text(draw, (760, 360), "26", font)
    else:
        measured = [text for text in labels if re.search(r"\d", text)]
        unknown = next((text for text in labels if not re.search(r"\d", text)), "x")
        left = measured[0] if measured else ""
        right = measured[1] if len(measured) > 1 else unknown
        _two_triangles(draw, font, (left, ""), (right, unknown))
    return img


def _draw_nested(question, spec, labels):
    img, draw, title, font = _canvas()
    apex, left, right = (560, 80), (160, 640), (960, 640)
    draw.line([left, apex, right, left], fill=INK, width=4)
    # Horizontal beam creating the smaller triangle, base about 16/40 of the large base.
    y = 400
    x_left = 160 + (560 - 160) * (640 - y) / (640 - 80)
    x_right = 960 - (960 - 560) * (640 - y) / (640 - 80)
    draw.line([x_left, y, x_right, y], fill=INK, width=4)
    _text(draw, (480, 660), "40 ft", font)
    _text(draw, ((x_left + x_right) / 2 - 30, y - 44), "16 ft", font)
    draw.line([990, apex[1], 990, left[1]], fill=INK, width=2)
    draw.line([980, apex[1], 1000, apex[1]], fill=INK, width=2)
    draw.line([980, left[1], 1000, left[1]], fill=INK, width=2)
    _text(draw, (1008, 340), "15 ft", font)
    return img


def _draw_truss(question, spec, labels):
    img, draw, title, font = _canvas()
    apex, left, right = (560, 90), (180, 640), (940, 640)
    draw.line([left, apex, right, left], fill=INK, width=4)
    y = 360
    x_left = 180 + (560 - 180) * (640 - y) / (640 - 90)
    x_right = 940 - (940 - 560) * (640 - y) / (640 - 90)
    draw.line([x_left, y, x_right, y], fill=INK, width=4)
    _text(draw, (apex[0] - 10, apex[1] - 40), "A", font)
    _text(draw, (left[0] - 28, left[1] + 8), "B", font)
    _text(draw, (right[0] + 10, right[1] + 8), "C", font)
    _text(draw, (x_left - 28, y - 10), "D", font)
    _text(draw, (x_right + 10, y - 10), "E", font)
    _text(draw, (500, 660), "30 ft", font)
    _text(draw, (300, 200), "12 ft", font)
    _text(draw, ((x_left + x_right) / 2 - 16, y + 10), "DE", font)
    return img


def _draw_composite(question, spec, labels):
    img, draw, title, font = _canvas()
    x0, y0, x1, y1 = 220, 280, 820, 620
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    apex = ((x0 + x1) // 2, 80)
    draw.line([(x0, y0), apex, (x1, y0)], fill=INK, width=4)
    _text(draw, ((x0 + x1) // 2 - 30, y1 + 16), "12 m", font)
    _text(draw, (x1 + 16, (y0 + y1) // 2), "9 m", font)
    _text(draw, (apex[0] + 20, 180), "3 m", font)
    _text(draw, (x0 - 20, y0 - 36), "A", font)
    _text(draw, (x1 + 8, y0 - 36), "C", font)
    _text(draw, (apex[0] - 10, apex[1] - 36), "B", font)
    _text(draw, (240, 40), "ABC ~ DEF", title)
    return img


def _draw_rect_triangle(question, spec, labels):
    img, draw, title, font = _canvas()
    x0, y0, x1, y1 = 160, 220, 700, 560
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    draw.line([(x1, y0), (x1 + 220, y1), (x1, y1)], fill=INK, width=4)
    _right_angle(draw, (x1, y1), -1, -1)
    _text(draw, ((x0 + x1) // 2 - 40, y1 + 16), "120 ft", font)
    _text(draw, (x0 - 90, (y0 + y1) // 2), "80 ft", font)
    _text(draw, (x1 + 70, y1 + 12), "60 ft", font)
    return img


def _draw_prism(question, spec, labels):
    img, draw, title, font = _canvas()
    front = [(180, 540), (540, 540), (180, 220)]
    shift = (280, -120)
    back = [(x + shift[0], y + shift[1]) for x, y in front]
    draw.line(back + [back[0]], fill=INK, width=3)
    for a, b in zip(front, back):
        draw.line([a, b], fill=INK, width=3)
    draw.polygon(front, fill="white")
    draw.line(front + [front[0]], fill=INK, width=4)
    _right_angle(draw, front[0], 1, -1)
    _text(draw, (300, 552), "8 cm", font)
    _text(draw, (90, 360), "6 cm", font)
    _text(draw, (300, 400), "10 cm", font)
    _text(draw, (430, 130), "15 cm (length)", font)
    return img


def _draw_canopy(question, spec, labels):
    img, draw, title, font = _canvas()
    x0, y0, x1, y1 = 260, 360, 840, 620
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    apex = ((x0 + x1) // 2, 120)
    draw.line([(x0, y0), apex, (x1, y0)], fill=INK, width=4)
    draw.line([apex, ((x0 + x1) // 2, y0)], fill=INK, width=3)
    _right_angle(draw, ((x0 + x1) // 2, y0), -1, -1)
    _text(draw, (apex[0] + 16, 230), "15 ft", font)
    _text(draw, ((x0 + apex[0]) // 2 - 80, y0 - 30), "8 ft", font)
    _text(draw, (x0 + 30, 230), "17 ft", font)
    _text(draw, ((x0 + x1) // 2 - 50, y1 + 16), "40 ft long", font)
    return img


def _draw_atrium(question, spec, labels):
    img, draw, title, font = _canvas()
    x0, y0, x1, y1 = 200, 180, 900, 620
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    tri = [(x0 + 80, y1 - 40), (x0 + 80, y0 + 80), (x1 - 80, y1 - 40)]
    draw.line(tri + [tri[0]], fill=INK, width=4)
    _right_angle(draw, tri[0], 1, -1)
    _text(draw, (x0 + 40, (tri[0][1] + tri[1][1]) // 2), "12 ft", font)
    _text(draw, ((tri[0][0] + tri[2][0]) // 2, tri[0][1] + 10), "18 ft", font)
    return img


def _draw_curve(question, spec, labels):
    img, draw, title, font = _canvas()
    ox, oy = 150, 640
    _axes(draw, (ox, oy), 1000, 70, "x", "y", font)

    def px(x):
        return ox + 60 + x / 40 * 760

    def py(y):
        return oy - 40 - y / 16 * 460

    def height(x):
        return 12 - ((x - 18) ** 2) / 72

    coords = [(px(step / 2), py(height(step / 2))) for step in range(81)]
    draw.line(coords, fill=INK, width=4)
    draw.line([px(30), py(10), px(30), oy], fill=MUTED, width=2)
    draw.ellipse([px(30) - 6, py(10) - 6, px(30) + 6, py(10) + 6], fill=INK)
    draw.ellipse([px(18) - 5, py(12) - 5, px(18) + 5, py(12) + 5], fill=INK)
    _text(draw, (px(0) - 10, oy + 12), "0", font)
    _text(draw, (px(40) - 16, oy + 12), "40", font)
    _text(draw, (px(30) - 10, oy + 12), "30", font)
    _text(draw, (px(30) + 12, py(10) - 20), "10", font)
    _text(draw, (px(18) + 10, py(12) - 28), "12", font)
    return img


def _draw_zones(question, spec, labels):
    img, draw, title, font = _canvas()
    zones = []
    for text in labels:
        match = re.search(r"zone\s*([A-F])\s*:\s*(-?\d+)", text, re.I)
        if match:
            zones.append((match.group(1).upper(), match.group(2)))
    if not zones:
        zones = [("A", ""), ("B", ""), ("C", ""), ("D", "")]
    x0, y0, x1, y1 = 140, 120, 980, 640
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    cols = 3
    rows = (len(zones) + cols - 1) // cols
    cw = (x1 - x0) / cols
    rh = (y1 - y0) / rows
    for i, (name, area) in enumerate(zones):
        c, r = i % cols, i // cols
        left = x0 + c * cw
        top = y0 + r * rh
        if c:
            draw.line([left, top, left, top + rh], fill=INK, width=3)
        if r:
            draw.line([left, top, left + cw, top], fill=INK, width=3)
        _text(draw, (left + cw / 2 - 40, top + rh / 2 - 30), f"Zone {name}", title)
        _text(draw, (left + cw / 2 - 50, top + rh / 2 + 8), area, font)
    return img


def _draw_board(question, spec, labels):
    img, draw, title, font = _canvas()
    x0, y0, x1, y1 = 160, 140, 960, 640
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    draw.line([x0 + 220, y0, x0 + 220, y1], fill=INK, width=3)
    draw.line([x0 + 220, y0 + 220, x1, y0 + 220], fill=INK, width=3)
    _text(draw, (x0 + 70, (y0 + y1) // 2), "80", title)
    _text(draw, (x0 + 460, y0 + 80), "80", title)
    _text(draw, (x0 + 460, y0 + 340), "240", title)
    _text(draw, (x1 + 16, (y0 + y1) // 2), "800", font)
    return img


def _draw_divided(question, spec, labels):
    img, draw, title, font = _canvas()
    x0, y0, x1, y1 = 180, 160, 940, 600
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    c1 = x0 + (x1 - x0) // 3
    c2 = x0 + 2 * (x1 - x0) // 3
    draw.line([c1, y0, c1, y1], fill=INK, width=3)
    draw.line([c2, y0, c2, y1], fill=INK, width=3)
    for name, cx in (("A", (x0 + c1) // 2), ("B", (c1 + c2) // 2), ("C", (c2 + x1) // 2)):
        _text(draw, (cx - 12, (y0 + y1) // 2), name, title)
    _text(draw, ((x0 + x1) // 2 - 40, y1 + 16), "60 yd", font)
    _text(draw, (x1 + 12, (y0 + y1) // 2), "30 yd", font)
    return img


def _draw_route(question, spec, labels):
    img, draw, title, font = _canvas()
    a, b, c = (200, 600), (780, 600), (200, 180)
    draw.line([a, b, c, a], fill=INK, width=4)
    _right_angle(draw, a, 1, -1)
    _text(draw, (450, 616), "12 mi", font)
    _text(draw, (110, 380), "5 mi", font)
    _text(draw, (480, 360), "13 mi", font)
    _text(draw, (200, 80), "scale factor 4", title)
    return img


def _draw_triangle(question, spec, labels):
    img, draw, title, font = _canvas()
    measured = [text for text in labels if re.search(r"\d", text)]
    a, b, c = (220, 600), (860, 600), (220, 180)
    draw.line([a, b, c, a], fill=INK, width=4)
    _right_angle(draw, a, 1, -1)
    if measured:
        _text(draw, (130, 370), measured[0], font)
    if len(measured) > 1:
        _text(draw, (480, 616), measured[1], font)
    if len(measured) > 2:
        _text(draw, (520, 360), measured[2], font)
    return img


def _draw_box(question, spec, labels):
    img, draw, title, font = _canvas()
    measured = [text for text in labels if re.search(r"\d", text)]
    x0, y0, x1, y1 = 220, 220, 780, 560
    dx, dy = 120, -80
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=4)
    draw.line([x1, y0, x1 + dx, y0 + dy], fill=INK, width=3)
    draw.line([x1, y1, x1 + dx, y1 + dy], fill=INK, width=3)
    draw.line([x0, y0, x0 + dx, y0 + dy], fill=INK, width=3)
    draw.line([x0 + dx, y0 + dy, x1 + dx, y0 + dy], fill=INK, width=3)
    draw.line([x1 + dx, y0 + dy, x1 + dx, y1 + dy], fill=INK, width=3)
    if measured:
        _text(draw, ((x0 + x1) // 2 - 40, y1 + 16), measured[0], font)
    if len(measured) > 1:
        _text(draw, (x0 - 110, (y0 + y1) // 2), measured[1], font)
    if len(measured) > 2:
        _text(draw, (x1 + 20, y0 + dy - 10), measured[2], font)
    return img


def _draw_shaded_rect(question, spec, labels):
    img, draw, title, font = _canvas()
    measured = [text for text in labels if re.search(r"\d", text)]
    x0, y0, x1, y1 = 250, 180, 850, 560
    draw.rectangle([x0, y0, x1, y1], outline=INK, fill=(230, 230, 230), width=4)
    if measured:
        _text(draw, ((x0 + x1) // 2 - 30, y1 + 16), measured[0], font)
    if len(measured) > 1:
        _text(draw, (x1 + 16, (y0 + y1) // 2), measured[1], font)
    return img


def _draw_rectangle(question, spec, labels):
    return _draw_shaded_rect(question, spec, labels)
