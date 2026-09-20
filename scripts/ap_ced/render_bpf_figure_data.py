#!/usr/bin/env python3
"""Render BPF figure_spec charts with Pillow (code-drawn, not AI images).

Used after Grok stimulus-set batch download to write PNGs under generated_diagrams/
and optionally copy into skillintns/public/drawings_images/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.generation_pipeline.question_format import normalize_mcq_document

DEFAULT_DIAGRAM_DIR = ROOT / "generated_diagrams"
DEFAULT_APP_DRAWINGS = Path("/Users/sarmakompalli/skillintns/public/drawings_images")
SUBJECT = "AP Business with Personal Finance"


def _font(size: int):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(name):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_bars(draw, origin_x, origin_y, chart_w, chart_h, labels, values, max_v, colors, title, fonts):
    title_f, label_f, small_f = fonts
    draw.text((origin_x, origin_y - chart_h - 50), title, fill="black", font=title_f)
    draw.line([origin_x, origin_y, origin_x + chart_w, origin_y], fill="black", width=2)
    draw.line([origin_x, origin_y, origin_x, origin_y - chart_h], fill="black", width=2)
    n = len(values)
    if n == 0 or max_v <= 0:
        return
    bar_w = max(36, min(70, (chart_w - 40) // max(n, 1) - 16))
    gap = (chart_w - 40 - n * bar_w) // max(n, 1)
    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        x0 = origin_x + 20 + i * (bar_w + gap)
        h = int(chart_h * (float(val) / max_v))
        y0 = origin_y - h
        draw.rectangle([x0, y0, x0 + bar_w, origin_y], fill=col, outline="black")
        draw.text((x0 + 4, y0 - 22), f"{val}", fill="black", font=label_f)
        words = str(lab).split()
        lines = [" ".join(words[:2]), " ".join(words[2:])] if len(words) > 2 else [lab]
        for j, line in enumerate(lines):
            if line:
                draw.text((x0 - 2, origin_y + 8 + j * 18), str(line)[:18], fill="black", font=small_f)
    step = max(1, int(max_v) // 4)
    for tick in range(0, int(max_v) + 1, step):
        y = origin_y - int(chart_h * (tick / max_v))
        draw.line([origin_x - 5, y, origin_x, y], fill="black")
        draw.text((origin_x - 36, y - 8), str(tick), fill="black", font=small_f)


def _draw_pie(draw, cx, cy, r, labels, values, colors, title, fonts):
    title_f, label_f, _ = fonts
    draw.text((cx - r, cy - r - 40), title, fill="black", font=title_f)
    start = -90
    total = sum(float(v) for v in values) or 1
    for lab, val, col in zip(labels, values, colors):
        extent = 360 * (float(val) / total)
        draw.pieslice(
            [cx - r, cy - r, cx + r, cy + r],
            start=start,
            end=start + extent,
            fill=col,
            outline="black",
        )
        start += extent
    lx, ly = cx - r, cy + r + 20
    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        y = ly + i * 28
        draw.rectangle([lx, y, lx + 22, y + 22], fill=col, outline="black")
        pct = round(100 * float(val) / total)
        draw.text((lx + 30, y + 2), f"{lab} {pct}%", fill="black", font=label_f)


def render_figure_spec(
    figure_spec: Dict[str, Any],
    path: Path,
    *,
    unit_label: str = "",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 1200, 700
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    fonts = (_font(20), _font(16), _font(13))
    kind = figure_spec.get("kind")
    colors = [
        (70, 130, 180),
        (220, 120, 60),
        (60, 160, 100),
        (150, 100, 180),
        (200, 80, 80),
        (100, 149, 237),
    ]

    if kind == "dual_bar":
        left, right = figure_spec["left"], figure_spec["right"]
        _draw_bars(
            draw, 60, 580, 500, 380,
            left["labels"], left["values"], float(left.get("max_v") or max(left["values"])),
            colors, left["title"], fonts,
        )
        _draw_bars(
            draw, 640, 580, 500, 380,
            right["labels"], right["values"], float(right.get("max_v") or max(right["values"])),
            colors[2:], right["title"], fonts,
        )
    elif kind == "pie_bar":
        pie, bar = figure_spec["pie"], figure_spec["bar"]
        _draw_pie(draw, 280, 320, 140, pie["labels"], pie["values"], colors, pie["title"], fonts)
        _draw_bars(
            draw, 560, 560, 580, 400,
            bar["labels"], bar["values"], float(bar.get("max_v") or max(bar["values"])),
            colors, bar["title"], fonts,
        )
    elif kind == "single_bar":
        bar = figure_spec["bar"]
        _draw_bars(
            draw, 120, 580, 960, 420,
            bar["labels"], bar["values"], float(bar.get("max_v") or max(bar["values"])),
            colors, bar["title"], fonts,
        )
    else:
        raise ValueError(f"Unsupported figure_spec.kind: {kind!r}")

    footer = f"{SUBJECT}"
    if unit_label:
        footer += f" — {unit_label}"
    footer += " — code-drawn"
    draw.text((40, 660), footer, fill=(100, 100, 100), font=fonts[2])
    img.save(path)
    return path


def flatten_stimulus_payload(
    payload: Dict[str, Any],
    *,
    skill_id: int,
    skill: str,
    subject: str,
    batch_id: Optional[str],
    model_name: str,
    diagram_dir: Path,
    copy_to_app: Optional[Path],
) -> List[Dict[str, Any]]:
    """Turn one Grok stimulus_set response into dryrun_questions docs + PNG."""
    ss = payload.get("stimulus_set") or {}
    questions = payload.get("questions") or []
    if not isinstance(ss, dict) or not isinstance(questions, list):
        raise ValueError("Expected stimulus_set object and questions list")

    set_id = ss.get("stimulus_set_id") or f"bpf_skill_{skill_id}_001"
    filename = f"{set_id}.png"
    figure_spec = ss.get("figure_spec")
    if not isinstance(figure_spec, dict):
        raise ValueError(f"Missing figure_spec for {set_id}")

    out_path = diagram_dir / filename
    render_figure_spec(figure_spec, out_path, unit_label=skill)
    if copy_to_app:
        copy_to_app.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, copy_to_app / filename)

    flat: List[Dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        doc = dict(q)
        doc["subject"] = doc.get("subject") or subject
        doc["subject_area"] = doc.get("subject_area") or subject
        doc["skill"] = doc.get("skill") or skill
        doc["skill_name"] = doc.get("skill_name") or f"{doc.get('level')}-{skill}"
        doc["skill_id"] = skill_id
        doc["model_name"] = model_name
        if batch_id:
            doc["batch_id"] = batch_id
        doc["requires_diagram"] = True
        doc["stimulus_set_id"] = set_id
        doc["stimulus_text"] = ss.get("stimulus_text") or doc.get("stimulus_text")
        doc["diagram_filename"] = filename
        doc["diagram_path"] = f"generated_diagrams/{filename}"
        doc["diagram_ids"] = [filename]
        doc["figure_data"] = ss.get("figure_data")
        doc["figure_spec"] = figure_spec
        flat.append(normalize_mcq_document(doc))
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Render figure_spec from a parsed BPF JSON")
    parser.add_argument("input_json", help="JSON with questions already flattened OR stimulus payloads")
    parser.add_argument("--diagram-dir", default=str(DEFAULT_DIAGRAM_DIR))
    parser.add_argument("--copy-to-app", action="store_true")
    parser.add_argument("--app-dir", default=str(DEFAULT_APP_DRAWINGS))
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    diagram_dir = Path(args.diagram_dir)
    app_dir = Path(args.app_dir) if args.copy_to_app else None

    # Support list of questions that already have figure_spec
    questions = data["questions"] if isinstance(data, dict) and "questions" in data else data
    if not isinstance(questions, list):
        raise SystemExit("Unsupported JSON shape")

    rendered = 0
    for q in questions:
        spec = q.get("figure_spec")
        fn = q.get("diagram_filename")
        if not spec or not fn:
            continue
        render_figure_spec(spec, diagram_dir / fn, unit_label=q.get("skill") or "")
        if app_dir:
            app_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(diagram_dir / fn, app_dir / fn)
        rendered += 1
    print(json.dumps({"rendered": rendered, "diagram_dir": str(diagram_dir)}, indent=2))


if __name__ == "__main__":
    main()
