"""Enhanced ACT Mathematics form blueprint (45 items, 50 minutes, A–D)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

FORM_ID = "act-math-enhanced-002"
FORM_QUESTION_COUNT = 45
FORM_MINUTES = 50
CHOICE_LETTERS = ("A", "B", "C", "D")

# Counts are inside the official reporting-category ranges and sum to 45.
# Preparing for Higher Math = units 1–5 = 36/45 = 80%.
# Integrating Essential Skills = 9/45 = 20%.
# Modeling is cross-cutting: 9/45 = 20%, not an extra item pool.
CATEGORY_COUNTS = (
    {"unit": "Number & Quantity", "skill_id": 320, "count": 5},
    {"unit": "Algebra", "skill_id": 321, "count": 9},
    {"unit": "Functions", "skill_id": 322, "count": 8},
    {"unit": "Geometry", "skill_id": 323, "count": 8},
    {"unit": "Statistics & Probability", "skill_id": 324, "count": 6},
    {"unit": "Integrating Essential Skills", "skill_id": 325, "count": 9},
)

FRAMEWORK_PATH = (
    Path(__file__).resolve().parent / "data" / "act_math_course_framework.json"
)


def difficulty_for_item(item_number: int) -> str:
    """ACT Math forms get harder toward the end. The easy band is short so the form is not soft."""
    if item_number <= 10:
        return "Easy"
    if item_number <= 25:
        return "Medium"
    return "Hard"


def _spread_categories() -> List[Dict[str, Any]]:
    remaining = {row["unit"]: row["count"] for row in CATEGORY_COUNTS}
    order = [row["unit"] for row in CATEGORY_COUNTS]
    meta = {row["unit"]: row for row in CATEGORY_COUNTS}
    sequence: List[Dict[str, Any]] = []
    last = None
    while sum(remaining.values()):
        candidates = [unit for unit in order if remaining[unit] > 0 and unit != last]
        if not candidates:
            candidates = [unit for unit in order if remaining[unit] > 0]
        pick = max(candidates, key=lambda unit: (remaining[unit], -order.index(unit)))
        remaining[pick] -= 1
        sequence.append(meta[pick])
        last = pick
    return sequence


def build_form_slots(form_number: int = 2) -> List[Dict[str, Any]]:
    categories = _spread_categories()
    if len(categories) != FORM_QUESTION_COUNT:
        raise RuntimeError(f"Blueprint has {len(categories)} items, expected {FORM_QUESTION_COUNT}")
    modeling_numbers = {5, 10, 15, 20, 25, 30, 35, 40, 45}
    form_id = f"act-math-enhanced-{form_number:03d}"
    slots = []
    for index, category in enumerate(categories, start=1):
        slots.append(
            {
                "form_id": form_id,
                "form_number": form_number,
                "item_number": index,
                "difficulty": difficulty_for_item(index),
                "domain": category["unit"],
                "skill": category["unit"],
                "skill_id": category["skill_id"],
                "modeling": index in modeling_numbers,
                "correct_answer": CHOICE_LETTERS[(index + form_number) % 4],
                "requires_diagram": False,
            }
        )
    _mark_diagrams(slots)
    return slots


def _mark_diagrams(slots: List[Dict[str, Any]]) -> None:
    """A fixed share of geometry, function, and statistics items carry a figure."""
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for slot in slots:
        by_domain.setdefault(slot["domain"], []).append(slot)
    chosen = set()
    chosen.update(id(slot) for slot in by_domain.get("Geometry", [])[::2])
    chosen.update(id(slot) for slot in by_domain.get("Functions", [])[::4][:2])
    stats = by_domain.get("Statistics & Probability", [])
    if stats:
        chosen.add(id(stats[len(stats) // 2]))
    for slot in slots:
        slot["requires_diagram"] = id(slot) in chosen


FORM_THEMES = {
    3: "school schedules, sports, and campus measurement",
    4: "science labs, motion, and physical measurement",
    5: "money, shopping, and travel",
    6: "design, architecture, and construction",
    7: "games, surveys, and data displays",
}


def objectives_for_domain(domain: str) -> str:
    framework = json.loads(FRAMEWORK_PATH.read_text(encoding="utf-8"))
    for unit in framework.get("units") or []:
        if unit.get("unit") != domain:
            continue
        lines = []
        for topic in unit.get("topics") or []:
            topic_name = topic.get("topic") or ""
            for obj in topic.get("objectives") or []:
                lines.append(f"- {topic_name}: {obj.get('description', '')}")
        return "\n".join(lines)
    return f"- Practice {domain}"


def format_slot_block(slot: Dict[str, Any]) -> str:
    modeling = "yes" if slot["modeling"] else "no"
    diagram = "yes" if slot.get("requires_diagram") else "no"
    diagram_rule = ""
    if slot.get("requires_diagram"):
        diagram_rule = (
            "- The stem must tell the student to use the figure. "
            "Put the given lengths, angles, coordinates, or data on figure_spec, not only in the stem.\n"
            "- figure_spec.kind is \"geometry\" or \"bars\". "
            "geometry elements use a 1000 by 700 canvas with origin at the top left: "
            "segment {x1,y1,x2,y2}, circle {cx,cy,r}, rect {x,y,w,h}, "
            "label {x,y,text}, point {x,y,text}. "
            "bars uses {title, labels, values}. Do not print the answer on the figure.\n"
        )
    return (
        f"Item {slot['item_number']}\n"
        f"- difficulty: {slot['difficulty']}\n"
        f"- reporting category: {slot['domain']}\n"
        f"- skill_id: {slot['skill_id']}\n"
        f"- modeling: {modeling}\n"
        f"- requires_diagram: {diagram}\n"
        f"{diagram_rule}"
        f"- correct_answer: {slot['correct_answer']}\n"
        f"- category skills:\n{objectives_for_domain(slot['domain'])}"
    )


def chunk_slots(slots: List[Dict[str, Any]], size: int = 3) -> List[List[Dict[str, Any]]]:
    return [slots[i : i + size] for i in range(0, len(slots), size)]
