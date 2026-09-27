"""ACT Math course framework helpers for metadata tagging."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FRAMEWORK_PATH = Path(__file__).resolve().parent / "data" / "act_math_course_framework.json"


@lru_cache(maxsize=1)
def load_framework() -> dict[str, Any]:
    return json.loads(FRAMEWORK_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def units_by_skill_id() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for unit in load_framework().get("units") or []:
        sid = unit.get("skill_id")
        if sid is not None:
            out[int(sid)] = unit
    return out


def topic_catalog_for_skill(skill_id: int) -> list[dict[str, Any]]:
    """Topics with objectives for one reporting category."""
    unit = units_by_skill_id().get(int(skill_id))
    if not unit:
        return []
    catalog = []
    for topic in unit.get("topics") or []:
        name = str(topic.get("topic") or "").strip()
        objectives = []
        for obj in topic.get("objectives") or []:
            desc = str(obj.get("description") or "").strip()
            code = str(obj.get("code") or "").strip()
            if desc:
                objectives.append({"code": code, "description": desc})
        if name:
            catalog.append({"topic": name, "objectives": objectives})
    return catalog


def format_catalog_for_prompt(skill_id: int) -> str:
    lines: list[str] = []
    for entry in topic_catalog_for_skill(skill_id):
        lines.append(f"Topic: {entry['topic']}")
        for obj in entry["objectives"]:
            code = obj.get("code")
            prefix = f"  - [{code}] " if code else "  - "
            lines.append(f"{prefix}{obj['description']}")
    return "\n".join(lines) if lines else "(no framework topics)"


def all_objective_descriptions_for_skill(skill_id: int) -> set[str]:
    descs: set[str] = set()
    for entry in topic_catalog_for_skill(skill_id):
        for obj in entry["objectives"]:
            descs.add(obj["description"])
    return descs


def objective_maps_for_skill(skill_id: int) -> tuple[dict[str, str], dict[str, str]]:
    """code -> description, description -> description (identity)."""
    by_code: dict[str, str] = {}
    by_desc: dict[str, str] = {}
    for entry in topic_catalog_for_skill(skill_id):
        for obj in entry["objectives"]:
            code = str(obj.get("code") or "").strip()
            desc = str(obj.get("description") or "").strip()
            if code:
                by_code[code] = desc
            if desc:
                by_desc[desc] = desc
    return by_code, by_desc


def resolve_learning_objective_strings(skill_id: int, raw_values: list[Any]) -> list[str]:
    """Map model output (codes, bracketed lines, or descriptions) to framework descriptions."""
    import re

    by_code, by_desc = objective_maps_for_skill(skill_id)
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_values or []:
        text = str(raw or "").strip()
        if not text:
            continue
        m = re.match(r"^\[(ACT-[^\]]+)\]\s*(.*)$", text)
        if m:
            code, rest = m.group(1).strip(), m.group(2).strip()
            desc = by_code.get(code)
            if desc and desc not in seen:
                resolved.append(desc)
                seen.add(desc)
                continue
            if rest and rest in by_desc and rest not in seen:
                resolved.append(rest)
                seen.add(rest)
                continue
        if text in by_code:
            desc = by_code[text]
            if desc not in seen:
                resolved.append(desc)
                seen.add(desc)
            continue
        if text in by_desc and text not in seen:
            resolved.append(text)
            seen.add(text)
    return resolved


def topic_names_for_skill(skill_id: int) -> set[str]:
    return {entry["topic"] for entry in topic_catalog_for_skill(skill_id)}


def resolve_topic_name(skill_id: int, raw_topic: str) -> str | None:
    """Map model topic strings to exact framework topic names."""
    text = str(raw_topic or "").strip()
    if not text:
        return None
    names = topic_names_for_skill(skill_id)
    if text in names:
        return text
    lower_map = {n.lower(): n for n in names}
    if text.lower() in lower_map:
        return lower_map[text.lower()]
    collapsed = " ".join(text.lower().replace("&", "and").split())
    for name in names:
        name_c = " ".join(name.lower().replace("&", "and").split())
        if collapsed == name_c:
            return name
        if set(collapsed.split()) == set(name_c.split()):
            return name
    text_l = text.lower()
    partial: list[str] = []
    for name in names:
        if text_l in name.lower():
            partial.append(name)
    if len(partial) == 1:
        return partial[0]
    return None


def validate_act_math_metadata(
    skill_id: int,
    topic: str,
    learning_objectives: list[str],
) -> list[str]:
    errors: list[str] = []
    topic = str(topic or "").strip()
    topics = topic_names_for_skill(skill_id)
    if topic not in topics:
        errors.append(f"invalid_topic:{topic!r}")
    allowed = all_objective_descriptions_for_skill(skill_id)
    los = [str(x).strip() for x in (learning_objectives or []) if str(x).strip()]
    if not los:
        errors.append("empty_learning_objectives")
    for lo in los:
        if lo not in allowed:
            errors.append(f"invalid_lo:{lo[:40]!r}")
    return errors


def normalize_act_math_metadata(
    question: dict[str, Any],
    *,
    topic: str,
    learning_objectives: list[str],
) -> dict[str, Any]:
    """Apply AP-style topic / LO fields on a copy of the question."""
    q = dict(question)
    los = [str(x).strip() for x in learning_objectives if str(x).strip()]
    topic = str(topic).strip()
    q["topic"] = topic
    q["matched_topics"] = [topic]
    q["learning_objectives"] = los
    return q
