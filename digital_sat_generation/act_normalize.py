"""Normalize ACT LLM question payloads to digital_sat_rw_questions shape."""

from __future__ import annotations

import re
from typing import Any, Dict

from digital_sat_generation.act_schemas import ACT_DOMAIN_MYSQL_MAP
from digital_sat_generation.act_stimulus_ui import fix_stimulus_for_sat_rw_ui


def _coerce_stimulus(stimulus: Any, shared_passage: str | None = None) -> Dict[str, Any]:
    if isinstance(stimulus, str):
        text = stimulus.strip()
        return {"format": "single_text", "text": text}
    if not isinstance(stimulus, dict):
        text = (shared_passage or "").strip()
        return {"format": "single_text", "text": text}

    if not stimulus.get("text") and not stimulus.get("passage_text"):
        for key in ("content", "body", "passage"):
            if stimulus.get(key):
                stimulus = {**stimulus, "passage_text": stimulus[key]}
                break

    if stimulus.get("text"):
        out = dict(stimulus)
        fixed, _ = fix_stimulus_for_sat_rw_ui(out)
        return fixed

    if stimulus.get("type") == "single_text" and isinstance(stimulus.get("sentences"), list):
        parts = []
        sentence_objs = []
        for i, s in enumerate(stimulus["sentences"], 1):
            if isinstance(s, str):
                parts.append(s)
                sentence_objs.append({"sentence_number": i, "text": s})
            elif isinstance(s, dict):
                txt = str(s.get("text") or "")
                parts.append(txt)
                sentence_objs.append(
                    {"sentence_number": s.get("sentence_number", i), "text": txt}
                )
        text = " ".join(p for p in parts if p).strip()
        if text:
            out = dict(stimulus)
            out["format"] = "single_text"
            out["text"] = text
            out["sentences"] = sentence_objs
            out.pop("type", None)
            return out

    nested = stimulus.get("single_text")
    if isinstance(nested, str) and nested.strip():
        out = dict(stimulus)
        out["format"] = "single_text"
        out["text"] = nested.strip()
        out.pop("single_text", None)
        return out
    if isinstance(nested, dict):
        sentences = nested.get("sentences")
        if isinstance(sentences, list):
            parts = []
            for s in sentences:
                if isinstance(s, str):
                    parts.append(s)
                elif isinstance(s, dict):
                    parts.append(str(s.get("text") or ""))
            passage_text = " ".join(p for p in parts if p).strip()
        else:
            passage_text = str(nested.get("text") or "").strip()
        if passage_text:
            out = dict(stimulus)
            out["format"] = "single_text"
            out["text"] = passage_text
            out.pop("single_text", None)
            return out

    passage_text = stimulus.get("passage_text") or shared_passage or ""
    out = dict(stimulus)
    out["text"] = passage_text
    out.pop("passage_text", None)
    fixed, _ = fix_stimulus_for_sat_rw_ui(out)
    if not fixed.get("format"):
        fixed["format"] = "single_text"
    return fixed


def normalize_act_question(
    question: Dict[str, Any],
    *,
    shared_passage: str | None = None,
    passage_topic: str | None = None,
    passage_set_id: str | None = None,
) -> Dict[str, Any]:
    q = dict(question)
    domain = str(q.get("domain") or "").strip()
    skill = str(q.get("skill") or "").strip()
    if domain not in ACT_DOMAIN_MYSQL_MAP:
        for key in ACT_DOMAIN_MYSQL_MAP:
            if key.lower() == domain.lower():
                domain = key
                break
    if skill.startswith("ACT-") or skill not in ACT_DOMAIN_MYSQL_MAP:
        if domain in ACT_DOMAIN_MYSQL_MAP:
            skill = domain
        elif skill in ACT_DOMAIN_MYSQL_MAP:
            domain = skill
    q["domain"] = domain
    q["skill"] = skill

    q["stimulus"] = _coerce_stimulus(q.get("stimulus"), shared_passage=shared_passage)
    if passage_topic:
        q["passage_topic"] = passage_topic
    if passage_set_id:
        q["passage_set_id"] = passage_set_id

    wrong = q.get("wrong_choice_explanations")
    if isinstance(wrong, dict) and len(wrong) != 3:
        correct = str(q.get("correct_answer", "")).upper()
        trimmed = {k: v for k, v in wrong.items() if k.upper() != correct}
        if len(trimmed) >= 3:
            q["wrong_choice_explanations"] = dict(list(trimmed.items())[:3])
        elif len(trimmed) > 0:
            q["wrong_choice_explanations"] = trimmed

    return q


def normalize_act_payload(data: Dict[str, Any], meta: Dict[str, Any]) -> list[Dict[str, Any]]:
    passage_id = data.get("passage_id") or meta.get("passage_id")
    passage_topic = data.get("passage_topic") or meta.get("passage_topic")
    shared = data.get("passage_text") or data.get("shared_passage")
    if not shared and isinstance(data.get("stimulus"), dict):
        shared = data["stimulus"].get("passage_text") or data["stimulus"].get("text")

    questions = data.get("questions") or []
    out = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        out.append(
            normalize_act_question(
                q,
                shared_passage=shared,
                passage_topic=passage_topic,
                passage_set_id=passage_id,
            )
        )
    return out
