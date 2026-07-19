"""Optional LLM quality review for Digital SAT questions."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from digital_sat_generation.app_config import PACKAGE_DIR, DigitalSatConfig
from digital_sat_generation.utils import parse_llm_json, strip_markdown_fences

import os

PROMPTS_DIR = os.path.join(PACKAGE_DIR, "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), encoding="utf-8") as f:
        return f.read().strip()


class QualityReviewer:
    def __init__(self, config: DigitalSatConfig):
        self.config = config
        from pipeline.pipeline_utils.llm_connections import LLMConnections

        self.llm = LLMConnections(config.llm_model_params)

    def review(self, question: Dict[str, Any]) -> Dict[str, Any]:
        system = _load_prompt("quality_review_system_prompt.txt")
        user_template = _load_prompt("quality_review_user_prompt.txt")
        user = user_template.format(
            domain=question.get("domain", ""),
            skill=question.get("skill", ""),
            difficulty=question.get("difficulty", ""),
            question_json=json.dumps(question, indent=2),
        )
        raw = self.llm.call_llm_api(
            provider=self.config.llm_model,
            system_prompt=system,
            user_prompt=user,
            temperature=0.1,
        )
        if not raw:
            return {
                "approved": False,
                "issues": ["Quality review LLM returned no response"],
                "confidence": 0.0,
            }
        try:
            result = parse_llm_json(raw)
        except json.JSONDecodeError:
            return {
                "approved": False,
                "issues": ["Quality review returned invalid JSON"],
                "confidence": 0.0,
            }
        return {
            "approved": bool(result.get("approved")),
            "issues": result.get("issues", []),
            "confidence": float(result.get("confidence", 0.0)),
        }

    def apply_review_to_document(
        self, question: Dict[str, Any], review: Dict[str, Any]
    ) -> Dict[str, Any]:
        doc = dict(question)
        doc["quality_review"] = {
            "status": "approved" if review.get("approved") else "rejected",
            "confidence": review.get("confidence", 0.0),
            "issues": review.get("issues", []),
        }
        return doc
