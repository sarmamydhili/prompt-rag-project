"""Orchestration for Digital SAT RW question generation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from digital_sat_generation.app_config import DigitalSatConfig
from digital_sat_generation.base_prompt import build_prompts
from digital_sat_generation.duplicate_checker import DuplicateChecker
from digital_sat_generation.persistence import DigitalSatPersistence
from digital_sat_generation.schemas import (
    GenerationRequest,
    GenerationStats,
    ItemPlan,
    build_generation_schedule,
)
from digital_sat_generation.utils import (
    normalize_llm_response,
    parse_llm_json,
    permute_question_to_target_answer,
    repair_explanation_letter_references,
)
from digital_sat_generation.validator import validate_batch_distribution, validate_question

logger = logging.getLogger(__name__)


def _get_llm_connections(config):
    from pipeline.pipeline_utils.llm_connections import LLMConnections

    return LLMConnections(config.llm_model_params)


class DigitalSatGenerator:
    def __init__(self, config: DigitalSatConfig, llm=None):
        self.config = config
        self._llm = llm
        self.persistence = DigitalSatPersistence(config)

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _get_llm_connections(self.config)
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    def generate(self, request: GenerationRequest) -> Tuple[List[Dict[str, Any]], GenerationStats]:
        stats = GenerationStats(
            requested_count=request.count,
            model_name=self.config.active_model_name,
        )

        schedule = build_generation_schedule(
            request.count,
            request.section,
            request.skill,
            request.subject_area,
            request.difficulty,
        )
        strict = request.strict or self.config.validation_mode == "strict"
        validated: List[Dict[str, Any]] = []
        rejected = 0
        generated_total = 0
        all_errors: List[str] = []

        for slot in schedule:
            if len(validated) >= request.count:
                break
            raw_questions, gen_count, gen_errors = self._generate_for_slot(
                request,
                slot,
                prior_errors=all_errors if all_errors else None,
            )
            generated_total += gen_count
            all_errors.extend(gen_errors)

            for q in raw_questions:
                q["subject_area"] = slot.passage_topic
                q.setdefault("difficulty", slot.difficulty)
                errors = validate_question(
                    q,
                    expected_domain=slot.domain,
                    expected_skill=slot.skill,
                    expected_difficulty=slot.difficulty,
                    expected_passage_topic=slot.passage_topic,
                    strict=strict,
                )
                if errors:
                    rejected += 1
                    all_errors.extend(errors)
                    logger.warning("Validation failed: %s", errors)
                    continue

                if self._should_check_duplicates(request):
                    dup_errors = self._check_duplicate(q, request)
                    if dup_errors:
                        rejected += 1
                        all_errors.extend(dup_errors)
                        continue

                if request.quality_review:
                    from digital_sat_generation.quality_review import QualityReviewer

                    reviewer = QualityReviewer(self.config)
                    review = reviewer.review(q)
                    q = reviewer.apply_review_to_document(q, review)
                    if not review.get("approved") and not request.override_quality_review:
                        rejected += 1
                        all_errors.append(
                            f"Quality review rejected: {review.get('issues', [])}"
                        )
                        continue

                validated.append(q)
                if len(validated) >= request.count:
                    break

        retry_round = 0
        while len(validated) < request.count and retry_round < self.config.max_retries:
            retry_round += 1
            slot = schedule[len(validated) % len(schedule)]
            raw_questions, gen_count, gen_errors = self._generate_for_slot(
                request,
                slot,
                prior_errors=all_errors,
            )
            generated_total += gen_count
            all_errors.extend(gen_errors)

            for q in raw_questions:
                q["subject_area"] = slot.passage_topic
                q.setdefault("difficulty", slot.difficulty)
                errors = validate_question(
                    q,
                    expected_domain=slot.domain,
                    expected_skill=slot.skill,
                    expected_difficulty=slot.difficulty,
                    strict=strict,
                )
                if errors:
                    rejected += 1
                    all_errors.extend(errors)
                    continue
                if self._should_check_duplicates(request):
                    dup_errors = self._check_duplicate(q, request)
                    if dup_errors:
                        rejected += 1
                        all_errors.extend(dup_errors)
                        continue
                if request.quality_review:
                    from digital_sat_generation.quality_review import QualityReviewer

                    reviewer = QualityReviewer(self.config)
                    review = reviewer.review(q)
                    q = reviewer.apply_review_to_document(q, review)
                    if not review.get("approved") and not request.override_quality_review:
                        rejected += 1
                        continue
                validated.append(q)
                if len(validated) >= request.count:
                    break

        validated = validated[: request.count]
        batch_errors = validate_batch_distribution(validated, request.count, strict=strict)
        if batch_errors and len(validated) == request.count:
            stats.validation_status = "batch_rejected"
            stats.errors.extend(batch_errors)
            stats.generated_count = generated_total
            stats.rejected_count = rejected + len(validated)
            stats.validated_count = 0
            return [], stats

        enriched = [
            self.persistence.enrich_document(q, self.config.active_model_name)
            for q in validated
        ]

        stats.generated_count = generated_total
        stats.validated_count = len(enriched)
        stats.rejected_count = rejected + (request.count - len(enriched))
        stats.correct_answer_distribution = _answer_distribution(enriched)
        stats.stimulus_formats = list(
            {q.get("stimulus", {}).get("format", "") for q in enriched}
        )
        stats.validation_status = "valid" if enriched else "failed"
        stats.errors = all_errors[:20]
        return enriched, stats

    def _should_check_duplicates(self, request: GenerationRequest) -> bool:
        if request.allow_duplicate:
            return False
        return self.config.enable_duplicate_check

    def _check_duplicate(
        self, question: Dict[str, Any], request: GenerationRequest
    ) -> List[str]:
        if self.persistence.collection is None and not request.dry_run:
            try:
                self.persistence.connect()
            except Exception:
                pass
        dup_checker = DuplicateChecker(
            collection=self.persistence.collection,
            allow_duplicate=request.allow_duplicate,
            enable_embedding_similarity=self.config.enable_embedding_similarity,
            similarity_threshold=self.config.similarity_threshold,
        )
        return dup_checker.check(question)

    def _generate_for_slot(
        self,
        request: GenerationRequest,
        slot: ItemPlan,
        prior_errors: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], int, List[str]]:
        return self._generate_batch(
            request,
            count=1,
            domain=slot.domain,
            skill=slot.skill,
            difficulty=slot.difficulty,
            subject_areas=[slot.passage_topic],
            target_correct_answer=slot.target_correct_answer,
            prior_errors=prior_errors,
        )

    def _generate_batch(
        self,
        request: GenerationRequest,
        count: int,
        subject_areas: List[str],
        domain: Optional[str] = None,
        skill: Optional[str] = None,
        difficulty: Optional[str] = None,
        target_correct_answer: Optional[str] = None,
        prior_errors: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], int, List[str]]:
        errors: List[str] = []
        resolved_domain = domain or request.domain or ""
        resolved_skill = skill or request.skill
        resolved_difficulty = difficulty or request.difficulty
        area = subject_areas[0] if len(set(subject_areas)) == 1 else "mixed"
        system_prompt, user_prompt = build_prompts(
            domain=resolved_domain,
            skill=resolved_skill,
            difficulty=resolved_difficulty,
            subject_area=area if area != "mixed" else "mixed",
            count=count,
            target_correct_answer=target_correct_answer,
            prior_errors=prior_errors,
        )

        raw = self.llm.call_llm_api(
            provider=self.config.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
        )
        if not raw:
            errors.append("LLM returned no response")
            return [], 0, errors

        try:
            parsed = parse_llm_json(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON from LLM: {exc}")
            return [], 0, errors

        questions = normalize_llm_response(parsed, resolved_skill)
        if target_correct_answer:
            questions = [
                permute_question_to_target_answer(q, target_correct_answer)
                for q in questions
            ]
        questions = [
            repair_explanation_letter_references(q) for q in questions
        ]

        if len(questions) != count:
            errors.append(
                f"Expected {count} questions, got {len(questions)}"
            )

        for i, q in enumerate(questions):
            q.setdefault("domain", resolved_domain)
            q.setdefault("skill", resolved_skill)
            q.setdefault("difficulty", resolved_difficulty)
            if i < len(subject_areas):
                q["subject_area"] = subject_areas[i]
            else:
                q.setdefault("subject_area", request.subject_area)

        return questions[:count], len(questions), errors


def _answer_distribution(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for q in questions:
        key = str(q.get("correct_answer", "")).strip().upper()
        dist[key] = dist.get(key, 0) + 1
    return dist
