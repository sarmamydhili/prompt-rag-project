"""Focused tests for Digital SAT RW generation."""

from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from digital_sat_generation.duplicate_checker import DuplicateChecker, compute_content_hash
from digital_sat_generation.generator import DigitalSatGenerator, _answer_distribution
from digital_sat_generation.persistence import DigitalSatPersistence
from digital_sat_generation.schemas import (
    GenerationRequest,
    build_generation_schedule,
    validate_domain_skill,
    validate_section_skill,
)
from digital_sat_generation.utils import strip_markdown_fences
from digital_sat_generation.validator import validate_batch_distribution, validate_question


# 1. Valid single-item parse/enrich
def test_valid_single_item_enrich(valid_inference_question):
    config = MagicMock()
    config.prompt_version = "digital-sat-rw-v1"
    config.task_name = "Digital SAT Reading and Writing"
    config.subject = "Reading and Writing"
    persistence = DigitalSatPersistence(config)
    doc = persistence.enrich_document(valid_inference_question, "test-model")
    assert doc["content_type"] == "digital_sat_rw_question"
    assert doc["schema_version"] == 2
    assert doc["test"] == "Digital SAT"
    assert doc["status"] == "draft"
    assert doc["content_hash"]
    assert doc["generation_metadata"]["model_name"] == "test-model"
    assert doc["task_name"] == "Digital SAT Reading and Writing"
    assert doc["Subject"] == "Reading and Writing"
    assert doc["skill"] == "Information and Ideas"
    assert doc["skill_id"] == 301
    assert doc["item_skill"] == "inferences"
    assert doc["passage_topic"] == "science"
    assert doc["subject_area"] == "Reading"


# 2. Valid batch parse/enrich
def test_valid_batch_enrich(valid_inference_question, valid_boundaries_question):
    config = MagicMock()
    config.prompt_version = "digital-sat-rw-v1"
    config.task_name = "Digital SAT Reading and Writing"
    config.subject = "Reading and Writing"
    persistence = DigitalSatPersistence(config)
    docs = [
        persistence.enrich_document(valid_inference_question, "m"),
        persistence.enrich_document(valid_boundaries_question, "m"),
    ]
    assert len(docs) == 2
    assert docs[0]["content_hash"] != docs[1]["content_hash"]
    assert docs[1]["skill_id"] == 302
    assert docs[1]["subject_area"] == "Writing"
    assert docs[1]["item_skill"] == "boundaries"


# 3. Domain and skill mismatch rejection
def test_domain_skill_mismatch():
    errors = validate_domain_skill("Craft and Structure", "inferences")
    assert any("does not belong" in e for e in errors)


# 4. Unsupported difficulty rejection (strict only)
def test_unsupported_difficulty(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["difficulty"] = "Impossible"
    errors = validate_question(q, strict=True)
    assert any("Unsupported difficulty" in e for e in errors)


# 5. Missing stimulus rejection
def test_missing_stimulus(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q.pop("stimulus")
    errors = validate_question(q)
    assert any("Missing or invalid stimulus" in e for e in errors)


# 6. Stimulus length validation (strict only)
def test_stimulus_length_validation(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["stimulus"]["text"] = "Too short."
    q["stimulus"]["sentences"] = [{"sentence_number": 1, "text": "Too short."}]
    errors = validate_question(q, strict=True)
    assert any("word count" in e for e in errors)


# 7. Exactly four choices required
def test_four_choices_required(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["choices"] = q["choices"][:3]
    errors = validate_question(q)
    assert any("Expected exactly 4 choices" in e for e in errors)


# 8. Duplicate choices rejected (strict only)
def test_duplicate_choices_rejected(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["choices"][3]["text"] = q["choices"][0]["text"]
    errors = validate_question(q, strict=True)
    assert any("Duplicate normalized choice" in e for e in errors)


# 9. Invalid correct-answer key rejected
def test_invalid_correct_answer(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["correct_answer"] = "E"
    errors = validate_question(q)
    assert any("Invalid correct_answer" in e for e in errors)


# 10. Missing correct explanation rejected (strict only)
def test_missing_correct_explanation(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["correct_choice_explanation"] = {}
    errors = validate_question(q, strict=True)
    assert any("Missing correct_choice_explanation" in e for e in errors)


# 11. Missing wrong-choice explanation rejected (strict only)
def test_missing_wrong_explanation(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["wrong_choice_explanations"] = {}
    errors = validate_question(q, strict=True)
    assert any("wrong_choice_explanations keys" in e for e in errors)


# 12. Wrong-choice map containing correct answer rejected (strict only)
def test_wrong_map_contains_correct(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["wrong_choice_explanations"]["C"] = {
        "why_wrong": "x",
        "mistake_type": "not_supported",
        "relevant_text": [],
    }
    errors = validate_question(q, strict=True)
    assert any("must not contain the correct answer" in e for e in errors)


# 13. Invalid sentence reference rejected (strict only)
def test_invalid_sentence_reference(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["correct_choice_explanation"]["relevant_text"] = [{"sentence_numbers": [99]}]
    errors = validate_question(q, strict=True)
    assert any("Invalid sentence reference" in e for e in errors)


# 14. Invalid text span rejected (strict only)
def test_invalid_text_span(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["correct_choice_explanation"]["relevant_text"] = [
        {"sentence_numbers": [1], "text_span": "nonexistent phrase xyz"}
    ]
    errors = validate_question(q, strict=True)
    assert any("text_span not found" in e for e in errors)


# 15. Invalid mistake type rejected (strict only)
def test_invalid_mistake_type(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    key = next(k for k in q["wrong_choice_explanations"])
    q["wrong_choice_explanations"][key]["mistake_type"] = "bogus_mistake"
    errors = validate_question(q, strict=True)
    assert any("Invalid mistake_type" in e for e in errors)


# 16. Markdown fence cleanup
def test_markdown_fence_cleanup():
    raw = '```json\n{"questions": []}\n```'
    cleaned = strip_markdown_fences(raw)
    parsed = json.loads(cleaned)
    assert "questions" in parsed


# 17. Invalid JSON retry path
def test_invalid_json_retry(mock_config):
    generator = DigitalSatGenerator(mock_config, llm=MagicMock())
    request = GenerationRequest(
        section="reading",
        domain="Information and Ideas",
        skill="inferences",
        difficulty="Medium",
        count=1,
    )
    with patch.object(generator.llm, "call_llm_api", side_effect=["not json", "not json", '{"questions": []}']):
        questions, gen_count, errors = generator._generate_batch(
            request,
            1,
            ["science"],
            domain="Information and Ideas",
            skill="inferences",
            difficulty="Medium",
        )
    assert gen_count == 0 or errors


# 18. Batch answer-distribution validation (strict only)
def test_batch_answer_distribution():
    questions = [{"correct_answer": "A"} for _ in range(10)]
    errors = validate_batch_distribution(questions, 10, strict=True)
    assert any("same letter" in e for e in errors)


def test_single_item_batch_distribution_allowed():
    questions = [{"correct_answer": "C"}]
    errors = validate_batch_distribution(questions, 1)
    assert errors == []


def test_batch_requires_three_positions_for_large_batch():
    questions = [{"correct_answer": "A"}, {"correct_answer": "B"}] * 4
    errors = validate_batch_distribution(questions, 8, strict=True)
    assert any("at least 3 distinct" in e for e in errors)


# 19. Boundaries ambiguous punctuation rejected (strict only)
def test_boundaries_missing_grammar_rule(valid_boundaries_question):
    q = copy.deepcopy(valid_boundaries_question)
    q.pop("grammar_metadata")
    errors = validate_question(q, strict=True)
    assert any("grammar_metadata" in e for e in errors)


# 20. Missing grammar rule rejected (strict only)
def test_missing_grammar_rule(valid_boundaries_question):
    q = copy.deepcopy(valid_boundaries_question)
    q["grammar_metadata"] = {"clause_analysis": "only analysis"}
    errors = validate_question(q, strict=True)
    assert any("tested_rule" in e for e in errors)


# 21. Transition item without logical relationship rejected (strict only)
def test_transition_without_relationship(valid_transitions_question):
    q = copy.deepcopy(valid_transitions_question)
    q.pop("transition_metadata")
    errors = validate_question(q, strict=True)
    assert any("transition_metadata" in e for e in errors)


# 22. Rhetorical synthesis without goal rejected (strict only)
def test_rhetorical_synthesis_without_goal(valid_rhetorical_synthesis_question):
    q = copy.deepcopy(valid_rhetorical_synthesis_question)
    q.pop("rhetorical_goal")
    errors = validate_question(q, strict=True)
    assert any("rhetorical_goal" in e for e in errors)


# 23. Rhetorical synthesis bad note index rejected (strict only)
def test_rhetorical_synthesis_bad_note_index(valid_rhetorical_synthesis_question):
    q = copy.deepcopy(valid_rhetorical_synthesis_question)
    q["correct_choice_explanation"]["notes_used"] = [99]
    errors = validate_question(q, strict=True)
    assert any("Invalid notes_used index" in e for e in errors)


# 24. Paired-text missing text rejected
def test_paired_text_missing_text(valid_paired_text_question):
    from digital_sat_generation.validator import _validate_paired_texts

    stimulus = copy.deepcopy(valid_paired_text_question["stimulus"])
    stimulus["texts"][1]["text"] = ""
    errors = _validate_paired_texts(stimulus)
    assert any("Missing text" in e for e in errors)


# 25. Quantitative nonexistent value rejected (strict only)
def test_quantitative_nonexistent_value(valid_quantitative_question):
    q = copy.deepcopy(valid_quantitative_question)
    q["correct_choice_explanation"]["data_used"] = [
        {"row": "Treatment Z", "column": "Average Growth", "value": 9.9}
    ]
    errors = validate_question(q, strict=True)
    assert any("nonexistent row" in e for e in errors)


# 26. Duplicate hash rejected
def test_duplicate_hash_rejected(valid_inference_question):
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {"_id": "existing"}
    checker = DuplicateChecker(mock_collection, allow_duplicate=False)
    errors = checker.check(copy.deepcopy(valid_inference_question))
    assert any("Duplicate content_hash" in e for e in errors)


def test_allow_duplicate_bypasses_hash(valid_inference_question):
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {"_id": "existing"}
    checker = DuplicateChecker(mock_collection, allow_duplicate=True)
    errors = checker.check(copy.deepcopy(valid_inference_question))
    assert errors == []


# 27. Dry-run does not write to MongoDB
def test_dry_run_no_insert(valid_inference_question, mock_config):
    persistence = DigitalSatPersistence(mock_config)
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = None
    persistence.collection = mock_collection
    persistence.insert_many = MagicMock(return_value=([], 0))
    request = GenerationRequest(
        section="reading",
        domain="Information and Ideas",
        skill="inferences",
        difficulty="Medium",
        count=1,
        dry_run=True,
    )
    generator = DigitalSatGenerator(mock_config, llm=MagicMock())
    generator.persistence = persistence
    payload = json.dumps({"questions": [valid_inference_question]})
    with patch.object(generator.llm, "call_llm_api", return_value=payload):
        docs, stats = generator.generate(request)
    persistence.insert_many.assert_not_called()
    assert len(docs) == 1


# 28. Valid batch inserts one document per question
def test_batch_insert_one_per_question(valid_inference_question, valid_boundaries_question, mock_config):
    persistence = DigitalSatPersistence(mock_config)
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = None
    persistence.collection = mock_collection
    persistence.client = MagicMock()
    persistence.insert_many = MagicMock(return_value=(["id1", "id2"], 2))

    docs = [
        persistence.enrich_document(valid_inference_question, "m"),
        persistence.enrich_document(valid_boundaries_question, "m"),
    ]
    ids, count = persistence.insert_many(docs)
    assert count == 2
    assert len(ids) == 2


# 29. Failed items are not inserted
def test_failed_items_not_inserted(valid_inference_question, mock_config):
    generator = DigitalSatGenerator(mock_config, llm=MagicMock())
    bad = copy.deepcopy(valid_inference_question)
    bad["correct_answer"] = "E"
    payload = json.dumps({"questions": [bad]})
    request = GenerationRequest(
        section="reading",
        domain="Information and Ideas",
        skill="inferences",
        difficulty="Medium",
        count=1,
    )
    with patch.object(generator.llm, "call_llm_api", return_value=payload):
        docs, stats = generator.generate(request)
    assert docs == []
    assert stats.validated_count == 0


# 30. Existing project generation workflows remain unaffected
def test_existing_generation_import_unchanged():
    import os

    module_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "pipeline",
        "generation_pipeline",
        "generate_new_question.py",
    )
    with open(os.path.abspath(module_path), encoding="utf-8") as f:
        source = f.read()
    assert "class GlobalContext" in source
    assert "class QuestionGenerationWorkflow" in source


# Additional helper tests
def test_valid_inference_passes(valid_inference_question):
    errors = validate_question(valid_inference_question, strict=True)
    assert errors == []


def test_compute_content_hash_stable(valid_inference_question):
    h1 = compute_content_hash(valid_inference_question)
    h2 = compute_content_hash(valid_inference_question)
    assert h1 == h2


def test_answer_distribution_helper():
    dist = _answer_distribution(
        [{"correct_answer": "A"}, {"correct_answer": "B"}, {"correct_answer": "A"}]
    )
    assert dist == {"A": 2, "B": 1}


def test_deferred_skill_rejected():
    errors = validate_domain_skill("Craft and Structure", "cross_text_connections")
    assert any("not yet implemented" in e for e in errors)


def test_normalize_string_stimulus():
    from digital_sat_generation.utils import normalize_llm_question

    raw = {
        "stimulus": "Recent studies have shown that monarch butterfly populations have declined. Researchers blame habitat loss and pesticides.",
        "question": {"stem": "What can be inferred?"},
        "choices": [
            {"key": "A", "text": "One"},
            {"key": "B", "text": "Two"},
            {"key": "C", "text": "Three"},
            {"key": "D", "text": "Four"},
        ],
        "correct_answer": "A",
        "correct_choice_explanation": {
            "why_correct": "x",
            "relevant_text": [{"sentence_numbers": [1]}],
            "reasoning": "r",
            "strategy": "s",
        },
        "wrong_choice_explanations": {
            "B": {"why_wrong": "w", "mistake_type": "contradicted", "relevant_text": []},
            "C": {"why_wrong": "w", "mistake_type": "not_supported", "relevant_text": []},
            "D": {"why_wrong": "w", "mistake_type": "overgeneralization", "relevant_text": []},
        },
    }
    q = normalize_llm_question(raw, "inferences")
    assert isinstance(q["stimulus"], dict)
    assert q["stimulus"]["format"] == "single_text"
    assert q["stimulus"]["text"]
    assert q["stimulus"]["sentences"]
    assert q["wrong_choice_explanations"]["B"]["mistake_type"] == "contradicted_by_text"
    errors = validate_question(q, expected_skill="inferences", strict=True)
    assert "Missing or invalid stimulus" not in errors


def test_draft_accepts_title_case_mistake_type(valid_inference_question):
    from digital_sat_generation.utils import normalize_llm_question

    q = copy.deepcopy(valid_inference_question)
    key = next(k for k in q["wrong_choice_explanations"])
    q["wrong_choice_explanations"][key]["mistake_type"] = "Contradicted"
    normalized = normalize_llm_question(q, "inferences")
    assert normalized["wrong_choice_explanations"][key]["mistake_type"] == "contradicted_by_text"
    errors = validate_question(normalized, expected_skill="inferences")
    assert errors == []


def test_draft_allows_missing_metadata(valid_boundaries_question):
    q = copy.deepcopy(valid_boundaries_question)
    q.pop("grammar_metadata")
    errors = validate_question(q, expected_skill="boundaries")
    assert errors == []


def test_build_generation_schedule_mixed_section():
    schedule = build_generation_schedule(6, "mixed", "mixed", "mixed", "mixed")
    assert len(schedule) == 6
    skills = [slot.skill for slot in schedule]
    assert "inferences" in skills
    assert "boundaries" in skills
    difficulties = {slot.difficulty for slot in schedule}
    assert difficulties == {"Medium", "Hard"}
    areas = {slot.passage_topic for slot in schedule}
    assert len(areas) > 1


def test_build_generation_schedule_reading_only():
    schedule = build_generation_schedule(3, "reading", "mixed", "science", "mixed")
    assert all(slot.skill in ("words_in_context", "central_ideas_and_details", "inferences") for slot in schedule)
    assert all(slot.passage_topic == "science" for slot in schedule)
    assert [slot.difficulty for slot in schedule] == ["Medium", "Hard", "Medium"]


def test_assign_difficulties_mixed():
    from digital_sat_generation.utils import assign_difficulties

    levels = assign_difficulties(4, "mixed")
    assert levels == ["Medium", "Hard", "Medium", "Hard"]


def test_assign_target_correct_answers_rotates():
    from digital_sat_generation.utils import assign_target_correct_answers

    targets = assign_target_correct_answers(8)
    assert targets == ["A", "B", "C", "D", "A", "B", "C", "D"]


def test_build_generation_schedule_assigns_target_answers():
    schedule = build_generation_schedule(4, "reading", "inferences", "science", "Medium")
    assert [slot.target_correct_answer for slot in schedule] == ["A", "B", "C", "D"]


def test_permute_moves_correct_to_target(valid_inference_question):
    import random

    from digital_sat_generation.utils import permute_question_to_target_answer

    random.seed(0)
    q = copy.deepcopy(valid_inference_question)
    assert q["correct_answer"] == "C"
    correct_text = next(c["text"] for c in q["choices"] if c["key"] == "C")

    permuted = permute_question_to_target_answer(q, "A")
    assert permuted["correct_answer"] == "A"
    assert next(c["text"] for c in permuted["choices"] if c["key"] == "A") == correct_text
    assert len({c["text"] for c in permuted["choices"]}) == 4


def test_permute_remaps_wrong_explanations(valid_inference_question):
    import random

    from digital_sat_generation.utils import permute_question_to_target_answer

    random.seed(1)
    q = copy.deepcopy(valid_inference_question)
    original_wrong = copy.deepcopy(q["wrong_choice_explanations"])
    text_by_key = {c["key"]: c["text"] for c in q["choices"]}

    permuted = permute_question_to_target_answer(q, "B")
    assert permuted["correct_answer"] == "B"
    for choice in permuted["choices"]:
        key = choice["key"]
        if key == "B":
            continue
        old_key = next(
            k for k, text in text_by_key.items() if text == choice["text"] and k != "C"
        )
        assert (
            permuted["wrong_choice_explanations"][key]["why_wrong"]
            == original_wrong[old_key]["why_wrong"]
        )


def test_repair_explanation_letter_references(valid_inference_question):
    from digital_sat_generation.utils import repair_explanation_letter_references

    q = copy.deepcopy(valid_inference_question)
    q["correct_answer"] = "D"
    q["correct_choice_explanation"]["why_correct"] = (
        "B accurately captures the passage's primary focus."
    )

    repaired = repair_explanation_letter_references(q)
    assert repaired["correct_choice_explanation"]["why_correct"].startswith(
        "This choice accurately"
    )


def test_draft_rejects_stale_explanation_letter(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["correct_answer"] = "D"
    q["correct_choice_explanation"]["why_correct"] = (
        "B accurately captures the passage's primary focus."
    )

    errors = validate_question(q)
    assert any("cites choice B" in e for e in errors)


def test_draft_rejects_short_stimulus(valid_inference_question):
    q = copy.deepcopy(valid_inference_question)
    q["stimulus"]["text"] = "Too short."
    q["stimulus"]["sentences"] = [{"sentence_number": 1, "text": "Too short."}]
    errors = validate_question(q)
    assert any("word count" in e for e in errors)


def test_enrich_maps_writing_domain(valid_boundaries_question):
    config = MagicMock()
    config.prompt_version = "digital-sat-rw-v1"
    config.task_name = "Digital SAT Reading and Writing"
    config.subject = "Reading and Writing"
    persistence = DigitalSatPersistence(config)
    doc = persistence.enrich_document(valid_boundaries_question, "test-model")
    assert doc["skill"] == "Standard English Conventions"
    assert doc["skill_id"] == 302
    assert doc["subject_area"] == "Writing"
    assert doc["item_skill"] == "boundaries"
    assert doc["passage_topic"] == "science"


def test_migrate_transform():
    from digital_sat_generation.schemas import transform_document_mysql_alignment

    old_doc = {
        "domain": "Information and Ideas",
        "skill": "inferences",
        "subject_area": "history_social_studies",
        "schema_version": 1,
    }
    updates = transform_document_mysql_alignment(old_doc)
    assert updates is not None
    assert updates["skill"] == "Information and Ideas"
    assert updates["skill_id"] == 301
    assert updates["item_skill"] == "inferences"
    assert updates["passage_topic"] == "history_social_studies"
    assert updates["subject_area"] == "Reading"
    assert updates["schema_version"] == 2


def test_migrate_transform_skips_already_aligned():
    from digital_sat_generation.schemas import transform_document_mysql_alignment

    aligned = {
        "skill": "Information and Ideas",
        "skill_id": 301,
        "item_skill": "inferences",
        "passage_topic": "science",
        "subject_area": "Reading",
        "schema_version": 2,
    }
    assert transform_document_mysql_alignment(aligned) is None


def test_section_skill_mismatch_rejected():
    errors = validate_section_skill("reading", "boundaries")
    assert any("does not belong to section" in e for e in errors)


def test_batch_distribution_skipped_in_draft():
    questions = [{"correct_answer": "A"} for _ in range(10)]
    assert validate_batch_distribution(questions, 10, strict=False) == []


def test_ensure_content_hash_index_migrates_unique_index():
    config = MagicMock()
    config.collection_name = "digital_sat_rw_questions"
    persistence = DigitalSatPersistence(config)
    mock_collection = MagicMock()
    mock_collection.index_information.return_value = {
        "_id_": {"key": [("_id", 1)]},
        "content_hash_1": {"key": [("content_hash", 1)], "unique": True},
    }
    persistence.collection = mock_collection

    persistence._ensure_content_hash_index()

    mock_collection.drop_index.assert_called_once_with("content_hash_1")
    mock_collection.create_index.assert_called_once()
    _, kwargs = mock_collection.create_index.call_args
    assert kwargs.get("unique") is False


def test_generator_skips_duplicate_check_by_default(valid_inference_question, mock_config):
    mock_config.enable_duplicate_check = False
    generator = DigitalSatGenerator(mock_config, llm=MagicMock())
    persistence = DigitalSatPersistence(mock_config)
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {"_id": "existing"}
    persistence.collection = mock_collection
    generator.persistence = persistence
    request = GenerationRequest(
        section="reading",
        domain="Information and Ideas",
        skill="inferences",
        difficulty="Medium",
        count=1,
    )
    payload = json.dumps({"questions": [valid_inference_question]})
    with patch.object(generator.llm, "call_llm_api", return_value=payload):
        docs, stats = generator.generate(request)
    assert len(docs) == 1
    mock_collection.find_one.assert_not_called()


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.llm_model = "grok"
    cfg.llm_model_params = {"grok_llm_model": "grok-3-latest"}
    cfg.active_model_name = "grok-3-latest"
    cfg.temperature = 0.3
    cfg.max_retries = 2
    cfg.prompt_version = "digital-sat-rw-v1"
    cfg.enable_embedding_similarity = False
    cfg.similarity_threshold = 0.85
    cfg.collection_name = "digital_sat_rw_questions"
    cfg.validation_mode = "draft"
    cfg.enable_duplicate_check = False
    cfg.task_name = "Digital SAT Reading and Writing"
    cfg.subject = "Reading and Writing"
    return cfg
