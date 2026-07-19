"""Content hash and duplicate detection."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import config
from digital_sat_generation.utils import collect_stimulus_text, normalize_whitespace


def compute_content_hash(question: Dict[str, Any]) -> str:
    stimulus_text = collect_stimulus_text(question.get("stimulus", {}))
    stem = (question.get("question") or {}).get("stem", "")
    choices = question.get("choices", [])
    choice_text = " ".join(
        f"{c.get('key', '')}:{c.get('text', '')}" for c in sorted(choices, key=lambda x: x.get("key", ""))
    )
    canonical = normalize_whitespace(f"{stimulus_text} {stem} {choice_text}")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalized_stem(question: Dict[str, Any]) -> str:
    return normalize_whitespace((question.get("question") or {}).get("stem", ""))


class DuplicateChecker:
    def __init__(
        self,
        collection,
        allow_duplicate: bool = False,
        enable_embedding_similarity: bool = False,
        similarity_threshold: float = 0.85,
    ):
        self.collection = collection
        self.allow_duplicate = allow_duplicate
        self.enable_embedding_similarity = enable_embedding_similarity
        self.similarity_threshold = similarity_threshold
        self._batch_hashes: set[str] = set()

    def check(self, question: Dict[str, Any]) -> List[str]:
        if self.allow_duplicate:
            return []
        errors: List[str] = []
        content_hash = compute_content_hash(question)
        question["content_hash"] = content_hash

        if content_hash in self._batch_hashes:
            errors.append(f"Duplicate content_hash within batch: {content_hash[:12]}...")
            return errors

        if self.collection is not None:
            existing = self.collection.find_one({"content_hash": content_hash})
            if existing:
                errors.append(f"Duplicate content_hash in database: {content_hash[:12]}...")

            stem = normalized_stem(question)
            if stem:
                stem_dup = self.collection.find_one(
                    {"question.stem": {"$regex": f"^{stem}$", "$options": "i"}}
                )
                if stem_dup:
                    errors.append("Duplicate normalized question stem in database")

            if self.enable_embedding_similarity and not errors:
                sim_error = self._check_embedding_similarity(question)
                if sim_error:
                    errors.append(sim_error)

        if not errors:
            self._batch_hashes.add(content_hash)
        return errors

    def _check_embedding_similarity(self, question: Dict[str, Any]) -> Optional[str]:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=config.OPENAI_API_KEY)
            text = collect_stimulus_text(question.get("stimulus", {}))
            stem = (question.get("question") or {}).get("stem", "")
            payload = f"{text} {stem}".strip()
            if not payload:
                return None

            response = client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=payload,
            )
            embedding = response.data[0].embedding

            recent = self.collection.find(
                {"embedding": {"$exists": True}},
                {"embedding": 1},
            ).limit(50)
            for doc in recent:
                stored = doc.get("embedding")
                if stored and _cosine_similarity(embedding, stored) >= self.similarity_threshold:
                    return "Highly similar item detected via embedding similarity"
        except Exception:
            return None
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
