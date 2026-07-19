"""MongoDB persistence for Digital SAT RW questions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from digital_sat_generation.app_config import DigitalSatConfig
from digital_sat_generation.duplicate_checker import compute_content_hash
from digital_sat_generation.schemas import (
    SCHEMA_VERSION,
    resolve_domain_from_document,
    resolve_mysql_fields,
    skill_from_display,
)
from pipeline.pipeline_utils.db_connections import get_mongo_connection


class DigitalSatPersistence:
    def __init__(self, config: DigitalSatConfig):
        self.config = config
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None

    def connect(self) -> Collection:
        self.client, db = get_mongo_connection()
        self.collection = db[self.config.collection_name]
        return self.collection

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None
            self.collection = None

    def ensure_indexes(self) -> None:
        if self.collection is None:
            self.connect()
        assert self.collection is not None
        self._ensure_content_hash_index()
        self.collection.create_index(
            [("domain", ASCENDING), ("skill", ASCENDING), ("difficulty", ASCENDING)]
        )
        self.collection.create_index(
            [("subject_area", ASCENDING), ("status", ASCENDING)]
        )
        self.collection.create_index(
            [("skill_id", ASCENDING), ("status", ASCENDING)]
        )
        self.collection.create_index(
            [("item_skill", ASCENDING), ("difficulty", ASCENDING)]
        )
        self.collection.create_index(
            [("passage_topic", ASCENDING), ("status", ASCENDING)]
        )
        self.collection.create_index(
            [("status", ASCENDING), ("created_at", DESCENDING)]
        )

    def _ensure_content_hash_index(self) -> None:
        assert self.collection is not None
        content_hash_key = [("content_hash", ASCENDING)]
        for name, spec in self.collection.index_information().items():
            if name == "_id_":
                continue
            if list(spec.get("key", [])) == content_hash_key and spec.get("unique"):
                self.collection.drop_index(name)
                break
        try:
            self.collection.create_index(content_hash_key, unique=False)
        except OperationFailure as exc:
            if exc.code != 86:
                raise
            self.collection.drop_index(content_hash_key)
            self.collection.create_index(content_hash_key, unique=False)

    def enrich_document(
        self,
        question: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        content_hash = question.get("content_hash") or compute_content_hash(question)
        doc = dict(question)

        item_skill = doc.get("item_skill") or doc.get("skill", "")
        passage_topic = doc.get("passage_topic") or doc.get("subject_area", "")
        domain = resolve_domain_from_document(doc) or doc.get("domain", "")
        if not domain:
            skill_enum = skill_from_display(str(item_skill))
            if skill_enum:
                from digital_sat_generation.schemas import domain_for_skill

                domain = domain_for_skill(skill_enum)

        mysql_fields = resolve_mysql_fields(domain) if domain else {}

        doc.update(
            {
                "content_type": "digital_sat_rw_question",
                "schema_version": SCHEMA_VERSION,
                "test": "Digital SAT",
                "section": "Reading and Writing",
                "status": "draft",
                "content_hash": content_hash,
                "task_name": self.config.task_name,
                "Subject": self.config.subject,
                "item_skill": item_skill,
                "passage_topic": passage_topic,
                "generation_metadata": {
                    "model_name": model_name,
                    "prompt_version": self.config.prompt_version,
                    "generated_at": now,
                },
                "created_at": now,
                "updated_at": now,
            }
        )
        if mysql_fields:
            doc.update(mysql_fields)
            doc["domain"] = domain

        return doc

    def insert_many(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], int]:
        if not documents:
            return [], 0
        if self.collection is None:
            self.connect()
        assert self.collection is not None
        result = self.collection.insert_many(documents)
        ids = [str(i) for i in result.inserted_ids]
        return ids, len(ids)
