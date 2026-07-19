"""MongoDB insert helpers for AP CED course framework documents."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient

import config

DEFAULT_DB = "adaptive_learning_docs"
DEFAULT_COLLECTION = "course_framework"


def get_framework_mongo(
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Tuple[MongoClient, Any, Any]:
    """Return (client, db, course_framework collection)."""
    load_dotenv()
    uri = mongo_uri or os.getenv("MONGODB_URI")
    if not uri:
        server = os.getenv("MONGODB_SERVER", getattr(config, "MONGODB_SERVER", "127.0.0.1"))
        port = os.getenv("MONGODB_PORT", getattr(config, "MONGODB_PORT", "27017"))
        user = getattr(config, "MONGODB_USER", None) or os.getenv("MONGODB_USER")
        password = getattr(config, "MONGODB_PASSWORD", None) or os.getenv("MONGODB_PASSWORD")
        if user and password:
            uri = f"mongodb://{user}:{password}@{server}:{port}/"
        else:
            uri = f"mongodb://{server}:{port}/"

    db_name = db_name or os.getenv("MONGO_DB_NAME", DEFAULT_DB)
    collection_name = collection_name or os.getenv(
        "MONGO_COURSE_FRAMEWORK_COLLECTION", DEFAULT_COLLECTION
    )

    client = MongoClient(uri)
    database = client[db_name]
    collection = database[collection_name]
    return client, database, collection


def upsert_course_framework(
    payload: Dict[str, Any],
    *,
    replace: bool = False,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Insert or replace a course_framework document keyed by payload['subject'].

    Returns a small result dict describing the action taken.
    """
    subject = payload.get("subject")
    if not subject:
        raise ValueError("Payload missing required 'subject' field")

    client, database, collection = get_framework_mongo(
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
    )
    try:
        existing = collection.find_one({"subject": subject}, {"_id": 1})
        if existing and not replace:
            return {
                "action": "skipped",
                "subject": subject,
                "_id": existing["_id"],
                "db": database.name,
                "collection": collection.name,
                "message": "Document already exists. Re-run with --replace to overwrite.",
            }

        if existing and replace:
            # Preserve _id when replacing the rest of the document
            payload_to_save = dict(payload)
            payload_to_save["_id"] = existing["_id"]
            collection.replace_one({"_id": existing["_id"]}, payload_to_save)
            return {
                "action": "replaced",
                "subject": subject,
                "_id": existing["_id"],
                "db": database.name,
                "collection": collection.name,
            }

        result = collection.insert_one(payload)
        return {
            "action": "inserted",
            "subject": subject,
            "_id": result.inserted_id,
            "db": database.name,
            "collection": collection.name,
        }
    finally:
        client.close()
