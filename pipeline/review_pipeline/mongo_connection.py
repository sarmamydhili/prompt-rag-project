from typing import Tuple

from pymongo import MongoClient
from pymongo.collection import Collection

import config
from pipeline.review_pipeline.review_context import ReviewContext


def get_review_collection(context: ReviewContext) -> Tuple[MongoClient, Collection]:
    """Connect to localhost MongoDB using review_config only."""
    if context.mongo_server not in ("127.0.0.1", "localhost"):
        raise ValueError(
            f"Review pipeline is localhost-only; refused server '{context.mongo_server}'"
        )

    uri = f"mongodb://{context.mongo_server}:{context.mongo_port}/"
    if config.MONGODB_USER and config.MONGODB_PASSWORD:
        uri = (
            f"mongodb://{config.MONGODB_USER}:{config.MONGODB_PASSWORD}"
            f"@{context.mongo_server}:{context.mongo_port}/"
        )

    client = MongoClient(uri)
    db = client[context.mongo_db_name]
    return client, db[context.mongo_questions_collection]
