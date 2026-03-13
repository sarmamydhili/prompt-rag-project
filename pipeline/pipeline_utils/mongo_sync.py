"""
Sync delta records from MongoDB Staging to Production.

Delta mode: compare by question_id. Reports added, deleted, changed.
Dry run: report numbers only, no writes.
"""

import argparse
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional, Set, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, ConnectionFailure

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _doc_content_hash(doc: Dict[str, Any], exclude_keys: Tuple[str, ...] = ("_id",)) -> str:
    """Return a stable hash of document content, excluding _id (and other keys)."""
    d = {k: v for k, v in doc.items() if k not in exclude_keys}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _build_uri(
    server: Optional[str],
    port: Optional[str],
    user: Optional[str],
    password: Optional[str],
    database: str = "adaptive_learning_docs",
    default_uri: str = "mongodb://localhost:27017",
) -> str:
    """Build MongoDB URI from server, port, optional user/password, and database."""
    if not server:
        return default_uri
    port = port or os.getenv("MONGODB_PORT", "27017")
    host = f"{server}:{port}"
    if user and password:
        return f"mongodb://{user}:{password}@{host}/{database}"
    return f"mongodb://{host}/{database}"


class MongoSync:
    """
    Sync delta documents from Staging to Production by question_id.

    Delta mode: added (in staging, not in prod), deleted (in prod, not in staging),
    changed (same question_id, different content). Dry run reports numbers only.
    """

    def __init__(
        self,
        source_uri: Optional[str] = None,
        target_uri: Optional[str] = None,
        database: str = "adaptive_learning_docs",
        collection: Optional[str] = None,
        id_field: str = "question_id",
    ):
        """
        Initialize sync utility.

        Args:
            source_uri: Staging MongoDB connection string.
            target_uri: Production MongoDB connection string.
            database: Database name.
            collection: Collection name (required: pass in or set MONGO_COLLECTION).
            id_field: Field used for delta comparison (default question_id).
        """
        self.database = database or os.getenv("MONGO_DATABASE", "adaptive_learning_docs")
        self.collection = (collection or os.getenv("MONGO_COLLECTION") or "").strip()
        if not self.collection:
            raise ValueError("Collection name is required. Set --collection / -c or MONGO_COLLECTION.")
        self.id_field = id_field
        _port = os.getenv("MONGODB_PORT", "27017")

        # Staging (source) URI: explicit > built from .env (stag_*, MONGODB_PORT)
        if source_uri:
            self.source_uri = source_uri
        else:
            self.source_uri = _build_uri(
                os.getenv("stag_mongodb_server"),
                _port,
                os.getenv("stag_mongo_db_user"),
                os.getenv("stag_mongo_db_password"),
                database=self.database,
                default_uri="mongodb://localhost:27017",
            )

        # Production (target) URI: explicit > built from .env (prod_*, MONGODB_PORT)
        if target_uri:
            self.target_uri = target_uri
        else:
            self.target_uri = _build_uri(
                os.getenv("prod_mongodb_server"),
                _port,
                os.getenv("prod_mongo_db_user"),
                os.getenv("prod_mongo_db_password"),
                database=self.database,
                default_uri="mongodb://localhost:27017",
            )

        self.source_client: Optional[MongoClient] = None
        self.target_client: Optional[MongoClient] = None

    def connect_databases(self) -> bool:
        """Establish connections to both source (staging) and target (prod)."""
        try:
            logger.info("Connecting to source (staging) database...")
            self.source_client = MongoClient(
                self.source_uri, serverSelectionTimeoutMS=5000
            )
            self.source_client.admin.command("ping")

            logger.info("Connecting to target (production) database...")
            self.target_client = MongoClient(
                self.target_uri, serverSelectionTimeoutMS=5000
            )
            self.target_client.admin.command("ping")

            logger.info("Successfully connected to both databases")
            return True

        except ConnectionFailure as e:
            logger.error("Failed to connect to database: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error during connection: %s", e)
            return False

    def get_source_question_ids(self) -> Set[Any]:
        """Return set of question_id (or id_field) values in source (staging)."""
        try:
            coll = self.source_client[self.database][self.collection]
            ids = set(
                doc[self.id_field]
                for doc in coll.find({}, {self.id_field: 1})
                if self.id_field in doc
            )
            logger.info("Source (%s): %d documents with %s", self.collection, len(ids), self.id_field)
            return ids
        except Exception as e:
            logger.error("Error fetching source %s: %s", self.id_field, e)
            return set()

    def get_target_question_ids(self) -> Set[Any]:
        """Return set of question_id (or id_field) values in target (prod)."""
        try:
            coll = self.target_client[self.database][self.collection]
            ids = set(
                doc[self.id_field]
                for doc in coll.find({}, {self.id_field: 1})
                if self.id_field in doc
            )
            logger.info("Target (%s): %d documents with %s", self.collection, len(ids), self.id_field)
            return ids
        except Exception as e:
            logger.error("Error fetching target %s: %s", self.id_field, e)
            return set()

    def compute_delta_stats(
        self, batch_size: int = 500
    ) -> Dict[str, Any]:
        """
        Compute added, deleted, changed counts by question_id.

        Returns:
            dict with keys: added_count, deleted_count, changed_count,
            added_ids, deleted_ids, changed_ids (sets).
        """
        source_coll = self.source_client[self.database][self.collection]
        target_coll = self.target_client[self.database][self.collection]

        source_ids = self.get_source_question_ids()
        target_ids = self.get_target_question_ids()

        added_ids = source_ids - target_ids
        deleted_ids = target_ids - source_ids
        common_ids = source_ids & target_ids

        changed_ids: Set[Any] = set()
        if common_ids:
            common_list = list(common_ids)
            for i in range(0, len(common_list), batch_size):
                chunk = common_list[i : i + batch_size]
                source_docs = {doc[self.id_field]: doc for doc in source_coll.find({self.id_field: {"$in": chunk}})}
                target_docs = {doc[self.id_field]: doc for doc in target_coll.find({self.id_field: {"$in": chunk}})}
                for qid in chunk:
                    sd = source_docs.get(qid)
                    td = target_docs.get(qid)
                    if sd is not None and td is not None:
                        if _doc_content_hash(sd) != _doc_content_hash(td):
                            changed_ids.add(qid)
            logger.info(
                "Delta stats: added=%d, deleted=%d, changed=%d",
                len(added_ids),
                len(deleted_ids),
                len(changed_ids),
            )
        else:
            logger.info(
                "Delta stats: added=%d, deleted=%d, changed=0",
                len(added_ids),
                len(deleted_ids),
            )

        return {
            "added_count": len(added_ids),
            "deleted_count": len(deleted_ids),
            "changed_count": len(changed_ids),
            "added_ids": added_ids,
            "deleted_ids": deleted_ids,
            "changed_ids": changed_ids,
        }

    def sync_documents(
        self, batch_size: int = 1000, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Delta sync by question_id: report added/deleted/changed, then copy added to prod.

        If dry_run=True, only compute and log numbers; no writes.

        Returns:
            dict with added_count, deleted_count, changed_count, synced_count.
        """
        if not self.connect_databases():
            return {"added_count": 0, "deleted_count": 0, "changed_count": 0, "synced_count": 0}

        try:
            source_coll = self.source_client[self.database][self.collection]
            target_coll = self.target_client[self.database][self.collection]

            stats = self.compute_delta_stats(batch_size=batch_size)
            added_count = stats["added_count"]
            deleted_count = stats["deleted_count"]
            changed_count = stats["changed_count"]
            added_ids = stats["added_ids"]

            # Always log delta numbers
            logger.info(
                "Delta (collection=%s, key=%s): added=%d, deleted=%d, changed=%d",
                self.collection,
                self.id_field,
                added_count,
                deleted_count,
                changed_count,
            )

            if dry_run:
                logger.info("Dry run: no records copied.")
                return {
                    "added_count": added_count,
                    "deleted_count": deleted_count,
                    "changed_count": changed_count,
                    "synced_count": 0,
                }

            if not added_ids:
                logger.info("No new records to copy.")
                return {
                    "added_count": added_count,
                    "deleted_count": deleted_count,
                    "changed_count": changed_count,
                    "synced_count": 0,
                }

            synced_count = 0
            added_list = list(added_ids)
            for i in range(0, len(added_list), batch_size):
                chunk = added_list[i : i + batch_size]
                cursor = source_coll.find({self.id_field: {"$in": chunk}})
                batch = list(cursor)
                if not batch:
                    continue
                try:
                    result = target_coll.insert_many(batch, ordered=False)
                    synced_count += len(result.inserted_ids)
                    logger.info(
                        "Synced batch of %d documents. Total: %d",
                        len(batch),
                        synced_count,
                    )
                except BulkWriteError as e:
                    inserted_count = len(e.details.get("insertedIds", []))
                    synced_count += inserted_count
                    logger.warning(
                        "Batch had %d failed insertions",
                        len(batch) - inserted_count,
                    )
                except Exception as e:
                    logger.error("Error inserting batch: %s", e)

            logger.info(
                "Sync completed. Delta: added=%d, deleted=%d, changed=%d; synced=%d",
                added_count,
                deleted_count,
                changed_count,
                synced_count,
            )
            return {
                "added_count": added_count,
                "deleted_count": deleted_count,
                "changed_count": changed_count,
                "synced_count": synced_count,
            }

        except Exception as e:
            logger.error("Error during sync: %s", e)
            return {"added_count": 0, "deleted_count": 0, "changed_count": 0, "synced_count": 0}
        finally:
            self.close_connections()

    def close_connections(self) -> None:
        """Close database connections."""
        if self.source_client:
            self.source_client.close()
            logger.info("Source (staging) connection closed")
        if self.target_client:
            self.target_client.close()
            logger.info("Target (production) connection closed")


def main() -> None:
    """Run delta sync from Staging to Prod; config via CLI and/or env."""
    parser = argparse.ArgumentParser(
        description="Sync delta records (by question_id) from MongoDB Staging to Production."
    )
    parser.add_argument(
        "--collection",
        "-c",
        default=os.getenv("MONGO_COLLECTION"),
        help="Collection name (required: set this or MONGO_COLLECTION env)",
    )
    parser.add_argument(
        "--database",
        "-d",
        default=os.getenv("MONGO_DATABASE", "adaptive_learning_docs"),
        help="Database name (default: adaptive_learning_docs)",
    )
    parser.add_argument(
        "--id-field",
        default="question_id",
        help="Field used for delta comparison (default: question_id)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1000,
        help="Batch size for inserts (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report added/deleted/changed numbers only; do not copy records",
    )
    parser.add_argument(
        "--staging-uri",
        help="Staging MongoDB URI (overrides .env)",
    )
    parser.add_argument(
        "--prod-uri",
        help="Production MongoDB URI (overrides .env)",
    )
    args = parser.parse_args()

    if not (args.collection or "").strip():
        parser.error("Collection name is required. Use --collection / -c or set MONGO_COLLECTION.")
    args.collection = args.collection.strip()

    try:
        sync_util = MongoSync(
            source_uri=args.staging_uri,
            target_uri=args.prod_uri,
            database=args.database,
            collection=args.collection,
            id_field=args.id_field,
        )
        result = sync_util.sync_documents(
            batch_size=args.batch_size, dry_run=args.dry_run
        )
        logger.info(
            "Result: added=%d, deleted=%d, changed=%d, synced=%d",
            result["added_count"],
            result["deleted_count"],
            result["changed_count"],
            result["synced_count"],
        )
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
    except Exception as e:
        logger.error("Unexpected error in main: %s", e)


if __name__ == "__main__":
    main()
