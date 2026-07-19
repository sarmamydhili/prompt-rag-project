#!/usr/bin/env python3
"""Migrate digital_sat_rw_questions to MySQL-aligned schema (schema_version 2)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from digital_sat_generation.app_config import DigitalSatConfig
from digital_sat_generation.persistence import DigitalSatPersistence
from digital_sat_generation.schemas import transform_document_mysql_alignment


def migrate_collection(
    apply: bool = False,
    force: bool = False,
) -> Dict[str, int]:
    config = DigitalSatConfig.load()
    persistence = DigitalSatPersistence(config)
    persistence.connect()
    assert persistence.collection is not None
    collection = persistence.collection

    stats = {
        "scanned": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
    }
    failures: List[str] = []

    for doc in collection.find({}):
        stats["scanned"] += 1
        doc_id = doc.get("_id")

        if not force and doc.get("schema_version") == 2:
            if doc.get("skill_id") and doc.get("item_skill") and doc.get("passage_topic"):
                stats["skipped"] += 1
                continue

        updates = transform_document_mysql_alignment(doc)
        if updates is None:
            stats["skipped"] += 1
            continue

        updates["updated_at"] = datetime.now(timezone.utc)

        if apply:
            try:
                collection.update_one({"_id": doc_id}, {"$set": updates})
                stats["updated"] += 1
            except Exception as exc:
                stats["failed"] += 1
                failures.append(f"{doc_id}: {exc}")
        else:
            stats["updated"] += 1
            if stats["updated"] <= 3:
                print(f"\nSample update for {doc_id}:")
                for key, value in updates.items():
                    print(f"  {key}: {value}")

    persistence.close()
    stats["failures"] = failures
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate digital_sat_rw_questions to MySQL-aligned schema"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates (default is dry-run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-migrate documents even if schema_version is 2",
    )
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Migration mode: {mode}")
    if args.force:
        print("Force: re-migrating all eligible documents")

    stats = migrate_collection(apply=args.apply, force=args.force)

    print("\nResults:")
    print(f"  Scanned:  {stats['scanned']}")
    print(f"  Updated:  {stats['updated']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Failed:   {stats['failed']}")

    if stats.get("failures"):
        print("\nFailures:")
        for err in stats["failures"][:10]:
            print(f"  - {err}")

    print("\nVerification (run in Mongosh):")
    print(
        'db.digital_sat_rw_questions.aggregate([\n'
        '  { $group: { _id: { skill_id: "$skill_id", subject_area: "$subject_area" }, count: { $sum: 1 } } }\n'
        "])"
    )

    if stats["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
