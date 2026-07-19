#!/usr/bin/env python3
"""Insert Digital SAT Reading and Writing course framework into MongoDB."""

from __future__ import annotations

import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from digital_sat_generation.app_config import DigitalSatConfig
from pipeline.pipeline_utils.db_connections import get_mongo_connection

FRAMEWORK_PATH = os.path.join(
    project_root,
    "digital_sat_generation",
    "data",
    "digital_sat_rw_course_framework.json",
)


def load_framework() -> dict:
    with open(FRAMEWORK_PATH, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert Digital SAT RW course framework document"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing document with the same subject",
    )
    args = parser.parse_args()

    config = DigitalSatConfig.load()
    framework = load_framework()
    subject = framework["subject"]

    client, db = get_mongo_connection()
    collection = db[config.mongo_course_framework_collection]

    existing = collection.find_one({"subject": subject})
    if existing and not args.replace:
        print(f"Document already exists for subject '{subject}' (_id={existing['_id']})")
        print("Use --replace to overwrite.")
        client.close()
        sys.exit(1)

    if existing and args.replace:
        collection.replace_one({"subject": subject}, framework)
        print(f"Replaced course framework for subject '{subject}'")
    else:
        result = collection.insert_one(framework)
        print(f"Inserted course framework for subject '{subject}' (_id={result.inserted_id})")

    print(f"  Units: {len(framework['units'])}")
    print(f"  Collection: {config.mongo_course_framework_collection}")
    client.close()


if __name__ == "__main__":
    main()
