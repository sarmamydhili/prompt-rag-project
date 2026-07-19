#!/usr/bin/env python3
"""Delete digital_sat_rw_questions by generation_metadata.model_name."""

from __future__ import annotations

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from digital_sat_generation.app_config import DigitalSatConfig
from digital_sat_generation.persistence import DigitalSatPersistence


def delete_by_model(model_name: str, apply: bool = False) -> dict:
    config = DigitalSatConfig.load()
    persistence = DigitalSatPersistence(config)
    persistence.connect()
    assert persistence.collection is not None
    collection = persistence.collection

    query = {"generation_metadata.model_name": model_name}
    count = collection.count_documents(query)
    deleted = 0
    if apply and count:
        result = collection.delete_many(query)
        deleted = result.deleted_count

    persistence.close()
    return {"model_name": model_name, "matched": count, "deleted": deleted}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete Digital SAT RW questions by model_name"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact generation_metadata.model_name value (e.g. grok-3-latest)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete documents (default is dry-run)",
    )
    args = parser.parse_args()

    stats = delete_by_model(args.model, apply=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] delete_digital_sat_rw_by_model")
    print(f"  Model:    {stats['model_name']}")
    print(f"  Matched:  {stats['matched']}")
    print(f"  Deleted:  {stats['deleted']}")


if __name__ == "__main__":
    main()
