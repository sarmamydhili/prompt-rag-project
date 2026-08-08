#!/usr/bin/env python3
"""Shuffle multiple-choice options to reduce correct-answer letter bias."""

import argparse
import os
import random
import re
import sys
from collections import Counter

# Project root on path for explanation remap helper
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pymongo import MongoClient


def get_mongodb_connection(connection_string):
    """Establish MongoDB connection based on connection string"""
    try:
        client = MongoClient(connection_string)
        client.admin.command("ping")
        print("Successfully connected to MongoDB")
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None


def _choice_text(choice: str) -> str:
    if not isinstance(choice, str):
        return str(choice)
    if ". " in choice[:4]:
        return choice.split(". ", 1)[1]
    return re.sub(r"^[A-Da-d][.)]\s*", "", choice)


def _answer_letter(answer) -> str:
    s = str(answer or "").strip().upper()
    m = re.match(r"^([A-D])\b", s)
    if m:
        return m.group(1)
    if s and s[0] in "ABCD":
        return s[0]
    raise ValueError(f"Unrecognized correct_answer: {answer!r}")


def shuffle_doc(doc):
    """Shuffle multiple choice options while preserving the correct answer text.

    Returns (new_choices, new_correct, old_letter, letter_map) where letter_map
    maps old letters -> new letters by option text (for remapping explanations).
    """
    choices = doc.get("multiple_choices") or []
    if len(choices) < 2:
        raise ValueError("Need at least 2 multiple_choices")

    texts = [_choice_text(c) for c in choices]
    old_texts = list(texts)
    old_letter = _answer_letter(doc["correct_answer"])
    old_idx = ord(old_letter) - ord("A")
    if old_idx < 0 or old_idx >= len(texts):
        raise ValueError(f"correct_answer {old_letter} out of range for {len(texts)} choices")
    correct_txt = texts[old_idx]

    for i in range(len(texts) - 1, 0, -1):
        j = random.randint(0, i)
        texts[i], texts[j] = texts[j], texts[i]

    new_choices = [f"{chr(65 + i)}. {texts[i]}" for i in range(len(texts))]
    new_idx = texts.index(correct_txt)
    new_correct = chr(65 + new_idx)

    letter_map = {}
    for old_i, text in enumerate(old_texts):
        new_i = texts.index(text)
        letter_map[chr(65 + old_i)] = chr(65 + new_i)

    return new_choices, new_correct, old_letter, letter_map


def shuffle_questions(collection, query, dry_run: bool = False, verbose: bool = False):
    """Shuffle questions matching query. Always writes when choices change."""
    from pipeline.generation_pipeline.question_explanation_validation import (
        remap_wrong_explanations_for_shuffle,
    )

    count = 0
    shuffled_count = 0
    skipped = 0
    before = Counter()
    after = Counter()

    for doc in collection.find(query):
        try:
            old_letter = _answer_letter(doc.get("correct_answer"))
            before[old_letter] += 1
            new_choices, new_correct, _, letter_map = shuffle_doc(doc)
        except Exception as e:
            skipped += 1
            print(f"Skip {_id_str(doc)}: {e}")
            continue

        after[new_correct] += 1
        changed = (
            new_choices != doc.get("multiple_choices")
            or new_correct != old_letter
        )
        if not changed:
            count += 1
            continue

        if verbose or shuffled_count < 5:
            print(f"{_id_str(doc)}: {old_letter} -> {new_correct}")

        if not dry_run:
            update_fields = {
                "multiple_choices": new_choices,
                "correct_answer": new_correct,
            }
            # Remap letter-keyed wrong explanations so they stay aligned with choices
            for field in ("wrong_choice_explanations", "wrong_choices"):
                if isinstance(doc.get(field), dict) and doc[field]:
                    update_fields[field] = remap_wrong_explanations_for_shuffle(
                        doc[field], letter_map
                    )
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": update_fields},
            )
        shuffled_count += 1
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} documents...")

    print(f"Processed {count} questions total.")
    print(f"Updated {shuffled_count} questions{' (dry-run)' if dry_run else ''}.")
    if skipped:
        print(f"Skipped {skipped} questions.")
    print(f"Letter distribution before: {dict(sorted(before.items()))}")
    print(f"Letter distribution after:  {dict(sorted(after.items()))}")
    return count, shuffled_count


def _id_str(doc) -> str:
    return str(doc.get("_id", "?"))


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle MC choices to reduce correct-answer letter bias"
    )
    parser.add_argument(
        "--uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection URI",
    )
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument(
        "--collection",
        default="dryrun_questions",
        help="Questions collection (default: dryrun_questions)",
    )
    parser.add_argument(
        "--subject",
        default="AP Cybersecurity",
        help='Subject filter (default: "AP Cybersecurity")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute shuffle stats without writing",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    query = {"subject": args.subject}
    print("Connecting to MongoDB...")
    print(f"URI: {args.uri}")
    print(f"Database: {args.database}")
    print(f"Collection: {args.collection}")
    print(f"Query: {query}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)

    client = get_mongodb_connection(args.uri)
    if not client:
        return 1

    try:
        collection = client[args.database][args.collection]
        shuffle_questions(
            collection,
            query,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error during shuffle operation: {e}")
        return 1
    finally:
        client.close()
        print("MongoDB connection closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
