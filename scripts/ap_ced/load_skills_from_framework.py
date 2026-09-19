#!/usr/bin/env python3
"""Load MySQL adaptive_skills + adaptive_tasks from a Mongo course_framework subject.

Mirrors the AP Cybersecurity pattern: one skill per unit; skill_name == unit title
(so generation can resolve LOs via get_unit_objectives(subject, skill_name)).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mysql.connector
from dotenv import dotenv_values
from pymongo import MongoClient


def _slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:200]


def _clean_topic(topic: str) -> str:
    return (
        topic.replace(" continued on next page", "")
        .replace(" ontinued on next page", "")
        .strip()
    )


def load_units_from_mongo(uri: str, database: str, subject: str):
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    doc = client[database]["course_framework"].find_one({"subject": subject})
    if not doc:
        raise SystemExit(f"No course_framework document for subject={subject!r}")
    units = doc.get("units") or []
    if not units:
        raise SystemExit(f"course_framework for {subject!r} has no units")
    return units


def load_units_from_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("units") or []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default="AP Business with Personal Finance",
        help="Exact subject name in course_framework / adaptive_skills.Subject",
    )
    parser.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017")
    parser.add_argument("--mongo-db", default="adaptive_learning_docs")
    parser.add_argument(
        "--from-json",
        type=Path,
        help="Optional framework JSON instead of Mongo",
    )
    parser.add_argument(
        "--task-name",
        help="adaptive_tasks.adaptive_task_name (default: --subject)",
    )
    parser.add_argument(
        "--package-id",
        type=int,
        default=1,
        help="adaptive_package_tasks package to attach (default: 1 High School)",
    )
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[2] / ".env"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subject = args.subject
    task_name = args.task_name or subject

    if args.from_json:
        units = load_units_from_json(args.from_json)
    else:
        units = load_units_from_mongo(args.mongo_uri, args.mongo_db, subject)

    env = dotenv_values(args.env_file)
    conn = mysql.connector.connect(
        host=env.get("MYSQL_HOST") or "127.0.0.1",
        user=env["MYSQL_USER"],
        password=env["MYSQL_PASSWORD"],
        database=env.get("MYSQL_DATABASE") or "adaptive_learning",
    )
    cur = conn.cursor(dictionary=True)

    # Idempotency: skip if task already exists with skills
    cur.execute(
        "SELECT adaptive_task_id FROM adaptive_tasks WHERE adaptive_task_name = %s",
        (task_name,),
    )
    existing_task = cur.fetchone()
    cur.execute(
        "SELECT skill_id, skill_name FROM adaptive_skills WHERE Subject = %s ORDER BY skill_id",
        (subject,),
    )
    existing_skills = cur.fetchall()
    if existing_task and existing_skills:
        print(f"Already loaded: task_id={existing_task['adaptive_task_id']}")
        for s in existing_skills:
            print(f"  {s['skill_id']}: {s['skill_name']}")
        conn.close()
        return

    if existing_skills and not existing_task:
        raise SystemExit(
            f"Found {len(existing_skills)} skills for Subject={subject!r} but no task "
            f"{task_name!r}. Resolve manually before re-running."
        )

    planned = []
    for u in units:
        unit_name = (u.get("unit") or "").strip()
        if not unit_name:
            continue
        topics = [_clean_topic(t.get("topic") or "") for t in (u.get("topics") or [])]
        topics = [t for t in topics if t]
        planned.append(
            {
                "skill_name": unit_name,  # MUST match course_framework units.unit
                "skill_details": "; ".join(topics) if topics else unit_name,
                "subject_area": subject,
                "Subject": subject,
                "display_skill": unit_name,
                "additional_details": "[]",
                "unit_code": u.get("unit_code"),
            }
        )

    print(f"Will insert {len(planned)} skills + task {task_name!r}")
    for p in planned:
        print(f"  {p['unit_code']}: {p['skill_name']}")

    if args.dry_run:
        print("Dry-run only; no writes.")
        conn.close()
        return

    skill_ids = []
    for p in planned:
        cur.execute(
            """
            INSERT INTO adaptive_skills
              (skill_name, skill_details, subject_area, Subject,
               subject_id, subject_area_id, display_skill, additional_details)
            VALUES (%s, %s, %s, %s, NULL, NULL, %s, %s)
            """,
            (
                p["skill_name"],
                p["skill_details"],
                p["subject_area"],
                p["Subject"],
                p["display_skill"],
                p["additional_details"],
            ),
        )
        skill_ids.append(cur.lastrowid)

    cur.execute(
        """
        INSERT INTO adaptive_tasks
          (adaptive_task_name, adaptive_task_description, task_designation)
        VALUES (%s, %s, %s)
        """,
        (task_name, f"{subject} test practice", "general"),
    )
    task_id = cur.lastrowid

    for sid in skill_ids:
        cur.execute(
            """
            INSERT INTO adaptive_task_skills (task_id, task_name, skill_id)
            VALUES (%s, %s, %s)
            """,
            (task_id, task_name, sid),
        )

    if args.package_id:
        cur.execute(
            """
            INSERT IGNORE INTO adaptive_package_tasks
              (adaptive_package_id, adaptive_task_id)
            VALUES (%s, %s)
            """,
            (args.package_id, task_id),
        )

    conn.commit()

    print("\nLoaded:")
    print(f"  task_id={task_id} name={task_name}")
    for sid, p in zip(skill_ids, planned):
        print(f"  skill_id={sid}  {p['skill_name']}")
    if args.package_id:
        print(f"  package_id={args.package_id}")

    conn.close()


if __name__ == "__main__":
    main()
