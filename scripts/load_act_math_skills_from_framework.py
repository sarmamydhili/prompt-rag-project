#!/usr/bin/env python3
"""Load MySQL adaptive_skills + ACT Math task from ACT course framework.

Uses explicit skill_id values from framework JSON. App-facing Subject is Math.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mysql.connector
from dotenv import dotenv_values

FRAMEWORK_JSON = (
    Path(__file__).resolve().parents[1]
    / "digital_sat_generation"
    / "data"
    / "act_math_course_framework.json"
)

TASK_NAME = "ACT Math"
APP_SUBJECT = "Math"
PACKAGE_ID = 3
SKILL_ID_MIN = 320
SKILL_ID_MAX = 325


def _clean_topic(topic: str) -> str:
    return topic.strip()


def load_framework(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework-json",
        type=Path,
        default=FRAMEWORK_JSON,
    )
    parser.add_argument("--package-id", type=int, default=PACKAGE_ID)
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[1] / ".env"),
    )
    parser.add_argument(
        "--mysql-host",
        help="Override MYSQL_HOST (e.g. 127.0.0.1 on prod EC2 via SSH tunnel)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    framework = load_framework(args.framework_json)
    app_subject = framework.get("app_subject") or APP_SUBJECT
    units = framework.get("units") or []

    env = dotenv_values(args.env_file)
    conn = mysql.connector.connect(
        host=args.mysql_host or env.get("MYSQL_HOST") or "127.0.0.1",
        user=env["MYSQL_USER"],
        password=env["MYSQL_PASSWORD"],
        database=env.get("MYSQL_DATABASE") or "adaptive_learning",
    )
    cur = conn.cursor(dictionary=True)

    cur.execute(
        "SELECT adaptive_task_id FROM adaptive_tasks WHERE adaptive_task_name = %s",
        (TASK_NAME,),
    )
    existing_task = cur.fetchone()
    cur.execute(
        """
        SELECT skill_id, skill_name FROM adaptive_skills
        WHERE skill_id BETWEEN %s AND %s ORDER BY skill_id
        """,
        (SKILL_ID_MIN, SKILL_ID_MAX),
    )
    existing_act_skills = cur.fetchall()
    if existing_task and len(existing_act_skills) >= len(units):
        print(
            f"Task already present (task_id={existing_task['adaptive_task_id']}); "
            "updating skill names and details from the framework."
        )

    planned = []
    for u in units:
        unit_name = (u.get("unit") or "").strip()
        skill_id = u.get("skill_id")
        if not unit_name or skill_id is None:
            continue
        topics = [_clean_topic(t.get("topic") or "") for t in (u.get("topics") or [])]
        topics = [t for t in topics if t]
        planned.append(
            {
                "skill_id": int(skill_id),
                "skill_name": unit_name,
                "skill_details": "; ".join(topics) if topics else unit_name,
                "subject_area": u.get("subject_area") or u.get("section_type") or "",
                "Subject": app_subject,
                "display_skill": unit_name,
                "unit_code": u.get("unit_code"),
            }
        )

    print(f"Will upsert {len(planned)} skills + task {TASK_NAME!r} (package {args.package_id})")
    for p in planned:
        print(f"  {p['skill_id']} {p['unit_code']}: {p['skill_name']} ({p['subject_area']})")

    if args.dry_run:
        print("Dry-run only; no writes.")
        conn.close()
        return

    for p in planned:
        cur.execute("SELECT skill_id FROM adaptive_skills WHERE skill_id = %s", (p["skill_id"],))
        if cur.fetchone():
            cur.execute(
                """
                UPDATE adaptive_skills SET
                  skill_name=%s, skill_details=%s, subject_area=%s, Subject=%s, display_skill=%s
                WHERE skill_id=%s
                """,
                (
                    p["skill_name"],
                    p["skill_details"],
                    p["subject_area"],
                    p["Subject"],
                    p["display_skill"],
                    p["skill_id"],
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO adaptive_skills
                  (skill_id, skill_name, skill_details, subject_area, Subject,
                   subject_id, subject_area_id, display_skill, additional_details)
                VALUES (%s, %s, %s, %s, %s, NULL, NULL, %s, %s)
                """,
                (
                    p["skill_id"],
                    p["skill_name"],
                    p["skill_details"],
                    p["subject_area"],
                    p["Subject"],
                    p["display_skill"],
                    "[]",
                ),
            )

    if existing_task:
        task_id = existing_task["adaptive_task_id"]
    else:
        cur.execute(
            """
            INSERT INTO adaptive_tasks
              (adaptive_task_name, adaptive_task_description, task_designation)
            VALUES (%s, %s, %s)
            """,
            (TASK_NAME, "ACT Math MCQ practice", "general"),
        )
        task_id = cur.lastrowid

    for p in planned:
        cur.execute(
            """
            INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id)
            VALUES (%s, %s, %s)
            """,
            (task_id, TASK_NAME, p["skill_id"]),
        )

    cur.execute(
        """
        INSERT IGNORE INTO adaptive_package_tasks
          (adaptive_package_id, adaptive_task_id)
        VALUES (%s, %s)
        """,
        (args.package_id, task_id),
    )

    conn.commit()
    print(f"\nLoaded task_id={task_id} name={TASK_NAME}")
    for p in planned:
        print(f"  skill_id={p['skill_id']}  {p['skill_name']}")
    print(f"  package_id={args.package_id}")
    conn.close()


if __name__ == "__main__":
    main()
