#!/usr/bin/env python3
"""Generate idempotent MySQL SQL for ACT Math on the Standardized Tests package."""

from __future__ import annotations

import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
FRAMEWORK = project_root / "digital_sat_generation/data/act_math_course_framework.json"
OUT = project_root / "scripts/sql/act_math_mysql_production.sql"

TASK_NAME = "ACT Math"
PACKAGE_ID = 3
SKILL_ID_MIN = 320
SKILL_ID_MAX = 325


def _sql_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "''")


def main() -> None:
    framework = json.loads(FRAMEWORK.read_text(encoding="utf-8"))
    app_subject = framework.get("app_subject") or "Math"
    units = framework.get("units") or []

    lines = [
        "-- ACT Math on Standardized Tests package (idempotent)",
        "USE adaptive_learning;",
        "",
    ]

    for u in units:
        skill_id = int(u["skill_id"])
        unit_name = _sql_escape((u.get("unit") or "").strip())
        subject_area = _sql_escape(u.get("subject_area") or u.get("section_type") or "")
        topics = "; ".join(
            t.get("topic", "").strip() for t in (u.get("topics") or []) if t.get("topic")
        )
        details = _sql_escape(topics or unit_name)
        display = unit_name
        lines.append(
            f"INSERT INTO adaptive_skills "
            f"(skill_id, skill_name, skill_details, subject_area, Subject, "
            f"subject_id, subject_area_id, display_skill, additional_details) "
            f"VALUES ({skill_id}, '{unit_name}', '{details}', '{subject_area}', "
            f"'{app_subject}', NULL, NULL, '{display}', '[]') "
            f"ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name), "
            f"skill_details=VALUES(skill_details), subject_area=VALUES(subject_area), "
            f"Subject=VALUES(Subject), display_skill=VALUES(display_skill);"
        )

    lines.extend(
        [
            "",
            f"INSERT INTO adaptive_tasks (adaptive_task_name, adaptive_task_description, task_designation)",
            f"SELECT '{TASK_NAME}', 'ACT Math MCQ practice', 'general'",
            f"WHERE NOT EXISTS (",
            f"  SELECT 1 FROM adaptive_tasks WHERE adaptive_task_name = '{TASK_NAME}'",
            f");",
            "",
            f"SET @act_task_id = (SELECT adaptive_task_id FROM adaptive_tasks",
            f"  WHERE adaptive_task_name = '{TASK_NAME}' LIMIT 1);",
            "",
        ]
    )

    for u in units:
        skill_id = int(u["skill_id"])
        lines.append(
            f"INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id) "
            f"VALUES (@act_task_id, '{TASK_NAME}', {skill_id});"
        )

    lines.extend(
        [
            "",
            f"INSERT IGNORE INTO adaptive_package_tasks (adaptive_package_id, adaptive_task_id)",
            f"VALUES ({PACKAGE_ID}, @act_task_id);",
            "",
            "SELECT adaptive_package_id, adaptive_package_name FROM adaptive_packages WHERE adaptive_package_id=3;",
            "SELECT adaptive_task_id, adaptive_task_name FROM adaptive_tasks WHERE adaptive_task_name='ACT Math';",
            f"SELECT skill_id, skill_name FROM adaptive_skills WHERE skill_id BETWEEN {SKILL_ID_MIN} AND {SKILL_ID_MAX} ORDER BY skill_id;",
            "",
        ]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
