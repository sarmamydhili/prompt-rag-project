"""Shared quality scoring and elimination rules for generation rinse cycles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

VALID = frozenset({"A", "B", "C", "D"})


def normalize_letter(value: Optional[str]) -> str:
    if not value:
        return "N/A"
    letter = str(value).strip().upper()[:1]
    return letter if letter in VALID else "N/A"


def passes_dual_model_qc(db_answer: str, deepseek: str, gemini: str) -> bool:
    db = normalize_letter(db_answer)
    ds = normalize_letter(deepseek)
    gm = normalize_letter(gemini)
    if db == "N/A" or ds == "N/A" or gm == "N/A":
        return False
    return ds == gm == db


def elimination_rule(
    db_answer: str,
    deepseek: str,
    gemini: str,
    *,
    hint_step_mismatch: bool = False,
) -> Optional[str]:
    if hint_step_mismatch:
        return "E4_hint_step_mismatch"
    db = normalize_letter(db_answer)
    ds = normalize_letter(deepseek)
    gm = normalize_letter(gemini)
    if ds == "N/A" or gm == "N/A":
        return "E3_model_na"
    if ds != gm:
        return "E2_models_disagree"
    if ds != db:
        return "E1_both_models_disagree_with_db"
    return None


@dataclass
class RinseScore:
    total: int
    passed: int
    pass_rate: float

    @property
    def meets_goal(self) -> bool:
        return self.pass_rate >= 0.90


def score_rows(rows: Iterable[dict]) -> RinseScore:
    rows = list(rows)
    total = len(rows)
    passed = sum(
        1
        for r in rows
        if passes_dual_model_qc(
            r.get("db_answer", ""),
            r.get("deepseek", ""),
            r.get("gemini", ""),
        )
    )
    rate = (passed / total) if total else 0.0
    return RinseScore(total=total, passed=passed, pass_rate=rate)


def write_qc_csv(path: Path | str, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "skill",
        "level",
        "requires_diagram",
        "db_answer",
        "deepseek",
        "gemini",
        "pass",
        "elimination_rule",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rule = elimination_rule(
                row.get("db_answer", ""),
                row.get("deepseek", ""),
                row.get("gemini", ""),
                hint_step_mismatch=bool(row.get("hint_step_mismatch")),
            )
            row = dict(row)
            row["pass"] = "yes" if not rule else "no"
            row["elimination_rule"] = rule or ""
            writer.writerow(row)


def write_eliminate_csv(path: Path | str, rows: List[dict]) -> int:
    path = Path(path)
    eliminate = [
        r
        for r in rows
        if elimination_rule(
            r.get("db_answer", ""),
            r.get("deepseek", ""),
            r.get("gemini", ""),
            hint_step_mismatch=bool(r.get("hint_step_mismatch")),
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "skill",
        "level",
        "db_answer",
        "deepseek",
        "gemini",
        "elimination_rule",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in eliminate:
            rule = elimination_rule(
                row.get("db_answer", ""),
                row.get("deepseek", ""),
                row.get("gemini", ""),
                hint_step_mismatch=bool(row.get("hint_step_mismatch")),
            )
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "skill": row.get("skill", ""),
                    "level": row.get("level", ""),
                    "db_answer": row.get("db_answer", ""),
                    "deepseek": row.get("deepseek", ""),
                    "gemini": row.get("gemini", ""),
                    "elimination_rule": rule,
                }
            )
    return len(eliminate)
