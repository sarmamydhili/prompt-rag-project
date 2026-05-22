from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional


VALID_CHOICES = frozenset({"A", "B", "C", "D"})


@dataclass
class ReviewDecision:
    recommended_answer: str
    review_flag: bool
    review_reason: str


def compute_review_decision(
    db_answer: str,
    model_responses: Dict[str, Optional[str]],
    requires_diagram: bool = False,
) -> ReviewDecision:
    """
    Option A: any disagreement with DB or uncertainty → manual review flag.
    Auto-correct is not performed by the review stage; apply only sets flags/metadata.
    """
    db_answer = (db_answer or "").strip().upper()
    if db_answer not in VALID_CHOICES:
        db_answer = db_answer or "N/A"

    if requires_diagram:
        return ReviewDecision(
            recommended_answer=db_answer if db_answer in VALID_CHOICES else "",
            review_flag=True,
            review_reason="Diagram review required",
        )

    valid = {
        model: response.strip().upper()
        for model, response in model_responses.items()
        if response and response.strip().upper() in VALID_CHOICES
    }

    if not valid:
        return ReviewDecision(
            recommended_answer=db_answer if db_answer in VALID_CHOICES else "",
            review_flag=True,
            review_reason="No valid model responses",
        )

    counts = Counter(valid.values())
    top_answer, top_count = counts.most_common(1)[0]
    tied = sum(1 for count in counts.values() if count == top_count) > 1

    if tied:
        return ReviewDecision(
            recommended_answer=db_answer if db_answer in VALID_CHOICES else top_answer,
            review_flag=True,
            review_reason="Model tie — no clear majority",
        )

    total = len(valid)
    if top_answer == db_answer:
        return ReviewDecision(
            recommended_answer=top_answer,
            review_flag=False,
            review_reason=f"Majority agrees with database ({top_count}/{total} models)",
        )

    return ReviewDecision(
        recommended_answer=top_answer,
        review_flag=True,
        review_reason=f"{top_count}/{total} models disagree with database",
    )
