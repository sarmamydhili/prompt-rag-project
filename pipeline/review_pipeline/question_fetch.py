import logging
from typing import Dict, List, Optional

from pipeline.review_pipeline.mongo_connection import get_review_collection
from pipeline.review_pipeline.review_context import ReviewContext

logger = logging.getLogger(__name__)


def fetch_questions(context: ReviewContext) -> List[Dict]:
    client, collection = get_review_collection(context)
    try:
        query: Dict = {"subject": context.subject}
        if context.skill:
            query["skill"] = context.skill
        if context.level_num_min is not None:
            query["level_num"] = {"$gte": context.level_num_min}

        cursor = collection.find(
            query,
            {
                "_id": 1,
                "question": 1,
                "multiple_choices": 1,
                "correct_answer": 1,
                "subject": 1,
                "skill": 1,
                "skill_name": 1,
                "requires_diagram": 1,
                "learning_objectives": 1,
                "level_num": 1,
            },
        )
        if context.limit:
            cursor = cursor.limit(context.limit)

        questions = list(cursor)
        logger.info("Fetched %s questions for subject=%s skill=%s", len(questions), context.subject, context.skill)
        return questions
    finally:
        client.close()
