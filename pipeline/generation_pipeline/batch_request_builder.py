import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple


def build_custom_id(skill_id: int, bloom_level: str) -> str:
    level_slug = re.sub(r"[^\w]+", "_", bloom_level.strip())
    return f"skill_{skill_id}_bloom_{level_slug}"


def build_chat_completion_request(
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
) -> Dict:
    """Build one xAI Batch API JSONL row for /v1/chat/completions."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        },
    }


def build_manifest_entry(
    custom_id: str,
    skill_id: int,
    skill: str,
    subject: str,
    subject_area: str,
    bloom_level: str,
    num_questions: int,
    output_collection: str,
    task_name: str = "",
) -> Dict:
    return {
        "custom_id": custom_id,
        "skill_id": skill_id,
        "skill": skill,
        "subject": subject,
        "subject_area": subject_area,
        "bloom_level": bloom_level,
        "num_questions": num_questions,
        "output_collection": output_collection,
        "task_name": task_name,
    }


def write_jsonl(requests: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


def write_manifest(entries: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"requests": entries}, f, indent=2, ensure_ascii=False)


def default_output_paths(output_dir: str, prefix: str = "generation_batch") -> Tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"{prefix}_{timestamp}")
    return f"{base}.jsonl", f"{base}_manifest.json"
